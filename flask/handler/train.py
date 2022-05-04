import random
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

import data as d
from arch import net_D, net_G
from utils import transform, loss, lr_scheduler, diffjpeg

class Trainer:
    def __init__(self, opt):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opt = opt
        self.path = opt['path']
        self.train_opt = opt['train']
        
        self.net_G = net_G.RRDBNet(**self.opt['network_g']).to(self.device)
        self.net_D = net_D.UNetDiscriminatorSN(**self.opt['network_d']).to(self.device)
        
        self.optimizers = []
        self.schedulers = []
        
        self.jpeger = diffjpeg.DiffJPEG(differentiable=False).to(self.device)
        self.usm_sharpener = transform.USMSharp().to(self.device)
        self.queue_size = opt.get('queue_size', 180)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(self.device)
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).to(self.device)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def load_network(self, net, load_path, strict=True, param_key='params'):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[:7]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
    
    def load_trainer_state(self):
        # load pretrain model
        load_path_g = self.path.get('pretrain_network_g', None)
        param_key_g = self.path.get('param_key_g', 'params')
        self.load_network(self.net_G, load_path_g, self.path.get('strict_load_g', True), param_key_g)

        load_path_d = self.path.get('pretrain_network_d', None)
        param_key_d = self.path.get('param_key_d', 'params')
        self.load_network(self.net_D, load_path_d, self.path.get('strict_load_d', True), param_key_d)
        
    def init_training_setting(self):
        self.net_G.train()
        self.net_D.train()
        
        # init loss function
        if self.train_opt.get('pixel_opt'):
            self.cri_pix = loss.L1loss(**self.train_opt['pixel_opt']).to(self.device)
        
        if self.train_opt.get('perceptual_opt'):
            self.cri_perceptual = loss.PerceptualLoss(**self.train_opt['perceptual_opt']).to(self.device)
        
        if self.train_opt.get('gan_opt'):
            self.cri_gan = loss.GanLoss(**self.train_opt['gan_opt']).to(self.device)
        
        self.net_d_iters = self.train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = self.train_opt.get('net_d_init_iters', 0)

        # init optimizer
        optim_g_opt = self.train_opt['optim_g']
        optim_d_opt = self.train_opt['optim_d']    
        self.optimizers_g = torch.optim.Adam(self.net_G.parameters(), **optim_g_opt)
        self.optimizers_d = torch.optim.Adam(self.net_D.parameters(), **optim_d_opt)
        self.optimizers.append(self.optimizers_g)
        self.optimizers.append(self.optimizers_d)
        
        # setup scheduler
        scheduler_type = self.train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **self.train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnelingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **self.train_opt['scheduler']))
    
    def update_learning_rate(self, current_iter):
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
            
    @torch.no_grad()
    def feed_data(self, data):
        if self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel_1'].to(self.device)
            self.kernel2 = data['kernel_2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = transform.filter2D(self.gt_usm, self.kernel1)
            
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = transform.random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = transform.random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = transform.filter2D(out, self.kernel2)
            
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = transform.random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = transform.random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = transform.filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = transform.filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = transform.paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
    
    def save_trainer_state(self):
        pass
    
    def _train_step(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt
        
        # optimize net g
        for p in self.net_D.parameters():
            p.requires_grad = False
        
        self.optimizers_g.zero_grad()
        self.output = self.net_G(self.lq)
        
        loss_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            loss_g_pix = self.cri_pix(self.output, l1_gt)
            loss_g_total += loss_g_pix
            loss_dict['loss_g_pix'] = loss_g_pix
            
            loss_g_percep, loss_g_style = self.cri_perceptual(self.output, percep_gt)
            loss_g_total += loss_g_percep
            loss_dict['loss_g_percep'] = loss_g_percep

            if loss_g_style is not None:
                loss_g_total += loss_g_style
                loss_dict['loss_g_style'] = loss_g_style
            
            fake_g_pred = self.net_D(self.output)
            loss_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)    
            loss_dict['loss_g_gan'] = loss_g_gan
            
            loss_g_total.backward()
            self.optimizers_g.step()
        
        # optimize net d
        for p in self.net_D.parameters():
            p.requires_grad = True
        
        self.optimizers_d.zero_grad()
        
        # real
        real_d_pred = self.net_D(gan_gt)
        loss_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['loss_d_real'] = loss_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_d_real.backward()
        
        #fake
        fake_d_pred = self.net_D(self.output.detach().clone())
        loss_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['loss_d_fake'] = loss_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        loss_d_fake.backward()
        self.optimizers_d.step()
        
        # ema_decay
        
        # log_dict
    
    def fit(self):
        train_loader, val_loader, total_epochs, total_iters = d.create_train_val_dataloader(self.opt)
        start_epoch = 0
        current_iter = 0
        self.load_trainer_state()
        self.init_training_setting()
        for epoch in range(start_epoch, total_epochs+1):
            for data in train_loader:
                current_iter += 1
                self.update_learning_rate(current_iter)
                self.feed_data(data)
                self._train_step(current_iter)
            print(epoch)
                
                # log
                
                # validation
                
    
    def evaluate(self):
        pass

if __name__ == '__main__':
    
    import json
 
    with open('config.json', 'r') as f:
        opt = json.load(f)
    
    trainer = Trainer(opt)
    trainer.fit()