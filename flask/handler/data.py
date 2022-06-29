import math
import os
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from utils import transform

class EsrganDataset(Dataset):
    def __init__(self, dir_path, opt):
        self.dir_path = dir_path
        self.img_list = os.listdir(dir_path)
        
        self.prep_opt = opt['preprocessing']
        
        # 1st degradation
        self.blur_kernel_size = self.prep_opt['blur_kernel_size']
        self.kerenl_list = self.prep_opt['kernel_list']
        self.kernel_prob = self.prep_opt['kernel_prob']
        self.blur_sigma = self.prep_opt['blur_sigma']
        self.betag_range = self.prep_opt['betag_range']
        self.betap_range = self.prep_opt['betap_range']
        self.sinc_prob = self.prep_opt['sinc_prob']
        
        # 2nd degradation
        self.blur_kernel_size2 = self.prep_opt['blur_kernel_size2']
        self.kerenl_list2 = self.prep_opt['kernel_list2']
        self.kernel_prob2 = self.prep_opt['kernel_prob2']
        self.blur_sigma2 = self.prep_opt['blur_sigma2']
        self.betag_range2 = self.prep_opt['betag_range2']
        self.betap_range2 = self.prep_opt['betap_range2']
        self.sinc_prob2 = self.prep_opt['sinc_prob2']
        
        self.final_sinc_prob = self.prep_opt['final_sinc_prob']
        
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
    
    def __getitem__(self, index):
        gt_path = os.path.join(self.dir_path, self.img_list[index])
        img_gt = cv2.imread(gt_path)
        img_gt = transform.augment(img_gt)
        img_gt = transform.random_crop(img_gt)
        
        kernel_1 = transform.generate_kernel(
            self.kernel_range, self.sinc_prob, self.kerenl_list, self.kernel_prob, self.blur_sigma, self.betag_range, self.betap_range
            )
        
        kernel_2 = transform.generate_kernel(
            self.kernel_range, self.sinc_prob2, self.kerenl_list2, self.kernel_prob2, self.blur_sigma2, self.betag_range2, self.betap_range2
            )
        
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = transform.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
            
        img_gt = img_gt.astype('float32')
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float()
        
        kernel_1 = torch.FloatTensor(kernel_1)
        kernel_2 = torch.FloatTensor(kernel_2)
        
        return_d = {'gt': img_gt, 'kernel_1':kernel_1, 'kernel_2':kernel_2, 'sinc_kernel':sinc_kernel, 'gt_path':gt_path}
        return return_d
    
    def __len__(self):
        return len(self.img_list)
    

def build_dataset(dataset_opt):
    dir_path = dataset_opt['root_path']
    dataset = EsrganDataset(dir_path, dataset_opt)
    return dataset

def build_dataloader(dataset, phase, dataset_opt, seed=None):
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        dataloader_arg = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    elif phase in ['val', 'test']:
        dataloader_arg = dict(dataset=dataset, batch_size=1, shuffle=False)
        
    return DataLoader(**dataloader_arg)

def create_train_val_dataloader(opt):
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['dataset'].items():
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(
                train_set,
                phase,
                dataset_opt,
                seed=opt['manual_seed']
            )
            num_iter_per_epoch = math.ceil(
                len(train_set) / (dataset_opt['batch_size'])
            )
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,
                phase,
                dataset_opt,
                seed=opt['manual_seed']
            )
    return train_loader, val_loader, total_epochs, total_iters
    


if __name__ == '__main__':
    import json
    
    with open('config.json', 'r') as f:
        opt = json.load(f)

    train_loader, val_loader = create_train_val_dataloader(opt)
    for data in train_loader:
        print(data['gt'].shape, data['kernel_1'].shape, data['kernel_2'].shape, data['gt_path'])
        break