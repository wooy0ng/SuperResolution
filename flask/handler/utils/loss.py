from collections import OrderedDict
from click import style

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg as vgg

NAMES = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
    'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
    'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]

def l1_loss(pred, target, weight, reduction='mean'):
    loss = F.l1_loss(pred, target, reduction='none')
    if weight is not None:
        loss = loss * weight
    if weight is None or reduction == 'sum':
        loss = loss.mean()
    elif reduction == 'mean':
        weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight
    return loss

class L1loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        
    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

class VGGExtractor(nn.Module):
    def __init__(self,
                 layer_name_list,
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGExtractor, self).__init__()
        
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        self.names = NAMES
        
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
                
        vgg_net = vgg.vgg19(pretrained=True)
        features = vgg_net.features[:max_idx + 1]
        modified_net = OrderedDict()
        
        for k, v in zip(self.names, features):
            if 'pool' in k:
                if remove_pooling:
                    continue
                else:
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v
        
        self.vgg_net = nn.Sequential(modified_net)
        
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        
        return output

class PerceptualLoss(nn.Module):
    def __init__(self,
                 layer_weights,
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGExtractor(
            layer_name_list=list(layer_weights.keys()),
            use_input_norm=use_input_norm,
            range_norm=range_norm
        )
        
        if criterion== 'l1':
            self.criterion = torch.nn.L1Loss()
    
    def forward(self, x, gt):
        
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        
        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        
        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        
        return percep_loss, style_loss

class GanLoss(nn.Module):
    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GanLoss, self).__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.BCEWithLogitsLoss()
        
    def get_target_label(self, input, target_is_real):
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val
    
    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
    
        return loss if is_disc else loss * self.loss_weight