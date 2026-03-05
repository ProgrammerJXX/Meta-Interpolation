import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.vgg import Vgg16, VGG19

loss_keys = ('l1_loss', 'ssim_loss', 'mse_loss', 'tv_loss', 'dis_loss', 'percept_loss', 'style_loss', 'l2_reg_loss')

class TVLoss(nn.Module):
    def __init__(self, reduction='mean', TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.reduction = reduction

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        if self.reduction == 'mean':
            return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        elif self.reduction == 'sum':
            return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w)

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class LossCalculator(nn.Module):
    def __init__(self, config, loss_objects, loss_weight, device='cuda:1'):
        super(LossCalculator, self).__init__()
        self.loss_objects = loss_objects
        self.loss_weight = loss_weight
        self.device = device
        self.config = config

    def forward(self, state, img_ori, img_out, mask=None, model=None,test=False):
        # 初始化所有损失项为0
        losses = {
            'l1_loss': torch.tensor(0.0, device=self.device),
            'ssim_loss': torch.tensor(0.0, device=self.device),
            'mse_loss': torch.tensor(0.0, device=self.device),
            'tv_loss': torch.tensor(0.0, device=self.device),
            'dis_loss': torch.tensor(0.0, device=self.device),
            'percept_loss': torch.tensor(0.0, device=self.device),
            'style_loss': torch.tensor(0.0, device=self.device),
            'l2_reg_loss': torch.tensor(0.0, device=self.device)
        }
        total_loss = torch.tensor(0.0, device=self.device)
        batch_size = img_ori.shape[0]
        mask_L1loss = torch.exp(1+abs(img_ori-img_out))
        weight_iter = iter(self.loss_weight)

        # 计算各个损失
        if 'L1Loss' in self.loss_objects:
            if self.config.l1_type == 'v1' or test:
                losses['l1_loss'] = next(weight_iter) * self.loss_objects['L1Loss'](img_out, img_ori).mean()
            elif self.config.l1_type == 'v2':
                base = (self.loss_objects['L1Loss'](mask * img_out, mask * img_ori)).mean()*32 #* batch_size
                term = self.config.alpha * (self.loss_objects['L1Loss']((1-mask)*img_out, (1-mask)*img_ori)).mean()*32 #* batch_size
                losses['l1_loss'] = next(weight_iter) * (base + term)
            elif self.config.l1_type == 'v3':
                base = (self.loss_objects['L1Loss'](mask * img_out, mask * img_ori)).mean()*32 #* batch_size
                term = 6 * (mask_L1loss * self.loss_objects['L1Loss']((1-mask)*img_out, (1-mask)*img_ori)).mean()*32 #* batch_size
                losses['l1_loss'] = next(weight_iter) * (base + term)
            else:
                losses['l1_loss'] = next(weight_iter) * self.loss_objects['L1Loss'](img_out, img_ori).mean()

        if 'SSIMLoss' in self.loss_objects:
            losses['ssim_loss'] = next(weight_iter) * self.loss_objects['SSIMLoss'](img_ori, img_out)*32 #* batch_size

        if 'MSELoss' in self.loss_objects:
            losses['mse_loss'] = next(weight_iter) * self.loss_objects['MSELoss'](img_out, img_ori)

        if 'TVLoss' in self.loss_objects:
            losses['tv_loss'] = next(weight_iter) * self.loss_objects['TVLoss'](img_out)

        if 'DisLoss' in self.loss_objects:
            losses['dis_loss'] = next(weight_iter) * self.loss_objects['DisLoss'](img_out, img_ori, mask)

        if 'PerceptLoss' in self.loss_objects:
            losses['percept_loss'] = next(weight_iter) * self.loss_objects['PerceptLoss'](img_out, img_ori)
            
        if 'StyleLoss' in self.loss_objects:
            losses['style_loss'] = next(weight_iter) * self.loss_objects['StyleLoss'](img_out * (1 - mask), img_ori * (1 - mask))


        if 'L2_reg' in self.loss_objects:
            # print("self.loss_objects['L2_reg'](model)", self.loss_objects['L2_reg'](model))
            losses['l2_reg_loss'] = next(weight_iter) * self.loss_objects['L2_reg'](model)

        # 累加总损失并更新state
        total_loss = sum(losses.values())
        state.update(losses)  # 单行替代多行赋值
        
        return total_loss, state



class DiscriminatorLoss(nn.Module):
    
    def __init__(self, D, dl_num: List[int] = [1, 2, 3, 4], freeze_params=True, use_eval=True):
        super(DiscriminatorLoss, self).__init__()
        self.dl_num = dl_num
        self.D = D
        if freeze_params:
            # 冻结D的所有参数
            for param in self.D.parameters():
                param.requires_grad = False  # 关键修改点
        if use_eval:
            self.D = self.D.eval()

    def forward(self, fake_img, real_img, mask=None):
        with torch.no_grad():
            d, real_feature = self.D(real_img.detach(), mask)
        d, fake_feature = self.D(fake_img, mask)
        
        D_penalty = 0
        for f_id in self.dl_num:
            D_penalty += F.l1_loss(fake_feature[int(f_id-1)],
                                              real_feature[int(f_id-1)])
        return D_penalty

    def set_ftr_num(self, dl_num):
        self.dl_num = dl_num

     
class PerceptLoss(nn.Module):
    
    def __init__(self, loss_type='L1', pl_num=3, vgg_type='vgg16', weights=[1.0, 1.0, 1.0, 1.0, 1.0], device=0, dtype=torch.float32):
        super(PerceptLoss, self).__init__()
        if vgg_type == 'vgg16':
            self.add_module('vgg', Vgg16().to(device, dtype=dtype))
        elif vgg_type == 'vgg19':
            self.add_module('vgg', VGG19().to(device, dtype=dtype))
            # print("vgg_type", vgg_type)
        self.pl_num = pl_num
        self.vgg_type = vgg_type
        self.weights = weights
        if loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'L2':
            self.criterion = nn.MSELoss()
    
    def forward(self, fake_img, real_img):

        real_img = real_img.repeat(1, 3, 1, 1)
        fake_img = fake_img.repeat(1, 3, 1, 1)
        # print('fake_img.shape', fake_img.shape)

        image_loss = self.vgg_loss(fake_img, real_img)
        # print('image_loss', image_loss)
        return image_loss
    
    def vgg_loss(self, fake_img, real_img):
        output_feature = self.vgg(fake_img)
        target_feature = self.vgg(real_img)
        
        loss = []
        if self.vgg_type == 'vgg16':
            loss.append(self.weights[0] * self.criterion(output_feature.relu1_2, target_feature.relu1_2))
            loss.append(self.weights[1] * self.criterion(output_feature.relu2_2, target_feature.relu2_2))
            loss.append(self.weights[2] * self.criterion(output_feature.relu3_3, target_feature.relu3_3))
            loss.append(self.weights[3] * self.criterion(output_feature.relu4_3, target_feature.relu4_3))
        
        elif self.vgg_type == 'vgg19':

            loss.append(self.weights[0] * self.criterion(output_feature['relu1_1'], target_feature['relu1_1']))
            loss.append(self.weights[1] * self.criterion(output_feature['relu2_1'], target_feature['relu2_1']))
            loss.append(self.weights[2] * self.criterion(output_feature['relu3_1'], target_feature['relu3_1']))
            loss.append(self.weights[3] * self.criterion(output_feature['relu4_1'], target_feature['relu4_1']))
            loss.append(self.weights[4] * self.criterion(output_feature['relu5_1'], target_feature['relu5_1']))
        
        D_penalty = 0
        for f_id in self.pl_num:
            D_penalty  += loss[f_id-1]
        return D_penalty

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    https://github.com/huangwenwenlili/spa-former/blob/main/model/loss.py
    """

    def __init__(self, device=0, dtype=torch.float32):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19().to(device, dtype=dtype))
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


# 定义L2正则化损失函数
class L2_regularization_loss(nn.Module):
    def __init__(self, p=2):
        super(L2_regularization_loss, self).__init__()
        self.p = p
    def forward(self, model):
        l2_reg = 0
        param_count = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=self.p) ** 2
            param_count += param.numel()  # 计算当前参数的元素个数
        # 计算均值
        l2_reg_mean = l2_reg / param_count if param_count > 0 else 0
        return l2_reg_mean
    
    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]