from typing import Any, List, Tuple, Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

from models.base_models import DeformConv2D, DeformConv2D_v2, PacConv2d, DynamicConv2D
from models.base_models import Auto_Attn, Self_Attn, MultiHeadAttention, Attention_C_M, ECA, ChannelAttention, DualBranchFusion
from utils.image_processing_utils import scale_img

##################################Face_Parsing_Network######################################
class Conv2dBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='none', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        
        
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
            
            
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)
            
            
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           # output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)
   
   
    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
##################################Face_Parsing_Network######################################


class Deform_Conv2dBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='bn', activation='relu', deform_conv_type='v1'):
        super(Deform_Conv2dBlock, self).__init__()
        self.use_bias = True
        
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

            
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


        # initialize convolution
        if deform_conv_type == 'v1':
            self.conv = DeformConv2D(input_dim, output_dim, kernel_size, padding, stride, bias=None, lr_ratio = 0.1)
        elif deform_conv_type == 'v2':
            self.conv = DeformConv2D_v2(input_dim, output_dim, kernel_size, padding, stride)
        elif deform_conv_type == 'pac':
            self.conv = PacConv2d(input_dim, output_dim, kernel_size, padding)
        elif deform_conv_type == 'dy':
            self.conv = DynamicConv2D(input_dim, output_dim, kernel_size, padding=1, K=4)
        else:
            self.conv = None
            print("deform_conv_type:",deform_conv_type)

    def forward(self, x):

        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Attention_Block(nn.Module):
    def __init__(self, input_dim, num_heads=1, bias=False, LayerNorm_type='WithBias', attention_type='v1', reduction_ratio=8):
        super(Attention_Block, self).__init__()
        
        # initialize attention
        if attention_type == 'auto':
            self.attn = Auto_Attn(input_dim, reduction_ratio)
        elif attention_type == 'self':
            self.attn = Self_Attn(input_dim, reduction_ratio)
        elif attention_type == 'mult':
            self.attn = MultiHeadAttention(n_head=num_heads)
        elif attention_type == 'spa':
            self.attn = Attention_C_M(input_dim, num_heads, bias, LayerNorm_type)
        elif attention_type == 'eca':
            self.attn = ECA(kernel_size=3)
        elif attention_type == 'ca':
            self.attn = ChannelAttention(dim=input_dim)
        else:
            self.attn = None
            print("attention_type:",attention_type)
            
    def forward(self, x):
        x, attn, v = self.attn(x)
        return x, attn, v

class Single_layer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, dilation=1, pad_type='zeros', activation='none'):
        super(Single_layer, self).__init__()
        
        self.use_bias = True

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
            
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                padding=padding, padding_mode=pad_type, dilation=dilation,
                                bias=self.use_bias)

   
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x
    
##################################Face_Parsing_Network######################################
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, filtersize = 3, stride=1,padding=1,dilation=1, act_func=nn.ReLU(inplace = True), use_attention=False, attention_type='spa', num_heads=1, reduction_ratio=16):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(in_channels, middle_channels, filtersize, stride,padding,dilation=dilation, bias=True)
        self.bn1 = nn.BatchNorm2d(middle_channels,affine=True)#nn.InstanceNorm2d(middle_channels,affine=True)#
        self.conv2 = nn.Conv2d(middle_channels, out_channels, filtersize, stride,padding,dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels,affine=True)#nn.InstanceNorm2d(out_channels,affine=True)#
        if use_attention:
            self.attn = Attention_Block(self.out_channels, attention_type=attention_type, num_heads=num_heads, reduction_ratio=reduction_ratio)
            # self.fuse = DualBranchFusion(self.out_channels)

    def forward(self, x, mask=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)
        
        if self.use_attention:
            x_attn, attn, v = self.attn(out) 
            x = x_attn
            return out, attn, v
        else:
            return out
    
    
##################################Unet######################################
   
def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    elif mode == 'bilinear':
        return nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True))
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)
   
    
class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, norm='none',pooling=True, pool='max',activation='relu', use_deform_conv=False,deform_conv_type='v2',DRU_Net=False, SEU_Net=False, MS_Unet=False, reduction=16):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.pooling = pooling
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.use_deform_conv = use_deform_conv
        self.DRU_Net = DRU_Net
        self.SEU_Net = SEU_Net
        self.MS_Unet = MS_Unet
        if DRU_Net:
            self.conv1 = Deform_Conv2dBlock(self.in_channels, self.out_channels, 3, 1, 1, deform_conv_type=deform_conv_type, norm='none', activation='none')
            self.conv2 = Deform_Conv2dBlock(self.out_channels, self.out_channels, 3, 1, 1, deform_conv_type=deform_conv_type, norm='none', activation='none') 
            self.res_block = resnet_block(self.out_channels)
        elif SEU_Net:
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.conv2 = conv3x3(self.out_channels, self.out_channels)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_channels // reduction, self.out_channels),
                nn.Sigmoid()
                )
        elif MS_Unet:
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.conv2 = conv3x3(self.out_channels, self.out_channels)
            self.conv3 = conv3x3(self.out_channels, self.out_channels)
            
        elif use_deform_conv:
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.conv2 = Deform_Conv2dBlock(self.out_channels, self.out_channels, 3, 1, 1, deform_conv_type=deform_conv_type, norm='none', activation='none')
        else:
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(self.out_channels,affine=True)
            self.bn2 = nn.BatchNorm2d(self.out_channels,affine=True)
            if self.MS_Unet:
                self.bn3 = nn.BatchNorm2d(self.out_channels,affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm2d(self.out_channels,affine=True)
            self.bn2 = nn.InstanceNorm2d(self.out_channels,affine=True)
            if self.MS_Unet:
                self.bn3 = nn.InstanceNorm2d(self.out_channels,affine=True)
        if self.pooling:
            if pool == 'max':
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            elif pool == 'avg':
                self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x, mask=None):
        
        if self.norm != 'none':
            x = self.activation(self.bn1(self.conv1(x)))
        else:
            x = self.activation(self.conv1(x))
        if self.DRU_Net:
            x = self.res_block(x)

        if self.norm != 'none':
            x_ = self.activation(self.bn2(self.conv2(x)))
            if self.MS_Unet:
                x = self.activation(self.bn3(self.conv3(x_)))
            _, _, h, w = x_.size()
            if mask is not None and self.use_deform_conv:
                mask = scale_img(mask, [h, w])
                x  = x + x_ * (1-mask)
            else:
                x = x_
        else:
            x = self.activation(self.conv2(x))
            if self.MS_Unet:
                x = self.activation(self.conv3(x))
            
        if self.SEU_Net:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            x = x * y.expand_as(x)
            
        before_pool = x
        
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose', norm='none',activation='relu',use_deform_conv=False,deform_conv_type='v2', use_attention=False, attention_type='spa', num_heads=1, MS_Unet=False, reduction_ratio=8):
        super(UpConv, self).__init__()

        self.in_channels = in_channels  # >out_channels
        self.out_channels = out_channels 
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.norm = norm
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.use_deform_conv = use_deform_conv
        self.use_attention = use_attention
        self.MS_Unet = MS_Unet

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            if self.up_mode == 'transpose':
                self.conv1 = conv3x3(
                    2*self.out_channels, self.out_channels)
            elif self.up_mode == 'bilinear':
                # print(self.in_channels//2*3, self.out_channels)
                self.conv1 = conv3x3(
                    self.in_channels//2*3, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        if use_deform_conv:
            self.conv2 = Deform_Conv2dBlock(self.out_channels, self.out_channels, 3, 1, 1, deform_conv_type=deform_conv_type, norm='none', activation='none')
        else:
            self.conv2 = conv3x3(self.out_channels, self.out_channels)
            if MS_Unet:
                self.conv3 = conv3x3(self.out_channels, self.out_channels)
        if use_attention:
            self.attn = Attention_Block(self.out_channels, attention_type=attention_type, num_heads=num_heads, reduction_ratio=reduction_ratio)
            self.fuse = DualBranchFusion(self.out_channels)

        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(self.out_channels,affine=True)
            self.bn2 = nn.BatchNorm2d(self.out_channels,affine=True)
            if MS_Unet:
                self.bn3 = nn.BatchNorm2d(self.out_channels,affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm2d(self.out_channels,affine=True)
            self.bn2 = nn.InstanceNorm2d(self.out_channels,affine=True)
            if MS_Unet:
                self.bn3 = nn.InstanceNorm2d(self.out_channels,affine=True)
            
    def forward(self, from_down, from_up, mask=None):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway         before_pool
            from_up: upconv'd tensor from the decoder pathway  x
        """
        attn = None
        v = None
        
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            # print(from_down.shape, from_up.shape)
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        if self.norm != 'none':
            x = self.activation(self.bn1(self.conv1(x)))
            x_ = self.activation(self.bn2(self.conv2(x)))
            if self.MS_Unet:
                x = self.activation(self.bn3(self.conv3(x_)))
            if self.use_attention and self.use_deform_conv and (mask is not None):
                _, _, h, w = x_.size()
                x_attn, attn, v = self.attn(x) 
                mask = scale_img(mask, [h, w])
                x  = x + x_ * (1-mask)
                x = self.fuse(x, x_attn)
            elif self.use_attention:
                x_attn, attn, v = self.attn(x_) 
                x = x_attn
            else:
                x = x_
        # if self.norm != 'none':
        #     x = self.activation(self.bn1(self.conv1(x)))
        #     x_ = self.activation(self.bn2(self.conv2(x)))
        #     if self.MS_Unet:
        #         x = self.activation(self.bn3(self.conv3(x_)))
        #     if self.use_attention:
        #         x_attn, attn, v = self.attn(x)  
        #     _, _, h, w = x_.size()
        #     if mask is not None and self.use_deform_conv:
        #         mask = scale_img(mask, [h, w])
        #         x  = x + x_ * (1-mask)
        #         if self.use_attention:
        #             x = self.fuse(x, x_attn)
        #     elif self.use_attention:
        #         x = x_attn
        #     else:
        #         x = x_
        else:
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            if self.MS_Unet:
                x = self.activation(self.conv3(x))
        if self.use_attention:
            return x, attn, v
        else:    
            return x

# DRU_Net resblock
class resnet_block(nn.Module):
    
    def __init__(self, channels):
        super(resnet_block, self).__init__()

        self.channels = channels  # >out_channels

        self.conv1 = conv3x3(channels, channels)
        self.conv2 = conv3x3(channels, channels)
        self.conv3 = conv3x3(channels, channels)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = residual + out
        return out

def make_layers(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, bias=True, batch_norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        #layer.append(nn.InstanceNorm2d(out_channels, affine=True))
        layer.append(nn.BatchNorm2d(out_channels,affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,batch_normid=False):
        super(ResidualBlock, self).__init__()
        self.conv_layer1 = make_layers(in_channels=in_channels, out_channels=in_channels, 
                            kernel_size=3, stride=1, padding=1, batch_norm=batch_normid,activation=True,is_relu=True)
        
        self.conv_layer2 = make_layers(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                            stride=1, padding=1, batch_norm=False,activation=False,is_relu=False)

    def forward(self, x):
        out = self.conv_layer2(self.conv_layer1(x))

        return x + out

class BottleneckDecoderBlock_ins(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
          # 类似于densenet的拼接模块  
        super(BottleneckDecoderBlock_ins, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.InstanceNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.InstanceNorm2d(in_planes + 2 * 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.InstanceNorm2d(in_planes + 3 * 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.InstanceNorm2d(in_planes + 4 * 32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.InstanceNorm2d(in_planes + 5 * 32)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn7 = nn.InstanceNorm2d(inter_planes)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_planes + 5 * 32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        # out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

# 用了数据变为(B, out_planes, H*2, W*2)
class TransitionBlock_ins(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
          # 类似于denset _Transition 但是没有池化操作  卷积变成反卷积
        super(TransitionBlock_ins, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2,
                               padding=1,output_padding=1, bias=True)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return out#F.upsample_nearest(out, scale_factor=2)  

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)
    

##################################Unet######################################



