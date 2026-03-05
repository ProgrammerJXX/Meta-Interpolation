import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as spectral_norm_ori
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from einops import rearrange as rearrange
import numbers
import torchvision.ops
from torchvision.ops import DeformConv2d
# from mmcv.ops import modulated_deform_conv2d
# from mmcv.ops import DeformConv2dPack as DCNv1
# from mmcv.ops import ModulatedDeformConv2dPack as DCNv2
import torch.utils.checkpoint as checkpoint

import math
from numbers import Number
from itertools import repeat
import numpy as np
try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass

################################################################COARSE TO REFINE################################################################

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, filtersize = 3, stride=1,padding=1,dilation=1, act_func=nn.ReLU(inplace = True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, filtersize, stride,padding,dilation=dilation, bias=True)
        self.bn1 = nn.BatchNorm2d(middle_channels,affine=True)#nn.InstanceNorm2d(middle_channels,affine=True)#
        self.conv2 = nn.Conv2d(middle_channels, out_channels, filtersize, stride,padding,dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels,affine=True)#nn.InstanceNorm2d(out_channels,affine=True)#

        #self.conv3 = nn.Conv2d(in_channels, middle_channels,  filtersize, stride,padding,dilation=dilation, bias=True)
        #self.bn3 = nn.InstanceNorm2d(middle_channels,affine=True)#nn.BatchNorm2d(out_channels,affine=True)#

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        #out3 = self.conv3(x)
        #out3 = self.bn3(out3)
        return out
    

class Coarse(nn.Module):
    def __init__(self, input_channels,output_channels):
        super().__init__()
        nb_filter = [16, 32, 64]

        self.pool = nn.AvgPool2d(2, 2)#nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0],filtersize = 5, stride=1,padding=2)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],filtersize = 5, stride=1,padding=2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],filtersize = 5, stride=1, padding=8, dilation=4)

        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],filtersize = 5, stride=1,padding=2)
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],filtersize = 5, stride=1,padding=2)

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = torch.sigmoid(self.final(x0_4))
        return output


class Refine(nn.Module):
    def __init__(self, input_channels,output_channels,ds=0):
        super().__init__()
        
        self.coarse = Coarse(1,1) # 输入/输出通道=1
        self.ds = ds
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.dropout0_1 = torch.nn.Dropout(0.5)
        # self.dropout1_1 = torch.nn.Dropout(0.5)
        # self.dropout2_1 = torch.nn.Dropout(0.5)
        # self.dropout3_1 = torch.nn.Dropout(0.5)
        # self.dropout0_2 = torch.nn.Dropout(0.5)
        # self.dropout1_2 = torch.nn.Dropout(0.5)
        # self.dropout2_2 = torch.nn.Dropout(0.5)
        # self.dropout0_3 = torch.nn.Dropout(0.5)
        # self.dropout1_3 = torch.nn.Dropout(0.5)
        # self.dropout0_4 = torch.nn.Dropout(0.5)
        #self.dropoutfinal = torch.nn.Dropout(0.5)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0],padding=4,dilation=4) # 分辨率H*W不变
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],padding=4,dilation=4)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],padding=4,dilation=4)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],padding=4,dilation=4)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],padding=4,dilation=4)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],padding=4,dilation=4)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)

        if ds:
            self.final1 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)



    def forward(self, input, mask):
        
        feat = []
        
        first_out = self.coarse(input)
        
        # Refinement
        second_in = input * mask + first_out * (1-mask)
        
        x0_0 = self.conv0_0(second_in)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        feat.append(x0_0)
        feat.append(x1_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        feat.append(x2_0)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        feat.append(x3_0)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        feat.append(x4_0)
        feat.append(x3_1)
        feat.append(x2_2)
        feat.append(x1_3)
        feat.append(x0_4)

        if self.ds:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            second_out = torch.sigmoid(self.final(x0_4))
            feature = [x0_0,x1_0,x2_0,x3_0,x4_0,x0_1,x1_1,x2_1,x3_1,x2_2,x1_2,x0_2,x1_3,x0_3,x0_4]
            # return first_out,second_out 
            return second_out, feat

    def forward_with_features(self, input, mask):
        return self.forward(input, mask) 


class Discriminator(nn.Module):
    def __init__(self, imgSize=256):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(1), # 输入张量的 所有四个边界（左、右、上、下）各填充 1个像素
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        #专门为marmousi模型数据加的一层
        # self.conv5 = nn.Sequential(
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0, bias=True),
        #     nn.InstanceNorm2d(512),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )
        self.output = nn.Sequential(
            nn.Linear(in_features=int(512*8*8), out_features=1, bias=True)#16*8#8*8  #512*16*7 #512*8*8
        )
        
    def forward(self,img, mask):
        
        feat = []
        
        # the input x should contain 2 channels because it is a combination of recon image and mask
        x = torch.cat((img, 1-mask), 1)       
        x = self.conv1(x) # 128->64
        feat.append(x)
        x = self.conv2(x) # 64->32
        feat.append(x)
        x = self.conv3(x) # 32->16
        feat.append(x)
        x = self.conv4(x) # 16->8
        feat.append(x)
        #x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        
        return x, feat
################################################################COARSE TO REFINE################################################################

################################################################ATTENTION################################################################

def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return spectral_norm_ori(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret

class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            output_nc = output_nc * 4
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass,)

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out

# https://github.com/lyndonzheng/Pluralistic-Inpainting/blob/master/model/base_function.py
class Auto_Attn(nn.Module):
    """ Short+Long attention Layer
    """

    def __init__(self, input_nc, reduction_ratio=4, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // reduction_ratio, kernel_size=1) # defult:input_nc // 4
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        # return out, attention
        return out, attention, proj_value
    
    
# https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, reduction_ratio=32):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction_ratio , kernel_size= 1)  # default in_dim//8
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction_ratio , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #在最后一个维度归一化  实际上是按列归一化行  得到权重  所以在最后一个维度上乘像素值
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        # return out,attention
        return out,attention, proj_value
    def forward_d(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention, proj_value
        # return out, attention
    
##MultiHeadAttention## 
 # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention 
    temperature:维度开根号
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn   


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module 
    n_head:头的数量（多头注意力机制）8
    d_model:输入向量最后一维长度 512
    d_k:每个头中 Query 和 Key 的向量维度 64
    d_v:每个头中 Value 的向量维度 64
    '''

    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn, v

##SPAMultiHeadAttention## 
#https://github.com/huangwenwenlili/spa-former/blob/main/model/netU_spa_former.py
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_C_M(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, LayerNorm_type='WithBias'):
        super(Attention_C_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_1 = self.norm1(x)
        g = self.gate(x_1)

        qkv = self.qkv_dwconv(self.qkv(x_1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b head c c
        #attn = attn.softmax(dim=-1)
        attn = F.relu(attn)
        #attn = torch.square(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out * g
        out = self.project_out(out)
        out = x + out
        # return out
        # return out, attn, x
        return out, attn, v

# https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out, y, x

# https://github.com/Andy-zhujunwen/danet-pytorch/blob/master/danet/danet.py
class ChannelAttention(nn.Module):
    """Channel attention module"""

    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.conv_c = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.ReLU(True))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x= self.conv_c(x)
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out, attention, feat_a

################################################################ATTENTION################################################################


################################################################DEFORMABLE################################################################
# https://github.com/dontLoveBugs/Deformable_ConvNet_pytorch/blob/master/network/deform_conv/deform_conv.py
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, lr_ratio=1.0):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        self.offset_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0)  # the offset learning are initialized with zero weights
        self.offset_conv.register_backward_hook(self._set_lr)

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.lr_ratio = lr_ratio

    # 对于卷积层y=wx+b，grad_input 包含三个元素grad_x, grad_weight, grad_bias
    # grad_output是grad_y
    def _set_lr(self, module, grad_input, grad_output):
        """将反向传播的梯度缩小为原来的lr_ratio倍"""
        # print('grad input:', grad_input)
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        # print('new grad input:', new_grad_input)
        return new_grad_input
    
    def forward(self, x):
        offset = self.offset_conv(x)
        dtype = torch.float32
        self.device = offset.device
        # print(", device=self.device", self.device)
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]).type_as(x).long()
        offsets_index.requires_grad = False
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        """
            if q is float, using bilinear interpolate, it has four integer corresponding position.
            The four position is left top, right top, left bottom, right bottom, defined as q_lt, q_rb, q_lb, q_rt
        """
        # (b, h, w, 2N)
        q_lt = p.detach().floor()

        """
            Because the shape of x is N, b, h, w, the pixel position is (y, x)
            *┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄→y
            ┊  .(y, x)   .(y+1, x)
            ┊   
            ┊  .(y, x+1) .(y+1, x+1)
            ┊
            ↓
            x

            For right bottom point, it'x = left top'y + 1, it'y = left top'y + 1
        """
        q_rb = q_lt + 1

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)

        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        """
            find p_y <= padding or p_y >= h - 1 - padding, find p_x <= padding or p_x >= x - 1 - padding
            This is to find the points in the area where the pixel value is meaningful.
        """
        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        # print('mask:', mask)

        floor_p = torch.floor(p)
        # print('floor_p = ', floor_p)

        """
           when mask is 1, take floor_p;
           when mask is 0, take original p.
           When thr point in the padding area, interpolation is not meaningful and we can take the nearest
           point which is the most possible to have meaningful value.
        """
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        """
            In the paper, G(q, p) = g(q_x, p_x) * g(q_y, p_y)
            g(a, b) = max(0, 1-|a-b|)
        """
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # print('g_lt size is ', g_lt.size())
        # print('g_lt unsqueeze size:', g_lt.unsqueeze(dim=1).size())

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        """
            In the paper, x(p) = ΣG(p, q) * x(q), G is bilinear kernal
        """
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        """
            x_offset is kernel_size * kernel_size(N) times x. 
        """
        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        """
            In torch 0.4.1 grid_x, grid_y = torch.meshgrid([x, y])
            In torch 1.0   grid_x, grid_y = torch.meshgrid(x, y)
        """
        p_n_x, p_n_y = torch.meshgrid(
            [torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
             torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).to(device=self.device, dtype=dtype)
        p_n.requires_grad = False
        # print('requires_grad:', p_n.requires_grad)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        """stride不默认为1的可变形卷积"""
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).to(device=self.device, dtype=dtype)
        p_0.requires_grad = False

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # print(p_0.device,  p_n.device,  offset.device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

# https://github.com/dontLoveBugs/Deformable_ConvNet_pytorch/blob/master/network/deform_conv/deform_conv_v2.py
class DeformConv2D_v2(nn.Module): # 生成b, c, h, w, 2N图像  N=k*k代表k*k卷积后的像素值
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True, lr_ratio = 0.1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2D_v2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0.5)
            self.m_conv.register_backward_hook(self._set_lr)

        self.lr_ratio = lr_ratio

    def _set_lr(self, module, grad_input, grad_output):
        # 在可变形卷积（Deformable Convolution）中，偏移量卷积层（self.p_conv）和调制卷积层（self.m_conv）比主卷积层的参数更新更缓慢
        # print('grad input:', grad_input)
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        # print('new grad input:', new_grad_input)
        return new_grad_input

    def forward(self, x):
        offset = self.p_conv(x)               # b, 2*N, h, w
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x)) # b, N, h, w

        dtype = torch.float32
        self.device = offset.device
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1) # (b, 1, h, w, N)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1) # 通道复制
            x_offset = x_offset*m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid([
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0).to(device=self.device, dtype=dtype)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # print("p_0_y.device", p_0_y.device)
        p_0 = torch.cat([p_0_x, p_0_y], 1).to(device=self.device, dtype=dtype)
        # print("p_0.device", p_0.device)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # print(dtype, p_0.device,  p_n.device,  offset.device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

# https://github.com/ChristophReich1996/Cell-DETR/blob/master/pixel_adaptive_convolution/pac.py
def _neg_idx(idx):
    return None if idx == 0 else -idx

def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output

def np_gaussian_2d(width, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d

class GaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff_sq = (cols - feat_0).pow(2)
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)
        output = torch.exp(-0.5 * diff_sq)
        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= \
            grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

        grad_input = F.fold(grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
                            (in_h, in_w), ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        return grad_input, None, None, None, None, None

class PacConv2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel

        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (in_mul_k, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,ojkl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            in_cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])

            grad_input = F.fold(grad_im2col_output,
                                ctx.input_size[:2], ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->ojkl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))

        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None

def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1,
                kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None

    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)

    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for (o, k, s, p, op, d) in
                      zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:]

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                          dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

    if native_impl:
        bs, k_ch, in_h, in_w = input.shape

        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = (int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2),
                              int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2))
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :,
                                    crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel,
                                stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])
        x = x - feat_0
        if kernel_type.find('_asym') >= 0:
            x = F.relu(x, inplace=True)
        # x.pow_(2)  # this causes an autograd issue in pytorch>0.4
        x = x * x
        if not channel_wise:
            x = torch.sum(x, dim=1, keepdim=True)
        if kernel_type == 'gaussian':
            x = torch.exp_(x.mul_(-0.5))  # TODO profiling for identifying the culprit of 5x slow down
            # x = torch.exp(-0.5 * x)
        elif kernel_type.startswith('inv_'):
            epsilon = 1e-4
            x = inv_alpha.view(1, -1, 1, 1, 1) \
                + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        else:
            raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert (smooth_kernel_type == 'none' and
                kernel_type == 'gaussian')
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise)

    if mask is not None:
        output = output * mask  # avoid numerical issue on masked positions

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        # output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output = output / (norm + empty_mask.clone().detach())
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask

def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False,
              native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols * kernel, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols * kernel, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    else:
        output = PacConv2dFn.apply(input, kernel, weight, bias, stride, padding, dilation, shared_filters)

    return output

class _PacConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, bias,
                 pool_only, kernel_type, smooth_kernel_type,
                 channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([p > d * (k - 1) / 2 for (p, d, k) in zip(padding, dilation, kernel_size)]):
            # raise ValueError('padding ({}) too large'.format(padding))
            pass  # TODO verify that this indeed won't cause issues
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0  # Eq.5, DenseCRF
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([.25, .5, .25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))

        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                        (-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                            (-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                       and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


class PacConv2d(_PacConvNd):
    r"""
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'
    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=False,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel=None, kernel=None, mask=None):
        if input_for_kernel==None:
            input_for_kernel = input_2d
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False, K=4):
        """
        动态卷积模块

        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - kernel_size: 卷积核大小
        - stride: 步长
        - padding: 填充
        - dilation: 膨胀率
        - bias: 是否使用偏置
        - K: 动态核个数（通常设置为4~8）
        """
        super(DynamicConv2D, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # K个卷积核参数（等价于 K 个普通卷积）
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_channels))
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 注意力模块生成 K 个权重系数（对每个样本）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, K, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B = x.size(0)
        # 生成注意力权重 [B, K, 1, 1]
        attention = self.attention(x).view(B, self.K, 1, 1, 1, 1)  # [B,K,1,1,1,1]
        attention = attention.expand(-1, -1, self.out_channels, self.in_channels, self.weight.size(3), self.weight.size(4))  # [B,K,C_out, C_in, k, k]
        # 动态卷积核融合：权重融合为 [B, C_out, C_in, k, k]
        agg_weight = (attention * self.weight.unsqueeze(0)).sum(1)  # sum over K
        
        # 合并批量维度以实现向量化计算
        x = x.view(1, B*self.in_channels, x.size(2), x.size(3))      # [1, B*C_in, H, W]
        agg_weight = agg_weight.view(B*self.out_channels, self.in_channels, 
                                    self.weight.size(3), self.weight.size(4))  # [B*C_out, C_in, k, k]
        
        # 批量卷积计算
        out = F.conv2d(x, agg_weight, bias=None, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=B)  # [1, B*C_out, H_out, W_out]
        out = out.view(B, self.out_channels, out.size(2), out.size(3))  # [B, C_out, H_out, W_out]
        
        # 处理偏置
        if self.bias is not None:
            agg_bias = (attention.squeeze() @ self.bias).view(B, self.out_channels, 1, 1)
            out += agg_bias
        return out




################################################################DEFORMABLE################################################################

# 双分支融合模块
class DualBranchFusion(nn.Module):
    # def __init__(self, dim, LocalDetailBranch, Attention, num_heads=8):
    def __init__(self, dim):
        super().__init__()
        # self.local_branch = LocalDetailBranch(dim)
        # self.semantic_branch = Attention(dim, num_heads)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, local_feat, global_feat):
        # local_feat = self.local_branch(x)
        # global_feat = self.semantic_branch(x)
        fused = torch.cat([local_feat, global_feat], dim=1)  # 通道拼接
        out = self.fusion(fused)
        return out

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
    
# Define the model
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
 
    def forward(self, input_tensor, mask):
        return self.model(input_tensor, mask)