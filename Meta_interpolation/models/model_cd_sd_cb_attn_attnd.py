import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

from models.base_models import Self_Attn
from models.model_blocks import Single_layer
from models.model_gen_functions import gen_conv, gen_deform_conv
    
    
class IN_net(nn.Module):
    def __init__(self, config, input_dim, cnum, deform_conv_type): # config is not used..
        super(IN_net, self).__init__()
        # 需要2的倍数构成的图像大小
        self.input_dim = input_dim      # input_channels
        self.output_num = input_dim     # output_channels
        self.cnum = cnum
    
        # 128 * 128 * cnum
        self.conv1_1 = gen_conv(self.input_dim, self.cnum, 7, 1, 1,)
        self.conv1_2 = gen_conv(self.cnum, self.cnum, 3, 2, 1)
        self.conv1_3 = gen_deform_conv(self.cnum, self.cnum, 3, 1, 1, deform_conv_type=deform_conv_type)
        
        # 64 * 64 * cnum
        self.conv2_1 = gen_conv(self.cnum, self.cnum * 2, 3, 1, 1)
        self.conv2_2 = gen_conv(self.cnum * 2, self.cnum * 2, 3, 2, 1)
        self.conv2_3 = gen_deform_conv(self.cnum * 2, self.cnum * 2, 3, 1, 1, deform_conv_type=deform_conv_type)
        
        # 32 * 32 * cnum
        self.conv3_1  = gen_conv(self.cnum * 2, self.cnum * 4, 3, 1, 1)
        self.conv3_2 = gen_conv(self.cnum * 4, self.cnum * 4, 3, 1, 1)
        self.conv3_3 = gen_conv(self.cnum * 4, self.cnum * 4, 3, 2, 1)
        self.conv3_4 = gen_deform_conv(self.cnum * 4, self.cnum * 4, 3, 1, 1, deform_conv_type=deform_conv_type)
        
        # 16 * 16 * cnum
        self.conv4_1 = gen_conv(self.cnum * 4, self.cnum * 8, 3, 1, 1)
        self.conv4_2 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv4_3 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 2, 1)
        self.conv4_4 = gen_deform_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1, deform_conv_type=deform_conv_type)
        #--self.pool4

        # 8 * 8 * cnum
        self.conv5_1 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv5_2 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv5_3 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 2, 1)
        self.conv5_4 = gen_deform_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1, deform_conv_type=deform_conv_type)
        #--self.pool5

        # 4 * 4 * cnum
        self.conv6_1 = gen_conv(self.cnum * 8, 4096, 3, 1, 1)
        self.conv6_2 = gen_deform_conv(4096, 4096, 3, 1, 1)
        self.attn6 = Self_Attn(4096)


        #-- self.deconv6
        self.conv7_1  = gen_conv(4096, self.cnum * 8, 5, 1, 2)
        self.conv7_2 = gen_deform_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1, deform_conv_type=deform_conv_type)
        self.attn7 = Self_Attn(self.cnum * 8)
        #-- dropout6

        # 8 * 8 * cnum
        #-- self.deconv7
        self.conv8_1 = gen_conv(self.cnum * 8, self.cnum * 8, 5, 1, 2)
        self.conv8_2 = gen_deform_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1, deform_conv_type=deform_conv_type)
        self.attn8 = Self_Attn(self.cnum * 8)
        # -- dropout7

        # 16 * 16 * cnum
        #--self.deconv8
        self.conv9_1 = gen_conv(self.cnum * 8, self.cnum * 8, 5, 1, 2)
        self.conv9_2 = gen_deform_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1, deform_conv_type=deform_conv_type)
        self.attn9 = Self_Attn(self.cnum * 8)
        # -- dropout8

        # 32 * 32 * cnum
        #-- self.deconv9
        self.conv10_1 = gen_conv(self.cnum * 8, self.cnum * 4, 5, 1, 2)
        self.conv10_2 = gen_deform_conv(self.cnum * 4, self.cnum * 4, 3, 1, 1, deform_conv_type=deform_conv_type)
        self.attn10 = Self_Attn(self.cnum * 4)
        # -- dropout9

        # 64 * 64 * cnum
        #--self.deconv10
        self.conv11_1 = gen_conv(self.cnum * 4, self.cnum * 2, 3, 1, 1)
        self.conv11_2 = gen_deform_conv(self.cnum * 2, self.cnum * 2, 3, 1, 1, deform_conv_type=deform_conv_type)
        self.attn11 = Self_Attn(self.cnum * 2)
        # -- dropout10

        # 128 * 128 * cnum
        #--self.deconv11
        self.conv12_1 = gen_conv(self.cnum * 2, self.cnum, 3, 1, 1)
        # -- dropout11

        # 128 * 128 * C+1 (FINAL)
        self.h_out = gen_conv(self.cnum, self.output_num, 3, 1, 1)         
        
    
    def forward(self, x, mask):
        
        Proj_value = []
        Attention = []
        
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)


        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)


        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        if self.deform_conv_type == 'pac':
            x_cb = self.conv5_4(x, x)
        else:
            x_cb = self.conv5_4(x)
        x = x + x_cb * mask
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv6
        x = self.conv6_1(x)
        #print(x.shape)

        # conv7
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv7_1(x)
        #print(x.shape)
        if self.deform_conv_type == 'pac':
            x_cb = self.conv7_2(x, x)
        else:
            x_cb = self.conv7_2(x)
        x_attn, proj_value, attention = self.attn7.forward_d(x)
        x_cb = x + x_cb * mask
        x = x_attn + x_cb
        x = F.dropout2d(x)

        # conv8
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8_1(x)
        
        #print(x.shape)

        # conv9
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv9_1(x)
        #print(x.shape)

        # conv10
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv10_1(x)
        #print(x.shape)

        # conv11
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv11_1(x)
        #print(x.shape)

        # conv12
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv12_1(x)
        #print(x.shape)

        # h_out
        x_out = self.h_out(x)
        #print(x_out.shape)

        return x_out

    def forward_with_features(self, x, mask):

        Attention = []
        Proj_value = []
        feat = []
        
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)


        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)


        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)

        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        if self.deform_conv_type == 'pac':
            x_cb = self.conv5_4(x, x)
        else:
            x_cb = self.conv5_4(x)
        x = x + x_cb * mask
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)

        # conv6
        x = self.conv6_1(x)
        #print(x.shape)

        # conv7
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv7_1(x)
        #print(x.shape)
        if self.deform_conv_type == 'pac':
            x_cb = self.conv7_2(x, x)
        else:
            x_cb = self.conv7_2(x)
        x_attn, proj_value, attention = self.attn7.forward_d(x)
        x_cb = x + x_cb * mask
        x = x_attn + x_cb
        Attention.append(attention)
        Proj_value.append(proj_value)
        x = F.dropout2d(x)
        feat.append(x)
        
        # conv8
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8_1(x)
        #print(x.shape)

        # conv9
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv9_1(x)
        #print(x.shape)

        # conv10
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv10_1(x)
        #print(x.shape)

        # conv11
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv11_1(x)
        #print(x.shape)

        # conv12
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv12_1(x)
        #print(x.shape)

        # h_out
        x_out = self.h_out(x)
        #print(x_out.shape)

        return x_out, feat, Attention, Proj_value
        
        
class AN_net(nn.Module):
    def __init__(self, config, input_dim, cnum): # config is not used..
        super(AN_net, self).__init__()
        # 需要2的倍数构成的图像大小
        self.input_dim = input_dim      # input_channels
        self.output_num = input_dim     # output_channels
        self.cnum = cnum
    
        # 128 * 128 * cnum
        self.conv1_1 = gen_conv(self.input_dim, self.cnum, 7, 1, 1,)
        self.conv1_2 = gen_conv(self.cnum, self.cnum, 3, 2, 1)
        
        # 64 * 64 * cnum
        self.conv2_1 = gen_conv(self.cnum, self.cnum * 2, 3, 1, 1)
        self.conv2_2 = gen_conv(self.cnum * 2, self.cnum * 2, 3, 2, 1)
        
        # 32 * 32 * cnum
        self.conv3_1  = gen_conv(self.cnum * 2, self.cnum * 4, 3, 1, 1)
        self.conv3_2 = gen_conv(self.cnum * 4, self.cnum * 4, 3, 1, 1)
        self.conv3_3 = gen_conv(self.cnum * 4, self.cnum * 4, 3, 2, 1)
        
        # 16 * 16 * cnum
        self.conv4_1 = gen_conv(self.cnum * 4, self.cnum * 8, 3, 1, 1)
        self.conv4_2 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv4_3 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 2, 1)
        #--self.pool4

        # 8 * 8 * cnum
        self.conv5_1 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv5_2 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 1, 1)
        self.conv5_3 = gen_conv(self.cnum * 8, self.cnum * 8, 3, 2, 1)
        #--self.pool5

        # 4 * 4 * cnum
        self.conv6_1 = gen_conv(self.cnum * 8, 4096, 3, 1, 1)


        #-- self.deconv6
        self.conv7_1  = gen_conv(4096, self.cnum * 8, 5, 1, 2)
        self.attn7 = Self_Attn(self.cnum * 8)
        #-- dropout6

        # 8 * 8 * cnum
        #-- self.deconv7
        self.conv8_1 = gen_conv(self.cnum * 8, self.cnum * 8, 5, 1, 2)
        # -- dropout7

        # 16 * 16 * cnum
        #--self.deconv8
        self.conv9_1 = gen_conv(self.cnum * 8, self.cnum * 8, 5, 1, 2)
        # -- dropout8

        # 32 * 32 * cnum
        #-- self.deconv9
        self.conv10_1 = gen_conv(self.cnum * 8, self.cnum * 4, 5, 1, 2)
        # -- dropout9

        # 64 * 64 * cnum
        #--self.deconv10
        self.conv11_1 = gen_conv(self.cnum * 4, self.cnum * 2, 3, 1, 1)
        # -- dropout10

        # 128 * 128 * cnum
        #--self.deconv11
        self.conv12_1 = gen_conv(self.cnum * 2, self.cnum, 3, 1, 1)
        # -- dropout11

        # 128 * 128 * C+1 (FINAL)
        self.h_out = gen_conv(self.cnum, self.output_num, 3, 1, 1)        
        
    
    def forward(self, x):
        
        Attention = []
        
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)


        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)


        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)

        # conv6
        x = self.conv6_1(x)
        #print(x.shape)

        # conv7
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv7_1(x)
        x, proj_value, attention = self.attn7.forward_d(x)
        
        #print(x.shape)
        x = F.dropout2d(x)

        # conv8
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8_1(x)
        #print(x.shape)

        # conv9
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv9_1(x)
        #print(x.shape)

        # conv10
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv10_1(x)
        #print(x.shape)

        # conv11
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv11_1(x)
        #print(x.shape)

        # conv12
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv12_1(x)
        #print(x.shape)

        # h_out
        x_out = self.h_out(x)
        #print(x_out.shape)

        return x_out
    
    def forward_with_features(self, x):
        
        Attention = []
        Proj_value = []
        feat = []
        
        # conv1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)


        # conv2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)


        # conv3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)

        # conv4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)

        # conv5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        #print(x.shape)
        #x = nn.MaxPool2d(x)
        feat.append(x)

        # conv6
        x = self.conv6_1(x)
        #print(x.shape)

        # conv7
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv7_1(x)
        x = self.conv7_1(x)
        x, proj_value, attention = self.attn7.forward_d(x)
        Attention.append(attention)
        Proj_value.append(proj_value)
        feat.append(x)
        #print(x.shape)
        x = F.dropout2d(x)

        # conv8
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8_1(x)
        #print(x.shape)

        # conv9
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv9_1(x)
        #print(x.shape)

        # conv10
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv10_1(x)
        #print(x.shape)

        # conv11
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv11_1(x)
        #print(x.shape)

        # conv12
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv12_1(x)
        #print(x.shape)

        # h_out
        x_out = self.h_out(x)
        #print(x_out.shape)

        return x_out, feat, Attention, Proj_value