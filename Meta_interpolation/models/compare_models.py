import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_gen_functions import gen_single_layer, gen_conv, gen_deform_conv, gen_vggblock, gen_downconv, gen_upconv, gen_resnet_block, gen_basic_block, gen_en_dense_block, gen_en_transition_block, gen_de_bottleneck_block, gen_de_transition_block, gen_residual_block
from models.model_blocks import conv3x3, conv1x1, make_layers

__all__ = ['AN_net_Unet', 'Unet', 'ResNet', 'CAE', 'DRU_Net', 'SEU_Net', 'MS_Unet', 'AN_net_Unetplus']

# https://github.com/jaxony/unet-pytorch/blob/master/model.py    
# AN_net_Unet v2
class AN_net_Unet(nn.Module):
    def __init__(self, config): 
        super(AN_net_Unet, self).__init__()
        self.config = config
        # 需要2的倍数构成的图像大小
        self.input_dim = config.input_dim      # input_channels
        self.output_num = config.input_dim     # output_channels
        self.start_filts = config.cnum
        self.up_mode = 'transpose'
        self.merge_mode = 'concat'
        self.depth = config.depth
        self.use_deform_layer_down = config.use_deform_layer_down
        self.use_deform_layer_up = config.use_deform_layer_up
        self.use_attn_layer_down = config.use_attn_layer_down
        self.use_attn_layer_up = config.use_attn_layer_up
            
        self.down_convs = []
        self.up_convs = []
        self.attn = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.input_dim if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            if (config.use_deform_conv and i in self.use_deform_layer_down):
                down_conv = gen_downconv(ins, outs, norm=config.norm, pooling=pooling, pool=config.pool,activation=config.activation, use_deform_conv=config.use_deform_conv,deform_conv_type=config.deform_conv_type)
            else:
                down_conv = gen_downconv(ins, outs, norm=config.norm, pooling=pooling, pool=config.pool,activation=config.activation)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            if (config.use_deform_conv and i in self.use_deform_layer_up) and (config.use_attention and i in self.use_attn_layer_up):
                up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm,activation=config.activation,use_deform_conv=config.use_deform_conv,deform_conv_type=config.deform_conv_type, use_attention=config.use_attention, attention_type=config.attention_type, num_heads=config.num_heads, reduction_ratio=config.reduction_ratio)
            elif  (not config.use_attention) and (config.use_deform_conv and i in self.use_deform_layer_up):
                up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm,activation=config.activation,use_deform_conv=config.use_deform_conv,deform_conv_type=config.deform_conv_type)
            elif (not config.use_deform_conv) and (config.use_attention and i in self.use_attn_layer_up):
                up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm,activation=config.activation, use_attention=config.use_attention, attention_type=config.attention_type, num_heads=config.num_heads, reduction_ratio=config.reduction_ratio)
            elif (config.use_deform_conv) and (config.use_attention and i in self.use_attn_layer_up):
                up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm,activation=config.activation, use_attention=config.use_attention, attention_type=config.attention_type, num_heads=config.num_heads, reduction_ratio=config.reduction_ratio)
            else:
                 up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm,activation=config.activation)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.output_num)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
    
    def forward(self, x, mask=None):
        feat = []
        attn = []
        v = []
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x,mask)
            # print(i, self.depth)
            encoder_outs.append(before_pool)
            feat.append(before_pool)
            
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            if (self.config.use_attention and i in self.use_attn_layer_up):
                x, attn_, v_ = module(before_pool, x ,mask)  # 池化前的特征 和 池化卷积后的特征
                attn.append(attn_)
                v.append(v_)
            else:
                x = module(before_pool, x ,mask)  # 池化前的特征 和 池化卷积后的特征
            feat.append(x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        
        # feat = encoder_outs[1:self.depth]
        # print(len(encoder_outs), self.depth-1, len(feat))
        if self.config.use_attention:
            return torch.sigmoid(x), feat, attn, v
        else:  
             return torch.sigmoid(x), feat
    
    def forward_with_features(self, x, mask=None):
        return self.forward(x,mask)


# AN_net_Unet v2
class Unet(nn.Module):
    def __init__(self, config): 
        super(Unet, self).__init__()
        # 需要2的倍数构成的图像大小
        self.input_dim = config.input_dim      # input_channels
        self.output_num = config.input_dim     # output_channels
        self.start_filts = config.cnum
        self.up_mode = 'bilinear' # 'transpose' 'bilinear' or 'bilinear_conv'
        self.merge_mode = 'concat'
        self.depth = config.depth
        
        self.down_convs = []
        self.up_convs = []
        self.attn = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.input_dim if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False

            down_conv = gen_downconv(ins, outs, norm=config.norm, pooling=pooling, pool=config.pool)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            
            up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.output_num)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
    
    def forward(self, x, mask=None):
        feat = []
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            # print(i, self.depth)
            encoder_outs.append(before_pool)
            feat.append(before_pool)
            
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)  # 池化前的特征 和 池化卷积后的特征
            feat.append(x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        
        # feat = encoder_outs[1:self.depth]
        # print(len(encoder_outs), self.depth-1, len(feat))
        return torch.sigmoid(x), feat
    
    def forward_with_features(self, x):
        return self.forward(x)
    
# AN_net_ResNet
class ResNet(nn.Module):
    def __init__(self, config):
            super(ResNet, self).__init__()
            
            input_dim = config.input_dim # 1
            in_channels = config.cnum    # 64
            depth = config.depth         # 3
            
            self.conv1 = make_layers(input_dim, in_channels, stride=1, padding=1, batch_norm=True, activation=True, is_relu=True)
            res_layers = []
            for i in range(depth):
                res_layers.append(gen_residual_block(in_channels=in_channels,batch_normid=True))
            self.res_blocks = nn.Sequential(*res_layers)
            self.conv2 = make_layers(in_channels, input_dim, stride=1, padding=1, batch_norm=False, activation=False)
    
    def forward(self, x, mask=None):
        
        feat = []
        
        out = self.conv1(x)
        feat.append(out)
        
        for i, layer in enumerate(self.res_blocks):
            out = layer(out)
            feat.append(out)
            
        out = self.conv2(out)
        feat.append(out)
        
        out = torch.sigmoid(out)+x

        return out, feat
    
    def forward_with_features(self, x):
        return self.forward(x)

# CAE
class CAE(nn.Module):
    def __init__(self, config):
        super(CAE, self).__init__()
        input_dim = config.input_dim # 1
        
        self.conv1=make_layers(input_dim, 32, stride=1, kernel_size=5, padding=2, batch_norm = False, activation=True, is_relu=True)
        self.conv2=make_layers(32, 16, stride=2, kernel_size=5, padding=2, batch_norm = False, activation=True, is_relu=True)
        self.conv3=make_layers(16, 16, stride=2, kernel_size=5, padding=2, batch_norm = False, activation=True, is_relu=True)

        self.deconv1=nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2=nn.ConvTranspose2d(16, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3=nn.ConvTranspose2d(32, input_dim, kernel_size=5, stride=1, padding=2, output_padding=0)

    def forward(self, x, mask=None):
        
        feat = []
        out = self.conv1(x)
        # n, c, h, w = out.shape[0],out.shape[1],out.shape[2],out.shape[3]
        # m = nn.LayerNorm([c,h,w],  elementwise_affine=True).cuda(0)
        # out = m(out)
        # n = nn.ReLU() 
        # out = n(out)
        out = self.conv2(out)
        # n, c, h, w = out.shape[0],out.shape[1],out.shape[2],out.shape[3]
        # m = nn.LayerNorm([c,h,w],  elementwise_affine=True).cuda(0)
        # out = m(out)
        # n = nn.ReLU() 
        # out = n(out)
        out = self.conv3(out)
        # n, c, h, w = out.shape[0],out.shape[1],out.shape[2],out.shape[3]
        # m = nn.LayerNorm([c,h,w],  elementwise_affine=True).cuda(0)
        # out = m(out)
        # n = nn.ReLU() 
        # out = n(out)

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = torch.sigmoid(self.deconv3(out))

        return out, feat
    def forward_with_features(self, x):
        return self.forward(x)


class DRU_Net(nn.Module):
    def __init__(self, config): 
        super(DRU_Net, self).__init__()
        # 需要2的倍数构成的图像大小
        self.input_dim = config.input_dim      # input_channels
        self.output_num = config.input_dim     # output_channels
        self.start_filts = config.cnum
        self.up_mode = 'transpose'
        self.merge_mode = 'concat'
        self.depth = config.depth
        
        self.down_convs = []
        self.up_convs = []
        self.attn = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.input_dim if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            if i == self.depth-1:
                down_conv = gen_downconv(ins, outs, pooling=pooling, DRU_Net=True)
            else:
                down_conv = gen_downconv(ins, outs, pooling=pooling)
            
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            
            up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.output_num)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
    
    def forward(self, x, mask=None):
        feat = []
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            # print(i, self.depth)
            encoder_outs.append(before_pool)
            feat.append(before_pool)
            
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)  # 池化前的特征 和 池化卷积后的特征
            feat.append(x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        
        # feat = encoder_outs[1:self.depth]
        # print(len(encoder_outs), self.depth-1, len(feat))
        return torch.sigmoid(x), feat
    
    def forward_with_features(self, x):
        return self.forward(x)
    
# https://github.com/miraclewkf/SENet-PyTorch/blob/master/se_resnet.py
class SEU_Net(nn.Module):
    def __init__(self, config): 
        super(SEU_Net, self).__init__()
        # 需要2的倍数构成的图像大小
        self.input_dim = config.input_dim      # input_channels
        self.output_num = config.input_dim     # output_channels
        self.start_filts = config.cnum
        self.up_mode = 'transpose'
        self.merge_mode = 'concat'
        self.depth = config.depth
        
        self.down_convs = []
        self.up_convs = []
        self.attn = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.input_dim if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            if i == self.depth-1:
                down_conv = gen_downconv(ins, outs, pooling=pooling, SEU_Net=True, reduction=16)
            else:
                down_conv = gen_downconv(ins, outs, pooling=pooling)
            
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            
            up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.output_num)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
    
    def forward(self, x, mask=None):
        feat = []
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            # print(i, self.depth)
            encoder_outs.append(before_pool)
            feat.append(before_pool)
            
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)  # 池化前的特征 和 池化卷积后的特征
            feat.append(x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        
        # feat = encoder_outs[1:self.depth]
        # print(len(encoder_outs), self.depth-1, len(feat))
        return torch.sigmoid(x), feat
    
    def forward_with_features(self, x):
        return self.forward(x)
    
# coarse_to_refine
# -----------------------------------------------
#                   Generator
# -----------------------------------------------
# Input: masked image + mask
# Output: filled image
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, filtersize = 3, stride=1,padding=1,dilation=1, act_func=nn.ReLU(inplace = True), norm='bn'):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, filtersize, stride,padding,dilation=dilation, bias=True)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(middle_channels,affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm2d(middle_channels,affine=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, filtersize, stride,padding,dilation=dilation, bias=True)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm2d(middle_channels,affine=True)
        elif norm == 'in':
            self.bn2 = nn.InstanceNorm2d(middle_channels,affine=True)

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
    

class Coarse(nn.Module): # Unet
    def __init__(self, input_channels,output_channels, norm='bn', pool='avg'):
        super().__init__()
        nb_filter = [16, 32, 64]
        if pool == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        elif pool == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0],filtersize = 5, stride=1,padding=2, norm=norm)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],filtersize = 5, stride=1,padding=2, norm=norm)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],filtersize = 5, stride=1, padding=8, dilation=4, norm=norm)

        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],filtersize = 5, stride=1,padding=2, norm=norm)
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],filtersize = 5, stride=1,padding=2, norm=norm)

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = torch.sigmoid(self.final(x0_4))
        return output
class Refine(nn.Module): # Unet + Unet++
    def __init__(self, input_channels,output_channels,ds=0, norm='bn', pool='avg'):
        super().__init__()
        
        self.coarse = Coarse(1,1, norm=norm, pool=pool) # 输入/输出通道=1
        self.ds = ds
        nb_filter = [32, 64, 128, 256, 512]

        if pool == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        elif pool == 'max':
            self.pool = nn.MaxPool2d(2, 2)
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
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0],padding=4,dilation=4, norm=norm) # 分辨率H*W不变
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],padding=4,dilation=4, norm=norm)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],padding=4,dilation=4, norm=norm)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],padding=4,dilation=4, norm=norm)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],padding=4,dilation=4, norm=norm)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4, norm=norm)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4, norm=norm)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4, norm=norm)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],padding=4,dilation=4, norm=norm)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4, norm=norm)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4, norm=norm)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4, norm=norm)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4, norm=norm)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4, norm=norm)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4, norm=norm)

        if ds:
            self.final1 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)



    def forward(self, input, mask):
        first_out = self.coarse(input)
        
        # Refinement
        second_in = input * mask + first_out * (1-mask)
        
        x0_0 = self.conv0_0(second_in)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.ds:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            second_out = torch.sigmoid(self.final(x0_4))
            feature = [x0_0,x1_0,x2_0,x3_0,x4_0,x0_1,x1_1,x2_1,x3_1,x2_2,x1_2,x0_2,x1_3,x0_3,x0_4]
            return first_out,second_out 
        
class Discriminator(nn.Module):
    def __init__(self, imgSize=256, config=None):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(1),
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
        # self.output = nn.Sequential(
        #     nn.Linear(in_features=int(512*16*7), out_features=1, bias=True)#16*8#8*8  #512*16*7 #512*8*8
        # )
        if config is not None:
            if config.dataset == 'MAVG' or config.dataset == 'MAVO':
                self.output = nn.Sequential(
                    nn.Linear(in_features=int(512*16*7), out_features=1, bias=True)#16*8#8*8  #512*16*7 #512*8*8
                    )
            elif config.dataset == 'SEGC3' or config.dataset == 'Model94':
                self.output = nn.Sequential(
                    nn.Linear(in_features=int(512*8*8), out_features=1, bias=True)#16*8#8*8  #512*16*7 #512*8*8
                    )
            else:
                print("config.dataset:", config.dataset)
        else:
            self.output = nn.Sequential(
                nn.Linear(in_features=int(512*16*7), out_features=1, bias=True)#16*8#8*8  #512*16*7 #512*8*8
            )

    def forward(self,img, mask):
        # the input x should contain 2 channels because it is a combination of recon image and mask
        x = torch.cat((img, 1-mask), 1)       
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

class MS_Unet(nn.Module):
    def __init__(self, config): 
        super(MS_Unet, self).__init__()
        self.config = config
        # 需要2的倍数构成的图像大小
        self.input_dim = config.input_dim      # input_channels
        self.output_num = config.input_dim     # output_channels
        self.start_filts = config.cnum
        self.up_mode = 'transpose'
        self.merge_mode = 'concat'
        self.depth = config.depth
        
        self.down_convs = []
        self.up_convs = []
            
        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.input_dim if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            down_conv = gen_downconv(ins, outs, norm=config.norm, pooling=pooling, pool=config.pool,activation=config.activation, MS_Unet=True)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            up_conv = gen_upconv(ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode, norm=config.norm,activation=config.activation, MS_Unet=True)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.output_num)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
    
    def forward(self, x, mask=None):
        feat = []
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x,mask)
            # print(i, self.depth)
            encoder_outs.append(before_pool)
            feat.append(before_pool)
            
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x ,mask)  # 池化前的特征 和 池化卷积后的特征
            feat.append(x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        
        # feat = encoder_outs[1:self.depth]
        # print(len(encoder_outs), self.depth-1, len(feat))
        return torch.sigmoid(x), feat
    
    def forward_with_features(self, x, mask=None):
        return self.forward(x,mask)
    
    
    
class AN_net_Unetplus(nn.Module):
    def __init__(self, config, ds=0): # config is not used..
        super(AN_net_Unetplus, self).__init__()
        # 需要2的倍数构成的图像大小
        self.config = config
        self.input_dim = config.input_dim      # input_channels
        self.output_num = config.input_dim     # output_channels
        nb_filter = config.nb_filter
        self.nb_filter = nb_filter
        self.ds = ds
        self.use_deform_layer_down = config.use_deform_layer_down
        self.use_deform_layer_up = config.use_deform_layer_up
        self.use_attn_layer_down = config.use_attn_layer_down
        self.use_attn_layer_up = config.use_attn_layer_up
        
        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = gen_vggblock(self.input_dim, nb_filter[0], nb_filter[0],padding=4,dilation=4)   # 分辨率H*W不变
        self.conv1_0 = gen_vggblock(nb_filter[0], nb_filter[1], nb_filter[1],padding=4,dilation=4)
        self.conv2_0 = gen_vggblock(nb_filter[1], nb_filter[2], nb_filter[2],padding=4,dilation=4)
        self.conv3_0 = gen_vggblock(nb_filter[2], nb_filter[3], nb_filter[3],padding=4,dilation=4)
        # self.conv4_0 = gen_vggblock(nb_filter[3], nb_filter[4], nb_filter[4],padding=4,dilation=4)

        self.conv0_1 = gen_vggblock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)
        self.conv1_1 = gen_vggblock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4)
        if len(self.use_attn_layer_up)>= 1:
            self.conv2_1 = gen_vggblock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4, use_attention=config.use_attention, attention_type=config.attention_type, num_heads=config.num_heads, reduction_ratio=config.reduction_ratio)
        else:
            self.conv2_1 = gen_vggblock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4)
        # self.conv3_1 = gen_vggblock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],padding=4,dilation=4)

        self.conv0_2 = gen_vggblock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)
        if len(self.use_attn_layer_up)>= 2:
            self.conv1_2 = gen_vggblock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4, use_attention=config.use_attention, attention_type=config.attention_type, num_heads=config.num_heads, reduction_ratio=config.reduction_ratio)
        else:
            self.conv1_2 = gen_vggblock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4)
        # self.conv2_2 = gen_vggblock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2],padding=4,dilation=4)
        if len(self.use_attn_layer_up)>= 3:
            self.conv0_3 = gen_vggblock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4, use_attention=config.use_attention, attention_type=config.attention_type, num_heads=config.num_heads, reduction_ratio=config.reduction_ratio)
        else:
            self.conv0_3 = gen_vggblock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)
        # self.conv1_3 = gen_vggblock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1],padding=4,dilation=4)

        # self.conv0_4 = gen_vggblock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0],padding=4,dilation=4)


        self.final = nn.Conv2d(nb_filter[0], self.output_num, kernel_size=1)



    def forward(self, input, mask=None):
        
        feat = []
        attn = []
        v = []
        
        x0_0 = self.conv0_0(input, mask)
        feat.append(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0), mask)
        feat.append(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0), mask)
        feat.append(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0), mask)
        feat.append(x3_0)
        if self.config.use_attention and len(self.use_attn_layer_up)>=1:
            x2_1, attn_, v_ = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1), mask)
            attn.append(attn_)
            v.append(v_)
            if len(self.use_attn_layer_up)>=2:
                # print(self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1), mask).shape)
                x1_2, attn_, v_ = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1), mask)
                attn.append(attn_)
                v.append(v_)
            else:
                x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1), mask)
            if len(self.use_attn_layer_up)>=3:
                x0_3, attn_, v_ = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1), mask)
                attn.append(attn_)
                v.append(v_)
            else:
                x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1), mask)
        else:
            
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1), mask)
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1), mask)
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1), mask)

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))


        # out = torch.sigmoid(self.final(x0_4))
        out = torch.sigmoid(self.final(x0_3))
        
        if self.config.use_attention:
            return out, feat, attn, v
        else:  
            return out, feat

    def forward_with_features(self, input, mask=None):
        return self.forward(input, mask)