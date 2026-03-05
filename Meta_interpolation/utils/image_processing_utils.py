import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/lyndonzheng/Pluralistic-Inpainting/blob/034a7588957adff1edb027af44f48a35d1c949d3/util/task.py#L94"

def scale_img(img, size): # 
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img

def scale_img_01(img, size):
    # 显式设置align_corners=False
    scaled_img = F.interpolate(img.float(), size=size, mode='bilinear', align_corners=False)
    scaled_img = (scaled_img > 0.5).float()
    return scaled_img
