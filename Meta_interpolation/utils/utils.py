import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import time
import random
import numpy as np
from pathlib import Path
from einops import rearrange as rearrange

from models import base_models, model_original, model_attn, model_cd, compare_models
from utils.image_processing_utils import scale_img, scale_img_01

###########################################Base Tools#################################

def load_model(config, train_source_or_compare_model, source_config=None, device='cpu'):

    if train_source_or_compare_model == True:
        model = compare_models.__dict__[config.source_model]

        # 实例化模型
        model = model(config)
        model = model.to(device, dtype=config.data_type)
        return model

    else:
        ############################################定义SOURCE模型############################################
        # load pretrained model's config
        if os.path.exists(config.model_source_load_path):
            # print(device)
            source_checkpoint = torch.load(config.model_source_load_path, map_location=device)
        if source_config == None and config.source_model != 'coarse_to_refine':
            print(config.model_source_load_path)
            source_config = source_checkpoint["config"]
            # print("source_config",source_config)

        if source_config.source_model == 'coarse_to_refine':
            source_model = base_models.Refine
            source_model = source_model(1,1)
        else:
            source_model = compare_models.__dict__[source_config.source_model]

        # 实例化模型
        if config.source_model != 'coarse_to_refine':
            try:
                if not hasattr(source_config, "pool"):
                    source_config.pool = 'max'
                    source_config.norm = 'none'
                if not hasattr(source_config, "use_deform_layer_down"):
                    source_config.use_deform_layer_down = [0, 1, 2, 3]
                    source_config.use_deform_layer_up = [0, 1, 2]
                    source_config.use_attn_layer_down = [100]
                    source_config.use_attn_layer_up = [0, 1, 2]
                    source_config.attention_type = 'self'
                    source_config.reduction_ratio = 8
                if source_config.nb_filter != config.nb_filter:
                    source_config.nb_filter = config.nb_filter
                    print("use target nb_filter!!!")
                source_model = source_model(source_config)
                print("use source_config")
            except Exception as e:
                print("use source_config=config")
                source_model = source_model(config)
                print("一个错误：", e)

        ############################################定义SOURCE模型############################################

        ############################################定义TARGET模型############################################
        target_model = model_cd.__dict__[config.target_model]

        # 实例化模型
        target_model = target_model(config)
        ############################################定义TARGET模型############################################

        return source_model, target_model, source_config


# Gets the path to the project
def get_project_path():

    # 获取当前栈帧的信息
    stack = inspect.stack()

    # 获取调用该函数的文件路径（即栈中的第二个帧）
    file_path = stack[1].filename  # stack[1] 代表调用该函数的地方

    # 获取当前文件的文件夹（Project名）
    folder_name = os.path.basename(os.path.dirname(file_path))

    # 获取获取当前文件的文件夹的绝对路径
    folder_path = os.path.abspath(os.path.dirname(file_path))

    p_path = folder_path[:folder_path.index(folder_name)]
    # print(file_path, folder_name, folder_path, p_path)
    return p_path


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_trained_model(path, model, device):
    if "train_compare_models" in path:
        model.load_state_dict(torch.load(path, map_location=device)['model'])
    elif "train_original" in path:
        model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
    elif "train_cd" in path:
        model.load_state_dict(torch.load(path, map_location=device)['model_target'])
    elif 'train_sd' in path:
        model.load_state_dict(torch.load(path, map_location=device)['model_target'])
    else:
        print("path is not used: ", path)

    model = model.to(device)
    return model


def signal_to_noise(ori_data, imputed_data):
  denominator = torch.sum(ori_data**2)
  numerator = torch.sum((ori_data-imputed_data)**2)
  SNR = 10*torch.log10(denominator/(numerator+1e-9))
  return SNR

def peak_signal_to_noise(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    """
    Calculates the PSNR between 2 images, independent of batch size.

    This function computes the Mean Squared Error (MSE) over all pixels in the
    entire input tensors, regardless of their batch size. This ensures that
    the PSNR value is consistent across different batch sizes when evaluating
    the same set of images.

    PSNR is Peak Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{N}\sum_{i=0}^{N-1} [I_i - T_i]^2

    and :math:`N` is the total number of elements in the tensors, and
    :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Args:
        input (torch.Tensor): the input image with arbitrary shape :math:`(*)`.
                               This tensor contains all images to be evaluated.
        target (torch.Tensor): the labels image with arbitrary shape :math:`(*)`.
                               This tensor contains all target images, must have
                               the same shape as input.
        max_val (float): The maximum value in the input tensor (e.g., 1.0 for
                         normalized float images, 255.0 for 8-bit images).

    Return:
        torch.Tensor: the computed loss as a scalar.

    Examples:
        >>> # Example with batch size 1
        >>> ones_b1 = torch.ones(1, 3, 256, 256)
        >>> target_b1 = 1.2 * ones_b1
        >>> psnr_independent_of_batchsize(ones_b1, target_b1, 2.)
        tensor(20.0000)

        >>> # Example with batch size 4
        >>> ones_b4 = torch.ones(4, 3, 256, 256)
        >>> target_b4 = 1.2 * ones_b4
        >>> psnr_independent_of_batchsize(ones_b4, target_b4, 2.)
        tensor(20.0000)

        >>> # Example with mixed images (demonstrating consistent calculation)
        >>> img1 = torch.rand(1, 3, 64, 64)
        >>> img2 = torch.rand(1, 3, 64, 64)
        >>> img3 = torch.rand(1, 3, 64, 64)
        >>> img4 = torch.rand(1, 3, 64, 64)
        >>>
        >>> # Batch size 2
        >>> input_b2 = torch.cat([img1, img2], dim=0)
        >>> target_b2 = torch.cat([img1 * 0.9, img2 * 1.1], dim=0) # create some difference
        >>> psnr_b2 = psnr_independent_of_batchsize(input_b2, target_b2, 1.0)
        >>> print(f"PSNR with batch size 2: {psnr_b2.item():.4f}")
        >>>
        >>> # Batch size 4 (all images together)
        >>> input_b4 = torch.cat([img1, img2, img3, img4], dim=0)
        >>> target_b4 = torch.cat([img1 * 0.9, img2 * 1.1, img3 * 0.95, img4 * 1.05], dim=0)
        >>> psnr_b4 = psnr_independent_of_batchsize(input_b4, target_b4, 1.0)
        >>> print(f"PSNR with batch size 4: {psnr_b4.item():.4f}")
        >>>
        >>> # For comparison, let's calculate PSNR with batch size 1 for each image individually
        >>> psnr_img1 = psnr_independent_of_batchsize(img1, img1 * 0.9, 1.0)
        >>> psnr_img2 = psnr_independent_of_batchsize(img2, img2 * 1.1, 1.0)
        >>> psnr_img3 = psnr_independent_of_batchsize(img3, img3 * 0.95, 1.0)
        >>> psnr_img4 = psnr_independent_of_batchsize(img4, img4 * 1.05, 1.0)
        >>> print(f"PSNR for img1: {psnr_img1.item():.4f}")
        >>> print(f"PSNR for img2: {psnr_img2.item():.4f}")
        >>> print(f"PSNR for img3: {psnr_img3.item():.4f}")
        >>> print(f"PSNR for img4: {psnr_img4.item():.4f}")
        >>> avg_psnr_individual = torch.mean(torch.tensor([psnr_img1, psnr_img2, psnr_img3, psnr_img4]))
        >>> print(f"Average PSNR of individual images: {avg_psnr_individual.item():.4f}")
        >>> # Note: avg_psnr_individual might differ slightly from psnr_b4 due to floating point precision and order of operations.
        >>> # The key is that psnr_b4 should be consistent regardless of how the images were batched.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for input but got {type(input)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for target but got {type(target)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    # Calculate the Mean Squared Error over ALL elements in the tensors
    # .numel() gives the total number of elements in the tensor
    # We square the difference and sum them up, then divide by the total count.
    mse_val = torch.sum((input - target) ** 2) / input.numel()

    # Handle potential division by zero if mse_val is 0 (perfect reconstruction)
    # In this case, PSNR is considered infinite.
    if mse_val == 0:
        # A very large number to represent infinity, or torch.inf if desired
        return torch.tensor(float('inf'), device=input.device)

    # Calculate PSNR
    psnr_val = 10. * torch.log10(max_val ** 2 / mse_val)

    return psnr_val


def count_parameters(model):
    if model == None:
        return  0
    else:
        if not isinstance(model, list):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model[0].parameters() if p.requires_grad)*len(model)


def format_parameters_as_million(num_parameters):
    """将参数数量转换为以'M'为单位的字符串格式。

    Args:
    num_parameters (int): 模型的参数数量。

    Returns:
    str: 以'M'为单位的参数数量字符串。
    """
    # 将参数数量转换为百万单位
    million_parameters = num_parameters / 1_000_000
    # 格式化输出，保留两位小数
    formatted_parameters = f"{million_parameters:.4f}M"
    return formatted_parameters
def format_times_as_hms(elapsed_time):

    hours = int(elapsed_time // 3600)  # 整除得到小时
    minutes = int((elapsed_time % 3600) // 60)  # 取余数，再整除得到分钟
    seconds = elapsed_time % 60  # 剩下的秒数

    return hours, minutes, seconds

def torch_dtype(value: str) -> torch.dtype:
    """将字符串转换为 torch.dtype"""
    value = value.lower()
    if value == "float32":
        return torch.float32
    elif value == "float64":
        return torch.float64
    elif value == "float16":
        return torch.float16
    else:
        raise argparse.ArgumentTypeError(
            f"无效数据类型: {value}，可选值为 'float16' 或 'float32' 或 'float64'"
        )


def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


###########################################Base Tools#################################

##########################################Distill Tools###############################
def _get_num_features(model, config):

    cnum = config.cnum
    depth = config.depth
    nb_filter = config.nb_filter
    num_init_features = config.num_init_features
    block_config = config.block_config
    growth_rate = config.growth_rate

    if model == ('Face_Parsing_Network_deep'):
        return [item * cnum for item in [2, 4, 8, 8]]

    if model == ('Face_Parsing_Network_shallow'):
        return [item * cnum for item in [1, 2, 4, 8]]

    if model == ('Face_Parsing_Network'):
        return [item * cnum for item in [1, 2, 4, 8, 8]]

    if model == ('AN_net_CTR') or model == ('IN_net_CTR'):
        return [nb_filter[1], nb_filter[2],  nb_filter[3]]

    if model == ('IN_net_Unetplus') or model == ('AN_net_Unetplus'):
        return [nb_filter[0], nb_filter[1], nb_filter[2],  nb_filter[3]]
    if model == ('IN_net_Unetplus_shallow'):
        return [nb_filter[0], nb_filter[1], nb_filter[2]]
    if model == ('IN_net_Unetplus_deep'):
        return [nb_filter[1], nb_filter[2],  nb_filter[3]]
    if model == ('IN_net_Unetplus_attn') or model == ('AN_net_Unetplus_attn'):
        return [nb_filter[2], nb_filter[1],  nb_filter[0]]


    if model == ('AN_net_DGII') or model == ('IN_net_DGII'):
        # return [item * cnum for item in [2, 4]] + [4096] # [item * cnum for item in [2, 4, 8*8]]
        # return [item * cnum for item in [2, 4, 8*8]] # AN_net_DGII v4
        # return [item * cnum for item in [1, 2, 4, 8]] + [4096] # AN_net_DGII v3
        return [item * cnum for item in [2, 4, 8]] # AN_net_DGII v5 v5.3
        # return [item * cnum for item in [2, 4]] + [4096] # AN_net_DGII v5.5

    if model == ('AN_net_Unet_deep') or model == ('IN_net_Unet_deep'):
        if depth ==3:
            return [item * cnum for item in [2, 4]]
        elif depth ==4:
            return [item * cnum for item in [2, 4, 8]]     # output channels
        elif depth == 5:
            return [item * cnum for item in [2, 4, 8, 16]]
    if model == ('AN_net_Unet_shallow') or model == ('IN_net_Unet_shallow'):
        if depth ==3:
            return [item * cnum for item in [1, 2]]
        elif depth ==4:
            return [item * cnum for item in [1, 2, 4]]     # output channels
        elif depth == 5:
            return [item * cnum for item in [1, 2, 4, 8]]         # output channels
    if model == ('AN_net_Unet') or model == ('IN_net_Unet'):
        if depth ==3:
            return [item * cnum for item in [1, 2, 4]]
        elif depth ==4:
            return [item * cnum for item in [1, 2, 4, 8]]      # output channels
        elif depth == 5:
            return [item * cnum for item in [1, 2, 4, 8, 16]]      # output channels
        else:
            print("depth unkown:", depth)
    if model == ('AN_net_Unet_attn') or model == ('IN_net_Unet_attn'):
        if depth ==3:
                return [item * cnum for item in [2, 1]]
        elif depth ==4:
            return [item * cnum for item in [4, 2, 1]]         # atten input channels 256 128 64
        elif depth == 5:
            return [item * cnum for item in [8, 4, 2, 1]]      # output channels
        else:
            print("depth unkown:", depth)

    if model == ('AN_net_ResNet') or model == ('IN_net_ResNet'):
        return [cnum] +  [cnum for item in range(depth)] + [1]

    if model == ('AN_net_DenseNet') or model == ('IN_net_DenseNet') or model == ('AN_net_DenseNet_Skip') or model == ('IN_net_DenseNet_Skip'):
        channels = []
        num_features = num_init_features
        for num_layers in block_config:
            num_features = num_features + num_layers * growth_rate
            channels.append(num_features)
            num_features = num_features // 2

        return channels
    if  model.startswith('coarse_to_refine'):
        nb_filter = [32, 64, 128, 256, 512]
        return [nb_filter[0], nb_filter[1], nb_filter[2], nb_filter[3], nb_filter[4], nb_filter[3], nb_filter[2], nb_filter[1], nb_filter[0]]


# paper page3 r_θ
class FeatureMatching(nn.ModuleList):
    def __init__(self, config, source_model, target_model, pairs, attnd=False, version='v2'):
        super(FeatureMatching, self).__init__()

# weight:w_c
class WeightNetwork(nn.ModuleList):
    def __init__(self, config, source_model, pairs, attnd=False): # Feature_matching已经将通道数变为和target了
        self.config = config
        super(WeightNetwork, self).__init__()
 
def pair_to_list(pairs_str):

    pairs = []
    for pair in pairs_str.split(','):
        pairs.append((int(pair.split('-')[0]),
                        int(pair.split('-')[1])))
    return pairs

##########################################Distill Tools###############################

##########################################Other Tools ##################################


def delete_checkpt_files(target_dir):
    """
    永久删除目标目录及其子目录下所有名为checkpt.pth的文件
    :param target_dir: 要删除文件的根目录路径
    """
    deleted_files = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file == "checkpt.pth":
                file_path = os.path.join(root, file)
                try:
                    # 确保是文件且存在
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # 或使用os.unlink()
                        print(f"已删除：{file_path}")
                        deleted_files += 1
                except Exception as e:
                    print(f"删除失败：{file_path}，错误：{str(e)}")

    print(f"\n操作完成,共删除{deleted_files}个文件")

def count_files_in_path(path):
    try:
        file_count = 0
        for root, dirs, files in os.walk(path):
            file_count += len(files)
        return file_count
    except FileNotFoundError:
        print(f"错误：路径 {path} 未找到。")
        return 0
    except PermissionError:
        print(f"错误：没有权限访问路径 {path}。")
        return 0

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

##########################################Other Tools ##################################

if __name__ == '__main__':

    print(get_project_path())