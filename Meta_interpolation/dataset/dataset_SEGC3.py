import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset

from dataset.dataset_base import BaseDataset
from dataset.mask import random_seismic_sampler_1, continuous_seismic_sampler_1


class Dataset_SEGC3(BaseDataset):
    def __init__(self, config, img_paths, sampler, mode, dtype=torch.float32, add_gussian=False):
        """
        Initialize the Dataset_SEGC3 class.

        This function sets up the dataset with the given configuration, image paths, sampler, and mode.

        Args:
            config (object): Configuration object containing dataset parameters.
            img_paths (str or list): Path(s) to the image files or directory.
            sampler (str): The type of sampling method to use.
            mode (str): The mode of operation ('train' or 'test').
            add_gussian (bool, optional): Flag to add Gaussian noise. Defaults to False.

        Returns:
            None

        Note:
            This method initializes various attributes of the Dataset_SEGC3 class,
            including configuration, image paths, seed, sampler, mode, and noise level.
        """
        super(Dataset_SEGC3, self).__init__() # 没有的话不继承BaseDataset的方法

        self.config = config

        if mode == 'train':
            self.img_paths = self.list_txt_files(img_paths)[:25000]  # 35000 25000
            if isinstance(config.data_use, float) and config.data_use != -1.0:
                # 使用固定的随机数种子
                random.seed(42)  #  <---  这里设置随机数种子，你可以选择任何整数
                print("data_use_seed:", 42)
                random.shuffle(self.img_paths)  # 随机打乱数据
                data_num = int(config.data_use * len(self.img_paths))
                self.img_paths = self.img_paths[:data_num]
        else:
            self.img_paths = self.list_txt_files(img_paths)[:10000]  # 17280 10000
            
        self.seed = self.config.seed
        self.sampler = sampler
        self.mode = mode
        self.dtype = dtype
        self.noise_level_img = self.config.noise_level_img


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        if self.mode == 'train':
            flag = False
        else:
            flag = True
        
        patch_path = self.img_paths[idx]

        # --------------------------------
        # get patch_H
        # --------------------------------

        patch_name = os.path.basename(patch_path)
        patch_H = np.loadtxt(patch_path)

        transform_image = self.get_transform(patch_H, (self.config.resolution_h, self.config.resolution_w))
        patch_H = transform_image(patch_H)
        _, traindim_time, traindim_trace = patch_H.size()

        # add AWGN
        if self.noise_level_img:
            patch_H_noise = patch_H * 2 - 1
            patch_H_noise = self.add_gussian_noise(patch_H_noise, self.noise_level_img, idx, flag) 
            patch_H_noise = patch_H_noise / 2 + 0.5
        else:
            patch_H_noise = patch_H

        # --------------------------------
        # get patch_mask
        # --------------------------------
        sampler = self.sampler_select(self.sampler)
        patch_mask, ratio = sampler(traindim_time, traindim_trace, self.config.missing_p, idx, flag, self.config.missing_p_continuous)

        # --------------------------------
        # get patch_L
        # --------------------------------
        patch_mask = patch_mask.unsqueeze(0)
        patch_L = patch_H_noise.clone()  # Numpy.copy()
        patch_L[patch_mask == 0] = 0
        patch_L = patch_L

        # dtype
        patch_H = patch_H.to(dtype=self.dtype) 
        patch_L = patch_L.to(dtype=self.dtype) 
        patch_H_noise = patch_H_noise.to(dtype=self.dtype)
        patch_mask = patch_mask.to(dtype=self.dtype)

        # Return images names  masks missing_ratio
        return patch_H, patch_L, patch_H_noise, patch_name, patch_mask, ratio

    def sampler_select(self, sampler_name):

        if sampler_name == 'random':
            sampler = self.random_seismic_sampler
        elif sampler_name == 'continus':
            sampler = self.continus_seismic_sampler
        elif sampler_name == 'multiple':
            sampler = self.modified_multiple_seismic_sampler
        elif sampler_name == 'half':
            sampler = self.binary_seismic_sampler_half
        elif sampler_name == 'regular':
            sampler = self.regular_seismic_sampler

        return sampler