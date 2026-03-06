import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os
import random

class BaseDataset(data.Dataset):
    """_summary_

    Args:
        data (_type_): _description_
    """
    
    def __init__(self, dtype=torch.dtype):
        super(BaseDataset, self).__init__()
        self.dtype = dtype
        
    def get_transform(self, patch, target_size, toCrop=False, pad=False):
        
        h, w = patch.shape
        
        transform_list = []
        transform_list.append(transforms.ToTensor())
        if toCrop =='CenterCrop':
            transform_list.append(transforms.CenterCrop(target_size))
        if pad == True:
            transform_list.append(transforms.Pad(padding=(0, 0, target_size-w, 0)))

        return transforms.Compose(transform_list)
    
    def add_gussian_noise(self, img, noise_level_img, index, flag=False):  # __类中的私有方法或属性  普通的可以调用

        seed = index
        if flag:
            torch.manual_seed(seed)  # 保证结果可复现
        
        # 确保输入是 tensor 类型
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=self.dtype)

        # 生成和 img 相同形状的高斯噪声
        if noise_level_img != [0.0]:
            if len(noise_level_img)>1:
                noise_std = random.uniform(noise_level_img[0], noise_level_img[1])
                gussian_noise = torch.normal(mean=0.0, std=noise_std, size=img.shape).to(img.device)
            else:
                gussian_noise = torch.normal(mean=0.0, std=noise_level_img[0], size=img.shape).to(img.device)
        else:
            gussian_noise = 0

        # 将高斯噪声加到原图像上
        img = img + gussian_noise

        # 如果需要，可以选择剪裁到 [0, 1] 范围（或者其他合适的范围）
        img = torch.clamp(img, -1, 1)
        return img
    
    def list_txt_files(self, directory):
        txt_files = []
        # 遍历目录中的所有文件
        for file in os.listdir(directory):
            # 检查文件是否以 .txt 结尾，并加入列表
            if file.endswith('.txt'):
                txt_files.append(os.path.join(directory, file))
        return txt_files

    def random_seismic_sampler(self, time, trace, p, index, flag=False, *arg):
        """随机去除道集的函数

        Args:
            time (_type_): 时间步数（每个样本的行数）
            trace (_type_): 道数（每个样本的列数）
            p (_type_): 最大缺失道数比例(范围从 0 到 1)
            flag (bool, optional):是否使用随机种子. Defaults to False.

        Returns:
            binary_random_matrix: 缺失数据的三维数组 ans(转换为浮点数格式)
            min_ritio: 缺失道的最小比例
            max_ritio: 最大比例
        """
        ans = torch.ones([time, trace], dtype=self.dtype)

        seed = index
        if flag:
            torch.manual_seed(seed)  # 保证结果可复现

        # 计算每个样本中应该缺失的最小和最大列数
        min_missing_traces = int(torch.ceil(torch.tensor(0.2 * trace)).item()) # 设定为总道数的 20%（向上取整）
        max_missing_traces = int(torch.floor(torch.tensor(p * trace)).item())  # 设定为总道数与比例 p 的乘积（向下取整）
        # 从[min_missing_traces, max_missing_traces]范围内随机选择一个值作为缺失列数
        num_missing_traces = torch.randint(min_missing_traces, max_missing_traces + 1, (1,)).item()

        # 随机选择列进行缺失
        missing_indices = torch.randperm(trace)[:num_missing_traces]
        ans[:, missing_indices] = 0

        ratio = (ans == 0).sum().item() / (time * trace)

        binary_random_matrix = ans
        return binary_random_matrix, ratio

    def continus_seismic_sampler(self, time, trace, p, index, flag=False, *arg):
        """下面写随机去除连续道集的函数

        Args:
            time (_type_): _description_
            trace (_type_): _description_
            p (_type_): _description_
            index (_type_): _description_
            flag (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        ans = torch.ones([time, trace], dtype=self.dtype)

        seed = index
        if flag:
            torch.manual_seed(seed)  # 设置随机种子

        unif_random = 0.1 + (p - 0.1) * torch.rand(1)  # 生成一个[0.1, p]之间的随机数
        start = torch.randint(int(0.1 * trace), int((1.0 - p - 0.1) * trace), (1,))  # 随机起始索引
        # start = torch.randint(int(0.1 * trace), int((1.0 - p) * trace), (1,))  # 随机起始索引
        end = (start + trace * unif_random).to(torch.int16)  # 计算结束索引

        star = start.item()  # 提取标量值
        en = end.item()

        ans[:, star:en] = 0
        ratio = (ans == 0).sum().item() / (time * trace)
        # print(ratio)

        binary_random_matrix = ans
        return binary_random_matrix, ratio

    def modified_multiple_seismic_sampler(self, time, trace, p, index, flag=True, p_continuous=0.1, *arg):
        """下面写多种缺失共存的道集的函数
        连续和随机缺失缺失:总缺失率 0.2-0.9(p), 连续缺失的比例 0.1-0.4(p_continuous)

        Args:
            time (_type_): _description_
            trace (_type_): _description_
            p (_type_): _description_
            index (_type_): _description_
            flag (bool, optional): _description_. Defaults to True.
            p_continuous (float, optional): _description_. Defaults to 0.4.

        Returns:
            _type_: _description_
        """
        # -------------------------
        ans = torch.ones([time, trace], dtype=self.dtype)

        seed = index
        if flag:
            torch.manual_seed(seed)

        # 随机生成总缺失率 0.2-0.9（p）
        total_missing_rate = 0.2 + (p - 0.2) * torch.rand(1).item()
        # 随机分配连续缺失的比例 0.1-0.4
        continuous_missing_rate = 0.1 + (p_continuous - 0.1) * torch.rand(1).item()
        # 计算连续缺失的列数
        num_continuous_missing_cols = int(trace * continuous_missing_rate)
        # 随机选择连续缺失的起始列
        start_col = torch.randint(0, trace - num_continuous_missing_cols + 1, (1,)).item()
        # 应用连续缺失
        ans[:, start_col:start_col + num_continuous_missing_cols] = 0

        # 应用随机缺失
        if p_continuous == 0.4:
            for col in range(trace):
                if col < start_col or col >= start_col + num_continuous_missing_cols:
                    if torch.rand(1).item() < (total_missing_rate - continuous_missing_rate):
                        ans[:, col] = 0

            ratio = (ans == 0).sum().item() / (time * trace)
        else:
            # 收集所有符合条件的列号
            eligible_cols = [
                col for col in range(trace) 
                if col < start_col or col >= (start_col + num_continuous_missing_cols)
            ]

            # 计算需要缺失的列数（按比例）
            # missing_rate = (total_missing_rate - continuous_missing_rate)
            missing_rate = (trace*total_missing_rate - trace*continuous_missing_rate)/(trace - trace*p_continuous)
            num_missing_cols = int(round(len(eligible_cols) * missing_rate))

            # 随机选取列并置零
            if eligible_cols and num_missing_cols > 0:
                selected_cols = torch.randperm(len(eligible_cols))[:num_missing_cols].tolist()
                for col_idx in selected_cols:
                    ans[:, eligible_cols[col_idx]] = 0

            ratio = (ans == 0).sum().item() / (time * trace)

        binary_random_matrix = ans
        return binary_random_matrix, ratio

    def binary_seismic_sampler_half(self, time, trace, p, index, flag=False, *arg):
        """下面写随机去除连续道集的函数 只在训练、验证默认为True
        从trace=0开始或者trace=num_trace结束
        这里注意flag=true 没有unif_random = 0.1 + (p - 0.1) * torch.rand(1)

        Args:
            time (_type_): _description_
            trace (_type_): _description_
            p (_type_): _description_
            index (_type_): _description_
            flag (bool, optional): True False None. Defaults to False.

        Returns:
            _type_: _description_
        """
        ans = torch.ones((time, trace), dtype=self.dtype)

        seed = index + 35000
        if flag == True:
            torch.manual_seed(seed)  # 设置随机种子
            unif_random = 0.1 + (p - 0.1) * torch.rand(1)  # 生成0.1-p均匀分布的随机数
        elif flag == False:
            unif_random = 0.1 + (p - 0.1) * torch.rand(1)
        else:
            unif_random = torch.ones(1) * p
        pflag = torch.rand(1)  # 生成pflag

        if pflag < 0.5:
            # 如果 pflag 小于 0.5 计算缺失的结束位置 end，将从开始到 end 的所有道的数据设置为 0
            end = (trace * unif_random).to(torch.int16)
            ans[:, :end.item() + 1] = 0  # 从开始到 end 的数据设置为 0
        else:
            # 如果 pflag 大于或等于 0.5 计算缺失的起始位置 start，将从 start（末尾） 到最后的所有道的数据设置为 0
            start = (trace * unif_random).to(torch.int16)
            ans[:, -start.item():] = 0  # 从 start 到最后的道设置为 0

        ratio = (ans == 0).sum().item() / (time * trace)
        binary_random_matrix = ans
        
        return binary_random_matrix, ratio

    def regular_seismic_sampler(self, time, trace, p, index, flag=False, num1=2, num2=5, *arg):
        """下面写规则去除道集的函数  只在测试用flag=None
        使用随机生成的缺失间隔和长度(flag=True)，或者通过固定参数进行缺失操作(flag=None)

        Args:
            time (_type_): _description_
            trace (_type_): _description_
            p (_type_): _description_
            index (_type_): _description_
            flag (bool, optional): _description_. Defaults to False.
            num1 (int, optional): _description_. Defaults to 2.
            num2 (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        ans = torch.ones([time, trace], dtype=self.dtype)  # 使用 PyTorch 张量初始化

        seed = index

        if flag == True:
            # 设置随机种子
            torch.manual_seed(seed)
            if num1 == 2:
                unif_random = 2  # 直接返回固定值
            else:
                unif_random = torch.randint(2, num1, (1,)).item()  # 使用torch.randint生成随机整数
            torch.manual_seed(seed + 234)
            randomnum = torch.randint(2, num2, (1,)).item()
        elif flag == False:
            if num1 == 2:
                unif_random = 2  
            else:
                unif_random = torch.randint(2, num1, (1,)).item()  
            randomnum = torch.randint(2, num2, (1,)).item()
        else:
            unif_random = int(num1)
            randomnum = int(num2)

        step = unif_random  # 缺失间隔
        length = randomnum  # 缺失长度
        start = step

        # 开始进行缺失道集的规则操作
        while start <= trace:
            end = int(start + length)
            ans[:, start:end] = 0  # 将指定区域设置为 0，表示缺失道
            start = end + step

        # 计算当前样本的缺失比例
        ratio = (ans == 0).sum().item() / (time * trace)  # 使用torch的sum方法

        binary_random_matrix = ans
        # 返回最终的二进制随机矩阵
        return binary_random_matrix, ratio
    


