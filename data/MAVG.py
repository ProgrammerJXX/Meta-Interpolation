import segyio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import math
import os
import random


def get_data(start,end,segy_data):
    """
    从segy中获得指定位置的初至时间与道集数据  1001*120
    1001 个检波点、120 个接收点，以及 1500 个时间步的数 inlinenum crosslinenum timenum
    """
    # print(np.array(segy_data.trace.raw[start: (start + end)]).shape)
    traces_shot = np.reshape(np.array(segy_data.trace.raw[start: (start + end)]),[1001,120,1500])

    return traces_shot.transpose(0,2,1)


def oridataset(segy_data):
    """
    将数据按照炮点取出 拼接后转化为numpy类型

    Args:
        segy_data (SegyFile): 输入的SEGY文件对象,包含地震数据和相关信息。

    Returns:
        np.array: [Receiver, time, trace]
    """
    traces_num = len(segy_data.trace)
    oridata = get_data(0, traces_num, segy_data)
    oridata = np.array(oridata)
    return oridata


def normalization(data):
    """将数据按照trace归一化为0~1  数据[time, trace]

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    n, time, trace = data.shape
    scaler = MinMaxScaler()    # 将数据按列进行缩放，使得每列的数据都落在 [0, 1] 区间
    ans = np.empty_like(data)  # 预分配空间

    for i in range(n):
        norm = scaler.fit_transform(data[i, :, :].reshape(-1, 1)).squeeze() #fit_transform 时，都会计算每列的最小值和最大值，然后将每个值缩放到 [0, 1] 的范围,d但是这里是全局的归一化
        ans[i, :, :] = np.reshape(norm, (time, trace))

    return ans

def get_patchsetrandom(oridata):

    data = normalization(oridata)

    return data

def main(name, path, mavg_path, start_time=100,end_time=988, start_trace=8, end_trace=120, patch_sizetime=128, patch_sizetrace=128):
    """随机生成训练测试和验证集的patch 并储存

    Args:
        name (_type_): 保存的文件名
        path (_type_): 存储文件位置
        mavg_path (_type_): 读取mavg文件的位置
        start_time (int, optional): 截取patch的起始时间. Defaults to 100.
        end_time (int, optional): 截取patch的终止时间,要确保剩余长度足够截取patch. Defaults to 988.
        start_trace (int, optional): _description_. Defaults to 8.
        end_trace (int, optional): _description_. Defaults to 120.
        patch_sizetime (int, optional): patch的时间维度大小. Defaults to 128.
        patch_sizetrace (int, optional): patch的trace维度大小. Defaults to 128.

    Returns:
        _type_: _description_
    """
    ###############判断是否存在训练集###############
    if os.path.exists(path+'/train/mavg_patchsets_train_'+name+'.npy'):
        print("Exist:", path+'/train/mavg_patchsets_train_'+name+'.npy')
        return  # 跳出函数

    seed = 124

    segy_filename = mavg_path + "/Mobil_Avo_Viking_Graben_Line_12.segy"
    print("segy_filename:", segy_filename)
    segy_data = segyio.open(segy_filename,'r',strict = False)
    mmap = segy_data.mmap()# Memory map the file
    segy_data = oridataset(segy_data)

    ################截取patch###############
    for j in range(10000):
        # print(j)
        n = np.random.randint(0, 1001)
        if j==0:
            start_time_random = np.random.randint(start_time,end_time)#544#xin4:745
            ans = segy_data[n:n+1,start_time_random:start_time_random+patch_sizetime,start_trace:end_trace] # 之前转置了，这里的顺序是检波点、时间、道数
        else:
            start_time_random = np.random.randint(start_time,end_time)#544#xin4:745
            start_trace_random = np.random.randint(0,start_trace)
            ans = np.concatenate((ans,segy_data[n:n+1,start_time_random:start_time_random+patch_sizetime,start_trace_random:start_trace_random+patch_sizetrace]),axis=0)
    segy_data = ans
    print('Shape of segy_data: ',segy_data.shape)

    # Normalize
    segy_data_norm = get_patchsetrandom(segy_data)

    ###############分割训练验证测试###############
    # 生成总范围
    total_numbers = set(range(10000))  # 使用集合提高效率
    # 测试集和验证集比例20%
    sample_size = int(10000 * 0.2)
    # Set the random number generator seed for the first choice
    random.seed(seed)
    # Randomly select a number from 0 to sample_size
    first_choice = set(random.sample(total_numbers, sample_size))
    # Set the random number generator seed again for the second choice
    # Choose a number from the remaining ones
    remaining_numbers = total_numbers - first_choice
    random.seed(seed+234)
    second_choice = random.sample(remaining_numbers, sample_size)
    # Form a set with the remaining numbers
    remaining_set = set(remaining_numbers) - set(second_choice)

    val_patch_sets = segy_data_norm[list(first_choice)]
    test_patch_sets = segy_data_norm[list(second_choice)]
    train_patch_sets = segy_data_norm[list(remaining_set)]

    np.random.seed(138)
    np.random.shuffle(train_patch_sets)
    print('Number of generate train patches: ',train_patch_sets.shape)
    np.save(path+'/train/mavg_patchsets_train_'+name+'.npy',train_patch_sets)
    print('Number of generate val patches: ',val_patch_sets.shape)
    np.save(path+'/val/mavg_patchsets_val_'+name+'.npy',val_patch_sets)
    print('Number of generate test patches: ',test_patch_sets.shape)
    np.save(path+'/test/mavg_patchsets_test_'+name+'.npy',test_patch_sets)


def plot_seismic(trace):
    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',
                            squeeze=False)
    axs = axs.ravel()
    axs[0].imshow(trace, cmap=plt.cm.seismic)
    plt.axis('off')  #去掉坐标轴

def check_txt_in_directories(directories):
    """查看文件夹下有没有txt格式文件

    Args:
        directories (_type_): _description_
    """
    for directory in directories:
        has_txt = False
        # 遍历目录中的所有文件
        for file in os.listdir(directory):
            # 检查文件是否以 .txt 结尾
            if file.endswith('.txt'):
                print(f"{directory} 中存在 .txt 文件: {file}")
                has_txt = True
    return has_txt

def save_txt(seicmic_path, save_path):
    patches = np.load(seicmic_path)
    for i, patch in enumerate(patches):
        np.savetxt(save_path+"/MAVG_patch_{}.txt".format(i), patch, fmt='%.2f')  # 保留2位小数，具体格式根据需求调整


if __name__ == '__main__':

    ###############参数设置###############
    name = 'within0_1000_scilcechoose'
    path = 'E:/PycharmProject1/Project1/Hint_seismic/dataset/mavg'
    mavo_path = "E:/PycharmProject1/Project1/Hint_seismic/data/MAVG"
    start_time= 100 #
    end_time = 988  #
    start_trace= 8
    end_trace = 120
    patch_sizetime=256 # 256
    patch_sizetrace=112

    ###############创建文件夹###############
    folders = [path+"/train", path+"/test", path+"/val"]  # 替换成你要检查的路径

    # 遍历每个文件夹路径
    for folder in folders:
        # 判断每个文件夹是否存在
        if not os.path.exists(folder):
            # 如果文件夹不存在，创建该文件夹
            os.makedirs(folder)
            print(f"文件夹 '{folder}' 已创建")
        else:
            print(f"文件夹 '{folder}' 已经存在")

    ###############裁剪的训练集 测试集和验证集生成###############
    if os.path.exists(path+'/train/mavg_patchsets_train_'+name+'.npy'):
        pass
    else:
        main(name, path, mavo_path, start_time, end_time, start_trace, end_trace, patch_sizetime, patch_sizetrace)


    ##############将数据存储为text文件################
    SAVE_PATH = [
        r'E:\PycharmProject1\Project1\Hint_seismic\dataset\mavg\train',
        r'E:\PycharmProject1\Project1\Hint_seismic\dataset\mavg\test',
        r'E:\PycharmProject1\Project1\Hint_seismic\dataset\mavg\val'
    ]
    if check_txt_in_directories(SAVE_PATH) is True:
        pass
    else:
        train_data_path = SAVE_PATH[0] + '\mavg_patchsets_train_within0_1000_scilcechoose.npy'
        save_txt(seicmic_path=train_data_path, save_path=SAVE_PATH[0])

        test_data_path = SAVE_PATH[1] + '\mavg_patchsets_test_within0_1000_scilcechoose.npy'
        save_txt(seicmic_path=test_data_path, save_path=SAVE_PATH[1])

        val_data_path = SAVE_PATH[2] + '\mavg_patchsets_val_within0_1000_scilcechoose.npy'
        save_txt(seicmic_path=val_data_path, save_path=SAVE_PATH[2])