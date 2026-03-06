
import os
import segyio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import random 
from numpy.lib.stride_tricks import as_strided


def plot_seismic(trace):
    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',
                            squeeze=False)
    axs = axs.ravel()
    axs[0].imshow(trace, cmap=plt.cm.seismic)

###############################################获取数据################################################

def get_data(start,end,segy_data):
    """
    从segy中获得指定位置的初至时间与道集数据  201*201=40401
     201 个xline、201 个yline,以及 625 个时间步的数
    """
    traces_shot = np.reshape(np.array(segy_data.trace.raw[start: (start + end)]),[201,201,625])
    # if plot:
    #     plot_seismic(traces_shot[189,:,:].T)
    return traces_shot


def getitem(index,segy_data):
    start = 40401*(index-1)
    end = 40401
    trace_shot = get_data(start,end,segy_data)
    return trace_shot
     
     
def oridataset(segy_data):
    """
    将数据按照炮点取出 拼接后转化为numpy类型

    Args:
        segy_data (SegyFile): 输入的SEGY文件对象,包含地震数据和相关信息。

    Returns:
        np.array: _description_
    """
    oridata = []
    for i in range(1,10):
        shot = getitem(i,segy_data)
        oridata.append(shot)
    oridata = np.array(oridata)
    return oridata
  
###############################################获取数据################################################

def normalization(data):
    """将数据按照trace归一化为0~1  并且将数据由[trace, time]转化为[time, trace]

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    n, time, trace = data.shape
    scaler = MinMaxScaler()    # 将数据按列进行缩放，使得每列的数据都落在 [0, 1] 区间
    ans = np.empty_like(data)  # 预分配空间

    for i in range(n):
        norm = scaler.fit_transform(data[i, :, :].reshape(-1, 1)).squeeze() #fit_transform 时，都会计算每列的最小值和最大值，然后将每个值缩放到 [0, 1] 的范围
        ans[i, :, :] = np.reshape(norm, (time, trace))

    return ans

###############################################裁剪patch方法################################################
# 1.get_patchsetrandom       随机生成
# 2.extract_patches_overlap  按照stride裁剪

def get_patchsetrandom(oridata, patch_wh, patchnum):
    patch_set = []
    num, inlinenum, crosslinenum, timenum = oridata.shape  # shots inlinenum crosslinenum timenum

    for i in range(num):
        for _ in range(patchnum):  # 假设每个样本仍需要迭代1000次
            n = np.random.randint(0, inlinenum)  # 从 [0, inlinenum) 这个区间内随机生成一个整数
            start_trace = np.random.randint(0, crosslinenum - patch_wh)
            start_time = np.random.randint(0, timenum - patch_wh)
            patch_data = oridata[i, n, start_trace:start_trace + patch_wh, start_time:start_time + patch_wh].T
            norm = normalization(patch_data[np.newaxis, :, :])
            patch_set.append(norm)

    return np.concatenate(patch_set, axis=0)


def calculate_padding_or_trimming(size, patch_size, stride):
    """
    Calculate the padding or trimming size based on the original size, patch size, and stride.
    """
    # Total size that can be covered by patches with the given stride
    total_covered = ((size - patch_size) // stride + 1) * stride + patch_size
    if total_covered == size:
        return 0
    elif total_covered > size:
        # Need padding
        return total_covered - size
    else:
        # Need trimming
        return size - total_covered
    

def extract_patches_overlap(data, time_size, trace_size, time_stride, trace_stride, padding=False):
    """
    对比main3 这个函数有固定的采样间隔  同样得到patch
    Extract patches from the seismic data with overlap, based on specified patch size and stride.
    Padding or trimming is applied based on the patch size and stride.
    """
    n_data, n_inlines, n_crosslines, n_times = data.shape

    if padding:
        pad_inline = calculate_padding_or_trimming(n_inlines, trace_size, trace_stride)
        pad_crossline = calculate_padding_or_trimming(n_crosslines, trace_size, trace_stride)
        pad_time = calculate_padding_or_trimming(n_times, time_size, time_stride)
        print("pad_inline:", pad_inline, "pad_crossline:", pad_crossline, "pad_time:", pad_time)

        data_padded = np.pad(data, ((0, 0), (0,pad_inline), (0, pad_crossline), (0, pad_time)), mode='constant')
    else:
        # Trimming the data if padding is not desired
        padded_shape = (0,0,0,0)
        data_padded = data

    _, padded_inlines, padded_crosslines, padded_times = data_padded.shape

    num_patches_inline = (padded_inlines - trace_size) // trace_stride + 1
    num_patches_crossline = (padded_crosslines - trace_size) // trace_stride + 1
    num_patches_time = (padded_times - time_size) // time_stride + 1
    print("num_patches_inline:", num_patches_inline, "num_patches_crossline:", num_patches_crossline, "num_patches_time:", num_patches_time)

    patches = []
    seed =123
    for i in range(num_patches_inline):
        for j in range(num_patches_crossline):
            for k in range(num_patches_time):
                inline_start = i * trace_stride
                crossline_start = j * trace_stride
                time_start = k * time_stride

                patch = data_padded[:, inline_start:inline_start+trace_size, crossline_start:crossline_start+trace_size, time_start:time_start+time_size]
                # print(i,j,k)
                # print(patch.shape)
                # random.seed(seed)
                # seed+=1
                # random_number = random.random()
                # if random_number > 0.5:
                patch = np.transpose(patch.reshape(patch.shape[0]*trace_size, trace_size,time_size), (0, 2, 1))
                norm = normalization(patch)
                patches.append(norm)
                
                # else: 
                #     trans_patch = np.transpose(patch, (0, 2, 1, 3))
                #     patches.append(np.transpose(trans_patch.reshape(patch.shape[0]*trace_size, trace_size, time_size), (0, 2, 1)))
    print(type(patches), np.array(patches).shape)

    return np.concatenate(np.array(patches),axis=0)


def unpatches_efficient(patches, original_shape, trace_size, time_size, time_stride, trace_stride, padding=True):
    """
    用于将patch恢复为原图
    """
    n_data, n_inlines, n_crosslines, n_times = original_shape

    if padding:
        pad_inline = (trace_size - n_inlines % trace_size) % trace_size  # 当n_inlines 已经是 trace_size 的倍数，pad_inline=0
        pad_crossline = (trace_size - n_crosslines % trace_size) % trace_size
        pad_time = (time_size - n_times % time_size) % time_size
        padded_shape = (n_data, n_inlines + pad_inline, n_crosslines + pad_crossline, n_times + pad_time)
    else:
        padded_shape = original_shape

    # reconstructed_data: 用于存储最终重构的数据，初始化为全零。
    # count: 用于记录每个位置被多少个补丁覆盖，初始化为全零。
    reconstructed_data = np.zeros(padded_shape)
    count = np.zeros(padded_shape)

    # 计算滑动窗口操作后，在每个维度上补丁的数量
    num_patches_inline = (padded_shape[1] - trace_size) // trace_stride + 1
    num_patches_crossline = (padded_shape[2] - trace_size) // trace_stride + 1
    num_patches_time = (padded_shape[3] - time_size) // time_stride + 1

    # 生成滑动窗口的索引
    inline_indices = np.arange(0, num_patches_inline * trace_stride, trace_stride)
    crossline_indices = np.arange(0, num_patches_crossline * trace_stride, trace_stride)
    time_indices = np.arange(0, num_patches_time * time_stride, time_stride)

    for i in inline_indices:
        for j in crossline_indices:
            for k in time_indices:
                patch = patches[:, i // trace_stride, j // trace_stride, k // time_stride, :, :, :] # 从 patches 中取出对应位置的补丁
                reconstructed_data[:, i:i+trace_size, j:j+trace_size, k:k+time_size] += patch       # 将补丁加到对应的 reconstructed_data 中相应的位置
                count[:, i:i+trace_size, j:j+trace_size, k:k+time_size] += 1                        # 对每个位置进行计数，每次补丁覆盖的区域计数加 1

    reconstructed_data /= count # 同一个补丁的不同位置可能会被不同补丁访问多次
    
    if padding:
        reconstructed_data = reconstructed_data[:, :n_inlines, :n_crosslines, :n_times] # 如果有填充，则去掉填充部分，只保留原始数据大小的部分

    return reconstructed_data

###############################################裁剪patch方法################################################


###############################################生成npy格式的训练、测试、验证集################################################

# 训练集划分patch  并裁剪出验证和测试集
def main(name, path, segc3_path, start_time=100,end_time=None, patch_sizetime=128, patch_sizetrace=128): 
    """
    按照炮点的维度为分割数据的index
    """
    val_data_all = []           
    test_data_all = []
    train_patch_sets = []
    for h in range(5):   
        segy_filename = segc3_path + "\SEG_45Shot_shots"+str(h*9+1)+"_"+str((h+1)*9)+".sgy"
        print("segy_filename:", segy_filename)
        segy_data = segyio.open(segy_filename,'r',strict = False)
        train_data = oridataset(segy_data)
        # Set the random number generator seed for the first choice
        random.seed(h)  
        # Randomly select a number from 0 to 8
        first_choice = random.randint(0, 8)
        # Set the random number generator seed again for the second choice
        # Choose a number from the remaining ones
        remaining_numbers = [i for i in range(0, 9) if i != first_choice]
        random.seed(h+234)  
        second_choice = random.choice(remaining_numbers)
        # Form a set with the remaining numbers
        remaining_set = set(remaining_numbers) - {second_choice}
        val_data = np.expand_dims(train_data[first_choice], axis=0)
        val_data_all.append(val_data)
        test_data = np.expand_dims(train_data[second_choice], axis=0)
        test_data_all.append(test_data)
        train_data = train_data[list(remaining_set)]
        if end_time is not None:
            train_data=train_data[:,:,:,start_time:end_time] 
        else:
            train_data=train_data[:,:,:,start_time:]
        print('Shape of train_data: ',train_data.shape) 
        patch_sets = get_patchsetrandom(train_data, patch_sizetime, patchnum=1000)
        train_patch_sets.append(patch_sets)
    val_data_all = np.concatenate(val_data_all,axis=0)
    test_data_all = np.concatenate(test_data_all,axis=0)
    train_patch_sets = np.concatenate(train_patch_sets,axis=0)
    
    np.random.seed(138)
    np.random.shuffle(train_patch_sets)
    print('Number of generate train patches: ',train_patch_sets.shape)
    np.save(path+'/train/segc3_patchsets_train_'+name+'.npy',train_patch_sets)
    if name == 'withinall_scilcechoose':
        np.save(path+'/val/segc3_val_withinall_scilcechoose.npy',val_data_all)
        np.save(path+'/test/segc3_test_withinall_scilcechoose.npy',test_data_all)


def main_xin(name, path, segc3_path, start_time=100, end_time=None, patch_sizetime=128, patch_sizetrace=128):
    """
    按照inline为分割数据的index

    Args:
        name (_type_): _description_
        path (_type_): 处理后文件存放位置
        segc3_path (_type_): 原始数据文件位置
        start_time (int, optional): 时间采样起始时间（裁剪用）. Defaults to 100.
        end_time (_type_, optional): 时间采样结束时间（裁剪用）. Defaults to None.
        patch_sizetime (int, optional): _description_. Defaults to 128.
        patch_sizetrace (int, optional): _description_. Defaults to 128.
    """
    val_data_all = []           
    test_data_all = []
    train_data_all = []
    train_patch_sets = []
    seed =0
    for h in range(5):   
        segy_filename = segc3_path + "\SEG_45Shot_shots"+str(h*9+1)+"_"+str((h+1)*9)+".sgy"
        print("segy_filename:", segy_filename)
        segy_data = segyio.open(segy_filename,'r',strict = False)
        train_data = oridataset(segy_data)
        for j in range(len(train_data)):  # 0-8
            # Set the random number generator seed for the first choice
            seed+=1
            random.seed(seed+123)  
            # Randomly select a number from 0 to 8
            first_choice = random.sample(range(train_data.shape[1]), 20)
            remaining_numbers_after_first = list(set(range(train_data.shape[1])) - set(first_choice))
            # Set the random number generator seed again for the second choice
            # Choose a number from the remaining ones
            random.seed(seed+234)  
            second_choice = random.sample(remaining_numbers_after_first, 20)
            remaining_set  = list(set(remaining_numbers_after_first) - set(second_choice))
            # Form a set with the remaining numbers
            val_data = np.expand_dims(train_data[j, first_choice], axis=0)
            val_data_all.append(val_data)
            test_data = np.expand_dims(train_data[j, second_choice], axis=0)
            test_data_all.append(test_data)
            train_data_all_each = np.expand_dims(train_data[j,remaining_set], axis=0)
            train_data_all.append(train_data_all_each)
    val_data_all = np.concatenate(val_data_all,axis=0)
    test_data_all = np.concatenate(test_data_all,axis=0)
    train_data_all = np.concatenate(train_data_all,axis=0)
    if end_time is not None:
        train_data_all=train_data_all[:,:,:,start_time:end_time] 
    else:
        train_data_all=train_data_all[:,:,:,start_time:]

    # 裁剪patch
    print('Shape of train_data: ',train_data_all.shape) 
    train_patch_sets = get_patchsetrandom(train_data_all, patch_sizetime, patchnum=700)

    np.random.seed(138)
    np.random.shuffle(train_patch_sets)
    print('Number of generate train patches: ',train_patch_sets.shape)
    train_patch_sets = train_patch_sets[:30000] # 5*9*700=31500
    print('Number of generate train patches finally: ',train_patch_sets.shape)
    np.save(path+'/train/segc3_patchsets_train_'+name+'.npy',train_patch_sets)
    if name == 'withinall_scilcechoose':
        np.save(path+'/val/segc3_val_'+name+'.npy',val_data_all)
        np.save(path+'/test/segc3_test_'+name+'.npy',test_data_all)

# 测试集合验证集划分patch
def main2(name, path, val_data_all,test_data_all, time_size, trace_size, time_stride, trace_stride):
    """
    用于测试集和验证集按照stride切割patch 并且去除重叠patch
    """
    segc3_patchsets_val = extract_patches_overlap(val_data_all,  time_size, trace_size, time_stride, trace_stride, padding=False) 
    segc3_patchsets_test = extract_patches_overlap(test_data_all,  time_size, trace_size, time_stride, trace_stride, padding=False) 
    print('Number of generate val patches: ',segc3_patchsets_val.shape)
    print('Number of generate test patches: ',segc3_patchsets_test.shape)
    np.save(path+'/val/segc3_patchsets_val_'+name+'.npy',segc3_patchsets_val )
    np.save(path+'/test/segc3_patchsets_test_'+name+'.npy',segc3_patchsets_test)


def main3(name, path, val_data_all,test_data_all, time_size, trace_size, time_stride, trace_stride):
    """
    用于验证集和测试集随机切割patch
    """
    segc3_patchsets_val = get_patchsetrandom(val_data_all, time_size, patchnum=134)
    segc3_patchsets_test = get_patchsetrandom(test_data_all, time_size, patchnum=134)  # 5 *9 *134=6030
    np.random.seed(567)
    np.random.shuffle(segc3_patchsets_val)
    np.random.seed(143)
    np.random.shuffle(segc3_patchsets_test)
    segc3_patchsets_val = segc3_patchsets_val[:6000]
    segc3_patchsets_test = segc3_patchsets_test[:6000]
    print('Number of generate val patches: ',segc3_patchsets_val.shape)
    print('Number of generate test patches: ',segc3_patchsets_test.shape)
    np.save(path+'/val/segc3_patchsets_val_'+name+'.npy',segc3_patchsets_val)
    np.save(path+'/test/segc3_patchsets_test_'+name+'.npy',segc3_patchsets_test)


      
###############################################生成npy格式的训练、测试、验证集################################################

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
        np.savetxt(save_path+"/SEGC3_patch_{}.txt".format(i), patch, fmt='%.2f')  # 保留2位小数，具体格式根据需求调整

if __name__ == '__main__':
    
    ##############参数设置##############
    name = 'within100_300_scilcechoose' #'within0_300_scilcechoose' #'withinall_scilcechoose' #'within100_300_scilcechoose' #
    path = 'E:\PycharmProject1\Project1\Hint_seismic\dataset\segc3'
    segc3_path = "E:\PycharmProject1\Project1\Hint_seismic\data\SEG C3"
    start_time= 100 # 0 100
    end_time = 300  # 300 None
    patch_sizetime=128
    patch_sizetrace=128
    
    ##############创建文件夹##############
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
    
    ##############裁剪的训练集生成  未裁剪的测试集和验证集生成##############
    if os.path.exists(path+'/train/segc3_patchsets_train_'+name+'.npy'):
        pass
    else:
        
        main(name, path, segc3_path, start_time, end_time, patch_sizetime, patch_sizetrace)

    ##############按照stride裁剪test和val的patch##############
    if os.path.exists(path+'/val/segc3_val_withinall_scilcechoose.npy') and os.path.exists(path+'/test/segc3_test_withinall_scilcechoose.npy'):
        pass
    else:
        val_data_all = np.load(path+'/val/segc3_val_withinall_scilcechoose.npy')
        test_data_all = np.load(path+'/test/segc3_test_withinall_scilcechoose.npy')
        start_time= 100 # 0 100
        end_time = 300  # 300 None
        if end_time is not None:
            val_data_all = val_data_all[:,:,:,start_time:end_time] 
            test_data_all = test_data_all[:,:,:,start_time:end_time] 
        else:
            val_data_all = val_data_all[:,:,:,start_time:]
            test_data_all = test_data_all[:,:,:,start_time:]
        print('Shape of val_data_all: ',val_data_all.shape)
        print('Shape of test_data_all: ',test_data_all.shape)
        
        time_stride= 32
        trace_stride= 32
        
        time_size, trace_size =128, 128
        main2(name, path, val_data_all, test_data_all, time_size, trace_size, time_stride, trace_stride)
    
    ##############将数据存储为text文件################
    SAVE_PATH = [
        r'E:\PycharmProject1\Project1\Hint_seismic\dataset\segc3\train',
        r'E:\PycharmProject1\Project1\Hint_seismic\dataset\segc3\test',
        r'E:\PycharmProject1\Project1\Hint_seismic\dataset\segc3\val'
    ]
    if check_txt_in_directories(SAVE_PATH) is True:
        pass
    else:
        train_data_path = SAVE_PATH[0] + '\segc3_patchsets_train_within100_300_scilcechoose.npy'
        save_txt(seicmic_path=train_data_path, save_path=SAVE_PATH[0])
        
        test_data_path = SAVE_PATH[1] + '\segc3_patchsets_test_within100_300_scilcechoose.npy'
        save_txt(seicmic_path=test_data_path, save_path=SAVE_PATH[1])
        
        val_data_path = SAVE_PATH[2] + '\segc3_patchsets_val_within100_300_scilcechoose.npy'
        save_txt(seicmic_path=val_data_path, save_path=SAVE_PATH[2])
