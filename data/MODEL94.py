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
from tqdm import tqdm


def get_data(data_path):
    # 打开SEGY文件
    with segyio.open(data_path, ignore_geometry=True) as segyfile:
        # 读取所有Trace的Field Record Number（Shot编号）
        field_records = segyfile.attributes(segyio.TraceField.FieldRecord)[:]
        # 读取所有Trace的数据（形状为 [总Trace数, 采样点数]）
        traces = segyfile.trace.raw[:]

    # 确定每个Shot的起始和结束位置
    shot_positions = {}
    current_shot = None
    start_idx = 0
    for i, fr in enumerate(field_records):
        # if i == 0:
        #     print("1_", fr)
        if fr != current_shot:
            if current_shot is not None:
                # print(fr, start_idx, i-1)
                shot_positions[current_shot] = (start_idx, i)
            current_shot = fr
            start_idx = i
    # 处理最后一个Shot
    shot_positions[current_shot] = (start_idx, len(field_records))

    # 筛选出包含480个Trace的Shot
    valid_shots = []
    for shot, (start, end) in shot_positions.items():
        if end - start == 480:
            valid_shots.append(shot)

    # 按Shot编号排序
    valid_shots_sorted = sorted(valid_shots)

    # 寻找连续198个的Shot序列
    selected_shot_start = None
    current_streak = 1
    for i in range(1, len(valid_shots_sorted)):
        if valid_shots_sorted[i] == valid_shots_sorted[i-1] + 1:
            current_streak += 1
            if current_streak == 198:
                selected_shot_start = i - 198 + 1
                break
        else:
            current_streak = 1

    if selected_shot_start is None:
        raise ValueError("未找到连续的198个完整Shot")

    selected_shots = valid_shots_sorted[selected_shot_start : selected_shot_start + 198]

    # 提取选定Shot的数据并组合为3D数组
    data_list = []
    for shot in selected_shots:
        start, end = shot_positions[shot]
        shot_data = traces[start:end]  # 形状为 [480, 采样点数]
        data_list.append(shot_data)

    # 合并为3D数组（198个Shot × 480个Trace × 采样点数）
    data_3d = np.stack(data_list, axis=0)

    print("提取的数据形状:", data_3d.shape)  # 应为 (198, 480, 采样点数)
    return data_3d.transpose(0,2,1)

def oridataset(segy_filename):
    # (198, 采样点数, 480)
    oridata = get_data(segy_filename)
    return oridata


def normalization(data):
    """将数据按照trace归一化为0~1  数据[time, trace]
    """
    n, time, trace = data.shape
    scaler = MinMaxScaler()    # 将数据按列进行缩放，使得每列的数据都落在 [0, 1] 区间
    ans = np.empty_like(data)  # 预分配空间

    for i in range(n):
        norm = scaler.fit_transform(data[i, :, :].reshape(-1, 1)).squeeze() #fit_transform 时，都会计算每列的最小值和最大值，然后将每个值缩放到 [0, 1] 的范围
        ans[i, :, :] = np.reshape(norm, (time, trace))

    return ans

def get_patchsetrandom(oridata):

    data = normalization(oridata)

    return data

def main(name, path, model94_path, start_time=8,end_time=850, start_trace=8, end_trace=350, patch_sizetime=128, patch_sizetrace=128):
    """随机生成训练测试和验证集的patch 并储存"""

    ###############判断是否存在训练集###############
    if os.path.exists(path+'/train/model94_patchsets_train_'+name+'.npy'):
        print("Exist:", path+'/train/model94_patchsets_train_'+name+'.npy')
        return  # 跳出函数

    seed = 124

    segy_filename = model94_path + "/Model94_shots.segy"
    print("segy_filename:", segy_filename)
    segy_data = oridataset(segy_filename)[:,200:1200,:]  # 裁剪全为零部分
    print("裁剪全为零部分后数据形状:", segy_data.shape)  # 应为 (198, 1000, 480)

    ################截取patch###############
    pbar = tqdm(total=10000, ncols=100)
    for j in range(10000):
        # print(j)
        n = np.random.randint(0, 198) # 0~197 因为出现198会导致198:199切片
        start_time_random = np.random.randint(start_time,end_time)
        start_trace_random = np.random.randint(start_trace, end_trace)

        patch = segy_data[n:n+1, start_time_random:start_time_random+patch_sizetime, start_trace_random:start_trace_random+patch_sizetrace]
        if patch.shape != (1, 128, 128):
            print(patch.shape)
            print(f"[Error] 无效切片位置: j={j}, time={start_time_random}, trace={start_trace_random}")
            continue  # 跳过无效切片

        if j == 0:
            ans = patch
        else:
            ans = np.concatenate((ans, patch), axis=0)
        pbar.set_description(' '.join(['Generate segy_data Step: [{0}/{1}]  '.format(j, 10000)]))
        pbar.update(1)
    pbar.close()
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
    np.save(path+'/train/model94_patchsets_train_'+name+'.npy',train_patch_sets)
    print('Number of generate val patches: ',val_patch_sets.shape)
    np.save(path+'/val/model94_patchsets_val_'+name+'.npy',val_patch_sets)
    print('Number of generate test patches: ',test_patch_sets.shape)
    np.save(path+'/test/model94_patchsets_test_'+name+'.npy',test_patch_sets)

def main_all(name, path, model94_path):
    """生成完整数据集并储存"""

    seed = 124

    segy_filename = model94_path + "/Model94_shots.segy"
    print("segy_filename:", segy_filename)
    segy_data = oridataset(segy_filename)[:,200:1200,:]  # 裁剪全为零部分
    print("裁剪全为零部分后数据形状:", segy_data.shape)  # 应为 (198, 1000, 480)

    print('Shape of segy_data: ',segy_data.shape)
    np.save(path+'/model94_all_'+name+'.npy', segy_data)

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
        np.savetxt(save_path+"/MODEL94_patch_{}.txt".format(i), patch, fmt='%.2f')  # 保留2位小数，具体格式根据需求调整

if __name__ == '__main__':

    ###############参数设置###############
    name = 'within0_1000_scilcechoose'
    path = 'H:/seismic data/Hint_seismic/dataset/model94'
    model94_path = "H:/seismic data/Hint_seismic/data/MODEL94"
    start_time= 8
    end_time = 850  # 850+128<1000
    start_trace= 8
    end_trace = 350 # 350+128<480
    patch_sizetime=128
    patch_sizetrace=128
    mode = 'all' #patch all

    if mode == 'patch':
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
        if os.path.exists(path+'/train/model94_patchsets_train_'+name+'.npy'):
            pass
        else:
            main(name, path, model94_path, start_time, end_time, start_trace, end_trace, patch_sizetime, patch_sizetrace)


        ##############将数据存储为text文件################
        SAVE_PATH = [
            r'G:/seismic data/Hint_seismic/dataset/model94/train',
            r'G:/seismic data/Hint_seismic/dataset/model94/test',
            r'G:/seismic data/Hint_seismic/dataset/model94/val'
        ]
        if check_txt_in_directories(SAVE_PATH) is True:
            pass
        else:
            train_data_path = SAVE_PATH[0] + '\model94_patchsets_train_within0_1000_scilcechoose.npy'
            save_txt(seicmic_path=train_data_path, save_path=SAVE_PATH[0])

            test_data_path = SAVE_PATH[1] + '\model94_patchsets_test_within0_1000_scilcechoose.npy'
            save_txt(seicmic_path=test_data_path, save_path=SAVE_PATH[1])

            val_data_path = SAVE_PATH[2] + '\model94_patchsets_val_within0_1000_scilcechoose.npy'
            save_txt(seicmic_path=val_data_path, save_path=SAVE_PATH[2])
    elif mode == 'all':
        name = 'not_norm'
        path = 'H:/seismic data/Hint_seismic/dataset/model94'
        main_all(name, path, model94_path)
