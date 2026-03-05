import pandas as pd
from datetime import datetime


def read_csv2numpy(paths, col, names=None, split=0):
    
    dicts = {}
    for name, path in zip(names, paths):
        df = pd.read_csv(path)
        column_data = df.iloc[:, col].to_numpy() 
        dicts[name] = column_data[split:]
    
    return dicts

import os
import re

def rename_files(path):
    """
    重命名指定路径下符合imputed_data2***.jpeg/txt格式的文件，
    在imputed_data2和数字之间添加下划线
    """
    # 验证路径是否存在
    if not os.path.exists(path):
        print(f"路径 {path} 不存在")
        return

    # 编译正则表达式模式（提高效率）
    pattern = re.compile(r'^imputed_data(\d+)\.(jpeg|txt)$', re.IGNORECASE)
    
    # 遍历目录中的所有文件
    for filename in os.listdir(path):
        # 匹配文件名模式
        match = pattern.match(filename)
        if match:
            # 提取数字部分和扩展名
            digits = match.group(1)
            ext = match.group(2).lower()  # 统一转为小写
            
            # 构建新文件名
            new_name = f"imputed_data_{digits}.{ext}"
            
            # 获取完整文件路径
            old_path = os.path.join(path, filename)
            new_path = os.path.join(path, new_name)
            
            # 执行重命名操作
            try:
                os.rename(old_path, new_path)
                print(f"成功重命名：{filename} -> {new_name}")
            except Exception as e:
                print(f"重命名 {filename} 失败：{str(e)}")



import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from kornia.losses import SSIMLoss, psnr, SSIM
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd

def load_txt_data(file_path):
    """
    从txt文件加载数据
    假设数据格式为trace*time的二维数组
    """
    try:
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_psnr_ssim(data1, data2):
    """
    计算两个数据之间的PSNR和SSIM
    """
    
    ssim = SSIM(window_size=int(3))
    
    if data1 is None or data2 is None:
        return None, None
    
    if data1.shape != data2.shape:
        print(f"Data shapes don't match: {data1.shape} vs {data2.shape}")
        return None, None
    
    try:
        # 计算PSNR (需要确保数据范围合适)
        psnr = peak_signal_noise_ratio(data1, data2, data_range=data1.max() - data1.min())

        # 计算SSIM
        ssim = structural_similarity(data1, data2, win_size=3, data_range=data1.max() - data1.min())
        
        return psnr, ssim
    except Exception as e:
        print(f"Error calculating PSNR/SSIM: {e}")
        return None, None

def analyze_data_comparison(method_paths, ours_method_name, name1_pattern, name2_pattern, M=5, N=5):
    """
    主要分析函数
    
    Parameters:
    method_paths: dict - 方法名到路径的字典，例如 {'Method1': '/path1', 'Ours': '/path2', ...}
    ours_method_name: str - "Ours"方法的名称
    name1_pattern: str - name1文件名模式
    name2_pattern: str - name2文件名模式
    M: int - PSNR差距最大的前M个数据
    N: int - SSIM差距最大的前N个数据
    """
    
    # 存储所有结果
    all_results = defaultdict(lambda: defaultdict(dict))  # {method: {filename: {'psnr': x, 'ssim': y}}}
    
    # 遍历每个方法路径
    for method_name, method_path in method_paths.items():
        print(f"Processing method: {method_name} at {method_path}")
        
        # 查找name1和name2文件
        name1_files = glob.glob(os.path.join(method_path, f"*{name1_pattern}*.txt"))
        name2_files = glob.glob(os.path.join(method_path, f"*{name2_pattern}*.txt"))
        
        # 创建文件名到路径的映射
        name1_dict = {os.path.basename(f).replace(name1_pattern, '').replace('.txt', ''): f for f in name1_files}
        name2_dict = {os.path.basename(f).replace(name2_pattern, '').replace('.txt', ''): f for f in name2_files}
        
        # 找到共同的文件名基础
        common_names = set(name1_dict.keys()) & set(name2_dict.keys())
        
        for base_name in common_names:
            # 加载数据
            data1 = load_txt_data(name1_dict[base_name])
            data2 = load_txt_data(name2_dict[base_name])
            
            # 计算PSNR和SSIM
            psnr, ssim = calculate_psnr_ssim(data1, data2)
            
            if psnr is not None and ssim is not None:
                all_results[method_name][base_name] = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'name1_file': name1_dict[base_name],
                    'name2_file': name2_dict[base_name]
                }
            else:
                print("method {} psnr&ssim is None!".format(method_name))
    
    # 获取所有共同的文件名
    all_filenames = set()
    for method_results in all_results.values():
        all_filenames.update(method_results.keys())
    
    # 筛选出所有方法都有的文件名
    common_filenames = set(all_filenames)
    for method_name in method_paths.keys():
        if method_name in all_results:
            common_filenames &= set(all_results[method_name].keys())
    
    print(f"Found {len(common_filenames)} common files across all methods")
    
    # 计算Ours方法PSNR排名第一的文件
    ours_psnr_first = []
    psnr_gaps = {}  # {filename: max_gap}
    
    for filename in common_filenames:
        # 获取所有方法对该文件的PSNR
        psnr_values = {}
        for method_name in method_paths.keys():
            if method_name in all_results and filename in all_results[method_name]:
                # print(method_name, filename)
                psnr_values[method_name] = all_results[method_name][filename]['psnr']
        
        if len(psnr_values) > 1 and ours_method_name in psnr_values:
            # 检查Ours是否PSNR最高
            ours_psnr = psnr_values[ours_method_name]
            max_psnr = max(psnr_values.values())
            
            if ours_psnr == max_psnr:
                ours_psnr_first.append(filename)
                
                # 计算与其他方法的最大差距
                other_psnrs = [v for k, v in psnr_values.items() if k != ours_method_name]
                if other_psnrs:
                    max_gap = ours_psnr - max(other_psnrs)
                    psnr_gaps[filename] = max_gap
    # print(" psnr_values:",  psnr_values)
    # print("ours_psnr_first:", ours_psnr_first)
    # print("psnr_gaps:", psnr_gaps)
    
    # 计算Ours方法SSIM排名第一的文件
    ours_ssim_first = []
    ssim_gaps = {}  # {filename: max_gap}
    
    for filename in common_filenames:
        # 获取所有方法对该文件的SSIM
        ssim_values = {}
        for method_name in method_paths.keys():
            if method_name in all_results and filename in all_results[method_name]:
                ssim_values[method_name] = all_results[method_name][filename]['ssim']
        
        if len(ssim_values) > 1 and ours_method_name in ssim_values:
            # 检查Ours是否SSIM最高
            ours_ssim = ssim_values[ours_method_name]
            max_ssim = max(ssim_values.values())
            
            if ours_ssim == max_ssim:
                ours_ssim_first.append(filename)
                
                # 计算与其他方法的最大差距
                other_ssims = [v for k, v in ssim_values.items() if k != ours_method_name]
                if other_ssims:
                    max_gap = ours_ssim - max(other_ssims)
                    ssim_gaps[filename] = max_gap
    
    # 找出PSNR差距最大的前M个
    top_psnr_gaps = sorted(psnr_gaps.items(), key=lambda x: x[1], reverse=True)[:M]
    
    # 找出SSIM差距最大的前N个
    top_ssim_gaps = sorted(ssim_gaps.items(), key=lambda x: x[1], reverse=True)[:N]
    
    # 生成结果报告
    results = {
        'all_results': dict(all_results),
        'ours_psnr_first_count': len(ours_psnr_first),
        'ours_ssim_first_count': len(ours_ssim_first),
        'ours_psnr_first_files': ours_psnr_first,
        'ours_ssim_first_files': ours_ssim_first,
        'top_psnr_gaps': top_psnr_gaps,
        'top_ssim_gaps': top_ssim_gaps
    }
    
    return results

def print_results(results, ours_method_name):
    """
    打印分析结果
    """
    print("\n" + "="*80)
    print("分析结果报告")
    print("="*80)
    
    print(f"\n{ours_method_name}方法PSNR排名第一的文件数量: {results['ours_psnr_first_count']}")
    print(f"{ours_method_name}方法SSIM排名第一的文件数量: {results['ours_ssim_first_count']}")
    
    print(f"\n{ours_method_name}方法PSNR差距最大的文件:")
    print("-" * 50)
    for i, (filename, gap) in enumerate(results['top_psnr_gaps'], 1):
        print(f"{i}. {filename}: PSNR差距 = {gap:.4f}")
    
    print(f"\n{ours_method_name}方法SSIM差距最大的文件:")
    print("-" * 50)
    for i, (filename, gap) in enumerate(results['top_ssim_gaps'], 1):
        print(f"{i}. {filename}: SSIM差距 = {gap:.4f}")

def save_results_to_csv(results, output_path="analysis_results.csv"):
    """
    将结果保存为CSV文件
    """
    # 准备数据用于CSV输出
    csv_data = []
    
    for method_name, method_results in results['all_results'].items():
        for filename, metrics in method_results.items():
            csv_data.append({
                'Method': method_name,
                'Filename': filename,
                'PSNR': metrics['psnr'],
                'SSIM': metrics['ssim'],
                'Name1_File': metrics['name1_file'],
                'Name2_File': metrics['name2_file']
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

if __name__ == '__main__':
    
    # names = ['λ1=1', 'λ1=0.1', 'λ1=0.5', 'λ1=2',]
    # paths = [r'E:/PycharmProject1/Project1/Distill_model/results/loss_abl/train_cd_continus_2025-06-28-14-11-22-2025-8-15_21_40_48.csv',
    #          r'E:/PycharmProject1/Project1/Distill_model/results/loss_abl/train_cd_continus_2025-07-05-23-30-52-2025-8-15_21_41_19.csv',
    #          r'E:/PycharmProject1/Project1/Distill_model/results/loss_abl/train_cd_continus_2025-07-06-12-11-04-2025-8-15_21_45_57.csv',
    #          r'E:/PycharmProject1/Project1/Distill_model/results/loss_abl/train_cd_continus_2025-07-06-15-54-16-2025-8-15_21_43_26.csv',
    #             ]
    
    # read_csv2numpy(paths, col=1, names=names)
    
    """修改路径下文件名"""
    # target_path = 'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_DRU_Net_4_64_L1Loss_v1/2025-08-08-16-29-31/test-2025-08-09-19-48-28\image'
    # rename_files(target_path)

    """查找PSNR和SSIM最高的几张图"""
    # 设置参数
    ours_method_name = 'Ours' # 'Oursplus'
    name1_pattern = 'test_outputs_'  # 根据实际情况修改
    name2_pattern = 'ori_data_'  # 根据实际情况修改
    M = 10  # PSNR差距最大的前M个
    N = 10  # SSIM差距最大的前N个
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # SEGC3 continus 0.3
    output_path_1 = "E:\PycharmProject1\Project1\Distill_model/results/analysis_results_SEGC3_continus_0.3_{}.csv".format(time_str)
    method_paths_1 = {
            'Ours': 'G:\Distill_model/results/train_cd_v4.1_T_2_L_1_v2_lc_beta_1_SEGC3_continus_0.1_0.3_AN_net_Unet_4_64_bn_avg_L1Loss_v2/2025-06-06-22-53-19/test-2025-09-02-11-37-44\image',
            'Unet':'G:\Distill_model/results/train_compare_models_SEGC3_continus_0.1_0.3_Unet_5_64_bn_max_MSELoss/2025-04-30-14-52-52/test-2025-08-07-22-35-07\image',
            'CAE': 'G:\Distill_model/results/train_compare_models_SEGC3/2025-04-11-14-18-56/test-2025-08-08-18-17-05/image',
            'Resnet': 'G:\Distill_model/results/train_compare_models_SEGC3/2025-04-12-11-07-08/test-2025-08-09-11-50-50\image',
            'CTR':'G:\coarse_to_refine/results/train_wonderfulxin_SEGC3/2025-04-10-21-48-40/test-2025-08-09-21-39-43/image',
            'DRU_Net':'G:\Distill_model/results/train_compare_models_SEGC3_continus_0.1_0.3_DRU_Net_4_64_L1Loss_v1/2025-08-30-18-44-40/test-2025-08-31-10-33-13/image',
            'SEU_Net':'G:\Distill_model/results/train_compare_models_SEGC3_continus_0.1_0.3_SEU_Net_4_64_L1Loss_v1/2025-08-15-15-14-12/test-2025-08-16-17-38-03/image',
            'MSU_Net':'G:\Distill_model/results/train_compare_models_SEGC3_continus_0.1_0.3_MS_Unet_4_64_none_max_L1Loss_v2/2025-08-04-11-47-53/test-2025-09-02-18-25-58/image'
         }
    
    # SEGC3 random 0.3
    output_path_2 = "E:\PycharmProject1\Project1\Distill_model/results/analysis_results_SEGC3_random_0.3_{}.csv".format(time_str)
    method_paths_2 = {
              'Ours': 'G:\Distill_model/results/train_cd_sd_cb_attn_attnd_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_SEGC3_random_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_spa/2025-08-28-10-42-36/test-2025-09-06-21-46-21\image',
              'Unet':'G:\Distill_model/results/train_compare_models_SEGC3_random_0.1_0.3_Unet_5_64_bn_max_MSELoss/2025-04-30-14-54-05/test-2025-08-08-11-23-40\image',
              'CAE': 'G:\Distill_model/results/train_compare_models_SEGC3_random_0.1_0.3_CAE_3_64_MSELoss/2025-04-23-15-41-01/test-2025-08-08-21-06-56/image',
              'Resnet': 'G:\Distill_model/results/train_compare_models_SEGC3_random_0.1_0.3_ResNet_3_64_MSELoss/2025-04-23-15-25-31/test-2025-08-09-15-13-40\image',
              'CTR':'G:\coarse_to_refine/results/train_wonderfulxin_SEGC3/2025-04-13-16-50-12/test-2025-08-10-00-03-01/image',
              'DRU_Net':'G:\Distill_model/results/train_compare_models_SEGC3_random_0.1_0.3_DRU_Net_4_64_L1Loss_v1/2025-08-09-15-13-45/test-2025-08-12-20-12-11/image',
              'SEU_Net':'G:\Distill_model/results/train_compare_models_SEGC3_random_0.1_0.3_SEU_Net_4_64_L1Loss_v1/2025-08-09-21-16-55/test-2025-09-03-22-08-17/image',
              'MSU_Net':'G:\Distill_model/results/train_compare_models_SEGC3_random_0.1_0.3_MS_Unet_4_64_none_max_MSELoss/2025-08-07-21-40-41/test-2025-09-04-11-50-44/image'
          }
    
    # MAVO continus 0.3
    output_path_3 = "E:\PycharmProject1\Project1\Distill_model/results/analysis_results_MAVO_continus_0.3_{}.csv".format(time_str)
    method_paths_3 = {
              'Ours': 'G:\Distill_model/results/train_cd_sd_cb_attn_attnd_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_continus_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_spa/2025-09-04-23-39-17/test-2025-09-09-15-41-27/image',
              'Oursplus': 'G:\Distill_model/results/train_cd_sd_cb_attn_attnd_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_continus_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_spa/2025-09-05-13-23-42/test-2025-09-11-20-50-12/image',
              'Unet':'G:\Distill_model/results/train_compare_models_MAVG_continus_0.1_0.3_Unet_5_64_bn_max_MSELoss/2025-06-16-19-58-00/test-2025-08-08-14-45-44\image',
              'CAE': 'G:\Distill_model/results/train_compare_models_MAVG_continus_0.1_0.3_CAE_3_64_MSELoss/2025-06-15-21-42-46/test-2025-08-09-10-47-56/image',
              'Resnet': 'G:\Distill_model/results/train_compare_models_MAVG_continus_0.1_0.3_ResNet_3_64_MSELoss/2025-06-15-22-03-19/test-2025-08-09-19-33-53\image',
              'CTR':'G:\coarse_to_refine/results/train_compare_models_MAVG_continus_0.1_0.3_CTR_bn_avg/2025-09-06-23-45-17/test-2025-09-08-15-53-49/image',
              'DRU_Net':'G:\Distill_model/results/train_compare_models_MAVG_continus_0.1_0.3_DRU_Net_4_64_bn_max_L1Loss_v1/2025-08-31-10-41-05/test-2025-09-01-14-43-26/image',
              'SEU_Net':'G:\Distill_model/results/train_compare_models_MAVG_continus_0.1_0.3_SEU_Net_4_64_L1Loss_v1/2025-08-15-15-13-40/test-2025-08-16-16-31-48/image',
              'MSU_Net':'G:\Distill_model/results/train_compare_models_MAVG_continus_0.1_0.3_MS_Unet_4_64_none_max_MSELoss/2025-08-04-11-54-25/test-2025-09-03-09-17-15/image'
          }

    # MAVO random 0.3
    output_path_4 = "E:\PycharmProject1\Project1\Distill_model/results/analysis_results_MAVO_random_0.3_{}.csv".format(time_str)
    method_paths_4 = {
              'Ours': 'G:\Distill_model/results/train_cd_sd_cb_attn_attnd_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_random_0.1_0.3_AN_net_Unet_4_64_bn_avg_dcn_v2_attn_spa/2025-09-05-17-21-01/test-2025-09-08-14-49-51/image',
              'Unet':'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_Unet_5_64_bn_max_MSELoss/2025-08-09-22-04-23/test-2025-08-11-11-10-25\image',
              'CAE': 'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_CAE_3_64_MSELoss/2025-06-15-21-52-53/test-2025-08-09-11-20-41/image',
              'Resnet': 'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_ResNet_3_64_MSELoss/2025-06-16-08-53-41/test-2025-08-09-20-41-21\image',
              'CTR':'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_CTR_bn_avg/2025-09-07-11-34-54/test-2025-09-08-11-57-09/image',
              'DRU_Net':'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_DRU_Net_4_64_bn_max_L1Loss_v1/2025-08-31-10-44-54/test-2025-09-01-15-08-42/image',
              'SEU_Net':'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_SEU_Net_4_64_L1Loss_v1/2025-08-09-10-58-54/test-2025-09-06-20-45-48/image',
              'MSU_Net':'G:\Distill_model/results/train_compare_models_MAVG_random_0.1_0.3_MS_Unet_4_64_none_max_MSELoss/2025-08-04-11-52-00/test-2025-09-02-21-19-47/image'
          }

    # MAVO continus 0.3 gussian hnoise 0.15-0.3
    output_path_5 = "E:\PycharmProject1\Project1\Distill_model/results/analysis_results_MAVO_continus_0.3_hnoise_0.15_0.3_{}.csv".format(time_str)
    method_paths_5 = {
              'Ours': 'G:\Distill_model/results/train_cd_sd_cb_attn_attnd_v4.1_M_30_T_1_L_10_lc_v2_ls_v1_la_v1_beta_1_MAVG_adhnoise_continus_0.1_0.3_AN_net_Unet_4_64/2025-11-14-12-06-03/test-2025-11-18-18-42-02/image',
              'Unet':'G:\Distill_model/results/train_compare_models_MAVG_adhnoise_continus_0.1_0.3_Unet_5_64_bn_max_MSELoss/2025-11-12-19-19-59/test-2025-11-18-17-35-22/image',
              'CTR':'G:\Distill_model/results/train_compare_models_MAVG_adhnoise_continus_0.1_0.3_CTR_bn_avg/2025-11-17-22-56-41/test-2025-11-18-16-15-20/image',
              'DRU_Net':'G:\Distill_model/results/train_compare_models_MAVG_adhnoise_continus_0.1_0.3_DRU_Net_4_64_bn_max_L1Loss_v1/2025-11-13-10-11-10/test-2025-11-18-16-43-45/image',
          }

    # 运行分析
    try:
        results = analyze_data_comparison(
            method_paths=method_paths_5,
            ours_method_name=ours_method_name,
            name1_pattern=name1_pattern,
            name2_pattern=name2_pattern,
            M=M,
            N=N
        )
    
        # 打印结果
        print_results(results, ours_method_name)
    
        # 保存结果
        save_results_to_csv(results, output_path=output_path_5)
    
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()