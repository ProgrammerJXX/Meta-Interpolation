import os
import argparse
import json
import wandb
import swanlab
import time
from datetime import datetime
from operator import itemgetter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Any, List, Tuple
from kornia.losses import SSIMLoss, psnr, SSIM
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import socket
# 获取当前主机的主机名
hostname = socket.gethostname()
swanlab.login(api_key='I4Ge4KHvI92q1Bg5DkrRl')

from models.base_models import Refine, Discriminator
from models.vgg import Vgg16
from dataset.dataset_SEGC3 import Dataset_SEGC3
from dataset.dataset_MAVG import Dataset_MAVG
from utils.meta_optimizers import MetaSGD
from utils.utils import WeightNetwork, LossWeightNetwork, FeatureMatching, inner_objective, outer_objective, seed_rng
from utils.utils import AverageMeter, makedirs, pair_to_list, get_project_path, load_model, load_trained_model, count_parameters, format_parameters_as_million, format_times_as_hms, torch_dtype
from utils.utils import signal_to_noise as snr
from utils.loss import TVLoss, DiscriminatorLoss, PerceptLoss, StyleLoss, LossCalculator, loss_keys
from utils.plot import train_loss_figure, val_loss_figure, train_SNR_figre, val_SNR_figre, \
    train_PSNR_figre, val_PSNR_figre, train_SSIM_figre, val_SSIM_figre, SNR_figre, PSNR_figre, SSIM_figre
from utils.plot import plot_seismic, plot_difference, plot_mask

#  创建一个字典，映射损失名称到损失函数的构造
loss_dict = {
    'L1Loss': lambda: nn.L1Loss(reduction='none').to(device),  # 默认 reduction='mean'
    'SSIMLoss': lambda: SSIMLoss(window_size=int(3)).to(device),
    'MSELoss': lambda: nn.MSELoss().to(device),
    'TVloss': lambda: TVLoss().to(device),
}


def get_args():

    parser = argparse.ArgumentParser()
    # Base setting
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--Epoch', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_source_load_path', type=str, default='.pt')
    parser.add_argument('--train_source_model', action='store_true', default=False)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--online_vis', type=str, default='swanlab', choices=['wandb', 'swanlab', 'not_use'])
    parser.add_argument('--resume', action='store_true', default=False)
    # Dataset dataloader
    parser.add_argument('--dataset', type=str, default='SEGC3', choices=['SEGC3', 'MAVG'])
    parser.add_argument('--data_use', type=float, default=-1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--missing_p', type=float, default=0.5)
    parser.add_argument('--missing_p_continuous', type=float, default=0.1)
    parser.add_argument('--flag', choices=['bool', 'str'], default=True)
    parser.add_argument('--noise_level_img', type=float, nargs='+', default=[0.05, 0.3])
    parser.add_argument('--meta_batch', type=int, default=16)
    parser.add_argument('--batch_train', type=int, default=16)
    parser.add_argument('--batch_val', type=int, default=1)
    parser.add_argument('--batch_test', type=int, default=1)
    parser.add_argument('--resolution_h', type=int, default=128)
    parser.add_argument('--resolution_w', type=int, default=128)
    parser.add_argument('--sampler', type=str, default='random', choices=['random', 'continus', 'multiple', 'half', 'regular'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_type', type=str, default='float32', choices=['float16', 'float32', 'float64'])
    # Model
    parser.add_argument('--input_dim', type=int, default=1, choices=[1, 3])
    parser.add_argument('--depth', type=int, default=5, choices=[3, 4, 5])
    parser.add_argument('--cnum', type=int, default=64, choices=[16, 32, 64, 128])
    parser.add_argument('--norm', type=str, default='none', choices=['bn', 'in', 'none'])
    parser.add_argument('--pool', type=str, default='max', choices=['max', 'avg', 'none'])
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'elu', 'lrelu', 'prelu', 'selu', 'tanh', 'sigmoid', 'none'])
    parser.add_argument('--nb_filter', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    # AN_net_DenseNet/AN_net_DenseNet_Skip
    parser.add_argument('--growth_rate', type=int, default=32)
    parser.add_argument('--block_config', type=Tuple[int, int, int, int], default=(6, 12, 24))
    parser.add_argument('--num_init_features', type=int, default=64)
    parser.add_argument('--bn_size', type=int, default=4)
    parser.add_argument('--drop_rate', type=float, default=0)
    # Optimizer & scheduler
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--betas', type=float, nargs='+',  default=(0.9, 0.999))
    parser.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'CosineAnnealingLR'])
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # Distill
    parser.add_argument('--distill_type', type=str, default='v3', choices=['v1', 'v2', 'v2.1', 'v3', 'v4', 'v4.1'], help='v1 for L2T-ww;v2 for inner matching_only=False;v3 for Distillation-guided Image Inpainting; v4 for L2T-ww do not rollback')
    parser.add_argument('--max_meta_step', type=int, default=2)
    parser.add_argument('--need_cd', action='store_true', default=False)
    parser.add_argument('--need_sd', action='store_true', default=False)
    parser.add_argument('--need_attnd', action='store_true', default=False)
    parser.add_argument('--source_model', type=str, default='AN_net_CTR', choices=['coarse_to_refine', 'Face_Parsing_Network', 'AN_net_DGII', 'AN_net_CTR', 'AN_net_Unet', 'AN_net_ResNet', 'AN_net_DenseNet', 'AN_net_DenseNet_Skip','AN_net_Unetplus'])
    parser.add_argument('--target_model', type=str, default='IN_net_CTR', choices=['Face_Parsing_Network', 'IN_net_DGII', 'IN_net_CTR', 'IN_net_Unet', 'IN_net_ResNet', 'IN_net_DenseNet', 'IN_net_DenseNet_Skip','IN_net_Unetplus'])
    parser.add_argument('--source_model_sd', type=str, default='Face_Parsing_Network_deep', choices=['Face_Parsing_Network', 'Face_Parsing_Network_deep', 'AN_net_Unet', 'AN_net_Unet_deep', 'IN_net_Unet_deep','IN_net_Unetplus_deep'])
    parser.add_argument('--target_model_sd', type=str, default='Face_Parsing_Network_shallow', choices=['Face_Parsing_Network', 'Face_Parsing_Network_shallow', 'AN_net_Unet', 'AN_net_Unet_shallow', 'IN_net_Unet_shallow','IN_net_Unetplus_shallow'])
    parser.add_argument('--source_model_attnd', type=str, default='AN_net_Unet_attn', choices=['AN_net_Unet_attn', 'AN_net_Unetplus_attn'])
    parser.add_argument('--target_model_attnd', type=str, default='IN_net_Unet_attn', choices=['IN_net_Unet_attn', 'IN_net_Unetplus_attn'])
    parser.add_argument('--pairs', type=str, default='0-0,1-1,2-2,3-3,4-4')
    parser.add_argument('--pairs_sd', type=str, default='0-0,1-1')
    parser.add_argument('--pairs_attnd', type=str, default='0-0')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--roll_back', action='store_true', default=False, help='config.distill_type in {v1, v2, v2.1}')
    parser.add_argument('--feature_matching_type', type=str, default='v1', choices=['v1', 'v2', 'v3'], help='v1 for L2T-ww;v2 for Distillation-guided Image Inpainting;v3 for v2 but no conv')
    parser.add_argument('--feature_matching_type_sd', type=str, default='v1', choices=['v1', 'v2', 'v3'], help='v1 for L2T-ww;v2 for Distillation-guided Image Inpainting;v3 for v2 but no conv')
    parser.add_argument('--feature_matching_type_attnd', type=str, default='v1', choices=['v1', 'v2', 'v3', 'not_use'], help='v1 for L2T-ww;v2 for Distillation-guided Image Inpainting;v3 for v2 but no conv')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='Initial learning rate for meta networks')
    parser.add_argument('--meta_wd', type=float, default=1e-4)
    parser.add_argument('--loss_weight_cd', action='store_true', default=False)
    parser.add_argument('--loss_weight_sd', action='store_true', default=False)
    parser.add_argument('--loss_weight_attnd', action='store_true', default=False)
    parser.add_argument('--loss_weight_type', type=str, default='relu6', choices=['relu', 'relu_avg', 'relu6'])
    parser.add_argument('--loss_weight_init', type=float, default=1.0)
    parser.add_argument('--loss_weight_init_sd', type=float, default=1.0)
    parser.add_argument('--loss_weight_init_attnd', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0, help='Used to adjust the scale of  weights:0, 0.5, 30')
    parser.add_argument('--beta_mt', type=float, default=0, help='Used to adjust the scale of  weights:0, 0.5, 30')
    parser.add_argument('--use_mask', action='store_true', default=False)
    # Loss
    parser.add_argument('--loss_list', type=str, nargs='+', default=['L1Loss', 'SSIMLoss', 'MSELoss', 'TVLoss', 'DisLoss', 'PerceptLoss', 'StyleLoss'], help='L1Loss, SSIMLoss, TVLoss, MSELoss, DisLoss, PerceptLoss, StyleLoss')
    parser.add_argument('--loss_weight', type=float, nargs='+', default=[1, 0, 0, 0, 1], help='L1Loss, SSIMLoss, TVLoss, MSELoss, DisLoss')
    parser.add_argument('--alpha', type=int, default=6, choices=[1, 2, 4, 6, 8, 10, 15, 20])
    parser.add_argument('--test_loss_list', type=str, nargs='+', default=['L1Loss', 'SSIMLoss', 'MSELoss', 'PerceptLoss'], help='L1Loss, SSIMLoss, TVLoss, MSELoss, DisLoss, PerceptLoss, StyleLoss')
    parser.add_argument('--test_loss_weight', type=float, nargs='+', default=[1, 1, 1, 1], help='L1Loss, SSIMLoss, TVLoss, MSELoss, DisLoss')
    parser.add_argument('--l1_type', type=str, default='v2', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--dl_num', type=int, nargs='+',default=[1, 2, 3, 4], help='Number of feature layers used by the discriminator to compute the loss. Have 4 layers')
    parser.add_argument('--pl_num', type=int, nargs='+',default=[1, 2, 3, 4, 5], help='Number of feature layers used by the vgg16 to compute the loss. Have 23 layers,4 use. Vgg19 5 use')
    parser.add_argument('--vgg_type', type=str,default='vgg19', choices=['vgg16', 'vgg19'])
    # Adaptive block
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--use_deform_conv', action='store_true', default=False)
    parser.add_argument('--deform_conv_type', type=str, default='v1', choices=['v1', 'v2', 'pac', 'dy'])
    parser.add_argument('--use_deform_layer_down', type=int, nargs='+',default=[0, 1, 2, 3])
    parser.add_argument('--use_deform_layer_up', type=int, nargs='+',default=[0, 1, 2])
    parser.add_argument('--attention_type', default='self', choices=['auto', 'self', 'mult', 'spa', 'eca', 'ca'])
    parser.add_argument('--use_attn_layer_down', type=int, nargs='+',default=[0, 1, 2, 3])
    parser.add_argument('--use_attn_layer_up', type=int, nargs='+',default=[0, 1, 2])
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--reduction_ratio', type=int, default=8)
    # Path
    parser.add_argument(
    '--root_path', type=str, default='{}'.format(get_project_path()),
    help='Root path (default: %(default)s)')
    parser.add_argument('--train_data_dir', type=str,
                        help='directory of training data',
                        default='Hint_seismic/dataset/segc3/train',
                        choices=['Hint_seismic/dataset/segc3/train', 'Hint_seismic/dataset/mavg/train',
                                 'Hint_seismic/dataset/segc3/train', 'Hint_seismic/dataset/mavg/train'])
    parser.add_argument('--val_data_dir', type=str,
                        help='directory of val data',
                        default='Hint_seismic/dataset/segc3/val',
                        choices=['Hint_seismic/dataset/segc3/val', 'Hint_seismic/dataset/mavg/val',
                                 'Hint_seismic/dataset/segc3/val', 'Hint_seismic/dataset/mavg/val'])
    parser.add_argument('--test_data_dir', type=str,
                        help='directory of testing data',
                        default='Hint_seismic/dataset/segc3/test',
                        choices=['Hint_seismic/dataset/segc3/test', 'Hint_seismic/dataset/mavg/test',
                                 'Hint_seismic/dataset/segc3/test', 'Hint_seismic/dataset/mavg/test'])
    parser.add_argument('--dis_load_path', type=str,
                        help='path of pre-trained Discriminator',
                        default='Distill_model/models/Dis_pt/refineGanSSIMdilation4xin_0.1-0.3/dis_60.pt',
                        choices=['Distill_model/models/Dis_pt/refineGanSSIMdilation4xin_0.1-0.3/dis_60.pt',
                                 'Distill_model/models/Dis_pt/segc3_continus01_06_100_300_mask_in_epochscheduler/dis_500.pt',]
                )
    parser.add_argument('--save',help='path of saving model parameters',default='Distill_model/results',type=str)
    parser.add_argument(
        '--pt_path',
        help='path of saving model parameters',
        default=os.path.join(parser.parse_known_args()[0].save, "checkpt.pth"),
        type=str)
    parser.add_argument('--save_test_plot', action='store_true', default=False)

    config = parser.parse_args()

    return config


def train(config, model, train_data_loader, optimizer, scheduler, loss_list, SNR_function, PSNR_function, SSIM_function, inner_objective, outer_objective,
          state, epoch, device, iter):
    
    # model
    model_target = model[0]
    model_source = model[1]
    wnet_cd = model[2]
    lwnet_cd = model[3]
    branch_target_cd = model[4]
    wnet_sd = model[5]
    lwnet_sd = model[6]
    branch_target_sd = model[7]
    wnet_attnd = model[8]
    lwnet_attnd = model[9]
    branch_target_attnd = model[10]
    
    model_target.train()
    model_source.eval()
    if config.need_cd:
        wnet_cd.train()
        branch_target_cd.train()
        if config.loss_weight_cd:
            lwnet_cd.train()
    if config.need_sd:
        wnet_sd.train()
        branch_target_sd.train()
        if config.loss_weight_sd:
            lwnet_sd.train()
    if config.need_attnd:
        wnet_attnd.train()
        branch_target_attnd.train()
        if config.loss_weight_attnd:
            lwnet_attnd.train()
    
    wnet = [wnet_cd, wnet_sd, wnet_attnd]
    lwnet = [lwnet_cd, lwnet_sd, lwnet_attnd]
    branch_target = [branch_target_cd, branch_target_sd, branch_target_attnd]
    
    # state
    state['epoch'] = epoch
    
    # optimizer
    optimizer_target = optimizer[0]
    optimizer_source = optimizer[1]

    # loss
    loss_objects = {name: func() for name, func in loss_dict.items() if name in loss_list}# 根据 loss_list 初始化损失函数
    loss_function = LossCalculator(config, loss_objects, config.loss_weight, device)

    running_total_and_match_loss = 0.0
    running_total_loss = 0.0
    running_match_loss = 0.0
    running_match_loss_cd = 0.0
    running_match_loss_sd = 0.0
    running_match_loss_attnd = 0.0
    running_l1_loss = 0.0
    running_ssim_loss = 0.0
    running_mse_loss = 0.0
    running_tv_loss = 0.0
    running_dis_loss = 0.0
    running_percept_loss = 0.0
    running_style_loss = 0.0

    # SNR PSNR SSIM
    running_SNR = 0.0
    running_PSNR = 0.0
    running_SSIM = 0.0

    for i, (patch_H, patch_L, patch_H_noise, patch_name, patch_mask, ratio) in enumerate(train_data_loader):
        
        if i==config.max_meta_step*2:
            break
        
        start_time_batch = time.time()
        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)
        data = [patch_H[:config.meta_batch], patch_L[:config.meta_batch], patch_mask[:config.meta_batch]]
        batch_size = patch_H.size(0)
        
        # inner loop
        if i % 2 ==0:
            if config.distill_type == 'v2.1':
                # version 2.1
                for _ in range(config.T):
                    optimizer_target.zero_grad()
                    optimizer_target.step(inner_objective, config, model_target, model_source, wnet, lwnet, branch_target, 
                                        data, state, loss_function, matching_only=False)
            elif config.distill_type == 'v4.1':
                # version 4.1
                for _ in range(config.T):
                    optimizer_target.zero_grad()
                    optimizer_target.step(inner_objective, config, model_target, model_source, wnet, lwnet, branch_target, 
                                        data, state, loss_function, matching_only=True)

        # outer loop 
        # if i % 2 ==0:
        else: 
                optimizer_target.zero_grad()
                optimizer_target.step(outer_objective, config, model_target, data, state, loss_function)
                
                optimizer_target.zero_grad()
                optimizer_source.zero_grad()
                total_loss, state, out = outer_objective(config, model_target, data, state, loss_function)
                total_loss.backward()
                
                start_time = time.time()
                optimizer_target.meta_backward(state, model_source)
                end_time = time.time()
                state['meta_backward_use_time'] = end_time-start_time
                optimizer_source.step()
                
                end_time_batch = time.time()
                state['meta_batch_use_time'] = end_time_batch-start_time_batch

    # Learning of Model
    num = 0
    min_ratio, max_ratio = 1.0, 0.0
    for step, (patch_H, patch_L, patch_H_noise, patch_name, patch_mask, ratio) in enumerate(train_data_loader):
    
        start_time_batch = time.time()
        
        num = step + 1
        max_val = ratio.max().item()
        min_val = ratio.min().item()
        if max_val >= max_ratio:
            max_ratio = max_val
        if min_val <= min_ratio:
            min_ratio = min_val

        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)
        data = [patch_H, patch_L, patch_mask]
        batch_size = patch_H.size(0)        
        
        for _ in range(config.L):
            optimizer_target.zero_grad()
            total_and_match_loss, total_loss, match_loss, state, out = inner_objective(config, model_target, model_source, wnet, lwnet, branch_target, data, state, loss_function, matching_only=False, meta_test=True)
            total_and_match_loss.backward()
            optimizer_target.step(None) # 执行一次普通的优化step 不用记录在优化器中

        total_and_match_loss = state['total_and_match_loss']
        match_loss = state['match_loss']
        matching_loss_cd = state['match_loss_cd']
        matching_loss_sd = state['match_loss_sd']
        matching_loss_attnd = state['match_loss_attnd']
            
        l1_loss, ssim_loss, mse_loss, tv_loss, dis_loss, percept_loss, style_loss, l2_reg_loss = itemgetter(*loss_keys)(state)

        start_time = time.time()
        running_total_and_match_loss += total_and_match_loss.data.cpu().detach().numpy()
        running_total_loss += total_loss.data.cpu().detach().numpy()
        running_match_loss += match_loss.data.cpu().detach().numpy()
        running_match_loss_cd += matching_loss_cd.data.cpu().detach().numpy()
        running_match_loss_sd += matching_loss_sd.data.cpu().detach().numpy()
        running_match_loss_attnd += matching_loss_attnd.data.cpu().detach().numpy()
        running_l1_loss += l1_loss.data.cpu().detach().numpy()
        running_ssim_loss += ssim_loss.data.cpu().detach().numpy()
        running_mse_loss += mse_loss.data.cpu().detach().numpy()
        running_tv_loss += tv_loss.data.cpu().detach().numpy()
        running_dis_loss += dis_loss.data.cpu().detach().numpy()
        running_percept_loss += percept_loss.data.cpu().detach().numpy()
        running_style_loss += style_loss.data.cpu().detach().numpy()

        running_SNR += (SNR_function(patch_H, out).mean()).data.cpu().detach().numpy()
        running_PSNR += PSNR_function(input=out, target=patch_H, max_val=1.0).mean().data.cpu().detach().numpy()
        # running_PSNR += PSNR_function(patch_H, out, channel_axis=-1, data_range=1).mean().data.cpu().detach().numpy()
        running_SSIM += SSIM_function(patch_H, out).mean().data.cpu().detach().numpy()
        
        end_time = time.time()
        state['loss_ind_use_time'] = end_time-start_time
        
        end_time_batch = time.time()
        state['batch_use_time'] = end_time_batch-start_time_batch
        
        if num % config.print_interval == 1 and epoch % 200 == 1:
            pbar.write(f"num:{num}, w_cd:{state['weights_cd']}, lw_cd:{state['loss_weights_cd']}; , w_sd:{state['weights_sd']}, lw_sd:{state['loss_weights_sd']};, w_attnd:{state['weights_attnd']}, lw_attnd:{state['loss_weights_attnd']};meta_backward_use_time:{state['meta_backward_use_time']:.5f}, loss_ind_use_time:{state['loss_ind_use_time']:.5f}, meta_batch_use_time:{state['meta_batch_use_time']:.5f}, batch_use_time:{state['batch_use_time']:.5f}")
             
        iter = iter + 1
        
    scheduler.step()

    return running_total_and_match_loss/num, running_total_loss/num, running_match_loss/num, running_match_loss_cd/num, running_match_loss_sd/num, running_match_loss_attnd/num, running_l1_loss/num, running_ssim_loss/num, running_mse_loss/num, running_tv_loss/num, running_dis_loss/num, running_percept_loss/num, running_style_loss/num, \
           running_SNR/num, running_PSNR/num, running_SSIM/num, min_ratio, max_ratio


def val(config, model, val_data_loader, loss_list, SNR_function, PSNR_function, SSIM_function, state, epoch, device):

    # model
    model.eval()

    # loss
    loss_objects = {name: func() for name, func in loss_dict.items() if name in loss_list} # 根据 loss_list 初始化损失函数
    loss_function = LossCalculator(config, loss_objects, config.test_loss_weight, device)

    running_total_loss = 0.0
    running_l1_loss = 0.0
    running_ssim_loss = 0.0
    running_mse_loss = 0.0
    running_tv_loss = 0.0
    running_dis_loss = 0.0
    running_percept_loss = 0.0
    running_style_loss = 0.0

    # SNR PSNR SSIM
    running_SNR = 0.0
    running_PSNR = 0.0
    running_SSIM = 0.0

    num = 0
    min_ratio, max_ratio = 1.0, 0.0

    for step, (patch_H, patch_L, patch_H_noise, patch_name, patch_mask, ratio) in enumerate(val_data_loader):
        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)

        num = step + 1
        max = ratio.max().item()
        min = ratio.min().item()
        if max >= max_ratio:
            max_ratio = max
        if min <= min_ratio:
            min_ratio = min

        batch_size = patch_H.size(0)

        with torch.no_grad():
            if config.use_attention:
                out, feat, attn, v = model(patch_L, mask=patch_mask)
            else:
                 out, feat = model(patch_L, mask=patch_mask)
            
            total_loss, state = loss_function(state, patch_H, out, patch_mask, test=True)
            l1_loss, ssim_loss, mse_loss, tv_loss, dis_loss, percept_loss, style_loss, l2_reg_loss = itemgetter(*loss_keys)(state)
            matching_loss_cd = state['match_loss_cd']
            matching_loss_sd = state['match_loss_sd']
            matching_loss_attnd = state['match_loss_attnd']

            running_total_loss += total_loss.data.cpu().detach().numpy()
            running_l1_loss += l1_loss.data.cpu().detach().numpy()
            running_ssim_loss += ssim_loss.data.cpu().detach().numpy()
            running_mse_loss += mse_loss.data.cpu().detach().numpy()
            running_tv_loss += tv_loss.data.cpu().detach().numpy()
            running_dis_loss += dis_loss.data.cpu().detach().numpy()
            running_percept_loss += percept_loss.data.cpu().detach().numpy()
            running_style_loss += style_loss.data.cpu().detach().numpy()

            running_SNR += (SNR_function(patch_H, out).mean()).data.cpu().detach().numpy()
            running_PSNR += PSNR_function(input=out, target=patch_H, max_val=1.0).mean().data.cpu().detach().numpy()
            running_SSIM += SSIM_function(patch_H, out).mean().data.cpu().detach().numpy()
            
    num = num

    return running_total_loss/num, running_l1_loss/num, running_ssim_loss/num, running_mse_loss/num, running_tv_loss/num, running_dis_loss/num, running_percept_loss/num, running_style_loss/num, \
           running_SNR/num, running_PSNR/num, running_SSIM/num, min_ratio, max_ratio

def test(config, model, test_data_loader, loss_list, SNR_function, PSNR_function, SSIM_function, state, device):
    model.eval()

    # loss
    loss_objects = {name: func() for name, func in loss_dict.items() if name in loss_list} # 根据 loss_list 初始化损失函数
    loss_function = LossCalculator(config, loss_objects, config.test_loss_weight, device)

    running_total_loss = 0.0
    running_l1_loss = 0.0
    running_ssim_loss = 0.0
    running_mse_loss = 0.0
    running_tv_loss = 0.0
    running_dis_loss = 0.0
    running_percept_loss = 0.0

    # SNR PSNR SSIM
    running_SNR = 0.0
    running_PSNR = 0.0
    running_SSIM = 0.0

    num = 0
    
    min_ratio, max_ratio = 1.0, 0.0

    pbar = tqdm(total=len(test_data_loader), ncols=200)
    for step, (patch_H, patch_L, patch_H_noise, patch_name, patch_mask, ratio) in enumerate(test_data_loader):
        
        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)
        
        num = step + 1
        max_val = ratio.max().item()
        min_val = ratio.min().item()
        if max_val >= max_ratio:
            max_ratio = max_val
        if min_val <= min_ratio:
            min_ratio = min_val
        
        batch_size = patch_H.size(0)

        with torch.no_grad():
            if config.use_attention:
                out, feat, attn, v = model(patch_L, mask=patch_mask)
            else:
                 out, feat = model(patch_L, mask=patch_mask)
            if step == 0:
                test_outputs = out.cpu()
                patch_Hs = patch_H.cpu()
                patch_Ls = patch_L.cpu()
                patch_masks = patch_mask.cpu()
            else:
                test_outputs = torch.cat((test_outputs, out.cpu()), dim=0)
                patch_Hs = torch.cat((patch_Hs, patch_H.cpu()), dim=0)
                patch_Ls = torch.cat((patch_Ls, patch_L.cpu()), dim=0)
                patch_masks = torch.cat((patch_masks, patch_mask.cpu()), dim=0)
        
            total_loss, state = loss_function(state, patch_H, out, patch_mask, test=True)
            l1_loss = state['l1_loss']
            ssim_loss = state['ssim_loss']
            mse_loss = state['mse_loss']
            tv_loss = state['tv_loss']
            dis_loss = state['dis_loss']
            percept_loss = state['percept_loss']

            running_total_loss += total_loss.data.cpu().detach().numpy()
            running_l1_loss += l1_loss.data.cpu().detach().numpy()
            running_ssim_loss += ssim_loss.data.cpu().detach().numpy()
            running_mse_loss += mse_loss.data.cpu().detach().numpy()
            running_tv_loss += tv_loss.data.cpu().detach().numpy()
            running_dis_loss += dis_loss.data.cpu().detach().numpy()
            running_percept_loss += percept_loss.data.cpu().detach().numpy()
            
            running_SNR += (SNR_function(patch_H, out).mean()).data.cpu().detach().numpy()
            running_PSNR += PSNR_function(input=out, target=patch_H, max_val=1.0).mean().data.cpu().detach().numpy()
            running_SSIM += SSIM_function(patch_H, out).mean().data.cpu().detach().numpy()
        
        pbar.set_description(' '.join(['Step: [{0}/{1}]  '.format(step, len(test_data_loader))] + 
                                      ['running_test_l1_loss {:0.3e}'.format(running_l1_loss / num)] + ['running_test_mse_loss {:0.3e}'.format(running_mse_loss / num)] + 
                                      ['running_test_snr {:0.5f}'.format(running_SNR / num)] + ['running_test_psnr {:0.5f}'.format(running_PSNR / num)] + ['running_test_ssim {:0.5f}'.format(running_SSIM / num)] ))
        pbar.update(1)
    pbar.close()

    test_l1_loss = running_l1_loss / num
    test_mse_loss = running_mse_loss / num
    testSNR = running_SNR / num
    testpsnr = running_PSNR / num
    testSSIM = running_SSIM / num

    imputed_data = patch_masks * patch_Hs + (1 - patch_masks) * test_outputs  # patch_Ls + (1 - patch_masks) * test_outputs
    test_outputs = test_outputs.cpu().detach().numpy()
    imputed_data = imputed_data.cpu().detach().numpy()

    return patch_Hs, patch_Ls, patch_masks, test_outputs, imputed_data, test_l1_loss, test_mse_loss, testSNR, testpsnr, testSSIM, min_ratio, max_ratio


if __name__ == '__main__':

    # Base setting
    config = get_args()
    seed_rng(config.seed)
    print("config:", json.dumps(vars(config), indent=4))
    print("Exp PID = {}".format(os.getpid()))
    device = torch.device('cuda:' + str(config.gpu) if torch.cuda.is_available() else 'cpu')
    config.data_type = torch_dtype(config.data_type)
    Epoch = config.Epoch
    pairs = pair_to_list(config.pairs)
    pairs_sd = pair_to_list(config.pairs_sd)
    pairs_attnd = pair_to_list(config.pairs_attnd)
    # roll_back = config.distill_type in {'v1', 'v2', 'v2.1'}
    if config.online_vis == 'wandb':
        run = wandb.init(
                        project="Distill_model", 
                         name='{}'.format('{}_{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], config.sampler, config.save.split('/')[-1])),
                        config=vars(config),
                        settings=wandb.Settings(init_timeout=300))
    elif config.online_vis == 'swanlab':
        run = swanlab.init(
                        project="Distill_model", 
                        experiment_name='{}'.format('{}_{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], config.sampler, config.save.split('/')[-1])),
                        config=vars(config),)
    else:
        run = None
    
    # Path
    root_path = config.root_path
    if hostname == "ubuntu-SYS-420GP-TNR":
        config.train_data_dir = os.path.join(root_path, "link8TB/dax_result/datasets", config.train_data_dir)
        config.val_data_dir = os.path.join(root_path, "link8TB/dax_result/datasets", config.val_data_dir)
        config.test_data_dir = os.path.join(root_path, "link8TB/dax_result/datasets", config.test_data_dir)
        config.model_source_load_path = os.path.join(root_path, "link8TB/dax_result/datasets/Hint_seismic", config.model_source_load_path)
        config.save = os.path.join(root_path, "link8TB/dax_result/datasets/Hint_seismic", config.save)
        config.pt_path = os.path.join(root_path, "link8TB/dax_result/datasets/Hint_seismic", config.pt_path)
    elif hostname == "ubuntu":
        config.train_data_dir = os.path.join(root_path.replace('code', 'link14TB'), config.train_data_dir)
        config.val_data_dir = os.path.join(root_path.replace('code', 'link14TB'), config.val_data_dir)
        config.test_data_dir = os.path.join(root_path.replace('code', 'link14TB'), config.test_data_dir)
        config.model_source_load_path = os.path.join(root_path.replace('code', 'link14TB'), config.model_source_load_path)
        config.save = os.path.join(root_path.replace('code', 'link14TB'), config.save)
        config.pt_path = os.path.join(root_path.replace('code', 'link14TB'), config.pt_path)
    elif hostname == "cxg3004-zhang":
        config.train_data_dir = os.path.join(root_path, config.train_data_dir)
        config.val_data_dir = os.path.join(root_path, config.val_data_dir)
        config.test_data_dir = os.path.join(root_path, config.test_data_dir)
        config.model_source_load_path = os.path.join(root_path, config.model_source_load_path)
        config.save = os.path.join(root_path, config.save)
        config.pt_path = os.path.join(root_path, config.pt_path)
    elif hostname == "xjtu-Sun-04":
        config.train_data_dir = os.path.join(root_path.replace('code', 'data'), config.train_data_dir)
        config.val_data_dir = os.path.join(root_path.replace('code', 'data'), config.val_data_dir)
        config.test_data_dir = os.path.join(root_path.replace('code', 'data'), config.test_data_dir)
        config.model_source_load_path = os.path.join(root_path, config.model_source_load_path)
        config.save = os.path.join(root_path, config.save)
        config.pt_path = os.path.join(root_path, config.pt_path)
    elif hostname == "xjtu-Sun-08":
        config.train_data_dir = os.path.join(root_path.replace('demo', 'data'), config.train_data_dir)
        config.val_data_dir = os.path.join(root_path.replace('demo', 'data'), config.val_data_dir)
        config.test_data_dir = os.path.join(root_path.replace('demo', 'data'), config.test_data_dir)
        config.save = os.path.join(root_path, config.save)
        config.pt_path = os.path.join(root_path, config.pt_path)
    elif hostname == "PC-20201112YXPH":
        config.train_data_dir = os.path.join('G:\seismic data', config.train_data_dir)
        config.val_data_dir = os.path.join('G:\seismic data', config.val_data_dir)
        config.test_data_dir = os.path.join('G:\seismic data', config.test_data_dir)
        config.model_source_load_path = os.path.join('G:', config.model_source_load_path)
        config.save = os.path.join('G:', config.save)
        config.pt_path = os.path.join('G:',  config.pt_path)
    else:
        print("hostname:", hostname, "Unkown")

    # Model initialization
    model_source, model_target, source_config = load_model(config, config.train_source_model, device=device)
    if config.need_cd:
        wnet_cd = WeightNetwork(config, config.source_model, pairs)
        branch_target_cd = FeatureMatching([config, source_config], config.source_model, config.target_model, pairs, version=config.feature_matching_type)
        if config.loss_weight_cd:
            lwnet_cd = LossWeightNetwork(source_config, config.source_model, pairs, config.loss_weight_type, config.loss_weight_init)
        else:
            lwnet_cd = None
    else:
        wnet_cd = None
        branch_target_cd =None
        lwnet_cd = None
    
    if config.need_sd:
        wnet_sd = WeightNetwork(config, config.source_model_sd, pairs_sd)
        branch_target_sd = FeatureMatching([config, config], config.source_model_sd, config.target_model_sd, pairs_sd, version=config.feature_matching_type_sd)
        if config.loss_weight_sd:
            lwnet_sd = LossWeightNetwork(config, config.source_model_sd, pairs_sd, config.loss_weight_type, config.loss_weight_init_sd)
        else:
            lwnet_sd = None
    else:
        wnet_sd = None
        branch_target_sd = None
        lwnet_sd = None

    if config.need_attnd:
        wnet_attnd = WeightNetwork(config, config.source_model_attnd, pairs_attnd, attnd=True)
        branch_target_attnd = FeatureMatching([config, source_config], config.source_model_attnd, config.target_model_attnd, pairs_attnd, attnd=True, version=config.feature_matching_type_attnd)
        if config.loss_weight_attnd:
            lwnet_attnd = LossWeightNetwork(source_config, config.source_model_attnd, pairs_attnd, config.loss_weight_type, config.loss_weight_init_attnd, attnd=True) 
        else:
            lwnet_attnd = None
    else:
        wnet_attnd = None
        branch_target_attnd = None
        lwnet_attnd = None

    ssim = SSIM(window_size=int(3))

    # Load AN pretrained weights
    model_source_load_path = config.model_source_load_path
    if os.path.exists(model_source_load_path):
        try:
            model_source.load_state_dict(torch.load(model_source_load_path, map_location=device)['state_dict'], strict=True) # strict=False
        except Exception as e:
            model_source.load_state_dict(torch.load(model_source_load_path, map_location=device)['model'], strict=True) # strict=False
    else:
        print(f"Warning: {model_source_load_path} does not exist. Skipping source model loading.")
        model_source = model_source

    # TO device
    model_target = model_target.to(device, dtype=config.data_type)
    model_source = model_source.to(device, dtype=config.data_type)
    print("model_source", model_source)
    print("model_target", model_target)
    if 'PerceptLoss' in config.loss_list:
        loss_dict.update({'PerceptLoss': lambda: PerceptLoss(pl_num=config.pl_num, vgg_type=config.vgg_type, device=device, dtype=config.data_type)})
    if 'StyleLoss' in config.loss_list:
        loss_dict.update({'StyleLoss': lambda: StyleLoss(device=device, dtype=config.data_type)})
    if config.need_cd:
        wnet_cd = wnet_cd.to(device, dtype=config.data_type)
        branch_target_cd = branch_target_cd.to(device, dtype=config.data_type)
        if config.loss_weight_cd:
            lwnet_cd = lwnet_cd.to(device, dtype=config.data_type)
    if config.need_sd:
        wnet_sd = wnet_sd.to(device, dtype=config.data_type)
        branch_target_sd = branch_target_sd.to(device, dtype=config.data_type)
        if config.loss_weight_sd:
            lwnet_sd = lwnet_sd.to(device, dtype=config.data_type)
    if config.need_attnd:
        wnet_attnd = wnet_attnd.to(device, dtype=config.data_type)
        branch_target_attnd = branch_target_attnd.to(device, dtype=config.data_type)
        if config.loss_weight_attnd:
            lwnet_attnd = lwnet_attnd.to(device, dtype=config.data_type)
    
    # Parameters
    wnet_params_cd = list(wnet_cd.parameters()) if config.need_cd else []
    wnet_params_sd = list(wnet_sd.parameters()) if config.need_sd else []
    wnet_params_attnd = list(wnet_attnd.parameters()) if config.need_attnd else []
    lwnet_params_cd = list(lwnet_cd.parameters()) if (config.need_cd and config.loss_weight_cd) else []
    lwnet_params_sd = list(lwnet_sd.parameters()) if (config.need_sd and config.loss_weight_sd) else []
    lwnet_params_attnd = list(lwnet_attnd.parameters()) if (config.need_attnd and config.loss_weight_attnd) else []
    branch_parameters_cd = list(branch_target_cd.parameters())  if config.need_cd else []
    branch_parameters_sd = list(branch_target_sd.parameters()) if config.need_sd else []
    branch_parameters_attnd = list(branch_target_attnd.parameters()) if config.need_attnd else []
    model_target_parameters = list(model_target.parameters())
    
    weight_params = wnet_params_cd + wnet_params_sd + wnet_params_attnd
    weight_params_count = count_parameters(wnet_cd) + count_parameters(wnet_sd)  + count_parameters(wnet_attnd) 
    if config.feature_matching_type == 'v2' or 'v3':
        weight_params = weight_params + branch_parameters_cd
        weight_params_count = weight_params_count + count_parameters(branch_target_cd)
    # if config.loss_weight_cd:
    weight_params = weight_params + lwnet_params_cd
    weight_params_count = weight_params_count + count_parameters(lwnet_cd)
    # if config.loss_weight_sd:
    weight_params = weight_params + lwnet_params_sd
    weight_params_count = weight_params_count + count_parameters(lwnet_sd)
    # if config.loss_weight_attnd:
    weight_params = weight_params + lwnet_params_attnd
    weight_params_count = weight_params_count + count_parameters(lwnet_attnd) 
    print("weight_params_count", format_parameters_as_million(weight_params_count))
    
    if config.feature_matching_type == 'v1' and config.feature_matching_type_sd == 'v1':
        target_params = model_target_parameters + branch_parameters_cd + branch_parameters_sd + branch_parameters_attnd
        target_params_count = count_parameters(model_target) + count_parameters(branch_target_cd) + count_parameters(branch_target_sd) + count_parameters(branch_target_attnd)
    elif (config.feature_matching_type == 'v2' or 'v3') and config.feature_matching_type_sd == 'v1':
        target_params = model_target_parameters + branch_parameters_sd + branch_parameters_attnd
        target_params_count = count_parameters(model_target) + count_parameters(branch_target_sd) + count_parameters(branch_target_attnd)
    elif (config.feature_matching_type_sd == 'v2' or 'v3') and config.feature_matching_type == 'v1':
        target_params = model_target_parameters + branch_parameters_cd + branch_parameters_attnd
        target_params_count = count_parameters(model_target) + count_parameters(branch_target_cd) + count_parameters(branch_target_attnd)
    elif (config.feature_matching_type == 'v2' or 'v3') and (config.feature_matching_type_sd == 'v2' or 'v3'):
        target_params = model_target_parameters + branch_parameters_attnd
        target_params_count = count_parameters(model_target) + count_parameters(branch_target_attnd)
    print("target_params_count", format_parameters_as_million(target_params_count))
    
    # Optimizers & scheduler
    if config.optimizer == 'ADAM':
        optimizer_source = optim.Adam(weight_params, lr=config.meta_lr, weight_decay=config.meta_wd)
    elif config.optimizer == 'SGD':
        optimizer_source = optim.SGD(weight_params, lr=config.meta_lr, weight_decay=config.meta_wd)
    else:
        optimizer_source = optim.Adam(weight_params, lr=config.meta_lr, weight_decay=config.meta_wd)
    if config.feature_matching_type == 'v1' and config.feature_matching_type_sd == 'v1':
        modules = [model_target, branch_target_cd, branch_target_sd, branch_target_attnd]
        valid_modules = [m for m in modules if m is not None]  # 移除所有None值模块
        optimizer_target = MetaSGD(target_params, valid_modules, lr=config.lr, weight_decay=config.wd, momentum=config.momentum, rollback=config.roll_back, cpu=config.T > 2)
    elif (config.feature_matching_type == 'v2' or 'v3') and config.feature_matching_type_sd == 'v1':
        modules = [model_target, branch_target_sd, branch_target_attnd]
        valid_modules = [m for m in modules if m is not None]  # 移除所有None值模块
        optimizer_target = MetaSGD(target_params, valid_modules, lr=config.lr, weight_decay=config.wd, momentum=config.momentum, rollback=config.roll_back, cpu=config.T > 2)
    elif (config.feature_matching_type_sd == 'v2' or 'v3') and config.feature_matching_type == 'v1':
        modules = [model_target, branch_target_cd, branch_target_attnd]
        valid_modules = [m for m in modules if m is not None]  # 移除所有None值模块
        optimizer_target = MetaSGD(target_params, valid_modules, lr=config.lr, weight_decay=config.wd, momentum=config.momentum, rollback=config.roll_back, cpu=config.T > 2)
    elif (config.feature_matching_type == 'v2' or 'v3') and (config.feature_matching_type_sd == 'v2' or 'v3'):
        modules = [model_target, branch_target_attnd]
        valid_modules = [m for m in modules if m is not None]  # 移除所有None值模块
        optimizer_target = MetaSGD(target_params, valid_modules, lr=config.lr, weight_decay=config.wd, momentum=config.momentum, rollback=config.roll_back, cpu=config.T > 2)
    
    if config.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer_target, step_size=config.step_size, gamma=config.gamma, last_epoch=-1)  # 每 100 个 epoch，学习率降低
    elif config.scheduler == 'CosineAnnealingLR':
        scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer_target, T_max=config.Epoch)
    
    epoch_start = 1
    if config.resume:
        pt = torch.load(config.pt_path, map_location=device)
        config = pt['config']
        model_source.load_state_dict(pt['model_source'], strict=True)
        optimizer_source.load_state_dict(pt['optimizer_source'], strict=True)
        model_target.load_state_dict(pt['model_target'], strict=True)
        optimizer_target.load_state_dict(pt['optimizer_target'], strict=True)
        scheduler.load_state_dict(pt['scheduler'], strict=True)
        epoch_start = pt['optimizer_source']
        if config.need_cd:
            wnet_cd.load_state_dict(pt['wnet_cd'], strict=True)
            branch_target_cd.load_state_dict(pt['branch_target_cd'], strict=True)
            if config.loss_weight_cd:
                lwnet_cd.load_state_dict(pt['lwnet_cd'], strict=True)
        if config.need_sd:
            wnet_sd.load_state_dict(pt['wnet_sd'], strict=True)
            branch_target_sd.load_state_dict(pt['branch_target_sd'], strict=True)
            if config.loss_weight_sd:
                lwnet_sd.load_state_dict(pt['lwnet_sd'], strict=True)
        if config.need_attnd:
            wnet_attnd.load_state_dict(pt['wnet_attnd'], strict=True)
            branch_target_attnd.load_state_dict(pt['branch_target_attnd'], strict=True)
            if config.loss_weight_attnd:
                lwnet_attnd.load_state_dict(pt['lwnet_attnd'], strict=True)

    # Information records & State
    train_total_and_match_loss_meter = AverageMeter()
    train_total_loss_meter = AverageMeter()
    train_match_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    train_snr_meter = AverageMeter()
    val_snr_meter = AverageMeter()
    train_psnr_meter = AverageMeter()
    val_psnr_meter = AverageMeter()
    train_ssim_meter = AverageMeter()
    val_ssim_meter = AverageMeter()
    
    state = {}
    
    # data
    if config.dataset == 'SEGC3':
        train_dataset = Dataset_SEGC3(config, config.train_data_dir, config.sampler, config.mode)
        val_dataset = Dataset_SEGC3(config, config.val_data_dir, config.sampler, 'val')
        test_dataset = Dataset_SEGC3(config, config.test_data_dir, config.sampler, 'test')
    elif config.dataset == 'MAVG':
        train_dataset = Dataset_MAVG(config, config.train_data_dir, config.sampler, config.mode)
        val_dataset = Dataset_MAVG(config, config.val_data_dir, config.sampler, 'val')
        test_dataset = Dataset_MAVG(config, config.test_data_dir, config.sampler, 'test')

    train_data_loader = DataLoader(train_dataset, config.batch_train, shuffle=True, drop_last=True, pin_memory=True, num_workers=config.num_workers)
    val_data_loader = DataLoader(val_dataset, config.batch_val, shuffle=False, drop_last=False, num_workers=config.num_workers)
    test_data_loader = DataLoader(test_dataset, config.batch_test, shuffle=False, drop_last=False, num_workers=config.num_workers)

    pbar = tqdm(total=Epoch, ncols=600)

    # pack
    model = [model_target, model_source, wnet_cd, lwnet_cd, branch_target_cd, wnet_sd, lwnet_sd, branch_target_sd, wnet_attnd, lwnet_attnd, branch_target_attnd]
    optimizer = [optimizer_target, optimizer_source]
    
    # Train & val
    best_loss = float("inf")
    best_snr = float("-inf")
    best_psnr = float("-inf")
    best_ssim = float("-inf")

    # Train loss every epoch
    Training_total_and_match_loss = []
    Training_total_loss = []
    Training_match_loss = []
    Training_match_loss_cd = []
    Training_match_loss_sd = []
    Training_match_loss_attnd = []
    Training_l1_loss = []
    Training_ssim_loss = []
    Training_mse_loss = []
    Training_tv_loss = []
    Training_dis_loss = []
    Training_percept_loss = []
    Training_style_loss = []
    Training_SNR = []
    Training_PSNR = []
    Training_SSIM = []

    # Val loss every epoch
    Val_total_loss = []
    Val_l1_loss = []
    Val_ssim_loss = []
    Val_mse_loss = []
    Val_tv_loss = []
    Val_dis_loss = []
    Val_percept_loss = []
    Val_style_loss = []
    Val_SNR = []
    Val_PSNR = []
    Val_SSIM = []

    iter = 0


    for epoch in range(epoch_start, Epoch + 1):


        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # TRAIN
        start_time = time.time()
        train_total_and_match_loss, train_total_loss, train_match_loss, train_match_loss_cd, train_match_loss_sd, train_match_loss_attnd, train_l1_loss, train_ssim_loss, train_mse_loss, train_tv_loss, train_dis_loss, train_percept_loss, train_style_loss, \
        train_SNR, train_PSNR, train_SSIM, train_min_ratio, train_max_ratio = \
        train(config, model, train_data_loader, optimizer, scheduler, config.loss_list, snr, psnr, ssim, inner_objective, outer_objective, 
              state, epoch, device, iter)

        # Log metrics to wandb.
        if config.online_vis != 'not_use':
            run.log({"train_total_and_match_loss":train_total_and_match_loss,"train_total_loss":train_total_loss, "train_match_loss":train_match_loss,"train_match_loss_cd":train_match_loss_cd, "train_match_loss_sd":train_match_loss_sd, "train_match_loss_attnd":train_match_loss_attnd,
                    "train_l1_loss": train_l1_loss, "train_ssim_loss": train_ssim_loss, "train_mse_loss": train_mse_loss, "train_percept_loss": train_percept_loss, "train_style_loss": train_style_loss,
                    "train_SNR":train_SNR, "train_PSNR":train_PSNR, "train_SSIM":train_SSIM})
        
        if epoch == 1:
            pbar.write("\n Train {} missing min_ratio {:0.3f}, {} missing max_ratio {:0.3f}".format(config.sampler, train_min_ratio, 
                                                                                        config.sampler,train_max_ratio))
            end_time = time.time()
            hours, minutes, seconds = format_times_as_hms(end_time-start_time)
            pbar.write(f"train_use_time: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
        
        Training_total_and_match_loss.append(train_total_and_match_loss)
        Training_total_loss.append(train_total_loss)
        Training_match_loss.append(train_match_loss)
        Training_match_loss_cd.append(train_match_loss_cd)
        Training_match_loss_sd.append(train_match_loss_sd)
        Training_match_loss_attnd.append(train_match_loss_attnd)
        Training_l1_loss.append(train_l1_loss)
        Training_ssim_loss.append(train_ssim_loss)
        Training_mse_loss.append(train_mse_loss)
        Training_tv_loss.append(train_tv_loss)
        Training_dis_loss.append(train_dis_loss)
        Training_percept_loss.append(train_percept_loss)
        Training_style_loss.append(train_style_loss)
        Training_SNR.append(train_SNR)
        Training_PSNR.append(train_PSNR)
        Training_SSIM.append(train_SSIM)

        train_total_and_match_loss_meter.update(train_total_and_match_loss)
        train_total_loss_meter.update(train_total_loss)
        train_match_loss_meter.update(train_match_loss)
        train_snr_meter.update(train_SNR)
        train_psnr_meter.update(train_PSNR)
        train_ssim_meter.update(train_SSIM)

        # VAL
        val_total_loss, val_l1_loss, val_ssim_loss, val_mse_loss, val_tv_loss, val_dis_loss, val_percept_loss, val_style_loss, \
        val_SNR, val_PSNR, val_SSIM, val_min_ratio, val_max_ratio = \
            val(config, model[0], val_data_loader, config.test_loss_list, snr, psnr, ssim, state, epoch, device)

        # Log metrics to wandb.
        if config.online_vis != 'not_use':
            run.log({"val_total_loss":val_total_loss, "val_l1_loss": val_l1_loss, "val_ssim_loss": val_ssim_loss, "val_mse_loss": val_mse_loss, "val_percept_loss": val_percept_loss, "val_style_loss": val_style_loss,
                    "val_SNR":val_SNR, "val_PSNR":val_PSNR, "val_SSIM":val_SSIM})

        if epoch == 1:
            end_time = time.time()
            hours, minutes, seconds = format_times_as_hms(end_time-start_time)
            pbar.write(f"train&val_use_time: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
            pbar.write("\n Val {} missing min_ratio {:0.3f}, {} missing max_ratio {:0.3f}".format(config.sampler, val_min_ratio, 
                                                                                      config.sampler,val_max_ratio))
        
        Val_total_loss.append(val_total_loss)
        Val_l1_loss.append(val_l1_loss)
        Val_ssim_loss.append(val_ssim_loss)
        Val_mse_loss.append(val_mse_loss)
        Val_tv_loss.append(val_tv_loss)
        Val_dis_loss.append(val_dis_loss)
        Val_percept_loss.append(val_percept_loss)
        Val_style_loss.append(val_style_loss)
        Val_SNR.append(val_SNR)
        Val_PSNR.append(val_PSNR)
        Val_SSIM.append(val_SSIM)

        val_loss_meter.update(val_total_loss)
        val_snr_meter.update(val_SNR)
        val_psnr_meter.update(val_PSNR)
        val_ssim_meter.update(val_SSIM)

        # 训练后更新state中的模型参数
        state['model_source'] = model_source.state_dict()
        state['optimizer_source'] = optimizer_source.state_dict()
        state['model_target'] = model_target.state_dict()
        state['optimizer_target'] = optimizer_target.state_dict()
        state['scheduler'] = scheduler.state_dict()
        state['epoch'] = epoch
        if config.need_cd:
            state['wnet_cd'] = wnet_cd.state_dict()
            state['branch_target_cd'] = branch_target_cd.state_dict()
            if config.loss_weight_cd:
                state['lwnet_cd'] = lwnet_cd.state_dict()
        if config.need_sd:
            state['branch_target_sd'] = branch_target_sd.state_dict()
            state['wnet_sd'] = wnet_sd.state_dict()
            if config.loss_weight_sd:
                state['lwnet_sd'] = lwnet_sd.state_dict()
        if config.need_attnd:
            state['branch_target_attnd'] = branch_target_attnd.state_dict()
            state['wnet_attnd'] = wnet_attnd.state_dict()
            if config.loss_weight_attnd:
                state['lwnet_attnd'] = lwnet_attnd.state_dict()
    
        # SAVE STATE USE PSNR
        val_psnr = val_psnr_meter.val
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            makedirs(config.save)
            if  best_psnr > 20:
                torch.save({
                    "config": config,
                    **state # 将state中的所有键值对解包并添加到字典中:model_target, branch_target, optimizer_target wnet, lwnet, loss_weights, loss
                }, os.path.join(config.save, "checkpt.pth"))
        if epoch % config.save_interval == 100000:
            torch.save({
            "config": config,
            **state,}, os.path.join(config.save, "checkpt_{}.pth".format(epoch)))
                    
        # PBAR
        description = ' '.join(['Epoch: [{0}/{1}]  '.format(epoch, Epoch)] +  ['total_and_match_loss {:0.5f}'.format(train_total_and_match_loss_meter.val)]
                                      + ['total_loss {:0.5f}'.format(train_total_loss_meter.val)] +  ['match_loss {:0.5f}'.format(train_match_loss_meter.val)]
                                      +  ['match_loss_cd {:0.5f}'.format(train_match_loss_cd)] +  ['match_loss_sd {:0.5f}'.format(train_match_loss_sd)] +  ['match_loss_attnd {:0.5f}'.format(train_match_loss_attnd)]
                                      + ['val_loss {:0.5f}'.format(val_loss_meter.val)] + ['val_snr {:0.5f}'.format(val_snr_meter.val)]
                                      + ['val_psnr {:0.5f}'.format(val_psnr_meter.val)] + ['val_ssim {:0.5f}'.format(val_ssim_meter.val)]
                                      + ['loss_weights_cd {}'.format(state['loss_weights_cd'])] + ['weights_cd {}'.format(state['weights_cd'])]
                                      + ['loss_weights_sd {}'.format(state['loss_weights_sd'])] + ['weights_cd {}'.format(state['weights_sd'])]
                                      + ['loss_weights_attnd {}'.format(state['loss_weights_attnd'])] + ['weights_attnd {}'.format(state['weights_attnd'])]
                                      + ['train_psnr {:0.5f}'.format(train_psnr_meter.val)])
                                      
        if 'SSIMLoss' in config.loss_list:
            description += ' train_ssim_loss {:0.5f}'.format(train_ssim_loss)
            description += ' val_ssim_loss {:0.5f}'.format(val_ssim_loss)

        if 'PerceptLoss' in config.loss_list:
            description += ' train_percept_loss {:0.5f}'.format(train_percept_loss)
            description += ' val_percept_loss {:0.5f}'.format(val_percept_loss)
           
        if 'StyleLoss' in config.loss_list:
            description += ' train_style_loss {:0.5f}'.format(train_style_loss)
            description += ' val_style_loss {:0.5f}'.format(val_style_loss)
                                   
        pbar.set_description(description)
        pbar.update(1)


        # Plot(every epoch)
        path_save = config.save
        if config.save_test_plot:
            # Training loss
            train_loss_figure(np.array(Training_total_and_match_loss), path_save, save_name='total_and_match', Epoch=Epoch, epoch=epoch,
                        save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_total_loss), path_save, save_name='total', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_match_loss), path_save, save_name='match', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_match_loss_cd), path_save, save_name='match_cd', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_match_loss_sd), path_save, save_name='match_sd', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_match_loss_sd), path_save, save_name='match_attnd', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_ssim_loss), path_save, save_name='ssim', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_l1_loss), path_save, save_name='l1', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_mse_loss), path_save, save_name='mse', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_tv_loss), path_save, save_name='tv', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_percept_loss), path_save, save_name='percept', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_loss_figure(np.array(Training_style_loss), path_save, save_name='style', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            # Val loss
            val_loss_figure(np.array(Val_total_loss), path_save, save_name='total', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            val_loss_figure(np.array(Val_ssim_loss), path_save, save_name='ssim', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            val_loss_figure(np.array(Val_l1_loss), path_save, save_name='l1', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            val_loss_figure(np.array(Val_mse_loss), path_save, save_name='mse', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            val_loss_figure(np.array(Val_tv_loss), path_save, save_name='tv', Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            # Training SNR PSNR SSIM
            train_SNR_figre(np.array(Training_SNR), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_PSNR_figre(np.array(Training_PSNR), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            train_SSIM_figre(np.array(Training_SSIM), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            # Val SNR PSNR SSIM
            val_SNR_figre(np.array(Val_SNR), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            val_PSNR_figre(np.array(Val_PSNR), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            val_SSIM_figre(np.array(Val_SSIM), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)

            # Training & Val SNR PSNR SSIM
            SNR_figre(np.array(Training_SNR), np.array(Val_SNR), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            PSNR_figre(np.array(Training_PSNR), np.array(Val_PSNR), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False)
            SSIM_figre(np.array(Training_SSIM), np.array(Val_SSIM), path_save, None, Epoch=Epoch, epoch=epoch,
                            save_as_txt=True, show=False) 
    pbar.close()
    print("Training Finished!")

    # Test
    start_time = time.time()
    model_opt = load_trained_model(config.pt_path, model[0], device)
    test_ori_x, test_norm_data_x, patch_masks, test_outputs, imputed_data, test_l1_loss, test_mse_loss, testSNR, testpsnr, testSSIM, \
    min_ratio, max_ratio = test(config, model_opt, test_data_loader,config.test_loss_list, snr, psnr, ssim, state, device)
    end_time = time.time()
    hours, minutes, seconds = format_times_as_hms(end_time-start_time)
    print(f"test time: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    
    # Log metrics to wandb.
    if config.online_vis != 'not_use':
        run.log({"test_l1_loss":test_l1_loss, "test_mse_loss": test_mse_loss, "testSNR": testSNR, "testpsnr": testpsnr, "testSSIM": testSSIM})

    print("Test {} missing min_ratio {:0.3f}, {} missing max_ratio {:0.3f}".format(config.sampler, min_ratio, 
                                                                                      config.sampler,max_ratio))
    print("test l1 loss {:0.3e} test mse loss {:0.3e} test SNR {:0.5f} test SSIM {:0.5f} test PSNR {:0.5f}".format(test_l1_loss, test_mse_loss, 
                                                                                                                   testSNR, testSSIM, testpsnr))
    print("Test Finishing")

    
    if config.save_test_plot:
        # Plot
        current_time = "test-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path_save = os.path.join(config.save, current_time)
        print("test_path_save:", path_save)
        
        pbar = tqdm(total=len(imputed_data), ncols=100)
        for i in range(0, len(imputed_data)):

            path_image = os.path.join(path_save, "image/")
            makedirs(path_image)

            plot_seismic(imputed_data[i,0,:,:], path_image, save_name='imputed_data_', index=i, save_as_txt=config.save_as_txt)
            plot_seismic(test_ori_x[i,0,:,:], path_image, save_name='ori_data_', index=i, save_as_txt=config.save_as_txt) # test_ori_x[i,:,:]
            plot_mask(patch_masks[i,0,:,:], path_image, save_name='mask_', index=i, save_as_txt=config.save_as_txt)
            plot_seismic(test_outputs[i,0,:,:], path_image, save_name='test_outputs_', index=i, save_as_txt=config.save_as_txt)
            plot_seismic(test_norm_data_x[i,0,:,:], path_image, save_name='miss_data_', index=i, save_as_txt=config.save_as_txt)

            # np.savetxt(path_image+"test_outputs_"+str(i)+".txt",test_outputs[i,0,:,:])
            

            path_difference = os.path.join(path_save, "difference/")
            makedirs(path_difference)

            plot_difference(test_ori_x[i, 0, :, :] - imputed_data[i, 0, :, :], path_difference, i, save_as_txt=config.save_as_txt)
            
            pbar.set_description(' '.join(['SAVE: [{0}/{1}]  '.format(i, len(imputed_data))] ))
            pbar.update(1)

        print("Save Image Finishing")

    # Finish the run and upload any remaining data.
    if config.online_vis != 'not_use':
        run.finish()   
