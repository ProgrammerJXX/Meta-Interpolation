import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft2, fftshift, fftfreq
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr
import seaborn as sns
from typing import Dict, List
matplotlib.use("Agg")  # 用纯图像渲染的方式工作，不弹窗，只输出文件
# plt.rcParams['font.family'] = 'Times New Roman' #默认：DejaVu Sans
plt.rcParams['font.size'] = 12

def SNR_figre(train_SNR, val_SNR, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置"非数学文本"字体类型  默认：DejaVu Sans
        "font.size": 15,                    # 字体的大小为 15
        # "mathtext.fontset":'stix',        # 渲染"数学公式"时使用的字体集
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(train_SNR, label='Train SNR', color='red')
    plt.plot(val_SNR, label='Val SNR', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('SNR')
    plt.legend()

    # save image
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'SNR.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_SNR.txt'), np.array(train_SNR))
        np.savetxt(os.path.join(path_save, 'val_SNR.txt'), np.array(val_SNR))


def PSNR_figre(train_PSNR, val_PSNR, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        # "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(train_PSNR, label='Train PSNR', color='red')
    plt.plot(val_PSNR, label='Val PSNR', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'PSNR.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_PSNR.txt'), np.array(train_PSNR))
        np.savetxt(os.path.join(path_save, 'val_PSNR.txt'), np.array(val_PSNR))


def SSIM_figre(train_SSIM, val_SSIM, path_save, save_name, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        # "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(train_SSIM, label='Train SSIM', color='red')
    plt.plot(val_SSIM, label='Val SSIM', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'SSIM.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_SSIM.txt'), np.array(train_SSIM))
        np.savetxt(os.path.join(path_save, 'val_SSIM.txt'), np.array(val_SSIM))
def loss_figure(Generator_loss,Discriminator_loss):
    # 设置西文字体为新罗马字体

  config = {
        #"font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
    #     "mathtext.fontset":'stix',
    }
  rcParams.update(config)

  #损失变化可视化
  fig = plt.figure(figsize = (8,4),dpi=100)
  plt.figure(1)
  # plt.plot(D_loss,label='Discriminator loss',color='red')
  # plt.plot(G_loss,label='Generator loss',color='blue')
  plt.plot(Generator_loss,label='Generator loss',color='red')
  plt.plot(Discriminator_loss,label='Discriminator loss',color='blue')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def train_loss_figure(train_loss, path_save, save_name, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(1)
    plt.plot(train_loss, label='Train loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'training_loss_{}.jpeg').format(save_name))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_loss_{}.txt').format(save_name), np.array(train_loss))


def val_loss_figure(val_loss, path_save, save_name, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        # "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(1)
    plt.plot(val_loss, label='Val loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'val_loss_{}.jpeg').format(save_name))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'val_loss_{}.txt').format(save_name), np.array(val_loss))


def train_SNR_figre(train_SNR, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(train_SNR, label='Train SNR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('SNR')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'training_SNR.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_SNR.txt'), np.array(train_SNR))


def train_PSNR_figre(train_PSNR, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(train_PSNR, label='Train PSNR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'training_PSNR.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_PSNR.txt'), np.array(train_PSNR))


def train_SSIM_figre(train_SSIM, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(train_SSIM, label='Train SSIM', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'training_SSIM.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'training_SSIM.txt'), np.array(train_SSIM))


def val_SNR_figre(val_SNR, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(val_SNR, label='Val SNR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('SNR')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'val_SNR.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'val_SNR.txt'), np.array(val_SNR))


def val_PSNR_figre(val_PSNR, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(val_PSNR, label='Val PSNR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'val_PSNR.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'val_PSNR.txt'), np.array(val_PSNR))


def val_SSIM_figre(val_SSIM, path_save=None, save_name=None, Epoch=100, epoch=0, save_as_txt=True, show=False):

    # 设置西文字体为新罗马字体
    config = {
        # "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 15,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    # 损失变化可视化
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.figure(2)
    plt.plot(val_SSIM, label='Val SSIM', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    if (epoch == Epoch) and show:
        # plt.show()
        pass
    plt.savefig(os.path.join(path_save, 'val_SSIM.jpeg'))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, 'val_SSIM.txt'), np.array(val_SSIM))


def plot_seismic(trace, path_save, save_name, index, save_as_txt=True, save_type='jpeg',colobar=True):
    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',  # 背景为白，边框为黑
                            squeeze=False)
    axs = axs.ravel()
    # axs[0].imshow(trace, cmap=plt.cm.seismic)
    im = axs[0].imshow(trace, cmap=plt.cm.seismic, vmin=0, vmax = 1)

    if colobar:
        l = 0.92
        b = 0.12
        w = 0.015
        h = 1 - 2*b
        #对应 l,b,w,h；设置colorbar位置；
        rect = [l,b,w,h] # l, b, w, h 分别表示 colorbar 的左边距、底边距、宽度和高度
        cbar_ax = fig.add_axes(rect)
        plt.colorbar(im, ax=axs[0], cax=cbar_ax)

    plt.savefig(os.path.join(path_save, '{}'.format(save_name) + str(index) + ".{}".format(save_type))) # jpeg
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, '{}'.format(save_name) + str(index) + ".txt"), trace)



def plot_difference(dif, path_save, index, save_as_txt=True, save_type='jpeg'):
    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',
                            squeeze=False) # squeeze=False 保持 axs 为二维数组，即使只有一个子图，axs 的形状仍然是 (1, 1)
    axs = axs.ravel() # 将 axs 从二维数组展平为一维数组
    h3 = axs[0].imshow(dif, cmap='Greys', vmin=-1, vmax = 1)
    l = 0.92
    b = 0.12
    w = 0.015
    h = 1 - 2*b
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] # l, b, w, h 分别表示 colorbar 的左边距、底边距、宽度和高度
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h3, cax=cbar_ax)

    plt.savefig(os.path.join(path_save, "difference_" + str(index) + ".{}".format(save_type)))
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, "difference_" + str(index) + ".txt"), dif)

# 保存掩码矩阵到指定路径
def plot_mask(mask_array, path_save, save_name, index, save_as_txt=True, save_type='jpeg'):

    # 创建图像并保存
    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',  # 背景为白，边框为黑
                            squeeze=False)
    axs = axs.ravel()
    im = axs[0].imshow(mask_array, cmap='gray', vmin=0, vmax=1)

    plt.savefig(os.path.join(path_save, '{}'.format(save_name) + str(index) + ".{}".format(save_type))) # jpeg
    plt.close()

    # save txt
    if save_as_txt:
        np.savetxt(os.path.join(path_save, '{}'.format(save_name) + str(index) + ".txt"), mask_array)

# 展示带colorbar的图像；框选重要区域；在右上角放大子图
def read_txt2pdf(path_txt, path_save, dataset_name='', dt=None, save_name=None, save_type='pdf', add_rectangle=False, rect_box=[15, 45, 20, 30], magnify=False, zoom_scale=0.3, rect_color='black', time_crop=None, dt_not_use=False, arrows_list=None, clean=False, Mask=False):
    """_summary_

    Args:
        dt: 时间采样间隔 SEGC3 8ms/MAVO 4ms
        rect_box (list, optional): 框选区域 (x_start, y_start, width, height),默认None;如果为None,则不显示框选和放大
        zoom_scale (float, optional): 放大区域占图像的比例,默认0.3
    """
    # 基本配置
    if dataset_name == 'SEGC3':
        dt = 0.008   # 采样间隔（单位：秒，对应8ms）
    elif dataset_name == 'MAVO' or dataset_name == 'MAVG':
        dt = 0.004   # 采样间隔（单位：秒，对应4ms）
    else:
        print("dataset_name is {dataset_name}")


    """主图"""
    if time_crop:
        trace = np.loadtxt(path_txt)[time_crop[0]:time_crop[1],:]
    else:
        trace = np.loadtxt(path_txt)

    if clean != True:
        fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',  # 背景为白，边框为黑
                            squeeze=False)

    else:
        fig, axs = plt.subplots(nrows=1, ncols=1,  facecolor='none',squeeze=False)# 背景透明

    axs = axs.ravel()

    if Mask:
        im = axs[0].imshow(trace, cmap='gray', vmin=0, vmax = 1)  # 设置范围为0-1，colorbar也会自动生成0-1
    else:
        im = axs[0].imshow(trace, cmap=plt.cm.seismic, vmin=0, vmax = 1)

    if clean != True:
        # 坐标轴位置
        axs[0].xaxis.tick_top()
        axs[0].xaxis.set_label_position('top')
        axs[0].yaxis.tick_left()
        axs[0].yaxis.set_label_position('left')

        # 设置坐标轴范围
        y_range, x_range = trace.shape
        # print(y_range, x_range)
        num_ticks = 5  # 刻度数量
        if dt != None:
            T = dt * y_range
            positions = np.linspace(0, y_range, num_ticks)                  # 从0到总时间T生成等间距刻度位置
            labels = np.round(np.linspace(0, T, num_ticks), 3)        # 直接映射实际时间值，保留3位小数（毫秒级）
        if dt_not_use:
            positions = np.linspace(0, y_range, num_ticks)            # 根据图像高度定
            labels = np.round(np.linspace(0, 1, num_ticks), 3)        # 映射为 0 到 1 的刻度标签  0 到 1 之间生成 6 个等间距的数值,保留1位小数

        axs[0].set_xlim((0,x_range))
        axs[0].set_yticks(positions)
        axs[0].set_yticklabels(labels)
        axs[0].tick_params(axis='both', labelsize=12)  # 同时设置x/y轴刻度标签大小
        # axs[0].invert_yaxis() #翻转y轴

        # 设置标签和标题
        x_label= 'Trace number'
        axs[0].set_xlabel(x_label, fontsize=15)
        y_label = 'Time (s)'
        axs[0].set_ylabel(y_label, fontsize=15)
        title  = ''
        axs[0].set_title(title)
    else:
        axs[0].axis('off')  # 关闭坐标轴（刻度 + 边框）

    if save_name == None:
        save_name = os.path.splitext(os.path.basename(path_txt))[0]

    """添加红色箭头 """
    # xy 是箭头尖端 (x_data, y_data)，xytext 是箭头尾部 (x_data, y_data)
    if arrows_list is not None and (clean != True):
        for arrow_info in arrows_list:
            # xy 是箭头的尖端，xytext 是箭头的尾部
            axs[0].annotate(
                '',
                xy=arrow_info['xy'],
                xytext=arrow_info['xytext'],
                arrowprops=dict(
                    arrowstyle=arrow_info.get('arrowstyle', '->'), # 默认箭头样式
                    color=arrow_info.get('color', 'red'),          # 默认红色
                    linewidth=arrow_info.get('linewidth', 2),      # 默认线宽
                    shrinkA=0, shrinkB=0,                          # 箭头尾部和头部不缩短
                    mutation_scale=arrow_info.get('mutation_scale', 20), # 控制箭头头部大小
                )
            )

    """添加黑色矩形框（右上角区域"""
    # 定义矩形的位置（百分比或坐标）
    # 比如：右上角的区域（x从0.8到1，y从0.8到1）
    if add_rectangle and (clean != True):
        x_start, y_start, width, height = rect_box
        rect_linewidth=1 # default=2

        # 在主图上添加黑色长方形框
        rect_main = patches.Rectangle((x_start, y_start), width, height,
                                    linewidth=rect_linewidth, edgecolor=rect_color, facecolor='none')
        axs[0].add_patch(rect_main)

        """放大子图"""
        if magnify:
            # 获取主图在 figure 中的位置（归一化坐标）
            main_pos = axs[0].get_position()

            # 设置子图宽高（按 figure 坐标比例）
            zoom_width = zoom_scale
            zoom_height = zoom_scale

            # 计算子图位置（右上角对齐）
            zoom_x = main_pos.x1 - zoom_width + 0.043      # 保证右边框对齐     # -0.043  # 0.043  # -0.303 # -0.347 # -0.328
            zoom_y = main_pos.y1 - zoom_height + 0.        # 保证上边框对齐     # 0       # 0      # 0.01   # 0      # 0

            # 创建放大子图（使用 fig.add_axes 的 figure 坐标）
            zoom_ax = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])

            # print("Main ax right:", axs[0].get_position().x1)
            # print("Zoom ax right:", zoom_ax.get_position().x1)

            # 提取框选区域的数据
            y_end = min(y_start + height, trace.shape[0])
            x_end = min(x_start + width, trace.shape[1])
            zoomed_data = trace[int(y_start):int(y_end), int(x_start):int(x_end)]

            # 在放大子图中显示框选区域
            zoom_ax.imshow(zoomed_data, cmap=plt.cm.seismic, vmin=0, vmax=1)

            # 设置放大子图的边框为黑色
            for spine in zoom_ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(rect_linewidth)

            # 可选：移除放大子图的坐标轴刻度
            zoom_ax.set_xticks([])
            zoom_ax.set_yticks([])
    # colorbar
    l = 0.82 # 0.92
    b = 0.12
    w = 0.015
    h = 1 - 2*b
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] # l, b, w, h 分别表示 colorbar 的左边距、底边距、宽度和高度
    if clean != True:
        cbar_ax = fig.add_axes(rect)
        plt.colorbar(im, ax=axs[0], cax=cbar_ax) # 默认范围 = [data.min(), data.max()]

    plt.savefig(os.path.join(path_save, '{}'.format(save_name) + ".{}".format(save_type)), bbox_inches='tight', pad_inches=0.1) # jpeg
    plt.close()

# 展示多个方法的wiggle图
def plot_wiggle_multimethod(traces_list, method_names, dataset_name, path_save, save_name="wiggle_plot", save_type='pdf', scale=1.0, colors=None, trace_index=None):
    """
    traces_list: list of np.ndarray, 每个元素是一个二维数组 (n_samples, n_traces)
                 代表不同方法的地震数据
    method_names: list of str, 每个方法名称
    dt: float, 时间采样间隔，默认4ms
    scale: float, 振幅放大倍数

    示例：
        plot_wiggle([data_method1, data_method2], ['Method1', 'Method2'])
    """

    # 基本配置
    if dataset_name == 'SEGC3':
        dt = 0.008   # 采样间隔（单位：秒，对应8ms）
    elif dataset_name == 'MAVO' or dataset_name == 'MAVG':
        dt = 0.004   # 采样间隔（单位：秒，对应4ms）
    else:
        print("dataset_name is {dataset_name}")

    if trace_index !=None:

        traces_list = [trace[:,trace_index[0]:trace_index[1]] for trace in traces_list]

    n_methods = len(traces_list)
    n_samples, n_traces = traces_list[0].shape
    # print(n_samples, n_traces)

    if colors is None:
        # 默认颜色
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple'][:n_methods]

    fig, ax = plt.subplots(figsize=(n_traces * 0.2 + 2, 6))
    time = np.arange(n_samples) * dt

    # 坐标轴位置
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    # 所有方法全局最大值归一化
    global_max = max(np.max(np.abs(data)) for data in traces_list)

    # 遍历每个方法
    for i, (data, name, color) in enumerate(zip(traces_list, method_names, colors)):
        for j in range(n_traces):
            trace = data[:, j] / global_max
            x = j + scale * trace

            # 画 wiggle 曲线
            ax.plot(x, time, color=color, linewidth=0.7, label=name if j==0 else "")

            # 填充正振幅
            # ax.fill_betweenx(time, j, x, where=(trace > 0), facecolor=color, alpha=0.3)

    ax.invert_yaxis()
    ax.set_ylabel('Time (s)', fontsize=15)
    ax.set_xlabel('Trace Number', fontsize=15)
    # ax.set_xlim(-1, n_traces)
    # ax.set_xlim(-1, n_traces)
    ax.tick_params(axis='both', labelsize=12)  # 同时设置x/y轴刻度标签大小
    ax.grid(False)
    ax.set_yticks(np.arange(0, time[-1]+0.05, 0.2))
    if trace_index:
        # 裁剪后的数据索引范围
        start, end = trace_index
        n_traces_cropped = end - start + 1  # 例如35-25+1=11

        # 设置刻度位置为裁剪后的坐标（0到n_traces_cropped-1）
        xticks_pos = np.arange(0, n_traces_cropped, step=4)  # 位置: 0,1,2,...,10

        # 标签对应原始索引25到35
        xticks_labels = np.arange(start, end + 1, step=4)  # 标签: 25,26,...,35

        # 应用设置
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks_labels)
    else:
        ax.set_xticks(np.arange(0, n_traces+1, max(1, n_traces // 10)))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# 展示FK图
def compute_fk_spectrum(data, dt, dx=None, positive_freq_only=True, remove_dc=True):
    """
    计算地震数据的f-k光谱

    参数:
    data: numpy数组, 地震数据 (time x space)
    dt: float, 时间采样间隔
    dx: float or None, 空间采样间隔。如果为None，使用归一化波数
    positive_freq_only: bool, 是否只显示正频率

    返回:
    fk_spectrum: f-k光谱的幅度
    freq_axis: 频率轴
    knum_axis: 波数轴
    k_label: 波数轴标签
    """

    # 对数据进行2D FFT

    fk_data = fft2(data)
    fk_spectrum = np.abs(fftshift(fk_data))
    # fk_spectrum = np.abs(fk_data)  # 修改：不使用fftshift

    # 创建频率和波数轴
    nt, nx = data.shape

    # 频率轴
    freq_axis = fftshift(fftfreq(nt, dt))
    # freq_axis = fftfreq(nt, dt)  # 修改：不使用fftshift

    # 波数轴处理
    if dx is None:
        # 如果没有dx，使用归一化波数 (cycles per sample)
        knum_axis = fftshift(fftfreq(nx, 1.0))  # 归一化波数
        k_label = 'Normalized Wave Number (cycles/sample)'
    else:
        # 使用真实波数
        knum_axis = fftshift(fftfreq(nx, dx))
        # knum_axis = fftfreq(nx, dx)  # 修改：不使用fftshift，真实波数
        k_label = 'Wave Number (1/m)'

    # 如果只显示正频率
    if positive_freq_only:
          # 找到正频率的索引
          pos_freq_idx = freq_axis >= 0
          freq_axis = freq_axis[pos_freq_idx]
          fk_spectrum = fk_spectrum[pos_freq_idx, :]

    return fk_spectrum, freq_axis, knum_axis, k_label
def plot_fk_spectrum(path_save, data, dataset_name='', dt=0.001, dx=None, positive_freq_only=True,
                    output_filename='fk_spectrum', save_type='pdf',
                    enhancement_method='',add_rectangle=False, rect_box=[15, 45, 20, 30], vmin=0, vmax=2):
    """
    绘制并保存f-k光谱图 (改进版本)

    参数:
    path_save: str, 保存路径
    data: numpy数组, 地震数据
    dataset_name: str, 数据集名称
    dt: float, 时间采样间隔 (秒)
    dx: float or None, 空间采样间隔 (米)
    positive_freq_only: bool, 是否只显示正频率
    output_filename: str, 输出文件名
    save_type: str, 保存格式
    rect_box (list, optional): 框选区域 (x_start, y_start, width, height),默认None;如果为None,则不显示框选和放大
    """

    if dataset_name == 'SEGC3':
        dt = 0.008   # 采样间隔（单位：秒，对应8ms）
        dx = 20
    elif dataset_name == 'MAVO' or dataset_name == 'MAVG':
        dt = 0.004   # 采样间隔（单位：秒，对应4ms）
        dx = 25
    else:
        print(f"dataset_name is {dataset_name}")


    # 计算f-k光谱
    fk_spectrum, freq_axis, knum_axis, k_label = compute_fk_spectrum(
        data, dt, dx, positive_freq_only)

    # 不同的增强方法
    if enhancement_method == 'adaptive_log':
        # 自适应对数增强
        mean_val = np.mean(fk_spectrum)
        fk_spectrum_processed = np.log10(fk_spectrum + mean_val * 0.01)
        fk_spectrum_processed = (fk_spectrum_processed - fk_spectrum_processed.min()) / (fk_spectrum_processed.max() - fk_spectrum_processed.min())

    elif enhancement_method == 'power_law':
        # 幂律变换（伽马校正）
        normalized = fk_spectrum / fk_spectrum.max()
        gamma = 0.3  # 小于1增强暗部细节
        fk_spectrum_processed = np.power(normalized, gamma)

    else:  # 'percentile_only'
        # 仅百分位裁剪
        fk_spectrum_processed = np.log10(fk_spectrum + np.finfo(float).eps)
        # vmax = np.percentile(fk_spectrum_processed, 99)
        # vmin = np.percentile(fk_spectrum_processed, 0)
        # fk_spectrum_processed = np.clip(fk_spectrum_processed, vmin, vmax)


    # 创建图形
    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',  # 背景为白，边框为黑
                            squeeze=False)
    axs = axs.ravel()

    # 绘制f-k光谱
    im = axs[0].imshow(fk_spectrum_processed,
                    extent=[knum_axis.min(), knum_axis.max(),
                           freq_axis.max(), freq_axis.min()],
                    aspect='auto',
                    cmap='jet',  # 可以尝试 'hot', 'plasma', 'turbo'
                    interpolation='bilinear',
                    vmin=vmin, vmax=vmax  # 去掉极端值噪声
                    )

    """添加黑色矩形框（右上角区域"""
    # 定义矩形的位置（百分比或坐标）
    # 比如：右上角的区域（x从0.8到1，y从0.8到1）
    if add_rectangle:
        if not isinstance(rect_box[0], list):
            x_start, y_start, width, height = rect_box
            rect_linewidth=2 # default=2

            # 在主图上添加黑色长方形框
            rect_main = patches.Rectangle((x_start, y_start), width, height,
                                        linewidth=rect_linewidth, edgecolor='white', facecolor='none')

            # 获取当前 Axes 对象
            ax = plt.gca()  # 关键修改：通过 gca() 获取当前 Axes
            ax.add_patch(rect_main)
        else:
            x_start0, y_start0, width0, height0 = rect_box[0]
            rect_linewidth=2 # default=2

            # 在主图上添加黑色长方形框
            rect_main0 = patches.Rectangle((x_start0, y_start0), width0, height0,
                                        linewidth=rect_linewidth, edgecolor='white', facecolor='none', linestyle='-')

            # 获取当前 Axes 对象
            ax = plt.gca()  # 关键修改：通过 gca() 获取当前 Axes
            ax.add_patch(rect_main0)

            x_start1, y_start1, width1, height1 = rect_box[1]
            rect_linewidth=2 # default=2

            # 在主图上添加黑色长方形框
            rect_main1 = patches.Rectangle((x_start1, y_start1), width1, height1,
                                        linewidth=rect_linewidth, edgecolor='white', facecolor='none', linestyle='--')

            # 获取当前 Axes 对象
            ax = plt.gca()  # 关键修改：通过 gca() 获取当前 Axes
            ax.add_patch(rect_main1)

    # 设置坐标轴标签和位置
    plt.xlabel(k_label, fontsize=15)
    plt.ylabel('Frequency (Hz)', fontsize=15)
    # 设置坐标轴范围
    y_range, x_range = fk_spectrum_processed.shape
    # print(y_range, x_range)
    num_ticks = 5  # 刻度数量
    positions = np.linspace(knum_axis.min(), knum_axis.max(), num_ticks)                  # 从0到总时间T生成等间距刻度位置
    labels = np.round(np.linspace(knum_axis.min(), knum_axis.max(), num_ticks), 2)        # 直接映射实际时间值，保留3位小数（毫秒级）
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', labelsize=12)  # 同时设置x/y轴刻度标签大小

    # 将x轴移到顶部
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')

    # 添加colorbar
    cbar = plt.colorbar(im, shrink=0.8, pad=0.02)

    cbar.set_label('Amplitude', fontsize=12)

    # 添加网格
    # plt.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(path_save, f"{output_filename}.{save_type}"),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# 画残差图
def plot_residual_advanced(residual_data, dataset_name, trace_range=None, time_axis=None,
                          vmin=None, vmax=None, figsize=(12, 8),
                          cmap='RdBu_r', title='',colorbar=False, path_save='', save_name=None, save_type='pdf', dt_not_use=False):
    """
    高级残差图绘制函数，支持更多自定义选项

    额外参数:
    vmin, vmax: float,颜色映射的值范围
    """

    if dataset_name == 'SEGC3':
        dt = 0.008   # 采样间隔（单位：秒，对应8ms）
    elif dataset_name == 'MAVO' or dataset_name == 'MAVG':
        dt = 0.004   # 采样间隔（单位：秒，对应4ms）
    else:
        print("dataset_name is {dataset_name}")


    n_time, n_trace = residual_data.shape

    if trace_range is None:
        start_trace, end_trace = 0, n_trace
    else:
        start_trace, end_trace = trace_range
        start_trace = max(0, start_trace)
        end_trace = min(n_trace, end_trace)

    plot_data = residual_data[:, start_trace:end_trace]

    num_ticks = 5
    if time_axis is None:
        if dt != None:
            T = dt * n_time
            positions = np.linspace(0, n_time, num_ticks)                  # 从0到总时间T生成等间距刻度位置
            labels = np.round(np.linspace(0, T, num_ticks), 3)        # 直接映射实际时间值，保留3位小数（毫秒级）
        if dt_not_use:
            positions = np.linspace(0, n_time, num_ticks)            # 根据图像高度定
            labels = np.round(np.linspace(0, 1, num_ticks), 3)        # 映射为 0 到 1 的刻度标签  0 到 1 之间生成 6 个等间距的数值,保留1位小数
        time_axis = np.arange(n_time)

    trace_axis = np.arange(start_trace, end_trace)
    # print(trace_axis)
    # 自动设置颜色范围
    if vmin is None or vmax is None:
        abs_max = np.abs(plot_data).max()
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(plot_data,
                   cmap=cmap,
                   aspect='auto',
                   origin='upper',
                   vmin=vmin,
                   vmax=vmax,
                    extent=[trace_axis[0], trace_axis[-1],
                            time_axis[-1], time_axis[0]]
                   )

    ax.set_xlabel('Trace Number', fontsize=15)
    ax.set_ylabel('Time', fontsize=15)
    ax.set_xlim((start_trace, end_trace))
    ax.set_xticks([start_trace, end_trace])
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', labelsize=12)  # 同时设置x/y轴刻度标签大小
    if title:
        ax.set_title(title, fontsize=15, pad=20)

    # x轴在上方
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    if colorbar:
        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Residual Amplitude', rotation=270, labelpad=20)

    # 添加统计信息
    residual_stats = f"Max: { plot_data.max():.3f}\nMin: {plot_data.min():.3f}\nStd: {  plot_data.std():.3f}"
    ax.text(0.03, 0.99, residual_stats, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)

    plt.tight_layout()

    plt.savefig(os.path.join(path_save, '{}'.format(save_name) + ".{}".format(save_type)), bbox_inches='tight', pad_inches=0.1) # jpeg
    plt.close()


# 画多方法同一道wiggle图，局部放大展示
def create_wiggle_plot(path_save, dataset_name, data_dict, highlight_time_range=None,
                      figsize=(14, 10), line_width=1.5, alpha=0.8, save_name="wiggle_comparison", save_type='pdf'):
    """
    创建多方法对比的wiggle图,支持局部放大和背景阴影

    参数:
    data_dict: dict, 格式为 {'方法名': 数据数组}
    highlight_time_range: tuple, 需要高亮显示的时间范围 (start_time, end_time)
    output_filename: str, 输出文件名
    figsize: tuple, 图形尺寸
    line_width: float, 线条宽度
    alpha: float, 线条透明度
    """

    if dataset_name == 'SEGC3':
        dt = 0.008   # 采样间隔（单位：秒，对应8ms）
        dx = 20
        times = 128
        traces = 128
    elif dataset_name == 'MAVO' or dataset_name == 'MAVG':
        dt = 0.004   # 采样间隔（单位：秒，对应4ms）
        dx = 25
        times = 256
        traces = 112
    else:
        print(f"dataset_name is {dataset_name}")
    time_axis = np.arange(0, times * dt, dt)

    # 获取原始数据（第一个数据）
    method_names = list(data_dict.keys())
    original_data = data_dict[method_names[0]]

    # 计算皮尔森相关系数并创建新的标签
    updated_data_dict = {}
    for i, (method_name, data) in enumerate(data_dict.items()):
        if i == 0:
            # 第一个数据是原始数据，不计算相关系数
            updated_label = method_name
        else:
            # 计算与原始数据的皮尔森相关系数
            correlation, p_value = pearsonr(original_data.flatten(), data.flatten())
            updated_label = f"{method_name}, r={correlation:.3f}"
            # print(f"{method_name} 与原始数据的皮尔森相关系数: {correlation:.4f} (p-value: {p_value:.6f})")

        updated_data_dict[updated_label] = data
    data_dict = updated_data_dict

    # 设置颜色方案
    colors = ['black', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['--', '-', '-', '-', '-', '-', '-', '-.', ':']

    # 创建主图
    fig, ax = plt.subplots(figsize=figsize) # constrained_layout=True

    # 绘制每个方法的数据
    for i, (method_name, data) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        ax.plot(time_axis, data, color=color, linestyle=line_style, linewidth=line_width,
                       alpha=alpha, label=method_name)

    # 设置主图的基本属性
    ax.set_xlabel('Time (s)', fontsize=25)
    ax.set_ylabel('Amplitude', fontsize=25)
    ax.set_title('Multi-Method Wiggle Trace Comparison', fontsize=25)
    # ax.grid(True, alpha=0.3)

    # 在左上角添加图例
    legend = ax.legend(
        loc='upper left',      # 位置控制
        frameon=True,          # 边框显示
        fancybox=True,         # 边框圆角
        shadow=True,           # 阴影效果
        fontsize=15            # 字体大小
    )
    legend.get_frame().set_facecolor('white') # 设置图例的背景颜色'red'、'#FF0000'、'lightblue
    legend.get_frame().set_alpha(0.9)         # 设置图例背景的透明度

    # 调整布局  在填加子图之前，不然会报错
    plt.tight_layout()

    # 如果指定了高亮时间范围,添加阴影和放大图
    if highlight_time_range is not None:
        start_time, end_time = highlight_time_range

        # 验证时间范围的有效性
        if start_time >= end_time:
            raise ValueError("开始时间必须小于结束时间")
        if start_time < time_axis.min() or end_time > time_axis.max():
            print(f"警告: 指定的时间范围 [{start_time}, {end_time}] 超出数据范围 [{time_axis.min():.3f}, {time_axis.max():.3f}]")

        # 添加灰色阴影背景
        ax.axvspan(
            start_time,         # 起始位置（x轴）
            end_time,           # 结束位置（x轴）
            alpha=0.2,          # 透明度
            color='gray',       # 填充颜色
            zorder=0,           # 图层叠放顺序 置于底层，避免遮挡数据线、点等
            label='Highlighted Region'  # 图例标签
        )


        # 找到对应的时间索引
        start_idx = np.argmin(np.abs(time_axis - start_time))
        end_idx = np.argmin(np.abs(time_axis - end_time))

        # 确保索引范围有效
        start_idx = max(0, start_idx)
        end_idx = min(len(time_axis) - 1, end_idx)

        # 提取高亮区域的数据
        highlight_time = time_axis[start_idx:end_idx+1]

        # 创建放大图（右上角插图）
        # 计算插图位置和大小
        inset_left = 0.40    # 插图左侧起始位置（相对主图的比例）
        inset_bottom = 0.72  # 插图底部起始位置（相对主图的比例）
        inset_width = 0.3   # 插图宽度比例
        inset_height = 0.2   # 插图高度比例

        # 创建插图
        ax_inset = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])

        # 在插图中绘制高亮区域的放大视图
        for i, (method_name, data) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            highlight_data = data[start_idx:end_idx+1]
            ax_inset.plot(highlight_time, highlight_data, color=color, linestyle=line_style,
                         linewidth=line_width, alpha=alpha)

        # # 设置插图属性
        # ax_inset.set_xlabel('Time (s)', fontsize=10)
        # ax_inset.set_ylabel('Amplitude', fontsize=10)
        # ax_inset.set_title(f'Zoomed View ({start_time:.2f}s - {end_time:.2f}s)',
        #                   fontsize=10, fontweight='bold')
        # ax_inset.grid(True, alpha=0.3)

        # 设置插图的坐标轴范围
        ax_inset.set_xlim(start_time, end_time)

        # 计算插图的y轴范围
        highlight_amplitudes = []
        for data in data_dict.values():
            highlight_amplitudes.extend(data[start_idx:end_idx+1])

        if highlight_amplitudes:
            highlight_min = np.min(highlight_amplitudes)  # 计算高亮区域数据最小值
            highlight_max = np.max(highlight_amplitudes)  # 计算最大值
            highlight_range = highlight_max - highlight_min  # 计算范围
            margin = highlight_range * 0.1 if highlight_range > 0 else 0.1  # 设置边距为范围的10%（防止边界贴数据）
            ax_inset.set_ylim(highlight_min - margin, highlight_max + margin)  # 设置插图的Y轴显示范围（包含边距）

        # 添加连接线,连接主图的高亮区域和插图
        from matplotlib.patches import ConnectionPatch

        # 连接线1：主图左边界高亮区域的底部 -> 插图左下角
        con1 = ConnectionPatch(
            xyA=(start_time, highlight_min),  # 主图连接点（数据坐标系）
            coordsA='data', axesA=ax,         # 主图的坐标系和轴对象
            xyB=(0, 0), coordsB='axes fraction',  # 插图连接点（相对坐标系，左下角） 左下(0, 0) 右下(1, 0) 右上(1, 1) 左下(0, 1)
            axesB=ax_inset,                 # 插图的轴对象
            color='gray', linestyle='--',   # 虚线灰色
            alpha=0.6, linewidth=1          # 半透明、细线
        )

        # 连接线2：主图右边界高亮区域的底部 -> 插图右下角
        con2 = ConnectionPatch(
            xyA=(end_time, highlight_min),  # 主图连接点（数据坐标系）
            coordsA='data', axesA=ax,
            xyB=(1, 0), coordsB='axes fraction',  # 插图右下角
            axesB=ax_inset,
            color='gray', linestyle='--',
            alpha=0.6, linewidth=1
        )

        fig.add_artist(con1)  # 将连接线1添加到画布（非轴对象，需直接添加到Figure）
        fig.add_artist(con2)  # 添加连接线2


        # 在主图上添加矩形框标识高亮区域
        rect = Rectangle((start_time, highlight_min), end_time - start_time,
                        highlight_max - highlight_min, linewidth=2,
                        edgecolor='red', facecolor='none', alpha=0.7)
        ax.add_patch(rect)

    # 调整布局
    # plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plt.show()

    # return fig, ax

# PSNR 和参数展示图
def create_psnr_parameter_plot(path_save, methods_data, figsize=(12, 8), min_size=50, max_size=500, alpha=0.7,
                               save_name="psnr_parameter", save_type='pdf'):
    """
    创建方法参数量与PSNR关系的散点图

    参数:
    methods_data: list of dict, 每个字典包含方法信息
                 格式: [{'name': '方法名', 'parameters': 参数量, 'psnr': PSNR值}, ...]
    output_filename: str, 输出文件名
    figsize: tuple, 图形尺寸
    min_size: float, 最小圆形大小
    max_size: float, 最大圆形大小
    alpha: float, 圆形透明度
    """

    # 提取数据
    method_names = [method['name'] for method in methods_data]
    parameters = np.array([method['parameters'] for method in methods_data])
    psnr_values = np.array([method['psnr'] for method in methods_data])

    # 数据验证
    if len(methods_data) == 0:
        raise ValueError("输入数据不能为空")

    # 计算圆形大小：根据参数量进行线性缩放
    param_min, param_max = parameters.min(), parameters.max()
    if param_max == param_min:
        # 如果所有参数量相同，使用中等大小
        circle_sizes = np.full(len(parameters), (min_size + max_size) / 2) # 创建一个形状为 shape 的数组，并用指定的 value 填充所有元素
    else:
        # 线性缩放到指定大小范围
        circle_sizes = min_size + (max_size - min_size) * \
                      (parameters - param_min) / (param_max - param_min)

    # 定义颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
              '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2']

    # 如果方法数量超过预定义颜色数量，生成随机颜色
    if len(methods_data) > len(colors):
        import matplotlib.cm as cm
        colormap = cm.get_cmap('tab20')
        colors = [colormap(i / len(methods_data)) for i in range(len(methods_data))]

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点图
    scatter_points = []
    for i, (name, param, psnr, size) in enumerate(zip(method_names, parameters,
                                                     psnr_values, circle_sizes)):
        color = colors[i % len(colors)]
        scatter = ax.scatter(param, psnr, s=size, c=[color], alpha=alpha,
                           edgecolors='black', linewidth=1.5, zorder=3)
        scatter_points.append(scatter)

        # 在圆形旁边添加方法名标签（可选，避免图形过于拥挤）
        # ax.annotate(name, (param, psnr), xytext=(5, 5),
        #             textcoords='offset points', fontsize=8, alpha=0.8)
        ax.annotate(
                text=f"{psnr:.2f}",       # 标注内容（保留2位小数）
                xy=(param, psnr),             # 标注锚点（数据点坐标）
                xytext=(-35, 20),         # 文本相对于锚点的偏移（像素）
                textcoords="offset points",  # 偏移单位（像素）
                ha='center',           # 横向对齐方式
                fontsize=15,            # 字体大小
                color='black'          # 字体颜色
                )

    # 设置坐标轴标签和标题
    ax.set_xlabel('Number of Parameters (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Method Performance: Parameters vs PSNR', fontsize=16, fontweight='bold')

    # 添加网格
    ax.grid(True, alpha=0.3, zorder=1)

    # 设置坐标轴范围，留出适当边距
    param_range = param_max - param_min
    psnr_range = psnr_values.max() - psnr_values.min()

    ax.set_xlim(param_min - 0.1 * param_range, param_max + 0.1 * param_range)
    ax.set_ylim(psnr_values.min() - 0.1 * psnr_range,
                psnr_values.max() + 0.1 * psnr_range)

    # 在左下角创建图例
    legend_elements = []
    for i, (name, param, psnr) in enumerate(zip(method_names, parameters, psnr_values)):
        color = colors[i % len(colors)]
        # 创建图例元素，显示方法名和关键信息
        legend_label = f"{name}" #\n({param/1e6:.1f}M, {psnr:.1f}dB)"
        legend_elements.append(patches.Patch(color=color, label=legend_label))

    # 添加图例到右下角
    legend = ax.legend(
        handles=legend_elements,
        loc='lower right',          # 基准定位点（图例右下角与锚点对齐）
        bbox_to_anchor=(1, 0), # 锚点位置（相对于坐标轴的归一化坐标）
        frameon=True,               # 显示边框
        fancybox=True,              # 圆角边框
        shadow=True,                # 添加阴影
        fontsize=9,                 # 文字大小
        title='Methods (Params, PSNR)', # 图例标题
        title_fontsize=10           # 标题字体大小
        )


    # 设置图例样式
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')


    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight')

# PSNR-参数量（M）表现效率
def create_performance_efficiency_analysis(path_save, methods_data, figsize=(12, 8), min_size=50, max_size=500, save_name="psnr_parameter", save_type='pdf'):
    """
    创建性能效率分析图，包含帕累托前沿

    参数:
    methods_data: list, 方法数据
    save_name: str, 输出文件名
    """

    parameters = np.array([method['parameters'] for method in methods_data])
    psnr_values = np.array([method['psnr'] for method in methods_data])
    method_names = [method['name'] for method in methods_data]
    # 定义颜色方案
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4',  '#DDA0DD', '#F7DC6F',
              '#FF6B6B', '#BB8FCE', '#85C1E9',
              '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2']

    # 计算圆形大小：根据参数量进行线性缩放
    param_min, param_max = parameters.min(), parameters.max()
    if param_max == param_min:
        # 如果所有参数量相同，使用中等大小
        circle_sizes = np.full(len(parameters), (min_size + max_size) / 2) # 创建一个形状为 shape 的数组，并用指定的 value 填充所有元素
    else:
        # 线性缩放到指定大小范围
        circle_sizes = min_size + (max_size - min_size) * \
                      (parameters - param_min) / (param_max - param_min)

    fig, ax = plt.subplots(figsize=figsize)

    # 计算效率指标 (PSNR per million parameters)
    efficiencys = psnr_values / (parameters / 1e6)
    vmin = np.min(efficiencys)
    vmax = np.max(efficiencys)
    for i, (name, param, psnr, eff, size) in enumerate(zip(method_names, parameters,
                                                     psnr_values, efficiencys, circle_sizes)):
        # 创建散点图
        scatter = ax.scatter(
            param/1e6,           # x轴：参数量（百万）
            psnr,              # y轴：PSNR值
            c=eff,             # 颜色映射：根据效率值着色
            s=size,                    # 点的大小
            alpha=0.7,                # 透明度（0透明，1不透明）
            cmap='viridis',           # 颜色映射方案
            edgecolors='black',       # 点的边框颜色
            linewidth=1.5,            # 边框宽度
            vmin=vmin, vmax=vmax
        )

    # 添加方法名标签
    for i, (name, param, psnr) in enumerate(zip(method_names, parameters, psnr_values)):
        color = colors[i % len(colors)]
        ax.annotate(name, (param/1e6, psnr),
                   xytext=(0, 30), textcoords='offset points', ha='center',
                  fontsize=20, alpha=0.8,color=color)
        # ax.annotate(
        # text=f"{psnr:.2f}",       # 标注内容（保留2位小数）
        # xy=(param/1e6, psnr),             # 标注锚点（数据点坐标）
        # xytext=(0, -35),         # 文本相对于锚点的偏移（像素）
        # textcoords="offset points",  # 偏移单位（像素）
        # ha='center',           # 横向对齐方式
        # fontsize=15,            # 字体大小
        # color=color,          # 字体颜色
        # alpha=0.8
        # )

    # 在左下角创建图例
    legend_elements = []
    for i, (name, param, psnr) in enumerate(zip(method_names, parameters, psnr_values)):
        color = colors[i % len(colors)]
        # 创建图例元素，显示方法名和关键信息
        legend_label = f"{name}" #\n({param/1e6:.1f}M, {psnr:.1f}dB)"
        legend_elements.append(patches.Patch(color=color, label=legend_label))

    # 添加图例到右下角
    legend = ax.legend(
        handles=legend_elements,
        loc='upper right',          # 基准定位点（图例右下角与锚点对齐）
        bbox_to_anchor=(1, 1), # 锚点位置（相对于坐标轴的归一化坐标）
        frameon=True,               # 显示边框
        fancybox=True,              # 圆角边框
        shadow=True,                # 添加阴影
        fontsize=20,                 # 文字大小
        title='Methods (Params, PSNR)', # 图例标题
        title_fontsize=20           # 标题字体大小
        )


    # 设置图例样式
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')


    # 添加colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency (PSNR/M)', fontsize=25)

    ax.set_xlabel('Parameters (Millions)', fontsize=25)
    ax.set_ylabel('PSNR (dB)', fontsize=25)
    #ax.set_title('Method Efficiency Analysis', fontsize=25)
    ax.grid(True, alpha=0.3)

    # 设置x轴范围，留出适当边距
    x_min, x_max = parameters.min()/1e6,  parameters.max()/1e6
    x_range = x_max - x_min
    if x_range > 0:
        ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    # 设置y轴范围，留出适当边距
    y_min, y_max = psnr_values.min(),  psnr_values.max()
    y_range = y_max - y_min
    if y_range > 0:
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)

    plt.tight_layout()

    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight')


# 超参图PSNR/MSE/...
def create_training_curves_plot(path_save, epoch_data, loss_data_dict, metric_name='Validation MSE loss',
                               title='Hybrid loss weight', figsize=(8, 6), line_width=2, alpha=0.8,
                               save_name="training_curves", save_type='pdf'):
    """
    创建训练曲线图，显示不同超参数配置下的损失/指标随epoch变化

    参数:
    epoch_data: array, epoch数组
    loss_data_dict: dict, 格式为 {'参数配置名': 损失值数组}
    metric_name: str, 指标名称 (如 'Validation MSE loss', 'PSNR', 'Training Loss')
    title: str, 图表标题
    output_filename: str, 输出文件名
    figsize: tuple, 图形尺寸
    line_width: float, 线条宽度
    alpha: float, 线条透明度
    """

    # # 设置中文字体和样式
    # plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 定义颜色和线型
    colors = ['#008000','#d40d8c',  '#FFA500', '#00b3b0', '#0070b8', '#c2d0ea']
    line_styles = ['-', '-', '-', '-', '--', '-.', ':']

    # 绘制每条曲线
    for i, (config_name, loss_values) in enumerate(loss_data_dict.items()):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]

        line, = ax.plot(epoch_data, loss_values,
                       color=color,
                       linestyle=line_style,
                       linewidth=line_width,
                       alpha=alpha,
                       label=config_name,
                       marker='o' if len(epoch_data) <= 100 else None, # epoch_data 的数据点数量 <= 20，则会在折线图的每个数据点上显示圆形标记 'o'
                       markersize=4 if len(epoch_data) <= 100 else 0)  # 数据点 <= 20 时，标记的大小为 4

    # 设置坐标轴标签
    ax.set_xlabel('Epoch', fontsize=25)
    ax.set_ylabel(metric_name, fontsize=25)

    # 设置坐标轴位置
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')

    # 添加网格
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 设置x坐标轴范围
    ax.set_xlim(epoch_data.min(), epoch_data.max())

    # 设置y轴范围，留出适当边距
    all_values = np.concatenate([values for values in loss_data_dict.values()])
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    if y_range > 0:
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)


    # 添加图例到右上角
    legend = ax.legend(title=title, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=20, title_fontsize=20)
    legend.get_frame().set_facecolor('white')  # 图例背景
    legend.get_frame().set_alpha(0.9)

    # 设置坐标轴刻度
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight')

    # return fig, ax

# 创建推理时间(采样时间)柱状图
def create_inference_time_plot(path_save, methods, inference_times, sampling_counts=None,
                              figsize=(10, 6), bar_width=0.8, alpha=0.8,
                              save_name="inference_time_comparison", save_type='pdf', bar_name1='Flops (G)', bar_name2='Inference time (s)'):
    """
    创建方法推理时间对比柱状图

    参数:
    methods: list, 方法名称列表
    inference_times: list, 对应的推理时间(秒)
    sampling_counts: list or None, 采样数量(可选,用于双y轴显示)
    output_filename: str, 输出文件名
    figsize: tuple, 图形尺寸
    bar_width: float, 柱状图宽度
    alpha: float, 透明度
    """

    # # 设置字体
    # plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    # plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, ax1 = plt.subplots(figsize=figsize)

    # 设置x轴位置
    x_positions = np.arange(len(methods))

    # 定义颜色
    primary_color = '#87CEEB'  # 深绿色,类似参考图
    secondary_color = '#2E8B57'  # 浅蓝色

    # 如果有采样数量数据,创建双y轴图
    if sampling_counts is not None:
        # 创建第二个y轴
        ax2 = ax1.twinx()

        # 绘制采样数量柱状图(深色)
        bars1 = ax1.bar(x_positions - bar_width/4, sampling_counts,
                        width=bar_width/2, alpha=alpha,
                        color=primary_color, label=bar_name1,
                        edgecolor='black', linewidth=0.5)

        # 绘制推理时间柱状图(浅色)
        bars2 = ax2.bar(x_positions + bar_width/4, inference_times,
                        width=bar_width/2, alpha=alpha,
                        color=secondary_color, label=bar_name2,
                        edgecolor='black', linewidth=0.5)

        # 设置y轴标签
        ax1.set_ylabel(bar_name1, fontsize=35)
        ax2.set_ylabel(bar_name2, fontsize=35)

        # 设置对数刻度
        # ax1.set_yscale('log')

        # 在柱状图上方添加数值标注
        for i, (bar1, bar2, count, time) in enumerate(zip(bars1, bars2, sampling_counts, inference_times)):
            # 标注采样数量
            height1 = bar1.get_height()
            ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + max(sampling_counts) * 0.01,
                    int(round(count)), ha='center', va='bottom', fontsize=15)

            # 标注推理时间
            height2 = bar2.get_height()
            ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + max(inference_times) * 0.01,
                     int(round(time)), ha='center', va='bottom', fontsize=15)
                    #f'{time:.3f}', ha='center', va='bottom', fontsize=15)

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                  frameon=True, fancybox=True, shadow=True, fontsize=20, title_fontsize=20)

    else:
        # 单一指标的柱状图
        bars = ax1.bar(x_positions, inference_times, width=bar_width,
                      alpha=alpha, color=primary_color,
                      edgecolor='black', linewidth=0.5)

        # 设置y轴标签
        ax1.set_ylabel('Inference Time (s)', fontsize=20)

        # 在柱状图上方添加数值标注
        for bar, time in zip(bars, inference_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(inference_times) * 0.02,
                    f'{time:.3f}', ha='center', va='bottom', fontsize=10)

    # 设置x轴
    ax1.set_xlabel('Methods', fontsize=35)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(methods, rotation=0, ha='center', fontsize=20)

    # 添加网格
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 设置坐标轴范围
    ax1.set_xlim(-0.5, len(methods) - 0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight')


# 局部相似度图像
def plot_similarity_map(dataset_name, similarity_map, path_save, save_name, save_type='pdf', threshold=0.5, dt_not_use=False):
    """绘制带 colorbar 的局部相似度图。"""

    # 基本配置
    if dataset_name == 'SEGC3':
        dt = 0.008   # 采样间隔（单位：秒，对应8ms）
    elif dataset_name == 'MAVO' or dataset_name == 'MAVG':
        dt = 0.004   # 采样间隔（单位：秒，对应4ms）
    else:
        print("dataset_name is {dataset_name}")

    fig, axs = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k',  # 背景为白，边框为黑
                            squeeze=False)
    axs = axs.ravel()

    similarity_map[similarity_map < threshold] = 0 # 设为0会显示为最深的蓝色
    #similarity_map[similarity_map > 0.95] = 0 # 设为0会显示为最深的蓝色
    # 使用 'jet' 或其他颜色映射，interpolation='nearest' 让图像更清晰
    im = axs[0].imshow(similarity_map, cmap='jet',clim=(0,1))
    # 如果希望彩色图范围与您的示例图(e)(f)类似，可以设置 vmin=0, vmax=0.6 或 vmin=-1, vmax=1

    # 坐标轴位置
    axs[0].xaxis.tick_top()
    axs[0].xaxis.set_label_position('top')
    axs[0].yaxis.tick_left()
    axs[0].yaxis.set_label_position('left')

    # 设置坐标轴范围
    y_range, x_range = similarity_map.shape
    # print(y_range, x_range)
    num_ticks = 5  # 刻度数量
    if dt != None:
        T = dt * y_range
        positions = np.linspace(0, y_range, num_ticks)                  # 从0到总时间T生成等间距刻度位置
        labels = np.round(np.linspace(0, T, num_ticks), 3)        # 直接映射实际时间值，保留3位小数（毫秒级）
    if dt_not_use:
        positions = np.linspace(0, y_range, num_ticks)            # 根据图像高度定
        labels = np.round(np.linspace(0, 1, num_ticks), 3)        # 映射为 0 到 1 的刻度标签  0 到 1 之间生成 6 个等间距的数值,保留1位小数
    axs[0].set_xlim((0,x_range))
    axs[0].set_yticks(positions)
    axs[0].set_yticklabels(labels)
    axs[0].tick_params(axis='both', labelsize=12)  # 同时设置x/y轴刻度标签大小

    # 设置标签和标题
    x_label= 'Trace number'
    axs[0].set_xlabel(x_label, fontsize=15)
    y_label = 'Time (s)'
    axs[0].set_ylabel(y_label, fontsize=15)
    title  = ''
    axs[0].set_title(title)

    # 添加 colorbar
    l = 0.82 # 0.92
    b = 0.12
    w = 0.015
    h = 1 - 2*b
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] # l, b, w, h 分别表示 colorbar 的左边距、底边距、宽度和高度
    cbar_ax = fig.add_axes(rect)
    plt.colorbar(im, ax=axs[0], cax=cbar_ax) # 默认范围 = [data.min(), data.max()]
    #cbar.set_label('Local Similarity Value')

    #plt.show()
    # 保存图像
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight')

# 概率密度图
def plot_pixel_kde(data_dict: Dict, title, path_save, save_name, save_type='pdf'):
    """
    计算并绘制多个二维矩阵(单通道图)的像素值概率密度曲线(KDE)。

    参数:
    data_dict (Dict[str, np.ndarray]): 输入字典，键为方法名(str)，值为对应的二维矩阵(np.ndarray)。
    title (str): 图像标题。
    """

    # 检查输入字典是否为空
    if not data_dict:
        print("警告：输入字典 data_dict 为空，未进行绘图。")
        return

    # 1. 初始化绘图
    plt.figure(figsize=(10, 6))

    # 获取所有的像素值，用于统一 x 轴范围
    all_values = []
    for array in data_dict.values():
        all_values.append(array.flatten())

    # 展平所有数据，找到全局的最小值和最大值，用于设置 x 轴范围
    global_min = np.min([v.min() for v in all_values])
    global_max = np.max([v.max() for v in all_values])
    print("global_min:", global_min, "global_max:", global_max)

    mean_values = {} # 用于存储每个方法的均值
    # 2. 遍历字典，对每个矩阵进行 KDE 拟合和绘图
    for method_name, matrix in data_dict.items():
        # 将二维矩阵展平为一维数组，提取所有像素值
        pixel_values = matrix.flatten()

        # 计算当前方法的均值
        mean_values[method_name] = np.mean(pixel_values)

        # 使用 seaborn 的 kdeplot 进行核密度估计 (KDE) 拟合和绘图
        # kdeplot 会自动计算和拟合概率密度曲线
        pixel_values = np.array(matrix).flatten()
        sns.kdeplot(
            pixel_values,
            label=method_name,
            linewidth=1,
            # 设置填充颜色，可选
            # fill=True,
            # alpha=0.1
        )

    # 3. 设置图表样式
    plt.title(title, fontsize=16)
    plt.xlabel("Pixel Value (Intensity)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)

    # 添加图例，根据标签(方法名)区分曲线
    plt.legend(title="Method", fontsize=12)

    # 4. 将均值展示在左上角
    # 构建均值文本
    mean_text = "Mean Values:\n"
    for method_name, mean_val in mean_values.items():
        mean_text += f"- {method_name}: {mean_val:.2f}\n" # 保留两位小数

    # 将文本添加到图的左上角
    # bbox_to_anchor 用于定位文本框，loc='upper left' 表示文本框的左上角对齐指定位置
    # xycoords='axes fraction' 表示坐标是相对于 axes 的比例 (0,0) 是左下角, (1,1) 是右上角
    ax = plt.gca() # Get Current Axes
    ax.text(0.02, 0.2, mean_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # 设置 x 轴范围，稍微超出数据范围，让曲线完整显示
    # 使用 np.ptp (Peak-to-Peak) 得到范围，然后稍微扩展
    x_range = global_max - global_min
    x_min = global_min# - 0.05 * x_range
    x_max = global_max + 0.05 * x_range
    plt.xlim(x_min, x_max)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # 自动调整布局

    # 保存图像
    plt.savefig(os.path.join(path_save, f"{save_name}.{save_type}"), dpi=300, bbox_inches='tight')

if __name__ == '__main__':


    path_txt = r'G:/Distill_model/results/train_compare_models_SEGC3_multiple_0.1_0.3_AN_net_Unet_4_64_bn_avg_L1Loss_v2/2025-04-28-00-03-37/test-2025-04-30-14-40-43/image/test_outputs_8763.txt'
    path_save ='F:/GraduateStudent/3.导师/论文/TGRS/manu20220612/figures/c3/舍弃'

    """输出带colorbar的图"""
    save_name = "segc3_test"
    read_txt2pdf(path_txt, path_save, save_name=None, save_type='pdf', add_rectangle=True, rect_box=[50, 45, 20, 30], magnify=True, zoom_scale=0.2)

    # # wiggle
    # # 生成3个方法的合成数据，每个1000采样点，10道
    # np.random.seed(0)
    # n_samples = 1000
    # n_traces = 10
    # t = np.linspace(0, 1, n_samples)

    # data_method1 = np.array([np.sin(2*np.pi*10*t + j*0.5) * np.exp(-5*t) for j in range(n_traces)]).T
    # data_method2 = np.array([np.sin(2*np.pi*15*t + j*0.2) * np.exp(-5*t) for j in range(n_traces)]).T
    # data_method3 = np.array([np.sin(2*np.pi*25*t + j*0.1) * np.exp(-5*t) for j in range(n_traces)]).T

    # plot_wiggle_multimethod(
    #      [data_method1, data_method2, data_method1],
    #      ['Method 1', 'Method 2', 'Method 3'],
    #      path_save='',
    #      save_name="test_outputs_8763_wiggle",
    #      save_type="pdf",
    #      dt=0.004,
    #      scale=1.0,
    #      colors=["black", "red", "blue"]
    #  )


    # FK
    # dt = 0.008  # 时间采样间隔 (8ms)
    # dx = 20     # 空间采样间隔 (20m)

    # path_txt = r'G:/Distill_model/results/train_compare_models_SEGC3_multiple_0.1_0.3_AN_net_Unet_4_64_bn_avg_L1Loss_v2/2025-04-28-00-03-37/test-2025-04-30-14-40-43/image/test_outputs_8763.txt'

    # # 创建包含不同传播速度的波的合成数据
    # data = np.loadtxt(path_txt)
    # nt, nx = data.shape

    # print("数据形状:", data.shape)
    # print("数据类型:", type(data))

    # # 绘制并保存f-k光谱
    # plot_fk_spectrum(path_save, data, dt=dt, dx=20, positive_freq_only=True, output_filename='seismic_fk_spectrum')

    # # 加载真实数据示例
    # trace = np.loadtxt(path_txt)
    # time_axis = np.arange(0, 128 * 0.008, 0.008)
    # highlight_range = (0.7, 0.8)  # 可以修改这个范围
    # original_data = trace[:,2]
    # method1_data = trace[:,3]
    # method2_data = trace[:,1]
    # real_data_dict = {
    #     'Original': original_data,
    #     'Method1': method1_data,
    #     'Method2': method2_data,
    #     # ... 更多方法
    # }

    # create_wiggle_plot(
    #     path_save='',
    #     time_axis=time_axis,
    #     data_dict=real_data_dict,
    #     highlight_time_range=highlight_range,  # 根据需要调整
    #     save_name="multi_method_wiggle_comparison", save_type='pdf'
    # )


    # # PSNR-参数图
    # methods_data = [
    #     {'name': 'ResNet-18', 'parameters': 11.7e6, 'psnr': 28.5},
    #     {'name': 'DenseNet-121', 'parameters': 8.0e6, 'psnr': 29.1},
    #     {'name': 'Vision Transformer', 'parameters': 86.6e6, 'psnr': 32.8},
    #     {'name': 'Swin Transformer', 'parameters': 28.3e6, 'psnr': 31.9},

    # ]

    # create_psnr_parameter_plot(
    #     path_save='',
    #     methods_data=methods_data,
    #     figsize=(12, 8),
    #     min_size=500,    # 最小圆形大小
    #     max_size=1000,   # 最大圆形大小
    #     alpha=0.7       # 透明度
    # )

    # # 创建效率分析图
    # print("\n创建效率分析图...")
    # create_performance_efficiency_analysis('',methods_data, 'efficiency_analysis')

    ## 超参图
    # epoch_data = np.arange(0, 50)
    # num_configs=4

    # loss_data_dict = {}
    # configs = [
    #     {'name': 'λ₁ =2.0', 'initial_loss': 0.010, 'decay_rate': 0.08, 'final_loss': 0.003},
    #     {'name': 'λ₁ = 1.0', 'initial_loss': 0.009, 'decay_rate': 0.09, 'final_loss': 0.0025},
    #     {'name': 'λ₁ = 0.1', 'initial_loss': 0.008, 'decay_rate': 0.07, 'final_loss': 0.0035},
    #     {'name': 'λ₁ = 0.01', 'initial_loss': 0.0075, 'decay_rate': 0.06, 'final_loss': 0.004}]


    # for config in configs[:num_configs]:
    #         # 生成指数衰减的损失曲线
    #     initial_loss = config['initial_loss']
    #     decay_rate = config['decay_rate']
    #     final_loss = config['final_loss']

    #     # 指数衰减 + 噪声
    #     loss_curve = final_loss + (initial_loss - final_loss) * np.exp(-decay_rate * epoch_data / 10)

    #     # 添加一些随机波动
    #     noise = np.random.normal(0, 0.0002, len(epoch_data))
    #     loss_curve += noise
    #     loss_data_dict[config['name']] = loss_curve

    # psnr_data_dict = {}
    # for config_name, mse_values in loss_data_dict.items():
    #     # 将MSE转换为近似的PSNR值
    #     psnr_values = 20 * np.log10(1.0) - 10 * np.log10(mse_values + 1e-8)
    #     psnr_data_dict[config_name] = psnr_values
    # # Loss
    # create_training_curves_plot(
    #     path_save='',
    #     epoch_data=epoch_data,
    #     loss_data_dict=loss_data_dict,
    #     metric_name='Validation MSE loss',
    #     title='Hybrid loss weight',
    #     figsize=(8, 6)
    #     )

    # create_training_curves_plot(
    #     path_save='',
    #     epoch_data=epoch_data,
    #     loss_data_dict=psnr_data_dict,
    #     metric_name='PSNR (dB)',
    #     title='Method Performance',
    #     figsize=(8, 6),
    #     save_name='training_curves_psnr'
    # )


    # # 您的真实数据
    # methods = ['Method A', 'Method B', 'Method C']
    # inference_times = [15.2, 8.7, 2.1]  # 推理时间(秒)
    # sampling_counts = [800, 500, 50]   # 可选：采样数量

    # # 创建单一指标的推理时间图
    # print("\\n创建单一推理时间对比图...")
    # create_inference_time_plot(
    #     path_save='',
    #     methods=methods,
    #     inference_times=inference_times,
    #     figsize=(8, 6),
    #     save_name='inference_time_only'
    # )

    # # 创建带采样数量的组合图
    # print("\\n创建组合对比图...")
    # create_inference_time_plot(
    #     path_save='',
    #     methods=methods,
    #     inference_times=inference_times,
    #     sampling_counts=sampling_counts,
    #     figsize=(10, 6),
    #     save_name='combined_comparison'
    # )