import os
import  re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
import cv2
from matplotlib.colors import LinearSegmentedColormap

def feature_visualization(path_save, name, features, image_mask=None, original_image=None, channel_index=0, alpha=0.5, save_type="1", colormap=cv2.COLORMAP_WINTER, type='jpg'):

    features = features.numpy()
    if original_image is not None:
        original_image = original_image.numpy()
    if image_mask is not None:
        image_mask = image_mask.numpy()
    # features = features.squeeze(0)

    # 获取单个通道的特征图
    channel = features[channel_index]
    #print(channel.min(), channel.max())
    # plt.imshow(data, cmap=plt.cm.seismic, vmin=0, vmax = 1)
    plt.imshow(channel, cmap=plt.cm.seismic)
    plt.axis('off')                      # 隐藏坐标轴和边框
    plt.savefig(os.path.join(path_save, '{}_seismic.{}'.format(name, type)), bbox_inches='tight', pad_inches=0) # bbox_inches='tight'：自动裁剪空白边缘 pad_inches=0：边界填充设为0英寸
    plt.close()
    # 归一化到0-1范围
    channel_data = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)

    if save_type == "1" or 'all':
        # 调整特征图尺寸)
        _, orig_w, orig_h = original_image.shape
        if orig_w==orig_h:
            data = cv2.resize(channel_data, (orig_w, orig_h))
        else:
            data = channel_data[:orig_h, :]
        # 方案1：直接保存灰度图
        gray_image = (data * 255).astype(np.uint8)
        Image.fromarray(gray_image).save(os.path.join(path_save, '{}_gray.{}'.format(name, type)))


    if save_type == "2" or 'all':
        # 方案2：伪彩色增强 (使用matplotlib colormap)
        plt.figure(figsize=(10,10))
        _, orig_w, orig_h = original_image.shape
        if orig_w==orig_h:
            color_image = (channel_data * 255).astype(np.uint8)
        else:
            color_image = (channel_data[:orig_h, :] * 255).astype(np.uint8)

        plt.imshow(color_image, vmin=0, vmax = 255) # cmap='jet'：使用彩虹色系（蓝-青-黄-红） gnuplot:深蓝 → 青 → 绿 → 黄
        plt.axis('off')                      # 隐藏坐标轴和边框
        plt.savefig(os.path.join(path_save, '{}_color.{}'.format(name, type)), bbox_inches='tight', pad_inches=0) # bbox_inches='tight'：自动裁剪空白边缘 pad_inches=0：边界填充设为0英寸
        plt.close()

    if save_type == "3" or 'all':
        # 方案2：伪彩色增强 (使用matplotlib colormap)
        plt.figure(figsize=(10,10))
        _, orig_w, orig_h = original_image.shape
        if orig_w==orig_h:
            color_image_mask = (image_mask * 255).squeeze(0)
        else:
            color_image_mask = (image_mask[:,:orig_h, :] * 255).squeeze(0)
        plt.imshow(color_image_mask, cmap=plt.cm.seismic, vmin=0, vmax = 255)
        # plt.imshow(color_image_mask, cmap='gnuplot') # cmap='jet'：使用彩虹色系（蓝-青-黄-红） gnuplot:深蓝 → 青 → 绿 → 黄
        plt.axis('off')                      # 隐藏坐标轴和边框
        plt.savefig(os.path.join(path_save, '{}_image_mask.{}'.format(name, type)), bbox_inches='tight', pad_inches=0) # bbox_inches='tight'：自动裁剪空白边缘 pad_inches=0：边界填充设为0英寸
        plt.close()

        plt.imshow(original_image[:,:orig_h, :].transpose(1, 2, 0), cmap=plt.cm.seismic, vmin=0, vmax = 1)
        # plt.imshow(color_image_mask, cmap='gnuplot') # cmap='jet'：使用彩虹色系（蓝-青-黄-红） gnuplot:深蓝 → 青 → 绿 → 黄
        plt.axis('off')                      # 隐藏坐标轴和边框
        plt.savefig(os.path.join(path_save, '{}_ori.{}'.format(name, type)), bbox_inches='tight', pad_inches=0) # bbox_inches='tight'：自动裁剪空白边缘 pad_inches=0：边界填充设为0英寸
        plt.close()

    if save_type == "4" or 'all':
        # 方案3：叠加在原图上 (需要尺寸匹配)
        if original_image is not None:
            # 调整特征图尺寸)
            c, orig_w, orig_h = original_image.shape
            channel_data = (channel_data * 255).astype(np.uint8)
            if orig_w==orig_h:
                heatmap = cv2.applyColorMap(cv2.resize(channel_data, (orig_w, orig_h)), colormap) # 对缩放后的数据应用JET颜色映射，将单通道数据转换为3通道的伪彩色BGR图像（蓝色表示低值，红色表示高值）
            else:
                # print(original_image.shape)
                heatmap = cv2.applyColorMap(cv2.resize(channel_data, (orig_h, orig_h)), colormap)
                original_image = original_image[:,:orig_h, :]
                # print(heatmap.shape, original_image.shape)
            # 转换为PIL格式并混合
            original_image = (original_image * 255)
            # print(original_image.shape, heatmap.shape)
            overlay = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)) # 将OpenCV的BGR格式热图转换为PIL兼容的RGB格式112, 256, 3->3, 256, 112
            original_image = Image.fromarray(original_image.squeeze(0))
            blended = Image.blend(original_image.convert('RGBA'),
                                overlay.convert('RGBA'), alpha=alpha)
            # print(np.array(blended).shape)
            blended = blended.convert('RGB')
            # print(np.array(blended).shape)

            blended.save(os.path.join(path_save, '{}_overlay.{}'.format(name, 'png'))) # 将原始&热图像转换为RGBA模式（添加透明度通道） 以50%的透明度混合原始图像和热图，生成叠加效果
            # plt.figure(figsize=(10,10))
            # plt.imshow(blended) # cmap='jet'
            # plt.axis('off')
            # plt.savefig(
            #     os.path.join(path_save, '{}_overlay.pdf'.format(name)),
            #     bbox_inches='tight',
            #     pad_inches=0,
            #     dpi=300
            # )
            # plt.close()  # 防止内存泄漏

def attention_visualization(path_save, name, attention, original_image, alpha=0.5, upscale_method=cv2.INTER_CUBIC, colormap=cv2.COLORMAP_WINTER):

    attention = attention.numpy()
    if original_image is not None:
        original_image = original_image.cpu().detach().numpy()

    # 上采样到原图尺寸
    attn_map = cv2.resize(attention, original_image.size, interpolation=upscale_method)

    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), colormap)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

    # 叠加显示
    blended = Image.blend(original_image.convert('RGBA'),
                         heatmap.convert('RGBA'),
                         alpha=alpha)

    blended.save(os.path.join(path_save, '{}_attention.png'.format(name)))


# 改文件名字
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
    pattern = re.compile(r'^imputed_data2(\d+)\.(jpeg|txt)$', re.IGNORECASE)

    # 遍历目录中的所有文件
    for filename in os.listdir(path):
        # 匹配文件名模式
        match = pattern.match(filename)
        if match:
            # 提取数字部分和扩展名
            digits = match.group(1)
            ext = match.group(2).lower()  # 统一转为小写

            # 构建新文件名
            new_name = f"imputed_data2_{digits}.{ext}"

            # 获取完整文件路径
            old_path = os.path.join(path, filename)
            new_path = os.path.join(path, new_name)

            # 执行重命名操作
            try:
                os.rename(old_path, new_path)
                print(f"成功重命名：{filename} -> {new_name}")
            except Exception as e:
                print(f"重命名 {filename} 失败：{str(e)}")