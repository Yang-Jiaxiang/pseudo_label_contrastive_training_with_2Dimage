# script to crop the images into the target resolution and save them.
import numpy as np
import pathlib

import nibabel as nib

import os.path
from os import path

from PIL import Image

import sys
sys.path.append("<path_to_git_code>/git_code")

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md', 'BTS'])

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

import experiment_init.init_BTS as cfg
from experiment_init.data_cfg_BTS import data_list


######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)


for file_name in data_list(cfg.data_path_tr + "/img"):
    file_path = os.path.join(cfg.data_path_tr + "/img", file_name + '.png')
    mask_path = os.path.join(cfg.data_path_tr + "/label", file_name + '.png')

    print(file_path)
    print(mask_path)
    # 检查图像文件是否存在
    if os.path.exists(file_path):
        print('Processing', file_path)
    else:
        print('Skipping', file_name)
        continue
    
    # 加载图像
    img_sys = Image.open(file_path)
    img_sys = np.array(img_sys)  # 如果需要，转换为numpy数组

    # 检查掩码是否存在并加载
    if os.path.exists(mask_path):
        label_sys = Image.open(mask_path)
        label_sys = np.array(label_sys)  # 如果需要，转换为numpy数组
    else:
        label_sys = np.zeros_like(img_sys)  # 创建一个全零的虚拟掩码

    # 预处理图像和掩码（根据需要修改此部分）
    cropped_img_sys = img_sys  # 这里添加裁剪或其他预处理代码
    cropped_mask_sys = label_sys  # 同上

    # 输出目录保存裁剪的图像和掩码
    save_dir_tmp = os.path.join(cfg.data_path_tr_cropped, file_name)
    print(save_dir_tmp)
    pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)

    # 保存裁剪的图像
    pred_filename = os.path.join(save_dir_tmp, 'img_cropped.png')
    Image.fromarray(cropped_img_sys).save(pred_filename)

    # 如果存在掩码，则保存裁剪的掩码
    if os.path.exists(mask_path):
        pred_filename = os.path.join(save_dir_tmp, 'mask_cropped.png')
        Image.fromarray(cropped_mask_sys).save(pred_filename)
