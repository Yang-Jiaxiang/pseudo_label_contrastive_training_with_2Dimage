import sys
import os
import numpy as np
from PIL import Image

def split_dataset_to_lists(base_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # 確認比例總和為1
    assert train_ratio + valid_ratio + test_ratio == 1
    
    # 獲取所有資料夾名稱
    all_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    np.random.shuffle(all_folders)
    
    # 計算分割點
    total_folders = len(all_folders)
    train_end = int(total_folders * train_ratio)
    valid_end = train_end + int(total_folders * valid_ratio)
    
    # 分配資料夾名稱到各個列表
    train_folders = all_folders[:train_end]
    valid_folders = all_folders[train_end:valid_end]
    test_folders = all_folders[valid_end:]
    
    return train_folders, valid_folders, test_folders

def data_list(train_path):
    """ 列出目錄中的所有檔案名稱 """
    # 獲取目錄中的所有條目
    files = os.listdir(train_path)
    # 過濾出檔案，排除資料夾
    files = [os.path.splitext(file)[0] for file in files if os.path.isfile(os.path.join(train_path, file))]
    return files

def data_imgs(train_path, target_size=(256, 256)):
    """列出目录中的所有图片，并将其统一尺寸后加载为单通道（灰度）numpy数组"""
    files = os.listdir(train_path)  # 获取目录下的所有文件和文件夹
    images = []
    for path in files:
        if os.path.isfile(os.path.join(train_path, path)):  # 确保路径是文件
            image_path = os.path.join(train_path, path)
            image = Image.open(image_path).convert('L')  # 转换为灰度图像
            image = image.resize(target_size, Image.ANTIALIAS)  # 改变图像尺寸
            images.append(np.array(image))
    return np.stack(images)  # 此时所有图像的尺寸都已统一，并且是单通道，可以安全地使用 np.stack


def train_data(train_path):
    # 假設你的資料夾路徑是 'Train/img'
    file_names = list_files(train_path)
    return file_names

def val_data(val_path):
    #print('val set list')
    file_names = list_files(val_path)
    return file_names

def test_data(test_path):
    #print('test set list')
    file_names = list_files(test_path)
    return file_names