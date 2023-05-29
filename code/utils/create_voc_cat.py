# 从VOC2007数据集中提取类别Cat的样本图片和标注文件

import os
import shutil
import random
import numpy as np
import xmltodict
from util import check_dir

suffix_xml = '.xml' # 标注文件后缀
suffix_jpeg = '.jpg' # 图片文件后缀
cat_train_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/cat_train.txt' # 训练集
cat_val_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/cat_val.txt' # 验证集
voc_annotation_dir = '../data/VOCdevkit/VOC2007/Annotations/' # 标注文件目录
voc_jpeg_dir = '../data/VOCdevkit/VOC2007/JPEGImages/' # 图片文件目录
cat_root_dir = '../data/voc_cat/' # 保存类别Cat的样本图片和标注文件的目录

def parse_train_val(data_path): # 解析训练集或验证集
    """
    提取指定类别图像
    """
    samples = []
    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split(' ')
            if len(res) == 3 and int(res[2]) == 1: # 仅提取类别Cat的图像
                samples.append(res[0])
    return np.array(samples) # 返回样本列表

def sample_train_val(samples):
    """
    随机采样样本，减少数据集个数（留下1/10）
    """
    for name in ['train', 'val']:
        dataset = samples[name] # 获取训练集或验证集
        length = len(dataset)
        random_samples = random.sample(range(length), int(length / 10)) # 随机采样样本
        # print(random_samples)
        new_dataset = dataset[random_samples] # 获取随机采样样本
        samples[name] = new_dataset  # 更新训练集或验证集

    return samples

def save_cat(cat_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    """
    保存类别Cat的样本图片和标注文件
    """
    for sample_name in cat_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml) # 源标注文件路径
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml) # 目标标注文件路径
        shutil.copyfile(src_annotation_path, dst_annotation_path) # 复制标注文件

        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg) # 源图片文件路径
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg) # 目标图片文件路径
        shutil.copyfile(src_jpeg_path, dst_jpeg_path) # 复制图片文件

    csv_path = os.path.join(data_root_dir, 'cat.csv') # 保存类别Cat的样本列表
    np.savetxt(csv_path, np.array(cat_samples), fmt='%s')


if __name__ == '__main__':
    samples = {'train': parse_train_val(cat_train_path), 'val': parse_train_val(cat_val_path)}
    print(samples)

    check_dir(cat_root_dir) # 检查目录是否存在，若不存在则创建
    for name in ['train', 'val']:
        data_root_dir = os.path.join(cat_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations') # 保存类别Cat的样本标注文件的目录
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages') # 保存类别Cat的样本图片文件的目录

        check_dir(data_root_dir)
        check_dir(data_annotation_dir)
        check_dir(data_jpeg_dir)
        save_cat(samples[name], data_root_dir, data_annotation_dir, data_jpeg_dir) # 保存类别Cat的样本图片和标注文件

    print('done')
