# 自定义微调数据集类
import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
sys.path.insert(0, 'utils')
from utils.util import parse_cat_csv

class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        samples = parse_cat_csv(root_dir) # 解析csv文件，获取所有样本的名称
        self.jpeg_images = [
            cv2.imread(os.path.join(root_dir, 'JPEGImages', f"{sample}.jpg")) # 读取所有样本的图像
            for sample in samples # 样本名称
        ]
        self.positive_annotations = [
            os.path.join(root_dir, 'Annotations', f"{sample}_1.csv") # 正样本的标注文件
            for sample in samples
        ]
        self.negative_annotations = [
            os.path.join(root_dir, 'Annotations', f"{sample}_0.csv") # 负样本的标注文件
            for sample in samples
        ]
        self.positive_sizes, self.positive_rects = self._parse_annotations(self.positive_annotations) # 解析正样本的标注文件
        self.negative_sizes, self.negative_rects = self._parse_annotations(self.negative_annotations) # 解析负样本的标注文件
        self.total_positive_num = int(np.sum(self.positive_sizes)) # 正样本的总数
        self.total_negative_num = int(np.sum(self.negative_sizes)) # 负样本的总数
        self.transform = transform # 数据增强

    def _parse_annotations(self, annotations):
        sizes, rects = [], [] # sizes: 样本中目标的数量，rects: 样本中目标的坐标
        for annotation_path in annotations: # 遍历所有样本的标注文件
            rects_array = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ') # 读取标注文件
            if len(rects_array.shape) == 1: # 如果样本中只有一个目标
                if rects_array.shape[0] == 4: # 如果目标的坐标是4个
                    rects.append(rects_array) # 添加目标的坐标
                    sizes.append(1) # 目标的数量为1
                else: # 如果目标的坐标不是4个
                    sizes.append(0) # 目标的数量为0
            else: # 如果样本中有多个目标
                rects.extend(rects_array) # 添加目标的坐标
                sizes.append(len(rects_array)) # 目标的数量为目标的个数
        return sizes, rects # 返回样本中目标的数量和坐标

    def __getitem__(self, index):
        if index < self.total_positive_num: # 如果索引小于正样本的总数
            # 正样本
            target = 1
            rect_index, image_id = self._get_rect_index_and_image_id(index, self.positive_sizes) # 获取正样本的坐标索引和图像索引
            xmin, ymin, xmax, ymax = self.positive_rects[rect_index] # 获取正样本的坐标
        else:
            # 负样本
            target = 0
            idx = index - self.total_positive_num # 获取负样本的索引
            rect_index, image_id = self._get_rect_index_and_image_id(idx, self.negative_sizes) # 获取负样本的坐标索引和图像索引
            xmin, ymin, xmax, ymax = self.negative_rects[rect_index] # 获取负样本的坐标

        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax] # 获取样本的图像
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self):
        return self.total_positive_num

    def get_negative_num(self):
        return self.total_negative_num

    def _get_rect_index_and_image_id(self, index, sizes):
        image_id = 0
        for i, size in enumerate(sizes): # 遍历所有样本
            if np.sum(sizes[:i]) <= index < np.sum(sizes[:i + 1]): # 如果索引在当前样本中
                rect_index = index - np.sum(sizes[:i]) # 获取坐标索引
                image_id = i # 获取图像索引
                break
        return rect_index, image_id

def test(idx):
    root_dir = '../data/finetune_cat/train'
    train_data_set = CustomFinetuneDataset(root_dir) # 创建数据集

    print('positive num: %d' % train_data_set.get_positive_num(1)) # 获取正样本的数量
    print('negative num: %d' % train_data_set.get_negative_num()) # 获取负样本的数量
    print('total num: %d' % train_data_set.__len__()) # 获取样本的总数


    image, target = train_data_set.__getitem__(idx) # 获取第idx个样本
    print('target: %d' % target) # 输出标签

    image = Image.fromarray(image) # 将numpy数组转换为PIL图像
    print(image) # 输出图像
    print(type(image)) # 输出图像的类型

if __name__ == '__main__':
    test(24768)