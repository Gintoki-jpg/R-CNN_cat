import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.util import parse_cat_csv

class CustomClassifierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        samples = parse_cat_csv(root_dir)
        self.root_dir = root_dir
        self.transform = transform # 转换器
        self.jpeg_images, self.positive_list, self.negative_list = [], [], [] # 图片，正样本，负样本
        for idx, sample_name in enumerate(samples): # 遍历所有样本
            jpeg_image = cv2.imread(os.path.join(root_dir, 'JPEGImages', f"{sample_name}.jpg")) # 读取图片
            self.jpeg_images.append(jpeg_image) # 添加图片

            positive_annotation_path = os.path.join(root_dir, 'Annotations', f"{sample_name}_1.csv") # 正样本标注路径
            positive_annotations = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ') # 读取正样本标注
            if len(positive_annotations.shape) == 1 and positive_annotations.shape[0] == 4: # 如果正样本标注只有一个
                positive_dict = {'rect': positive_annotations, 'image_id': idx}
                self.positive_list.append(positive_dict) # 添加正样本
            elif len(positive_annotations.shape) > 1: # 如果正样本标注有多个
                for positive_annotation in positive_annotations: # 遍历所有正样本
                    positive_dict = {'rect': positive_annotation, 'image_id': idx}
                    self.positive_list.append(positive_dict) # 添加正样本

            negative_annotation_path = os.path.join(root_dir, 'Annotations', f"{sample_name}_0.csv") # 负样本标注路径
            negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.int, delimiter=' ') # 读取负样本标注
            if len(negative_annotations.shape) == 1 and negative_annotations.shape[0] == 4: # 如果负样本标注只有一个
                negative_dict = {'rect': negative_annotations, 'image_id': idx}
                self.negative_list.append(negative_dict) # 添加负样本
            elif len(negative_annotations.shape) > 1: # 如果负样本标注有多个
                for negative_annotation in negative_annotations: # 遍历所有负样本
                    negative_dict = {'rect': negative_annotation, 'image_id': idx}
                    self.negative_list.append(negative_dict) # 添加负样本

    def __getitem__(self, index):
        if index < len(self.positive_list): # 如果索引小于正样本数量
            target = 1 # 标签为1
            positive_dict = self.positive_list[index] # 获取正样本
            cache_dict = positive_dict # 缓存字典
        else:
            target = 0 # 标签为0
            idx = index - len(self.positive_list) # 获取负样本索引
            negative_dict = self.negative_list[idx] # 获取负样本
            cache_dict = negative_dict # 缓存字典

        image_id = cache_dict['image_id'] # 获取图片id
        xmin, ymin, xmax, ymax = cache_dict['rect'] # 获取标注框
        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax] # 获取图片
        if self.transform: # 数据增强操作
            image = self.transform(image)

        return image, target, cache_dict

    def __len__(self):
        return len(self.positive_list) + len(self.negative_list)

    def get_transform(self):
        return self.transform

    def get_jpeg_images(self):
        return self.jpeg_images

    def get_positive_num(self):
        return len(self.positive_list)

    def get_negative_num(self):
        return len(self.negative_list)

    def get_positives(self):
        return self.positive_list

    def get_negatives(self):
        return self.negative_list

    def set_negative_list(self, negative_list):
        self.negative_list = negative_list

def test(idx):
    root_dir = '../data/classifier_cat/val'
    train_data_set = CustomClassifierDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    image, target, cache_dict = train_data_set.__getitem__(idx)
    print('target: %d' % target)
    print('dict: ' + str(cache_dict))

    image = Image.fromarray(image)
    print(image)
    print(type(image))

if __name__ == '__main__':
    test(24768)