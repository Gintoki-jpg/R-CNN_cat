# 难例挖掘
import torch.nn as nn
from torch.utils.data import Dataset
import sys
sys.path.insert(0, 'utils')
from utils.custom_svm_dataset import CustomClassifierDataset

class CustomHardNegativeMiningDataset(Dataset): # 难例挖掘数据集类
    def __init__(self, negative_list, jpeg_images, transform=None):
        self.negative_list = negative_list # 负样本列表
        self.jpeg_images = jpeg_images # JPEG图像列表
        self.transform = transform # 图像变换操作

    def __len__(self):
        return len(self.negative_list)

    def __getitem__(self, index):
        negative_dict = self.negative_list[index] # 获取负样本字典
        xmin, ymin, xmax, ymax = negative_dict['rect'] # 获取负样本矩形框坐标
        image_id = negative_dict['image_id'] # 获取负样本图像ID

        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax] # 获取负样本图像

        if self.transform:
            image = self.transform(image)

        target = 0  # 负样本标签为0

        return image, target, negative_dict

def test():
    root_dir = '../data/classifier_cat/train'
    data_set = CustomClassifierDataset(root_dir)  # 读取训练数据集

    negative_list = data_set.get_negatives()  # 获取负样本列表
    jpeg_images = data_set.get_jpeg_images()  # 获取JPEG图像列表
    transform = data_set.get_transform()  # 获取图像变换操作

    hard_negative_dataset = CustomHardNegativeMiningDataset(negative_list, jpeg_images,
                                                            transform=transform)  # 创建难例挖掘数据集对象
    image, target, negative_dict = hard_negative_dataset.__getitem__(100)  # 获取第100个样本

    print(image.shape)  # 打印图像形状
    print(target)  # 打印标签
    print(negative_dict)  # 打印负样本字典

if __name__ == '__main__':
    test()