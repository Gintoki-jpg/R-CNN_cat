# 自定义批量采样器
import numpy as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
sys.path.insert(0, 'utils')
from utils.custom_alex_dataset import CustomFinetuneDataset
import torch

class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative):
        self.num_positive = num_positive # 正样本数量
        self.num_negative = num_negative # 负样本数量
        self.batch_positive = batch_positive # 正样本批量大小
        self.batch_negative = batch_negative # 负样本批量大小

        self.positive_idx = torch.randperm(num_positive).tolist() # 正样本索引
        self.negative_idx = torch.randperm(num_negative).tolist() # 负样本索引

        self.num_iter = (num_positive // batch_positive) + (num_negative // batch_negative) # 迭代次数
        self.batch = batch_positive + batch_negative # 批量大小

    def __iter__(self):
        sampler_list = [] # 采样器列表
        for i in range(self.num_iter):
            start_pos = i * self.batch_positive # 起始位置
            end_pos = (i + 1) * self.batch_positive # 结束位置
            positive_batch = self.positive_idx[start_pos:end_pos] # 正样本批量
            negative_batch = random.sample(self.negative_idx, self.batch_negative) # 负样本批量
            batch = positive_batch + negative_batch
            random.shuffle(batch) # 打乱
            sampler_list.extend(batch) # 添加到采样器列表
        return iter(sampler_list)

    def __len__(self):
        return self.num_iter * self.batch

    def get_num_batch(self):
        return self.num_iter

def test():
    root_dir = '../data/finetune_cat/train' # 数据集根目录
    train_data_set = CustomFinetuneDataset(root_dir) # 数据集
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96) # 自定义批量采样器

    print('sampler len: %d' % train_sampler.__len__()) # 批量采样器长度
    print('sampler batch num: %d' % train_sampler.get_num_batch()) # 批量采样器批量数量

    first_idx_list = list(train_sampler.__iter__())[:128] # 批量采样器迭代器
    # print('first_idx_list：',first_idx_list) # 打印批量采样器迭代器
    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517)) # 单次批量中正样本个数

def test2():
    root_dir = '../data/finetune_cat/train'
    transform = transforms.Compose([
        transforms.ToPILImage(), # 转换为PIL图像
        transforms.Resize((227, 227)), # 调整图像大小
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 标准化
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform) # 数据集
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96) # 自定义批量采样器
    data_loader = DataLoader(train_data_set, batch_size=128, sampler=train_sampler, num_workers=8, drop_last=True) # 数据加载器

    inputs, targets = next(data_loader.__iter__())  # 获取数据
    print(targets) # 打印标签
    print(inputs.shape) # 打印输入张量形状

if __name__ == '__main__':
    # test()
    test2()