# 分类器训练，R-CNN在完成卷积模型的微调后，额外使用了线性SVM分类器，采用负样本挖掘方法进行训练
import time
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet
import sys
sys.path.insert(0, 'utils')
from utils.custom_svm_dataset import CustomClassifierDataset
from utils.custom_hnm_dataset import CustomHardNegativeMiningDataset
from utils.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir
from utils.util import save_model

batch_positive = 32 
batch_negative = 96
batch_total = 128

def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    remain_negative_list = []
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        data_set = CustomClassifierDataset(data_dir, transform=transform)

        if name == 'train':
            # Use hard negative mining
            pos_list = data_set.get_positives()
            neg_list = data_set.get_negatives()
            init_neg_idxs = random.sample(range(len(neg_list)), len(pos_list))
            init_neg_list = [neg_list[idx] for idx in init_neg_idxs]
            remain_negative_list = [neg_list[idx] for idx in range(len(neg_list)) if idx not in init_neg_idxs]
            data_set.set_negative_list(init_neg_list)
            data_loaders['remain'] = remain_negative_list

        sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                     batch_positive, batch_negative)

        data_loader = DataLoader(data_set, batch_size=batch_total, sampler=sampler, num_workers=8, drop_last=True)
        data_loaders[name] = data_loader
        data_sizes[name] = len(sampler)

    return data_loaders, data_sizes

def hinge_loss(outputs, labels, margin=1.0):
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T
    margins = outputs - corrects + margin
    loss = torch.mean(torch.max(margins, dim=1)[0])
    return loss

def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))

def get_hard_negatives(preds, cache_dicts):
    fp_mask = preds == 1
    tn_mask = preds == 0

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_ids = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_ids[idx]} for idx in range(len(fp_rects))]
    easy_negative_list = [{'rect': tn_rects[idx], 'image_id': tn_image_ids[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negative_list

def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict()) # 深拷贝
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            is_training = phase == 'train'
            model.train(is_training) # 设置模型为训练模式或验证模式
            running_loss, running_corrects = 0.0, 0 # 记录损失和正确的样本数

            data_set = data_loaders[phase].dataset # 获取数据集
            print(f'{phase} - positive_num: {data_set.get_positive_num()} - negative_num: {data_set.get_negative_num()} - data size: {data_sizes[phase]}') # 打印数据集信息

            for inputs, labels, cache_dicts in data_loaders[phase]: # 获取一个batch的数据
                inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到GPU上

                optimizer.zero_grad() # 梯度清零
                with torch.set_grad_enabled(is_training): # 设置是否计算梯度
                    outputs = model(inputs) # 前向传播
                    _, preds = torch.max(outputs, 1) # 获取预测结果
                    loss = criterion(outputs, labels) # 计算损失
                    if is_training:
                        loss.backward() # 反向传播
                        optimizer.step() # 更新参数

                running_loss += loss.item() * inputs.size(0) # 累加损失
                running_corrects += torch.sum(preds == labels.data) # 累加正确的样本数

            if is_training:
                lr_scheduler.step() # 更新学习率

            epoch_loss = running_loss / data_sizes[phase] # 计算平均损失
            epoch_acc = running_corrects.double() / data_sizes[phase] # 计算平均正确率

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        train_dataset = data_loaders['train'].dataset # 获取训练数据集
        remain_negative_list = data_loaders['remain'] # 获取剩余的负样本
        jpeg_images = train_dataset.get_jpeg_images() # 获取训练数据集的图片
        transform = train_dataset.get_transform() # 获取训练数据集的transform

        with torch.set_grad_enabled(False):
            remain_dataset = CustomHardNegativeMiningDataset(remain_negative_list, jpeg_images, transform=transform) # 构建剩余负样本数据集
            remain_data_loader = DataLoader(remain_dataset, batch_size=batch_total, num_workers=8, drop_last=True) # 构建剩余负样本数据集的数据加载器

            negative_list = train_dataset.get_negatives() # 获取训练数据集的负样本
            add_negative_list = data_loaders.get('add_negative', []) # 获取已经添加的负样本
            running_corrects = 0 # 记录正确的样本数

            for inputs, labels, cache_dicts in remain_data_loader:
                inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到GPU上

                optimizer.zero_grad() # 梯度清零
                outputs = model(inputs) # 前向传播
                _, preds = torch.max(outputs, 1) # 获取预测结果
                running_corrects += torch.sum(preds == labels.data)

                hard_negative_list, easy_neagtive_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)

            remain_acc = running_corrects.double() / len(remain_negative_list)
            print(f'remiam negative size: {len(remain_negative_list)}, acc: {remain_acc:.4f}')

            train_dataset.set_negative_list(negative_list) # 更新训练数据集的负样本
            tmp_sampler = CustomBatchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                                             batch_positive, batch_negative)
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_total, sampler=tmp_sampler,
                                               num_workers=8, drop_last=True)
            data_loaders['add_negative'] = add_negative_list
            data_sizes['train'] = len(tmp_sampler) # 更新训练数据集的大小

        # 每训练一轮就保存
        save_model(model, './models/linear_svm_alexnet_cat_%d.pth' % epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型
    model.load_state_dict(best_model_weights)
    return model

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_loaders,data_sizes =load_data('./data/classifier_cat')
    # 加载CNN模型
    model_path = './models/alexnet_cat.pth'
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 固定特征提取
    for param in model.parameters():
        param.requires_grad = False
    # 创建SVM分类器
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # print(model)
    model = model.to(device)

    criterion = hinge_loss
    # 由于初始训练集数量很少，所以降低学习率
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # 共训练10轮，每隔4论减少一次学习率
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=10, device=device)
    # 保存最好的模型参数
    save_model(best_model, './models/best_linear_svm_alexnet_cat.pth')










