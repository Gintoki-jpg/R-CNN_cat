# 边界框回归训练
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import AlexNet
import sys
sys.path.insert(0, 'utils')
from utils.custom_bbox_dataset import BBoxRegressionDataset
import utils.util as util

def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset = BBoxRegressionDataset(data_root_dir, transform=transform) # 加载数据集
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8) # 构建数据加载器

    return dataloader


def train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    model.train()
    loss_list = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}') # 打印当前训练轮数
        print('-' * 10) # '-' * 10表示打印10个'-'

        running_loss = 0.0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.float().to(device) # 将数据移动到指定设备

            with torch.set_grad_enabled(True):
                features = feature_model.features(inputs) # 提取特征
                features = features.view(inputs.size(0), -1) # 展平特征

                optimizer.zero_grad() # 梯度清零

                outputs = model(features) # 前向传播
                loss = criterion(outputs, targets) # 计算损失

                loss.backward() # 反向传播
                optimizer.step() # 更新参数

                running_loss += loss.item() * inputs.size(0) # 累加损失
                lr_scheduler.step() # 更新学习率

        epoch_loss = running_loss / len(data_loader.dataset) # 计算平均损失
        loss_list.append(epoch_loss) # 记录损失

        print(f'{epoch} Loss: {epoch_loss:.4f}') # 打印损失
        util.save_model(model, f'./models/bbox_regression_{epoch}.pth') # 保存模型

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return loss_list

def get_model(device=None):
    # 加载CNN模型
    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_cat.pth'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model

if __name__ == '__main__':
    data_loader = load_data('./data/bbox_regression')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 选择设备
    feature_model = get_model(device) # 加载CNN模型

    # AlexNet最后一个池化层计算得到256*6*6输出
    in_features = 256 * 6 * 6
    out_features = 4
    model = nn.Linear(in_features, out_features)
    model.to(device)

    criterion = nn.MSELoss() # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Adam优化器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # 学习率衰减

    loss_list = train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, device=device,
                            num_epochs=12) # 训练模型
    util.plot_loss(loss_list) # 绘制损失曲线