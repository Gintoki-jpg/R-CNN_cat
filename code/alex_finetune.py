# 微调实现，依赖自定义微调数据集类以及自定义批量采样器类
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import sys
sys.path.insert(0, 'utils')
from utils.custom_alex_dataset import CustomFinetuneDataset
from utils.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir

def load_data(data_root_dir, batch_size=128, num_workers=8):
    # 定义应用于数据集中每个图像的转换
    data_transforms = transforms.Compose([
        transforms.Resize((227, 227)), # 将图像大小调整为AlexNet所需的大小
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 将图像转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 标准化图像
    ])

    # 使用CustomFinetuneDataset创建训练和验证数据集
    train_dataset = CustomFinetuneDataset(os.path.join(data_root_dir, 'train'), transform=data_transforms)
    val_dataset = CustomFinetuneDataset(os.path.join(data_root_dir, 'val'), transform=data_transforms)

    # 使用CustomBatchSampler平衡每个批次中的正样本和负样本数量
    train_sampler = CustomBatchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(), 32, 96)

    # 使用DataLoader创建训练和验证数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # 返回一个带有训练和验证数据加载器及其各自数据集大小的字典
    return {'train': train_loader, 'val': val_loader}, {'train': train_sampler.__len__(), 'val': val_dataset.__len__()}

def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None): # 训练模型
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict()) # 保存最好的模型参数
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}") # 打印当前训练轮数
        print("-" * 10) # 打印分割线

        for phase in ["train", "val"]: # 遍历训练和验证阶段
            is_train = phase == "train"
            model.train(is_train)
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in data_loaders[phase]: # 遍历数据加载器
                inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到device上(GPU/CPU)
                optimizer.zero_grad() # 梯度清零

                with torch.set_grad_enabled(is_train): # 设置梯度计算开关
                    outputs = model(inputs) # 前向传播
                    _, preds = torch.max(outputs, 1) # 获取预测结果
                    loss = criterion(outputs, labels) # 计算损失

                    if is_train:
                        loss.backward() # 反向传播
                        optimizer.step() # 更新参数

                running_loss += loss.item() * inputs.size(0) # 累加损失
                running_corrects += torch.sum(preds == labels.data) # 累加正确预测的样本数

            epoch_loss = running_loss / len(data_loaders[phase].dataset) # 计算平均损失
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset) # 计算平均准确率
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}") # 打印损失和准确率

            if phase == "val" and epoch_acc > best_acc: # 如果是验证阶段且准确率更高
                best_acc = epoch_acc # 更新最好的准确率
                best_model_weights = copy.deepcopy(model.state_dict()) # 更新最好的模型参数

            if is_train:
                lr_scheduler.step() # 更新学习率

        print()

    time_elapsed = time.time() - since # 计算训练时间
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s") # 打印训练时间
    print(f"Best val Acc: {best_acc:.4f}") # 打印最好的准确率
    model.load_state_dict(best_model_weights) # 加载最好的模型参数
    return model

if __name__ == '__main__':
    print('start')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 检查是否有GPU可用
    print('1')
    dataloaders, dataset_sizes = load_data(data_root_dir="./data/finetune_cat", batch_size=128, num_workers=8) # 加载数据
    print('2')
    model = models.alexnet(pretrained=True) # 加载预训练模型
    num_ftrs = model.classifier[6].in_features # 获取分类器的输入特征数
    model.classifier[6] = nn.Linear(num_ftrs, 2) # 将分类器的输出特征数改为2
    model = model.to(device) # 将模型移动到device上(GPU/CPU)

    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # 定义优化器
    ls_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # 定义学习率调度器

    best_model = train_model(dataloaders, model, criterion, optimizer, ls_scheduler, num_epochs=25, device=device) # 训练模型
    check_dir("./models") # 检查模型保存目录是否存在
    torch.save(best_model.state_dict(), "./models/alexnet_cat.pth") # 保存模型参数
