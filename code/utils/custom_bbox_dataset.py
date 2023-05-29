import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils.util as util

class BBoxRegressionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.transform = transform

        samples = util.parse_cat_csv(root_dir)
        self.jpeg_list = [cv2.imread(os.path.join(root_dir, 'JPEGImages', s + '.jpg')) for s in samples]
        self.bndbox_list = [np.loadtxt(os.path.join(root_dir, 'bndboxs', s + '.csv'), dtype=np.int, delimiter=' ') for s in samples]
        self.positive_list = [np.loadtxt(os.path.join(root_dir, 'positive', s + '.csv'), dtype=np.int, delimiter=' ') for s in samples]

        self.box_list = []
        for i, (bndboxes, positives) in enumerate(zip(self.bndbox_list, self.positive_list)):
            if len(positives.shape) == 1:
                bndbox = self.get_bndbox(bndboxes, positives)
                self.box_list.append({'image_id': i, 'positive': positives, 'bndbox': bndbox})
            else:
                for positive in positives:
                    bndbox = self.get_bndbox(bndboxes, positive)
                    self.box_list.append({'image_id': i, 'positive': positive, 'bndbox': bndbox})

    def __len__(self):
        return len(self.box_list)

    def __getitem__(self, index):
        assert index < self.__len__(), f"Index {index} is out of range for dataset of size {self.__len__()}"

        box_dict = self.box_list[index] # 获取box字典
        image_id = box_dict['image_id'] # 图片id
        positive = box_dict['positive'] # 正样本
        bndbox = box_dict['bndbox'] # 目标框

        jpeg_img = self.jpeg_list[image_id] # 获取图片
        xmin, ymin, xmax, ymax = positive # 正样本坐标
        image = jpeg_img[ymin:ymax, xmin:xmax] # 截取正样本

        if self.transform:
            image = self.transform(image)

        p_w, p_h = xmax - xmin, ymax - ymin # 正样本宽高
        p_x, p_y = xmin + p_w / 2, ymin + p_h / 2 # 正样本中心点

        xmin, ymin, xmax, ymax = bndbox # 目标框坐标
        g_w, g_h = xmax - xmin, ymax - ymin # 目标框宽高
        g_x, g_y = xmin + g_w / 2, ymin + g_h / 2 # 目标框中心点

        t_x, t_y = (g_x - p_x) / p_w, (g_y - p_y) / p_h # 坐标偏移
        t_w, t_h = np.log(g_w / p_w), np.log(g_h / p_h) # 宽高缩放

        return image, np.array((t_x, t_y, t_w, t_h))

    def get_bndbox(self, bndboxes, positive):
        if len(bndboxes.shape) == 1: # 只有一个bndbox
            return bndboxes
        else: # 多个bndbox
            scores = util.iou(positive, bndboxes) # 计算IOU
            return bndboxes[np.argmax(scores)]

def test():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root_dir = '../data/bbox_regression'
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)

    print(data_set.__len__())
    image, target = data_set.__getitem__(10)
    print(image.shape)
    print(target)
    print(target.dtype)

def test2():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root_dir = '../data/bbox_regression'
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)

    items = next(data_loader.__iter__())
    datas, targets = items
    print(datas.shape)
    print(targets.shape)
    print(targets.dtype)

if __name__ == '__main__':
    # test()
    test2()