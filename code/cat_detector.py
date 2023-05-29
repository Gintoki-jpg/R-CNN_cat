# 检测器实现
import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import sys
sys.path.insert(0, 'utils')
import sel_search as selectivesearch
import utils.util as util

def get_model(device=None):
    # 加载CNN模型
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_cat.pth'))
    model.eval()
    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)
    return model

def nms(rect_list, score_list, iou_threshold=0.3):

    # Convert input lists to arrays
    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # Sort detections by descending score
    sorted_idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[sorted_idxs]
    score_array = score_array[sorted_idxs]

    # Initialize lists for non-suppressed detections
    nms_rects = []
    nms_scores = []

    while len(score_array) > 0:
        # Add highest-scoring detection to non-suppressed list
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])

        if len(score_array) == 1:
            break

        # Calculate IoU between highest-scoring detection and remaining detections
        iou_scores = util.iou(np.array(nms_rects[-1]), rect_array[1:])
        # Remove detections with IoU >= threshold
        idxs = np.where(iou_scores < iou_threshold)[0]
        rect_array = rect_array[1:][idxs]
        score_array = score_array[1:][idxs]

    return nms_rects, nms_scores

def draw_box_with_text(img, rect_list, score_list):
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i] # rect_list[i]是一个tuple
        score = score_list[i] # score_list[i]是一个float
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1) # 画矩形框
        cv2.putText(img, "{:.3f}".format(score), (xmin+20, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) # 写上分类概率

def pic_detect(img_path): # 目标检测算法
    # 设置设备和变换
    # 通过torch.cuda.is_available()判断是否可用CUDA加速，如果可用则使用GPU设备，否则使用CPU设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 创建一个变换对象transform，该对象包含了一系列的图像变换操作，包括将图像转换为PIL图像对象、调整大小为(227, 227)、随机水平翻转、转换为张量、以及归一化处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载模型和创建selective search对象
    model = get_model(device) # 调用get_model(device)函数获取模型，并将其加载到设备上
    gs = selectivesearch.get_selective_search() # 创建selective search对象gs
    # 读取图像和初始化变量
    img = cv2.imread(img_path) # 读取图片，并将其保存为img变量
    dst = copy.deepcopy(img) # 通过深拷贝创建dst变量，用于绘制矩形框
    svm_thresh = 0.60 # 设置SVM分类阈值
    positive_list = []
    score_list = []

    # 执行selective search和分类
    selectivesearch.config(gs, img, strategy='f') # 配置selective search对象，将其应用于图像 -- 最好不要使用s模式，尽管速度快但准确率太低了，f模式相对来说速度和准确率折中
    rects = selectivesearch.get_rects(gs) # 获取selective search返回的候选框列表rects
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]
        rect_transform = transform(rect_img).to(device) # 对于每个候选框，将其在原图像中截取出来，并应用之前定义的变换transform进行处理
        output = model(rect_transform.unsqueeze(0))[0] # 将处理后的图像输入模型，获取输出结果output

        # 如果输出结果中最大值的索引为1，表示猫的概率最高，则将其加入到positive_list列表
        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()
            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
    # 执行非极大值抑制，对positive_list中的矩形框进行非极大值抑制，得到抑制后的矩形框列表nms_rects和得分列表nms_scores
    # nms_rects, nms_scores = nms(positive_list, [1.0] * len(positive_list))
    nms_rects, nms_scores = nms(positive_list, score_list)
    # draw_box_with_text(dst, nms_rects, nms_scores)  # 绘制边框及其分类概率
    # cv2.imshow('img', dst)
    # cv2.waitKey(0)
    return nms_rects, nms_scores


def vedio_detect(video_path):
    # 使用OpenCV库中的cv2.VideoCapture打开目标视频
    video = cv2.VideoCapture(video_path)

    # 创建一个cv2.TrackerCSRT_create对象，用于跟踪猫的位置
    # tracker = cv2.TrackerCSRT_create()
    tracker = cv2.TrackerKCF_create()

    # 初始化变量
    frame_count = 0
    bbox = None
    frame_rate = 300 # 帧率：每秒显示的帧数，这个设置大一些就可以实现视频的不间断检测（前提是视频中的图像位置变化不大）
    detection_interval = 1 * frame_rate  # 每隔一定帧数(frame_rate)进行一次猫的检测

    while True:
        # 通过video.read()读取视频的下一帧
        success, frame = video.read()
        if not success: # 读取失败时跳出循环，读取成功则继续
            break

        if frame_count % detection_interval == 0: # 控制检测频率（每隔一定帧数进行猫的检测）
            # 将当前帧保存为一张图片，并通过pic_detect函数检测其中的猫的位置
            img_path = f'./images/frame_{frame_count}.jpg'
            cv2.imwrite(img_path, frame)
            boxes, scores = pic_detect(img_path)
            # 如果检测到猫的位置（len(boxes) > 0），则将第一个检测到的猫的位置作为跟踪目标，并使用tracker.init(frame, bbox)来初始化跟踪器
            if len(boxes) > 0:
                bbox = boxes[0] # 将第一个检测到的猫的位置作为跟踪目标
                tracker.init(frame, bbox) # 使用tracker.init(frame, bbox)来初始化跟踪器

        # 使用跟踪器来更新猫的位置
        if bbox is not None:
            ret, bbox = tracker.update(frame) # 通过ret, bbox = tracker.update(frame)获取更新后的位置
            if ret: # 如果更新成功（ret为True），则根据猫的位置在帧上绘制矩形框，并用绿色标记
                x1, y1, w, h = [int(i) for i in bbox] # 将bbox中的位置信息转换为整数
                x2, y2 = x1 + w, y1 + h # 计算矩形框的右下角坐标

                # factor = 1.5  # 矩形框大小的缩放因子
                # x2, y2 = x1 + int(factor * w), y1 + int(factor * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 使用cv2.imshow显示处理后的帧，如果按下键盘上的'q'键，循环退出
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 循环结束后，释放视频资源并关闭窗口
    video.release()
    cv2.destroyAllWindows()



def test():
    # 测试图像检测
    img_path = './images/test_2.jpg'
    boxes, scores = pic_detect(img_path)
    print(boxes[0])
    print(scores)

def test_2():
    # 测试视频检测
    vedio_detect('./images/cat.mp4')


if __name__ == '__main__':
    # pass
    test()
    # test_2()