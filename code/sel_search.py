# 区域候选建议算法，借助cv2库实现
import sys
import cv2

def get_selective_search(): # 获取选择性搜索算法对象
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs

def config(gs, img, strategy='q'): # 配置选择性搜索算法
    gs.setBaseImage(img) # 设置基础图像
    if (strategy == 's'):
        gs.switchToSingleStrategy() # 设置单一策略
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast() # 设置快速策略
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality() # 设置质量策略
    else:
        print(__doc__) # 打印脚本说明文档
        sys.exit(1)

def get_rects(gs): # 获取候选区域
    rects = gs.process()
    rects[:, 2] += rects[:, 0] # 将候选区域的坐标转换为(x1, y1, x2, y2)的形式
    rects[:, 3] += rects[:, 1]
    return rects

def test():
    gs = get_selective_search()
    img = cv2.imread('./images/test.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='s')  # single模式
    rects = get_rects(gs)
    print(rects)
    # 在原图上绘制候选区域
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    test()






