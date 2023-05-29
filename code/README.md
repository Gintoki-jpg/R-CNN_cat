# 帮助文档

本项目是一个基于R-CNN目标检测算法实现的对视频中的cat进行检测的目标检测系统。该系统可以检测并识别视频中的猫咪，只需在可视化界面中输入需要检测的视频路径并点击“start”即可开始进行目标检测

![image-20230529111614615](images/image-20230529111614615.png)

该系统具有简洁美观的可视化界面，并且操作简单易上手。可将本系统作为组件嵌入到流浪猫搜寻系统、猫咪看护机器人等，可极大的提升这些系统的性能。

项目的部署方式如下。首先在电脑上安装pytorch的虚拟环境以及相关的依赖（详情可以参考目录下的enviroment.yml环境配置文件），或者使用命令

```
conda env create -n troch -f environment.yml
```

利用环境配置文件新建一个虚拟环境，激活该虚拟环境后，进入code目录，输入以下命令

```
python window.py
```

即可启动系统，在系统界面的输入框中输入需要检测的视频文件路径并点击"Start"检测按钮即可开始检测

![image-20230529111943045](images/image-20230529111943045.png)