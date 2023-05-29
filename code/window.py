# 可视化界面搭建
import cat_detector
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QIcon
import os

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和图标
        self.setWindowTitle('Cat_Detector')
        self.setWindowIcon(QIcon('./images/icon.jpg'))

        # 创建提示标签
        self.label = QLabel('Please enter your file path:', self)

        # 创建输入框
        self.input_box = QLineEdit(self)

        # 创建Start按钮
        self.button = QPushButton('Start', self)
        self.button.clicked.connect(self.start_function)

        # 创建一个垂直布局
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.input_box)
        vbox.addWidget(self.button)

        # 创建一个水平布局
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)
        hbox.addStretch(1)

        # 将水平布局设置为窗口的布局
        self.setLayout(hbox)

        # 设置窗口的大小和位置
        self.resize(400, 150)
        self.center()
        self.show()

    def center(self):
        # 设置窗口居中
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def start_function(self):
        # 获取视频路径
        file_path = self.input_box.text()
        # 若输入的视频路径为空，弹出提示
        if file_path == '':
            self.label.setText('Error: Please enter your file path!')
            return
        # 若输入的视频路径不存在，弹出提示
        elif not os.path.exists(file_path):
            self.label.setText('Error: The file path does not exist!')
            return
        # 若输入的不是视频文件，弹出提示
        elif not file_path.endswith('.mp4'):
            self.label.setText('Error: Please enter a video file path!')
            return
        # 开始检测
        else:
            self.label.setText('Please wait for the result...')
            cat_detector.vedio_detect(file_path)
            # 重新输入文件
            self.label.setText('Please enter your file path:')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())