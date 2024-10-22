# Dorisoy.HandTracking

使用 python 脚本和 C#脚本的组合来创建一个具有手部跟踪和手势识别功能的 Unity 应用程序。使用 python 脚本来检测手部运动和相对手势。

## 要求

项目需要[Vuforia](https://developer.vuforia.com/)如果你想自己安装，下载 Asset 文件夹（在 Unity Project 文件夹内）。

## 手检测

1.使用3x3高斯滤波器去噪图像。

2.转换图像到HSV颜色空间。

3.应用阈值操作以获得二进制图像（肤色像素变为白色，其他为黑色）。

4.应用膨胀和腐蚀填充可能的孔洞。

5.再次应用高斯滤波器以平滑边缘。

6.使用OpenCV的findContours()函数从二进制图像获取轮廓。

7.使用Suzuky和Abe开发的算法并通过OpenCV的convexHull()实现Sklansky算法找到凸包。

## 手跟踪（左侧）

1.将点放置在之前找到的轮廓的最高点上（y值最小）。

2.实现稳定器例程以避免点围绕指尖摆动。

## 手势识别（右侧）

1.基于右手每帧的手指数量进行手势识别。

2.使用余弦定理从凸包和轮廓中找到手的缺陷。

3.为每个缺陷形成三角形，并计算红点（对应于缺陷）处的角度：


[\theta = \arccos\left(\frac{b^2 + c^2 - a^2}{2bc}\right) \times \frac{180}{\pi}]

其中 ( \theta ) 是度数形式的角度，a、b、c是图中三角形的边。


## 使用RCNN进行手跟踪

1.使用Fast-RCNN进行手部追踪，训练了40个epoch。使用了迁移学习（仅训练最后一层）。

## 使用CNN进行手指计数

1.开发了一个接受手部图像作为输入并输出手指数量的简单CNN。

2.网络结构包括4个卷积层、每层后跟一个最大池化层和最后的两个全连接层，激活函数选择ReLU。

3.在200个图像上训练了250个epoch，使用Cross Entropy作为损失函数和AdaDelta作为优化器。


## Python

```python
import numpy as np
import cv2
import math
import socket
import time

from hand_detector_utils import *

UDP_IP = "127.0.0.1"
UDP_PORT = 5065

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

last = []

good_condition = False
drawing_box = True
full_frame = False
stabilize_highest_point = True

old_highest_point = (-1, -1)

x1_crop = 0
y1_crop = 60
x2_crop = 320
y2_crop = 420

# Open Camera
try:
    default = 1 # Try Changing it to 1 if webcam not found
    capture = cv2.VideoCapture(default)
except:
    print("No Camera Source Found!")

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    width = frame.shape[1]

    img_left = frame[y1_crop:y2_crop, 0:int(width/3)]
    img_right = frame[y1_crop:y2_crop, int(width/3 * 2): int(width)]

    try:

        contour_left = detectHand(img_left)
        contour_right = detectHand(img_right)

        defects_left, drawing_left = findDefects(img_left, contour_left)
        defects_right, drawing_right = findDefects(img_right, contour_right)

        # Count defects (in the right image)
        count_defects = countDefects(defects_right, contour_right, img_right)

        # Track highest point (in the left image)
        highest_point = trackHighestPoint(defects_left, contour_left)

        if(stabilize_highest_point):
            if( old_highest_point == (-1, -1)): old_highest_point = highest_point
            else:
                # Evaluate the magnitude of the difference
                diag_difference = np.linalg.norm(np.asarray(old_highest_point) - np.asarray(highest_point))

                # If the difference is bigger than a threshold then I actually moved my finger
                if(diag_difference >= 9.5):
                    # print("diag_difference = ", diag_difference)
                    old_highest_point = highest_point
                else: highest_point = old_highest_point;

        if(full_frame):
            highest_point = (highest_point[0], highest_point[1])
            cv2.circle(frame, highest_point, 10, [255,0,255], -1)
        else:
            cv2.circle(img_left, highest_point, 10, [255,0,255], -1)
            highest_point = (highest_point[0] + x1_crop, highest_point[1] + y1_crop)
            cv2.circle(frame, highest_point, 10, [255,0,255], -1)

        # Print number of fingers
        textDefects(frame, count_defects,debug_var = False)

        # Show required images
        if(drawing_box):
            cv2.rectangle(frame, (x1_crop, y1_crop), (int(width/3), y2_crop),(0,0,255), 1)
            cv2.rectangle(frame, (int(width/3 * 2), y1_crop), (int(width), y2_crop),(0,0,255), 1)
        cv2.imshow("Full Frame", frame)

        all_image_left = np.hstack((drawing_left, img_left))
        cv2.imshow('Recognition Left', all_image_left)

        all_image_right = np.hstack((drawing_right, img_right))
        cv2.imshow('Recognition Right', all_image_right)

        last.append(count_defects)
        if(len(last) > 5):
            last = last[-5:]
            # last = []


        # Check if previously hand was wide open (3/4 fingers in previous frames), and is now a fist (0 fingers)
        if(good_condition):
            if(count_defects == 0 and 4 in last):
                last = []
                sendCommand(sock, UDP_IP, UDP_PORT, "ACTION")

            elif(count_defects == 0 and 2 in last):
                last = []
                sendCommand(sock, UDP_IP, UDP_PORT, "BACK")

        else:
            if(count_defects == 0 and 4 in last):
                last = []
                sendCommand(sock, UDP_IP, UDP_PORT, "ACTION")

        command = "l " + str(highest_point[0]) + " " + str(highest_point[1])


        sendCommand(sock, UDP_IP, UDP_PORT, command, debug_var = False)

    except:
        pass

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
```

## 屏幕

https://github.com/dorisoy/Dorisoy.HandTracking/blob/main/Video/826bc7f6c296dfae62b87318218fe8eb.mp4
