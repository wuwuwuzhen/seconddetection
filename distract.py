#主函数
from sixdrepnet import SixDRepNet
import cv2
from lanelines.rundef import cv_imread
import os

def Distraction(Test_path):
    if Test_path[0][-3:]=='jpg':
        model = SixDRepNet()
        Flag=[0]*len(Test_path)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                img = cv_imread(Test_path[i])
                pitch, yaw, roll = model.predict(img)
                if 20<=pitch<=90  or -90<=pitch<=-20 or 20<=yaw<=90  or -90<=yaw<=-20 or 20<=roll<=90  or -90<=roll<=-20: #分心驾驶判定阈值 后续需要具体确定有一个方向的头部偏移大于30度
                    Flag[i]=1
            else:
                Flag[i] = 1


    if Test_path[0][-3:]=='mp4':
        model = SixDRepNet()
        image_dir=[]
        Flag = [0]*len(Test_path)
        for i in range(len(Test_path)):#保存文件路径
            if os.path.exists(Test_path[i]):
                video = cv2.VideoCapture(Test_path[i])
                # 获取视频的帧率
                fps = video.get(cv2.CAP_PROP_FPS)
                # 计算每秒的中间帧索引
                mid_frame_index = fps // 2
                frame_count = 0  # 帧计数器
                frames = []  # 用于保存帧的列表
                while True:
                    # 读取视频帧
                    ret, frame = video.read()
                    # 如果视频帧读取失败，退出循环
                    if not ret:
                        break
                    # 检查当前帧是否是每秒的第一帧或中间帧
                    if frame_count % fps == 0 or frame_count % fps == mid_frame_index:
                        # 将帧添加到列表中
                        frames.append(frame)
                    frame_count += 1
                # 释放视频对象和关闭窗口
                video.release()
                cv2.destroyAllWindows()
                image_dir.append(frames)
            else:
                image_dir.append(0)
        for i in range(len(image_dir)):
            if image_dir[i]==0:
                continue
            else:
                for j in range(len(image_dir[i])):
                    pitch, yaw, roll = model.predict(image_dir[i][j])
                    if 20 <= pitch <= 90 or -90 <= pitch <= -20 or 20 <= yaw <= 90 or -90 <= yaw <= -20 or 20 <= roll <= 90 or -90 <= roll <= -20:  # 分心驾驶判定阈值 后续需要具体确定有一个方向的头部偏移大于30度
                        Flag[i]=1
                        break
    return Flag