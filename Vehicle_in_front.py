import numpy as np
import cv2 as cv
import math
import filetype
from PIL import Image
from PIL import ImageEnhance
def hisEqulColor1(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img
# 彩色图像进行自适应直方图均衡化
def hisEqulColor2(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img

def VehicleInFront(path):
    if path[-3:] == 'jpg':
        flag = 0
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            print("视频无法打开")
            exit()
        framenumber = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                #print("视频播放完毕")
                break
            width = frame.shape[1]
            hight = frame.shape[0]
            # frame = hisEqulColor1(frame)
            # frame = np.uint8(np.clip((cv.add(3 * frame, -30)), 0, 255))
            #frame = cv.convertScaleAbs(frame, alpha=2.0,beta=-20)
            # 提取特定区域
            # 处理帧， 将画面转化为灰度图
            gray = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
            blur = cv.GaussianBlur(gray, (7, 5), 0)
            img = cv.Canny(blur, 20, 40)
            poly_pts = np.array([[[int(width * 0), int(hight * 1)], [int(width * 0), int(hight * 7 / 10)],
                                  [int(width * 1), int(hight * 7 / 10)], [int(width * 1), int(hight * 1)]]])
            mask = np.zeros_like(img)
            cv.fillPoly(mask, pts=poly_pts, color=255)
            img_mask = cv.bitwise_and(img, mask)
            # cv.imshow('frame_window', img_mask)
            # if cv.waitKey(3000) == ord('q'):
            #     break

            front_lines = []
            front_lane = [0, 0, 0, 0]
            # 霍夫变换（参数5 累加平面阈值 参数6 最短长度 参数7 最大间隔）
            lines = cv.HoughLinesP(img_mask, 1, np.pi / 180, 10, minLineLength=100, maxLineGap=10)
            if lines is None:
                #print("NAN,当前帧未检测到前车")
                #cv.imshow("lane_video", frame)
                #cv.waitKey(25)  # 延时30ms
                flag=0
                continue
            for i in range(len(lines)):
                l = lines[i]
                if l[0][3] == l[0][1] or l[0][2] == l[0][0]:
                    continue
                slope = float(l[0][3] - l[0][1]) / float(l[0][2] - l[0][0])
                if (0 < math.fabs(slope) < 0.1
                        or l[0][2] == l[0][0]):
                    front_lines.append(l[0])
            if len(front_lines) == 0:
                #print("NAN,当前帧未检测到前车")
                #cv.imshow("lane_video", frame)
                #cv.waitKey(25)  # 延时30ms
                flag = 0
                continue
            #print("当前帧检测到前车")
            max_len = 0
            # 找出线最长的
            for l in front_lines:
                length = math.sqrt((l[3] - l[1]) ** 2 + (l[2] - l[0]) ** 2)
                if length > max_len:
                    max_len = length
                    front_lane = l
            # 画出车道
            if len(front_lines) != 0:
                cv.line(frame, (front_lane[0], front_lane[1]),(front_lane[2], front_lane[3]),(0, 255, 0), 3)
            if front_lane[1] >= (7 * hight / 10) or front_lane[3] >= (7 * hight / 10):
                cv.putText(frame, "Too close!", (int(0.3 * width), 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                #print("Too close!")
                flag=1
            framenumber += 1
            fpsString = f"framenumber:{framenumber}"
            #cv.putText(frame, fpsString, (40, 40), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
            #cv.imshow("lane_video", frame)
            #cv.waitKey(5000)  # 延时30ms
            #print(path,flag)
        cap.release()
        cv.destroyAllWindows()
        return flag

    if path[-3:] == 'mp4':
        flag = 0
        final_flag = 0
        Flag=[]
        cap = cv.VideoCapture(path)

        if not cap.isOpened():
            print("视频无法打开")
            exit()

        fps = cap.get(cv.CAP_PROP_FPS)
        # 计算每秒的中间帧索引
        mid_frame_index = fps // 2
        frame_count = 0  # 帧计数器
        while True:
            ret, frame = cap.read()
            if not ret:
                # print("视频播放完毕")
                break
            if frame_count % fps == 0 or frame_count % fps == mid_frame_index:
                width = frame.shape[1]
                hight = frame.shape[0]
                # frame = hisEqulColor1(frame)
                # frame = np.uint8(np.clip((cv.add(3 * frame, -30)), 0, 255))
                # frame = cv.convertScaleAbs(frame, alpha=2.0,beta=-20)
                # 提取特定区域
                # 处理帧， 将画面转化为灰度图
                gray = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
                blur = cv.GaussianBlur(gray, (7, 5), 0)
                img = cv.Canny(blur, 20, 40)
                poly_pts = np.array([[[int(width * 0), int(hight * 1)], [int(width * 0), int(hight * 7 / 10)],
                                      [int(width * 1), int(hight * 7 / 10)], [int(width * 1), int(hight * 1)]]])
                mask = np.zeros_like(img)
                cv.fillPoly(mask, pts=poly_pts, color=255)
                img_mask = cv.bitwise_and(img, mask)
                # cv.imshow('frame_window', img_mask)
                # if cv.waitKey(3000) == ord('q'):
                #     break
                front_lines = []
                front_lane = [0, 0, 0, 0]
                # 霍夫变换（参数5 累加平面阈值 参数6 最短长度 参数7 最大间隔）
                lines = cv.HoughLinesP(img_mask, 1, np.pi / 180, 10, minLineLength=100, maxLineGap=10)
                if lines is None:
                    # print("NAN,当前帧未检测到前车")
                    # cv.imshow("lane_video", frame)
                    # cv.waitKey(25)  # 延时30ms
                    continue
                for i in range(len(lines)):
                    l = lines[i]
                    if l[0][3] == l[0][1] or l[0][2] == l[0][0]:
                        continue
                    slope = float(l[0][3] - l[0][1]) / float(l[0][2] - l[0][0])
                    if (0 < math.fabs(slope) < 0.1
                            or l[0][2] == l[0][0]):
                        front_lines.append(l[0])
                if len(front_lines) == 0:
                    # print("NAN,当前帧未检测到前车")
                    # cv.imshow("lane_video", frame)
                    # cv.waitKey(25)  # 延时30ms
                    continue
                # print("当前帧检测到前车")
                max_len = 0
                # 找出线最长的
                for l in front_lines:
                    length = math.sqrt((l[3] - l[1]) ** 2 + (l[2] - l[0]) ** 2)
                    if length > max_len:
                        max_len = length
                        front_lane = l
                # 画出车道
                if len(front_lines) != 0:
                    cv.line(frame, (front_lane[0], front_lane[1]), (front_lane[2], front_lane[3]), (0, 255, 0), 3)
                if front_lane[1] >= (7 * hight / 10) or front_lane[3] >= (7 * hight / 10):
                    cv.putText(frame, "Too close!", (int(0.3 * width), 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                    # print("Too close!")
                    flag = 1
                # cv.imshow("lane_video", frame)
                # cv.waitKey(5000)  # 延时30ms
                # print(path, flag)
                Flag.append(flag)
            frame_count += 1
        cap.release()
        cv.destroyAllWindows()

        for i in range(len(Flag)):
            if Flag[i]!=0:
                final_flag=Flag[i]
                break
            else:
                continue
        return final_flag



