import cv2 as cv
import numpy as np
from yolov8 import YOLOv8
import math
from PIL import Image
import os
import model_path

def cv_imread(file_path):
    cv_img=cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv_imwrite(file_path,img):
    cv_img=cv.imencode('.jpg', img)[1].tofile(file_path)
    return cv_img
def vehicle_collision(Test_path):
    #Initialize yolov8 object detector
    yolo_model_path = os.path.join(model_path.root_path, 'models/yolov8m.onnx')
    yolov8_detector = YOLOv8(yolo_model_path, conf_thres=0.2, iou_thres=0.3)
    if Test_path[0][-3:] == 'jpg':
        Flag=[0]*len(Test_path)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                cap = cv.VideoCapture(Test_path[i])
                try:
                    Image.open(Test_path[i])
                except (IOError, SyntaxError) as e:
                    Flag[i] = 4
                    continue
                if not cap.isOpened():
                    Flag[i] = 4
                    continue
                img=cv_imread(Test_path[i])
                # Detect Objects
                boxes, scores, class_ids = yolov8_detector(img)
                Area=[]
                for k in range(len(class_ids)):
                    if class_ids[k] in [2, 5, 7]:
                        # print(boxes[i].astype(int))
                        x1, y1, x2, y2 = boxes[k].astype(int)
                        area = math.fabs((y2 - y1) * (x2 - x1))
                        Area.append(area)
                    else:
                        continue
                # Draw detections
                if len(Area) == 0:
                    Flag[i] = 2
                    continue
                else:
                    max_value = max(Area) # 求列表最大值
                    max_idx = Area.index(max_value)
                    picture_area=img.shape[0]*img.shape[1]
                    # print(max_value,picture_area,(max_value/picture_area)*100,'%')
                    if max_value/picture_area>0.09:#(碰撞)
                        Flag[i]=1
                    else:
                        Flag[i]=2
            else:
                Flag[i] = 4

    if Test_path[0][-3:] == 'mp4':
        Flag = [0] * len(Test_path)
        image_dir = []
        for i in range(len(Test_path)):  # 保存文件路径
            if os.path.exists(Test_path[i]):
                try:
                    cap = cv.VideoCapture(Test_path[i])
                    if not cap.isOpened():
                        image_dir.append(0)
                        continue
                    else:
                        while True:
                            ret, frame = cap.read()
                            # 如果读取到最后一帧则结束循环
                            if not ret:
                                break
                        cap.release()
                except Exception as e:
                    image_dir.append(0)
                    continue
                video = cv.VideoCapture(Test_path[i])
                total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
                frames_interval = total_frames // 10
                for j in range(0, total_frames, frames_interval):
                    video.set(cv.CAP_PROP_POS_FRAMES, j)
                    ret, frame = video.read()
                    if ret:
                        cv_imwrite('yolo_picture/'+str(i)+'_'+str(j)+'.jpg', frame)#从视频提取jpg
                video.release()
                cv.destroyAllWindows()
                image_dir.append(1)
            else:
                image_dir.append(0)
        for i in range(len(image_dir)):
            if image_dir[i] == 0:
                Flag[i] = 4
                continue
            else:
                for file in os.listdir('yolo_picture/'):
                    if (file.split('_')[0]) == str(i):
                        path = 'yolo_picture/' + file
                        img = cv_imread(path)
                        # Detect Objects
                        boxes, scores, class_ids = yolov8_detector(img)
                        Area = []
                        for k in range(len(class_ids)):
                            if class_ids[k] in [2, 5 ,7]:
                                x1, y1, x2, y2 = boxes[k].astype(int)
                                # print(boxes[i].astype(int))
                                area = math.fabs((y2 - y1) * (x2 - x1))
                                Area.append(area)
                            else:
                                continue
                        # Draw detections
                        if len(Area) == 0:
                            Flag[i] = 2
                            continue
                        else:
                            max_value = max(Area)  # 求列表最大值
                            max_idx = Area.index(max_value)
                            picture_area = img.shape[0] * img.shape[1]
                            # print(max_value,picture_area,(max_value/picture_area)*100,'%')
                            if max_value / picture_area > 0.09:  # (碰撞)
                                Flag[i] = 1
                if Flag[i] != 1:
                    Flag[i] =2
        dir_path = os.path.join(model_path.root_path, 'yolo_picture')
        for filename in os.listdir(dir_path):
            # 构造文件的完整路径
            file_path = os.path.join(dir_path, filename)
            # 如果是jpg图片，则删除
            if os.path.isfile(file_path) and filename.endswith('.jpg'):
                os.remove(file_path)
    return Flag


# combined_img = yolov8_detector.draw_detections(img)
# cv.namedWindow("Detected Objects", cv.WINDOW_NORMAL)
# cv.imshow("Detected Objects", combined_img)
# cv.imwrite("doc/img/test_result.png", combined_img)
# cv.waitKey(100000)
# cv.destroyAllWindows()