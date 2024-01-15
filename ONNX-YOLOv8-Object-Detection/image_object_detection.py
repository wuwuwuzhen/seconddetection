import cv2 as cv
import numpy as np
from yolov8 import YOLOv8
import math
def cv_imread(file_path):
    cv_img=cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv_imwrite(file_path,img):
    cv_img=cv.imencode('.jpg', img)[1].tofile(file_path)
    return cv_img

#Initialize yolov8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
path="doc/img/test.png"
img=cv_imread(path)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)
Area=[]
for i in range(len(class_ids)):
    if class_ids[i]!=0:
        continue
    print(boxes[i].astype(int))
    x1, y1, x2, y2 = boxes[i].astype(int)
    area=math.fabs((y2-y1)*(x2-x1))
    Area.append(area)
# Draw detections
max_value = max(Area) # 求列表最大值
max_idx = Area.index(max_value)
picture_area=img.shape[0]*img.shape[1]
print(max_value,picture_area,(max_value/picture_area)*100,'%')
if max_value/picture_area>0.1:#(碰撞)
    flag=1
else:
    flag=2

# combined_img = yolov8_detector.draw_detections(img)
# cv.namedWindow("Detected Objects", cv.WINDOW_NORMAL)
# cv.imshow("Detected Objects", combined_img)
# cv.imwrite("doc/img/test_result.png", combined_img)
# cv.waitKey(100000)
# cv.destroyAllWindows()