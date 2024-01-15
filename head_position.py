#主函数
from sixdrepnet import SixDRepNet
import cv2

def Distraction(path):
    model = SixDRepNet()
    img = cv2.imread(path)
    pitch, yaw, roll = model.predict(img)
    if pitch>0 or yaw>0 or yaw>0: #分心驾驶判定阈值 后续需要具体确定
        return 1
    else:
        return 0

# # model.draw_axis(img, yaw, pitch, roll)
# # cv2.imshow("test_window", img)
# # cv2.waitKey(0)

# import face_alignment
# from skimage import io
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
# io.imread('D:\Phd\Bus program/align.jpg')
# output = fa.get_landmarks(input)
# print(output)
# save_image(output, 'D:\Phd\Bus program/aligned.jpg')