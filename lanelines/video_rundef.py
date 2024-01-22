import torch
from lanelines.dataset import RoadSequenceDataset, RoadSequenceDatasetList
from lanelines.model import generate_model
from torchvision import transforms
from PIL import Image
import time
import argparse
import numpy as np
import cv2 as cv
import os
import pandas as pd
import math
import shutil
import sys

# globel param
# dataset setting
img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2

# path
#home_path="D:/college/研究生/驾驶行为检测/Demo_picture_video_12.25"

home_path = os.path.dirname(__file__)
resize_path=home_path+"/lanelines/video_resize"
test_path = home_path+"/lanelines/video.txt"
fps_path = home_path+"/lanelines/video_fps.txt"
video_pos_path = home_path+"/lanelines/video_position.txt"
save_path = home_path+"/lanelines/video_result/"
result_path = home_path+"/lanelines/video_result_resize/"
pretrained_path=home_path+'/lanelines/pretrained/unetlstm.pth'
lane_path = home_path+"/lanelines/video_result_lane/"


# weight
class_weight = [0.02, 1.02]


###读取中文路径图片
def cv_imread(file_path):
    cv_img=cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv_imwrite(file_path,img):
    cv_img=cv.imencode('.jpg', img)[1].tofile(file_path)
    return cv_img

def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def Remove():
    with open(test_path, 'a+', encoding='utf-8') as test1:
        test1.truncate(0)
    with open(fps_path, 'a+', encoding='utf-8') as test3:
        test3.truncate(0)
    with open(video_pos_path, 'a+', encoding='utf-8') as test4:
        test4.truncate(0)
    RemoveDir(resize_path)
    RemoveDir(save_path)
    RemoveDir(result_path)
    RemoveDir(lane_path)


def check_video(filepath):
    try:
        cap = cv.VideoCapture(filepath)
        if not cap.isOpened():
            print("无法打开视频文件")
        else:
            while True:
                ret, frame = cap.read()
                # 如果读取到最后一帧则结束循环
                if not ret:
                    break
            cap.release()
            print("视频正常")
    except Exception as e:
        print("发生错误:", str(e))

# 调用函数进行视频检测

####批量处理256*128
def resize(video_path):
    Remove()
    saveFile = resize_path
    fpslist=[]
    video_position=[]
    # print("图片视频数量",len(video_path))
    for i in range(len(video_path)):
        if (os.path.exists(video_path[i][0])):
            #check_video(video_path[i][0])
            filepath = video_path[i][0]
            file = video_path[i][1]
            cap = cv.VideoCapture(filepath)
            if not cap.isOpened():
                # print("损坏位置",i)
                continue
            else:
                c = 0
                j = 0
                fps = cap.get(cv.CAP_PROP_FPS)  # 获取帧率
                size = (int(256), int(128))

                while True:
                    # 捕获视频帧，返回ret，frame
                    # ret的true与false反应是否捕获成功，frame是画面
                    ret, frame = cap.read()
                    if not ret:
                        #print("视频播放完毕")
                        break
                    if ret:
                        if (c % round(fps) == 0):  # 每隔fps帧进行操作
                            file_path = saveFile + '/' + file.split('.')[0] + '_' + str(j) + '.jpg'
                            shrink = cv.resize(frame, size, interpolation=cv.INTER_AREA)
                            cv_imwrite(file_path, shrink)
                            j = j + 1
                            if j == 10:
                                break
                        cv.waitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)
                        c = c + 1
                    else:
                        break
                fpslist.append(j)
                video_position.append(i)
        else:
            continue
    if len(video_position)==0:
        # print("无视频")
        return
    df = pd.DataFrame(fpslist)
    df.to_csv(fps_path, sep='\t', index=False, header=False)
    df_p = pd.DataFrame(video_position)
    df_p.to_csv(video_pos_path, sep='\t', index=False, header=False)
###写成深度学习格式
def unetlstmtxt():
    os.chdir(resize_path)
    list = []
    path_list = os.listdir(resize_path)
    #path_list.sort(key=lambda x: int((x.split('.')[0]).split('_')[5]))
    for file in path_list:
    #file_chdir = os.getcwd()  # 获得工作目录
    #for root, dirs, files in os.walk(file_chdir):  # os.walk会便利该目录下的所有文件
        #for file in files:
        list1 = []
        for i in range(6):
            file_path = os.path.join(home_path,'video_resize', file)
            list1.append(file_path)
        list.append(list1)
    df = pd.DataFrame(list)
    df.to_csv(test_path,sep='\t', index=False, header=False)

def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model',type=str, default='UNet-ConvLSTM',help='( UNet-ConvLSTM | SegNet-ConvLSTM | UNet | SegNet | ')
    parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args([])
    return args


def output_result(args, model, test_loader, device):
    model.eval()
    k = 0
    feature_dic=[]
    with torch.no_grad():
        for sample_batched in test_loader:
            k+=1
            # print(k)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output,feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
            img = Image.fromarray(img.astype(np.uint8))

            data = torch.squeeze(data).cpu().numpy()
            if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
                data = np.transpose(data[-1], [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = (img.getpixel((i, j)))
                    if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data.save(save_path + "%s_data.jpg" % k)#red line on the original image
            img.save(save_path + "%s_pred.jpg" % k)#prediction result

def get_parameters(model, layer_name):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.UpsamplingBilinear2d
    )
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma

###检测后的图片改名
def rename():
    if not os.path.getsize(test_path):
        return
    saveFile = result_path
    pathin = save_path
    filelist = os.listdir(pathin)  # 获取文件路径
    total_num = len(filelist)  # 获取文件长度（文件夹下图片个数）
    data = pd.read_csv(test_path,sep='\t',header=None)

    i = 0  # 表示文件的命名是从1开始的
    size = (int(1280), int(720))
    for item in filelist:
        if item[-8:-4]!='pred':  # 初始的图片的格式为jpg格式的
            continue
        else:
            for i in range(int(total_num/2)):
                if item==(format(str(i + 1)) + '_pred.jpg'):
                    filename = data.loc[i][0]
                    filename = filename[23:]
                    file_path = os.path.join(saveFile, item)
                    item_path = pathin+"//" + item
                    image1 = cv_imread(item_path)
                    shrink = cv.resize(255-image1, size, interpolation=cv.INTER_AREA)
                    cv_imwrite(file_path, shrink)
                    break
                else:
                    i = i + 1

def LaneLines(path,file):

    index=int(file.split('_')[3])
    flag = 0
    flagl = 0
    flagr = 0
    flagm = 0
    cap = cv.VideoCapture(path)
    # 检查是否导入视频成功

    framenumber = 0

    while True:
        # 捕获视频帧，返回ret，frame
        # ret的true与false反应是否捕获成功，frame是画面
        ret, frame = cap.read()
        origin = np.copy(frame)
        #frame=hisEqulColor2(frame)
        if not ret:
            #print("视频播放完毕")
            break
        width = frame.shape[1]
        hight = frame.shape[0]

        # 提取特定区域
        # 处理帧， 将画面转化为灰度图
        gray = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
        blur = cv.GaussianBlur(gray, (9, 7), 0)
        img = cv.Canny(blur, 20, 80)
        poly_pts = np.array([[[int(width * 0), int(hight * 1)], [int(width * 0), int(hight * 2 / 5)],
                              [int(width * 1), int(hight * 2 / 5)], [int(width * 1), int(hight * 1)]]])
        mask = np.zeros_like(img)
        cv.fillPoly(mask, pts=poly_pts, color=255)
        img_mask = cv.bitwise_and(img, mask)
        if cv.waitKey(25) == ord('q'):
            break

        lines = []
        right_lines = []
        left_lines = []
        max_len = 0
        right_lane = [0, 0, 0, 0]
        left_lane = [0, 0, 0, 0]

        # 霍夫变换（参数5 累加平面阈值 参数6 最短长度 参数7 最大间隔）
        lines = cv.HoughLinesP(img_mask, 1, np.pi / 180, 10, minLineLength=100, maxLineGap=100)

        if lines is None:
            if index == 6 or index == 5:
                flagr = 1
                flag = 1
            elif index == 4 or index == 3:
                flagl = 1
                flag = 2
            continue

        for i in range(len(lines)):
            l = lines[i]
            if l[0][3] == l[0][1] or l[0][2] == l[0][0]:
                continue
            slope = float(l[0][3] - l[0][1]) / float(l[0][2] - l[0][0])
            if 0.5 < slope < 3 and l[0][0] > img.shape[1] / 2:
                right_lines.append(l[0])
            elif -3 < slope < -0.5 and l[0][2] < img.shape[1] / 2:
                left_lines.append(l[0])

        if len(right_lines) == 0 or len(left_lines) == 0:
            if index == 6 or index == 5:
                flagr = 1
                flag = 1
            elif index == 4 or index == 3:
                flagl = 1
                flag = 2
            continue

        max_len = 0
        max_slope = 0
        # 找出右车道线最长的
        for l in right_lines:
            length = math.sqrt((l[3] - l[1]) ** 2 + (l[2] - l[0]) ** 2)
            slope = math.fabs(float(l[3] - l[1]) / float(l[2] - l[0]))
            if slope > max_slope:
                max_slope = slope
                right_lane = l
        # 找出左车道线最长的
        max_len = 0
        max_slope = 0
        for l in left_lines:
            length = math.sqrt((l[3] - l[1]) ** 2 + (l[2] - l[0]) ** 2)
            slope = math.fabs(float(l[3] - l[1]) / float(l[2] - l[0]))
            if slope > max_slope:
                max_slope = slope
                left_lane = l

        # 画出车道
        if len(right_lines) != 0:
            cv.line(frame, (right_lane[0], right_lane[1]),(right_lane[2], right_lane[3]),(0, 255, 0), 3)
        if len(left_lines) != 0:
            cv.line(frame, (left_lane[0], left_lane[1]),(left_lane[2], left_lane[3]),(0, 255, 0), 3)
        right_x, left_x, sloper, slopel, br, bl, xm = 0, 0, 0, 0, 0, 0, 0
        sloper = (right_lane[3] - right_lane[1]) / (right_lane[2] - right_lane[0])
        right_x = right_lane[0] - (right_lane[1] / sloper)
        br = sloper * right_x

        slopel = (left_lane[1] - left_lane[3]) / (left_lane[2] - left_lane[0])
        left_x = left_lane[2] + (left_lane[3] / slopel)
        bl = slopel * left_x
        xm = (bl + br) / (sloper + slopel)
        if right_x < gray.shape[1] and left_x > 0:
            if xm < 0.4 * width:
                cv.putText(frame, "right", (450, 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                flagr=1
                flag=1
            elif xm > 0.6 * width:
                cv.putText(frame, "left", (80, 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                flagl=1
                flag = 2
            elif xm <  0.5*width+3 and xm >  0.5*width-3:
                cv.putText(frame, "middle", (80, 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                flagm = 1
                flag = 5
            else:
                if(slopel<sloper):
                    cv.putText(frame, "right", (450, 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                    flagr = 1
                    flag = 1
                else:
                    cv.putText(frame, "left", (80, 200), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
                    flagl = 1
                    flag = 2
        framenumber += 1
        cv_imwrite(lane_path+file, frame)
    cap.release()
    cv.destroyAllWindows()
    return flag

def detection():
    if not os.path.getsize(video_pos_path):
        return
    sub_Flag=[]
    final_Flag=[]
    path_list = os.listdir(result_path)
    path_list.sort(key=lambda x: int(x.split('_')[0]))
    data = pd.read_csv(fps_path, sep='\t', header=None)
    data_name = pd.read_csv(test_path, sep='\t', header=None)

    i=0
    j=0
    m=0
    for file in path_list:
        if file == (format(str(m + 1)) + '_pred.jpg'):
            fps = data.loc[j][0]
            filename = data_name.loc[m][0]
            filename = filename.split('/')[4]
            file_path = result_path + file
            flag = LaneLines(file_path, filename)
            sub_Flag.append(flag)
            i = i+1
            m = m+1
            if i==fps:
                sub_Flag_5 = sub_Flag[:5]
                final_Flag.append(max(sub_Flag_5, key=sub_Flag_5.count))
                j=j+1
                i=0
                sub_Flag=[]
            if j == len(data):
                break
        else:
            continue
    return final_Flag