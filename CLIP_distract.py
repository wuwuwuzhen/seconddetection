import os
import clip
import torch
import cv2 as cv
from PIL import Image

### 0 正常 ### 1 疲劳 ### 2 吸烟 ### 3 接听电话 ### 4 墨镜
def Cdistract(Test_path):
    if Test_path[0][-3:] == 'jpg':
        Flag=[0]*len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model, preprocess = clip.load('ViT-L/14', device)#RN50x64
        model, preprocess = clip.load('./ViT-L-14.pt', device)#RN50x64
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking one side.", 'A person closes eyes.', "A person is looking straight ahead.",
             "A person wears a mask", "A person wears sunglasses."])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                cap = cv.VideoCapture(Test_path[i])
                try:
                    img = Image.open(Test_path[i])
                except (IOError, SyntaxError) as e:
                    Flag[i] = 1
                    continue
                if not cap.isOpened():
                    Flag[i] = 1
                    continue
                c = 0
                fps = cap.get(cv.CAP_PROP_FPS)  # 获取帧率
                while True:
                    # 捕获视频帧，返回ret，frame
                    # ret的true与false反应是否捕获成功，frame是画面
                    ret, frame = cap.read()
                    if not ret:
                        #print("视频播放完毕")
                        break
                    if ret:
                        if (c % round(fps) == 0):  # 每隔fps帧进行操作
                            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                            image_input = preprocess(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model.encode_image(image_input)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                            a=similarity[0].tolist()
                            re_a = sorted(a, reverse=True)
                            index=a.index(max(a))
                            index_2 = a.index(re_a[1])  # 认定Top2
                            if (index in [0,1]) or (index_2 in [0,1]):
                                Flag[i]=1
                            else:
                                Flag[i]=0
                    else:
                        break
            else:
                Flag[i] = 1

    if Test_path[0][-3:] == 'mp4':
        Flag = [0] * len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dir = []
        # model, preprocess = clip.load('ViT-L/14', device)  # RN50x64
        model, preprocess = clip.load('./ViT-L-14.pt', device)  # RN50x64
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking one side.", 'A person closes eyes.', "A person is looking straight ahead.",
             "A person wears a mask", "A person wears sunglasses."])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
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
                # 获取视频的帧率
                fps = video.get(cv.CAP_PROP_FPS)
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
                        frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                    frame_count += 1
                # 释放视频对象和关闭窗口
                video.release()
                cv.destroyAllWindows()
                image_dir.append(frames)
            else:
                image_dir.append(0)
        for i in range(len(image_dir)):
            if image_dir[i] == 0:
                continue
            else:
                for j in range(len(image_dir[i])):
                    image_input = preprocess(image_dir[i][j]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    a = similarity[0].tolist()
                    re_a = sorted(a, reverse=True)
                    index = a.index(max(a))
                    index_2 = a.index(re_a[1])  # 认定Top2
                    if (index in [0, 1]) or (index_2 in [0, 1]):
                        Flag[i] = 1
                        break
    return Flag
