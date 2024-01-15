import os
import clip
import torch
import cv2 as cv
from PIL import Image

### 0 正常 ### 1 疲劳 ### 2 吸烟 ### 3 接听电话 ### 4 墨镜
def Behavior(Test_path):
    if Test_path[0][-3:] == 'jpg' :
        Flag_1 = [0]*len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model, preprocess = clip.load('ViT-L/14', device)#RN50x64
        model, preprocess = clip.load('./ViT-L-14.pt', device)#./RN50x64.pt
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.", "A person is smoking.",
             "A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                flag=0
                try:
                    img = Image.open(Test_path[i])
                except (IOError, SyntaxError) as e:
                    Flag_1[i]=1
                    continue
                cap = cv.VideoCapture(Test_path[i])
                if not cap.isOpened():
                    #print("视频无法打开")
                    Flag_1[i] = 1
                    continue
                c = 0
                fps = cap.get(cv.CAP_PROP_FPS)  # 获取帧率
                while True:
                    # 捕获视频帧，返回ret，frame
                    # ret的true与false反应是否捕获成功，frame是画面
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if ret:
                        if (c % round(fps) == 0):  # 每隔fps帧进行操作
                            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                            image_input = preprocess(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model.encode_image(image_input)
                                text_features = model.encode_text(text_inputs)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                            a=similarity[0].tolist()
                            re_a=sorted(a,reverse=True)
                            index=a.index(max(a))
                            index_2=a.index(re_a[1])#认定Top2
                            if (index in [1, 2]) or (index_2 in [1, 2]):
                                flag = 1
                            elif index in [0, 5]:
                                flag = 0
                            elif index == 3:
                                flag=2
                            elif index == 4:
                                flag=3
                            Flag_1[i]=flag
                        cv.waitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)
                        c = c + 1
                    else:
                        break
            else:
                Flag_1[i]=1

        Flag_2 = [0]*len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model, preprocess = clip.load('RN50x64', device)  # 闭眼判定的鲁棒性
        model, preprocess = clip.load('./RN50x64.pt', device)  # 闭眼判定的鲁棒性
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.","A person is smoking.",
             "A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                flag = 0
                try:
                    img = Image.open(Test_path[i])
                except (IOError, SyntaxError) as e:
                    Flag_2[i]=1
                    continue
                cap = cv.VideoCapture(Test_path[i])
                if not cap.isOpened():
                    # print("视频无法打开")
                    Flag_2[i] = 1
                    continue
                # print('WIDTH', cap.get(3))
                # print('HEIGHT', cap.get(4))
                c = 0
                fps = cap.get(cv.CAP_PROP_FPS)  # 获取帧率
                while True:
                    # 捕获视频帧，返回ret，frame
                    # ret的true与false反应是否捕获成功，frame是画面
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if ret:
                        if (c % round(fps) == 0):  # 每隔fps帧进行操作
                            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                            image_input = preprocess(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model.encode_image(image_input)
                                text_features = model.encode_text(text_inputs)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                            a = similarity[0].tolist()
                            re_a = sorted(a, reverse=True)
                            index = a.index(max(a))
                            index_2 = a.index(re_a[1])  # 认定Top2
                            if (index in [1, 2]) or (index_2 in [1, 2]):
                                flag = 1
                            elif index in [0, 5]:
                                flag = 0
                            elif index == 3:
                                flag = 2
                            elif index == 4:
                                flag = 3
                            Flag_2[i]=flag
                        cv.waitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)
                        c = c + 1
                    else:
                        break
            else:
                Flag_2[i] = 1

        Flag=[0]*len(Test_path)
        for i in range(len(Flag_1)):
            Flag[i]=Flag_1[i]
            if Flag_2[i]==1:
                Flag[i]=1

    if Test_path[0][-3:] == 'mp4':
        Flag_1 = [0] * len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dir = []
        # model, preprocess = clip.load('ViT-L/14', device)  # RN50x64
        model, preprocess = clip.load('./ViT-L-14.pt', device)  # ./RN50x64.pt
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.", "A person is smoking.",
             "A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
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
                t_flag = []  # 存储每帧的结果，输出为最多的类
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
                    if (index in [1, 2]) or (index_2 in [1, 2]):
                        flag = 1
                    elif index in [0, 5]:
                        flag = 0
                    elif index == 3:
                        flag = 2
                    elif index == 4:
                        flag = 3
                    t_flag.append(flag)
            if 1 in t_flag:
                Flag_1[i] = 1
            else:
                Flag_1[i] = max(set(t_flag), key=t_flag.count)

        Flag_2 = [0] * len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dir = []
        # model, preprocess = clip.load('RN50x64', device)  # RN50x64
        model, preprocess = clip.load('./RN50x64.pt', device)  # ./RN50x64.pt
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.",
             "A person is smoking.","A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
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
                t_flag = []  # 存储每帧的结果，输出为最多的类
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
                    if (index in [1, 2]) or (index_2 in [1, 2]):
                        flag = 1
                    elif index in [0, 5]:
                        flag = 0
                    elif index == 3:
                        flag = 2
                    elif index == 4:
                        flag = 3
                    t_flag.append(flag)
            if 1 in t_flag:
                Flag_2[i] = 1
            else:
                Flag_2[i] = max(set(t_flag), key=t_flag.count)

        Flag = [0] * len(Test_path)
        for i in range(len(Flag_1)):
            Flag[i] = Flag_1[i]
            if Flag_2[i] == 1:
                Flag[i] = 1
    return Flag