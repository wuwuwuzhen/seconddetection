import os
import clip
import torch
import cv2 as cv
from PIL import Image
import logging
import config

### 0 正常 ### 1 疲劳 ### 2 吸烟 ### 3 接听电话 ### 4 墨镜
def Behavior(Test_path):
    # print('fatigue')
    if Test_path[0][-3:] == 'jpg' :
        Flag_1 = [0]*len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(config.vit_l_14_path, device)
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.", "A person is smoking.",
             "A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                #logging.info(f"path: {Test_path[i]}")
                flag=0
                try:
                    img = Image.open(Test_path[i])
                except (IOError, SyntaxError) as e:
                    Flag_1[i]=4
                    continue
                cap = cv.VideoCapture(Test_path[i])
                if not cap.isOpened():
                    #print("视频无法打开")
                    Flag_1[i] = 4
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
                Flag_1[i]=4

        Flag_2 = [0]*len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(config.rn_50_x64_path, device)  
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.","A person is smoking.",
             "A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):
            if os.path.exists(Test_path[i]):
                #logging.info(f"path: {Test_path[i]}")
                flag = 0
                try:
                    img = Image.open(Test_path[i])
                except (IOError, SyntaxError) as e:
                    Flag_2[i]=4
                    continue
                cap = cv.VideoCapture(Test_path[i])
                if not cap.isOpened():
                    # print("视频无法打开")
                    Flag_2[i] =4
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
                Flag_2[i] = 4

        Flag=[0]*len(Test_path)
        for i in range(len(Flag_1)):
            Flag[i]=Flag_1[i]
            if Flag_2[i]==1:
                Flag[i]=1



    if Test_path[0][-3:] == 'mp4':
        Flag_1 = [0] * len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dir = []
        image_dir_list = []
        image_dir_half = []
        model, preprocess = clip.load(config.vit_l_14_path, device)  
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.", "A person is smoking.",
             "A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):  # 保存文件路径
            if os.path.exists(Test_path[i]):
                #logging.info(f"path: {Test_path[i]}")
                try:
                    cap = cv.VideoCapture(Test_path[i])
                    if not cap.isOpened():
                        image_dir.append(0)
                        image_dir_list.append(0)
                        image_dir_half.append(0)
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
                    image_dir_list.append(0)
                    image_dir_half.append(0)
                    continue
                video = cv.VideoCapture(Test_path[i])
                # print(Test_path[i], 1)
                total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
                if 8 > total_frames > 1:
                    frames_interval = total_frames // (total_frames - 1)
                elif total_frames <= 1:
                    frames_interval = 1
                    # continue
                else:
                    frames_interval = total_frames // 8  # 等间距取8张
                fps = video.get(cv.CAP_PROP_FPS)
                half_interval = fps // 2  # 0.5s的帧
                frames = []  # 用于保存帧的列表
                frames_list = []  # 保存8张对应正帧的索引
                # print(frames_interval, total_frames, frames_interval)
                for j in range(frames_interval, total_frames, frames_interval):
                    video.set(cv.CAP_PROP_POS_FRAMES, j)
                    # 记录8张索引的帧序列
                    frames_list.append(j)
                    ret, frame = video.read()
                    if ret:
                        frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                video.release()
                cv.destroyAllWindows()
                if len(frames) == 0:
                    image_dir.append(1)
                else:
                    image_dir.append(frames)
                image_dir_list.append(frames_list)
                image_dir_half.append(half_interval)
            else:
                image_dir.append(0)
                image_dir_list.append(0)
                image_dir_half.append(0)
            print(len(image_dir_list))
        for i in range(len(image_dir)):
            frames_list = image_dir_list[i]
            half_interval = image_dir_half[i]
            # if os.path.exists(Test_path[i]):
                # print(Test_path[i], 'max')
            if image_dir[i] == 1:
                Flag_1[i] = 1
                continue
            elif image_dir[i] == 0:
                Flag_1[i]=4
                continue
            else:
                # print(image_dir[i])
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
                # print(t_flag)
                second_flag = 0  # 标记是否相邻两张图片通过检测
                if len(image_dir[i]) in [1,2]:
                    second_flag = 1
                else:
                    for j in range(len(t_flag)):
                        t_frame = []
                        if t_flag[j] == 1:
                            video = cv.VideoCapture(Test_path[i])
                            if int(frames_list[j] - half_interval) > 0:
                                video.set(cv.CAP_PROP_POS_FRAMES, int(frames_list[j] - half_interval))
                                ret, frame = video.read()
                                if ret:
                                    t_frame.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                            if int(frames_list[j] + half_interval) < total_frames:
                                video.set(cv.CAP_PROP_POS_FRAMES, int(frames_list[j] + half_interval))
                                ret, frame = video.read()
                                if ret:
                                    t_frame.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                            ad_flag = [0, 0]
                            # print(len(t_frame))
                            video.release()
                            cv.destroyAllWindows()
                            for k in range(len(t_frame)):
                                image_input = preprocess(t_frame[k]).unsqueeze(0).to(device)
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
                                    ad_flag[k] = 1
                            if ad_flag[0] == 1 or ad_flag[1] == 1:
                                second_flag = 1
                                break

            if second_flag == 1:
                Flag_1[i] = 1
            else:
                Flag_1[i] = max(set(t_flag), key=t_flag.count)

        Flag_2 = [0] * len(Test_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dir = []
        image_dir_list = []
        image_dir_half = []
        model, preprocess = clip.load(config.rn_50_x64_path, device)  
        text_inputs = torch.cat([clip.tokenize(
            ["A person is looking straight ahead.", "A person closes eyes.", "A person is yawning.",
             "A person is smoking.","A person is using a phone.", 'A person wears sunglasses or mask'])]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        for i in range(len(Test_path)):  # 保存文件路径
            if os.path.exists(Test_path[i]):
                #logging.info(f"path: {Test_path[i]}")
                try:
                    cap = cv.VideoCapture(Test_path[i])
                    if not cap.isOpened():
                        image_dir.append(0)
                        image_dir_list.append(0)
                        image_dir_half.append(0)
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
                    image_dir_list.append(0)
                    image_dir_half.append(0)
                    continue
                # 等距抽取8帧
                video = cv.VideoCapture(Test_path[i])
                print(Test_path[i],2)
                total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
                if 8 > total_frames > 1:
                    frames_interval = total_frames // (total_frames - 1)
                elif total_frames <= 1:
                    frames_interval = 1
                    # continue
                else:
                    frames_interval = total_frames // 8  # 等间距取8张
                fps = video.get(cv.CAP_PROP_FPS)
                half_interval = fps // 2  # 0.5s的帧
                frames = []  # 用于保存帧的列表
                frames_list = []  # 保存8张对应正帧的索引
                print(frames_interval, total_frames, frames_interval)
                for j in range(frames_interval, total_frames, frames_interval):
                    video.set(cv.CAP_PROP_POS_FRAMES, j)
                    # 记录8张索引的帧序列
                    frames_list.append(j)
                    ret, frame = video.read()
                    if ret:
                        frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                video.release()
                cv.destroyAllWindows()
                if len(frames) == 0:
                    image_dir.append(1)
                else:
                    image_dir.append(frames)
                image_dir_list.append(frames_list)
                image_dir_half.append(half_interval)
            else:
                image_dir.append(0)
                image_dir_list.append(0)
                image_dir_half.append(0)

        for i in range(len(image_dir)):
            frames_list = image_dir_list[i]
            half_interval = image_dir_half[i]
            if image_dir[i] == 1:
                Flag_2[i] = 1
                continue
            elif image_dir[i] == 0:
                Flag_2[i] = 4
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
                second_flag = 0  # 标记是否相邻两张图片通过检测
                if len(image_dir[i]) in [1,2,3]:
                    second_flag = 1
                else:
                    for j in range(len(t_flag)):
                        t_frame = []
                        if t_flag[j] == 1:
                            video = cv.VideoCapture(Test_path[i])
                            if int(frames_list[j] - half_interval) > 0:
                                video.set(cv.CAP_PROP_POS_FRAMES, int(frames_list[j] - half_interval))
                                ret, frame = video.read()
                                if ret:
                                    t_frame.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                            if int(frames_list[j] + half_interval) < total_frames:
                                video.set(cv.CAP_PROP_POS_FRAMES, int(frames_list[j] + half_interval))
                                ret, frame = video.read()
                                if ret:
                                    t_frame.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
                            ad_flag = [0, 0]
                            video.release()
                            cv.destroyAllWindows()
                            for k in range(len(t_frame)):
                                image_input = preprocess(t_frame[k]).unsqueeze(0).to(device)
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
                                    ad_flag[k] = 1
                            if ad_flag[0] == 1 or ad_flag[1] == 1:
                                second_flag = 1
                                break

            if second_flag == 1:
                Flag_2[i] = 1
            else:
                Flag_2[i] = max(set(t_flag), key=t_flag.count)

        Flag = [0] * len(Test_path)
        for i in range(len(Flag_1)):
            Flag[i] = Flag_1[i]
            if Flag_2[i] == 1:
                Flag[i] = 1

    return Flag