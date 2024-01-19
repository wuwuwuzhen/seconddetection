import os
import pandas as pd
import xlwt
import lanelines
from CLIP_distract import Cdistract
from driver_behavior import Behavior
from pc_detection import pedestrian_collision
import lanelines.Lanelines
import lanelines.video_Lanelines
import torch
import time
from fv_detection import vehicle_collision
import logging

def is_repet (plate_type, Time, t_plate_type, alarm_time):
    t1 = pd.Timestamp(alarm_time)
    for i in range(len(plate_type)):
        t2 = pd.Timestamp(Time[i])
        t_dif = abs(t1 - t2).total_seconds()
        if plate_type[i]==t_plate_type and t_dif<=30:#同车同类型相隔30s以内的事件
            time_str = Time[i].strftime('%Y%m%d_%H%M%S')
            return 1,f"{plate_type[i]}_{time_str}"#返回重复以及车牌_类型_时间
    return 0,0

# if __name__ == '__main__':
def sample_selection(df):
    pid = os.getpid() 
    try:
        home_path = os.getcwd()
        # path = home_path+'/Input.xlsx'
        # path = home_path + "/Input.xlsx"
        # df = pd.read_excel(path)
        exception_type = df['exception_type'].tolist()
        lane_departure = [];
        behavior = [];
        distance = [];
        distracted = [];
        pedestrian=[]
        Other = []  # fatigue 包括疲劳、吸烟、接打电话
        for i in range(len(exception_type)):
            if int(exception_type[i]) in [3, 4, 5, 6]:
                lane_departure.append(i)
                continue
            if int(exception_type[i]) in [14, 15, 18, 19]:
                behavior.append(i)
                continue
            if int(exception_type[i]) in [1, 2, 7, 8]:
                distance.append(i)
                continue
            if int(exception_type[i]) in [16, 17]:
                distracted.append(i)
                continue
            if int(exception_type[i]) in [9, 10]:
                pedestrian.append(i)
            else:
                Other.append(i)
        # 将样本按类别规整便于调用模型
        lane_departure_samples = df.iloc[lane_departure].copy()  # 复制到新的Dataframe
        distance_samples = df.iloc[distance].copy()
        behavior_samples = df.iloc[behavior].copy()
        distracted_samples = df.iloc[distracted].copy()
        pedestrian_samples = df.iloc[pedestrian].copy()
        Other_samples = df.iloc[Other].copy()

        Second_det_distracted = [0] * len(distracted_samples);
        Merge_dis_distracted = [''] * len(distracted_samples)
        Second_det_lane_departure = [0] * len(lane_departure_samples);
        Merge_dis_lane_departure = [''] * len(lane_departure_samples)
        Second_det_distance = [0] * len(distance_samples);
        Merge_dis_distance = [''] * len(distance_samples)
        Second_det_behavior = [0] * len(behavior_samples);
        Merge_dis_behavior = [''] * len(behavior_samples)
        Second_det_pedestrian = [0] * len(pedestrian_samples);
        Merge_dis_pedestrian = [''] * len(pedestrian_samples)
        Second_det_Other = [4] * len(Other_samples);
        Merge_dis_Other = [''] * len(Other_samples)  # 无法判定的其他类别设置为1 仅筛除复报告情况

        picture_path = home_path + '/picture/'
        video_path=home_path+'/video/'
        video_det_distracted=[0]*len(distracted_samples);
        video_det_lane_departure=[0]*len(lane_departure_samples);
        video_det_distance=[0]*len(distance_samples);
        video_det_behavior=[0]*len(behavior_samples);
        video_det_pedestrian = [0] * len(pedestrian_samples);
        logging.info(f'PID {pid}|Succeed in classifying')
    except Exception as e:
        logging.error(f'PID {pid}|An error occurred: Failed to classify')


# 跟车距离检测
    try:
        t1 = time.time()
        logging.info(f"PID {pid}|len(distance_samples): {len(distance_samples)}")
        Test_path = [];Test_path_video = []
        if len(distance_samples) > 0:
            for index, row in distance_samples.iterrows():
                plate = row['plate'][1:]
                t_type = row['exception_type']
                t_plate_type = f"{plate}_{t_type}"
                alarm_time = pd.to_datetime(row['alarm_begin_time'])
                time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
                file_name = f"{plate}_{time_str}_{t_type}.jpg"
                test_path = picture_path + file_name
                Test_path.append(test_path)
                file_name_video = f"{plate}_{time_str}_{t_type}.mp4"
                test_path_video = video_path + file_name_video
                Test_path_video.append(test_path_video)
            Second_det_distance = vehicle_collision(Test_path)  # CLIP判定
            Second_det_distance_video = vehicle_collision(Test_path_video)  # CLIP判定
            # 视频检测
            for i in range(len(Second_det_distance)):
                if Second_det_distance_video[i] == 1 or Second_det_distance[i]==1:
                    Second_det_distance[i] = 1
                elif Second_det_distance_video[i] == 4 and Second_det_distance[i]==2:
                    Second_det_distance[i] = 2
                elif Second_det_distance_video[i] == 2 and Second_det_distance[i]==4:
                    Second_det_distance[i] = 2
                elif Second_det_distance_video[i] == 4 and Second_det_distance[i]==4:
                    Second_det_distance[i] = 3
            a = 0
            plate_type = [];
            Time = []  # 记录事件的车牌号、类型、时间
            for index, row in distance_samples.iterrows():
                plate = row['plate'][1:]
                t_type = row['exception_type']
                t_plate_type = f"{plate}_{t_type}"
                alarm_time = pd.to_datetime(row['alarm_begin_time'])
                repet, repet_label = is_repet(plate_type, Time, t_plate_type, alarm_time)
                if (Second_det_distance[a]==1) and (not repet):  # 二次检测通过且未重复
                    plate_type.append(t_plate_type)
                    Time.append(alarm_time)
                    a += 1
                    continue
                if (Second_det_distance[a]==1) and repet:  # 二次检测通过且重复
                    Merge_dis_distance[a] = repet_label
                    logging.info(f"PID {pid}|index {index}|Merge_dis_distance: {Merge_dis_distance[a]}")
                    a += 1
                    continue
                a += 1
            a = 0
            for index, row in distance_samples.iterrows():
                plate = row['plate'][1:]
                t_type = row['exception_type']
                t_plate_type = f"{plate}_{t_type}"
                alarm_time = pd.to_datetime(row['alarm_begin_time'])
                time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
                if (t_plate_type + '_' + time_str) in Merge_dis_distance:
                    Merge_dis_distance[a] = t_plate_type + '_' + time_str
                a += 1
            distance_samples['checkStatus'] = Second_det_distance
            distance_samples['mergeUUId'] = Merge_dis_distance
            logging.info(f"PID {pid}|index {index}|Second_det_distance: {Second_det_distance}")
            logging.info(f"PID {pid}|index {index}|Merge_dis_distance: {Merge_dis_distance}")
            t2 = time.time()
            # print(t2-t1)
        logging.info(f'PID {pid}|Succeed in distance detection')
    except Exception as e:
        logging.error(f'PID {pid}|An error occurred: Distance detection failed')
#
# #分心行为检测
    logging.info(f"len(distracted_samples): {len(distracted_samples)}")
    #print(len(distracted_samples))
    t1=time.time()
    Test_path=[];Test_path_video=[]
    if len(distracted_samples)>0:
        for index, row in distracted_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            file_name = f"{plate}_{time_str}_{t_type}.jpg"
            test_path = picture_path + file_name
            Test_path.append(test_path)
            file_name_video = f"{plate}_{time_str}_{t_type}.mp4"
            test_path_video=video_path+file_name_video
            Test_path_video.append(test_path_video)
        Second_det_distracted_2=Cdistract(Test_path)#CLIP判定
        Second_det_distracted_2_video = Cdistract(Test_path_video)  # CLIP判定
        for i in range(len(Second_det_distracted_2)):
            if Second_det_distracted_2[i] == 1 or Second_det_distracted_2_video[i]==1:
                Second_det_distracted_2[i] = 1
            elif Second_det_distracted_2[i] == 4 and Second_det_distracted_2_video[i]==2:
                Second_det_distracted_2[i] = 2
            elif Second_det_distracted_2[i] == 2 and Second_det_distracted_2_video[i]==4:
                Second_det_distracted_2[i] = 2
            elif Second_det_distracted_2[i] == 4 and Second_det_distracted_2_video[i]==4:
                Second_det_distracted_2[i] = 3

        a = 0
        plate_type = [];Time = []  # 记录事件的车牌号、类型、时间
        for index, row in distracted_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            repet, repet_label = is_repet(plate_type, Time, t_plate_type, alarm_time)
            if (Second_det_distracted_2[a]==1) and (not repet):  # 二次检测通过且未重复
                plate_type.append(t_plate_type)
                Time.append(alarm_time)
                a += 1
                continue
            if (Second_det_distracted_2[a]==1) and repet:  # 二次检测通过且重复
                Merge_dis_distracted[a] = repet_label
                a += 1
                continue
            a += 1

        a = 0
        for index, row in distracted_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            if (t_plate_type + '_' + time_str) in Merge_dis_distracted:
                Merge_dis_distracted[a] = t_plate_type + '_' + time_str
            a += 1
        distracted_samples['checkStatus']=Second_det_distracted_2
        distracted_samples['mergeUUId']=Merge_dis_distracted
        logging.info(f"PID {pid}|Second_det_distracted_2: {Second_det_distracted_2}")
        logging.info(f"PID {pid}|Merge_dis_distracted: {Merge_dis_distracted}")
        t2=time.time()
        # print(t2-t1)
    logging.info(f'PID {pid}|Succeed in distracted detection')
#
# # #行人碰撞检测
    # print(len(pedestrian_samples))
    logging.info(f"PID {pid}|len(pedestrian_samples): {len(pedestrian_samples)}")
    Test_path = [];Test_path_video = []
    t1 = time.time()
    if len(pedestrian_samples) > 0:
        for index, row in pedestrian_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            file_name = f"{plate}_{time_str}_{t_type}.jpg"
            test_path = picture_path + file_name
            Test_path.append(test_path)
            file_name_video = f"{plate}_{time_str}_{t_type}.mp4"
            test_path_video = video_path + file_name_video
            Test_path_video.append(test_path_video)
        Second_det_pedestrian = pedestrian_collision(Test_path)  # CLIP判定
        Second_det_pedestrian_video = pedestrian_collision(Test_path_video)  # CLIP判定

        for i in range(len(Second_det_pedestrian)):
            if Second_det_pedestrian[i] == 1 or Second_det_pedestrian_video[i]==1:
                Second_det_pedestrian[i] = 1
            elif Second_det_pedestrian[i] == 4 and Second_det_pedestrian_video[i]==2:
                Second_det_pedestrian[i] = 2
            elif Second_det_pedestrian[i] == 2 and Second_det_pedestrian_video[i]==4:
                Second_det_pedestrian[i] = 2
            elif Second_det_pedestrian[i] == 4 and Second_det_pedestrian_video[i]==4:
                Second_det_pedestrian[i] = 3
        a = 0
        plate_type = [];
        Time = []  # 记录事件的车牌号、类型、时间
        for index, row in pedestrian_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            repet, repet_label = is_repet(plate_type, Time, t_plate_type, alarm_time)
            if (Second_det_pedestrian[a]==1 )and (not repet):  # 二次检测通过且未重复
                plate_type.append(t_plate_type)
                Time.append(alarm_time)
                a += 1
                continue
            if (Second_det_pedestrian[a]==1) and repet:  # 二次检测通过且重复
                Merge_dis_pedestrian[a] = repet_label
                a += 1
                continue
            a += 1

        a = 0
        for index, row in pedestrian_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            if (t_plate_type + '_' + time_str) in Merge_dis_pedestrian:
                Merge_dis_pedestrian[a] = t_plate_type + '_' + time_str
            a += 1
        pedestrian_samples['checkStatus'] = Second_det_pedestrian
        pedestrian_samples['mergeUUId'] = Merge_dis_pedestrian
        logging.info(f"PID {pid}|Second_det_pedestrian: {Second_det_pedestrian}")
        logging.info(f"PID {pid}|Merge_dis_pedestrian: {Merge_dis_pedestrian}")
        t2 = time.time()
        # print(t2-t1)
    logging.info(f'PID {pid}|Succeed in pedestrian detection')
# #驾驶行为检测
    # print(len(behavior_samples))
    logging.info(f"PID {pid}|len(behavior_samples): {len(behavior_samples)}")
    t1=time.time()
    Test_path=[]
    Test_path_video = []
    if len(behavior_samples)>0:
        for index, row in behavior_samples.iterrows():#调用大模型前一次性读取所有路径
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            file_name = f"{plate}_{time_str}_{t_type}.jpg"
            test_path = picture_path + file_name
            Test_path.append(test_path)
            file_name_video = f"{plate}_{time_str}_{t_type}.mp4"
            test_path_video=video_path+file_name_video
            Test_path_video.append(test_path_video)
        Second_det_behavior=Behavior(Test_path)
        Second_det_behavior_video=Behavior(Test_path_video)

        a=0
        plate_type=[];Time=[]#记录事件的车牌号、类型、时间
        for index, row in behavior_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            repet, repet_label = is_repet(plate_type, Time, t_plate_type, alarm_time)
            t_type=int(t_type)
            if (((Second_det_behavior[a]==1 or Second_det_behavior_video[a]==1)and t_type in [14,15])
                    or ((Second_det_behavior[a]==2 or Second_det_behavior_video[a]==2)and t_type==19)
                    or ((Second_det_behavior[a]==3 or Second_det_behavior_video[a]==3) and t_type==18)):
                Second_det_behavior[a]=1
            else:
                Second_det_behavior[a] = 2

            if (Second_det_behavior[a]==1 )and (not repet):  # 二次检测通过且未重复
                plate_type.append(t_plate_type)
                Time.append(alarm_time)
                a += 1
                continue

            if (Second_det_behavior[a]==1) and repet:  # 二次检测通过且重复
                Merge_dis_behavior[a] = repet_label
                a += 1
                continue
            a += 1
        #logging.info(f"behavior: {Second_det_behavior}")
        for i in range(len(Second_det_behavior)):
            if (not os.path.exists(Test_path[i])) and (not os.path.exists(Test_path_video[i])):  # 是视频和图像都没有则标注为3
                Second_det_behavior[i] = 3

        a = 0
        for index, row in behavior_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            if (t_plate_type + '_' + time_str) in Merge_dis_behavior:
                Merge_dis_behavior[a] = t_plate_type + '_' + time_str
            a += 1
        behavior_samples['checkStatus']=Second_det_behavior
        behavior_samples['mergeUUId']=Merge_dis_behavior
        logging.info(f"PID {pid}|Second_det_behavior: {Second_det_behavior}")
        logging.info(f"PID {pid}|Merge_dis_behavior: {Merge_dis_behavior}")
        t2=time.time()
    logging.info(f'PID {pid}|Succeed in behavior detection')

#道路偏离检测
    # print(len(lane_departure_samples))
    logging.info(f"PID {pid}|len(lane_departure_samples): {len(lane_departure_samples)}")
    t1 = time.time()
    Test_path = [];
    Test_path_2 = []
    Test_path_video = [];
    Test_path_video_2 = []
    if len(lane_departure_samples) > 0:
        for index, row in lane_departure_samples.iterrows():  # 调用大模型前一次性读取所有路径
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            file_name = f"{plate}_{time_str}_{t_type}.jpg"
            test_path = picture_path + file_name
            Test_path_2.append(test_path)
            Test_path.append([test_path, file_name])
            file_name_video = f"{plate}_{time_str}_{t_type}.mp4"
            test_path_video = video_path + file_name_video
            Test_path_video.append([test_path_video, file_name_video])
            Test_path_video_2.append(test_path_video)
        Second_det_lane_departure = lanelines.Lanelines.main(Test_path)
        Second_det_lane_departure_video = lanelines.video_Lanelines.main(Test_path_video)
        length = len(lane_departure_samples)
        picture_pos_path = home_path + "/lanelines/picture_position.txt"
        video_pos_path = home_path + "/lanelines/video_position.txt"
        logging.info(f"PID {pid}|Second_det_lane_departure|picture_pos_path:{picture_pos_path}")
        result = [4] * length

        if not os.path.getsize(video_pos_path) and os.path.getsize(picture_pos_path) != 0:  ###没有视频
            data_picture = pd.read_csv(picture_pos_path, sep='\t', header=None)
            #logging.info(f"lane_departure: {Second_det_lane_departure}")
            for i in range(len(data_picture)):
                picture_position = data_picture.loc[i][0]
                picture_result = Second_det_lane_departure[i]
                result[picture_position] = picture_result
            Second_det_lane_departure = result
            #logging.info(f"lane_departure: {result}")

            # Second_det_lane_departure_video = Second_det_lane_departure.copy()
            Second_det_lane_departure_video =Second_det_lane_departure.copy()
        elif not os.path.getsize(picture_pos_path) and os.path.getsize(video_pos_path) != 0:  ###没有图片
            data_video = pd.read_csv(video_pos_path, sep='\t', header=None)
            for i in range(len(data_video)):
                video_position = data_video.loc[i][0]
                video_result = Second_det_lane_departure_video[i]
                result[video_position] = video_result
            Second_det_lane_departure_video = result
            Second_det_lane_departure = Second_det_lane_departure_video.copy()

        elif not os.path.getsize(picture_pos_path) and not os.path.getsize(video_pos_path):  ###没有图片视频
            result = [3] * length
            Second_det_lane_departure_video = result
            Second_det_lane_departure = result

        else:  ###有视频图片
            data_picture = pd.read_csv(picture_pos_path, sep='\t', header=None)
            data_video = pd.read_csv(video_pos_path, sep='\t', header=None)
            for i in range(len(data_picture)):
                picture_position = data_picture.loc[i][0]
                picture_result = Second_det_lane_departure[i]
                result[picture_position] = picture_result
            for i in range(len(data_video)):
                video_position = data_video.loc[i][0]
                video_result = Second_det_lane_departure_video[i]
                result[video_position] = video_result
            Second_det_lane_departure_video = result
            Second_det_lane_departure = result

        #logging.info(f"lane_departure: {Second_det_lane_departure}")
        a = 0
        plate_type = [];
        Time = []  # 记录事件的车牌号、类型、时间

        for index, row in lane_departure_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            repet, repet_label = is_repet(plate_type, Time, t_plate_type, alarm_time)
            t_type=int(t_type)
            # logging.info(f"t_type: {t_type}")
            # logging.info(f"Second_det_lane_departure: {Second_det_lane_departure[a]}")
            # logging.info(f"Second_det_lane_departure_video: {Second_det_lane_departure_video[a]}")
            if (((Second_det_lane_departure[a] == 1 or Second_det_lane_departure_video[a] == 1) and t_type in [5,6])
                    or ((Second_det_lane_departure[a] == 2 or Second_det_lane_departure_video[a] == 2) and t_type in [3, 4])):
                Second_det_lane_departure[a] = 1
                #logging.info(f"good")
            elif ((Second_det_lane_departure[a] == 5 or Second_det_lane_departure_video[a] == 5) and t_type in [3,4,5,6]):
                Second_det_lane_departure[a] = 2  ###居中检测
                #logging.info(f"bad")
            elif (((Second_det_lane_departure[a] == 1 or Second_det_lane_departure_video[a] == 1) and t_type in [3,4])
                  or (( Second_det_lane_departure[a] == 2 or Second_det_lane_departure_video[a] == 2) and t_type in [5, 6])):
                Second_det_lane_departure[a] = 2  ###偏离方向相反
                #logging.info(f"bad")
            # else:
            #     Second_det_lane_departure[a] = 1

            if Second_det_lane_departure[a] == 1 and (not repet):  # 二次检测通过且未重复
                plate_type.append(t_plate_type)
                Time.append(alarm_time)
                a += 1
                continue

            if Second_det_lane_departure[a] == 1 and repet:  # 二次检测通过且重复
                Merge_dis_lane_departure[a] = repet_label
                # Merge_dis_lane_departure[a]=10
                a += 1
                continue
            a += 1
        #logging.info(f"Second_det_lane_departure: {Second_det_lane_departure}")
        for i in range(len(Second_det_lane_departure)):
            if Second_det_lane_departure[i] != 1:
                Second_det_lane_departure[i] = 2
            if (not os.path.exists(Test_path_2[i])) and (
            not os.path.exists(Test_path_video_2[i])):  # 是视频和图像都没有则标注为3
                Second_det_lane_departure[i] = 3

        a = 0
        for index, row in lane_departure_samples.iterrows():
            plate = row['plate'][1:]
            t_type = row['exception_type']
            t_plate_type = f"{plate}_{t_type}"
            alarm_time = pd.to_datetime(row['alarm_begin_time'])
            time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
            if (t_plate_type + '_' + time_str) in Merge_dis_lane_departure:
                Merge_dis_lane_departure[a] = t_plate_type + '_' + time_str
            a += 1
        lane_departure_samples['checkStatus'] = Second_det_lane_departure
        lane_departure_samples['mergeUUId'] = Merge_dis_lane_departure

        logging.info(f"PID {pid}|Second_det_lane_departure: {Second_det_lane_departure}")
        logging.info(f"PID {pid}|Merge_dis_lane_departure: {Merge_dis_lane_departure}")
        t2 = time.time()
        # print(t2 - t1)
    logging.info(f'PID {pid}|Succeed in lane_departure detection')

    # 合并Others的重复项
    # print(len(Other_samples))
    t1 = time.time()
    a = 0
    plate_type = [];
    Time = []  # 记录事件的车牌号、类型、时间
    for index, row in Other_samples.iterrows():
        plate = row['plate'][1:]
        t_type = row['exception_type']
        t_plate_type = f"{plate}_{t_type}"
        alarm_time = pd.to_datetime(row['alarm_begin_time'])
        time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
        repet, repet_label = is_repet(plate_type, Time, t_plate_type, alarm_time)
        if (not repet):
            plate_type.append(t_plate_type)
            Time.append(alarm_time)
            a += 1
            continue
        if repet:
            Merge_dis_Other[a] = repet_label
            a += 1
            continue
        else:
            a += 1
            continue
    a = 0
    for index, row in Other_samples.iterrows():
        plate = row['plate'][1:]
        t_type = row['exception_type']
        t_plate_type = f"{plate}_{t_type}"
        alarm_time = pd.to_datetime(row['alarm_begin_time'])
        time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
        if (t_plate_type + '_' + time_str) in Merge_dis_Other:
            Merge_dis_Other[a] = t_plate_type + '_' + time_str
        a += 1
    Other_samples['checkStatus'] = Second_det_Other
    Other_samples['mergeUUId'] = Merge_dis_Other
    logging.info(f"PID {pid}|Second_det_Other: {Second_det_Other}")
    logging.info(f"PID {pid}|Merge_dis_Other: {Merge_dis_Other}")
    t2 = time.time()
    # print(t2 - t1)

    # outpath = home_path+'/Output_1_16_cpu.xlsx'
    # outpath = home_path+'/test_0117.xlsx'
    df_combined = pd.concat([lane_departure_samples, distance_samples, behavior_samples, distracted_samples, pedestrian_samples,Other_samples], axis=0)
    # df_combined.to_excel(outpath, index=False)
    return df_combined