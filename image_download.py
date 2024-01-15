
from urllib import request
import ssl
import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor


def download_image(url, file_name, save_directory):

    proxy_support = request.ProxyHandler()
    context = ssl._create_unverified_context()

    opener = request.build_opener(proxy_support)
    request.install_opener(opener)
    try:
        # 使用urllib发起请求
        response = request.urlopen(url, context=context)

        if response.getcode() == 200:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_path = os.path.join(save_directory, file_name)
            with open(save_path, "wb") as file:
                file.write(response.read())
            # print(f"Image downloaded and saved as {file_name}")
            return 0
        else:
            print("Failed to download the image. Status code:",
                  response.status_code)
            return -1
    except Exception as e:
        print(e)
        return -1


def download_image_wrapper(row):
    plate = row['plate'][1:]
    alarm_time = pd.to_datetime(row['alarm_begin_time'])
    time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
    t_type = row['exception_type']

    video_url = row["video_url"][0]
    picture_url = row["picture_url"][0]

    video_file_name = f"{plate}_{time_str}_{t_type}.mp4"
    pic_file_name = f"{plate}_{time_str}_{t_type}.jpg"

    # 下载视频
    error_message = download_image(video_url, video_file_name, "./video")
    if error_message != 0:
        print(f"Failed to download video from {video_url}")
    else:
        print(f"Successfully downloaded {video_file_name}")
    # 下载图片
    error_message = download_image(picture_url, pic_file_name, "./picture")
    if error_message != 0:
        print(f"Failed to download picture from {picture_url}")
    else:
        print(f"Successfully downloaded {pic_file_name}")


def image_thread_pool_executor(df):
    with ThreadPoolExecutor(max_workers=2) as executor:
        _ = [executor.submit(download_image_wrapper, row)
             for index, row in df.iterrows()]
