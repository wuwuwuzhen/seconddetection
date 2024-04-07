
from urllib import request
import ssl
import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
import logging
import config
import pandas as pd

def download_file(url, file_name, save_directory):

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
            return 0
        else:
            return -1
    except Exception as e:
        logging.error(f"[download_file] exception:{e}")
        return -1


def download_image_wrapper(req, temp_file_path):
    plate = req['plate'][1:]
    alarm_time = pd.to_datetime(req['alarm_begin_time'])
    time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
    t_type = req['exception_type']

    # 下载图片
    picture_urls = transfer_url_list(req.get("picture_url", None))
    if picture_urls is None:
        logging.warning("[download_image_wrapper] no picture url")
    else:
        for picture_url in picture_urls:
            pic_file_name = f"{plate}_{time_str}_{t_type}.jpg"
            error_message = download_file(
                picture_url, pic_file_name, os.path.join(temp_file_path, 'picture'))
            if error_message != 0:
                logging.warning(f"[download_image_wrapper] failed to download picture from {picture_url}")
            else: 
                logging.debug(f"[download_image_wrapper] succeeded in downloading picture {pic_file_name}")
    # 下载视频
    video_urls = transfer_url_list(req.get("video_url", None))
    if video_urls is not None:
        for video_url in video_urls:
            video_file_name = f"{plate}_{time_str}_{t_type}.mp4"
            error_message = download_file(
                video_url, video_file_name, os.path.join(temp_file_path,"video"))
            if error_message != 0:
                logging.info(f"[download_image_wrapper] failed to download video from {video_url}")


def transfer_url_list(urls) -> list:
    try:
        if urls is None :
            return None
        if type(urls) == list:
            return urls
        if type(urls) == str:
            return json.loads(urls)
    except:
        return None


def image_thread_pool_executor(df):
    with ThreadPoolExecutor(max_workers=2) as executor:
        _ = [executor.submit(download_image_wrapper, row)
             for index, row in df.iterrows()]


def download_image_from_req(req_list: list) -> None:
    for req in req_list:
        download_image_wrapper(req, config.root_path)


def df_row_download(row, temp_file_path):
    plate = row['plate'][1:]
    alarm_time = pd.to_datetime(row['alarm_begin_time'])
    time_str = alarm_time.strftime('%Y%m%d_%H%M%S')
    t_type = row['exception_type']

    if pd.notna(row['video_url']) and len(row['video_url']) != 0:

        video_url = row["video_url"][2:-2]
        # 下载视频
        video_file_name = f"{plate}_{time_str}_{t_type}.mp4"
        video_file_path = os.path.join(temp_file_path, "video")
        error_message = download_file(video_url, video_file_name, video_file_path)
        if error_message != 0:
            print(f"Failed to download video from {video_url}")
        else:
            print(f"Successfully downloaded {video_file_name}")
    
    picture_url = row["picture_url"][2:-2]
    pic_file_name = f"{plate}_{time_str}_{t_type}.jpg"
    # 下载图片
    picture_file_path = os.path.join(temp_file_path, "picture")
    error_message = download_file(picture_url, pic_file_name, picture_file_path)
    if error_message != 0:
        print(f"Failed to download picture from {picture_url}")
    else:
        print(f"Successfully downloaded {pic_file_name}")
