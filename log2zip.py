import json
import pandas as pd
import os
import config
from image_download import df_row_download, download_image_wrapper
import shutil
import re
import sys

# 函数用于从包含 'req:' 的行中提取 JSON 数据
def extract_json_from_line(line):
    try:
        parts = re.split('req:|resp:', line, 1)
        part = parts[1]
        json_str = part.replace("\'", "\"").replace(
            '\"[', '[').replace(']\"', ']')
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 如果无法解析为 JSON，返回 None
        return None
    except Exception as e:
        # print(f"Error: {e}, line: {line}")
        return None



if __name__ == "__main__":
    log_time = "2024_03_24"
    temp_file_path = os.path.join(config.root_path, log_time)
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)
    
    log_file_path = os.path.join(config.log_dir, log_time + '.log')
    csv_file_path = os.path.join(temp_file_path, log_time + '.csv')

    with open(log_file_path, 'r', encoding='utf-8') as file:
        # 新建一个空的df
        req_df = pd.DataFrame()
        resp_df = pd.DataFrame()
        for line in file:
            json_data = extract_json_from_line(line)
            if json_data:
                if 'alarms' in json_data:
                    # 如果是resp
                    details = []
                    for alarm in json_data['alarms']:
                        # append进details
                        for item in alarm['alarmDetails']:
                            details.append(item)
                    temp = pd.DataFrame(details)
                    resp_df = pd.concat([resp_df, temp], ignore_index=True)
                # 如果是req
                else:
                    # 如果没有 video_url, 新增一个空的 video_url 字段
                    if 'video_url' not in json_data[0]:
                        json_data[0]['video_url'] = ""
                    temp = pd.DataFrame(json_data)
                    req_df = pd.concat([req_df, temp], ignore_index=True)
            else:
                continue
        mode = 'a' if os.path.exists(csv_file_path) else 'w'
        # 将req_df和resp_df按照id左连接，取resp的checkStatus和mergeUUId
        df = pd.merge(req_df, resp_df[['id', 'checkStatus', 'mergeUUId']],
                      left_on='id', right_on='id', how='left')
        df.to_csv(csv_file_path, mode=mode, index=False,
                  encoding='utf-8-sig', header=not os.path.exists(csv_file_path))
    
    # 筛选
    filtered_df = df[df['exception_name'].isin(config.filter_exception_name)]
    exception_str = '+'.join(config.filter_exception_name)
    filtered_df.to_excel(os.path.join(
        temp_file_path, exception_str + '.xlsx'), index=False)

    for index, row in filtered_df.iterrows():
        # df_row_download(row, temp_file_path)
        download_image_wrapper(row, temp_file_path)

    print(
        f"Total images: {len(os.listdir(os.path.join(temp_file_path, 'picture')))}")
    print(
        f"Total videos: {len(os.listdir(os.path.join(temp_file_path, 'video')))}")

    shutil.make_archive(temp_file_path, 'zip', temp_file_path)

    # 删除临时文件夹
    shutil.rmtree(temp_file_path)
