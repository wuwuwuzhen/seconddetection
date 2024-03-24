import json
import pandas as pd
import os
import config
from image_download import df_row_download
import shutil


# 函数用于从包含 'req:' 的行中提取 JSON 数据
def extract_json_from_line(line):
    if "req" in line:
        parts = line.split("req:", 1)
        part = parts[1]
        json_str = part.replace("\'", "\"").replace(
            '\"[', '[').replace(']\"', ']')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果无法解析为 JSON，返回 None
            return None


if __name__ == "__main__":
    log_time = "2024_03_23"
    temp_file_path = os.path.join(config.root_path, log_time)
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)

    log_file_path = os.path.join(config.log_dir, log_time + '.log')
    csv_file_path = os.path.join(temp_file_path, log_time + '.csv')

    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = extract_json_from_line(line)
            if json_data:
                # 如果没有 video_url, 新增一个空的 video_url 字段
                if 'video_url' not in json_data[0]:
                    json_data[0]['video_url'] = ""
                temp = pd.DataFrame(json_data)
            # 拼接temp
            if 'df' in locals():
                df = pd.concat([df, temp], ignore_index=True)
            else:
                df = temp
        mode = 'a' if os.path.exists(csv_file_path) else 'w'
        df.to_csv(csv_file_path, mode=mode, index=False,
                  encoding='utf-8-sig', header=not os.path.exists(csv_file_path))

    # 筛选
    # df = pd.read_csv(csv_file_path)
    filtered_df = df[df['exception_name'].isin(config.filter_exception_name)]
    exception_str = '+'.join(config.filter_exception_name)
    filtered_df.to_excel(os.path.join(
        temp_file_path, exception_str + '.xlsx'), index=False)

    for index, row in filtered_df.iterrows():
        df_row_download(row, temp_file_path)

    print(
        f"Total images: {len(os.listdir(os.path.join(temp_file_path, 'picture')))}")
    print(
        f"Total videos: {len(os.listdir(os.path.join(temp_file_path, 'video')))}")

    shutil.make_archive(temp_file_path, 'zip', temp_file_path)

    # 删除临时文件夹
    shutil.rmtree(temp_file_path)
