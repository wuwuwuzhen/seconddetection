import json
import pandas as pd
import os 
import config 
from image_download import download_image_from_req

# 函数用于从包含 'req:' 的行中提取 JSON 数据
def extract_json_from_line(line):
    if "req" in line:
        parts = line.split("req:", 1)  
        part = parts[1]
        json_str = part.replace("\'", "\"").replace('\"[', '[').replace(']\"', ']')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果无法解析为 JSON，返回 None
            return None


log_file_path = 'logs\hf_bus_2024-01-24.log' 
if not os.path.exists(config.csv_path):
    os.makedirs(config.csv_path)

# 使用 UTF-8 编码打开文件
with open(log_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        json_data = extract_json_from_line(line)
        if json_data:
            download_image_from_req(json_data)
            df = pd.DataFrame(json_data)
            csv_file_path = os.path.join(config.csv_path, '2024_01_24.csv')
            mode = 'a' if os.path.exists(csv_file_path) else 'w'
            df.to_csv(csv_file_path, mode=mode, index=False, encoding='utf-8-sig', header=not os.path.exists(csv_file_path))

