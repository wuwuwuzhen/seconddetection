import pandas as pd
import json

# 读取 Excel 文件
file_path = "20250107.xlsx"  # 替换为实际的文件路径
df = pd.read_excel(file_path)

# 定义转换为 JSON 格式的函数
def generate_json(row):
    if pd.isna(row["plate"]):
        return None  # 如果 plate 为空，返回 None
    result = {
        "id": row["id"],
        "plate": row["plate"],
        "day": row["day"],
        "alarm_begin_time": row["alarm_begin_time"],
        "alarm_end_time": row["alarm_end_time"],
        "exception_name": row["exception_name"],
        "exception_type": row["exception_type"],
    }
    # 如果 video_url 不为空，则添加到 JSON 中
    if pd.notna(row["video_url"]):  # 检查是否为空
        result["video_url"] = row["video_url"] # 转换为列表
    if pd.notna(row["picture_url"]):  # 检查是否为空
        result["picture_url"] = row["picture_url"] # 转换为列表
    return result

# 转换为 JSON 格式
json_data = [generate_json(row) for _, row in df.iterrows()]
json_data = [data for data in json_data if data is not None]

# 保存为 JSON 文件
output_path = "output_1_8.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"JSON 数据已保存到 {output_path}")