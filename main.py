from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS  # 防止CORS错误
import os
from detection import sample_selection
from image_download import image_thread_pool_executor
from pack_req import pack_req
import requests


app = Flask(__name__)
CORS(app)  # 允许跨域请求

photo_path = './picture/'
video_path = './video/'
url = 'http://10.2.137.136:9202/alarm/filter/receive'


# @app.route('/seconddetection/', methods=['POST'])
# def seconddetection():
#     df = pd.DataFrame(request.json)
#     # thread_pool_executor(df)
#     df_combined = sample_selection(df)
#     json_payload = pack_req(df_combined)   
#     response = requests.post(url, json=json_payload)
#     if response.status_code == 200:
#         print('Request sent successfully.')
#     else:
#         print(f'Failed to send request: {response.status_code}, {response.text}')
    
#     df_dict = df_combined.to_dict(orient='records')

#     # 需要保留中文不转义
#     res = json.dumps(df_dict, ensure_ascii=False)
#     return res, 200

@app.route('/seconddetection/', methods=['POST'])
def seconddetection():
    try:
        df = pd.DataFrame(request.json)
        image_thread_pool_executor(df) 
        df_combined = sample_selection(df)
        json_payload = pack_req(df_combined)   
        response = requests.post(url, json=json_payload)
        if response.status_code == 200:
            # 如果请求成功，返回状态码 200 和成功信息
            return jsonify({"message": "Request sent successfully", "data": response.json()}), 200
        else:
            # 如果请求失败，返回状态码 500 和错误信息
            return jsonify({"message": "Failed to send request", "error": response.text}), 500
    except Exception as e:
        # 如果在处理中出现异常，返回状态码 500 和异常信息
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

with app.app_context():
    if not os.path.exists(photo_path):
        os.makedirs(photo_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

if __name__ == '__main__':
    app.run(debug=True)
