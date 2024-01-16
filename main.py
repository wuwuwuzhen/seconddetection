from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS  # 防止CORS错误
import os
from detection import sample_selection
from image_download import image_thread_pool_executor, download_image_from_req
from pack_req import pack_req
import requests
import logging

app = Flask(__name__)
CORS(app)  # 允许跨域请求

photo_path = './picture/'
video_path = './video/'
url = 'http://10.2.137.136:9202/alarm/filter/receive'
logging.basicConfig(filename='log.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


@app.route('/seconddetection/', methods=['POST'])
def seconddetection():
    logging.info(f'Request received: {request.json}')
    try:
        # 下载资源
        download_image_from_req(request.json)
        # 处理数据
        df = pd.DataFrame(request.json)
        df_combined = sample_selection(df)
        json_payload = pack_req(df_combined)
        logging.info(f'{json_payload}')
        response = requests.post(url, json=json_payload)
        if response.status_code == 200:
            # 如果请求成功，返回状态码 200 和成功信息
            return jsonify({"message": "Request sent successfully", "data": response.json()}), 200
        else:
            # 如果请求失败，返回状态码 500 和错误信息
            logging.error(
                f'Failed to send request: {response.status_code}, {response.text}')
            return jsonify({"message": "Failed to send request", "error": response.text}), 500
    except Exception as e:
        # 如果在处理中出现异常，返回状态码 500 和异常信息
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


with app.app_context():
    if not os.path.exists(photo_path):
        os.makedirs(photo_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

if __name__ == '__main__':
    app.run(debug=True)
