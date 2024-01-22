from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS  # 防止CORS错误
import os
from detection import sample_selection
from image_download import image_thread_pool_executor, download_image_from_req
from pack_req import pack_req
import requests
import log
import logging
import traceback
import config

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化日志
log.init_log()

@app.route('/seconddetection/', methods=['POST'])
def seconddetection():
    logging.debug("[seconddetection] request received")
    logging.info(f"[seconddetection] req:{request.json}")
    try:
        # 下载资源
        download_image_from_req(request.json)
        logging.debug("[seconddetection] succeed in downloading images and videos")
        # 处理数据
        df = pd.DataFrame(request.json)
        logging.debug(f'[seconddetection] succeed in converting json to df')
        df_combined = sample_selection(df)
        logging.debug(f'[seconddetection] succeed in selecting samples')
        json_payload = pack_req(df_combined)
        logging.info(f'[seconddetection] resp: {json_payload}')
        response = requests.post(config.resp_url, json=json_payload)
        if response.status_code == 200:
            # 如果请求成功，返回状态码 200 和成功信息
            return jsonify({"message": "Request sent successfully", "data": response.json()}), 200
        else:
            # 如果请求失败，返回状态码 500 和错误信息
            logging.error(
                f"[seconddetection] failed to send request|status_code: {response.status_code}|text: {response.text}")
            return jsonify({"message": "Failed to send request", "error": response.text}), 500
    except Exception as e:
        # 如果在处理中出现异常，返回状态码 500 和异常信息
        logging.error(f"[seconddetection] an error occurred|exception: {str(e)}|traceback: {traceback.format_exc()}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


with app.app_context():
    if not os.path.exists(config.photo_path):
        os.makedirs(config.photo_path)
    if not os.path.exists(config.video_path):
        os.makedirs(config.video_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
