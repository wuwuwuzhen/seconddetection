import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler
'''
# 配置定时任务
执行 crontab -e
按实际情况输入  
0 0 * * * python /path/to/your/script/delete_file.py
保存并退出

# 使脚本可执行
chmod +x /path/to/your/script/delete_file.py

'''


def init_log():
    log_dir = 'delete_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'hf_bus_delete.log')
    log_formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler = TimedRotatingFileHandler(
        log_file_path, when="midnight", interval=1, backupCount=0)
    handler.setFormatter(log_formatter)
    handler.suffix = "%Y-%m-%d"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def delete_files_in_paths(expire_time, *paths):
    # print(f'expire_time: {expire_time}, paths: {paths}')
    logging.info(
        f'start to delete files, expire_time: {expire_time}, paths: {paths}')
    now_time = time.time()

    def delete_files_in_path(file_path):
        for root, dirs, files in os.walk(file_path):
            for file in files:
                file_name = os.path.join(root, file)
                mtime = os.path.getmtime(file_name)
                if now_time - mtime > expire_time:
                    # 测试没问题后取消注释
                    # os.remove(file_name)
                    logging.info(
                        f'{file_name} deleted, mtime is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))}')
            for dir in dirs:
                delete_files_in_path(os.path.join(root, dir))

    for path in paths:
        delete_files_in_path(path)


if __name__ == '__main__':
    init_log()
    # 获取当前文件的完整路径
    current_file_path = os.path.realpath(__file__)
    # 获取当前文件所在的目录
    # 7days
    current_dir = os.path.dirname(current_file_path)
    delete_files_in_paths(3600 * 24 * 7, os.path.join(current_dir,
                          'video'), os.path.join(current_dir, 'picture'))
