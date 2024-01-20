import logging
from logging.handlers import TimedRotatingFileHandler
import os
# 初始化log
def init_log():    
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 设置日志文件的完整路径
    log_file_path = os.path.join(log_dir, 'hf_bus.log')
    # 设置日志的格式
    log_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    # 创建TimedRotatingFileHandler
    # 参数 'midnight' 表示日志会在午夜轮转
    handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=0)
    handler.setFormatter(log_formatter)
    handler.suffix = "%Y-%m-%d"  # 你可以选择你想要的时间格式
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)