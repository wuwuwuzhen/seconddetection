import logging
from logging.handlers import TimedRotatingFileHandler
import os

def init_log():    
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'hf_bus.log')
    log_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=0)
    handler.setFormatter(log_formatter)
    handler.suffix = "%Y-%m-%d" 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)