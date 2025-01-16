import logging
import os

class Logger():
    def __init__(self, name='', log_dir='./output/tensorboard_res/logs'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        logging.getLogger('PIL').setLevel(logging.WARNING)
        fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
        date_fmt='%Y-%m-%d %H:%M:%S'

        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
        self.logger.addHandler(file_handler)