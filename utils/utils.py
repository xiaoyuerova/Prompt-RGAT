import os.path
import sys
import random
import logging

import numpy as np
import torch
import pickle


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(
        name: str,
        level='INFO',
        stream_handler=True
):
    log_file_name = './save_log/log_' + name + '.txt'

    if not os.path.exists(os.path.dirname(log_file_name)):
        os.makedirs(os.path.dirname(log_file_name))
    if not os.path.exists(log_file_name):
        with open(log_file_name, 'w') as file:
            print(f"File {log_file_name} created.")

    logger = logging.getLogger(name)
    if not logger.handlers:  # 如果记录器没有处理程序，则添加处理程序
        if level == 'DEBUG':
            logger.setLevel(level=logging.DEBUG)
        elif level == 'INFO':
            logger.setLevel(level=logging.INFO)
        else:
            logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_file_name:
            handler = logging.FileHandler(log_file_name)
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        if stream_handler:
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(formatter)
            logger.addHandler(console)
    return logger


# 数据类型验证
class DataType(object):
    def __init__(self,
                 text_src: list[str],
                 text_tensor: torch.Tensor,
                 speaker: list,
                 label: list):
        self.text_src = text_src
        self.text_tensor = text_tensor
        self.speaker = speaker
        self.label = label


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file) -> list:
    with open(file, "rb") as f:
        return pickle.load(f)
