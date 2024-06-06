from .Configs import Configs
import pandas as pd
import os.path
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


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file) -> list:
    with open(file, "rb") as f:
        return pickle.load(f)


class DataType(object):
    def __init__(self,
                 speaker: list,
                 text: list[str],
                 label: list):
        self.speaker = speaker
        self.text = text
        self.label = label


def load_src_for_train(data, args: Configs, file_type='csv', shuffle=False):
    """
    加载数据. 从收集的包含各种信息的结构化数据中采集训练需要的数据
    :param data: 数据源
    :param args:
    :param file_type: 数据源的文件类型
    :param shuffle:
    :return:
    """
    if file_type == 'csv':
        data = pd.read_csv(data)
    elif file_type == 'xlsx':
        data = pd.read_excel(data)
    elif file_type == 'pkl':
        data = load_pkl(data)
    elif file_type == 'list':
        if args.src_type == 'mdla_pisa':
            data = pd.DataFrame(data, columns=['lesson_key',
                                               'TeamID',
                                               'group_number',
                                               'login_account',
                                               'Player',
                                               'LevelNumber',
                                               'DataCode',
                                               'Data01',
                                               'Data02',
                                               'PISACode'])
    else:
        print('数据格式不支持')
        return ['数据格式不支持']

    if args.src_type == 'mdla_pisa':
        data = data[['Group', 'Player', 'False_Name', 'Text', 'Label']]
        src = []
        for team, group in data.groupby('Group'):
            seq = []
            for index in group.index:
                seq.append(DataType(**{
                    'speaker': group['False_Name'][index],
                    'text': group['Text'][index],
                    'label': args.label_to_index[group['Label'][index]]
                }))
            src.append(seq)
        if shuffle:
            random.shuffle(src)
        return src

    if args.src_type == 'sdz_pisa':
        data = data[['TeamID', 'DataCode', 'PlayerID', 'Player', 'Data01', 'PISACode']]
        data = data[data.DataCode == 5000]
        data = data[~data.Data01.isnull()]
        data = data[~data.PISACode.isnull()]
        print(data.head())
        src = []
        for team, group in data.groupby('TeamID'):
            seq = []
            for index in group.index:
                seq.append(DataType(**{
                    'speaker': group['Player'][index],
                    'text': group['Data01'][index],
                    'label': args.label_to_index[group['PISACode'][index]]
                }))
            src.append(seq)
        if shuffle:
            random.shuffle(src)
        return src

    if args.src_type == 'sdz_aj':
        data = data[['NewName', 'DataCode', 'StudentID', 'Player', 'Data01', 'Jess0']]
        data = data[data.DataCode == 5000]
        data = data[~data.Data01.isnull()]
        data = data[~data.Jess0.isnull()]
        # print(data[data.Jess0.isin(['CEU', 'CE', 'CM'])])
        data = data[~data.Jess0.isin(['CEU', 'CE', 'CM'])]

        src = []
        for team, group in data.groupby('NewName'):
            seq = []
            for index in group.index:
                seq.append(DataType(**{
                    'speaker': group['Player'][index],
                    'text': group['Data01'][index],
                    'label': args.label_to_index[group['Jess0'][index]]
                }))
            src.append(seq)
        if shuffle:
            random.shuffle(src)
        return src
