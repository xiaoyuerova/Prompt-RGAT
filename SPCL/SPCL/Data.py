import random

import torch
from torch.utils.data import DataLoader, Subset
from .Configs import Configs
from .MyDataset import MyDataset as Dataset
from .MyDataset import DataUnit


class Data:
    def __init__(self, dataset: Dataset, args: Configs, train=False, shuffle=None):
        self.train = train
        self.args = args
        if shuffle is None:
            shuffle = args.shuffle
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                     collate_fn=self._collate_batch)

    def _collate_batch(self, batch: list[DataUnit]):
        """
        将对话数据组织成（上下文+查询句，标签）的训练对
        """
        input_ids, labels = [], []
        for data_unit in batch:
            input_ids.append(data_unit.input_id)
            labels.append(data_unit.label)

        data = {
            "input_ids": torch.LongTensor(input_ids).to(self.args.device),
            "labels": torch.LongTensor(labels).to(self.args.device),
        }

        return data


class EpochData:
    def __init__(self, dataset: Dataset,
                 selection_ids: list[int],
                 args: Configs,
                 shuffle=False):
        self.args = args
        self.subset = Subset(dataset, selection_ids)
        self.dataloader = DataLoader(self.subset, batch_size=args.batch_size, shuffle=shuffle,
                                     collate_fn=self._collate_batch)

    def _collate_batch(self, batch: list[DataUnit]):
        input_ids, labels, cluster_ids = [], [], []
        for data_unit in batch:
            input_ids.append(data_unit.input_id)
            labels.append(data_unit.label)
            cluster_ids.append(data_unit.cluster_id)

        data = {
            "input_ids": torch.LongTensor(input_ids).to(self.args.device),
            "labels": torch.LongTensor(labels).to(self.args.device),
            "cluster_ids": torch.LongTensor(cluster_ids).to(self.args.device)
        }

        return data



