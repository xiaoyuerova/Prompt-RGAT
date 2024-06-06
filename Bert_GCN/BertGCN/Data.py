import torch
from torch.utils.data import DataLoader
from .parameter import Parameter
from .MyDataset import Dataset
from .utils import DataType


class Data:
    def __init__(self, dataset: Dataset, args: Parameter):
        self.args = args
        self.label_pipeline = lambda x: [int(item) for item in x]
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self._collate_batch)

    def _collate_batch(self, batch: list[DataType]):
        """
        每段对话长度补齐
        设定每句话的维度为100。这里只要了文本的特征
        :param batch: batch段对话
        :return:
        data = {
            "text_len_tensor": 每段对话的长度，[bz],
            "text_tensor": [bz, mx, 100],
            "speaker_tensor": 每段对话中，每句话的speaker。[bz, mx],
            "label_tensor": 一维向量，每段话中每句话的标签汇总
        }
        """
        batch_size = len(batch)
        text_len_tensor = torch.tensor([len(s.text_src) for s in batch]).long()
        mx = torch.max(text_len_tensor).item()
        text_tensor = torch.zeros((batch_size, mx, self.args.text_dim))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        for i, item in enumerate(batch):
            cur_len = len(item.text_tensor)
            # print('item.text_tensor', cur_len, item.text_tensor.size())
            text_tensor[i, :cur_len, :] = item.text_tensor
            speaker_tensor[i, :cur_len] = torch.tensor(item.speaker)
            labels.extend(item.label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor.to(self.args.device),
            "text_tensor": text_tensor.to(self.args.device),
            "speaker_tensor": speaker_tensor.to(self.args.device),
            "label_tensor": label_tensor.to(self.args.device)
        }

        return data
