import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from .Configs import Configs
from .Mydataset import MyDataset as Dataset


class Data:
    def __init__(self, dataset: Dataset, args: Configs):
        self.args = args
        self.tokenizer = args.tokenizer
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                     collate_fn=self._collate_batch)

    def _collate_batch(self, batch):
        """
        :param batch: batch段对话
        :return:
        data = {
            "inputs_previous": ,
            "pre_pointer": list,
            "inputs_subsequent": ,
            "sub_pointer": list,
            "label_tensor":
        }
        """
        text_previous, pre_pointer, text_subsequent, sub_pointer, labels = [], [], [], [], []

        for item in batch:
            text_previous.append(item['text_previous'])
            pre_pointer.append(item['pre_pointer'])
            text_subsequent.append(item['text_subsequent'])
            sub_pointer.append(item['sub_pointer'])
            labels.append(item['labels'])

        inputs_previous = self.tokenizer(text_previous,
                                         return_tensors='pt',
                                         padding=True,
                                         truncation=True,
                                         max_length=64,
                                         )
        inputs_subsequent = self.tokenizer(text_subsequent,
                                           return_tensors='pt',
                                           padding=True,
                                           truncation=True,
                                           max_length=64,
                                           )
        label_tensor = torch.tensor(labels).long()
        data = {
            "inputs_previous": inputs_previous.to(self.args.device),
            "pre_pointer": pre_pointer,
            "inputs_subsequent": inputs_subsequent.to(self.args.device),
            "sub_pointer": sub_pointer,
            "label_tensor": label_tensor.to(self.args.device)
        }
        return data
