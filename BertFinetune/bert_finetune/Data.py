import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .Configs import Configs
from .MyDataset import MyDataset


class Data:
    def __init__(self, dataset: MyDataset, args: Configs, shuffle=None):
        self.args = args
        self.dataset = dataset
        self.tokenizer = args.tokenizer
        if shuffle is None:
            shuffle = args.shuffle
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                     collate_fn=self._collate_batch)

    def _collate_batch(self, batch):
        texts, labels = [], []

        for item in batch:
            texts.append(item['texts'])
            labels.append(item['labels'])

        texts_tensor = self.tokenizer(texts,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True,
                                      max_length=self.args.max_length,
                                      )
        labels_tensor = torch.tensor(labels).long()
        data = {
            "texts_tensor": texts_tensor.to(self.args.device),
            "labels_tensor": labels_tensor.to(self.args.device)
        }
        return data
