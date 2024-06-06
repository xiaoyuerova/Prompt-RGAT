from torch.utils.data import Dataset
from .utils import DataType


class MyDataset(Dataset):
    def __init__(self, src: list[list[DataType]], tokenizer):
        self.src = src
        self.preprocessed_data = self.preprocess(tokenizer)

    def __getitem__(self, index):
        return self.preprocessed_data[index]

    def __len__(self):
        return len(self.preprocessed_data)

    def preprocess(self, tokenizer):
        texts, labels = [], []
        for seq in self.src:
            for i, item in enumerate(seq):
                texts.append(item.text)
                labels.append(item.label)
        data = {
            "texts": texts,
            "labels": labels
        }
        return [
            {
                "texts": texts[i],
                "labels": labels[i]
            }
            for i in range(len(data['labels']))
        ]
