from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, src: list):
        self.src = src

    def __getitem__(self, index):
        return self.src[index]

    def __len__(self):
        return len(self.src)
