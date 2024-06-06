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
        end_token = '[SEP]'

        text_previous = []
        pre_pointer = []
        text_subsequent = []
        sub_pointer = []
        labels = []

        for seq in self.src:
            for i, item in enumerate(seq):
                pre = []
                sub = []

                # 给句子限长，32个单词
                sentence_len = []
                inputs = tokenizer(item.text,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=32,
                                   )
                text_src = []
                for sentence in inputs['input_ids']:
                    s = ''
                    s_l = 0
                    for ids in sentence:
                        if ids.item() == 0 or ids.item() == 101 or ids.item() == 102:
                            continue
                        # s = s + tokenizer.ids_to_tokens.get(ids.item()) + ' '
                        s = s + tokenizer.decode(ids) + ' '
                        s_l += 1
                    text_src.append(s)
                    sentence_len.append(s_l)

                # 计算
                pre.append(end_token + ' ' + text_src[0])
                for idx in range(len(text_src) - 1):
                    pre.append(text_src[idx] + ' ' + end_token + ' ' + text_src[idx + 1])
                    sub.append(text_src[idx] + ' ' + end_token + ' ' + text_src[idx + 1])
                sub.append(text_src[-1] + ' ' + end_token)

                text_previous.extend(pre)
                text_subsequent.extend(sub)
                for index, l in enumerate(sentence_len):
                    if index == 0:
                        pre_pointer.append(l + 1)
                    else:
                        pre_pointer.append(sentence_len[index - 1] + l + 2)
                    sub_pointer.append(l + 1)

                labels.append(item.label)

        data = {
            "text_previous": text_previous,
            "pre_pointer": pre_pointer,
            "text_subsequent": text_subsequent,
            "sub_pointer": sub_pointer,
            "labels": labels
        }
        return [
            {
                "text_previous": text_previous[i],
                "pre_pointer": pre_pointer[i],
                "text_subsequent": text_subsequent[i],
                "sub_pointer": sub_pointer[i],
                "labels": labels[i]
            }
            for i in range(len(data['labels']))
        ]
