import os
import random

import torch
from torch.utils.data import Dataset
from .Configs import Configs
from .utils import get_logger

log = get_logger(name=__name__, level="INFO")


class DataUnit(object):
    def __init__(self,
                 idx: int,
                 text: str,
                 input_id: int,
                 label: int,
                 speaker_id: int,
                 cluster_id: int = None):
        self.idx = idx
        self.text = text
        self.input_id = input_id
        self.label = label
        self.speaker_id = speaker_id
        self.cluster_id = cluster_id


class MyDataset(Dataset):
    def __init__(self, src: list, args: Configs, train=False, tmp_name: str = None):
        """

        :param src:
        :param args:
        :param train:
        :param tmp_name: str 处理后数据临时保存的文件名。可以是train_data,test_data等
        """
        self.train = train
        self.tmp_name = tmp_name
        self.args = args
        self.dataset: dict = self.init_data_with_context(src)
        self.all_cluster_centers: [torch.Tensor] = None
        self.dataset["cluster_ids"] = None

    def __getitem__(self, index):
        ret = {}
        for key in self.dataset.keys():
            if self.dataset[key] is None:
                ret[key[:-1]] = None
            elif isinstance(self.dataset[key], list):
                ret[key[:-1]] = self.dataset[key][index]
            elif isinstance(self.dataset[key], str):
                ret[key] = self.dataset[key]
            else:
                print('key = {} 的数据类型错误！'.format(key))

        return DataUnit(**ret)

    def __len__(self):
        return len(self.dataset['labels'])

    def init_data_with_context(self, src: list):
        QUERY = 'For utterance:'
        PROMPT = 'the cooperative problem-solving skills {} has shown is <mask>'
        if self.args.src_type == 'mdla_pisa':
            QUERY = '上文中的这句话：'
            PROMPT = '体现了 {} 的合作解决问题技能是 <mask>'
        ret_idx = []
        ret_text = []
        ret_speakers_id = []
        ret_utterances = []
        ret_labels = []
        for dialogue in src:
            utterance_ids = []
            speaker_to_id = {}
            query = QUERY
            query_ids = self.args.tokenizer(query)['input_ids'][1:-1]
            for idx, turn_data in enumerate(dialogue):
                # item_check = []
                text_with_speaker = turn_data.speaker + ':' + str(turn_data.text)
                token_ids = self.args.tokenizer(text_with_speaker)['input_ids'][1:]
                utterance_ids.append(token_ids)
                # utterance_check.append(turn_data.loc)

                full_context = [self.args.CLS]
                # 给对话历史限长。两个条件：1、对话历史的tokens数小于max_len；2、历史的句子数小于8
                l_idx = 0
                for l_idx in range(idx):
                    total_len = sum([len(item) for item in utterance_ids[l_idx:]]) + self.args.max_utterance_history_len
                    if total_len + len(utterance_ids[idx]) <= self.args.max_sentence_len:
                        break
                l_idx = max(l_idx, idx - self.args.max_utterance_history_len)
                for i, item in enumerate(utterance_ids[l_idx:]):
                    full_context.extend(item)
                    # item_check.append(utterance_check[l_idx:][i])

                query_idx = idx
                prompt = PROMPT.format(turn_data.speaker)
                full_query = query_ids + utterance_ids[query_idx] + self.args.tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, self.args.max_sentence_len, self.args.pad_value)
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx].label)
                if dialogue[query_idx].speaker in speaker_to_id:
                    ret_speakers_id.append(speaker_to_id[dialogue[query_idx].speaker])
                else:
                    speaker_to_id[dialogue[query_idx].speaker] = len(speaker_to_id)
                    ret_speakers_id.append(speaker_to_id[dialogue[query_idx].speaker])
                ret_idx.append(len(ret_idx))
                ret_text.append(dialogue[query_idx].text)

                # 为解决语言模型无法定位目标句，在训练时设置额外的训练目标。
                if self.train and idx > 3:
                    for query_idx in range(l_idx, idx):
                        if torch.rand(1).item() < self.args.extra_target_ratio:
                            # query_idx = random.randint(lidx, idx - 1)
                            prompt = PROMPT.format(turn_data.speaker)
                            full_query = query_ids + utterance_ids[query_idx] + self.args.tokenizer(prompt)[
                                                                                    'input_ids'][1:]
                            input_ids = full_context + full_query
                            input_ids = pad_to_len(input_ids, self.args.max_sentence_len, self.args.pad_value)
                            ret_utterances.append(input_ids)
                            ret_labels.append(dialogue[query_idx].label)
                            if dialogue[query_idx].speaker in speaker_to_id:
                                ret_speakers_id.append(speaker_to_id[dialogue[query_idx].speaker])
                            else:
                                speaker_to_id[dialogue[query_idx].speaker] = len(speaker_to_id)
                                ret_speakers_id.append(speaker_to_id[dialogue[query_idx].speaker])
                            ret_idx.append(len(ret_idx))
                            ret_text.append(dialogue[query_idx].text)

        data = {
            "idxs": ret_idx,
            "texts": ret_text,
            "input_ids": ret_utterances,
            "labels": ret_labels,
            "speaker_ids": ret_speakers_id,
        }
        return data

    def init_data_with_context2(self, src: list):
        if self.tmp_name:
            if os.path.exists(self.tmp_name):
                data = torch.load(os.path.join(self.args.output_dir, self.tmp_name + '.pt'))
                return data
        ret_utterances = []
        ret_labels = []
        data_check = []
        for dialogue in src:
            utterance_ids = []
            utterance_check = []
            query = 'For utterance:'
            query_ids = self.args.tokenizer(query)['input_ids'][1:-1]
            for idx, turn_data in enumerate(dialogue):
                item_check = []
                # text_with_speaker = turn_data.speaker + ':' + turn_data.text
                text_with_speaker = turn_data.text
                token_ids = self.args.tokenizer(text_with_speaker)['input_ids'][1:]
                utterance_ids.append(token_ids)
                utterance_check.append(turn_data.loc)
                if turn_data.label < 0:
                    continue
                full_context = [self.args.CLS]
                # 给对话历史限长。两个条件：1、对话历史的tokens数小于max_len；2、历史的句子数小于8
                lidx = 0
                for lidx in range(idx):
                    total_len = sum([len(item) for item in utterance_ids[lidx:]]) + self.args.max_utterance_history_len
                    if total_len + len(utterance_ids[idx]) <= self.args.max_sentence_len:
                        break
                lidx = max(lidx, idx - self.args.max_utterance_history_len)
                for i, item in enumerate(utterance_ids[lidx:]):
                    full_context.extend(item)
                    # print(i, lidx, len(utterance_check), utterance_check)
                    item_check.append(utterance_check[lidx:][i])

                query_idx = idx
                # prompt = dialogue[query_idx].speaker + ' feels <mask>'
                prompt = ' feels <mask>'
                full_query = query_ids + utterance_ids[query_idx] + self.args.tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, self.args.max_sentence_len, self.args.pad_value)
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx].label)
                data_check.append(item_check + [turn_data.loc])

                # 为解决语言模型无法定位目标句，在训练时设置额外的训练目标。
                if self.train and idx > 3 and torch.rand(1).item() < 0.4:
                    query_idx = random.randint(lidx, idx - 1)
                    # if dialogue[query_idx]['label'] < 0:
                    #     continue
                    # prompt = dialogue[query_idx].speaker + ' feels <mask>'
                    prompt = ' feels <mask>'
                    full_query = query_ids + utterance_ids[query_idx] + self.args.tokenizer(prompt)['input_ids'][1:]
                    input_ids = full_context + full_query
                    input_ids = pad_to_len(input_ids, self.args.max_sentence_len, self.args.pad_value)
                    ret_utterances.append(input_ids)
                    ret_labels.append(dialogue[query_idx].label)
                    data_check.append(item_check + [dialogue[query_idx].loc])

        data = {
            "input_ids": ret_utterances,
            "labels": ret_labels
        }
        print("train: {}, Data check: {}".format(self.train, data_check[:10]))

        if self.tmp_name:
            torch.save(data, os.path.join(self.args.output_dir, self.tmp_name + '.pt'))

        return data

    def init_data_no_context(self, src: list):
        ret_utterances = []
        ret_labels = []
        for dialogue in src:
            for idx, turn_data in enumerate(dialogue):
                input_ids = self.args.tokenizer(turn_data.text + 'was <mask>')['input_ids']
                ret_labels.append(turn_data.label)
                input_ids = pad_to_len(input_ids, self.args.max_sentence_len, self.args.pad_value)
                ret_utterances.append(input_ids)

        data = {
            "input_ids": ret_utterances,
            "labels": ret_labels
        }

        return data


def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data
