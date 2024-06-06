import os
import random

import pandas as pd
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .Configs import Configs
from .Data import Data
from .model import SPCL


# 数据类型验证
class DataType(object):
    def __init__(self,
                 speaker: str,
                 text: str,
                 label: int,
                 loc: tuple = None):
        if loc is None:
            loc = (-1, -1)
        self.speaker = speaker
        self.text = text
        self.label = label
        self.loc = loc


def load_src(file_name: str, args: Configs, replace_name=True, shuffle=False) -> list[list[DataType]]:
    if not replace_name:
        data = pd.read_csv(file_name)
        data = data[['TeamID', 'Player', 'LevelNumber', 'DataCode', 'Data01', 'Data02', 'PISACode']]
        data = data[data.DataCode == 5000]
        data = data[~data.PISACode.isnull()]
        data = data[~data.Data01.isnull()]
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
        for i, dialogue in enumerate(src):
            for j, turn_data in enumerate(dialogue):
                turn_data.loc = (i, j)

        return src

    data = pd.read_csv(file_name)
    data = data[['TeamID', 'Player', 'LevelNumber', 'DataCode', 'Data01', 'Data02', 'PISACode']]
    data = data[data.DataCode == 5000]
    data = data[~data.PISACode.isnull()]
    data = data[~data.Data01.isnull()]

    with open(args.names_pool_dir, 'r') as f:
        names_pool = f.read().split('\n')

    print('loaded names pool: ... {}'.format(names_pool[-5:]))

    src = []
    for team, group in data.groupby('TeamID'):
        seq = []
        mapping_table = {}
        for index in group.index:
            name = group['Player'][index]
            if name not in mapping_table:
                select_name = ''
                while select_name == '' or select_name in mapping_table:
                    select_name = names_pool[random.randint(0, len(names_pool) - 1)]
                mapping_table[name] = select_name

            seq.append(DataType(**{
                'speaker': mapping_table[name],
                'text': group['Data01'][index],
                'label': args.label_to_index[group['PISACode'][index]]
            }))
        src.append(seq)
    if shuffle:
        random.shuffle(src)
    for i, dialogue in enumerate(src):
        for j, turn_data in enumerate(dialogue):
            turn_data.loc = (i, j)

    return src

def get_params_group(model, args: Configs, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.ptmlr

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.learning_rate
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            args.ptmlr / 4 if id(param) in bert_params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params


def gen_all_reps(model: SPCL, data: Data):
    model.eval()
    results = []
    label_results = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data.dataloader),
                               desc="generate representations for all data",
                               total=len(data.dataloader)):
            results.append(model.gen_f_reps(batch["input_ids"]))
            label_results.append(batch["labels"])

        results = torch.cat(results, 0).squeeze()
        label_results = torch.cat(label_results, 0).squeeze()

        return results, label_results


def cluster(reps, labels, args: Configs, init_centers=None, init_centers_mask=None, epoch=0):
    """
    :param reps: 表示序列表示的张量，维度为(data_size, hidden_size)
    :param labels: 表示序列标签的张量，维度为(data_size)。
    :param args:
    :param init_centers: 初始聚类中心张量，维度为(num_classes, num_clusters, hidden_size)
    :param init_centers_mask: 初始聚类中心掩码张量，维度为(num_classes, num_clusters)
    :param epoch:
    :return:
    centers: 是所有聚类中心的tensor，tensor([[[1., 2., 1., 4., 3.]],
                                        [[1., 2., 1., 4., 3.]],
                                        [[1., 2., 1., 4., 3.]]], device='cuda:0')
    centers_mask: tensor([[1.],[1.],[1.]], device='cuda:0')
    cluster2dataid: {0: [2], 1: [0, 3], 2: [1]}
    cluster2classid: {0: 0, 1: 1, 2: 2}
    all_centers:是所有聚类中心tensor的列表 [tensor([1., 2., 1., 4., 3.]),
                                        tensor([1., 2., 1., 4., 3.]),
                                        tensor([1., 2., 1., 4., 3.])]
    """
    label_space = {}
    label_space_dataid = {}
    centers = []
    for idx in range(args.num_classes):
        label_space[idx] = []       # [[标签为0的所有特征],[]]
        label_space_dataid[idx] = []        # [[标签为0的所有特征对应的id（batch内id）],[]]
    for idx, turn_reps in enumerate(reps):
        label = labels[idx].item()
        if label < 0:
            continue
        label_space[label].append(turn_reps)
        label_space_dataid[label].append(idx)
    # clustering for each emotion class
    dim = label_space[0][0].shape[-1]

    max_num_clusters = 0
    cluster2dataid = {}
    cluster2classid = {}
    total_clusters = 0
    all_centers = []
    for label in range(args.num_classes):

        x = torch.stack(label_space[label], 0).reshape(-1, dim)     # [tensor, tensor, tensor] -> tensor
        # if init_centers is not None and init_centers_mask is not None:
        #     init = init_centers[
        #         label, :init_centers_mask[label].sum(), :]
        # else:
        #     init = []
        # kmeans_pytorch
        # num_clusters = x.shape[0] // CONFIG['avg_cluster_size']
        # if num_clusters > 1:
        #     flag = True
        #     while flag and num_clusters > 1:
        #         flag = False
        #         cluster_idxs, cluster_centers = kmeans(
        #             X=x,
        #             num_clusters=num_clusters,
        #             cluster_centers=[],
        #             distance=CONFIG['dist_func'],
        #             device=torch.device('cpu'),
        #             tqdm_flag=False,
        #         )
        #         for c_idx in range(num_clusters):
        #             c_size = (cluster_idxs == c_idx).sum()
        #             if c_size < CONFIG['avg_cluster_size']//2:
        #                 flag = True
        #                 num_clusters -= 1
        #             logging.info('decrease num_cluster')
        # if num_clusters <= 1:
        num_clusters = 1
        cluster_idxs = torch.zeros(x.shape[0]).long()
        cluster_centers = x.mean(0).unsqueeze(0).cpu()
        logging.info('{} clusters for emotion {}'.format(num_clusters, label))
        centers.append(cluster_centers)

        max_num_clusters = max(num_clusters, max_num_clusters)
        # 记录聚类中心到数据索引的映射，由此来构造对比学习的样本
        cluster_idxs += total_clusters
        for d_idx, c_idx in enumerate(cluster_idxs.numpy().tolist()):
            if c_idx < 0:
                continue
            if cluster2dataid.get(c_idx) is None:
                cluster2dataid[c_idx] = []
            cluster2classid[c_idx] = label
            cluster2dataid[c_idx].append(
                label_space_dataid[label][d_idx])
        total_clusters += num_clusters
        for c_idx in range(num_clusters):
            all_centers.append(cluster_centers[c_idx, :])

    centers_mask = []
    for label in range(args.num_classes):
        num_clusters, dim = centers[label].shape[0], centers[
            label].shape[-1]
        centers_mask.append(torch.zeros(max_num_clusters))
        centers_mask[label][:num_clusters] = 1
        centers[label] = torch.cat(
            (centers[label],
             torch.ones(max_num_clusters - num_clusters, dim)), 0)
    centers = torch.stack(centers, 0).to(args.device)
    centers_mask = torch.stack(centers_mask, 0).to(args.device)
    return centers, centers_mask, cluster2dataid, cluster2classid, all_centers


def dist(x, y):
    return (1-F.cosine_similarity(x, y, dim=-1))/2 + 1e-8


def selection(reps, all_centers, cluster2dataid, selection_ratio):
    """
    返回分数靠前[数据id]
    1. 计算每个特征的 scores = 到自己类别聚类中心的距离 / 到其它聚类中心距离的和
    2. 对按scores进行排序，返回分数靠前[数据id]，距离分数越小越好
    :param reps:
    :param all_centers:
    :param cluster2dataid:
    :param selection_ratio: 挑选的占全部数据的比率
    :return:
    """
    total_cluster = len(all_centers)
    data2clusterid = {}
    for c_idx in range(total_cluster):
        for data_id in cluster2dataid[c_idx]:
            data2clusterid[data_id] = c_idx
    all_centers = torch.stack(all_centers, 0).to(reps.device)
    # difficult measure function
    dis_scores = []
    for idx, rep in tqdm(enumerate(reps),
                         desc='compute distance scores.',
                         total=len(reps)):
        self_center = all_centers[data2clusterid[idx]]
        self_dis = dist(rep, self_center)
        sum_dis = dist(
            rep.unsqueeze(0).expand_as(all_centers),
            all_centers
        )
        dis_scores.append(self_dis / sum_dis.sum())
    dis_scores = torch.FloatTensor(dis_scores)
    # torch.argsort()是PyTorch库中的一个函数，用于对张量中的元素进行排序，并返回排序后元素的索引
    priority_seq = torch.argsort(dis_scores, descending=False).cpu().numpy().tolist()

    num_selection = int(selection_ratio * len(priority_seq))
    select_data_idx = priority_seq[:num_selection]

    return select_data_idx

def gen_cl_data(reps,
                all_centers,
                cluster2dataid,
                cluster2classid,
                args: Configs,
                epoch=0):
    """
    生成course learning数据，将容易区分的数据排在前面
    :param reps:
    :param all_centers:
    :param cluster2dataid:
    :param cluster2classid:
    :param args:
    :param epoch:
    :return:
    seed_list: [data_size * args.cl_selection_ratio] 挑选了距离分数靠前的数据，返回所挑数据的ID
    cluster_idxs：[data_size] 每个数据对应的聚类中心id
    """
    batch_size = args.batch_size
    num_data = reps.shape[0]
    dim = reps.shape[-1]
    total_cluster = len(all_centers)

    cluster_idxs = [0] * num_data     # [0]*data_size
    labels = torch.zeros(num_data).long()   # [0]*data_size

    for c_idx in range(total_cluster):
        for data_id in cluster2dataid[c_idx]:
            cluster_idxs[data_id] = c_idx
            labels[data_id] = cluster2classid[c_idx]
    selection_ids = selection(reps, all_centers, cluster2dataid, args.cl_selection_ratio)
    # plot_data(reps, labels, epoch, seed_list)
    return selection_ids, cluster_idxs
