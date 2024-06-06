import copy
import logging
import time
from sklearn.metrics import cohen_kappa_score, classification_report, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn import metrics
from transformers import get_cosine_schedule_with_warmup
from .Configs import Configs
from .model import SPCL, FGM
from utils import get_logger
from .utils import gen_all_reps, cluster, gen_cl_data
from .spcl_loss import SupProtoConLoss
from .Data import Data, EpochData
from .MyDataset import MyDataset as Dataset

log = get_logger(name=__name__, level='INFO')


class Coach:

    def __init__(self, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset, model: SPCL, optimizer, args: Configs):
        self.train_dataset = train_dataset
        self.train_data = Data(train_dataset, args, train=True, shuffle=False)
        self.valid_data = Data(valid_dataset, args, train=False, shuffle=False)
        self.test_data = Data(test_dataset, args, train=False, shuffle=False)
        self.epoch_train_data = None
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.label_to_idx = args.label_to_index
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.train_dataloader = None
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            # cluster
            all_reps, all_corr_labels = gen_all_reps(self.model, self.train_data)

            log.info('***clustering...***********************')
            centers, centers_mask, cluster2dataid, cluster2classid, all_centers = cluster(
                all_reps, all_corr_labels, self.args, epoch=epoch)

            self.model.centers = nn.Parameter(centers, requires_grad=False)
            self.model.centers_mask = nn.Parameter(centers_mask, requires_grad=False)

            # log.info('***course learning...***********************')
            selection, cluster_ids = gen_cl_data(all_reps,
                                                  all_centers,
                                                  cluster2dataid,
                                                  cluster2classid,
                                                  self.args,
                                                  epoch=epoch)
            # st = 1 - epoch / 10
            # ed = epoch / 10
            st = 1 - epoch / self.args.epochs
            ed = epoch / self.args.epochs
            num_data = len(selection)
            selection = torch.LongTensor(selection)
            prob_list = [
                st + (ed - st) / (num_data - 1) * i for i in range(num_data)
            ]
            prob_tensor = torch.FloatTensor(prob_list)
            rand_prob_tensor = torch.bernoulli(torch.Tensor([0.5] * num_data))  # 这里rand是random的意思
            if self.args.cl:
                sample = torch.bernoulli(prob_tensor).long()
            else:
                sample = torch.bernoulli(rand_prob_tensor).long()
            selection = selection * sample
            sample_results = selection[torch.nonzero(selection)]   # 原本数据id=0的数据被误删
            all_idxs = sample_results.squeeze().numpy().tolist()
            self.train_dataset.dataset["cluster_ids"] = cluster_ids
            self.epoch_train_data = EpochData(self.train_dataset, all_idxs, self.args)
            num_training_steps = len(self.epoch_train_data.subset) // (self.args.batch_size * self.args.accumulation_steps)
            num_warmup_steps = min(self.args.warm_up, num_training_steps) if epoch == 0 else 0
            lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            train_direct_f1 = self.train_epoch(epoch, lr_scheduler)
            log.info("[Train set] [direct_f1 {:.4f}]"
                     .format(train_direct_f1))

            dev_direct_f1, dev_cluster_f1, dev_acc, dev_kappa = self.evaluate()
            log.info("[Dev set] [direct_f1 {:.4f}] [cluster_f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]"
                     .format(dev_direct_f1, dev_cluster_f1, dev_acc, dev_kappa))
            if best_dev_f1 is None or train_direct_f1 > best_dev_f1:
                best_dev_f1 = train_direct_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

        test_direct_f1, test_cluster_f1, test_acc, test_kappa = self.evaluate(test=True)
        log.info("[Test set] [direct_f1 {:.4f}] [cluster_f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]"
                 .format(test_direct_f1, test_cluster_f1, test_acc, test_kappa))

        return best_dev_f1, best_epoch, best_state

    def train_epoch(self,
                    epoch,
                    lr_scheduler,
                    max_step=-1,
                    ):
        y_true_list = []
        direct_list = []

        self.model.train()
        spcl_loss = SupProtoConLoss(
            num_classes=self.args.num_classes,
            temp=self.args.temperature,
            pool_size=self.args.pool_size,
            support_set_size=self.args.support_set_size,
            centers=self.model.centers)

        fgm = FGM(self.model)
        for idx, batch in tqdm(enumerate(self.epoch_train_data.dataloader),
                               desc="train epoch {}".format(epoch),
                               total=len(self.epoch_train_data.dataloader)):
            loss, direct_predict = self.calc_loss(batch["input_ids"],
                                  batch["labels"],
                                  batch["cluster_ids"],
                                  spcl_loss)
            loss = loss / self.args.accumulation_steps
            loss.backward()
            direct_list.append(direct_predict.cpu())
            y_true_list.append(batch["labels"].cpu())

            if self.args.fgm:
                fgm.attack()
                loss, direct_predict = self.calc_loss(batch["input_ids"],
                                  batch["labels"],
                                  batch["cluster_ids"],
                                  spcl_loss)
                loss = loss / self.args.accumulation_steps
                loss.backward()
                fgm.restore()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)

            if idx % self.args.accumulation_steps == 0:
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
            if 0 < max_step < idx:
                self.optimizer.zero_grad()
                break
        self.optimizer.zero_grad()

        direct_list = torch.cat(direct_list, -1).numpy()
        y_true_list = torch.cat(y_true_list, -1).numpy()

        direct_f1_scores = f1_score(y_true=y_true_list,
                                    y_pred=direct_list,
                                    average='weighted')

        return direct_f1_scores


    def evaluate(self, test=False, desc=''):
        data = self.test_data if test else self.valid_data
        # log.debug('len(data.dataloader) = {}'.format(len(data.dataloader)))
        self.model.eval()
        with torch.no_grad():
            y_true_list = []
            direct_list = []
            cluster_list = []
            for idx, batch in tqdm(enumerate(data.dataloader),
                                   desc="test" if test else "dev",
                                   total=len(data.dataloader)):
                with torch.no_grad():
                    cluster_outputs, direct_outputs = self.model(batch)
                    labels = batch["labels"]
                    direct_list.append(direct_outputs.detach().to('cpu'))
                    cluster_list.append(cluster_outputs.detach().to('cpu'))
                    y_true_list.append(labels.to('cpu'))

        direct_list = torch.cat(direct_list, -1).numpy()
        cluster_list = torch.cat(cluster_list, -1).numpy()
        y_true_list = torch.cat(y_true_list, -1).numpy()

        direct_f1_scores = f1_score(y_true=y_true_list,
                     y_pred=direct_list,
                     average='weighted')
        cluster_f1_scores = f1_score(y_true=y_true_list,
                     y_pred=cluster_list,
                     average='weighted')
        acc = metrics.accuracy_score(y_true_list, direct_list)
        kappa = cohen_kappa_score(y_true_list, direct_list)

        # f1 = max(max(direct_f1_scores), max(cluster_f1_scores))
        self.optimizer.zero_grad()

        return direct_f1_scores, cluster_f1_scores, acc, kappa


    def calc_loss(self, sentences, emotion_idxs, labels, spcl_loss: SupProtoConLoss, train_obj=None):
        if train_obj is None:
            train_obj = self.args.train_obj
        ce_loss_func = self.ce_loss_func


        ccl_reps = self.model.gen_f_reps(sentences)

        if train_obj == 'ce':
            direct_outputs = self.model.predictor(ccl_reps)
            direct_loss = ce_loss_func(direct_outputs, emotion_idxs)
        else:
            direct_outputs = self.model.predictor(ccl_reps.detach())
            direct_loss = ce_loss_func(direct_outputs,emotion_idxs)
        ins_cl_loss = torch.zeros(1).to(self.args.device)
        if train_obj == 'spcl':
            ins_cl_loss = spcl_loss(ccl_reps, labels)
        if train_obj == 'spdcl':
            ins_cl_loss = spcl_loss(ccl_reps, labels, decoupled=True)

        loss = direct_loss + ins_cl_loss

        direct_outputs = torch.argmax(direct_outputs, -1)

        return loss, direct_outputs

