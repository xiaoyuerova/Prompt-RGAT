import torch
import torch.nn as nn
from transformers import AutoModel
from .Configs import Configs


class BertSeq(nn.Module):
    def __init__(self, args: Configs):
        super(BertSeq, self).__init__()
        # self.bert_pre = BertModel.from_pretrained("bert-base-uncased")
        # self.bert_sub = BertModel.from_pretrained("bert-base-uncased")
        self.bert_pre = AutoModel.from_pretrained(args.premodel_path, local_files_only=True)
        self.bert_sub = AutoModel.from_pretrained(args.premodel_path, local_files_only=True)

        self.fc = nn.Sequential(
            nn.Linear(768 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, args.tag_size),
            nn.ReLU()
        )
        if args.class_weight:
            self.loss_weights = args.class_weight.to(args.device)
            self.nll_loss = nn.NLLLoss(self.loss_weights)
        else:
            self.nll_loss = nn.NLLLoss()

        self.fc.apply(self._init_weights)
        self.computed_params()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data):
        pre_pointer = data["pre_pointer"]
        sub_pointer = data["sub_pointer"]
        pre = self.bert_pre(**data["inputs_previous"])  # [B*l, S_L, 768]
        sub = self.bert_sub(**data["inputs_subsequent"])
        # 从bert输出中选取[SEP]对应的tensor
        pre = torch.stack([item[pre_pointer[i]] for i, item in enumerate(pre['last_hidden_state'])], 0)  # [B*l, 768]
        sub = torch.stack([item[sub_pointer[i]] for i, item in enumerate(sub['last_hidden_state'])], 0)
        log_prob = self.fc(torch.concat([pre, sub], dim=1))  # torch.concat([pre, sub], dim=1) -> [B*l, 768*2]
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat

    def get_loss(self, data):
        pre_pointer = data["pre_pointer"]
        sub_pointer = data["sub_pointer"]
        pre = self.bert_pre(**data["inputs_previous"])  # [B*l, S_L, 768]
        sub = self.bert_sub(**data["inputs_subsequent"])
        # 从bert输出中选取[SEP]对应的tensor
        pre = torch.stack([item[pre_pointer[i]] for i, item in enumerate(pre['last_hidden_state'])], 0)  # [B*l, 768]
        sub = torch.stack([item[sub_pointer[i]] for i, item in enumerate(sub['last_hidden_state'])], 0)
        log_prob = self.fc(torch.concat([pre, sub], dim=1))  # [B*l, tag_size]
        loss = self.nll_loss(log_prob, data["label_tensor"])

        return loss

    def computed_params(self):
        total = 0
        total2 = 0
        for param in self.parameters():
            total += param.nelement()
            if param.requires_grad:
                total2 += param.nelement()
        print("Number of parameter: %.2fM" % (total / 1e6))
        print("Number of training parameter: %.2fM" % (total2 / 1e6))
