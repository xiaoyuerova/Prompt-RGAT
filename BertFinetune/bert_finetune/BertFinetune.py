import torch
import torch.nn as nn
from transformers import logging
from transformers import AutoModel
from .Configs import Configs

logging.set_verbosity_warning()
logging.set_verbosity_error()


class BertFinetune(nn.Module):
    def __init__(self, args: Configs):
        super(BertFinetune, self).__init__()
        self.bert = AutoModel.from_pretrained(args.premodel_path, local_files_only=True)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            # nn.ReLU(),
            nn.Linear(256, args.tag_size),
            # nn.ReLU()
        )
        self.fc.apply(self._init_weights)
        if args.class_weight:
            self.loss_weights = args.class_weight.to(args.device)
            self.nll_loss = nn.NLLLoss(self.loss_weights)
        else:
            self.nll_loss = nn.NLLLoss()
        self.computed_params()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data):
        feature = self.gen_f_reps(data)
        log_prob = self.fc(feature)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, data):
        feature = self.gen_f_reps(data)
        log_prob = self.fc(feature)
        loss = self.nll_loss(log_prob, data["labels_tensor"])
        return loss

    def gen_f_reps(self, data):
        out = self.bert(**data["texts_tensor"])  # [B*l, S_L, 768]
        feature = out[1]
        return feature

    def computed_params(self):
        total = 0
        total2 = 0
        for param in self.parameters():
            total += param.nelement()
            if param.requires_grad:
                total2 += param.nelement()
        print("Number of parameter: %.2fM" % (total / 1e6))
        print("Number of training parameter: %.2fM" % (total2 / 1e6))
