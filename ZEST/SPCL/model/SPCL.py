import torch
import torch.nn as nn
from transformers import AutoModel
from ..Configs import Configs


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        nn.init.constant_(m.bias, 0.0)


class SPCL(nn.Module):
    def __init__(self, args: Configs):
        super(SPCL, self).__init__()
        self.args = args
        self.centers = torch.nn.Parameter(torch.randn(args.num_classes, 1, args.feature_dim), requires_grad=False)
        self.centers_mask = torch.nn.Parameter(torch.zeros((args.num_classes, 1), dtype=torch.bool), requires_grad=False)
        self.f_context_encoder = AutoModel.from_pretrained(args.premodel_path,
                                                           local_files_only=True)
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, args.num_classes),
        )
        self.g = nn.Sequential(
            nn.Linear(self.dim, self.dim),
        )
        self.g.apply(_init_weights)

    def gen_f_reps(self, sentences):
        '''
        generate vector representations for each turn of conversation
        '''
        batch_size, max_len = sentences.shape[0], sentences.shape[-1]
        sentences = sentences.reshape(-1, max_len)
        mask = 1 - (sentences == (self.args.pad_value)).long()
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = (sentences == (self.args.MASK)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]
        # feature = torch.dropout(mask_outputs, 0.1, train=self.training)
        feature = mask_outputs
        # if self.config['output_mlp']:
        feature = self.g(feature)
        return feature

    def compute_scores(self, reps):

        num_classes, num_centers = self.args.num_classes, len(self.centers)
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_centers, -1)
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_classes, num_centers, -1)

        centers = self.centers.unsqueeze(0).expand(reps.shape[0], -1, -1, -1)
        # batch * turn, num_classes, num_centers
        sim_matrix = self.args.score_func(reps, centers)

        # batch * turn, num_calsses
        scores = sim_matrix
        return scores

    def forward(self, batch):
        ccl_reps = self.gen_f_reps(batch["input_ids"])
        outputs = self.compute_scores(ccl_reps)  #
        # outputs -= (~self.centers_mask) * 2
        outputs -= (1 - self.centers_mask) * 2
        cluster_outputs = torch.argmax(outputs.max(-1)[0], -1)
        direct_outputs = torch.argmax(self.predictor(ccl_reps), -1)

        return direct_outputs, cluster_outputs

