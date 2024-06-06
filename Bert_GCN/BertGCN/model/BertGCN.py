import torch
import torch.nn as nn
from ..parameter import Parameter
from .EdgeAtt import EdgeAtt
from .functions import batch_graphify
from .RGAT import RGAT
from .Classifier import Classifier


def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class BertGCN(nn.Module):

    def __init__(self, input_dim, args: Parameter):
        super(BertGCN, self).__init__()
        u_dim = input_dim
        g_dim = 400
        h1_dim = 100
        h2_dim = 100
        hc_dim = 256
        tag_size = args.tag_size

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.fc = nn.Sequential(
            nn.Linear(u_dim, g_dim),
            nn.ReLU()
            )
        self.fc.apply(_init_weights)

        self.edge_att = EdgeAtt(g_dim, args)
        self.rgat = RGAT(g_dim, h1_dim, h2_dim, args)
        self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)
        # self.clf = Classifier(g_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)

        # edge_type_to_idx = {'0': 0, '1': 1}

        # edge_type_to_idx = {}
        # for j in range(args.n_speakers):
        #     for k in range(args.n_speakers):
        #         edge_type_to_idx[str(j) + str(k)] = len(edge_type_to_idx)
        #         edge_type_to_idx[str(j) + str(k)] = len(edge_type_to_idx)

        self.edge_type_to_idx = edge_type_to_idx
        self.computed_params()

    def get_rep(self, data):
        node_features = self.fc(data["text_tensor"])  # [batch_size, mx_len, D_g]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.rgat(features, edge_index, edge_type)

        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        # out = self.clf(data["text_tensor"], data["text_len_tensor"])

        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        # loss = self.clf.get_loss(data["text_tensor"],
        #                          data["label_tensor"], data["text_len_tensor"])

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


# class BertGCN(nn.Module):
#
#     def __init__(self, args: Parameter):
#         super(BertGCN, self).__init__()
#         u_dim = args.feature_dim
#         g_dim = 400
#         hc_dim = 200
#         tag_size = 13
#
#         self.wp = args.wp
#         self.wf = args.wf
#         self.device = args.device
#
#         self.fc = nn.Sequential(
#             nn.Linear(u_dim, g_dim),
#             nn.ReLU()
#             )
#         self.fc.apply(_init_weights)
#
#         self.clf = Classifier(g_dim, hc_dim, tag_size, args)
#
#         self.computed_params()
#
#     def get_rep(self, data):
#         node_features = self.fc(data["text_tensor"])  # [batch_size, mx_len, D_g]
#         return node_features
#
#     def forward(self, data):
#         node_features = self.get_rep(data)
#         out = self.clf(node_features.view(-1, 400), data["text_len_tensor"])
#
#         return out
#
#     def get_loss(self, data):
#         node_features = self.get_rep(data)
#         loss = self.clf.get_loss(node_features.view(-1, 400),
#                                  data["label_tensor"], data["text_len_tensor"])
#
#         return loss
#
#     def computed_params(self):
#         total = 0
#         total2 = 0
#         for param in self.parameters():
#             total += param.nelement()
#             if param.requires_grad:
#                 total2 += param.nelement()
#         print("Number of parameter: %.2fM" % (total / 1e6))
#         print("Number of training parameter: %.2fM" % (total2 / 1e6))
