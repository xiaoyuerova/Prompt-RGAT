import torch.nn as nn
from torch_geometric.nn.conv import RGATConv, GraphConv
from ..parameter import Parameter


class RGAT(nn.Module):

    def __init__(self,
                 g_dim,
                 h1_dim,
                 h2_dim,
                 args: Parameter):
        super(RGAT, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGATConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)

        return x






















