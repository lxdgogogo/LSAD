import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.dataloading import DataLoader
import dgl.nn.pytorch.conv as dglnn
from utils.functions import create_activation, create_norm


class GCN(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout, activation, norm, encoding=True):
        super(GCN, self).__init__()
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = activation if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(dglnn.GraphConv(in_dim, out_dim, activation=activation, allow_zero_in_degree=True))
        else:
            self.gcn_layers.append(dglnn.GraphConv(in_dim, num_hidden, activation=activation, allow_zero_in_degree=True))
            for l in range(1, num_layers - 1):
                self.gcn_layers.append(dglnn.GraphConv(num_hidden, num_hidden, activation=activation))
            self.gcn_layers.append(dglnn.GraphConv(num_hidden, out_dim, activation=last_activation))
        self.norms = norm

    def forward(self, graph, inputs: torch.Tensor):
        h = inputs
        hidden_list = []
        for layer in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[layer](graph, h)
            if layer != self.num_layers - 1:
                h = self.norms(h)
            hidden_list.append(h)
        return h

    def inference(self, g, device="cuda", batch_size=128):
        feat = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        all_nid = torch.arange(g.num_nodes()).to(g.device)
        dataloader = DataLoader(g, all_nid, sampler, batch_size=batch_size, shuffle=False, drop_last=False)
        for i, layer in enumerate(self.gcn_layers):
            y = torch.empty(
                g.num_nodes(), self.num_hidden if i != self.num_layers - 1 else self.out_dim, device=device)
            for input_nodes, output_nodes, blocks in dataloader:
                x = feat[input_nodes]
                graph = blocks[0]
                h = layer(graph, x)
                if self.norms is not None and i != self.num_layers - 1:
                    h = self.norms[i](h)
                y[output_nodes[0]:output_nodes[-1] + 1] = h
            feat = y
        return feat

