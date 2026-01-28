import dgl
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from GNN.gcn import GCN
from utils.functions import create_activation, create_norm
from torch.utils.data import DataLoader


class AnomalyClip(nn.Module):
    def __init__(self, input_size, last_hidden_size, hidden_size, num_layers, dropout, norm, activation, alpha, batch_size, device):
        super().__init__()
        self.language_size = last_hidden_size
        self.encoder = GCN(in_dim=input_size, num_hidden=hidden_size, out_dim=hidden_size,
                           num_layers=num_layers, dropout=dropout, activation=create_activation(activation),
                           norm=create_norm(norm, hidden_size))
        self.decoder = GCN(in_dim=hidden_size, num_hidden=hidden_size, out_dim=input_size,
                           num_layers=num_layers, dropout=dropout, activation=create_activation(activation),
                           norm=create_norm(norm, hidden_size))
        self.device = device
        initial_value = torch.log(torch.tensor(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_value)
        self.text_projection = nn.Linear(self.language_size, hidden_size)
        self.alpha = alpha
        self.batch_size = batch_size

    def encode_gnn(self, graph, x: torch.Tensor):
        emb = self.encoder(graph, x)
        return emb

    def decode_gnn(self, graph, x: torch.Tensor):
        emb = self.decoder(graph, x)
        return emb

    def forward(self, graph, embeddings, feature, pre):
        s = self.encode_gnn(graph, feature)
        s_hat = self.decode_gnn(graph, s)
        x = self.text_projection(embeddings)
        s = s / s.norm(dim=0, keepdim=True)
        x = x / x.norm(dim=0, keepdim=True)
        logit_scale = self.logit_scale.exp()
        dataloader = DataLoader(graph.nodes(), batch_size=self.batch_size)
        score = torch.zeros_like(graph.nodes()).float()
        loss_sum = 0
        for batch in dataloader:
            # print(x.shape, s.shape)
            labels = torch.arange(batch.shape[0]).to(self.device)
            logits = logit_scale * x[batch] @ s[batch].T
            loss_i = F.cross_entropy(logits, labels, reduction='none')
            loss_t = F.cross_entropy(logits.T, labels, reduction='none')
            loss_clip = (loss_i + loss_t) / 2
            loss_anomaly = torch.norm(s_hat[batch] - feature[batch], dim=1)
            if pre:
                loss = loss_clip
            else:
                loss = (1 - self.alpha) * loss_anomaly + self.alpha * loss_clip
            score[batch] = (loss - torch.min(loss)) / (torch.max(loss) - torch.min(loss))
            loss_sum += loss.sum()
        return score, loss_sum
