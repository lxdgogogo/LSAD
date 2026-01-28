import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler, NeighborSampler
from utils.trick import gumbel_softmax


class PG_few(nn.Module):
    def __init__(self, text_embedding, lamb, K=8, device="cuda"):
        super().__init__()
        self.device = device
        self.lamb = lamb
        self.K = K

        self.embedding = F.normalize(text_embedding, p=2, dim=1).to(device)

        d = self.embedding.shape[1]

        mean_dir = torch.mean(self.embedding.detach(), dim=0, keepdim=True)  # (1,d)
        mean_dir = F.normalize(mean_dir, dim=1)

        init = mean_dir.repeat(K, 1)  # (K,d)
        init = init + 0.01 * torch.randn_like(init)
        init = F.normalize(init, dim=1)

        # virtual: (K, d)
        self.virtual = nn.Parameter(init)

        self.bce_loss = nn.BCELoss()

    
    @staticmethod
    def remove_projection(x: torch.Tensor, virtual: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        # (K,K)
        G = virtual @ virtual.T # (K,K)
        K = G.shape[0]
        G = G + eps * torch.eye(K, device=G.device, dtype=G.dtype)

        # (N,K)
        coef = (x @ virtual.T) @ torch.inverse(G)  # (N * D @ D * K ) @ (K * K) -> N * K
        # (N,d)
        proj = coef @ virtual # N * D
        return x - proj


    def s_diff(self, edges_adj, sim_sub):
        diff = torch.norm(edges_adj - sim_sub, 2, dim=1)
        return torch.mean(diff)

    def forward(self, graph, train_nodes: torch.Tensor, train_label: torch.Tensor):
        sampler = MultiLayerFullNeighborSampler(1)

        train_nodes = train_nodes.to(torch.int32).to(self.device)
        score = torch.zeros_like(train_nodes, dtype=torch.float32, device=self.device)

        V = F.normalize(self.virtual, dim=1)  # (K,d)

        text_embedding_local = self.remove_projection(self.embedding, V)
        text_embedding_local = F.normalize(text_embedding_local, p=2, dim=1)

        for idx, node in enumerate(train_nodes):
            idx_batch, _, blocks = sampler.sample(graph, node)
            idx_batch = torch.sort(idx_batch)[0].to(self.device)

            text_embedding_batch = text_embedding_local[idx_batch]  # (n,d)
            sim = text_embedding_batch @ text_embedding_batch.T     # (n,n)

            sim = sim.clone()
            sim.fill_diagonal_(-1e9)

            sim_sub = gumbel_softmax(sim)

            edges_ego = blocks[0].edges()
            edges_ego = torch.stack(edges_ego, dim=0)

            n = idx_batch.shape[0]
            edges_adj = torch.zeros((n, n), device=self.device)
            edges_adj[edges_ego[0], edges_ego[1]] = 1

            score[idx] = self.s_diff(edges_adj, sim_sub)

        eps = 1e-8
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score) + eps)

        loss = self.bce_loss(score, train_label.float())

        return score, loss

    def compute_local_embedding(self):
        V = F.normalize(self.virtual, dim=1)
        text_embedding_local = self.remove_projection(self.embedding, V)
        text_embedding_local = F.normalize(text_embedding_local, p=2, dim=1)
        return text_embedding_local
    
    @torch.no_grad()
    def inference(self, graph, nodes: torch.Tensor):
        sampler = MultiLayerFullNeighborSampler(1)

        nodes = nodes.to(torch.int32).to(self.device)
        score = torch.zeros_like(nodes, dtype=torch.float32, device=self.device)

        V = F.normalize(self.virtual, dim=1)
        text_embedding_local = self.remove_projection(self.embedding, V)
        text_embedding_local = F.normalize(text_embedding_local, p=2, dim=1)

        for idx, node in enumerate(nodes):
            idx_batch, _, blocks = sampler.sample(graph, node)
            idx_batch = torch.sort(idx_batch)[0].to(self.device)

            text_embedding_batch = text_embedding_local[idx_batch]
            sim = text_embedding_batch @ text_embedding_batch.T

            sim = sim.clone()
            sim.fill_diagonal_(-1e9)

            sim_sub = gumbel_softmax(sim)

            edges_ego = blocks[0].edges()
            edges_ego = torch.stack(edges_ego, dim=0)

            n = idx_batch.shape[0]
            edges_adj = torch.zeros((n, n), device=self.device)
            edges_adj[edges_ego[0], edges_ego[1]] = 1

            score[idx] = self.s_diff(edges_adj, sim_sub)
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        return score

