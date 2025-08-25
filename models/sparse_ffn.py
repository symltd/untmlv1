import torch
import torch.nn as nn

class UltraEfficientSparseFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, k=32, sparsity=0.5):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.k = k
        self.sparsity = sparsity

    def forward(self, x):
        h = self.fc1(x)
        if self.k < h.size(-1):
            topk_vals, topk_idx = torch.topk(h.abs(), self.k, dim=-1)
            mask = torch.zeros_like(h).scatter_(-1, topk_idx, 1.0)
            h = h * mask
        h = h * h * h
        if self.sparsity > 0.0:
            keep = int(h.size(-1) * (1 - self.sparsity))
            idx = torch.topk(h.abs(), keep, dim=-1)[1]
            mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
            h = h * mask
        return self.fc2(h)
