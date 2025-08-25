import torch
import torch.nn as nn
import torch.nn.functional as F

class UltraEfficientSparseFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, k=32, sparsity=0.5):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.k = k
        self.sparsity = sparsity

    def forward(self, x):
        # First projection
        h = self.fc1(x)

        # Spectral sparsity (top-k by magnitude)
        if self.k < h.size(-1):
            topk_vals, topk_idx = torch.topk(h.abs(), self.k, dim=-1)
            mask = torch.zeros_like(h).scatter_(-1, topk_idx, 1.0)
            h = h * mask

        # Polynomial nonlinearity (replace GELU)
        h = h * h * h  # cubic activation (cheap FLOPs)

        # Structured sparsity (drop ratio)
        if self.sparsity > 0.0:
            keep = int(h.size(-1) * (1 - self.sparsity))
            idx = torch.topk(h.abs(), keep, dim=-1)[1]
            mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
            h = h * mask

        return self.fc2(h)
