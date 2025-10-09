# UltraEfficientSparseFFN
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def _shape_check(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [batch, seq, hidden], got {tuple(x.shape)}")
    return x

class SparseSpectral(nn.Module):
    def __init__(self, hidden_dim: int, k_freq: int = 128, learnable_bias: bool = True):
        super().__init__()
        self.d = hidden_dim
        self.k = max(1, min(k_freq, self.d // 2))
        self.gains = nn.Parameter(torch.ones(self.k))
        self.bias = nn.Parameter(torch.zeros(hidden_dim)) if learnable_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _shape_check(x)
        X = torch.fft.rfft(x, dim=-1)  # complex
        mag = torch.abs(X)
        rlen = X.size(-1)
        k = min(self.k, rlen)
        vals, idx = torch.topk(mag, k, dim=-1)
        mask = torch.zeros_like(X)
        mask.scatter_(-1, idx, 1.0)
        Xs = X * mask
        gains = self.gains.view(*(1,)* (Xs.dim()-1), -1)
        scale = torch.zeros_like(X, dtype=Xs.real.dtype)
        scale.scatter_(-1, idx, gains.expand_as(idx).to(scale.dtype))
        Xs = Xs * scale.to(Xs.dtype)
        y = torch.fft.irfft(Xs, n=self.d, dim=-1)
        if self.bias is not None:
            y = y + self.bias
        return y

class SparsePolynomial(nn.Module):
    def __init__(self, hidden_dim: int, degree: int = 3, keep_ratio: float = 0.5, soft_ratio: float = 0.1):
        super().__init__()
        self.d = hidden_dim
        self.K = max(1, degree)
        self.keep_ratio = float(min(1.0, max(0.0, keep_ratio)))
        self.soft_ratio = float(min(1.0, max(0.0, soft_ratio)))
        self.coeffs = nn.Parameter(torch.randn(self.K) / (self.K ** 0.5))
        self.importance = nn.Parameter(torch.zeros(self.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        keep = max(1, int(D * self.keep_ratio))
        use_soft = self.training and (random.random() < self.soft_ratio)

        if use_soft:
            # differentiable sigmoid gating
            mask = torch.sigmoid(self.importance)
            mask = mask / (mask.mean() + 1e-6)
            x_masked = x * mask
        else:
            # fast top-k gating
            with torch.no_grad():
                topk_idx = torch.topk(self.importance, keep, dim=-1).indices
            x_masked = torch.zeros_like(x)
            x_masked[..., topk_idx] = x[..., topk_idx]

        # polynomial expansion
        y = torch.zeros_like(x_masked)
        x_power = x_masked
        for k in range(self.K):
            y = y + self.coeffs[k] * x_power
            x_power = x_power * x_masked

        if not use_soft:
            y_full = x.clone()
            y_full[..., topk_idx] = y[..., topk_idx]
            y = y_full

        return y

class SparseMicroRefine(nn.Module):
    def __init__(self, hidden_dim: int, steps: int = 2, keep_ratio: float = 0.25,
                 activation: str = "silu", soft_ratio: float = 0.1):
        super().__init__()
        self.d = hidden_dim
        self.steps = max(0, steps)
        self.keep_ratio = float(min(1.0, max(0.0, keep_ratio)))
        self.soft_ratio = float(min(1.0, max(0.0, soft_ratio)))
        self.importance = nn.Parameter(torch.zeros(self.d))
        self.act = nn.SiLU() if activation == "silu" else nn.GELU()
        self.linears = nn.ModuleList([nn.Linear(1, 1, bias=True) for _ in range(self.steps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.steps == 0 or self.keep_ratio <= 0.0:
            return x
        B, T, D = x.shape
        keep = max(1, int(D * self.keep_ratio))
        use_soft = self.training and (random.random() < self.soft_ratio)

        if use_soft:
            mask = torch.sigmoid(self.importance)
            mask = mask / (mask.mean() + 1e-6)
            x_masked = x * mask
        else:
            with torch.no_grad():
                idx = torch.topk(self.importance, keep, dim=-1).indices
            x_masked = torch.zeros_like(x)
            x_masked[..., idx] = x[..., idx]

        # refinement microsteps
        xt = x_masked.reshape(B*T, D, 1)
        for lin in self.linears:
            xt = self.act(lin(xt))
        y = xt.reshape(B, T, D)

        if not use_soft:
            y_full = x.clone()
            y_full[..., idx] = y[..., idx]
            y = y_full

        return y

class UltraEfficientSparseFFN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        k_freq: int = 128,
        poly_degree: int = 3,
        poly_keep_ratio: float = 0.5,
        micro_steps: int = 2,
        micro_keep_ratio: float = 0.25,
        dropout: float = 0.0,
        use_spectral: bool = True,
        use_polynomial: bool = True,
        use_micro: bool = True,
        residual_gate_init: float = 1.0,
    ):
        super().__init__()
        self.d = hidden_dim
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.use_spectral = use_spectral
        self.use_polynomial = use_polynomial
        self.use_micro = use_micro

        if use_spectral:
            self.spectral = SparseSpectral(hidden_dim, k_freq=k_freq, learnable_bias=True)
        if use_polynomial:
            self.poly = SparsePolynomial(hidden_dim, degree=poly_degree, keep_ratio=poly_keep_ratio)
        if use_micro:
            self.micro = SparseMicroRefine(hidden_dim, steps=micro_steps, keep_ratio=micro_keep_ratio)

        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _shape_check(x)
        h = self.ln_in(x)
        if self.use_spectral:
            h = self.spectral(h)
        if self.use_polynomial:
            h = self.poly(h)
        if self.use_micro:
            h = self.micro(h)
        h = self.ln_out(h)
        h = self.drop(self.residual_proj(h))
        return x + self.gate * h
