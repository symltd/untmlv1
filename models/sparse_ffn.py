# UltraEfficientSparseFFN
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def _shape_check(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [batch, seq, hidden], got {tuple(x.shape)}")
    return x


class SparsePolynomial(nn.Module):
    """
    Differentiable sparse polynomial activation.

    During training:
      - Uses a soft sigmoid-based mask on `importance` for differentiable gating.
      - No need for find_unused_parameters=True under DDP.

    During inference (model.eval()):
      - Applies hard top-k selection for true sparsity.

    Args:
        hidden_dim: Transformer hidden size.
        degree: Polynomial expansion degree.
        keep_ratio: Fraction of dimensions to keep active.
    """

    def __init__(self, hidden_dim: int, degree: int = 3, keep_ratio: float = 0.5):
        super().__init__()
        self.d = hidden_dim
        self.K = max(1, degree)
        self.keep_ratio = float(min(1.0, max(0.0, keep_ratio)))

        self.coeffs = nn.Parameter(torch.randn(self.K) / (self.K ** 0.5))
        self.importance = nn.Parameter(torch.zeros(self.d))  # learnable selector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _shape_check(x)
        B, T, D = x.shape

        if self.training:
            # ---- Soft differentiable selection ----
            mask = torch.sigmoid(self.importance).view(1, 1, D)  # [1,1,D]
            mask = mask / (mask.mean(dim=-1, keepdim=True) + 1e-6)
            x_masked = x * mask

            # Polynomial expansion over all dims (weighted by mask)
            y = torch.zeros_like(x_masked)
            x_power = x_masked
            for k in range(self.K):
                y = y + self.coeffs[k] * x_power
                x_power = x_power * x_masked

        else:
            # ---- Hard top-k selection for inference ----
            keep = max(1, int(D * self.keep_ratio))
            idx = torch.topk(self.importance, keep, dim=-1).indices  # [keep]
            x_act = x[..., idx]  # [B,T,keep]
            y_act = torch.zeros_like(x_act)
            x_power = x_act
            for k in range(self.K):
                y_act = y_act + self.coeffs[k] * x_power
                x_power = x_power * x_act
            y = x.clone()
            y[..., idx] = y_act

        return y

class SparseMicroRefine(nn.Module):
    """
    Differentiable sparse refinement block.

    During training:
      - Uses a soft sigmoid-based mask (importance -> [0,1]) for differentiable gating.
      - No need for find_unused_parameters=True under DDP.

    During inference (model.eval()):
      - Uses hard top-k selection for actual sparsity.

    Args:
        hidden_dim: Transformer hidden size.
        steps: Number of micro refinement layers.
        keep_ratio: Fraction of neurons to refine (sparsity level).
        activation: 'silu' or 'gelu'.
    """

    def __init__(
        self,
        hidden_dim: int,
        steps: int = 2,
        keep_ratio: float = 0.25,
        activation: str = "silu",
    ):
        super().__init__()
        self.d = hidden_dim
        self.steps = max(0, steps)
        self.keep_ratio = float(min(1.0, max(0.0, keep_ratio)))
        self.importance = nn.Parameter(torch.zeros(hidden_dim))  # learnable selector

        self.act = nn.SiLU() if activation == "silu" else nn.GELU()
        self.linears = nn.ModuleList([nn.Linear(1, 1, bias=True) for _ in range(self.steps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.steps == 0 or self.keep_ratio <= 0.0:
            return x

        B, T, D = x.shape
        h = x.clone()

        if self.training:
            # ---- Soft differentiable selection ----
            # Learnable continuous mask in [0, 1]
            mask = torch.sigmoid(self.importance).view(1, 1, D)  # [1,1,D]
            # Normalize to keep energy comparable
            mask = mask / (mask.mean(dim=-1, keepdim=True) + 1e-6)
            # Apply mask softly
            h = h * mask
            xt = h.reshape(B * T * D, 1)
            for lin in self.linears:
                xt = self.act(lin(xt))
            y = xt.reshape(B, T, D)
            # blend back (residual)
            out = x + y

        else:
            # ---- Hard top-k for inference ----
            keep = max(1, int(D * self.keep_ratio))
            idx = torch.topk(self.importance, keep, dim=-1).indices
            xt = h[..., idx].reshape(B * T, keep, 1)
            for lin in self.linears:
                xt = self.act(lin(xt))
            h[..., idx] = xt.reshape(B, T, keep)
            out = h

        return out


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
