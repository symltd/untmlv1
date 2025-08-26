# profile_compare_full_with_flops.py
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN
import time
import gc

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Config
batch_size = 16 if device=="cuda" else 16
seq_len = 128 if device=="cuda" else 128
hidden_size = 768*8 if device=="cuda" else 768*4
vocab_size = 50257
n_layers = 4 if device=="cuda" else 2
runs = 5

dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# Replace FFN with sparse version
def patch_sparse_ffn(model, hidden_size):
    for block in model.transformer.h:
        block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
    return model

# FLOPs estimation for FFN
def estimate_ffn_flops(d_model, d_ff, seq_len, batch_size, sparsity=0.0, n_layers=1):
    dense_flops_per_token = 2 * d_model * d_ff * 2  # multiply + add
    total_dense_flops = dense_flops_per_token * seq_len * batch_size * n_layers
    total_sparse_flops = total_dense_flops * (1 - sparsity)
    return total_dense_flops, total_sparse_flops

# Profiling function using torch.profiler
def profile_model(model, name="Model", runs=5):
    model.eval().to(device)
    gc.collect()
    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    if device=="cuda":
        torch.cuda.synchronize()
    avg_time = (time.time() - start)/runs

    mem = torch.cuda.max_memory_allocated()/1024**2 if device=="cuda" else sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    return avg_time, mem

# === Baseline GPT-2 ===
config = GPT2Config(n_embd=hidden_size, n_layer=n_layers, n_head=16 if device=="cuda" else 12)
baseline_model = GPT2LMHeadModel(config)
baseline_time, baseline_mem = profile_model(baseline_model, "Baseline GPT-2", runs=runs)
baseline_dense_flops, baseline_sparse_flops = estimate_ffn_flops(
    d_model=hidden_size,
    d_ff=hidden_size*4,
    seq_len=seq_len,
    batch_size=batch_size,
    sparsity=0.0,
    n_layers=n_layers
)

# === Sparse FFN GPT-2 ===
sparse_model = GPT2LMHeadModel(config)
sparse_model = patch_sparse_ffn(sparse_model, hidden_size)
sparse_time, sparse_mem = profile_model(sparse_model, "Sparse FFN GPT-2", runs=runs)
sparse_dense_flops, sparse_sparse_flops = estimate_ffn_flops(
    d_model=hidden_size,
    d_ff=hidden_size*4,
    seq_len=seq_len,
    batch_size=batch_size,
    sparsity=0.5,   # assumed 50% sparsity
    n_layers=n_layers
)

# === Compute speedup and memory savings ===
speedup = baseline_time / sparse_time
mem_saving = baseline_mem - sparse_mem

# === Side-by-side table ===
print("\n" + "="*90)
print(f"{'Model':<25} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10} {'FFNs FLOPs (GFLOPs)':<20}")
print("="*90)
print(f"{'Baseline GPT-2':<25} {baseline_time*1000:<12.2f} {baseline_mem:<12.2f} {'1.00x':<10} {baseline_dense_flops/1e9:<20.2f}")
print(f"{'Sparse FFN GPT-2':<25} {sparse_time*1000:<12.2f} {sparse_mem:<12.2f} {speedup:.2f}x {sparse_sparse_flops/1e9:<20.2f}")
print("="*90)
