# profile_compare_full_per_layer.py
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
ffn_expansion = 4  # GPT-2 default

dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# Replace FFN with sparse version
def patch_sparse_ffn(model, hidden_size):
    for block in model.transformer.h:
        block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
    return model

# FLOPs estimation per FFN layer
def estimate_ffn_flops_per_layer(d_model, d_ff, seq_len, batch_size, sparsity=0.0):
    """
    Returns FLOPs for a single FFN layer
    """
    dense_flops = 2 * d_model * d_ff * 2 * seq_len * batch_size  # 2 matmuls (mul+add)
    sparse_flops = dense_flops * (1 - sparsity)
    return dense_flops, sparse_flops

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

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    if device=="cuda":
        torch.cuda.synchronize()
    avg_time = (time.time() - start)/runs

    # Memory
    mem = torch.cuda.max_memory_allocated()/1024**2 if device=="cuda" else sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    return avg_time, mem

# === Baseline GPT-2 ===
config = GPT2Config(n_embd=hidden_size, n_layer=n_layers, n_head=16 if device=="cuda" else 12)
baseline_model = GPT2LMHeadModel(config)
baseline_time, baseline_mem = profile_model(baseline_model, "Baseline GPT-2", runs=runs)

baseline_ffn_flops = []
for _ in range(n_layers):
    dense, sparse = estimate_ffn_flops_per_layer(hidden_size, hidden_size*ffn_expansion, seq_len, batch_size, sparsity=0.0)
    baseline_ffn_flops.append((dense, sparse))

# === Sparse FFN GPT-2 ===
sparse_model = GPT2LMHeadModel(config)
sparse_model = patch_sparse_ffn(sparse_model, hidden_size)
sparse_time, sparse_mem = profile_model(sparse_model, "Sparse FFN GPT-2", runs=runs)

sparse_ffn_flops = []
sparsity = 0.5  # assumed 50% neurons active
for _ in range(n_layers):
    dense, sparse_f = estimate_ffn_flops_per_layer(hidden_size, hidden_size*ffn_expansion, seq_len, batch_size, sparsity=sparsity)
    sparse_ffn_flops.append((dense, sparse_f))

# === Compute speedup and memory savings ===
speedup = baseline_time / sparse_time
mem_saving = baseline_mem - sparse_mem

# === Print summary table ===
print("\n" + "="*100)
print(f"{'Model':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10} {'FFN Layer FLOPs (GFLOPs)':<40}")
print("="*100)
baseline_flops_total = sum([f[0] for f in baseline_ffn_flops])
sparse_flops_total = sum([f[1] for f in sparse_ffn_flops])

print(f"{'Baseline GPT-2':<20} {baseline_time*1000:<12.2f} {baseline_mem:<12.2f} 1.00x{' ':<10}{baseline_flops_total/1e9:<40.2f}")
print(f"{'Sparse FFN GPT-2':<20} {sparse_time*1000:<12.2f} {sparse_mem:<12.2f} {speedup:.2f}x{' ':<10}{sparse_flops_total/1e9:<40.2f}")
print("="*100)

# === Detailed per-layer FLOPs ===
print("\nPer-layer FFN FLOPs (GFLOPs):")
for i, (b, s) in enumerate(zip(baseline_ffn_flops, sparse_ffn_flops)):
    print(f"Layer {i+1}: Baseline={b[0]/1e9:.2f} GFLOPs, Sparse={s[1]/1e9:.2f} GFLOPs, Reduction={(1 - s[1]/b[0])*100:.1f}%")
