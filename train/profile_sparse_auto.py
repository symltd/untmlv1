# profile_sparse_auto.py
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN
import time
import gc

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

batch_size = 16 if device=="cuda" else 2
seq_len = 128 if device=="cuda" else 32
hidden_size = 1024 if device=="cuda" else 768
vocab_size = 50257
n_layers = 4 if device=="cuda" else 2
runs = 5

dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# === Utility to replace FFN ===
def patch_sparse_ffn(model, hidden_size):
    for block in model.transformer.h:
        block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
    return model

# === Profiling function ===
def profile_forward(model, name="Model", runs=5):
    model.eval()
    model.to(device)
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
        if device=="cuda":
            from torch.amp import autocast
            autocast_ctx = autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        with autocast_ctx:
            for _ in range(runs):
                _ = model(dummy_input)
    if device=="cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs

    # Memory usage estimate
    if device=="cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
    else:
        mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    return elapsed, mem

# === Baseline GPT-2 ===
config = GPT2Config(n_embd=hidden_size, n_layer=n_layers, n_head=16 if device=="cuda" else 12)
baseline_model = GPT2LMHeadModel(config)
baseline_time, baseline_mem = profile_forward(baseline_model, "Baseline GPT-2", runs=runs)

# === Sparse FFN GPT-2 ===
sparse_model = GPT2LMHeadModel(config)
sparse_model = patch_sparse_ffn(sparse_model, hidden_size)
sparse_time, sparse_mem = profile_forward(sparse_model, "Sparse FFN GPT-2", runs=runs)

# === Compute speedup and memory savings ===
speedup = baseline_time / sparse_time
mem_saving = baseline_mem - sparse_mem

# === Print comparison table ===
print("\n" + "="*50)
print(f"{'Model':<20} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
print("="*50)
print(f"{'Baseline GPT-2':<20} {baseline_time*1000:<15.2f} {baseline_mem:<15.2f} {'1.00x':<10}")
print(f"{'Sparse FFN GPT-2':<20} {sparse_time*1000:<15.2f} {sparse_mem:<15.2f} {speedup:.2f}x")
print("="*50)
