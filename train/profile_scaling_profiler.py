# profile_scaling_profiler_tb.py
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN
import time
import gc
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Base config
vocab_size = 50257
ffn_expansion = 4
runs = 5

# Define scaling steps
scales = [
    {"batch": 16, "seq": 128, "hidden": 1536, "layers": 4},
    {"batch": 24, "seq": 192, "hidden": 2304, "layers": 6},
    {"batch": 32, "seq": 256, "hidden": 3072, "layers": 8},
]

# Replace FFN with sparse version
def patch_sparse_ffn(model, hidden_size):
    for block in model.transformer.h:
        block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
    return model

# FLOPs estimation per FFN layer
def estimate_ffn_flops(d_model, d_ff, seq_len, batch_size, sparsity=0.0, n_layers=1):
    dense_flops_per_token = 2 * d_model * d_ff * 2  # matmul + add
    total_dense_flops = dense_flops_per_token * seq_len * batch_size * n_layers
    total_sparse_flops = total_dense_flops * (1 - sparsity)
    return total_dense_flops, total_sparse_flops

# Profiling function
def profile_model(model, dummy_input, runs=5):
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

# Store results
results = []

# Loop over scales
for scale in scales:
    b = scale["batch"]
    s = scale["seq"]
    h = scale["hidden"]
    l = scale["layers"]

    print(f"\n=== Profiling: batch={b}, seq={s}, hidden={h}, layers={l} ===")
    dummy_input = torch.randint(0, vocab_size, (b, s), device=device)
    tb_writer = SummaryWriter(log_dir=f"profiler_logs/B{b}_S{s}_H{h}_L{l}")

    # Baseline GPT-2
    config = GPT2Config(n_embd=h, n_layer=l, n_head=12)
    baseline_model = GPT2LMHeadModel(config)
    dense_time, dense_mem = profile_model(baseline_model, dummy_input, runs)
    dense_flops, _ = estimate_ffn_flops(h, h*ffn_expansion, s, b, sparsity=0.0, n_layers=l)

    # Sparse FFN GPT-2
    sparse_model = GPT2LMHeadModel(config)
    sparse_model = patch_sparse_ffn(sparse_model, h)
    sparse_time, sparse_mem = profile_model(sparse_model, dummy_input, runs)
    _, sparse_flops = estimate_ffn_flops(h, h*ffn_expansion, s, b, sparsity=0.5, n_layers=l)

    speedup = dense_time / sparse_time
    mem_saving = dense_mem - sparse_mem

    # Log scalars to TensorBoard
    tb_writer.add_scalar("Dense/AvgTime_ms", dense_time*1000, 0)
    tb_writer.add_scalar("Sparse/AvgTime_ms", sparse_time*1000, 0)
    tb_writer.add_scalar("Dense/Memory_MB", dense_mem, 0)
    tb_writer.add_scalar("Sparse/Memory_MB", sparse_mem, 0)
    tb_writer.add_scalar("Sparse/Speedup", speedup, 0)
    tb_writer.add_scalar("Sparse/MemorySaving_MB", mem_saving, 0)

    # Per-layer FLOPs
    for i in range(l):
        tb_writer.add_scalar(f"Layer_{i+1}/Dense_GFLOPs", dense_flops/l/1e9, 0)
        tb_writer.add_scalar(f"Layer_{i+1}/Sparse_GFLOPs", sparse_flops/l/1e9, 0)

    tb_writer.close()

    results.append({
        "batch": b,
        "seq": s,
        "hidden": h,
        "layers": l,
        "dense_time": dense_time*1000,
        "dense_mem": dense_mem,
        "dense_flops": dense_flops/1e9,
        "sparse_time": sparse_time*1000,
        "sparse_mem": sparse_mem,
        "sparse_flops": sparse_flops/1e9,
        "speedup": speedup,
        "mem_saving": mem_saving
    })

# === Print table ===
print("\n" + "="*110)
print(f"{'Scale':<25} {'Dense Time(ms)':<15} {'Sparse Time(ms)':<15} {'Speedup':<10} {'Dense FLOPs(G)':<15} {'Sparse FLOPs(G)':<15}")
print("="*110)
for r in results:
    scale_name = f"B{r['batch']}-S{r['seq']}-H{r['hidden']}-L{r['layers']}"
    print(f"{scale_name:<25} {r['dense_time']:<15.2f} {r['sparse_time']:<15.2f} {r['speedup']:<10.2f} {r['dense_flops']:<15.2f} {r['sparse_flops']:<15.2f}")
print("="*110)

# === Plot results ===
scale_labels = [f"B{r['batch']}-S{r['seq']}-H{r['hidden']}-L{r['layers']}" for r in results]
dense_times = [r['dense_time'] for r in results]
sparse_times = [r['sparse_time'] for r in results]
speedups = [r['speedup'] for r in results]

plt.figure(figsize=(10,5))
plt.plot(scale_labels, dense_times, marker='o', label="Baseline GPT-2")
plt.plot(scale_labels, sparse_times, marker='o', label="Sparse FFN GPT-2")
plt.xticks(rotation=45)
plt.ylabel("Avg Forward Time (ms)")
plt.title("Scaling Profiling: Time vs Model Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(scale_labels, speedups, marker='o', color='green')
plt.xticks(rotation=45)
plt.ylabel("Speedup")
plt.title("Sparse FFN Speedup vs Baseline")
plt.grid(True)
plt.tight_layout()
plt.show()
