# profile_compare_profiler.py
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN
import time
import gc
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Config
batch_size = 16 if device=="cuda" else 16
seq_len = 128 if device=="cuda" else 128
hidden_size = 1024 if device=="cuda" else 768*8
vocab_size = 50257
n_layers = 4 if device=="cuda" else 4
ffn_expansion = 4
runs = 10

dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# Patch GPT-2 with sparse FFN
def patch_sparse_ffn(model, hidden_size):
    for block in model.transformer.h:
        block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
    return model

# Estimate FLOPs per FFN layer
def estimate_ffn_flops(d_model, d_ff, seq_len, batch_size, sparsity=0.0):
    dense_flops = 2 * d_model * d_ff * 2 * seq_len * batch_size  # 2 matmuls per FFN
    sparse_flops = dense_flops * (1 - sparsity)
    return dense_flops, sparse_flops

# Profile function using torch.profiler
def profile_model(model, name="Model", runs=5, log_dir="profiler_logs"):
    model.eval().to(device)
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_input)

    # Profiler context
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    log_path = log_dir + f"/{name.replace(' ', '_')}"
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(log_path),
    ) as prof:
        with torch.no_grad():
            for step in range(runs):
                with record_function(name):
                    _ = model(dummy_input)
                prof.step()

    # ---- Time & Memory ----
    if device == "cuda":
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**2
    else:
        mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    events = prof.key_averages()
    if len(events) > 0:
        total_time = sum(evt.cpu_time_total for evt in events) / 1e6 / runs  # sec
    else:
        total_time = float("nan")

    print(f"[{name}] Avg forward time: {total_time*1000:.2f} ms, Peak memory: {mem:.2f} MB")
    print(f"[INFO] Profiler traces saved to: {log_path}")

    # ---- TensorBoard scalars ----
    writer = SummaryWriter(log_path + "/scalars")
    writer.add_scalar("AvgTime_ms", total_time * 1000, 0)
    writer.add_scalar("PeakMemory_MB", mem, 0)

    for i in range(n_layers):
        dense, sparse = estimate_ffn_flops(
            hidden_size,
            hidden_size * ffn_expansion,
            seq_len,
            batch_size,
            sparsity=0.5 if "Sparse" in name else 0.0,
        )
        writer.add_scalar(f"Layer_{i+1}_Dense_GFLOPs", dense / 1e9, 0)
        writer.add_scalar(f"Layer_{i+1}_Sparse_GFLOPs", sparse / 1e9, 0)
    writer.close()

    # ---- Console per-layer FLOPs ----
    print(f"\nPer-layer FFN FLOPs (GFLOPs) for {name}:")
    for i in range(n_layers):
        dense, sparse = estimate_ffn_flops(
            hidden_size,
            hidden_size * ffn_expansion,
            seq_len,
            batch_size,
            sparsity=0.5 if "Sparse" in name else 0.0,
        )
        print(
            f"Layer {i+1}: Dense={dense/1e9:.2f} GFLOPs, "
            f"Sparse={sparse/1e9:.2f} GFLOPs, "
            f"Reduction={(1 - sparse/dense) * 100:.1f}%"
        )

    return total_time, mem, prof

# === Baseline GPT-2 ===
config = GPT2Config(n_embd=hidden_size, n_layer=n_layers, n_head=16 if device=="cuda" else 12)
baseline_model = GPT2LMHeadModel(config)
baseline_time, baseline_mem, baseline_prof = profile_model(baseline_model, "Baseline GPT-2")

# === Sparse FFN GPT-2 ===
sparse_model = GPT2LMHeadModel(config)
sparse_model = patch_sparse_ffn(sparse_model, hidden_size)
sparse_time, sparse_mem, sparse_prof = profile_model(sparse_model, "Sparse FFN GPT-2")

# Summary Table
speedup = baseline_time / sparse_time
print("\n" + "="*80)
print(f"{'Model':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
print("="*80)
print(f"{'Baseline GPT-2':<20} {baseline_time*1000:<12.2f} {baseline_mem:<12.2f} {'1.00x':<10}")
print(f"{'Sparse FFN GPT-2':<20} {sparse_time*1000:<12.2f} {sparse_mem:<12.2f} {speedup:.2f}x")
print("="*80)
print("\n[INFO] Profiler traces saved to TensorBoard logs under 'profiler_logs/'")
