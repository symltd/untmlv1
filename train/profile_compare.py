# profile_compare_profiler.py
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size, seq_len = 2, 32
hidden_size = 768
dummy_input = torch.randint(0, 50257, (batch_size, seq_len), device=device)

def profile_model(model, name="Model"):
    print(f"\n=== Profiling {name} ===")
    model.eval()
    model.to(device)
    # Clear CUDA memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./logs/{name.replace(' ', '_')}"),
    ) as prof:
        with torch.no_grad():
            out = model(dummy_input)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[{name}] Peak memory: {mem:.2f} MB")
    return out

# Baseline GPT-2
config = GPT2Config(n_embd=hidden_size, n_layer=2, n_head=12)
baseline_model = GPT2LMHeadModel(config)
profile_model(baseline_model, name="Baseline GPT-2")

# Sparse FFN GPT-2
sparse_model = GPT2LMHeadModel(config)
for block in sparse_model.transformer.h:
    block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
profile_model(sparse_model, name="Sparse FFN GPT-2")

print("\n[✔] Profiling complete. View logs with:")
print("    tensorboard --logdir=./logs")
