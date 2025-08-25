# profile_sparse.py
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN

# === Replace GPT-2 FFN with our UltraEfficientSparseFFN ===
def patch_gpt2_with_sparse_ffn(model, hidden_size):
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for i, sub in enumerate(module):
                if isinstance(sub, nn.Linear) and sub.in_features == hidden_size:
                    # Replace first linear layer with UltraEfficientSparseFFN
                    module[i] = UltraEfficientSparseFFN(hidden_size)
                    print(f"Replaced FFN at {name}.{i} with UltraEfficientSparseFFN")
                    return model
    return model

# === Main profiling routine ===
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPT-2 small config
    config = GPT2Config(n_embd=768, n_layer=2, n_head=12)
    model = GPT2LMHeadModel(config).to(device)

    # Patch FFN
    model = patch_gpt2_with_sparse_ffn(model, hidden_size=config.n_embd)

    # Dummy input
    batch_size, seq_len = 2, 32
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Profile with torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
    ) as prof:
        for _ in range(5):  # warmup + multiple runs
            outputs = model(dummy_input)

    # Print profiling summary
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    ))

    print("\n[✔] Profiling complete. Open with:")
    print("    tensorboard --logdir=./logs")
    print("Then visit http://localhost:6006 in your browser.")

if __name__ == "__main__":
    main()
