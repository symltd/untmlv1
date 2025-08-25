"""
Profile FLOPs, CUDA time, and memory for GPT-2 with UltraEfficientSparseFFN.
Modular design: swap between Dense FFN and Sparse FFN.
Compatible with mixed precision (fp16/bf16).
Logs and visualizes FLOP savings over training steps.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Config, GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN  # Custom FFN
import logging
import matplotlib.pyplot as plt

# --- Configurable parameters ---
MODEL_NAME = "gpt2"  # Use "gpt2" for small, "gpt2-medium" for medium
USE_SPARSE_FFN = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # Change to torch.bfloat16 if needed
PROFILE_STEPS = 20

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Profiler")

# --- Utility: Patch GPT-2 MLP with UltraEfficientSparseFFN ---
def patch_gpt2_ffn(model, use_sparse=True):
    for block in model.transformer.h:
        if use_sparse:
            block.mlp = UltraEfficientSparseFFN(
                block.mlp.c_fc.in_features,
                block.mlp.c_fc.out_features,
                top_k=8,  # Example hyperparam
                proj_dim=256,
                structured_sparsity=True,
            )
        # else: keep default dense MLP
    return model

# --- Load model and patch FFN ---
config = GPT2Config.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel(config)
model = patch_gpt2_ffn(model, use_sparse=USE_SPARSE_FFN)
model.to(DEVICE, dtype=DTYPE)
model.eval()

# --- Dummy input for profiling ---
input_ids = torch.randint(0, config.vocab_size, (1, 32), device=DEVICE)

# --- Profiling loop ---
flops_log, cuda_time_log, mem_log = [], [], []

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    use_cuda=torch.cuda.is_available(),
) as prof:
    for step in range(PROFILE_STEPS):
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
            with record_function("model_inference"):
                outputs = model(input_ids)
        prof.step()
        # Log metrics
        flops = prof.key_averages().total("flops")
        cuda_time = prof.key_averages().total("cuda_time")
        mem = prof.key_averages().total("self_cuda_memory_usage")
        flops_log.append(flops)
        cuda_time_log.append(cuda_time)
        mem_log.append(mem)
        logger.info(f"Step {step}: FLOPs={flops}, CUDA Time={cuda_time}, Mem={mem}")

# --- Visualization ---
plt.figure(figsize=(10, 4))
plt.plot(flops_log, label="FLOPs")
plt.plot(cuda_time_log, label="CUDA Time")
plt.plot(mem_log, label="Memory")
plt.xlabel("Step")
plt.ylabel("Metric Value")
plt.title("Profiling UltraEfficientSparseFFN vs Dense FFN")
plt.legend()
plt.tight_layout()
plt.savefig("profile_metrics.png")
logger.info("Saved profiling plot to profile_metrics.png")

# --- Notes ---
# - To profile dense FFN, set USE_SPARSE_FFN=False.
# - Extend UltraEfficientSparseFFN in models/sparse_ffn.py for more features.
# - For training/fine-tuning, use Hugging Face Trainer in train/train_sparse.py.