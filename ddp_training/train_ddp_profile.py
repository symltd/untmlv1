# profile_compare_profiler.py
# from models.sparse_ffn import UltraEfficientSparseFFN
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import argparse
import gc
import os
import random
import string
import time
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# CLI Arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--per_device_batch", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--logdir", type=str, default="./logs")
args = parser.parse_args()

# ------------------------------
# Random Text Dataset
# ------------------------------
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=10000, seq_len=128):
        self.tokenizer = tokenizer
        self.samples = []
        for _ in range(num_samples):
            text = ''.join(random.choices(string.ascii_letters + ' ', k=seq_len))
            enc = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
            self.samples.append(enc["input_ids"].squeeze(0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch, batch  # input_ids and labels

# ------------------------------
# DDP Setup
# ------------------------------
def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return rank, world_size

# ------------------------------
# Patch GPT-2 with sparse FFN
# ------------------------------
# def patch_sparse_ffn(model, hidden_size, device):
#     for block in model.transformer.h:
#         block.mlp = UltraEfficientSparseFFN(hidden_size).to(device)
#     return model

# ------------------------------
# Estimate FLOPs per FFN layer
# ------------------------------
def estimate_ffn_flops(d_model, d_ff, seq_len, batch_size, sparsity=0.0):
    dense_flops = 2 * d_model * d_ff * 2 * seq_len * batch_size  # 2 matmuls per FFN
    sparse_flops = dense_flops * (1 - sparsity)
    return dense_flops, sparse_flops

# ------------------------------
# === Baseline GPT-2 ===
# config = GPT2Config(n_embd=hidden_size, n_layer=n_layers, n_head=16 if device=="cuda" else 12)
# baseline_model = GPT2LMHeadModel(config)
# baseline_time, baseline_mem, baseline_prof = profile_model(baseline_model, "Baseline GPT-2")

# EXCLUDE for starter === Sparse FFN GPT-2 ===
# sparse_model = GPT2LMHeadModel(config)
# sparse_model = patch_sparse_ffn(sparse_model, hidden_size)
# sparse_time, sparse_mem, sparse_prof = profile_model(sparse_model, "Sparse FFN GPT-2")

# EXCLUDE for starter Summary Table
# speedup = baseline_time / sparse_time
# print("\n" + "="*80)
# print(f"{'Model':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
# print("="*80)
# print(f"{'Baseline GPT-2':<20} {baseline_time*1000:<12.2f} {baseline_mem:<12.2f} {'1.00x':<10}")
# print(f"{'Sparse FFN GPT-2':<20} {sparse_time*1000:<12.2f} {sparse_mem:<12.2f} {speedup:.2f}x")
# print("="*80)
# print("\n[INFO] Profiler traces saved to TensorBoard logs under 'profiler_logs/'")


# ------------------------------
# Training Loop
# ------------------------------
def train(name, rank, world_size):
    device = torch.device(f"cuda:{rank}")

    # TensorBoard logger (only rank 0)
    log_path = args.logdir + f"/{name.replace(' ', '_')}"
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=log_path)
    
    # Load tokenizer (reuse GPT-2 vocab)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Config
    vocab_size=len(tokenizer)
    n_positions=128
    n_ctx=128
    n_embd=768
    n_layer=4
    n_head=12
    batch_size=args.per_device_batch
    ffn_expansion=4


    # GPT-2 config (train from scratch)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )
    model = GPT2LMHeadModel(config)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    # Dataset / loader
    dataset = RandomTextDataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        sampler=sampler,
        collate_fn=collate_fn
    )

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Profiler (optional)
    prof = None
    if args.profile and rank == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(log_path),
        )
        prof.__enter__()

    step_count = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for input_ids, labels in dataloader:
            if args.max_steps > 0 and step_count >= args.max_steps:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with record_function("model_forward"):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss

            with record_function("model_backward"):
                loss.backward()
                optimizer.step()

            if rank == 0:
                #if step_count % 50 == 0:
                if step_count % 5 == 0:
                    print(f"[Rank {rank}] Epoch {epoch} Step {step_count} Loss {loss.item():.4f}")
                    if writer:
                        writer.add_scalar("Loss/train", loss.item(), step_count)
                        for i in range(n_layer):
                            dense, sparse = estimate_ffn_flops(
                                n_embd,
                                n_embd * ffn_expansion,
                                n_ctx,
                                batch_size,
                                sparsity=0.5 if "Sparse" in name else 0.0,
                            )
                            writer.add_scalar(f"Layer_{i+1}_Dense_GFLOPs", dense / 1e9, step_count)
                            writer.add_scalar(f"Layer_{i+1}_Sparse_GFLOPs", sparse / 1e9, step_count)
                            
                            mem = torch.cuda.max_memory_allocated() / 1024**2
                            writer.add_scalar(f"PeakMemoryMB", mem, step_count)

                # Log profiler stats
                # if prof is not None and step_count % 20 == 0 and step_count > 0:
                if prof is not None:
                    if step_count % 2 == 0 and step_count > 0:
                        torch.cuda.synchronize()
                        mem = torch.cuda.max_memory_allocated() / 1024**2
                    prof.step()
            step_count += 1

    # Close profiler and writer
    if prof is not None:
        prof.__exit__(None, None, None)
    if writer is not None:
        writer.close()



# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    rank, world_size = setup_ddp()
    train("Baseline-GPT2", rank, world_size)
