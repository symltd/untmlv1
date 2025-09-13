import os
import argparse
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.profiler import profile, record_function, ProfilerActivity

# ------------------------------
# CLI Arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--per_device_batch", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--profile", action="store_true")
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
    return rank, world_size

# ------------------------------
# Training Loop
# ------------------------------
def train(rank, world_size):
    device = torch.device(f"cuda:{rank}")

    # Load model and tokenizer if you look for fine-tuning
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # else, for training from scratch:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Define new model config
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_ctx=128,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    # Dataset and DataLoader
    dataset = RandomTextDataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        sampler=sampler,
        collate_fn=collate_fn
    )

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Profiler context
    prof = None
    if args.profile and rank == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
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

            if rank == 0 and step_count % 50 == 0:
                print(f"[Rank {rank}] Epoch {epoch} Step {step_count} Loss {loss.item():.4f}")

            step_count += 1

    # Finish profiler
    if prof is not None:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace("ddp_gpt2_profiler_trace.json")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    rank, world_size = setup_ddp()
    train(rank, world_size)
