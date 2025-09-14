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
from torch.utils.tensorboard import SummaryWriter

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
    return rank, world_size

# ------------------------------
# Training Loop
# ------------------------------
def train(rank, world_size):
    device = torch.device(f"cuda:{rank}")

    # TensorBoard logger (only rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)

    # Load tokenizer (reuse GPT-2 vocab)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # GPT-2 config (train from scratch)
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
            with_stack=False,  # keep lighter
            on_trace_ready=None  # we handle logging manually
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
                if step_count % 50 == 0:
                    print(f"[Rank {rank}] Epoch {epoch} Step {step_count} Loss {loss.item():.4f}")
                    if writer:
                        writer.add_scalar("Loss/train", loss.item(), step_count)

                # Log profiler stats
                if prof is not None:
                    if step_count % 20 == 0 and step_count > 0:
                        events = prof.key_averages(group_by_input_shape=True)
                        for evt in events:
                            # memory in MB
                            mem = (evt.cuda_memory_usage / (1024 * 1024)) if evt.cuda_memory_usage else 0
                            flops = evt.flops if hasattr(evt, "flops") else 0
                            # sanitize layer name
                            name = evt.key.replace("/", "_").replace(" ", "_")
                            writer.add_scalar(f"FLOPs/{name}", flops, step_count)
                            writer.add_scalar(f"MemoryMB/{name}", mem, step_count)
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
    train(rank, world_size)
