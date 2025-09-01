import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Config, GPT2LMHeadModel
import utils

class RandomTextDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=32, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = x.clone()
        return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_batch", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    local_rank, rank, world_size = utils.setup_ddp()

    # Model
    config = GPT2Config(
        vocab_size=50257,
        n_layer=2,
        n_head=2,
        n_embd=128,
    )
    model = GPT2LMHeadModel(config).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Data
    dataset = RandomTextDataset(num_samples=2000)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.per_device_batch, sampler=sampler)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    step = 0
    def train_loop(prof=None):
        nonlocal step
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            for x, y in dataloader:
                x, y = x.cuda(local_rank), y.cuda(local_rank)
                outputs = model(x, labels=y)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if rank == 0 and step % 10 == 0:
                    print(f"Step {step} | Loss {loss.item():.4f}")
                step += 1
                if step >= args.max_steps:
                    return

    if args.profile and rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                     record_shapes=True,
                     with_stack=True) as prof:
            train_loop(prof)
            prof.export_chrome_trace("trace.json")
    else:
        train_loop()

    utils.cleanup_ddp()

if __name__ == "__main__":
    main()
