import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


from collections import defaultdict


# ---------------- FLOPs HOOKS ---------------- #
FLOP_COUNTER = defaultdict(int)
BLOCK_COMPONENTS = defaultdict(lambda: defaultdict(int))  # block_id -> component -> flops

def count_component_flops(module, input, output, block_idx, component):
    """Count FLOPs for each MLP component and store per-block."""
    batch_size, seq_len, hidden_size = input[0].shape
    dense_flops = 2 * batch_size * seq_len * hidden_size * hidden_size

    flops = int(dense_flops)
    FLOP_COUNTER[f"block_{block_idx}_{component}"] += flops
    BLOCK_COMPONENTS[block_idx][component] += flops

def attach_flop_hooks(model):
    for i, block in enumerate(model.transformer.h):
        mlp = block.mlp
        if hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
            mlp.c_fc.register_forward_hook(lambda m, inp, out, i=i: count_component_flops(m, inp, out, i, "c_fc"))
            mlp.c_proj.register_forward_hook(lambda m, inp, out, i=i: count_component_flops(m, inp, out, i, "c_proj"))

def log_flops_to_tensorboard(step, writer):
    # Log per-component FLOPs
    for mod_name, flops in FLOP_COUNTER.items():
        writer.add_scalar(f"FLOPs/{mod_name}", flops, step)

    # Log cumulative per-block sparse FFN FLOPs
    for block_idx, comps in BLOCK_COMPONENTS.items():
        block_total = sum(comps.values())
        writer.add_scalar(f"FLOPs/block_{block_idx}_total", block_total, step)

    # Log total FLOPs across all modules
    total_flops = sum(FLOP_COUNTER.values())
    writer.add_scalar("FLOPs/total", total_flops, step)

    # Clear counters
    FLOP_COUNTER.clear()
    BLOCK_COMPONENTS.clear()

# ---------------- DATASET ---------------- #
def get_wikitext_dataloader(tokenizer, block_size=128, batch_size=8):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=block_size)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)

# ---------------- TRAINING ---------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = get_wikitext_dataloader(tokenizer)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )

    model = GPT2LMHeadModel(config).to(device)
    attach_flop_hooks(model)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    writer = SummaryWriter(log_dir="./runs/dense_gpt2")

    global_step = 0
    model.train()

    for epoch in range(3):
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            writer.add_scalar("Loss/train", loss.item(), global_step)

            # Log FLOPs
            log_flops_to_tensorboard(global_step, writer)

            if global_step % 50 == 0:
                print(f"Step {global_step} | Loss {loss.item():.4f}")
            global_step += 1

    writer.close()

if __name__ == "__main__":
    main()
