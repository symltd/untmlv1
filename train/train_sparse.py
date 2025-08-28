import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2Tokenizer

from models import GPT2Sparse, GPT2SparseConfig, UltraEfficientSparseFFN

from collections import defaultdict

# ---------------- FLOPs HOOKS ---------------- #
FLOP_COUNTER = defaultdict(int)
BLOCK_COMPONENTS = defaultdict(lambda: defaultdict(int))  # block_id -> component -> flops

def count_linear_flops(module, input, output):
    """Count FLOPs for a single nn.Linear."""
    batch_size = input[0].shape[0]
    in_features = module.in_features
    out_features = module.out_features
    FLOP_COUNTER[f"{module.__class__.__name__}"] += 2 * batch_size * in_features * out_features

def count_sparse_component_flops(module, input, output, block_idx, component):
    """Count FLOPs for each UltraEfficientSparseFFN component and store per-block."""
    batch_size, seq_len, hidden_size = input[0].shape
    dense_flops = 2 * batch_size * seq_len * hidden_size * hidden_size

    sparsity_factor = 1.0
    if isinstance(module, UltraEfficientSparseFFN):
        if component == "spectral":
            sparsity_factor *= module.k / hidden_size if hasattr(module, "k") else 1.0
        elif component == "poly":
            sparsity_factor *= module.K / hidden_size if hasattr(module, "K") else 1.0
        elif component == "micro":
            sparsity_factor *= module.steps / hidden_size if hasattr(module, "steps") else 1.0

    flops = int(dense_flops * sparsity_factor)
    FLOP_COUNTER[f"block_{block_idx}_{component}"] += flops
    BLOCK_COMPONENTS[block_idx][component] += flops

def attach_flop_hooks(model):
    for i, block in enumerate(model.transformer.h):
        mlp = block.mlp
        if isinstance(mlp, UltraEfficientSparseFFN):
            if mlp.use_spectral:
                mlp.spectral.register_forward_hook(
                    lambda m, inp, out, i=i: count_sparse_component_flops(m, inp, out, i, "spectral")
                )
            if mlp.use_polynomial:
                mlp.poly.register_forward_hook(
                    lambda m, inp, out, i=i: count_sparse_component_flops(m, inp, out, i, "poly")
                )
            if mlp.use_micro:
                mlp.micro.register_forward_hook(
                    lambda m, inp, out, i=i: count_sparse_component_flops(m, inp, out, i, "micro")
                )
        # Count any dense linear layers as before
        for name, module in mlp.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(count_linear_flops)

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

    config = GPT2SparseConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        # Add your SparseFFN-specific config here if needed
    )

    model = GPT2Sparse(config).to(device)
    attach_flop_hooks(model)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    writer = SummaryWriter(log_dir="./runs/sparse_gpt2")

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
