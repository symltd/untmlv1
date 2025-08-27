import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2Tokenizer

from models import GPT2Sparse, GPT2SparseConfig, UltraEfficientSparseFFN

# ---------------- FLOPs HOOKS ---------------- #
from collections import defaultdict

FLOP_COUNTER = defaultdict(int)
PER_LAYER_COUNTER = defaultdict(lambda: {"dense": 0, "sparse": 0})

def count_flops_hook(module, input, output):
    """Hook to count FLOPs dynamically for each forward pass."""
    # Use module name as unique key
    name = getattr(module, "_flop_name", module.__class__.__name__)

    # Dense Linear FLOPs
    if isinstance(module, nn.Linear):
        batch_size = input[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features
        FLOP_COUNTER[f"{name}"] += 2 * batch_size * in_features * out_features

    # UltraEfficientSparseFFN FLOPs
    elif isinstance(module, UltraEfficientSparseFFN):
        batch_size, seq_len, hidden_size = input[0].shape
        dense_flops = 2 * batch_size * seq_len * hidden_size * hidden_size

        # sparsity_factor = 1.0
        # if hasattr(module, "use_spectral") and module.use_spectral:
        #     sparsity_factor *= module.spectral.k / hidden_size
        # if hasattr(module, "use_polynomial") and module.use_polynomial:
        #     sparsity_factor *= module.poly.K / hidden_size
        # if hasattr(module, "use_micro") and module.use_micro:
        #     sparsity_factor *= module.micro.steps / hidden_size
        # Apply effective sparsity factors
        spectral_factor = (module.spectral.k / hidden_size) if module.use_spectral else 1.0
        poly_factor = (module.poly.K / hidden_size) if module.use_polynomial else 1.0
        micro_factor = (module.micro.steps / hidden_size) if module.use_micro else 1.0

        # Sparse FLOPs estimate
        effective_factor = spectral_factor * poly_factor * micro_factor
        sparse_flops = int(dense_flops * effective_factor)
        #sparse_flops = int(dense_flops * sparsity_factor)
        FLOP_COUNTER[f"{name}"] += sparse_flops
        # Per-layer logging
        PER_LAYER_COUNTER[module] = {"dense": dense_flops, "sparse": sparse_flops}

def attach_flop_hooks(model):
    """Attach hooks to all Linear and UltraEfficientSparseFFN layers."""
    for module in model.modules():
    # for name, module in model.named_modules():
    #     if isinstance(module, UltraEfficientSparseFFN):
    #         module._flop_name = f"sparse_ffn_{name.replace('.', '_')}"
        if isinstance(module, (nn.Linear, UltraEfficientSparseFFN)):
            module.register_forward_hook(count_flops_hook)

def log_flops_to_tensorboard(step, writer):
    """Log total and per-layer FLOPs to TensorBoard."""
    total_flops = sum(FLOP_COUNTER.values())
    writer.add_scalar("FLOPs/total", total_flops, step)
    # for mod_name, flops in FLOP_COUNTER.items():
    #     writer.add_scalar(f"FLOPs/{mod_name}", flops, step)
    for idx, (module, flops_dict) in enumerate(PER_LAYER_COUNTER.items()):
        writer.add_scalar(f"FLOPs/layer_{idx}_dense", flops_dict["dense"], step)
        writer.add_scalar(f"FLOPs/layer_{idx}_sparse", flops_dict["sparse"], step)
        writer.add_scalar(f"FLOPs/layer_{idx}_savings", flops_dict["dense"] - flops_dict["sparse"], step)
        writer.add_scalar(f"FLOPs/layer_{idx}_speedup",
                          flops_dict["dense"] / max(flops_dict["sparse"], 1), step)
    # Clear counters after logging
    FLOP_COUNTER.clear()
    PER_LAYER_COUNTER.clear()

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

    #config = GPT2SparseConfig(
    #    vocab_size=tokenizer.vocab_size,
    #    n_layer=4,
    #    n_head=4,
    #    n_embd=128,
    #)
    #model = GPT2Sparse(config).to(device)
    model = GPT2Sparse.from_pretrained_sparse("gpt2").to(device)
    #model = GPT2Sparse.from_pretrained_sparse("gpt2", revision="main", cache_dir="./hf_cache")

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

            # Log FLOPs (total + per-layer)
            log_flops_to_tensorboard(global_step, writer)

            if global_step % 100 == 0:
                print(f"Step {global_step} | Loss {loss.item():.4f}")
            global_step += 1

    writer.close()

if __name__ == "__main__":
    main()
