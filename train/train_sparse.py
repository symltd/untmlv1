# train/train_sparse.py
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2Tokenizer

from models import GPT2Sparse, GPT2SparseConfig
from models.sparse_ffn import UltraEfficientSparseFFN

# ---------------- FLOPs HOOKS ---------------- #
FLOP_COUNTER = defaultdict(int)

def count_flops_hook(module, input, output):
    """Hook to count FLOPs dynamically for each forward pass."""
    name = module.__class__.__name__

    # Dense Linear FLOPs
    if isinstance(module, nn.Linear):
        batch_size = input[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features
        FLOP_COUNTER[f"{name}"] += 2 * batch_size * in_features * out_features

    # UltraEfficientSparseFFN FLOPs (rough estimate)
    elif isinstance(module, UltraEfficientSparseFFN):
        B, T, D = input[0].shape
        dense_flops = 2 * B * T * D * D
        sparsity_factor = 1.0
        if hasattr(module, "use_spectral") and module.use_spectral:
            sparsity_factor *= module.spectral.k / D
        if hasattr(module, "use_polynomial") and module.use_polynomial:
            sparsity_factor *= module.poly.K / D
        if hasattr(module, "use_micro") and module.use_micro:
            sparsity_factor *= max(1, module.micro.steps) / D
        sparse_flops = int(dense_flops * sparsity_factor)
        FLOP_COUNTER[f"{name}"] += sparse_flops

def attach_flop_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, UltraEfficientSparseFFN)):
            module.register_forward_hook(count_flops_hook)

def log_flops_to_tensorboard(step, writer):
    total_flops = sum(FLOP_COUNTER.values())
    writer.add_scalar("FLOPs/total", total_flops, step)
    for mod_name, flops in FLOP_COUNTER.items():
        writer.add_scalar(f"FLOPs/{mod_name}", flops, step)
    FLOP_COUNTER.clear()

# ---------------- DATASET ---------------- #
def get_wikitext_dataloader(tokenizer, block_size=128, batch_size=4):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size
        )
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)

# ---------------- TRAINING ---------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- TOKENIZER ---------------- #
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- DATALOADER ---------------- #
    dataloader = get_wikitext_dataloader(tokenizer)

    # ---------------- MODEL ---------------- #
    config = GPT2SparseConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        ffn_k_freq=64,
        poly_degree=3,
        poly_keep_ratio=0.5,
        micro_steps=2,
        micro_keep_ratio=0.25,
        ffn_dropout=0.0,
        use_spectral=True,
        use_polynomial=True,
        use_micro=True,
        residual_gate_init=1.0
    )
    model = GPT2Sparse(config).to(device)
    attach_flop_hooks(model)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    writer = SummaryWriter(log_dir="./runs/sparse_gpt2")

    global_step = 0
    model.train()

    # ---------------- TRAIN LOOP ---------------- #
    for epoch in range(3):
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---------------- LOGGING ---------------- #
            writer.add_scalar("Loss/train", loss.item(), global_step)
            log_flops_to_tensorboard(global_step, writer)

            if global_step % 50 == 0:
                print(f"Step {global_step} | Loss {loss.item():.4f}")
            global_step += 1

    writer.close()

if __name__ == "__main__":
    main()
