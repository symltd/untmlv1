import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from collections import defaultdict

from collections import defaultdict
import torch.nn as nn

# ---------------- Robust FLOPs hooks (replace previous versions) ---------------- #
FLOP_COUNTER = defaultdict(int)             # global counters by layer name
BLOCK_COMPONENTS = defaultdict(lambda: defaultdict(int))  # block_idx -> {layer_name: flops}

def _infer_linear_shapes_from_module(module, input_tensor, output):
    """
    Robustly infer (batch, seq_len, in_features, out_features) for
    both nn.Linear and HF Conv1D-like modules that store `weight`.
    Returns (batch, seq_len, in_f, out_f).
    """
    x = input_tensor
    # x may be [B, S, D] or [B, D] or other — handle common cases
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
    elif x.dim() == 2:
        batch_size, seq_len = x.shape[0], 1
    else:
        batch_size, seq_len = x.shape[0], 1

    # try to get in/out features from attributes if present
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        in_f = int(module.in_features)
        out_f = int(module.out_features)
    elif hasattr(module, "weight") and module.weight is not None:
        # weight typically has shape (out_f, in_f) for HF Conv1D
        wshape = tuple(module.weight.shape)
        if len(wshape) == 2:
            out_f, in_f = int(wshape[0]), int(wshape[1])
        else:
            # fallback: try transpose
            in_f, out_f = int(wshape[-1]), int(wshape[0])
    else:
        # last-resort: infer from input and output tensors
        try:
            out_f = output.shape[-1]
            in_f = x.shape[-1]
        except Exception:
            raise RuntimeError("Unable to infer linear shapes for FLOPs counting.")
    return batch_size, seq_len, in_f, out_f

def count_linear_flops(module, inputs, output, block_idx, layer_name):
    """
    Hook: compute FLOPs = 2 * B * S * in_f * out_f
    Supports nn.Linear and HF Conv1D (modules with .weight).
    """
    # inputs may be a tuple; use inputs[0]
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    B, S, in_f, out_f = _infer_linear_shapes_from_module(module, x, output)

    flops = 2 * B * S * in_f * out_f
    # Global and per-block accounting
    FLOP_COUNTER[layer_name] += flops
    BLOCK_COMPONENTS[block_idx][layer_name] += flops

def attach_flop_hooks(model):
    """
    Attach hooks to `c_fc` and `c_proj` or all Linear-like submodules inside each block.mlp,
    binding block index and layer name correctly (no late-binding lambda).
    """
    for i, block in enumerate(model.transformer.h):
        if not hasattr(block, "mlp"):
            continue
        # prefer to directly attach to known sublayers if present
        # GPT-2 MLP typically has c_fc and c_proj (Conv1D wrappers)
        mlp = block.mlp
        # If the module has attributes c_fc / c_proj, attach to them
        if hasattr(mlp, "c_fc") and hasattr(mlp, "c_proj"):
            def make_hook_fc(idx, lname):
                def hook(m, inp, out):
                    count_linear_flops(m, inp, out, block_idx=idx, layer_name=lname)
                return hook
            mlp.c_fc.register_forward_hook(make_hook_fc(i, f"block_{i}_c_fc"))
            mlp.c_proj.register_forward_hook(make_hook_fc(i, f"block_{i}_c_proj"))
        else:
            # fallback: attach hooks to every nn.Linear / conv-like in the mlp
            for name, sub in mlp.named_modules():
                if isinstance(sub, nn.Linear) or (hasattr(sub, "weight") and getattr(sub, "weight") is not None):
                    def make_hook(idx, lname):
                        def hook(m, inp, out):
                            count_linear_flops(m, inp, out, block_idx=idx, layer_name=lname)
                        return hook
                    sub.register_forward_hook(make_hook(i, f"block_{i}_{name}"))

def log_flops_to_tensorboard(step, writer):
    """Log per-layer and per-block FLOPs, then clear counters for next step."""
    # per-layer scalars
    for layer_name, flops in FLOP_COUNTER.items():
        writer.add_scalar(f"FLOPs/{layer_name}", flops, step)

    # per-block totals
    for block_idx, comps in BLOCK_COMPONENTS.items():
        block_total = sum(comps.values())
        writer.add_scalar(f"FLOPs/block_{block_idx}_total", block_total, step)

    # global total
    total = sum(FLOP_COUNTER.values())
    writer.add_scalar("FLOPs/total", total, step)

    # clear for next forward
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
