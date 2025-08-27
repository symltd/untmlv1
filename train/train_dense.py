# train/train_dense.py

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

# ---------------- FLOPs HOOKS ---------------- #

from collections import defaultdict
FLOP_COUNTER = defaultdict(int)
PER_LAYER_COUNTER = defaultdict(int)

def count_dense_flops_hook(module, input, output):
    """Hook to count FLOPs for each nn.Linear forward pass."""
    if isinstance(module, nn.Linear):
        batch_size = input[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features
        flops = 2 * batch_size * in_features * out_features
        FLOP_COUNTER["total"] += flops
        PER_LAYER_COUNTER[module] += flops

def attach_flop_hooks(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(count_dense_flops_hook)

def log_flops_to_tensorboard(step, writer):
    total_flops = FLOP_COUNTER["total"]
    writer.add_scalar("FLOPs/total", total_flops, step)
    for idx, (module, flops) in enumerate(PER_LAYER_COUNTER.items()):
        writer.add_scalar(f"FLOPs/layer_{idx}_dense", flops, step)
    FLOP_COUNTER.clear()
    PER_LAYER_COUNTER.clear()

# ---------------- TRAINING ---------------- #

def main():
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset["train"]["text"]

    # Tokenizer (reuse GPT2 tokenizer from HF)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Model config + dense GPT-2
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(config)

    # Attach FLOPs hooks
    attach_flop_hooks(model)

    # TensorBoard logging
    log_dir = os.path.join("runs", "dense_gpt2")
    writer = SummaryWriter(log_dir=log_dir)

    # Training args
    training_args = TrainingArguments(
        output_dir="./results_dense",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=1,
        logging_dir=log_dir,
        logging_steps=10,
        report_to=[],  # disable HF default logging
    )

    # Custom trainer to inject FLOPs logging
    class DenseTrainer(Trainer):
        def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            # Log FLOPs
            log_flops_to_tensorboard(self.state.global_step, writer)

            return loss.detach()

    trainer = DenseTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    trainer.train()
    writer.close()


if __name__ == "__main__":
    main()
