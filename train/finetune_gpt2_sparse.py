import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from models.sparse_ffn import UltraEfficientSparseFFN
import torch.profiler as profiler

# Load tokenizer and baseline GPT-2 small
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Replace the feedforward (mlp) in each transformer block with UltraEfficientSparseFFN
for block in model.transformer.h:
    hidden_dim = block.mlp.c_fc.in_features
    ffn_dim = block.mlp.c_fc.out_features
    block.mlp = UltraEfficientSparseFFN(hidden_dim, ffn_dim, k=64, sparsity=0.5)

# Tiny dataset example
texts = [
    "Ultra efficient sparse feedforward networks are powerful.",
    "Transformers can be optimized with sparse layers.",
    "Reducing FLOPs and power improves efficiency.",
]
encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attn_mask = encodings["attention_mask"]
    def __len__(self):
        return self.input_ids.size(0)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "labels": self.input_ids[idx],
        }

dataset = TinyDataset(encodings)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=5,
    save_total_limit=1,
    logging_steps=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Use profiler on a single training step
def profile_step():
    batch = dataset[0]
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    print("=== Profiling one step ===")
    profile_step()
    print("=== Starting training ===")
    trainer.train()
