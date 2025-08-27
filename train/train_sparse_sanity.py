# train_sparse_sanity.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import GPT2Sparse, UltraEfficientSparseFFN

# ---------------- Gradient Logging Helper ---------------- #
def log_sparse_ffn_grads(model, writer, step):
    for name, module in model.named_modules():
        if isinstance(module, UltraEfficientSparseFFN):
            for pname, param in module.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f"GradNorm/{name}.{pname}", param.grad.norm(), step)

# ---------------- Config ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
seq_len = 16
lr = 1e-4
num_steps = 10
log_dir = "./runs/sparse_ffn_sanity"

# ---------------- Model ---------------- #
model = GPT2Sparse.from_pretrained_sparse("gpt2")  # small GPT-2
model.to(device)
model.train()

# ---------------- Dummy Input ---------------- #
vocab_size = model.config.vocab_size
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# ---------------- Loss & Optimizer ---------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# ---------------- TensorBoard ---------------- #
writer = SummaryWriter(log_dir=log_dir)

# ---------------- Training Loop ---------------- #
for step in range(num_steps):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(dummy_input, labels=dummy_input)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Log loss
    writer.add_scalar("Loss/train", loss.item(), step)
    
    # log gradients per UltraEfficientSparseFFN block
    log_sparse_ffn_grads(model, writer, step)
    
    print(f"Step {step} | Loss: {loss.item():.4f}")

writer.close()
print("Sanity check completed. Loss and gradients logged to TensorBoard.")
