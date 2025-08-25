import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.gpt2_sparse import GPT2Sparse

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Sparse.from_pretrained("gpt2")

text = "Ultra efficient sparse feedforward networks are powerful"
inputs = tokenizer(text, return_tensors="pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(10):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step}, Loss: {loss.item():.4f}")
