# GPT-2 DDP Training with Profiling

This repo demonstrates how to train a small GPT-2 model with **Distributed Data Parallel (DDP)** on multiple GPUs (single-node) with profiling enabled.

## Requirements
```bash
pip install -r requirements.txt
```

## Run (1–8 GPUs)
```bash
torchrun --nproc_per_node=4 train_ddp.py --epochs 1 --per_device_batch 8 --max_steps 200
```

To enable profiling:
```bash
torchrun --nproc_per_node=2 train_ddp.py --epochs 1 --per_device_batch 4 --max_steps 10 --profile
```

Profiling output (`trace.json`) can be viewed in Chrome tracing or TensorBoard.
