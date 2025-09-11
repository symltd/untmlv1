#!/bin/bash
# run.sh - Example DDP training runs for GPT-2 (4 GPUs)

# ------------------------------
# Quick test with profiling (small number of steps)
# ------------------------------
torchrun --nproc_per_node=4 train_ddp.py \
    --epochs 1 \
    --per_device_batch 16 \
    --max_steps 20 \
    --profile

# ------------------------------
# Full run (uncomment for full training)
# ------------------------------
# torchrun --nproc_per_node=4 train_ddp.py \
#     --epochs 2 \
#     --per_device_batch 16 \
#     --max_steps 10000
