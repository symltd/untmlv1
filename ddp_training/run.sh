#!/bin/bash
# Example run script

# 2 GPUs with profiling
torchrun --nproc_per_node=2 train_ddp.py --epochs 1 --per_device_batch 4 --max_steps 10 --profile

# 8 GPUs full run
# torchrun --nproc_per_node=8 train_ddp.py --epochs 1 --per_device_batch 8 --max_steps 200
