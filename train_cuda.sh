#!/bin/bash

# CUDA-optimized training configuration for RTX 5060
# This configuration maximizes CUDA performance with mixed precision

echo "Starting CUDA-optimized training for RTX 5060..."

python -m src.train \
  --epochs 8 \
  --batch_size 8 \
  --lr 5e-4 \
  --num_workers 8 \
  --prefetch_factor 2 \
  --pin_memory \
  --accumulate 1 \
  --img_size 640 \
  --trainable_layers 3 \
  --fast_aug \
  --val_every 2 \
  --train_ratio 0.9 \
  --max_steps 100 \
  --device cuda

echo "Training completed!"
