#!/bin/bash
# Continue U-Net training if loss is still improving
# Usage: ./scripts/continue_unet_training.sh [additional_epochs]

ADDITIONAL_EPOCHS=${1:-500}
CURRENT_CHECKPOINT="study/checkpoints/unet-otcfm-10k-none/best.pt"

# Get current epoch from checkpoint
CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CURRENT_CHECKPOINT', map_location='cpu'); print(ckpt['epoch'])")
NEW_TOTAL=$((CURRENT_EPOCH + ADDITIONAL_EPOCHS))

echo "Current epoch: $CURRENT_EPOCH"
echo "Training to epoch: $NEW_TOTAL"

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled uv run python -m study.flow_matching.train \
    --arch unet \
    --flow otcfm \
    --dataset 10k \
    --aug none \
    --epochs $NEW_TOTAL \
    --batch-size 256 \
    --lr 1e-4 \
    --group unet-continued \
    --resume $CURRENT_CHECKPOINT \
    2>&1 | tee study/results/unet_continued_$(date +%Y%m%d_%H%M%S).log
