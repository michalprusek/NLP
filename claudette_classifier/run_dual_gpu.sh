#!/bin/bash

# Run Claudette binary classifier training with dual L40S GPU support (48GB VRAM each)
# This script uses PyTorch Distributed Data Parallel (DDP) for efficient multi-GPU training

set -e

# Configuration
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}  # Per GPU batch size (total: 512 with 2 GPUs)
LR=${LR:-3e-5}
ENCODER_LR=${ENCODER_LR:-1e-5}
HIDDEN_DIMS=${HIDDEN_DIMS:-"1024 512 256"}  # Larger capacity for dual GPU
RESIDUAL_BLOCKS=${RESIDUAL_BLOCKS:-4}  # More residual blocks
DROPOUT=${DROPOUT:-0.4}  # Higher dropout for regularization
TRAIN_ENCODER=${TRAIN_ENCODER:-"true"}  # Enable BERT fine-tuning by default

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1
WORLD_SIZE=2

echo "=========================================="
echo "Claudette Binary Classifier - Dual L40S GPU Training"
echo "=========================================="
echo "GPUs: 0, 1 (2x L40S 48GB)"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE (total: $((BATCH_SIZE * WORLD_SIZE)))"
echo "Learning rate (classifier): $LR"
echo "Learning rate (encoder): $ENCODER_LR"
echo "Hidden dims: [$HIDDEN_DIMS]"
echo "Residual blocks: $RESIDUAL_BLOCKS"
echo "Dropout: $DROPOUT"
if [ "$TRAIN_ENCODER" = "true" ]; then
    echo "Encoder: TRAINABLE (fine-tuning Legal-BERT)"
else
    echo "Encoder: FROZEN (only training MLP)"
fi
echo "=========================================="
echo ""

export PATH="$HOME/.local/bin:$PATH"

# Build command
CMD="torchrun --nproc_per_node=$WORLD_SIZE --master_port=29500"
CMD="$CMD -m claudette_classifier.main"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --lr $LR"
CMD="$CMD --encoder-lr $ENCODER_LR"
CMD="$CMD --hidden-dims $HIDDEN_DIMS"
CMD="$CMD --num-residual-blocks $RESIDUAL_BLOCKS"
CMD="$CMD --dropout $DROPOUT"
CMD="$CMD --device cuda"

# Add train-encoder flag if enabled
if [ "$TRAIN_ENCODER" = "true" ]; then
    CMD="$CMD --train-encoder"
fi

# Execute
eval $CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "Results saved to: results/claudette_classifier/"
echo "=========================================="
