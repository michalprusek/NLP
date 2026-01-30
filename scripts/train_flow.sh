#!/bin/bash
# Training script for EcoFlow flow model
# Run in tmux for long-running process

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/flow_model_${TIMESTAMP}"
LOG_FILE="results/train_flow_${TIMESTAMP}.log"

echo "Starting flow model training..."
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=0 uv run python -m src.ecoflow.train_flow \
    --data-path datasets/sonar_embeddings.pt \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 1024 \
    --lr 1e-4 \
    --epochs 100 \
    --hidden-dim 512 \
    --num-layers 6 \
    --num-heads 8 \
    --ema-decay 0.9999 \
    --grad-clip 1.0 \
    --warmup-steps 1000 \
    --val-interval 10 \
    --save-interval 20 \
    --seed 42 \
    2>&1 | tee "${LOG_FILE}"

echo "Training complete. Checkpoint saved to ${OUTPUT_DIR}"
