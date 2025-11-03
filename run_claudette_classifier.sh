#!/bin/bash
# Quick training script for Claudette binary classifier

set -e

# Default parameters
EPOCHS=50
BATCH_SIZE=32
LR=2e-5
ENCODER_LR=1e-5
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --train-encoder)
            TRAIN_ENCODER="--train-encoder"
            shift
            ;;
        --no-focal-loss)
            NO_FOCAL="--no-focal-loss"
            shift
            ;;
        --no-oversampling)
            NO_OVERSAMPLE="--no-oversampling"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--epochs N] [--batch-size N] [--device auto|cuda|mps|cpu] [--train-encoder] [--no-focal-loss] [--no-oversampling]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Training Claudette Binary Classifier"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Learning rate (classifier): $LR"
echo "Learning rate (encoder): $ENCODER_LR"
if [ -z "$TRAIN_ENCODER" ]; then
    echo "Encoder: FROZEN (only training MLP)"
else
    echo "Encoder: TRAINABLE (fine-tuning)"
fi
echo ""

uv run python -m claudette_classifier.main \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --encoder-lr "$ENCODER_LR" \
    --device "$DEVICE" \
    $TRAIN_ENCODER \
    $NO_FOCAL \
    $NO_OVERSAMPLE

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: results/claudette_classifier/"
echo "=========================================="
