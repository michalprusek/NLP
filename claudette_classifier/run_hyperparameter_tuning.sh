#!/bin/bash
# Run hyperparameter tuning for Claudette binary classifier

set -e

# Configuration
N_TRIALS=${N_TRIALS:-50}
DEVICE=${DEVICE:-"auto"}
STUDY_NAME=${STUDY_NAME:-"claudette_tuning"}

echo "=========================================="
echo "Claudette Hyperparameter Tuning"
echo "=========================================="
echo "Number of trials: $N_TRIALS"
echo "Device: $DEVICE"
echo "Study name: $STUDY_NAME"
echo "=========================================="
echo ""

export PATH="$HOME/.local/bin:$PATH"

# Run tuning
uv run python -m claudette_classifier.hyperparameter_tuning \
    --n-trials "$N_TRIALS" \
    --device "$DEVICE" \
    --study-name "$STUDY_NAME"

echo ""
echo "=========================================="
echo "Hyperparameter tuning complete!"
echo "Results saved to: results/claudette_classifier/"
echo "=========================================="
