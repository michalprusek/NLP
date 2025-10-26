#!/bin/bash

# Run prompt optimization with dual GPU support (tensor parallelism)
# This script uses both NVIDIA L40S GPUs to accelerate a single optimization run

set -e

# Configuration
TASK_MODEL=${TASK_MODEL:-"Equall/Saul-7B-Instruct-v1"}  # Legal domain LLM (Mistral-based, instruction-tuned)
META_MODEL=${META_MODEL:-""}  # Use Claude Haiku for meta-optimization (Saul-Instruct can handle meta-prompts)
METHOD=${METHOD:-"protegi"}
TASK=${TASK:-"claudette_binary"}

echo "=========================================="
echo "Legal Domain Prompt Optimization"
echo "Backend: Transformers (single GPU)"
echo "=========================================="
echo "Task: $TASK"
echo "Method: $METHOD"
echo "Task Model: $TASK_MODEL (Legal Domain)"
if [ -n "$META_MODEL" ]; then
    echo "Meta-optimizer Model: $META_MODEL"
else
    echo "Meta-optimizer Model: (same as task model)"
fi
echo "Device: CUDA (GPU 0)"
echo "=========================================="
echo ""

export PATH="$HOME/.local/bin:$PATH"

# Build command with optional meta-model
CMD="uv run python main.py"
CMD="$CMD --task $TASK"
CMD="$CMD --method $METHOD"
CMD="$CMD --model $TASK_MODEL"
CMD="$CMD --backend transformers"
CMD="$CMD --device cuda"
CMD="$CMD --iterations 10"
CMD="$CMD --minibatch-size 100"  # Smaller batch for transformers backend
CMD="$CMD --beam-size 4"
CMD="$CMD --num-candidates 8"

# Add meta-model if specified
if [ -n "$META_MODEL" ]; then
    CMD="$CMD --meta-model $META_MODEL"
fi

# Execute
eval $CMD

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Check results/ directory for outputs"
echo "=========================================="