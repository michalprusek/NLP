#!/bin/bash

# Run prompt optimization on a single GPU
# Use this if you want to run on just one GPU or if tensor parallelism has issues
#
# Usage:
#   ./run_single_gpu.sh [method] [iterations] [minibatch_size]
#
# Examples:
#   ./run_single_gpu.sh protegi 5 20
#   TASK_MODEL="Qwen/Qwen2.5-3B-Instruct" META_MODEL="claude-3-5-sonnet-20241022" ./run_single_gpu.sh protegi 10 50
#   META_MODEL="claude-3-haiku-20240307" ./run_single_gpu.sh opro 5 20

set -e

# Configuration
TASK_MODEL=${TASK_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
META_MODEL=${META_MODEL:-""}  # Leave empty to use same as task model
TASK=${TASK:-"gsm8k"}

# Command-line arguments
METHOD=${1:-protegi}
ITERATIONS=${2:-3}
MINIBATCH_SIZE=${3:-10}

echo "=========================================="
echo "Single GPU Prompt Optimization"
echo "=========================================="
echo "Task: $TASK"
echo "Method: $METHOD"
echo "Task Model: $TASK_MODEL"
if [ -n "$META_MODEL" ]; then
    echo "Meta-optimizer Model: $META_MODEL"
else
    echo "Meta-optimizer Model: (same as task model)"
fi
echo "GPU: 0"
echo "Iterations: $ITERATIONS"
echo "Minibatch Size: $MINIBATCH_SIZE"
echo "=========================================="
echo ""

export PATH="$HOME/.local/bin:$PATH"

# Build command with optional meta-model
CMD="uv run python main.py \
    --task $TASK \
    --method $METHOD \
    --model $TASK_MODEL \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-ids \"0\" \
    --iterations $ITERATIONS \
    --minibatch-size $MINIBATCH_SIZE \
    --beam-size 4 \
    --num-candidates 8"

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

