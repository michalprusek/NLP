#!/bin/bash

# Run prompt optimization with dual GPU support (tensor parallelism)
# This script uses both NVIDIA L40S GPUs to accelerate a single optimization run

set -e

# Configuration
TASK_MODEL=${TASK_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
META_MODEL=${META_MODEL:-""}  # Leave empty to use same as task model
METHOD=${METHOD:-"protegi"}
TASK=${TASK:-"gsm8k"}

echo "=========================================="
echo "Dual GPU Prompt Optimization"
echo "Backend: vLLM (with tensor parallelism)"
echo "=========================================="
echo "Task: $TASK"
echo "Method: $METHOD"
echo "Task Model: $TASK_MODEL"
if [ -n "$META_MODEL" ]; then
    echo "Meta-optimizer Model: $META_MODEL"
else
    echo "Meta-optimizer Model: (same as task model)"
fi
echo "GPUs: 0, 1"
echo "Tensor Parallel Size: 2"
echo "=========================================="
echo ""

export PATH="$HOME/.local/bin:$PATH"

# Build command with optional meta-model
CMD="uv run python main.py \
    --task $TASK \
    --method $METHOD \
    --model $TASK_MODEL \
    --backend vllm \
    --tensor-parallel-size 2 \
    --gpu-ids \"0,1\" \
    --iterations 5 \
    --minibatch-size 150 \
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