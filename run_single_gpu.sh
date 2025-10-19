#!/bin/bash

# Run prompt optimization on a single GPU
# Use this if you want to run on just one GPU or if tensor parallelism has issues

set -e

echo "=========================================="
echo "Single GPU Prompt Optimization"
echo "=========================================="
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "GPU: 0"
echo "=========================================="
echo ""

# Default method is protegi, but can be overridden
METHOD=${1:-protegi}
ITERATIONS=${2:-3}
MINIBATCH_SIZE=${3:-10}

export PATH="$HOME/.local/bin:$PATH"

uv run python main.py \
    --method "$METHOD" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-ids "0" \
    --iterations "$ITERATIONS" \
    --minibatch-size "$MINIBATCH_SIZE" \
    --beam-size 4 \
    --num-candidates 8 \
    --initial-prompt "Let's solve this step by step."

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Check results/ directory for outputs"
echo "=========================================="

