#!/bin/bash

# Run prompt optimization with dual GPU support (tensor parallelism)
# This script uses both NVIDIA L40S GPUs to accelerate a single optimization run

set -e

echo "=========================================="
echo "Dual GPU Prompt Optimization"
echo "Backend: vLLM (with tensor parallelism)"
echo "=========================================="
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "GPUs: 0, 1 (both L40S)"
echo "Tensor Parallel Size: 2"
echo "=========================================="
echo ""

# Default method is protegi, but can be overridden
METHOD=${1:-protegi}
ITERATIONS=${2:-5}
MINIBATCH_SIZE=${3:-150}

export PATH="$HOME/.local/bin:$PATH"

uv run python main.py \
    --method "$METHOD" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --tensor-parallel-size 2 \
    --gpu-ids "0,1" \
    --iterations "$ITERATIONS" \
    --minibatch-size "$MINIBATCH_SIZE" \
    --beam-size 4 \
    --num-candidates 8 \
    --initial-prompt 'Work through the problem piece by piece and box your final answer as #### NUMBER.'

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Check results/ directory for outputs"
echo "=========================================="

