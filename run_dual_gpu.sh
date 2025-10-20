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

export PATH="$HOME/.local/bin:$PATH"

uv run python main.py \
    --task claudette \
    --method protegi \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --tensor-parallel-size 2 \
    --gpu-ids "0,1" \
    --iterations 5 \
    --minibatch-size 1500 \
    --beam-size 4 \
    --num-candidates 8 \
    --initial-prompt 'Identify key provisions related to contractual obligations, liability limits, and dispute resolution methods, then report all fitting labels as: LABELS: <comma-separated numbers>'

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Check results/ directory for outputs"
echo "=========================================="