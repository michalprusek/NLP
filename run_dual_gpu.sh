#!/bin/bash

# Run prompt optimization with dual GPU support (tensor parallelism)
# This script uses both NVIDIA L40S GPUs to accelerate a single optimization run

set -e

# Configuration
TASK_MODEL=${TASK_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}  # Qwen/Qwen2.5-7B-Instruct or Equall/Saul-7B-Instruct-v1
META_MODEL=${META_MODEL:-""}  # haiku
METHOD=${METHOD:-"opro"}
TASK=${TASK:-"claudette_binary"}

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
CMD="uv run python main.py"
CMD="$CMD --mode optimize"
CMD="$CMD --task $TASK"
CMD="$CMD --method $METHOD"
CMD="$CMD --model $TASK_MODEL"
CMD="$CMD --backend vllm"
CMD="$CMD --tensor-parallel-size 2"
CMD="$CMD --gpu-ids 0,1"
CMD="$CMD --iterations 10" # 6 ProTeGi, 200 OPRO
CMD="$CMD --minibatch-size 300" # 64 ProTeGi, 260 OPRO
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