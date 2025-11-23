#!/bin/bash

# Run prompt optimization
# Supports both local models (vLLM/transformers) and API models (OpenAI/Claude)

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Configuration
TASK_MODEL=${TASK_MODEL:-"gpt-3.5-turbo"}  # gpt-3.5-turbo, claude-3-haiku-20240307, or Qwen/Qwen2.5-7B-Instruct
META_MODEL=${META_MODEL:-""}  # haiku, sonnet, or leave empty to use same as task model
METHOD=${METHOD:-"opro"}
TASK=${TASK:-"gsm8k"}


echo "=========================================="
echo "Prompt Optimization"
echo "Backend: auto (detects based on model)"
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
CMD="$CMD --iterations 200" # 6 ProTeGi, 200 OPRO
CMD="$CMD --minibatch-size 261" # 64 ProTeGi, 260 OPRO
CMD="$CMD --beam-size 4"
CMD="$CMD --num-candidates 8"

# Add meta-model if specified
if [ -n "$META_MODEL" ]; then
    CMD="$CMD --meta-model $META_MODEL"
fi

# Add save-intermediate-prompts flag if enabled
if [ "$SAVE_INTERMEDIATE" = "true" ]; then
    CMD="$CMD --save-intermediate-prompts"
    echo "Saving intermediate prompts to results JSON for debugging"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Execute and capture exit code
set +e  # Temporarily disable exit on error to catch status
echo "Starting optimization at $(date)..."
eval $CMD 2>&1
EXIT_CODE=$?
echo "Finished at $(date) with exit code $EXIT_CODE"
set -e  # Re-enable exit on error

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Optimization complete!"
    echo "Check results/ directory for outputs"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Optimization failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi