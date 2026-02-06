#!/bin/bash
# RieLBO-GSM8K benchmark — single seed, full test set eval
# Usage: tmux new-session -d -s gsm8k_bench "bash rielbo_gsm8k/run_benchmark.sh"

set -e
export CUDA_VISIBLE_DEVICES=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p rielbo_gsm8k/results

echo "RieLBO-GSM8K | seed=42 | eval=1319 | $(date)"

uv run python -m rielbo_gsm8k.run \
    --preset geodesic --subspace-dim 16 \
    --n-cold-start 30 --iterations 70 --seed 42 \
    --split test --eval-size 1319 \
    --sonar-device cpu \
    --incremental-json rielbo_gsm8k/results/rielbo_s42.json \
    2>&1 | tee "rielbo_gsm8k/results/rielbo_s42_${TIMESTAMP}.log"

echo "Generating convergence plot..."
uv run python -m shared.plot_prompt_convergence 2>&1 || true

echo "Done: $(date)"
