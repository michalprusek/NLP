#!/bin/bash
# Sequential run of all 4 prompt optimization methods with 50k LLM call budget
# OPRO -> ProTeGi -> GEPA -> NFBO
# Using single GPU (tensor_parallel_size=1) to avoid Ray disk space issues

set -e  # Exit on error

# Clean up Ray cache to prevent disk full errors
cleanup_ray() {
    echo "Cleaning up Ray/vLLM cache..."
    rm -rf /tmp/ray* 2>/dev/null || true
    rm -rf /tmp/vllm* 2>/dev/null || true
    sleep 2
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results_50k_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Running all optimizers with 50k LLM call budget"
echo "Output directory: $LOG_DIR"
echo "Start time: $(date)"
echo "Using single GPU to avoid Ray disk issues"
echo "=============================================="

# Initial cleanup
cleanup_ray

# ================================================
# 1. OPRO (50k budget)
# Budget counting: budget = prompts_evaluated × minibatch_size
# With minibatch=261: 50000/261 ≈ 191 prompt evaluations
# ================================================
echo ""
echo "=============================================="
echo "1/4: OPRO - Starting at $(date)"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0 uv run python -m opro.run \
    --model qwen \
    --backend vllm \
    --tensor-parallel-size 1 \
    --budget 50000 \
    --minibatch-size 261 \
    --num-candidates 8 \
    --iterations 500 \
    --output-dir opro/results \
    2>&1 | tee "$LOG_DIR/opro_${TIMESTAMP}.log"

echo "OPRO completed at $(date)"
cleanup_ray
echo ""

# ================================================
# 2. ProTeGi (50k budget)
# Budget counting: budget = prompts_evaluated × minibatch_size
# With minibatch=64: 50000/64 ≈ 781 prompt evaluations
# Paper defaults: beam=4, steps=6, gradients=4, mc_samples=2
# ================================================
echo "=============================================="
echo "2/4: ProTeGi - Starting at $(date)"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0 uv run python -m protegi.run \
    --model qwen \
    --backend vllm \
    --tensor-parallel-size 1 \
    --budget 50000 \
    --minibatch-size 64 \
    --beam-size 4 \
    --steps 100 \
    --gradients 4 \
    --mc-samples 2 \
    --max-successors 8 \
    --output-dir protegi/results \
    2>&1 | tee "$LOG_DIR/protegi_${TIMESTAMP}.log"

echo "ProTeGi completed at $(date)"
cleanup_ray
echo ""

# ================================================
# 3. GEPA (50k budget)
# Budget counting: direct LLM call count
# 50000 calls with minibatch=64 → ~781 prompt evaluations
# Default: pareto=10, mutations=4, exploit_prob=0.8
# ================================================
echo "=============================================="
echo "3/4: GEPA - Starting at $(date)"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0 uv run python -m gepa.run \
    --model qwen \
    --backend vllm \
    --tensor-parallel-size 1 \
    --budget 50000 \
    --minibatch-size 64 \
    --pareto-size 10 \
    --mutations 4 \
    --exploit-prob 0.8 \
    --output-dir gepa/results \
    2>&1 | tee "$LOG_DIR/gepa_${TIMESTAMP}.log"

echo "GEPA completed at $(date)"
cleanup_ray
echo ""

# ================================================
# 4. NFBO (50k budget)
# Budget: n_initial × eval_subset + iterations × eval_subset
# With n_initial=20, eval_subset=150: 20×150=3000 initial
# Remaining: (50000-3000)/150 ≈ 313 iterations
# Paper defaults from Lee et al. 2025: top_k_percentile=20, flow_epochs=50
# ================================================
echo "=============================================="
echo "4/4: NFBO - Starting at $(date)"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0 uv run python -m nfbo.run \
    --model qwen \
    --backend vllm \
    --tensor-parallel-size 1 \
    --iterations 313 \
    --n-initial 20 \
    --n-candidates 64 \
    --eval-subset-size 150 \
    --top-k-percentile 20 \
    --flow-epochs 50 \
    --flow-layers 6 \
    --hidden-dim 512 \
    --checkpoint-dir nfbo/results \
    --checkpoint-freq 50 \
    2>&1 | tee "$LOG_DIR/nfbo_${TIMESTAMP}.log"

echo "NFBO completed at $(date)"
echo ""

# ================================================
# Summary
# ================================================
echo "=============================================="
echo "ALL OPTIMIZERS COMPLETE"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - OPRO:    opro/results/"
echo "  - ProTeGi: protegi/results/"
echo "  - GEPA:    gepa/results/"
echo "  - NFBO:    nfbo/results/"
echo "  - Logs:    $LOG_DIR/"
echo ""
echo "To compare results, check the JSON files in each results directory."
