#!/bin/bash
# RieLBO v2 Benchmark Script
#
# Runs 8 configurations × 2 tasks × 10 seeds = 160 runs
#
# Usage:
#   chmod +x rielbo/benchmark/run_v2_benchmark.sh
#   ./rielbo/benchmark/run_v2_benchmark.sh
#
# Or with specific GPU:
#   CUDA_VISIBLE_DEVICES=1 ./rielbo/benchmark/run_v2_benchmark.sh

set -e

# Configuration
CONFIGS="baseline order2 whitening geodesic adaptive prob_norm product full"
TASKS="adip med2"
SEEDS="42 43 44 45 46 47 48 49 50 51"  # 10 seeds

N_COLD_START=100
ITERATIONS=500

LOG_DIR="rielbo/results/v2_benchmark_logs"
mkdir -p "$LOG_DIR"

# Count total runs
n_configs=$(echo $CONFIGS | wc -w)
n_tasks=$(echo $TASKS | wc -w)
n_seeds=$(echo $SEEDS | wc -w)
total_runs=$((n_configs * n_tasks * n_seeds))

echo "============================================="
echo "RieLBO v2 Benchmark"
echo "============================================="
echo "Configurations: $CONFIGS"
echo "Tasks: $TASKS"
echo "Seeds: $SEEDS"
echo "Total runs: $total_runs"
echo "Cold start: $N_COLD_START"
echo "Iterations: $ITERATIONS"
echo "Log dir: $LOG_DIR"
echo "============================================="
echo ""

run_count=0

for config in $CONFIGS; do
    for task in $TASKS; do
        for seed in $SEEDS; do
            run_count=$((run_count + 1))

            log_file="${LOG_DIR}/${config}_${task}_s${seed}.log"

            echo "[$run_count/$total_runs] Running: config=$config, task=$task, seed=$seed"

            if [ -f "$log_file" ] && grep -q "Final best score" "$log_file"; then
                echo "  -> Skipping (already completed)"
                continue
            fi

            uv run python -m rielbo.run_guacamol_subspace_v2 \
                --preset "$config" \
                --task-id "$task" \
                --seed "$seed" \
                --n-cold-start "$N_COLD_START" \
                --iterations "$ITERATIONS" \
                2>&1 | tee "$log_file"

            echo ""
        done
    done
done

echo "============================================="
echo "Benchmark complete!"
echo "Results saved to: rielbo/results/guacamol_v2/"
echo "Logs saved to: $LOG_DIR"
echo "============================================="
