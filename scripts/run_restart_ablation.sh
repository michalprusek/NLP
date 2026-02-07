#!/bin/bash
# Restart frequency ablation: fast vs vfast (baseline already has data from V2 geodesic runs)
# Runs on GPU 1. Two tmux sessions (one per config), seeds sequential within each.
# Run: bash scripts/run_restart_ablation.sh

set -e

TASK="adip"
ITERS=500
SEEDS="42 43 44 45 46"

mkdir -p rielbo/results/restart_ablation

for config in fast vfast; do
    SESSION="restart_${config}"
    echo "=== Launching $SESSION ==="
    tmux new-session -d -s "$SESSION" "
        for seed in $SEEDS; do
            echo '========================================'
            echo \"[RUN] ${config} seed=\${seed}\"
            echo '========================================'
            CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_restart_ablation.py \
                --config $config --seed \$seed --task-id $TASK --iterations $ITERS --device cuda \
                2>&1 | tee rielbo/results/restart_ablation/${config}_${TASK}_s\${seed}.log
            echo \"[DONE] ${config} seed=\${seed}\"
        done
        echo 'ALL DONE for ${config}'
        exec bash
    "
    echo "  -> tmux session: $SESSION"
done

echo ""
echo "Monitor: tmux ls | grep restart_"
echo "Check:   tmux capture-pane -t restart_fast -p | tail -5"
