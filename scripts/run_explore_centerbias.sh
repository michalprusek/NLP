#!/bin/bash
# ArcCosine explore with CENTER-BIASED cap sampling (original/reverted).
# Should recover ~0.5555 pre-fix performance.
# Expected runtime: ~40 min on A100

set -euo pipefail

RESULTS_DIR="rielbo/results/guacamol_v2_centerbias"
mkdir -p "$RESULTS_DIR"

echo "=== ArcCosine Explore (Center-Biased) ==="
echo "Start: $(date)"

for seed in 42 43 44 45 46 47 48 49 50 51; do
    echo "[$(date +%H:%M:%S)] Starting seed $seed..."

    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset explore \
        --ur-std-low 0.05 \
        --task-id adip \
        --n-cold-start 100 \
        --iterations 500 \
        --seed $seed

    LATEST=$(ls -t rielbo/results/guacamol_v2/v2_explore_adip_s${seed}_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$RESULTS_DIR/"
        echo "[$(date +%H:%M:%S)] s${seed}: $(python3 -c "import json; d=json.load(open('$LATEST')); print(f'best={d[\"best_score\"]:.4f}')")"
    fi
done

echo "=== Summary ==="
python3 -c "
import json, glob, numpy as np
scores = []
for seed in range(42, 52):
    files = sorted(glob.glob(f'$RESULTS_DIR/v2_explore_adip_s{seed}_*.json'))
    if files:
        with open(files[-1]) as f:
            d = json.load(f)
        scores.append(d['best_score'])
        print(f'  s{seed}: {d[\"best_score\"]:.4f}')
if scores:
    print(f'  Mean: {np.mean(scores):.4f} ± {np.std(scores):.3f}')
    print(f'  Expected: ~0.5555 (pre-fix)')
"
