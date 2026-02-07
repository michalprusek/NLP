#!/bin/bash
# GeodesicMatern ν=0.5 + explore with CENTER-BIASED cap sampling.
# The definitive test: does the learnable lengthscale help?
# Expected runtime: ~45 min on RTX A5000

set -euo pipefail

RESULTS_DIR="rielbo/results/guacamol_v2_centerbias"
mkdir -p "$RESULTS_DIR"

echo "=== GeoMat ν=0.5 (Center-Biased) ==="
echo "Start: $(date)"

for seed in 42 43 44 45 46 47 48 49 50 51; do
    echo "[$(date +%H:%M:%S)] Starting seed $seed..."

    CUDA_VISIBLE_DEVICES=1 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset explore \
        --kernel-type geodesic_matern \
        --kernel-order 0 \
        --ur-std-low 0.05 \
        --task-id adip \
        --n-cold-start 100 \
        --iterations 500 \
        --seed $seed

    LATEST=$(ls -t rielbo/results/guacamol_v2/v2_explore_geodesic_matern_adip_s${seed}_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$RESULTS_DIR/"
        echo "[$(date +%H:%M:%S)] s${seed}: $(python3 -c "import json; d=json.load(open('$LATEST')); print(f'best={d[\"best_score\"]:.4f}')")"
    fi
done

echo "=== Summary ==="
python3 -c "
import json, glob, numpy as np
from scipy import stats
scores = []
for seed in range(42, 52):
    files = sorted(glob.glob(f'$RESULTS_DIR/v2_explore_geodesic_matern_adip_s{seed}_*.json'))
    if files:
        with open(files[-1]) as f:
            d = json.load(f)
        scores.append(d['best_score'])
        print(f'  s{seed}: {d[\"best_score\"]:.4f}')
if scores:
    arr = np.array(scores)
    print(f'  Mean: {np.mean(arr):.4f} ± {np.std(arr):.3f}')
    t, p = stats.ttest_1samp(arr, 0.5555)
    print(f'  vs ArcCosine explore (0.5555): t={t:.3f}, p={p:.4f}')
"
