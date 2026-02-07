#!/bin/bash
# Full 10-seed benchmark: GeodesicMatern ν=0.5 + explore preset.
#
# Pilot results (3 seeds): 0.5782 ± 0.008
# Compared to: ArcCosine explore = 0.5555±0.013 (pre-audit)
# Expected runtime: ~10 x 4min = ~40min on RTX A5000

set -euo pipefail

RESULTS_DIR="rielbo/results/guacamol_v2_audit"
mkdir -p "$RESULTS_DIR"

echo "=== GeodesicMatern ν=0.5 Full Benchmark ==="
echo "Start: $(date)"
echo ""

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
        echo "[$(date +%H:%M:%S)] Seed $seed done: $(python3 -c "import json; d=json.load(open('$LATEST')); print(f'best={d[\"best_score\"]:.4f}')")"
    else
        echo "[$(date +%H:%M:%S)] WARNING: No result file found for seed $seed"
    fi
    echo ""
done

echo "=== Full Summary ==="
echo "End: $(date)"
python3 -c "
import json, glob, numpy as np
scores = []
for seed in range(42, 52):
    files = sorted(glob.glob(f'$RESULTS_DIR/v2_explore_geodesic_matern_adip_s{seed}_*.json'))
    if files:
        with open(files[-1]) as f:
            data = json.load(f)
        score = data['best_score']
        scores.append(score)
        print(f'  s{seed}: {score:.4f}')
if scores:
    arr = np.array(scores)
    print(f'  Mean: {np.mean(arr):.4f} ± {np.std(arr):.3f}')
    print(f'  Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f}')
    # Compare to baselines
    baseline = 0.5424
    print(f'  vs geodesic baseline (0.5424): {np.mean(arr)-baseline:+.4f}')
    from scipy import stats
    t, p = stats.ttest_1samp(arr, baseline)
    print(f'  One-sample t-test vs 0.5424: t={t:.3f}, p={p:.4f}')
"
