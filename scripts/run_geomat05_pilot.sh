#!/bin/bash
# Pilot experiment: GeodesicMaternKernel ν=0.5 with explore preset.
#
# Tests whether a single learnable lengthscale (exp(-d_g/ℓ))
# helps over parameterless ArcCosine (1 - d_g/π).
#
# Matérn ν=0.5 is provably PD on all spheres (Schoenberg's theorem).
# Only 1 hyperparameter (lengthscale) vs 0 for ArcCosine.
#
# Run 5 seeds as quick pilot (100 iter), then decide if worth 10 seeds × 500 iter.

set -euo pipefail

RESULTS_DIR="rielbo/results/guacamol_v2_audit"
mkdir -p "$RESULTS_DIR"

echo "=== GeodesicMatern ν=0.5 Pilot ==="
echo "Start: $(date)"
echo ""

# Quick pilot: 3 seeds × 500 iter
for seed in 42 43 44; do
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

    # Find the result
    LATEST=$(ls -t rielbo/results/guacamol_v2/v2_explore_geodesic_matern_adip_s${seed}_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$RESULTS_DIR/"
        echo "[$(date +%H:%M:%S)] Seed $seed done: $(python3 -c "import json; d=json.load(open('$LATEST')); print(f'best={d[\"best_score\"]:.4f}')")"
    else
        echo "[$(date +%H:%M:%S)] WARNING: No result file found for seed $seed"
    fi
    echo ""
done

echo "=== Pilot Summary ==="
echo "End: $(date)"
python3 -c "
import json, glob, numpy as np
scores = []
for seed in [42, 43, 44]:
    files = sorted(glob.glob(f'$RESULTS_DIR/v2_explore_geodesic_matern_adip_s{seed}_*.json'))
    if files:
        with open(files[-1]) as f:
            data = json.load(f)
        score = data['best_score']
        scores.append(score)
        print(f'  s{seed}: {score:.4f}')
if scores:
    print(f'  Mean: {np.mean(scores):.4f} ± {np.std(scores):.3f}')
    print(f'  Compare: ArcCosine explore = 0.5555±0.013 (pre-audit), geodesic baseline = 0.5424±0.017')
"
