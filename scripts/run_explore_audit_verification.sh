#!/bin/bash
# Clean 10-seed explore verification run after audit fixes.
#
# Fixes applied:
# 1. Cap sampling: area-uniform (rejection sampling with sin^(d-2)(θ))
# 2. ur_std_collapse: noise-scaled consistently with other thresholds
# 3. acqf_schedule: noise-relative thresholds matching UR-TR
# 4. GeodesicMaternKernel PD warning (documentation only)
#
# Expected runtime: ~10 x 12min = ~2 hours on A100

set -euo pipefail

RESULTS_DIR="rielbo/results/guacamol_v2_audit"
mkdir -p "$RESULTS_DIR"

echo "=== Explore Audit Verification Run ==="
echo "Start: $(date)"
echo "Results: $RESULTS_DIR"
echo ""

for seed in 42 43 44 45 46 47 48 49 50 51; do
    echo "[$(date +%H:%M:%S)] Starting seed $seed..."

    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset explore \
        --ur-std-low 0.05 \
        --task-id adip \
        --n-cold-start 100 \
        --iterations 500 \
        --seed $seed

    # Move the latest result to audit directory
    LATEST=$(ls -t rielbo/results/guacamol_v2/v2_explore_adip_s${seed}_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$RESULTS_DIR/"
        echo "[$(date +%H:%M:%S)] Seed $seed done: $(python3 -c "import json; d=json.load(open('$LATEST')); print(f'best={d[\"best_score\"]:.4f}')")"
    else
        echo "[$(date +%H:%M:%S)] WARNING: No result file found for seed $seed"
    fi
    echo ""
done

echo "=== Summary ==="
echo "End: $(date)"
python3 -c "
import json, glob, numpy as np
scores = []
for seed in range(42, 52):
    files = sorted(glob.glob(f'$RESULTS_DIR/v2_explore_adip_s{seed}_*.json'))
    if files:
        with open(files[-1]) as f:
            data = json.load(f)
        score = data['best_score']
        acqf = data.get('config', {}).get('acqf_schedule', 'N/A')
        scores.append(score)
        print(f'  s{seed}: {score:.4f} (acqf_schedule={acqf})')
if scores:
    print(f'  Mean: {np.mean(scores):.4f} ± {np.std(scores):.3f}')
    print(f'  Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}')
"
