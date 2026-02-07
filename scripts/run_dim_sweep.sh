#!/bin/bash
# Dimension sweep: V2 geodesic on adip, d=8..20, 10 seeds each
# ~24h total on single GPU
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_dim_sweep.sh
#
# Results saved to: rielbo/results/dim_sweep/d{DIM}_seed{SEED}.json

set -euo pipefail

TASK="adip"
DIMS=$(seq 8 20)
SEEDS=$(seq 42 51)
ITERS=500
COLD=100
OUTDIR="rielbo/results/dim_sweep"

mkdir -p "$OUTDIR"

total=$((13 * 10))
done=0

for d in $DIMS; do
  for seed in $SEEDS; do
    outfile="$OUTDIR/d${d}_seed${seed}.json"

    # Skip existing
    if [ -f "$outfile" ]; then
      done=$((done + 1))
      echo "[SKIP] d=$d seed=$seed ($outfile exists) [$done/$total]"
      continue
    fi

    done=$((done + 1))
    echo "========================================"
    echo "[RUN] d=$d seed=$seed  [$done/$total]"
    echo "========================================"

    uv run python -m rielbo.run_guacamol_subspace_v2 \
      --preset geodesic \
      --task-id "$TASK" \
      --subspace-dim "$d" \
      --n-cold-start "$COLD" \
      --iterations "$ITERS" \
      --seed "$seed" \
      2>&1 | tail -5

    # Find the most recent result file and move it
    latest=$(ls -t rielbo/results/guacamol_v2/v2_geodesic_${TASK}_s${seed}_*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
      mv "$latest" "$outfile"
      # Extract best score
      best=$(python3 -c "import json; print(f'{json.load(open(\"$outfile\"))[\"best_score\"]:.4f}')")
      echo "[DONE] d=$d seed=$seed -> best=$best"
    else
      echo "[WARN] d=$d seed=$seed: no output file found"
    fi

    echo ""
  done
done

echo "========================================"
echo "All runs complete. Aggregating..."
echo "========================================"

uv run python scripts/aggregate_dim_sweep.py
