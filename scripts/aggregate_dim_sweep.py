"""Aggregate dimension sweep results.

Usage:
    uv run python scripts/aggregate_dim_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np

OUTDIR = Path("rielbo/results/dim_sweep")


def main():
    results = {}  # d -> [best_scores]

    for f in sorted(OUTDIR.glob("d*_seed*.json")):
        parts = f.stem.split("_")
        d = int(parts[0][1:])  # "d16" -> 16
        seed = int(parts[1][4:])  # "seed42" -> 42

        with open(f) as fh:
            data = json.load(fh)

        best = data["best_score"]
        results.setdefault(d, []).append(best)

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*65}")
    print(f"Dimension Sweep: V2 Geodesic on adip (500 iter, 100 cold start)")
    print(f"{'='*65}")
    print(f"{'d':>4} | {'n':>3} | {'mean':>7} | {'std':>6} | {'min':>7} | {'max':>7} | scores")
    print("-" * 65)

    for d in sorted(results.keys()):
        scores = results[d]
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        lo = np.min(scores)
        hi = np.max(scores)
        sc_str = ", ".join(f"{s:.4f}" for s in sorted(scores))
        marker = " ***" if mean == max(np.mean(results[dd]) for dd in results) else ""
        print(f"{d:4d} | {n:3d} | {mean:.4f} | {std:.4f} | {lo:.4f} | {hi:.4f} | {sc_str}{marker}")

    # Best dimension
    best_d = max(results, key=lambda d: np.mean(results[d]))
    best_mean = np.mean(results[best_d])
    best_std = np.std(results[best_d])
    print(f"\n>>> Best d = {best_d}  (mean = {best_mean:.4f} ± {best_std:.4f})")

    # Check completeness
    total = sum(len(v) for v in results.values())
    expected = 13 * 10
    if total < expected:
        missing = expected - total
        print(f"\n[WARN] {missing} runs still missing ({total}/{expected} complete)")
        for d in range(8, 21):
            n = len(results.get(d, []))
            if n < 10:
                print(f"  d={d}: {n}/10 seeds")


if __name__ == "__main__":
    main()
