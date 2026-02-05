"""Aggregate RieLBO v2 benchmark results.

Usage:
    uv run python -m rielbo.benchmark.aggregate_v2_results
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: str = "rielbo/results/guacamol_v2") -> list[dict]:
    """Load all result JSON files."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    for f in results_path.glob("v2_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return results


def aggregate_by_config_and_task(results: list[dict]) -> dict:
    """Group results by (config_name, task_id)."""
    grouped = defaultdict(list)

    for r in results:
        key = (r.get("config_name", "unknown"), r.get("task_id", "unknown"))
        grouped[key].append(r["best_score"])

    return grouped


def compute_statistics(scores: list[float]) -> dict:
    """Compute mean, std, min, max from scores."""
    if not scores:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}

    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "n": len(scores),
    }


def print_results_table(grouped: dict):
    """Print results as a markdown table."""
    # Get all configs and tasks
    configs = sorted(set(k[0] for k in grouped.keys()))
    tasks = sorted(set(k[1] for k in grouped.keys()))

    print("\n## RieLBO v2 Benchmark Results\n")

    # Header
    header = "| Config |"
    for task in tasks:
        header += f" {task} |"
    print(header)

    # Separator
    sep = "|--------|"
    for _ in tasks:
        sep += "----------|"
    print(sep)

    # Data rows
    for config in configs:
        row = f"| {config} |"
        for task in tasks:
            key = (config, task)
            if key in grouped:
                stats = compute_statistics(grouped[key])
                row += f" {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']}) |"
            else:
                row += " - |"
        print(row)

    print("")


def print_detailed_statistics(grouped: dict):
    """Print detailed statistics for each configuration."""
    print("\n## Detailed Statistics\n")

    for (config, task), scores in sorted(grouped.items()):
        stats = compute_statistics(scores)
        print(f"### {config} on {task}")
        print(f"- n: {stats['n']}")
        print(f"- mean: {stats['mean']:.4f}")
        print(f"- std: {stats['std']:.4f}")
        print(f"- min: {stats['min']:.4f}")
        print(f"- max: {stats['max']:.4f}")
        print(f"- scores: {[f'{s:.4f}' for s in sorted(scores)]}")
        print("")


def compare_to_baseline(grouped: dict):
    """Compare all configurations to baseline."""
    print("\n## Improvement over Baseline\n")

    tasks = sorted(set(k[1] for k in grouped.keys()))

    for task in tasks:
        print(f"### Task: {task}\n")

        baseline_key = ("baseline", task)
        if baseline_key not in grouped:
            print("Baseline not found\n")
            continue

        baseline_stats = compute_statistics(grouped[baseline_key])
        baseline_mean = baseline_stats["mean"]

        improvements = []
        for (config, t), scores in grouped.items():
            if t != task or config == "baseline":
                continue

            stats = compute_statistics(scores)
            improvement = ((stats["mean"] - baseline_mean) / baseline_mean) * 100
            improvements.append((config, stats["mean"], improvement, stats["n"]))

        # Sort by improvement
        improvements.sort(key=lambda x: -x[2])

        print(f"Baseline: {baseline_mean:.4f}\n")
        print("| Config | Mean | Improvement |")
        print("|--------|------|-------------|")
        for config, mean, imp, n in improvements:
            sign = "+" if imp >= 0 else ""
            print(f"| {config} | {mean:.4f} | {sign}{imp:.2f}% (n={n}) |")
        print("")


def main():
    results = load_results()

    if not results:
        print("No results found. Run the benchmark first.")
        return

    print(f"Loaded {len(results)} result files\n")

    grouped = aggregate_by_config_and_task(results)

    print_results_table(grouped)
    compare_to_baseline(grouped)
    print_detailed_statistics(grouped)


if __name__ == "__main__":
    main()
