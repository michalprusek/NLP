"""Combined plotting for benchmark runner + V2 results.

Generates convergence plots showing explore preset alongside baseline methods.

Usage:
    uv run python -m rielbo.benchmark.plot_combined
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.guacamol.constants import GUACAMOL_TASKS, GUACAMOL_TASK_DESCRIPTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Methods to plot (order = legend order)
METHOD_ORDER = ["explore", "turbo", "cmaes", "invbo", "lolbo"]

METHOD_COLORS = {
    "explore": "#e41a1c",   # Bright red (ours, BEST)
    "subspace": "#ff7f00",  # Orange (ours, V1)
    "turbo": "#7b8fa1",     # Steel blue
    "lolbo": "#a0b070",     # Olive green
    "cmaes": "#c4956a",     # Tan
    "invbo": "#6aadba",     # Dusty teal
    "random": "#b0b0b0",    # Light gray
}

METHOD_LABELS = {
    "explore": "Ours: Explore (S\u00B9\u2075)",
    "subspace": "Subspace BO V1 (S\u00B9\u2075)",
    "turbo": "TuRBO (R\u00B2\u2075\u2076)",
    "lolbo": "LOL-BO (DKL)",
    "cmaes": "CMA-ES (R\u00B2\u2075\u2076)",
    "invbo": "InvBO (DKL+Inv)",
    "random": "Random Search",
}

METHOD_LINESTYLES = {
    "explore": "-",
    "subspace": "--",
    "turbo": "-.",
    "lolbo": ":",
    "cmaes": "-.",
    "invbo": ":",
    "random": ":",
}


def load_all_results(
    benchmark_dir: str = "rielbo/results/benchmark/full",
    v2_dir: str = "rielbo/results/guacamol_v2",
    v2_configs: list[str] | None = None,
    target_iterations: int = 500,
) -> dict:
    """Load results from both benchmark runner and V2 directories.

    Returns:
        Dict mapping (method, task_id) -> list of history dicts
    """
    if v2_configs is None:
        v2_configs = ["explore"]

    results = defaultdict(list)

    benchmark_path = Path(benchmark_dir)
    if benchmark_path.exists():
        for filepath in benchmark_path.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            method = data.get("method")
            task = data.get("task_id")
            history = data.get("history", {})

            if not method or not task or "best_score" not in history:
                continue

            results[(method, task)].append({
                "best_score": history["best_score"],
                "final_best": data.get("best_score"),
                "seed": data.get("seed"),
            })

    v2_path = Path(v2_dir)
    if v2_path.exists():
        for filepath in v2_path.glob("v2_*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            config_name = data.get("config_name", "")
            task = data.get("task_id", "")
            history = data.get("history", {})
            iterations = data.get("args", {}).get("iterations", 0)

            if config_name not in v2_configs:
                continue
            if iterations != target_iterations:
                continue
            if "best_score" not in history:
                continue

            results[(config_name, task)].append({
                "best_score": history["best_score"],
                "final_best": data.get("best_score"),
                "seed": data.get("args", {}).get("seed"),
            })

    return results


def compute_statistics(histories: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std from multiple histories."""
    if not histories:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(h["best_score"]) for h in histories)
    scores = np.array([h["best_score"][:min_len] for h in histories])

    iterations = np.arange(min_len)
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    return iterations, mean_scores, std_scores


def plot_convergence(
    task_id: str,
    results: dict,
    output_path: str,
    methods: list[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Create convergence plot for one task."""
    fig, ax = plt.subplots(figsize=figsize)

    if methods is None:
        available = set(m for m, t in results.keys() if t == task_id)
        methods = [m for m in METHOD_ORDER if m in available]

    plotted = False
    for method in methods:
        histories = results.get((method, task_id), [])
        if not histories:
            continue

        iterations, mean_scores, std_scores = compute_statistics(histories)
        if len(iterations) == 0:
            continue

        color = METHOD_COLORS.get(method, "#95a5a6")
        label = METHOD_LABELS.get(method, method)
        ls = METHOD_LINESTYLES.get(method, "-")
        lw = 2.5 if method == "explore" else 1.5

        # Offset by cold start (100 points)
        iters_with_cold = iterations + 100

        ax.plot(
            iters_with_cold, mean_scores,
            color=color, label=f"{label} (n={len(histories)})",
            linewidth=lw, linestyle=ls, zorder=10 if method == "explore" else 5,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    title = GUACAMOL_TASK_DESCRIPTIONS.get(task_id, task_id)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Total evaluations", fontsize=11)
    ax.set_ylabel("Best score", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_summary_grid(
    results: dict,
    output_path: str,
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    cols: int = 4,
) -> None:
    """Create grid of convergence plots for all tasks."""
    if tasks is None:
        tasks = [t for t in GUACAMOL_TASKS if any(t == tt for _, tt in results.keys())]

    n = len(tasks)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, task_id in enumerate(tasks):
        ax = axes_flat[idx]

        if methods is None:
            available = set(m for m, t in results.keys() if t == task_id)
            plot_methods = [m for m in METHOD_ORDER if m in available]
        else:
            plot_methods = methods

        for method in plot_methods:
            histories = results.get((method, task_id), [])
            if not histories:
                continue

            iterations, mean_scores, std_scores = compute_statistics(histories)
            if len(iterations) == 0:
                continue

            color = METHOD_COLORS.get(method, "#95a5a6")
            label = METHOD_LABELS.get(method, method)
            ls = METHOD_LINESTYLES.get(method, "-")
            lw = 2.0 if method == "explore" else 1.2

            iters_with_cold = iterations + 100

            ax.plot(iters_with_cold, mean_scores, color=color, label=label,
                    linewidth=lw, linestyle=ls, zorder=10 if method == "explore" else 5)

        ax.set_title(task_id.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluations", fontsize=8)
        ax.set_ylabel("Best score", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for idx in range(len(tasks), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       ncol=min(len(labels), 6), fontsize=9,
                       bbox_to_anchor=(0.5, 1.0))
            break

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary grid: {output_path}")


def generate_results_table(results: dict, output_path: str | None = None) -> str:
    """Generate markdown table of final scores."""
    tasks = [t for t in GUACAMOL_TASKS if any(t == tt for _, tt in results.keys())]
    methods = []
    for m in METHOD_ORDER:
        if any(m == mm for mm, _ in results.keys()):
            methods.append(m)

    lines = []
    header = "| Task |" + " | ".join(METHOD_LABELS.get(m, m) for m in methods) + " |"
    sep = "|------|" + " | ".join(["------"] * len(methods)) + " |"
    lines.append(header)
    lines.append(sep)

    for task in tasks:
        row = [f"| {task} "]
        best_mean = -1

        for method in methods:
            histories = results.get((method, task), [])
            if histories:
                final_scores = [h["final_best"] for h in histories if h.get("final_best")]
                if final_scores:
                    mean = np.mean(final_scores)
                    std = np.std(final_scores)
                    n = len(final_scores)
                    if mean > best_mean:
                        best_mean = mean
                    row.append(f"| {mean:.4f}±{std:.3f} (n={n}) ")
                else:
                    row.append("| - ")
            else:
                row.append("| - ")

        row.append("|")
        lines.append("".join(row))

    table = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table + "\n")
        logger.info(f"Saved table: {output_path}")

    return table


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Combined benchmark + V2 plots")
    parser.add_argument("--benchmark-dir", default="rielbo/results/benchmark/full")
    parser.add_argument("--v2-dir", default="rielbo/results/guacamol_v2")
    parser.add_argument("--output-dir", default="rielbo/results/plots")
    parser.add_argument("--v2-configs", default="explore", help="V2 configs to include (comma-sep)")
    parser.add_argument("--tasks", default=None, help="Tasks to plot (comma-sep, default=all)")
    parser.add_argument("--summary-grid", action="store_true", default=True)
    parser.add_argument("--table", action="store_true", default=True)
    args = parser.parse_args()

    v2_configs = args.v2_configs.split(",") if args.v2_configs else ["explore"]
    tasks = args.tasks.split(",") if args.tasks else None

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_all_results(args.benchmark_dir, args.v2_dir, v2_configs)

    logger.info(f"Loaded results for {len(set(t for _, t in results.keys()))} tasks, "
                f"{len(set(m for m, _ in results.keys()))} methods")

    plot_tasks = tasks or [t for t in GUACAMOL_TASKS
                           if any(t == tt for _, tt in results.keys())]
    for task_id in plot_tasks:
        plot_convergence(
            task_id, results,
            os.path.join(args.output_dir, f"convergence_{task_id}.png"),
        )

    if args.summary_grid:
        plot_summary_grid(
            results,
            os.path.join(args.output_dir, "summary_grid.png"),
            tasks=plot_tasks,
        )

    if args.table:
        table = generate_results_table(
            results,
            os.path.join(args.output_dir, "results_table.md"),
        )
        print(table)


if __name__ == "__main__":
    main()
