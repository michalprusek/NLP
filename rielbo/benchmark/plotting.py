"""Plotting module for benchmark results.

Generates convergence plots comparing BO methods across tasks.

Usage:
    uv run python -m rielbo.benchmark.plotting \
        --results-dir rielbo/results/benchmark \
        --output-dir rielbo/results/benchmark/plots
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.guacamol.constants import GUACAMOL_TASK_DESCRIPTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color scheme for methods
METHOD_COLORS = {
    "subspace": "#d62728",  # Red (ours)
    "turbo": "#7b8fa1",     # Steel blue
    "lolbo": "#a0b070",     # Olive green
    "baxus": "#9b8bb4",     # Lavender
    "cmaes": "#c4956a",     # Tan
    "invbo": "#6aadba",     # Dusty teal
    "random": "#b0b0b0",    # Light gray
}

METHOD_LABELS = {
    "subspace": "Subspace BO (S^15)",
    "turbo": "TuRBO (R^256)",
    "lolbo": "LOLBO (DKL)",
    "baxus": "BAxUS (Adaptive)",
    "cmaes": "CMA-ES (R^256)",
    "invbo": "InvBO (DKL+Inv)",
}


def load_results(
    results_dir: str,
    methods: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict:
    """Load benchmark results from JSON files.

    Returns:
        Dict mapping (method, task) -> list of history dicts
    """
    results = defaultdict(list)

    for filepath in Path(results_dir).glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to load {filepath}")
            continue

        method = data.get("method")
        task = data.get("task_id")
        history = data.get("history", {})

        if methods and method not in methods:
            continue
        if tasks and task not in tasks:
            continue

        if "best_score" in history:
            results[(method, task)].append({
                "iteration": history.get("iteration", []),
                "best_score": history["best_score"],
                "seed": data.get("seed"),
                "final_best": data.get("best_score"),
            })

    return results


def compute_statistics(histories: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std from multiple histories.

    Returns:
        iterations, mean_scores, std_scores
    """
    if not histories:
        return np.array([]), np.array([]), np.array([])

    # Find minimum length (some runs may have different lengths)
    min_len = min(len(h["best_score"]) for h in histories)

    # Stack scores
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
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    show_std: bool = True,
) -> None:
    """Create convergence plot for one task.

    Args:
        task_id: GuacaMol task ID
        results: Dict from load_results()
        output_path: Path to save figure
        methods: Methods to include (None = all found)
        title: Custom title (default: task description)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if methods is None:
        methods = sorted(set(m for m, t in results.keys() if t == task_id))

    for method in methods:
        histories = results.get((method, task_id), [])
        if not histories:
            continue

        iterations, mean_scores, std_scores = compute_statistics(histories)

        if len(iterations) == 0:
            continue

        color = METHOD_COLORS.get(method, "#95a5a6")
        label = METHOD_LABELS.get(method, method)

        plt.plot(
            iterations,
            mean_scores,
            color=color,
            label=f"{label} (n={len(histories)})",
            linewidth=2,
        )

        if show_std:
            plt.fill_between(
                iterations,
                mean_scores - std_scores,
                mean_scores + std_scores,
                color=color,
                alpha=0.2,
            )

    if title is None:
        title = GUACAMOL_TASK_DESCRIPTIONS.get(task_id, task_id)
    plt.title(f"Convergence: {title}", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Score", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)

    all_scores = []
    for method in methods:
        histories = results.get((method, task_id), [])
        for h in histories:
            all_scores.extend(h["best_score"])

    if all_scores:
        y_min = min(all_scores)
        y_max = max(all_scores)
        padding = (y_max - y_min) * 0.05
        plt.ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved convergence plot to {output_path}")


def plot_all_tasks(
    results_dir: str,
    output_dir: str,
    methods: list[str] | None = None,
    tasks: list[str] | None = None,
    show_std: bool = True,
) -> None:
    """Generate convergence plots for all tasks.

    Args:
        results_dir: Directory containing JSON results
        output_dir: Directory to save plots
        methods: Methods to include (None = all)
        tasks: Tasks to include (None = all found)
    """
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_dir, methods, tasks)

    if tasks is None:
        tasks = sorted(set(t for _, t in results.keys()))

    logger.info(f"Generating plots for {len(tasks)} tasks")

    for task_id in tasks:
        output_path = os.path.join(output_dir, f"convergence_{task_id}.png")
        plot_convergence(
            task_id=task_id,
            results=results,
            output_path=output_path,
            methods=methods,
            show_std=show_std,
        )


def plot_summary_grid(
    results_dir: str,
    output_path: str,
    methods: list[str] | None = None,
    tasks: list[str] | None = None,
    cols: int = 4,
    show_std: bool = True,
) -> None:
    """Create a grid of convergence plots for all tasks.

    Args:
        results_dir: Directory containing JSON results
        output_path: Path to save combined figure
        methods: Methods to include
        tasks: Tasks to include
        cols: Number of columns in grid
    """
    results = load_results(results_dir, methods, tasks)

    if tasks is None:
        tasks = sorted(set(t for _, t in results.keys()))

    n_tasks = len(tasks)
    rows = (n_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, task_id in enumerate(tasks):
        ax = axes[idx]

        # Determine methods to plot
        plot_methods = methods or sorted(set(m for m, t in results.keys() if t == task_id))

        for method in plot_methods:
            histories = results.get((method, task_id), [])
            if not histories:
                continue

            iterations, mean_scores, std_scores = compute_statistics(histories)
            if len(iterations) == 0:
                continue

            color = METHOD_COLORS.get(method, "#95a5a6")
            label = METHOD_LABELS.get(method, method)

            ax.plot(iterations, mean_scores, color=color, label=label, linewidth=1.5)
            if show_std:
                ax.fill_between(
                    iterations,
                    mean_scores - std_scores,
                    mean_scores + std_scores,
                    color=color,
                    alpha=0.2,
                )

        ax.set_title(task_id, fontsize=11)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Best Score", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(len(tasks), len(axes)):
        axes[idx].set_visible(False)

    # Single legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for legend
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved summary grid to {output_path}")


def generate_results_table(
    results_dir: str,
    methods: list[str] | None = None,
    tasks: list[str] | None = None,
    output_path: str | None = None,
) -> str:
    """Generate markdown results table.

    Args:
        results_dir: Directory containing JSON results
        methods: Methods to include
        tasks: Tasks to include
        output_path: Optional path to save table

    Returns:
        Markdown table string
    """
    results = load_results(results_dir, methods, tasks)

    all_methods = sorted(set(m for m, _ in results.keys()))
    all_tasks = sorted(set(t for _, t in results.keys()))

    if methods:
        all_methods = [m for m in methods if m in all_methods]
    if tasks:
        all_tasks = [t for t in tasks if t in all_tasks]

    lines = []
    header = "| Task | Cold Start |" + " | ".join(all_methods) + " | Best Δ |"
    separator = "|------|------------|" + " | ".join(["------"] * len(all_methods)) + " | ----- |"
    lines.append(header)
    lines.append(separator)

    for task in all_tasks:
        row = [f"| {task} "]

        cold_start_scores = []
        for method in all_methods:
            histories = results.get((method, task), [])
            for h in histories:
                if h["best_score"]:
                    cold_start_scores.append(h["best_score"][0])

        cold_start = np.mean(cold_start_scores) if cold_start_scores else 0.0
        row.append(f"| {cold_start:.4f} ")

        best_mean = -float("inf")
        best_method = ""

        for method in all_methods:
            histories = results.get((method, task), [])
            if histories:
                final_scores = [h["final_best"] for h in histories if "final_best" in h]
                if not final_scores:
                    final_scores = [h["best_score"][-1] for h in histories if h["best_score"]]

                if final_scores:
                    mean = np.mean(final_scores)
                    std = np.std(final_scores)
                    row.append(f"| {mean:.4f} ± {std:.4f} ")

                    if mean > best_mean:
                        best_mean = mean
                        best_method = method
                else:
                    row.append("| - ")
            else:
                row.append("| - ")

        improvement = ((best_mean - cold_start) / cold_start * 100) if cold_start > 0 else 0
        row.append(f"| **{best_method}** +{improvement:.1f}% |")
        lines.append("".join(row))

    table = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved results table to {output_path}")

    return table


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="rielbo/results/benchmark",
        help="Directory containing JSON results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rielbo/results/benchmark/plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Methods to include (comma-separated)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Tasks to include (comma-separated)",
    )
    parser.add_argument(
        "--summary-grid",
        action="store_true",
        help="Generate combined grid plot",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Generate markdown results table",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable std shading in convergence plots",
    )

    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods else None
    tasks = args.tasks.split(",") if args.tasks else None

    os.makedirs(args.output_dir, exist_ok=True)

    if args.table:
        table = generate_results_table(
            args.results_dir,
            methods,
            tasks,
            os.path.join(args.output_dir, "results_table.md"),
        )
        print(table)

    show_std = not args.no_std

    if args.summary_grid:
        plot_summary_grid(
            args.results_dir,
            os.path.join(args.output_dir, "summary_grid.png"),
            methods,
            tasks,
            show_std=show_std,
        )

    plot_all_tasks(
        args.results_dir,
        args.output_dir,
        methods,
        tasks,
        show_std=show_std,
    )


if __name__ == "__main__":
    main()
