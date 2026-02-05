"""Convergence plots for LOL-BO vs RieLBO (Subspace) vs TuRBO."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_DIR = Path("rielbo/results/benchmark/full")
OUTPUT_DIR = Path("rielbo/results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "subspace": "#2196F3",  # blue
    "lolbo": "#FF9800",     # orange
    "turbo": "#4CAF50",     # green
}
METHOD_LABELS = {
    "subspace": "RieLBO (Subspace S¹⁵)",
    "lolbo": "LOL-BO",
    "turbo": "TuRBO",
}


def load_results(method: str, task: str) -> list[dict]:
    """Load all seed results for a method/task."""
    results = []
    for f in sorted(RESULT_DIR.glob(f"{method}_{task}_seed*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def get_convergence_curves(results: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mean and std of best_score over iterations."""
    if not results:
        return np.array([]), np.array([]), np.array([])

    all_best = []
    for r in results:
        h = r["history"]
        all_best.append(h["best_score"])

    # Align to min length
    min_len = min(len(b) for b in all_best)
    all_best = np.array([b[:min_len] for b in all_best])

    iterations = np.arange(min_len)
    mean = all_best.mean(axis=0)
    std = all_best.std(axis=0)

    return iterations, mean, std


def plot_task(task: str, methods: list[str]):
    """Create convergence plot for one task."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for method in methods:
        results = load_results(method, task)
        if not results:
            continue

        iters, mean, std = get_convergence_curves(results)
        if len(iters) == 0:
            continue

        color = METHOD_COLORS.get(method, "#999")
        label = f"{METHOD_LABELS.get(method, method)} (n={len(results)})"

        ax.plot(iters, mean, color=color, linewidth=2, label=label)
        ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)

    # Cold start line
    cold_start_scores = []
    for method in methods:
        results = load_results(method, task)
        for r in results:
            if r["history"]["best_score"]:
                cold_start_scores.append(r["history"]["best_score"][0])
    if cold_start_scores:
        cs = np.mean(cold_start_scores)
        ax.axhline(y=cs, color="gray", linestyle="--", alpha=0.5, label=f"Cold start ({cs:.4f})")

    ax.set_xlabel("BO Iteration", fontsize=12)
    ax.set_ylabel("Best Score", fontsize=12)
    ax.set_title(f"Convergence — {task.upper()}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"convergence_{task}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    # Tasks with both lolbo and subspace
    tasks_with_both = []
    all_tasks = set()

    for f in RESULT_DIR.glob("*.json"):
        parts = f.stem.split("_")
        method, task = parts[0], parts[1]
        all_tasks.add(task)

    for task in sorted(all_tasks):
        lolbo_results = list(RESULT_DIR.glob(f"lolbo_{task}_seed*.json"))
        subspace_results = list(RESULT_DIR.glob(f"subspace_{task}_seed*.json"))
        if lolbo_results and subspace_results:
            tasks_with_both.append(task)

    print(f"Tasks with both LOL-BO and RieLBO: {tasks_with_both}")

    methods = ["subspace", "lolbo", "turbo"]

    for task in tasks_with_both:
        available = [m for m in methods if list(RESULT_DIR.glob(f"{m}_{task}_seed*.json"))]
        print(f"\n=== {task.upper()} === methods: {available}")
        plot_task(task, available)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
