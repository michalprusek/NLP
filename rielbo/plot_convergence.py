"""Convergence plots for RieLBO / LOL-BO / TuRBO.

Generates:
1. Individual per-task convergence plots (adip, med2, etc.)
2. Combined multi-panel figure for all tasks
3. V2 ablation plot (baseline, order2, whitening, geodesic)
4. Summary table
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FULL_DIR = Path("rielbo/results/benchmark/full")
V2_DIR = Path("rielbo/results/guacamol_v2")
OUTPUT_DIR = Path("rielbo/results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

METHOD_COLORS = {
    "rielbo": "#1565C0",     # dark blue
    "lolbo": "#E65100",      # dark orange
    "turbo": "#2E7D32",      # dark green
    "baxus": "#9b59b6",      # purple
    "cmaes": "#f39c12",      # orange-gold
    "invbo": "#1abc9c",      # teal
}
METHOD_LABELS = {
    "rielbo": "RieLBO (S\u00b9\u2075)",
    "lolbo": "LOL-BO",
    "turbo": "TuRBO",
    "baxus": "BAxUS",
    "cmaes": "CMA-ES",
    "invbo": "InvBO",
}

# GuacaMol task full names
TASK_NAMES = {
    "adip": "Amlodipine MPO",
    "med1": "Median 1",
    "med2": "Median 2",
    "osmb": "Osimertinib MPO",
    "pdop": "Perindopril MPO",
    "rano": "Ranolazine MPO",
    "dhop": "Deco Hop",
    "shop": "Scaffold Hop",
    "siga": "Sitagliptin MPO",
    "valt": "Valsartan SMARTS",
    "zale": "Zaleplon MPO",
}


def load_full_results(method: str, task: str) -> list[dict]:
    """Load all seed results from the full benchmark directory."""
    results = []
    for f in sorted(FULL_DIR.glob(f"{method}_{task}_seed*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def load_v2_results(variant: str, task: str) -> list[dict]:
    """Load v2 results for a given variant (baseline, geodesic, etc.)."""
    results = []
    for f in sorted(V2_DIR.glob(f"v2_{variant}_{task}_s*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def load_rielbo(task: str) -> list[dict]:
    """Load best RieLBO variant: v2_geodesic if available, else subspace v1."""
    v2 = load_v2_results("geodesic", task)
    if v2:
        return v2
    return load_full_results("subspace", task)


def get_convergence_curves(results: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mean and std of best_score over iterations."""
    if not results:
        return np.array([]), np.array([]), np.array([])

    all_best = []
    for r in results:
        all_best.append(r["history"]["best_score"])

    min_len = min(len(b) for b in all_best)
    all_best = np.array([b[:min_len] for b in all_best])

    iterations = np.arange(min_len)
    mean = all_best.mean(axis=0)
    std = all_best.std(axis=0)

    return iterations, mean, std


def plot_on_ax(ax, task: str, show_legend: bool = True):
    """Plot convergence curves for one task onto a given axes."""
    plotted = []

    # RieLBO (v2 geodesic)
    rielbo = load_rielbo(task)
    if rielbo:
        iters, mean, std = get_convergence_curves(rielbo)
        if len(iters) > 0:
            c = METHOD_COLORS["rielbo"]
            label = f"{METHOD_LABELS['rielbo']} (n={len(rielbo)})"
            ax.plot(iters, mean, color=c, linewidth=2.5, label=label)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=c)
            plotted.append(("rielbo", rielbo))

    # LOL-BO
    lolbo = load_full_results("lolbo", task)
    if lolbo:
        iters, mean, std = get_convergence_curves(lolbo)
        if len(iters) > 0:
            c = METHOD_COLORS["lolbo"]
            label = f"{METHOD_LABELS['lolbo']} (n={len(lolbo)})"
            ax.plot(iters, mean, color=c, linewidth=2, label=label)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=c)
            plotted.append(("lolbo", lolbo))

    # TuRBO
    turbo = load_full_results("turbo", task)
    if turbo:
        iters, mean, std = get_convergence_curves(turbo)
        if len(iters) > 0:
            c = METHOD_COLORS["turbo"]
            label = f"{METHOD_LABELS['turbo']} (n={len(turbo)})"
            ax.plot(iters, mean, color=c, linewidth=2, label=label)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=c)
            plotted.append(("turbo", turbo))

    # BAxUS, CMA-ES, InvBO
    for method in ("baxus", "cmaes", "invbo"):
        results = load_full_results(method, task)
        if results:
            iters, mean, std = get_convergence_curves(results)
            if len(iters) > 0:
                c = METHOD_COLORS[method]
                label = f"{METHOD_LABELS[method]} (n={len(results)})"
                ax.plot(iters, mean, color=c, linewidth=2, label=label)
                ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=c)
                plotted.append((method, results))

    # Cold start line
    cold_start_scores = []
    for _, results in plotted:
        for r in results:
            bs = r["history"]["best_score"]
            if bs:
                cold_start_scores.append(bs[0])
    if cold_start_scores:
        cs = np.mean(cold_start_scores)
        ax.axhline(y=cs, color="gray", linestyle="--", alpha=0.5,
                    label=f"Cold start ({cs:.4f})")

    task_name = TASK_NAMES.get(task, task.upper())
    ax.set_title(task_name, fontweight="bold")
    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Best Score")
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    return len(plotted) > 0


def plot_individual(task: str):
    """Create a standalone convergence plot for one task."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    has_data = plot_on_ax(ax, task, show_legend=True)
    if not has_data:
        plt.close(fig)
        return
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"convergence_{task}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined(tasks: list[str]):
    """Create a combined multi-panel figure for all tasks."""
    n = len(tasks)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        plot_on_ax(axes[i], task, show_legend=True)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("GuacaMol Convergence — LSBO Method Comparison",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "convergence_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_v2_ablation():
    """Create an ablation plot comparing v2 variants on adip."""
    v2_variants = {
        "baseline": ("V2 Baseline", "#90CAF9"),
        "order2": ("V2 Order-2 Kernel", "#CE93D8"),
        "whitening": ("V2 Whitening", "#A5D6A7"),
        "geodesic": ("V2 Geodesic TR", "#1565C0"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, task in enumerate(["adip", "med2"]):
        ax = axes[task_idx]
        for variant, (label, color) in v2_variants.items():
            results = load_v2_results(variant, task)
            if not results:
                continue
            iters, mean, std = get_convergence_curves(results)
            if len(iters) == 0:
                continue
            ax.plot(iters, mean, color=color, linewidth=2,
                    label=f"{label} (n={len(results)})")
            ax.fill_between(iters, mean - std, mean + std, alpha=0.12, color=color)

        # Also overlay LOL-BO and TuRBO for reference
        for method, (c, ls) in [("lolbo", ("#E65100", "-")), ("turbo", ("#2E7D32", "-"))]:
            results = load_full_results(method, task)
            if not results:
                continue
            iters, mean, std = get_convergence_curves(results)
            if len(iters) > 0:
                ax.plot(iters, mean, color=c, linewidth=1.5, linestyle="--", alpha=0.7,
                        label=f"{METHOD_LABELS[method]} (n={len(results)})")

        task_name = TASK_NAMES.get(task, task.upper())
        ax.set_title(f"{task_name}", fontweight="bold")
        ax.set_xlabel("BO Iteration")
        ax.set_ylabel("Best Score")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)

    fig.suptitle("RieLBO V2 Ablation — Variant Comparison",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "v2_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary_table():
    """Print a summary table of final scores for all methods/tasks."""
    all_tasks = set()
    if FULL_DIR.exists():
        for f in FULL_DIR.glob("*.json"):
            if f.suffix == '.json':
                parts = f.stem.split("_")
                all_tasks.add(parts[1])

    # Also add tasks from v2 results
    if V2_DIR.exists():
        for f in V2_DIR.glob("v2_geodesic_*_s*.json"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                all_tasks.add(parts[2])

    tasks = sorted(all_tasks)

    all_methods = ["rielbo", "lolbo", "turbo", "baxus", "cmaes", "invbo"]
    header_parts = [f"{'Task':<8}"]
    for m in all_methods:
        header_parts.append(f"{METHOD_LABELS.get(m, m):>18}")
    header_parts.append(f"{'Winner':>12}")
    header = " ".join(header_parts)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for task in tasks:
        row = {}

        # RieLBO V2
        rielbo = load_rielbo(task)
        if rielbo:
            scores = [r["best_score"] for r in rielbo]
            row["rielbo"] = (np.mean(scores), np.std(scores), len(scores))

        # Other methods from full benchmark dir
        for method in ["lolbo", "turbo", "baxus", "cmaes", "invbo"]:
            results = load_full_results(method, task)
            if results:
                scores = [r["best_score"] for r in results]
                row[method] = (np.mean(scores), np.std(scores), len(scores))

        def fmt(key, width=18):
            if key not in row:
                return "\u2014".center(width)
            m, s, n = row[key]
            return f"{m:.4f}\u00b1{s:.3f}({n})".center(width)

        # Find winner
        best_method = max(row.keys(), key=lambda k: row[k][0]) if row else "\u2014"
        winner = METHOD_LABELS.get(best_method, best_method)

        parts = [f"{task:<8}"]
        for m in all_methods:
            parts.append(f"{fmt(m)}")
        parts.append(f"{winner:>12}")
        print(" ".join(parts))

    print(sep)


def main():
    # Discover all tasks from the full benchmark + v2
    all_tasks = set()
    if FULL_DIR.exists():
        for f in FULL_DIR.glob("*.json"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                all_tasks.add(parts[1])

    # Always include adip
    all_tasks.add("adip")

    # Skip valt (all zeros — model can't produce SMARTS patterns)
    tasks = sorted(t for t in all_tasks if t != "valt")

    print(f"Tasks to plot: {tasks}")

    # Individual plots
    for task in tasks:
        plot_individual(task)

    # Combined multi-panel
    if len(tasks) > 1:
        plot_combined(tasks)

    # V2 ablation (adip + med2)
    plot_v2_ablation()

    # Summary table
    print_summary_table()

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
