"""Convergence plots for prompt optimization methods.

Generates convergence curves showing best accuracy vs prompts evaluated
for OPRO, ProTeGi, GEPA, and RieLBO-GSM8K.

Data sources:
- OPRO: opro/results/benchmark_100prompts_sonnet.json
- ProTeGi: protegi/results/benchmark_100prompts.json
- GEPA: gepa_gsm8k/results/benchmark_100prompts_batched.json
- RieLBO-GSM8K: rielbo_gsm8k/results/rielbo_s*.json

Usage:
    uv run python -m shared.plot_prompt_convergence
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("shared/results/plots")
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
    "opro": "#E65100",       # dark orange
    "protegi": "#2E7D32",    # dark green
    "gepa": "#7B1FA2",       # purple
    "rielbo": "#1565C0",     # dark blue
}

METHOD_LABELS = {
    "opro": "OPRO",
    "protegi": "ProTeGi",
    "gepa": "GEPA",
    "rielbo": "RieLBO-GSM8K",
}


def load_incremental_json(path: str | Path) -> list[dict] | None:
    """Load an IncrementalPromptSaver JSON file.

    Returns list of {eval_id, score} or None if file missing/empty.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    prompts = data.get("evaluated_prompts", [])
    if not prompts:
        return None
    return prompts


def extract_running_max(scores: list[float]) -> np.ndarray:
    """Compute running maximum over a list of scores."""
    running_max = []
    best = float("-inf")
    for s in scores:
        best = max(best, s)
        running_max.append(best)
    return np.array(running_max)


def load_opro_data() -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Load OPRO benchmark data. Returns list of (x, running_max) per seed."""
    paths = sorted(Path("opro/results").glob("benchmark_*sonnet*.json"))
    if not paths:
        paths = sorted(Path("opro/results").glob("benchmark_*.json"))
    if not paths:
        return None

    all_curves = []
    for p in paths:
        prompts = load_incremental_json(p)
        if prompts is None:
            continue
        scores = [ep["score"] for ep in prompts]
        x = np.arange(1, len(scores) + 1)
        y = extract_running_max(scores)
        all_curves.append((x, y))
    return all_curves if all_curves else None


def load_protegi_data() -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Load ProTeGi benchmark data."""
    paths = sorted(Path("protegi/results").glob("benchmark_*.json"))
    if not paths:
        return None

    all_curves = []
    for p in paths:
        prompts = load_incremental_json(p)
        if prompts is None:
            continue
        scores = [ep["score"] for ep in prompts]
        x = np.arange(1, len(scores) + 1)
        y = extract_running_max(scores)
        all_curves.append((x, y))
    return all_curves if all_curves else None


def load_gepa_data() -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Load GEPA benchmark data."""
    paths = sorted(Path("gepa_gsm8k/results").glob("benchmark_*.json"))
    if not paths:
        return None

    all_curves = []
    for p in paths:
        prompts = load_incremental_json(p)
        if prompts is None:
            continue
        scores = [ep["score"] for ep in prompts]
        x = np.arange(1, len(scores) + 1)
        y = extract_running_max(scores)
        all_curves.append((x, y))
    return all_curves if all_curves else None


def load_rielbo_gsm8k_data() -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Load RieLBO-GSM8K benchmark data."""
    paths = sorted(Path("rielbo_gsm8k/results").glob("rielbo_s*.json"))
    if not paths:
        return None

    all_curves = []
    for p in paths:
        prompts = load_incremental_json(p)
        if prompts is None:
            # Try history format (like GuacaMol)
            with open(p) as f:
                data = json.load(f)
            if "history" in data and "best_score" in data["history"]:
                bs = data["history"]["best_score"]
                x = np.arange(1, len(bs) + 1)
                y = np.array(bs)
                all_curves.append((x, y))
            continue
        scores = [ep["score"] for ep in prompts]
        x = np.arange(1, len(scores) + 1)
        y = extract_running_max(scores)
        all_curves.append((x, y))
    return all_curves if all_curves else None


def get_mean_std_curve(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std from multiple curves, handling different lengths."""
    if not curves:
        return np.array([]), np.array([]), np.array([])

    if len(curves) == 1:
        x, y = curves[0]
        return x, y, np.zeros_like(y)

    min_len = min(len(y) for _, y in curves)
    all_y = np.array([y[:min_len] for _, y in curves])
    x = np.arange(1, min_len + 1)
    mean = all_y.mean(axis=0)
    std = all_y.std(axis=0)
    return x, mean, std


def plot_convergence():
    """Create the main convergence plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    loaders = {
        "opro": load_opro_data,
        "protegi": load_protegi_data,
        "gepa": load_gepa_data,
        "rielbo": load_rielbo_gsm8k_data,
    }

    plotted = []
    for method, loader in loaders.items():
        curves = loader()
        if curves is None:
            print(f"  {method}: no data found, skipping")
            continue

        x, mean, std = get_mean_std_curve(curves)
        if len(x) == 0:
            continue

        c = METHOD_COLORS[method]
        n_seeds = len(curves)
        label = f"{METHOD_LABELS[method]} (n={n_seeds})"

        lw = 2.5 if method == "rielbo" else 2.0
        ax.plot(x, mean * 100, color=c, linewidth=lw, label=label)
        if n_seeds > 1:
            ax.fill_between(
                x,
                (mean - std) * 100,
                (mean + std) * 100,
                alpha=0.15,
                color=c,
            )
        plotted.append(method)

    if not plotted:
        print("No data found for any method. Nothing to plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Prompts Evaluated")
    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title("GSM8K Prompt Optimization — Convergence", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, None)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "prompt_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save as PDF for paper
    pdf_path = OUTPUT_DIR / "prompt_convergence.pdf"
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    for method, loader in loaders.items():
        curves = loader()
        if curves is None:
            continue
        x, mean, std = get_mean_std_curve(curves)
        if len(x) == 0:
            continue
        c = METHOD_COLORS[method]
        n_seeds = len(curves)
        label = f"{METHOD_LABELS[method]} (n={n_seeds})"
        lw = 2.5 if method == "rielbo" else 2.0
        ax2.plot(x, mean * 100, color=c, linewidth=lw, label=label)
        if n_seeds > 1:
            ax2.fill_between(x, (mean - std) * 100, (mean + std) * 100, alpha=0.15, color=c)

    ax2.set_xlabel("Prompts Evaluated")
    ax2.set_ylabel("Best Accuracy (%)")
    ax2.set_title("GSM8K Prompt Optimization — Convergence", fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, None)
    plt.tight_layout()
    fig2.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {pdf_path}")


def print_summary():
    """Print a summary table of final scores."""
    print("\n" + "=" * 70)
    print(f"{'Method':<18} {'Best Acc':>10} {'Prompts':>10} {'Seeds':>8}")
    print("=" * 70)

    loaders = {
        "opro": load_opro_data,
        "protegi": load_protegi_data,
        "gepa": load_gepa_data,
        "rielbo": load_rielbo_gsm8k_data,
    }

    for method, loader in loaders.items():
        curves = loader()
        if curves is None:
            print(f"{METHOD_LABELS[method]:<18} {'—':>10} {'—':>10} {'—':>8}")
            continue

        final_scores = [y[-1] for _, y in curves]
        n_prompts = [len(y) for _, y in curves]

        mean_score = np.mean(final_scores)
        std_score = np.std(final_scores) if len(final_scores) > 1 else 0
        mean_prompts = int(np.mean(n_prompts))

        if len(final_scores) > 1:
            score_str = f"{mean_score*100:.1f}±{std_score*100:.1f}%"
        else:
            score_str = f"{mean_score*100:.1f}%"

        print(
            f"{METHOD_LABELS[method]:<18} {score_str:>10} {mean_prompts:>10} {len(curves):>8}"
        )

    print("=" * 70)


def main():
    print("Prompt Optimization Convergence Plot")
    print("=" * 40)

    plot_convergence()
    print_summary()

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
