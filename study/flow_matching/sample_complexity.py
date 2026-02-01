#!/usr/bin/env python3
"""Sample complexity analysis for flow matching models.

Fits power law scaling: loss ~ N^(-alpha) where N is dataset size.
This is essential for demonstrating sample efficiency claims in NeurIPS papers.

Usage:
    # Analyze from checkpoint directories
    uv run python -m study.flow_matching.sample_complexity \
        --checkpoints study/checkpoints/mlp-icfm-1k-none \
                      study/checkpoints/mlp-icfm-5k-none \
                      study/checkpoints/mlp-icfm-10k-none

    # Analyze from results JSON
    uv run python -m study.flow_matching.sample_complexity \
        --results study/results/ablation_dataset_*.json

    # Generate plot
    uv run python -m study.flow_matching.sample_complexity \
        --checkpoints study/checkpoints/mlp-*-none \
        --output study/results/sample_complexity.png
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Dataset size mapping
DATASET_SIZES = {
    "1k": 1000,
    "5k": 5000,
    "10k": 10000,
}


@dataclass
class SampleComplexityResult:
    """Results from sample complexity analysis."""

    # Raw data
    dataset_sizes: np.ndarray  # [K]
    losses: np.ndarray  # [K] or [K, S] if multi-seed

    # Fitted parameters: loss = C * N^(-alpha)
    alpha: float  # Scaling exponent
    alpha_ci: tuple[float, float]  # 95% CI for alpha
    log_c: float  # Log of constant factor
    log_c_ci: tuple[float, float]  # 95% CI for log_c

    # Fit quality
    r_squared: float  # Coefficient of determination
    residuals: np.ndarray  # Residuals from fit

    def __repr__(self) -> str:
        return (
            f"SampleComplexityResult(\n"
            f"  alpha={self.alpha:.4f} (95% CI: [{self.alpha_ci[0]:.4f}, {self.alpha_ci[1]:.4f}]),\n"
            f"  log_c={self.log_c:.4f} (95% CI: [{self.log_c_ci[0]:.4f}, {self.log_c_ci[1]:.4f}]),\n"
            f"  R²={self.r_squared:.4f}\n"
            f")"
        )

    def predict(self, n: int | np.ndarray) -> float | np.ndarray:
        """Predict loss for given dataset size(s)."""
        return np.exp(self.log_c) * np.power(n, -self.alpha)

    def extrapolate(self, target_loss: float) -> int:
        """Estimate dataset size needed to achieve target loss."""
        # loss = C * N^(-alpha)  =>  N = (loss / C)^(-1/alpha)
        return int(np.power(target_loss / np.exp(self.log_c), -1 / self.alpha))


def load_checkpoint_loss(checkpoint_path: Path) -> tuple[int, float]:
    """Load validation loss from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or best.pt file.

    Returns:
        (dataset_size, best_val_loss)
    """
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract dataset size from run name or path
    run_name = checkpoint_path.parent.name
    dataset_size = None
    for size_str, size_int in DATASET_SIZES.items():
        if f"-{size_str}-" in run_name or run_name.endswith(f"-{size_str}"):
            dataset_size = size_int
            break

    if dataset_size is None:
        raise ValueError(f"Could not determine dataset size from: {run_name}")

    # Get best validation loss (try multiple possible keys)
    best_val_loss = checkpoint.get("best_loss")  # Primary key
    if best_val_loss is None:
        best_val_loss = checkpoint.get("best_val_loss")
    if best_val_loss is None:
        best_val_loss = checkpoint.get("val_loss")

    if best_val_loss is None:
        raise ValueError(f"Could not find validation loss in checkpoint: {checkpoint_path}. Keys: {list(checkpoint.keys())}")

    return dataset_size, float(best_val_loss)


def fit_power_law(
    sizes: np.ndarray,
    losses: np.ndarray,
    confidence: float = 0.95,
) -> SampleComplexityResult:
    """Fit power law: loss = C * N^(-alpha).

    Uses linear regression in log-log space:
    log(loss) = log(C) - alpha * log(N)

    Args:
        sizes: Dataset sizes [K].
        losses: Corresponding losses [K] or [K, S] for multi-seed.
        confidence: Confidence level for intervals (default: 0.95).

    Returns:
        SampleComplexityResult with fitted parameters and statistics.
    """
    # Handle multi-seed case: use mean losses
    if losses.ndim == 2:
        mean_losses = losses.mean(axis=1)
        std_losses = losses.std(axis=1)
    else:
        mean_losses = losses
        std_losses = None

    # Log transform
    log_n = np.log(sizes)
    log_loss = np.log(mean_losses)

    # Weighted least squares (weight by inverse variance if available)
    if std_losses is not None:
        # Propagate uncertainty: d(log(x))/dx = 1/x
        weights = mean_losses / std_losses  # Inverse of log-space std
        weights = weights / weights.sum()  # Normalize
    else:
        weights = np.ones_like(log_n) / len(log_n)

    # Fit: log(loss) = log_c - alpha * log(n)
    # Using weighted linear regression
    n = len(sizes)
    w_sum = weights.sum()

    x_mean = np.sum(weights * log_n) / w_sum
    y_mean = np.sum(weights * log_loss) / w_sum

    # Slope (negative of alpha)
    numerator = np.sum(weights * (log_n - x_mean) * (log_loss - y_mean))
    denominator = np.sum(weights * (log_n - x_mean) ** 2)

    slope = numerator / denominator
    alpha = -slope
    log_c = y_mean + alpha * x_mean

    # Residuals and R²
    predicted = log_c - alpha * log_n
    residuals = log_loss - predicted
    ss_res = np.sum(weights * residuals ** 2)
    ss_tot = np.sum(weights * (log_loss - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors and confidence intervals
    # SE of slope: sqrt(MSE / sum((x - x_mean)^2))
    mse = ss_res / (n - 2) if n > 2 else 0.0
    se_slope = np.sqrt(mse / denominator) if denominator > 0 else 0.0
    se_alpha = se_slope

    # SE of intercept
    se_intercept = se_slope * np.sqrt(np.sum(weights * log_n ** 2) / w_sum)
    se_log_c = se_intercept

    # t-value for confidence interval
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, df=max(n - 2, 1))

    alpha_ci = (alpha - t_value * se_alpha, alpha + t_value * se_alpha)
    log_c_ci = (log_c - t_value * se_log_c, log_c + t_value * se_log_c)

    return SampleComplexityResult(
        dataset_sizes=sizes,
        losses=losses,
        alpha=alpha,
        alpha_ci=alpha_ci,
        log_c=log_c,
        log_c_ci=log_c_ci,
        r_squared=r_squared,
        residuals=residuals,
    )


def plot_sample_complexity(
    result: SampleComplexityResult,
    output_path: Optional[Path] = None,
    title: str = "Sample Complexity Analysis",
    show_extrapolation: bool = True,
) -> None:
    """Generate log-log plot of sample complexity.

    Args:
        result: Fitted SampleComplexityResult.
        output_path: Path to save plot (optional).
        title: Plot title.
        show_extrapolation: Whether to show extrapolation curve.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot actual data points
    if result.losses.ndim == 2:
        # Multi-seed: plot mean with error bars
        mean_losses = result.losses.mean(axis=1)
        std_losses = result.losses.std(axis=1)
        ax.errorbar(
            result.dataset_sizes,
            mean_losses,
            yerr=std_losses,
            fmt="o",
            markersize=10,
            capsize=5,
            label="Observed (mean ± std)",
            color="tab:blue",
            zorder=3,
        )
    else:
        ax.scatter(
            result.dataset_sizes,
            result.losses,
            s=100,
            label="Observed",
            color="tab:blue",
            zorder=3,
        )

    # Plot fitted power law
    if show_extrapolation:
        # Extrapolate to larger dataset sizes
        n_range = np.logspace(
            np.log10(result.dataset_sizes.min() * 0.5),
            np.log10(result.dataset_sizes.max() * 10),
            100,
        )
    else:
        n_range = np.logspace(
            np.log10(result.dataset_sizes.min()),
            np.log10(result.dataset_sizes.max()),
            100,
        )

    predicted = result.predict(n_range)
    ax.plot(
        n_range,
        predicted,
        "--",
        linewidth=2,
        label=f"Fit: loss ∝ N^(-{result.alpha:.3f})",
        color="tab:orange",
    )

    # Confidence band (approximate)
    alpha_lo, alpha_hi = result.alpha_ci
    log_c_lo, log_c_hi = result.log_c_ci

    # Upper and lower bounds
    upper = np.exp(log_c_hi) * np.power(n_range, -alpha_lo)
    lower = np.exp(log_c_lo) * np.power(n_range, -alpha_hi)
    ax.fill_between(n_range, lower, upper, alpha=0.2, color="tab:orange", label="95% CI")

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (N)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(f"{title}\n(α = {result.alpha:.3f}, R² = {result.r_squared:.3f})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Add data size labels
    for n, loss in zip(result.dataset_sizes,
                        result.losses.mean(axis=1) if result.losses.ndim == 2 else result.losses):
        ax.annotate(
            f"N={n:,}",
            (n, loss),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def analyze_checkpoints(checkpoint_paths: list[Path]) -> SampleComplexityResult:
    """Analyze sample complexity from checkpoint files.

    Args:
        checkpoint_paths: List of checkpoint directories or files.

    Returns:
        Fitted SampleComplexityResult.
    """
    # Load data from checkpoints
    data: dict[int, list[float]] = {}

    for path in checkpoint_paths:
        path = Path(path)
        try:
            size, loss = load_checkpoint_loss(path)
            if size not in data:
                data[size] = []
            data[size].append(loss)
            logger.info(f"  {path.parent.name}: N={size:,}, loss={loss:.6f}")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"  Skipping {path}: {e}")

    if len(data) < 2:
        raise ValueError(f"Need at least 2 different dataset sizes, got {len(data)}")

    # Convert to arrays
    sizes = np.array(sorted(data.keys()))

    # Check if we have multiple seeds per size
    max_seeds = max(len(v) for v in data.values())
    if max_seeds > 1:
        # Multi-seed: create [K, S] array
        losses = np.zeros((len(sizes), max_seeds))
        for i, size in enumerate(sizes):
            for j, loss in enumerate(data[size]):
                losses[i, j] = loss
            # Fill missing seeds with NaN (will be handled in fit)
            for j in range(len(data[size]), max_seeds):
                losses[i, j] = np.nan
        # Remove NaN columns if all seeds present
        if not np.any(np.isnan(losses)):
            pass  # Keep as is
        else:
            # Use only valid entries
            losses = np.array([data[size] for size in sizes], dtype=object)
            # Convert back to regular array with mean
            losses = np.array([np.mean(l) for l in losses])
    else:
        losses = np.array([data[size][0] for size in sizes])

    return fit_power_law(sizes, losses)


def main():
    parser = argparse.ArgumentParser(
        description="Sample complexity analysis for flow matching models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Checkpoint directories or files to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (PNG/PDF)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sample Complexity Analysis",
        help="Plot title",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Output path for JSON results",
    )

    args = parser.parse_args()

    if not args.checkpoints:
        # Default: analyze dataset ablation
        checkpoint_dir = Path("study/checkpoints")
        args.checkpoints = sorted(checkpoint_dir.glob("mlp-icfm-*-none/best.pt"))
        if not args.checkpoints:
            parser.error("No checkpoints found. Specify --checkpoints or run dataset ablation first.")

    # Expand glob patterns
    from glob import glob
    expanded = []
    for pattern in args.checkpoints:
        matches = glob(str(pattern))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    checkpoint_paths = [Path(p) for p in expanded]
    logger.info(f"Analyzing {len(checkpoint_paths)} checkpoints...")

    # Analyze
    result = analyze_checkpoints(checkpoint_paths)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE COMPLEXITY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"\nFitted power law: loss = {np.exp(result.log_c):.4f} × N^(-{result.alpha:.4f})")
    logger.info(f"Scaling exponent α = {result.alpha:.4f} (95% CI: [{result.alpha_ci[0]:.4f}, {result.alpha_ci[1]:.4f}])")
    logger.info(f"R² = {result.r_squared:.4f}")

    # Extrapolation examples
    logger.info("\nExtrapolation (predicted loss):")
    for n in [100, 1000, 10000, 100000]:
        pred = result.predict(n)
        logger.info(f"  N = {n:>6,}: loss = {pred:.6f}")

    # Inverse extrapolation
    logger.info("\nDataset size needed for target loss:")
    current_min = result.losses.min() if result.losses.ndim == 1 else result.losses.mean(axis=1).min()
    for target in [current_min * 0.5, current_min * 0.25, current_min * 0.1]:
        try:
            n_needed = result.extrapolate(target)
            logger.info(f"  loss = {target:.6f}: N = {n_needed:,}")
        except (ValueError, OverflowError):
            pass

    # Generate plot
    if args.output or args.output is None:
        output_path = Path(args.output) if args.output else Path("study/results/sample_complexity.png")
        plot_sample_complexity(result, output_path, title=args.title)

    # Save JSON
    if args.json:
        json_data = {
            "alpha": result.alpha,
            "alpha_ci": result.alpha_ci,
            "log_c": result.log_c,
            "log_c_ci": result.log_c_ci,
            "r_squared": result.r_squared,
            "dataset_sizes": result.dataset_sizes.tolist(),
            "losses": result.losses.tolist() if result.losses.ndim == 1 else result.losses.mean(axis=1).tolist(),
        }
        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"\nResults saved to: {args.json}")


def compare_scaling_laws(
    configs: list[str] = None,
    checkpoint_dir: Path = Path("study/checkpoints"),
    output_path: Optional[Path] = None,
) -> dict[str, SampleComplexityResult]:
    """Compare sample complexity across multiple configurations.

    Args:
        configs: List of config patterns (e.g., ["mlp-icfm", "unet-icfm"]).
        checkpoint_dir: Directory containing checkpoints.
        output_path: Path to save comparison plot.

    Returns:
        Dict mapping config name to SampleComplexityResult.
    """
    if configs is None:
        # Default: compare architectures with best flow (icfm based on results)
        configs = ["mlp-icfm", "dit-icfm", "unet-icfm"]

    results: dict[str, SampleComplexityResult] = {}

    for config in configs:
        # Find checkpoints matching pattern
        pattern = f"{config}-*-none"  # Only no-augmentation for fair comparison
        checkpoints = sorted(checkpoint_dir.glob(pattern))

        if len(checkpoints) < 2:
            logger.warning(f"Skipping {config}: need at least 2 dataset sizes, found {len(checkpoints)}")
            continue

        try:
            result = analyze_checkpoints(checkpoints)
            results[config] = result
            logger.info(f"\n{config}: α={result.alpha:.4f}, R²={result.r_squared:.4f}")
        except Exception as e:
            logger.warning(f"Failed to analyze {config}: {e}")

    # Generate comparison plot
    if output_path and results:
        _plot_comparison(results, output_path)

    return results


def _plot_comparison(
    results: dict[str, SampleComplexityResult],
    output_path: Path,
) -> None:
    """Generate comparison plot of multiple scaling laws."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping comparison plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10.colors

    for i, (name, result) in enumerate(results.items()):
        color = colors[i % len(colors)]

        # Plot data points
        losses = result.losses.mean(axis=1) if result.losses.ndim == 2 else result.losses
        ax.scatter(result.dataset_sizes, losses, s=100, color=color, zorder=3, label=f"{name} (data)")

        # Plot fitted line
        n_range = np.logspace(
            np.log10(min(r.dataset_sizes.min() for r in results.values()) * 0.5),
            np.log10(max(r.dataset_sizes.max() for r in results.values()) * 5),
            100,
        )
        predicted = result.predict(n_range)
        ax.plot(n_range, predicted, "--", linewidth=2, color=color,
                label=f"{name}: α={result.alpha:.3f} (R²={result.r_squared:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (N)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Sample Complexity Comparison\nloss ∝ N^(-α)", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"\nComparison plot saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
