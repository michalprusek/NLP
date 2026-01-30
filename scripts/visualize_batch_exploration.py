#!/usr/bin/env python3
"""
Visualize batch BO exploration using UMAP projection.

Creates publication-quality figures showing how Local Penalization
spreads candidates across the embedding space vs greedy clustering.

Features:
- UMAP projection of 1024D SONAR embeddings to 2D
- Temporal coloring (iteration progression)
- LP vs Greedy comparison
- Batch diversity metrics overlay
- Publication-ready export (300 dpi, vector graphics)

Usage:
    # From checkpoint
    uv run python scripts/visualize_batch_exploration.py \
        --checkpoint results/stress_test/checkpoint_final.pt \
        --output results/figures/exploration_map.pdf

    # From multiple checkpoints (compare runs)
    uv run python scripts/visualize_batch_exploration.py \
        --checkpoints results/run1/checkpoint.pt results/run2/checkpoint.pt \
        --labels "LP (batch=8)" "Greedy (batch=8)" \
        --output results/figures/lp_vs_greedy.pdf

    # Real-time monitoring (updates every N seconds)
    uv run python scripts/visualize_batch_exploration.py \
        --checkpoint results/running/checkpoint.pt \
        --watch --interval 30
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    """Load optimization checkpoint."""
    data = torch.load(path, map_location=device, weights_only=False)
    return data


def load_metrics_json(path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """
    Project embeddings to 2D using UMAP.

    Args:
        embeddings: [N, D] array of embeddings
        n_neighbors: UMAP neighbor parameter (larger = more global structure)
        min_dist: UMAP min_dist parameter (smaller = tighter clusters)
        metric: Distance metric
        random_state: For reproducibility

    Returns:
        [N, 2] array of 2D coordinates
    """
    try:
        import umap
    except ImportError:
        logger.error("UMAP not installed. Run: uv pip install umap-learn")
        sys.exit(1)

    logger.info(f"Computing UMAP projection for {len(embeddings)} embeddings...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
        verbose=False,
    )

    projection = reducer.fit_transform(embeddings)
    logger.info(f"UMAP projection complete: {projection.shape}")

    return projection


def compute_batch_diversity(embeddings: np.ndarray, batch_size: int) -> List[float]:
    """
    Compute mean pairwise distance within each batch.

    Args:
        embeddings: [N, D] array
        batch_size: Size of each batch

    Returns:
        List of diversity scores per batch
    """
    from scipy.spatial.distance import pdist

    n_batches = len(embeddings) // batch_size
    diversities = []

    for i in range(n_batches):
        batch = embeddings[i * batch_size : (i + 1) * batch_size]
        if len(batch) > 1:
            distances = pdist(batch, metric="euclidean")
            diversities.append(np.mean(distances))
        else:
            diversities.append(0.0)

    return diversities


def create_exploration_figure(
    projections: np.ndarray,
    iterations: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "Batch BO Exploration Map",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    alpha: float = 0.7,
    show_colorbar: bool = True,
    highlight_best: bool = True,
) -> plt.Figure:
    """
    Create main exploration visualization.

    Args:
        projections: [N, 2] UMAP coordinates
        iterations: [N] iteration index for each point
        scores: [N] optional scores for sizing/coloring
        title: Figure title
        figsize: Figure size in inches
        cmap: Colormap for iteration coloring
        alpha: Point transparency
        show_colorbar: Whether to show iteration colorbar
        highlight_best: Highlight best-scoring point

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Size points by score if available
    if scores is not None:
        sizes = 20 + 80 * (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        sizes = 40

    # Main scatter plot colored by iteration
    scatter = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        c=iterations,
        s=sizes,
        alpha=alpha,
        cmap=cmap,
        edgecolors="white",
        linewidths=0.5,
    )

    # Highlight best point
    if highlight_best and scores is not None:
        best_idx = np.argmax(scores)
        ax.scatter(
            projections[best_idx, 0],
            projections[best_idx, 1],
            s=200,
            c="red",
            marker="*",
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
            label=f"Best (acc={scores[best_idx]:.3f})",
        )
        ax.legend(loc="upper right", fontsize=10)

    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Iteration", fontsize=12)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Remove spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def create_comparison_figure(
    projections_list: List[np.ndarray],
    iterations_list: List[np.ndarray],
    labels: List[str],
    scores_list: Optional[List[np.ndarray]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Create side-by-side comparison of multiple runs (e.g., LP vs Greedy).

    Args:
        projections_list: List of [N, 2] UMAP projections
        iterations_list: List of [N] iteration arrays
        labels: Labels for each run
        scores_list: Optional list of [N] score arrays
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_runs = len(projections_list)
    fig, axes = plt.subplots(1, n_runs, figsize=figsize)

    if n_runs == 1:
        axes = [axes]

    for i, (proj, iters, label) in enumerate(zip(projections_list, iterations_list, labels)):
        ax = axes[i]

        scores = scores_list[i] if scores_list else None
        sizes = 40 if scores is None else 20 + 80 * (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        scatter = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=iters,
            s=sizes,
            alpha=0.7,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.5,
        )

        # Highlight best
        if scores is not None:
            best_idx = np.argmax(scores)
            ax.scatter(
                proj[best_idx, 0],
                proj[best_idx, 1],
                s=150,
                c="red",
                marker="*",
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
            )

        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add colorbar to last subplot
        if i == n_runs - 1:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("Iteration", fontsize=10)

    plt.tight_layout()
    return fig


def create_diversity_figure(
    diversities_list: List[List[float]],
    labels: List[str],
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Create batch diversity comparison plot.

    Args:
        diversities_list: List of diversity scores per batch for each run
        labels: Labels for each run
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(diversities_list)))

    for divs, label, color in zip(diversities_list, labels, colors):
        iterations = range(1, len(divs) + 1)
        ax.plot(iterations, divs, "-o", label=label, color=color, markersize=6, linewidth=2)

    ax.set_xlabel("Batch Index", fontsize=12)
    ax.set_ylabel("Mean Pairwise Distance", fontsize=12)
    ax.set_title("Batch Diversity Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def create_score_progression_figure(
    scores_list: List[np.ndarray],
    labels: List[str],
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Create best-so-far score progression plot.

    Args:
        scores_list: List of score arrays for each run
        labels: Labels for each run
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_list)))

    for scores, label, color in zip(scores_list, labels, colors):
        # Compute best-so-far
        best_so_far = np.maximum.accumulate(scores)
        evaluations = range(1, len(scores) + 1)

        ax.plot(evaluations, best_so_far, "-", label=label, color=color, linewidth=2)
        ax.scatter(evaluations, scores, color=color, alpha=0.3, s=20)

    ax.set_xlabel("Number of Evaluations", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Sample Efficiency: Best Score vs Evaluations", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def extract_data_from_checkpoint(checkpoint: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings, iterations, and scores from checkpoint.

    Returns:
        embeddings: [N, D]
        iterations: [N] - which iteration each point came from
        scores: [N]
    """
    train_X = checkpoint["train_X"]
    train_Y = checkpoint["train_Y"]

    if isinstance(train_X, torch.Tensor):
        train_X = train_X.cpu().numpy()
    if isinstance(train_Y, torch.Tensor):
        train_Y = train_Y.cpu().numpy()

    # Flatten Y if needed
    if train_Y.ndim > 1:
        train_Y = train_Y.squeeze()

    # Infer iterations (assume batch_size from checkpoint or default)
    n_samples = len(train_X)

    # Try to get batch info from metrics
    metrics = checkpoint.get("metrics", {})
    n_observations = metrics.get("n_observations", [])

    if n_observations:
        # Reconstruct which iteration each sample came from
        iterations = np.zeros(n_samples, dtype=int)
        prev_n = 0
        for i, n in enumerate(n_observations):
            iterations[prev_n:n] = i
            prev_n = n
    else:
        # Fallback: assume sequential
        iterations = np.arange(n_samples)

    return train_X, iterations, train_Y


def main():
    parser = argparse.ArgumentParser(
        description="Visualize batch BO exploration using UMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to single checkpoint file",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Paths to multiple checkpoints for comparison",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each checkpoint (for comparison)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/exploration_map.pdf",
        help="Output path for figure (supports .pdf, .png, .svg)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output (default: 300)",
    )

    # UMAP parameters
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )

    # Visualization options
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[10, 8],
        help="Figure size in inches (default: 10 8)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap (default: viridis)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Batch BO Exploration Map",
        help="Figure title",
    )

    # Watch mode for real-time monitoring
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: update figure periodically",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Watch interval in seconds (default: 30)",
    )

    # Additional outputs
    parser.add_argument(
        "--save-diversity",
        action="store_true",
        help="Also save diversity comparison figure",
    )
    parser.add_argument(
        "--save-progression",
        action="store_true",
        help="Also save score progression figure",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for diversity computation (default: 8)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.checkpoint and not args.checkpoints:
        parser.error("Must provide --checkpoint or --checkpoints")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.checkpoint:
        # Single checkpoint mode
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = load_checkpoint(args.checkpoint)
        embeddings, iterations, scores = extract_data_from_checkpoint(checkpoint)

        # Compute UMAP
        projection = compute_umap_projection(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

        # Create figure
        fig = create_exploration_figure(
            projection,
            iterations,
            scores=scores,
            title=args.title,
            figsize=tuple(args.figsize),
            cmap=args.cmap,
        )

        # Save
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        logger.info(f"Saved exploration map: {args.output}")

        # Optional: diversity figure
        if args.save_diversity:
            diversities = compute_batch_diversity(embeddings, args.batch_size)
            div_fig = create_diversity_figure([diversities], ["Run 1"])
            div_path = output_path.with_name(output_path.stem + "_diversity" + output_path.suffix)
            div_fig.savefig(div_path, dpi=args.dpi, bbox_inches="tight")
            logger.info(f"Saved diversity plot: {div_path}")

        # Optional: progression figure
        if args.save_progression:
            prog_fig = create_score_progression_figure([scores], ["Run 1"])
            prog_path = output_path.with_name(output_path.stem + "_progression" + output_path.suffix)
            prog_fig.savefig(prog_path, dpi=args.dpi, bbox_inches="tight")
            logger.info(f"Saved progression plot: {prog_path}")

        plt.close("all")

    else:
        # Comparison mode
        labels = args.labels or [f"Run {i+1}" for i in range(len(args.checkpoints))]
        if len(labels) != len(args.checkpoints):
            parser.error("Number of --labels must match number of --checkpoints")

        all_embeddings = []
        all_iterations = []
        all_scores = []

        for path in args.checkpoints:
            logger.info(f"Loading checkpoint: {path}")
            checkpoint = load_checkpoint(path)
            emb, iters, scores = extract_data_from_checkpoint(checkpoint)
            all_embeddings.append(emb)
            all_iterations.append(iters)
            all_scores.append(scores)

        # Compute joint UMAP (fit on all, transform each)
        combined = np.vstack(all_embeddings)
        logger.info(f"Computing joint UMAP for {len(combined)} total embeddings...")

        try:
            import umap
        except ImportError:
            logger.error("UMAP not installed. Run: uv pip install umap-learn")
            sys.exit(1)

        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=42,
            n_components=2,
            verbose=False,
        )
        combined_proj = reducer.fit_transform(combined)

        # Split back
        projections = []
        idx = 0
        for emb in all_embeddings:
            projections.append(combined_proj[idx : idx + len(emb)])
            idx += len(emb)

        # Create comparison figure
        fig = create_comparison_figure(
            projections,
            all_iterations,
            labels,
            scores_list=all_scores,
            figsize=(7 * len(labels), 6),
        )

        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        logger.info(f"Saved comparison map: {args.output}")

        # Optional: diversity comparison
        if args.save_diversity:
            diversities = [compute_batch_diversity(e, args.batch_size) for e in all_embeddings]
            div_fig = create_diversity_figure(diversities, labels)
            div_path = output_path.with_name(output_path.stem + "_diversity" + output_path.suffix)
            div_fig.savefig(div_path, dpi=args.dpi, bbox_inches="tight")
            logger.info(f"Saved diversity comparison: {div_path}")

        # Optional: progression comparison
        if args.save_progression:
            prog_fig = create_score_progression_figure(all_scores, labels)
            prog_path = output_path.with_name(output_path.stem + "_progression" + output_path.suffix)
            prog_fig.savefig(prog_path, dpi=args.dpi, bbox_inches="tight")
            logger.info(f"Saved progression comparison: {prog_path}")

        plt.close("all")

    # Print summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("Summary Statistics")
    logger.info("=" * 50)

    if args.checkpoint:
        logger.info(f"Total samples: {len(embeddings)}")
        logger.info(f"Iterations: {iterations.max() + 1}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info(f"Best score: {scores.max():.4f} (idx={np.argmax(scores)})")

        if args.save_diversity:
            logger.info(f"Mean batch diversity: {np.mean(diversities):.4f}")
    else:
        for label, scores in zip(labels, all_scores):
            logger.info(f"\n{label}:")
            logger.info(f"  Samples: {len(scores)}")
            logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            logger.info(f"  Best: {scores.max():.4f}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
