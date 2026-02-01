"""Normalization utilities for SONAR embeddings.

Per-dimension z-score normalization with stored statistics for flow model training.
Statistics are computed from training data ONLY to prevent data leakage.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Import centralized constants
from study.data import (
    DEFAULT_STATS_PATH,
    EMBEDDING_DIM,
    EPSILON,
    SPLITS_DIR,
)

# Default training path for computing stats
DEFAULT_TRAIN_PATH = f"{SPLITS_DIR}/10k/train.pt"


def compute_stats(embeddings: Tensor) -> dict:
    """
    Compute per-dimension mean and std from embeddings.

    IMPORTANT: Call ONLY on training data to prevent data leakage.

    Args:
        embeddings: Tensor of shape [N, D] where D is embedding dimension (1024).

    Returns:
        Dictionary containing:
            - mean: Tensor of shape [D], per-dimension means
            - std: Tensor of shape [D], per-dimension stds (with epsilon for stability)
            - n_samples: Number of samples used
            - source: Provenance string
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [N, D], got shape {embeddings.shape}")

    n_samples, dim = embeddings.shape
    logger.info(f"Computing normalization stats from {n_samples} samples, dim={dim}")

    # Compute per-dimension statistics
    mean = embeddings.mean(dim=0)  # [D]
    std = embeddings.std(dim=0)    # [D]

    # Add epsilon for numerical stability (prevent division by zero)
    std = std.clamp(min=EPSILON)

    # Verify shapes
    assert mean.shape == (dim,), f"Mean shape mismatch: {mean.shape}"
    assert std.shape == (dim,), f"Std shape mismatch: {std.shape}"

    # Verify no zero std (after clamping)
    assert (std > 0).all(), "Some dimensions have zero std"

    logger.info(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    logger.info(f"Std range: [{std.min():.4f}, {std.max():.4f}]")

    return {
        "mean": mean.float(),  # Ensure float32
        "std": std.float(),    # Ensure float32
        "n_samples": n_samples,
        "source": "10k_train",
    }


def save_stats(stats: dict, path: str) -> None:
    """
    Save normalization statistics to file.

    Args:
        stats: Dictionary from compute_stats()
        path: Output path for .pt file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(stats, path)
    logger.info(f"Saved normalization stats to {path}")

    # Verify save
    loaded = torch.load(path, weights_only=False)
    assert torch.allclose(loaded["mean"], stats["mean"])
    assert torch.allclose(loaded["std"], stats["std"])
    logger.info("Verified saved stats match original")


def load_stats(path: str = DEFAULT_STATS_PATH) -> dict:
    """
    Load normalization statistics from file.

    Args:
        path: Path to .pt file with stats

    Returns:
        Dictionary with mean, std, n_samples, source
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    stats = torch.load(path, weights_only=False)

    # Validate structure
    required_keys = {"mean", "std", "n_samples", "source"}
    if not required_keys.issubset(stats.keys()):
        missing = required_keys - set(stats.keys())
        raise ValueError(f"Stats file missing keys: {missing}")

    logger.info(f"Loaded stats from {path} (n_samples={stats['n_samples']}, source={stats['source']})")
    return stats


def normalize(embeddings: Tensor, stats: dict) -> Tensor:
    """
    Apply z-score normalization: (x - mean) / std

    Args:
        embeddings: Tensor of shape [N, D] or [D]
        stats: Dictionary with mean and std tensors

    Returns:
        Normalized embeddings with same shape as input
    """
    mean = stats["mean"]
    std = stats["std"]

    # Handle device mismatch
    if embeddings.device != mean.device:
        mean = mean.to(embeddings.device)
        std = std.to(embeddings.device)

    # Handle 1D case
    if embeddings.dim() == 1:
        return (embeddings - mean) / std

    # 2D case: broadcast mean and std
    return (embeddings - mean.unsqueeze(0)) / std.unsqueeze(0)


def denormalize(embeddings: Tensor, stats: dict) -> Tensor:
    """
    Reverse z-score normalization: x * std + mean

    Args:
        embeddings: Normalized tensor of shape [N, D] or [D]
        stats: Dictionary with mean and std tensors

    Returns:
        Denormalized embeddings with same shape as input

    Warns:
        If embeddings appear to already be denormalized (mean far from 0)
    """
    mean = stats["mean"]
    std = stats["std"]

    # Handle device mismatch
    if embeddings.device != mean.device:
        mean = mean.to(embeddings.device)
        std = std.to(embeddings.device)

    # Warning check: if embeddings have large mean, they might already be denormalized
    emb_mean = embeddings.mean().abs().item()
    if emb_mean > 5.0:
        logger.warning(
            f"Embeddings have large mean ({emb_mean:.2f}), "
            "they may already be denormalized. Double-denormalization will corrupt data."
        )

    # Handle 1D case
    if embeddings.dim() == 1:
        return embeddings * std + mean

    # 2D case: broadcast mean and std
    return embeddings * std.unsqueeze(0) + mean.unsqueeze(0)


def verify_round_trip(stats: dict, embeddings: Tensor, atol: float = 1e-5) -> bool:
    """
    Verify that normalize -> denormalize is identity within tolerance.

    Args:
        stats: Normalization statistics
        embeddings: Original embeddings to test
        atol: Absolute tolerance for comparison

    Returns:
        True if round-trip preserves embeddings within tolerance
    """
    normalized = normalize(embeddings, stats)
    recovered = denormalize(normalized, stats)

    max_diff = (embeddings - recovered).abs().max().item()
    logger.info(f"Round-trip max difference: {max_diff:.2e}")

    return torch.allclose(embeddings, recovered, atol=atol)


def verify_normalized_stats(embeddings: Tensor, rtol: float = 0.1) -> Tuple[float, float]:
    """
    Check that normalized embeddings have approximately zero mean and unit std.

    Args:
        embeddings: Normalized embeddings
        rtol: Relative tolerance

    Returns:
        Tuple of (mean_of_means, mean_of_stds)
    """
    # Per-dimension statistics
    dim_means = embeddings.mean(dim=0)
    dim_stds = embeddings.std(dim=0)

    mean_of_means = dim_means.mean().item()
    mean_of_stds = dim_stds.mean().item()

    logger.info(f"Normalized stats - mean of means: {mean_of_means:.4f}, mean of stds: {mean_of_stds:.4f}")

    return mean_of_means, mean_of_stds


def main():
    """Compute and optionally verify normalization statistics."""
    parser = argparse.ArgumentParser(description="Compute normalization statistics from training data")
    parser.add_argument("--compute-stats", action="store_true", help="Compute stats from 10K train split")
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH, help="Path to training split")
    parser.add_argument("--output", default=DEFAULT_STATS_PATH, help="Output path for stats file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not args.compute_stats and not args.verify:
        parser.print_help()
        sys.exit(1)

    if args.compute_stats:
        # Load training data
        logger.info(f"Loading training data from {args.train_path}")
        train_data = torch.load(args.train_path, weights_only=False)
        embeddings = train_data["embeddings"]

        logger.info(f"Loaded {embeddings.shape[0]} training embeddings")

        # Compute stats
        stats = compute_stats(embeddings)

        # Save stats
        save_stats(stats, args.output)

        logger.info(f"Stats computed from {stats['n_samples']} samples and saved to {args.output}")

    if args.verify:
        logger.info("Running verification checks...")

        # Load stats
        stats = load_stats(args.output)

        # Check 1: Stats file has correct shape
        assert stats["mean"].shape == torch.Size([EMBEDDING_DIM]), f"Mean shape wrong: {stats['mean'].shape}"
        assert stats["std"].shape == torch.Size([EMBEDDING_DIM]), f"Std shape wrong: {stats['std'].shape}"
        logger.info(f"[PASS] Stats have correct shape [{EMBEDDING_DIM}]")

        # Check 2: No zero std
        assert (stats["std"] > 0).all(), "Some dimensions have zero std"
        logger.info("[PASS] No zero std values")

        # Check 3: Round-trip test
        test_embeddings = torch.randn(100, 1024)
        assert verify_round_trip(stats, test_embeddings), "Round-trip failed"
        logger.info("[PASS] Round-trip is numerically stable")

        # Check 4: Normalized embeddings have ~0 mean and ~1 std
        train_data = torch.load(args.train_path, weights_only=False)
        normalized = normalize(train_data["embeddings"], stats)
        mean_of_means, mean_of_stds = verify_normalized_stats(normalized)

        assert abs(mean_of_means) < 0.01, f"Normalized mean too far from 0: {mean_of_means}"
        assert abs(mean_of_stds - 1.0) < 0.01, f"Normalized std too far from 1: {mean_of_stds}"
        logger.info("[PASS] Normalized embeddings have ~0 mean and ~1 std")

        # Check 5: n_samples matches training set
        assert stats["n_samples"] == train_data["embeddings"].shape[0], \
            f"n_samples mismatch: {stats['n_samples']} vs {train_data['embeddings'].shape[0]}"
        logger.info(f"[PASS] n_samples={stats['n_samples']} matches training set")

        logger.info("All verification checks passed!")


if __name__ == "__main__":
    main()
