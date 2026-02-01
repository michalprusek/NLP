"""Flow model evaluation: reconstruction MSE and text generation.

This module provides evaluation utilities for trained flow matching models,
including reconstruction quality measurement and text generation verification.

Usage:
    # CLI evaluation
    python -m study.flow_matching.evaluate \\
        --checkpoint study/checkpoints/mlp-icfm-1k-none/best.pt \\
        --arch mlp \\
        --test-split study/datasets/splits/1k/test.pt

    # Programmatic usage
    from study.flow_matching.evaluate import compute_reconstruction_mse, generate_and_decode
    mse_results = compute_reconstruction_mse(model, test_embeddings, n_samples=100, n_steps=100, device="cuda:0")
    texts = generate_and_decode(model, stats, decoder, n_samples=5, n_steps=100, device="cuda:0")
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from study.flow_matching.models import create_model
from study.data.normalize import denormalize, load_stats, DEFAULT_STATS_PATH
from ecoflow.decoder import SonarDecoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def euler_ode_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    show_progress: bool = True,
) -> Tensor:
    """
    Integrate ODE from t=0 to t=1 using Euler method.

    The flow ODE is: dx/dt = v(x, t) where v is the learned velocity network.
    Euler integration: x_{t+dt} = x_t + dt * v(x_t, t)

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Number of integration steps.
        device: Computation device.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    dt = 1.0 / n_steps
    x = x0.to(device)

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="ODE integration", leave=False)

    for i in iterator:
        t = i / n_steps
        # Expand t to batch dimension
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        # Velocity prediction
        v = model(x, t_batch)
        # Euler step
        x = x + dt * v

    return x


@torch.no_grad()
def compute_reconstruction_mse(
    model: torch.nn.Module,
    test_embeddings: Tensor,
    n_samples: int,
    n_steps: int,
    device: str | torch.device,
) -> dict:
    """
    Compute reconstruction MSE by comparing flow-generated embeddings to targets.

    For each test embedding x1 (target):
    1. Sample random noise x0 ~ N(0, I)
    2. Run Euler ODE integration from t=0 to t=1 to get x1_hat
    3. Compute MSE = mean((x1 - x1_hat)^2)

    Args:
        model: Velocity network in eval mode.
        test_embeddings: Normalized test embeddings of shape [N, D].
        n_samples: Number of samples to evaluate.
        n_steps: Number of ODE integration steps.
        device: Computation device.

    Returns:
        Dictionary with:
            - mse: Mean squared error (float)
            - std: Standard deviation of per-sample MSE (float)
            - n_samples: Number of samples evaluated
            - n_steps: Number of integration steps used
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    # Limit to n_samples
    n_available = test_embeddings.shape[0]
    n_actual = min(n_samples, n_available)
    if n_actual < n_samples:
        logger.warning(f"Only {n_available} test samples available, using {n_actual}")

    # Get target embeddings (normalized)
    x1_target = test_embeddings[:n_actual].to(device)

    # Sample random noise as starting points
    x0 = torch.randn_like(x1_target)

    # Run ODE integration
    logger.info(f"Running ODE integration for {n_actual} samples with {n_steps} steps...")
    x1_hat = euler_ode_integrate(model, x0, n_steps, device, show_progress=True)

    # Compute per-sample MSE
    per_sample_mse = ((x1_target - x1_hat) ** 2).mean(dim=1)  # [N]

    mse = per_sample_mse.mean().item()
    std = per_sample_mse.std().item()

    logger.info(f"Reconstruction MSE: {mse:.6f} +/- {std:.6f}")

    return {
        "mse": mse,
        "std": std,
        "n_samples": n_actual,
        "n_steps": n_steps,
    }


@torch.no_grad()
def generate_and_decode(
    model: torch.nn.Module,
    stats: dict,
    decoder: SonarDecoder,
    n_samples: int,
    n_steps: int,
    device: str | torch.device,
) -> list[str]:
    """
    Generate embeddings from noise and decode to text.

    Process:
    1. Sample random noise x0 ~ N(0, I) of shape [n_samples, 1024]
    2. Run Euler ODE integration to get x1_hat (normalized)
    3. Denormalize x1_hat using stats
    4. Decode using SonarDecoder

    Args:
        model: Velocity network in eval mode.
        stats: Normalization statistics dict with mean and std.
        decoder: SonarDecoder instance for embedding-to-text.
        n_samples: Number of samples to generate.
        n_steps: Number of ODE integration steps.
        device: Computation device.

    Returns:
        List of decoded text strings.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    # Sample random noise
    x0 = torch.randn(n_samples, 1024, device=device)

    # Run ODE integration
    logger.info(f"Generating {n_samples} samples with {n_steps} steps...")
    x1_hat_normalized = euler_ode_integrate(model, x0, n_steps, device, show_progress=True)

    # Denormalize
    x1_hat_denormalized = denormalize(x1_hat_normalized, stats)
    logger.info(f"Denormalized embeddings, mean: {x1_hat_denormalized.mean():.4f}")

    # Decode to text
    logger.info("Decoding embeddings to text...")
    texts = decoder.decode(x1_hat_denormalized)

    return texts


def load_checkpoint(checkpoint_path: str, arch: str, device: str | torch.device) -> tuple:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        arch: Architecture name ('mlp' or 'dit').
        device: Device to load model onto.

    Returns:
        Tuple of (model, stats) where stats is normalization statistics.
    """
    device = torch.device(device) if isinstance(device, str) else device

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Create model
    model = create_model(arch)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded {arch} model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Get stats from checkpoint (preferred) or default path
    if "normalization_stats" in checkpoint:
        stats = checkpoint["normalization_stats"]
        logger.info("Using normalization stats from checkpoint")
    else:
        logger.warning("Checkpoint missing normalization_stats, loading from default path")
        stats = load_stats(DEFAULT_STATS_PATH)

    return model, stats


def main():
    """CLI entry point for flow model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate flow model reconstruction and text generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate MLP checkpoint
    python -m study.flow_matching.evaluate \\
        --checkpoint study/checkpoints/mlp-icfm-1k-none/best.pt \\
        --arch mlp \\
        --test-split study/datasets/splits/1k/test.pt

    # Evaluate DiT checkpoint with more samples
    python -m study.flow_matching.evaluate \\
        --checkpoint study/checkpoints/dit-icfm-1k-none/best.pt \\
        --arch dit \\
        --n-mse-samples 200 \\
        --n-gen-samples 10
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["mlp", "dit"],
        help="Model architecture",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="study/datasets/splits/1k/test.pt",
        help="Path to test split file (default: study/datasets/splits/1k/test.pt)",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="Path to normalization stats (default: use checkpoint stats)",
    )
    parser.add_argument(
        "--n-mse-samples",
        type=int,
        default=100,
        help="Number of samples for MSE computation (default: 100)",
    )
    parser.add_argument(
        "--n-gen-samples",
        type=int,
        default=5,
        help="Number of samples for text generation (default: 5)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="ODE integration steps (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Verify paths exist
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.test_split).exists():
        logger.error(f"Test split not found: {args.test_split}")
        sys.exit(1)

    # Load model and stats
    model, stats = load_checkpoint(args.checkpoint, args.arch, args.device)

    # Override stats if path provided
    if args.stats_path:
        stats = load_stats(args.stats_path)
        logger.info(f"Overriding stats with {args.stats_path}")

    # Load test embeddings
    logger.info(f"Loading test split from {args.test_split}")
    test_data = torch.load(args.test_split, weights_only=False)
    test_embeddings = test_data["embeddings"]
    logger.info(f"Loaded {test_embeddings.shape[0]} test embeddings")

    # Compute reconstruction MSE
    print("\n" + "=" * 50)
    print("=== Reconstruction MSE ===")
    print("=" * 50)

    mse_results = compute_reconstruction_mse(
        model=model,
        test_embeddings=test_embeddings,
        n_samples=args.n_mse_samples,
        n_steps=args.n_steps,
        device=args.device,
    )

    print(f"MSE: {mse_results['mse']:.6f} +/- {mse_results['std']:.6f}")
    print(f"Samples: {mse_results['n_samples']}")
    print(f"Steps: {mse_results['n_steps']}")

    # Generate and decode text
    print("\n" + "=" * 50)
    print("=== Generated Text Samples ===")
    print("=" * 50)

    # Initialize decoder
    decoder = SonarDecoder(device=args.device)

    texts = generate_and_decode(
        model=model,
        stats=stats,
        decoder=decoder,
        n_samples=args.n_gen_samples,
        n_steps=args.n_steps,
        device=args.device,
    )

    for i, text in enumerate(texts, 1):
        print(f"[{i}] \"{text}\"")

    print("\n" + "=" * 50)
    print("Evaluation complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
