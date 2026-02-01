"""Flow model evaluation: distribution MSE and text generation.

This module provides evaluation utilities for trained flow matching models,
including distribution quality measurement and text generation verification.

NOTE ON ICFM EVALUATION:
ICFM (Independent Conditional Flow Matching) learns to transport samples from
N(0,I) noise to the data distribution. This is NOT reconstruction - the model
generates NEW samples in the data distribution, not specific target samples.

The "distribution MSE" measures how far generated samples are from random test
samples. Expected value ~1.0 for normalized data (variance ~1 per dimension).
This validates that generated samples have similar statistics to the data
distribution. The key quality metric is coherent text generation.

Solvers:
- euler: 1st order, simple but slow
- heun: 2nd order, better accuracy
- rk4: 4th order, highest accuracy
- dpm++: DPM-Solver++ 2nd order, fast convergence

Usage:
    # CLI evaluation
    python -m study.flow_matching.evaluate \\
        --checkpoint study/checkpoints/mlp-icfm-1k-none/best.pt \\
        --arch mlp \\
        --test-split study/datasets/splits/1k/test.pt \\
        --solver heun --n-steps 50

    # Programmatic usage
    from study.flow_matching.evaluate import compute_distribution_mse, generate_and_decode
    mse_results = compute_distribution_mse(model, test_embeddings, n_samples=100, n_steps=100, device="cuda:0", solver="heun")
    texts = generate_and_decode(model, stats, decoder, n_samples=5, n_steps=100, device="cuda:0", solver="heun")
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from study.flow_matching.models import create_model
from study.flow_matching.solvers import get_solver, list_solvers, euler_integrate
from study.data.normalize import denormalize, load_stats, DEFAULT_STATS_PATH
from ecoflow.decoder import SonarDecoder

logger = logging.getLogger(__name__)

# Type alias for solver names
SolverName = Literal["euler", "heun", "rk4", "dpm++", "dpm_solver_pp", "adaptive_heun"]


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
def compute_distribution_mse(
    model: torch.nn.Module,
    test_embeddings: Tensor,
    n_samples: int,
    n_steps: int,
    device: str | torch.device,
    solver: SolverName = "euler",
) -> dict:
    """
    Compute distribution MSE by comparing flow-generated embeddings to random test samples.

    NOTE: This is NOT reconstruction. ICFM generates NEW samples from the learned
    distribution, not reconstructions of specific targets. The MSE measures how
    similar generated samples are to the test set distribution.

    Expected MSE ~1.0 for normalized data, indicating:
    - Generated samples have unit variance (like the data)
    - Model successfully transports noise to data distribution

    Process:
    1. Sample random noise x0 ~ N(0, I)
    2. Run ODE integration from t=0 to t=1 to get x1_hat
    3. Compare x1_hat to random test embeddings (NOT paired)
    4. Compute MSE = mean((x1_test - x1_hat)^2)

    Args:
        model: Velocity network in eval mode.
        test_embeddings: Normalized test embeddings of shape [N, D].
        n_samples: Number of samples to evaluate.
        n_steps: Number of ODE integration steps.
        device: Computation device.
        solver: ODE solver to use (euler, heun, rk4, dpm++).

    Returns:
        Dictionary with:
            - mse: Mean squared error (float) - expected ~1.0 for normalized data
            - std: Standard deviation of per-sample MSE (float)
            - n_samples: Number of samples evaluated
            - n_steps: Number of integration steps used
            - solver: Solver name used
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

    # Run ODE integration with selected solver
    solver_fn = get_solver(solver)
    logger.info(f"Running {solver} ODE integration for {n_actual} samples with {n_steps} steps...")
    x1_hat = solver_fn(model, x0, n_steps, device, show_progress=True)

    # Compute per-sample MSE
    per_sample_mse = ((x1_target - x1_hat) ** 2).mean(dim=1)  # [N]

    mse = per_sample_mse.mean().item()
    std = per_sample_mse.std().item()

    logger.info(f"Distribution MSE: {mse:.6f} +/- {std:.6f} (expected ~1.0 for normalized data)")

    return {
        "mse": mse,
        "std": std,
        "n_samples": n_actual,
        "n_steps": n_steps,
        "solver": solver,
    }


# Backward compatibility alias
compute_reconstruction_mse = compute_distribution_mse


@torch.no_grad()
def compute_path_straightness(
    model: torch.nn.Module,
    test_embeddings: Tensor,
    n_samples: int,
    n_steps: int,
    device: str | torch.device,
) -> dict:
    """
    Measure how straight the flow trajectories are.

    Compares actual ODE trajectories to ideal straight lines from noise to
    generated samples. OT-CFM should produce straighter paths than I-CFM.

    Process:
    1. Sample noise x0 ~ N(0, I)
    2. Integrate ODE to get trajectory [x_0, x_1, ..., x_T]
    3. Compute ideal straight line from x0 to x_T (final generated sample)
    4. Measure deviation from ideal at each step

    Args:
        model: Velocity network in eval mode.
        test_embeddings: Normalized test embeddings [N, D] (unused, for consistency).
        n_samples: Number of trajectories to evaluate.
        n_steps: Number of ODE integration steps.
        device: Computation device.

    Returns:
        Dictionary with:
            - mean_path_deviation: Average L2 deviation from straight line
            - max_path_deviation: Maximum deviation across all samples/steps
            - path_variance: Variance of deviation along trajectories
            - n_samples: Number of samples evaluated
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    # Sample noise starting points
    x0 = torch.randn(n_samples, 1024, device=device)

    dt = 1.0 / n_steps
    x = x0.clone()
    trajectory = [x.clone()]

    # Integrate ODE and record trajectory
    logger.info(f"Computing path straightness for {n_samples} samples, {n_steps} steps...")
    for i in tqdm(range(n_steps), desc="ODE integration", leave=False):
        t = i / n_steps
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        v = model(x, t_batch)
        x = x + dt * v
        trajectory.append(x.clone())

    trajectory = torch.stack(trajectory)  # [n_steps+1, n_samples, 1024]
    x_T = trajectory[-1]  # Final generated samples

    # Compute ideal straight path from x0 to x_T
    ts = torch.linspace(0, 1, n_steps + 1, device=device).view(-1, 1, 1)
    ideal_path = (1 - ts) * x0.unsqueeze(0) + ts * x_T.unsqueeze(0)

    # Compute deviation from ideal straight line
    # L2 distance at each step
    deviation = ((trajectory - ideal_path) ** 2).sum(dim=-1).sqrt()  # [steps+1, n_samples]

    # Skip t=0 and t=1 (always zero deviation by construction)
    deviation_interior = deviation[1:-1]  # [steps-1, n_samples]

    mean_deviation = deviation_interior.mean().item()
    max_deviation = deviation_interior.max().item()
    path_variance = deviation_interior.var(dim=0).mean().item()  # avg variance along paths

    logger.info(f"Path straightness: mean={mean_deviation:.4f}, max={max_deviation:.4f}, var={path_variance:.6f}")

    return {
        "mean_path_deviation": mean_deviation,
        "max_path_deviation": max_deviation,
        "path_variance": path_variance,
        "n_samples": n_samples,
    }


@torch.no_grad()
def generate_and_decode(
    model: torch.nn.Module,
    stats: dict,
    decoder: SonarDecoder,
    n_samples: int,
    n_steps: int,
    device: str | torch.device,
    solver: SolverName = "euler",
) -> List[str]:
    """
    Generate embeddings from noise and decode to text.

    Process:
    1. Sample random noise x0 ~ N(0, I) of shape [n_samples, 1024]
    2. Run ODE integration to get x1_hat (normalized)
    3. Denormalize x1_hat using stats
    4. Decode using SonarDecoder

    Args:
        model: Velocity network in eval mode.
        stats: Normalization statistics dict with mean and std.
        decoder: SonarDecoder instance for embedding-to-text.
        n_samples: Number of samples to generate.
        n_steps: Number of ODE integration steps.
        device: Computation device.
        solver: ODE solver to use (euler, heun, rk4, dpm++).

    Returns:
        List of decoded text strings.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()

    # Sample random noise
    x0 = torch.randn(n_samples, 1024, device=device)

    # Run ODE integration with selected solver
    solver_fn = get_solver(solver)
    logger.info(f"Generating {n_samples} samples with {n_steps} {solver} steps...")
    x1_hat_normalized = solver_fn(model, x0, n_steps, device, show_progress=True)

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
        choices=["mlp", "dit", "unet", "mamba"],
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
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=list_solvers(),
        help=f"ODE solver (default: euler). Available: {list_solvers()}",
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

    # Compute distribution MSE
    print("\n" + "=" * 50)
    print("=== Distribution MSE ===")
    print("=" * 50)
    print("(Expected ~1.0 for normalized data - validates distribution match)")

    mse_results = compute_distribution_mse(
        model=model,
        test_embeddings=test_embeddings,
        n_samples=args.n_mse_samples,
        n_steps=args.n_steps,
        device=args.device,
        solver=args.solver,
    )

    print(f"Solver: {mse_results['solver']}")
    print(f"MSE: {mse_results['mse']:.6f} +/- {mse_results['std']:.6f}")
    print(f"Samples: {mse_results['n_samples']}")
    print(f"Steps: {mse_results['n_steps']}")

    # Compute path straightness
    print("\n" + "=" * 50)
    print("=== Path Straightness ===")
    print("=" * 50)
    print("(Lower values = straighter paths = better ODE integration)")

    straightness_results = compute_path_straightness(
        model=model,
        test_embeddings=test_embeddings,
        n_samples=args.n_mse_samples,
        n_steps=args.n_steps,
        device=args.device,
    )

    print(f"Mean deviation: {straightness_results['mean_path_deviation']:.4f}")
    print(f"Max deviation: {straightness_results['max_path_deviation']:.4f}")
    print(f"Path variance: {straightness_results['path_variance']:.6f}")
    print(f"Samples: {straightness_results['n_samples']}")

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
        solver=args.solver,
    )

    for i, text in enumerate(texts, 1):
        print(f"[{i}] \"{text}\"")

    print("\n" + "=" * 50)
    print("Evaluation complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
