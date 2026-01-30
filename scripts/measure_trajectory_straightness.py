#!/usr/bin/env python3
"""
Trajectory Straightness Measurement for Flow Models.

This script measures the straightness of flow model trajectories, a key metric
for evaluating flow matching quality and determining if reflow is needed.

Metrics computed:
1. Straightness ratio: straight_distance / path_length
   - 1.0 = perfectly straight trajectory
   - < 1.0 = curved trajectory (lower = more curved)

2. Curvature: Angular change between consecutive velocity vectors
   - Mean and max curvature across trajectories
   - Low curvature indicates consistent velocity direction

Theory:
- Flow matching learns a velocity field v(z, t) that transports noise to data
- Optimal Transport CFM (OT-CFM) produces straighter paths via OT coupling
- Standard CFM with good training can also achieve near-straight paths
- Reflow iterations further straighten paths without OT cost

Interpretation guide (from 09-RESEARCH.md):
- Straightness > 0.95: Flow is well-trained, no reflow needed
- Straightness 0.8-0.95: Consider reflow
- Straightness < 0.8: Reflow strongly recommended

Usage:
    python scripts/measure_trajectory_straightness.py \
        --checkpoint results/flow_checkpoints/best_flow.pt \
        --n-samples 1000 \
        --num-steps 100 \
        --output results/trajectory_analysis.json

Author: EcoFlow Team
Date: 2026-01-30
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_trajectory_with_steps(
    model,
    n_samples: int,
    num_steps: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate trajectories recording all intermediate steps.

    Args:
        model: FlowMatchingModel instance
        n_samples: Number of trajectories to generate
        num_steps: Number of ODE integration steps
        device: Device for computation

    Returns:
        trajectory: Tensor [num_steps+1, n_samples, input_dim]
    """
    model.velocity_net.eval()
    input_dim = model.input_dim

    # Start from noise at t=0
    z = torch.randn(n_samples, input_dim, device=device)
    trajectory = [z.clone()]

    dt = 1.0 / num_steps

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            v = model.ode_func(t, z)
            z = z + v * dt
            trajectory.append(z.clone())

    # Stack: [num_steps+1, n_samples, input_dim]
    return torch.stack(trajectory, dim=0)


def compute_velocities_along_trajectory(
    model,
    trajectory: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute velocity vectors at each point along trajectories.

    Args:
        model: FlowMatchingModel instance
        trajectory: Tensor [num_steps+1, n_samples, input_dim]
        device: Device for computation

    Returns:
        velocities: Tensor [num_steps, n_samples, input_dim]
    """
    num_steps = trajectory.shape[0] - 1
    n_samples = trajectory.shape[1]

    velocities = []
    dt = 1.0 / num_steps

    model.velocity_net.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            z = trajectory[i].to(device)
            v = model.ode_func(t, z)
            velocities.append(v)

    return torch.stack(velocities, dim=0)


def compute_straightness_metrics(trajectory: torch.Tensor) -> dict:
    """
    Compute straightness metrics from trajectory.

    Straightness = straight_distance / path_length
    - 1.0 = perfectly straight
    - < 1.0 = curved (lower = more curved)

    Args:
        trajectory: Tensor [num_steps+1, n_samples, input_dim]

    Returns:
        Dictionary with straightness statistics
    """
    # Compute path length: sum of L2 distances between consecutive points
    # diffs: [num_steps, n_samples, input_dim]
    diffs = trajectory[1:] - trajectory[:-1]

    # step_lengths: [num_steps, n_samples]
    step_lengths = diffs.norm(dim=-1)

    # path_lengths: [n_samples]
    path_lengths = step_lengths.sum(dim=0)

    # Compute straight-line distance: L2(z_final - z_initial)
    # straight_distances: [n_samples]
    straight_distances = (trajectory[-1] - trajectory[0]).norm(dim=-1)

    # Straightness ratio (1.0 = perfectly straight)
    # straightness: [n_samples]
    straightness = straight_distances / (path_lengths + 1e-8)

    return {
        "mean": float(straightness.mean()),
        "std": float(straightness.std()),
        "min": float(straightness.min()),
        "max": float(straightness.max()),
        "median": float(straightness.median()),
    }


def compute_curvature_metrics(velocities: torch.Tensor) -> dict:
    """
    Compute curvature metrics from velocity vectors.

    Curvature is measured as angular change between consecutive velocity vectors.
    Lower values indicate more consistent velocity direction (straighter paths).

    Args:
        velocities: Tensor [num_steps, n_samples, input_dim]

    Returns:
        Dictionary with curvature statistics
    """
    # Normalize velocity vectors
    v_norm = velocities / (velocities.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute cosine similarity between consecutive velocities
    # cos_sim: [num_steps-1, n_samples]
    cos_sim = (v_norm[1:] * v_norm[:-1]).sum(dim=-1)

    # Clamp to [-1, 1] for numerical stability
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    # Angular change in radians
    angles = torch.acos(cos_sim)

    # Mean angle change per sample
    # mean_angles: [n_samples]
    mean_angles = angles.mean(dim=0)

    # Max angle change per sample
    max_angles = angles.max(dim=0).values

    return {
        "mean_angle_change": float(mean_angles.mean()),
        "std_angle_change": float(mean_angles.std()),
        "max_angle_change": float(max_angles.mean()),
        "max_single_angle": float(angles.max()),
    }


def compute_path_length_metrics(trajectory: torch.Tensor) -> dict:
    """
    Compute path length statistics.

    Args:
        trajectory: Tensor [num_steps+1, n_samples, input_dim]

    Returns:
        Dictionary with path length statistics
    """
    diffs = trajectory[1:] - trajectory[:-1]
    step_lengths = diffs.norm(dim=-1)
    path_lengths = step_lengths.sum(dim=0)

    return {
        "mean": float(path_lengths.mean()),
        "std": float(path_lengths.std()),
        "min": float(path_lengths.min()),
        "max": float(path_lengths.max()),
    }


def load_flow_model(
    checkpoint_path: str,
    device: str = "cuda",
    use_ema: bool = True,
):
    """
    Load FlowMatchingModel from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        use_ema: Whether to use EMA weights

    Returns:
        FlowMatchingModel instance
    """
    from src.ecoflow.velocity_network import VelocityNetwork
    from src.ecoflow.flow_model import FlowMatchingModel

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", 512)
    num_layers = args.get("num_layers", 6)
    num_heads = args.get("num_heads", 8)

    logger.info(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}")

    # Create model
    velocity_net = VelocityNetwork(
        input_dim=1024,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Load weights
    if use_ema and "ema_shadow" in checkpoint:
        logger.info("Loading EMA weights")
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = {}
        for name, param in velocity_net.named_parameters():
            if name in ema_shadow:
                state_dict[name] = ema_shadow[name]
            else:
                logger.warning(f"Parameter {name} not found in EMA shadow")
        velocity_net.load_state_dict(state_dict, strict=False)
    else:
        logger.info("Loading model state dict")
        velocity_net.load_state_dict(checkpoint["model_state_dict"])

    velocity_net = velocity_net.to(device)
    velocity_net.eval()

    # Load normalization stats if available
    norm_stats = checkpoint.get("norm_stats", None)
    if norm_stats:
        logger.info("Loaded normalization stats")

    # Wrap in FlowMatchingModel
    model = FlowMatchingModel(velocity_net, norm_stats=norm_stats)

    best_loss = checkpoint.get('best_loss') or checkpoint.get('best_val_loss')
    loss_str = f"{best_loss:.6f}" if best_loss is not None else "N/A"
    logger.info(f"Model loaded. Epoch: {checkpoint.get('epoch', 'N/A')}, Best loss: {loss_str}")

    return model


def measure_straightness(
    checkpoint_path: str,
    n_samples: int = 1000,
    num_steps: int = 100,
    device: str = "cuda",
    output_path: Optional[str] = None,
    batch_size: int = 256,
) -> dict:
    """
    Measure trajectory straightness for a flow model checkpoint.

    Args:
        checkpoint_path: Path to flow model checkpoint
        n_samples: Number of trajectories to generate
        num_steps: Number of ODE integration steps
        device: Device for computation
        output_path: Optional path to save JSON results
        batch_size: Batch size for trajectory generation

    Returns:
        Dictionary with all metrics
    """
    # Load model
    model = load_flow_model(checkpoint_path, device=device)

    logger.info(f"Generating {n_samples} trajectories with {num_steps} steps...")

    # Generate trajectories in batches to manage memory
    all_trajectories = []
    remaining = n_samples

    while remaining > 0:
        batch_n = min(batch_size, remaining)
        logger.info(f"  Generating batch of {batch_n} trajectories...")

        trajectory = compute_trajectory_with_steps(
            model, batch_n, num_steps, device
        )
        all_trajectories.append(trajectory.cpu())
        remaining -= batch_n

    # Concatenate all batches: [num_steps+1, total_samples, input_dim]
    trajectory = torch.cat(all_trajectories, dim=1)
    logger.info(f"Total trajectory shape: {trajectory.shape}")

    # Compute metrics
    logger.info("Computing straightness metrics...")
    straightness = compute_straightness_metrics(trajectory)
    logger.info(f"  Straightness: {straightness['mean']:.4f} +/- {straightness['std']:.4f}")
    logger.info(f"  Range: [{straightness['min']:.4f}, {straightness['max']:.4f}]")

    # Interpretation
    if straightness['mean'] >= 0.95:
        interpretation = "EXCELLENT - No reflow needed"
    elif straightness['mean'] >= 0.8:
        interpretation = "GOOD - Consider reflow for improvement"
    else:
        interpretation = "POOR - Reflow strongly recommended"
    logger.info(f"  Interpretation: {interpretation}")

    logger.info("Computing curvature metrics...")
    # Reload model for velocity computation
    model_for_vel = load_flow_model(checkpoint_path, device=device)
    velocities = compute_velocities_along_trajectory(model_for_vel, trajectory.to(device), device)
    curvature = compute_curvature_metrics(velocities.cpu())
    logger.info(f"  Mean angle change: {curvature['mean_angle_change']:.4f} rad")
    logger.info(f"  Max angle change: {curvature['max_angle_change']:.4f} rad")

    logger.info("Computing path length metrics...")
    path_lengths = compute_path_length_metrics(trajectory)
    logger.info(f"  Path length: {path_lengths['mean']:.4f} +/- {path_lengths['std']:.4f}")

    # Compile results
    results = {
        "checkpoint": checkpoint_path,
        "n_samples": n_samples,
        "num_steps": num_steps,
        "straightness": straightness,
        "curvature": curvature,
        "path_lengths": path_lengths,
        "interpretation": interpretation,
    }

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure trajectory straightness for flow models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to flow model checkpoint",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of trajectories to generate (default: 1000)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of ODE integration steps (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON results (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for trajectory generation (default: 256)",
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        args.output = f"results/trajectory_analysis_{checkpoint_name}.json"

    # Run measurement
    results = measure_straightness(
        checkpoint_path=args.checkpoint,
        n_samples=args.n_samples,
        num_steps=args.num_steps,
        device=args.device,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY STRAIGHTNESS ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {results['n_samples']}")
    print(f"Steps: {results['num_steps']}")
    print("-" * 60)
    print(f"Straightness: {results['straightness']['mean']:.4f} +/- {results['straightness']['std']:.4f}")
    print(f"  Range: [{results['straightness']['min']:.4f}, {results['straightness']['max']:.4f}]")
    print(f"Mean angle change: {results['curvature']['mean_angle_change']:.4f} rad")
    print(f"Path length: {results['path_lengths']['mean']:.4f} +/- {results['path_lengths']['std']:.4f}")
    print("-" * 60)
    print(f"Interpretation: {results['interpretation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
