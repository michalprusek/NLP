"""
Flow Curvature Uncertainty (FCU) for Active Learning.

This module implements FCU - a novel uncertainty metric based on the
geometry of learned flow trajectories. The key insight is:

    Straight trajectories → Model is confident
    Curved trajectories → Model is uncertain

This is because:
1. Reflow training straightens trajectories in high-data regions
2. Out-of-distribution regions have chaotic velocity fields
3. Curvature measures velocity field consistency

FCU is used for cost-aware acquisition:
    - High FCU → Evaluate with expensive LLM
    - Low FCU → Trust cheap value head prediction
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass

from lido_pp.flow.ode_solver import compute_curvature_only, integrate


@dataclass
class FCUResult:
    """Result of FCU computation."""

    # Raw curvature values
    curvature: torch.Tensor  # (B,)

    # Normalized curvature (0-1 scale based on batch statistics)
    normalized_curvature: torch.Tensor  # (B,)

    # Binary decision: should evaluate with LLM?
    should_evaluate: torch.Tensor  # (B,) bool

    # Threshold used
    threshold: float


def compute_flow_curvature(
    model: nn.Module,
    z: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Compute Flow Curvature Uncertainty (FCU).

    FCU is the sum of velocity changes along the trajectory:
        K(z) = Σ ||v(x_t, t) - v(x_{t-dt}, t-dt)||

    For a perfectly straight trajectory (after perfect Reflow),
    velocity is constant so K = 0.

    For uncertain regions (chaotic velocity field), K is large.

    Args:
        model: FlowDiT model
        z: Starting noise samples (B, latent_dim)
        context: Optional context (B, num_ctx, context_dim)
        num_steps: Steps for curvature estimation

    Returns:
        curvature: (B,) curvature values (unnormalized)
    """
    return compute_curvature_only(model, z, context, num_steps)


def compute_fcu_with_threshold(
    model: nn.Module,
    z: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
    percentile: float = 90.0,
    min_samples_for_stats: int = 10,
) -> FCUResult:
    """
    Compute FCU with automatic threshold determination.

    Uses percentile-based threshold: samples with curvature above
    the threshold (e.g., top 10%) should be evaluated with LLM.

    Args:
        model: FlowDiT model
        z: Starting noise samples (B, latent_dim)
        context: Optional context
        num_steps: Steps for curvature estimation
        percentile: Threshold percentile (90 = top 10% get evaluated)
        min_samples_for_stats: Minimum samples for reliable statistics

    Returns:
        FCUResult with curvature, normalized values, and decisions
    """
    # Compute raw curvature
    curvature = compute_flow_curvature(model, z, context, num_steps)

    # Compute statistics for normalization
    batch_size = z.shape[0]

    if batch_size >= min_samples_for_stats:
        # Use batch statistics
        k_min = curvature.min()
        k_max = curvature.max()
        k_range = k_max - k_min + 1e-8

        # Normalize to [0, 1]
        normalized = (curvature - k_min) / k_range

        # Compute threshold
        threshold = torch.quantile(curvature, percentile / 100.0).item()
    else:
        # Too few samples - use heuristic
        normalized = curvature / (curvature.mean() + 1e-8)
        threshold = curvature.mean().item()

    # Determine which samples should be evaluated
    should_evaluate = curvature > threshold

    return FCUResult(
        curvature=curvature,
        normalized_curvature=normalized,
        should_evaluate=should_evaluate,
        threshold=threshold,
    )


class FlowCurvatureEstimator(nn.Module):
    """
    Module for efficient FCU computation with caching.

    Provides:
    1. Curvature computation with configurable steps
    2. Running statistics for normalization
    3. Adaptive threshold based on history
    """

    def __init__(
        self,
        flow_model: nn.Module,
        num_steps: int = 20,
        percentile: float = 90.0,
        momentum: float = 0.1,  # For running statistics
    ):
        super().__init__()
        self.flow_model = flow_model
        self.num_steps = num_steps
        self.percentile = percentile
        self.momentum = momentum

        # Running statistics (not trainable parameters)
        self.register_buffer("running_mean", torch.tensor(0.0))
        self.register_buffer("running_std", torch.tensor(1.0))
        self.register_buffer("num_samples", torch.tensor(0))

    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        update_stats: bool = True,
    ) -> FCUResult:
        """
        Compute FCU for batch of samples.

        Args:
            z: Starting noise (B, latent_dim)
            context: Optional context
            update_stats: Update running statistics

        Returns:
            FCUResult
        """
        with torch.no_grad():
            curvature = compute_flow_curvature(
                self.flow_model, z, context, self.num_steps
            )

        # Update running statistics
        if update_stats:
            batch_mean = curvature.mean()
            batch_std = curvature.std() + 1e-8

            if self.num_samples == 0:
                self.running_mean = batch_mean
                self.running_std = batch_std
            else:
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean +
                    self.momentum * batch_mean
                )
                self.running_std = (
                    (1 - self.momentum) * self.running_std +
                    self.momentum * batch_std
                )

            self.num_samples += z.shape[0]

        # Normalize using running statistics
        normalized = (curvature - self.running_mean) / self.running_std

        # Compute threshold (percentile of normalized values)
        if z.shape[0] >= 10:
            threshold_normalized = torch.quantile(
                normalized, self.percentile / 100.0
            ).item()
        else:
            threshold_normalized = 1.0  # Top ~16% assuming standard normal

        should_evaluate = normalized > threshold_normalized

        # Convert threshold back to raw scale
        threshold_raw = (
            threshold_normalized * self.running_std + self.running_mean
        ).item()

        return FCUResult(
            curvature=curvature,
            normalized_curvature=normalized,
            should_evaluate=should_evaluate,
            threshold=threshold_raw,
        )

    def get_statistics(self) -> Dict[str, float]:
        """Get running statistics."""
        return {
            "running_mean": self.running_mean.item(),
            "running_std": self.running_std.item(),
            "num_samples": self.num_samples.item(),
        }


def batch_fcu_analysis(
    model: nn.Module,
    z_batch: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    num_steps: int = 20,
) -> Dict[str, float]:
    """
    Analyze FCU distribution for a batch.

    Returns statistics useful for:
    1. Monitoring training progress (curvature should decrease)
    2. Setting thresholds for active learning
    3. Debugging flow model behavior

    Args:
        model: FlowDiT model
        z_batch: Batch of noise samples (B, latent_dim)
        context: Optional context
        num_steps: Steps for curvature

    Returns:
        Dict with statistics
    """
    curvature = compute_flow_curvature(model, z_batch, context, num_steps)

    return {
        "curvature_mean": curvature.mean().item(),
        "curvature_std": curvature.std().item(),
        "curvature_min": curvature.min().item(),
        "curvature_max": curvature.max().item(),
        "curvature_median": curvature.median().item(),
        "curvature_p90": torch.quantile(curvature, 0.9).item(),
        "curvature_p95": torch.quantile(curvature, 0.95).item(),
        "curvature_p99": torch.quantile(curvature, 0.99).item(),
    }


if __name__ == "__main__":
    from lido_pp.flow.flow_dit import FlowDiT

    print("Testing Flow Curvature Uncertainty...")

    # Create model
    model = FlowDiT(latent_dim=32, hidden_dim=256, num_layers=4)

    # Test batch
    z = torch.randn(100, 32)
    context = torch.randn(100, 4, 768)

    # Basic curvature
    print("\n1. Basic curvature computation:")
    curvature = compute_flow_curvature(model, z, context, num_steps=20)
    print(f"   Shape: {curvature.shape}")
    print(f"   Mean: {curvature.mean():.4f}")
    print(f"   Std: {curvature.std():.4f}")

    # FCU with threshold
    print("\n2. FCU with threshold (90th percentile):")
    fcu_result = compute_fcu_with_threshold(model, z, context, percentile=90.0)
    print(f"   Threshold: {fcu_result.threshold:.4f}")
    print(f"   Should evaluate: {fcu_result.should_evaluate.sum().item()}/{z.shape[0]}")

    # Batch analysis
    print("\n3. Batch FCU analysis:")
    stats = batch_fcu_analysis(model, z, context)
    for k, v in stats.items():
        print(f"   {k}: {v:.4f}")

    # FCU estimator with running stats
    print("\n4. FCU estimator with running stats:")
    estimator = FlowCurvatureEstimator(model, num_steps=20, percentile=90.0)

    for i in range(5):
        z_batch = torch.randn(50, 32)
        ctx_batch = torch.randn(50, 4, 768)
        result = estimator(z_batch, ctx_batch)
        stats = estimator.get_statistics()
        print(f"   Batch {i+1}: mean={stats['running_mean']:.4f}, "
              f"std={stats['running_std']:.4f}, "
              f"eval_rate={result.should_evaluate.float().mean():.2%}")

    print("\nAll tests passed!")
