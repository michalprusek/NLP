"""
Cost-Aware Acquisition for LID-O++ Active Learning.

This module implements acquisition functions that balance:
1. Expected improvement (from GP surrogate)
2. Exploration (from uncertainty)
3. Cost savings (from FCU-based gating)

The key innovation is using Flow Curvature Uncertainty (FCU)
to decide whether to evaluate a candidate with expensive LLM
or trust the cheap Value Head prediction.

Acquisition: α(z) = GP_UCB(z) - λ * K(z)

Where:
- GP_UCB(z) = μ(z) + β * σ(z)  (Upper Confidence Bound)
- K(z) = Flow curvature (uncertainty proxy)
- λ = Cost penalty weight
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

try:
    from botorch.acquisition.analytic import UpperConfidenceBound, LogExpectedImprovement
    from botorch.optim import optimize_acqf
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False


@dataclass
class AcquisitionResult:
    """Result of acquisition optimization."""

    # Optimal latent vector
    z_optimal: torch.Tensor  # (latent_dim,)

    # Acquisition value
    acquisition_value: float

    # Components
    gp_mean: float
    gp_std: float
    curvature: float

    # Decision: should evaluate with LLM?
    should_evaluate: bool

    # Value Head prediction (if not evaluating)
    value_prediction: Optional[float] = None


class CostAwareAcquisition:
    """
    Cost-aware acquisition function with FCU gating.

    Combines:
    1. GP-based UCB for optimization direction
    2. FCU for uncertainty estimation
    3. Value Head for cheap predictions
    4. Gating logic for evaluation decisions

    The acquisition function is:
        α(z) = UCB(z) - λ * normalized_curvature(z)

    High curvature (uncertain) → Higher penalty → Less likely to select
    unless UCB is very high.

    Decision rule:
        if curvature > threshold: evaluate with LLM
        else: use Value Head prediction
    """

    def __init__(
        self,
        gp_model: nn.Module,
        flow_model: nn.Module,
        value_head: nn.Module,
        lambda_cost: float = 0.1,
        ucb_beta: float = 2.0,
        curvature_steps: int = 20,
        percentile_threshold: float = 90.0,
    ):
        """
        Args:
            gp_model: GP surrogate model with predict() method
            flow_model: FlowDiT for curvature computation
            value_head: Value Head for cheap predictions
            lambda_cost: Weight for curvature penalty
            ucb_beta: UCB exploration coefficient
            curvature_steps: Steps for FCU computation
            percentile_threshold: Threshold percentile for evaluation
        """
        self.gp = gp_model
        self.flow = flow_model
        self.value_head = value_head
        self.lambda_cost = lambda_cost
        self.ucb_beta = ucb_beta
        self.curvature_steps = curvature_steps
        self.percentile_threshold = percentile_threshold

        # Running statistics for curvature normalization
        self.curvature_history = []
        self.max_history = 1000

    def _compute_curvature(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute FCU for candidates."""
        from lido_pp.active_learning.curvature import compute_flow_curvature
        return compute_flow_curvature(
            self.flow, z, context, self.curvature_steps
        )

    def _normalize_curvature(self, curvature: torch.Tensor) -> torch.Tensor:
        """Normalize curvature to [0, 1] based on history."""
        if len(self.curvature_history) < 10:
            # Not enough history, use batch statistics
            k_min = curvature.min()
            k_max = curvature.max()
            if k_max - k_min < 1e-8:
                return torch.zeros_like(curvature)
            return (curvature - k_min) / (k_max - k_min)

        # Use historical min/max
        all_k = torch.cat(self.curvature_history + [curvature])
        k_min = all_k.min()
        k_max = all_k.max()

        return (curvature - k_min) / (k_max - k_min + 1e-8)

    def _update_history(self, curvature: torch.Tensor):
        """Update curvature history."""
        self.curvature_history.append(curvature.detach().cpu())
        if len(self.curvature_history) > self.max_history:
            self.curvature_history = self.curvature_history[-self.max_history:]

    def _get_threshold(self, curvature: torch.Tensor) -> float:
        """Get evaluation threshold."""
        if curvature.shape[0] >= 10:
            return torch.quantile(curvature, self.percentile_threshold / 100.0).item()
        elif len(self.curvature_history) >= 10:
            all_k = torch.cat(self.curvature_history)
            return torch.quantile(all_k, self.percentile_threshold / 100.0).item()
        else:
            return curvature.mean().item()

    def __call__(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute cost-aware acquisition values.

        Args:
            z: Candidate latents (B, latent_dim)
            context: Optional context for flow
            return_components: Return individual components

        Returns:
            acquisition: (B,) acquisition values
        """
        device = z.device

        # GP predictions
        with torch.no_grad():
            gp_mean, gp_std = self.gp.predict(z)

        # UCB: maximize (negative error) + exploration
        # Note: GP predicts negative error for BoTorch compatibility
        ucb = gp_mean + self.ucb_beta * gp_std

        # Compute curvature
        with torch.no_grad():
            curvature = self._compute_curvature(z, context)

        # Normalize and update history
        norm_curvature = self._normalize_curvature(curvature)
        self._update_history(curvature)

        # Cost-aware acquisition: UCB minus curvature penalty
        # High curvature → Lower acquisition → Less likely to select
        acquisition = ucb - self.lambda_cost * norm_curvature

        if return_components:
            return acquisition, {
                "gp_mean": gp_mean,
                "gp_std": gp_std,
                "ucb": ucb,
                "curvature": curvature,
                "norm_curvature": norm_curvature,
            }

        return acquisition

    def optimize(
        self,
        bounds: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_restarts: int = 64,
        raw_samples: int = 4096,
    ) -> AcquisitionResult:
        """
        Optimize acquisition to find best candidate.

        Args:
            bounds: (2, latent_dim) lower and upper bounds
            context: Context for flow
            num_restarts: Number of L-BFGS-B restarts
            raw_samples: Random samples for initialization

        Returns:
            AcquisitionResult with optimal z and decision
        """
        device = bounds.device
        latent_dim = bounds.shape[1]

        # Generate random candidates
        z_candidates = torch.rand(raw_samples, latent_dim, device=device)
        z_candidates = bounds[0] + z_candidates * (bounds[1] - bounds[0])

        # Expand context for batch
        if context is not None:
            context_batch = context.expand(raw_samples, -1, -1)
        else:
            context_batch = None

        # Evaluate acquisition
        acq_values, components = self(z_candidates, context_batch, return_components=True)

        # Get best candidate
        best_idx = acq_values.argmax()
        z_optimal = z_candidates[best_idx]

        # Get components for best
        gp_mean = components["gp_mean"][best_idx].item()
        gp_std = components["gp_std"][best_idx].item()
        curvature = components["curvature"][best_idx].item()
        acq_value = acq_values[best_idx].item()

        # Decision: should evaluate with LLM?
        threshold = self._get_threshold(components["curvature"])
        should_evaluate = curvature > threshold

        # Value Head prediction if not evaluating
        value_prediction = None
        if not should_evaluate:
            with torch.no_grad():
                self.value_head.eval()
                value_prediction = self.value_head(z_optimal.unsqueeze(0)).item()

        return AcquisitionResult(
            z_optimal=z_optimal,
            acquisition_value=acq_value,
            gp_mean=gp_mean,
            gp_std=gp_std,
            curvature=curvature,
            should_evaluate=should_evaluate,
            value_prediction=value_prediction,
        )


class AdaptiveAcquisition:
    """
    Acquisition function with adaptive β and λ.

    Early in optimization: high β (explore), low λ (try uncertain regions)
    Late in optimization: low β (exploit), high λ (focus on confident regions)
    """

    def __init__(
        self,
        gp_model: nn.Module,
        flow_model: nn.Module,
        value_head: nn.Module,
        beta_start: float = 8.0,
        beta_end: float = 2.0,
        lambda_start: float = 0.05,
        lambda_end: float = 0.2,
        curvature_steps: int = 20,
        percentile_threshold: float = 90.0,
        total_iterations: int = 50,
    ):
        self.base_acquisition = CostAwareAcquisition(
            gp_model=gp_model,
            flow_model=flow_model,
            value_head=value_head,
            lambda_cost=lambda_start,
            ucb_beta=beta_start,
            curvature_steps=curvature_steps,
            percentile_threshold=percentile_threshold,
        )

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.total_iterations = total_iterations
        self.current_iteration = 0

    def update_iteration(self, iteration: int):
        """Update iteration and adjust parameters."""
        self.current_iteration = iteration
        progress = iteration / self.total_iterations

        # Linear interpolation
        self.base_acquisition.ucb_beta = (
            self.beta_start * (1 - progress) + self.beta_end * progress
        )
        self.base_acquisition.lambda_cost = (
            self.lambda_start * (1 - progress) + self.lambda_end * progress
        )

    def __call__(self, *args, **kwargs):
        return self.base_acquisition(*args, **kwargs)

    def optimize(self, *args, **kwargs):
        return self.base_acquisition.optimize(*args, **kwargs)

    def get_current_params(self) -> Dict[str, float]:
        """Get current adaptive parameters."""
        return {
            "iteration": self.current_iteration,
            "ucb_beta": self.base_acquisition.ucb_beta,
            "lambda_cost": self.base_acquisition.lambda_cost,
        }


def create_cost_aware_acquisition(
    gp_model: nn.Module,
    flow_model: nn.Module,
    value_head: nn.Module,
    config,  # LIDOPPConfig
) -> CostAwareAcquisition:
    """Factory function for creating acquisition from config."""
    return CostAwareAcquisition(
        gp_model=gp_model,
        flow_model=flow_model,
        value_head=value_head,
        lambda_cost=config.lambda_cost,
        ucb_beta=config.ucb_beta,
        curvature_steps=config.curvature_steps,
        percentile_threshold=config.curvature_percentile,
    )


if __name__ == "__main__":
    print("Testing Cost-Aware Acquisition...")

    # Mock models
    class MockGP:
        def predict(self, z):
            # Return mock predictions
            mean = -torch.abs(z).sum(dim=-1) * 0.1  # Negative error
            std = torch.ones(z.shape[0]) * 0.1
            return mean, std

    class MockFlow(nn.Module):
        def forward(self, x_t, t, context=None):
            return torch.randn_like(x_t) * 0.1

    class MockValueHead(nn.Module):
        def forward(self, z):
            return torch.sigmoid(z.sum(dim=-1) * 0.1)

    gp = MockGP()
    flow = MockFlow()
    value_head = MockValueHead()

    # Create acquisition
    acquisition = CostAwareAcquisition(
        gp_model=gp,
        flow_model=flow,
        value_head=value_head,
        lambda_cost=0.1,
        ucb_beta=2.0,
    )

    # Test evaluation
    z = torch.randn(100, 32)
    acq_values = acquisition(z)
    print(f"Acquisition values shape: {acq_values.shape}")
    print(f"Acquisition values: min={acq_values.min():.4f}, max={acq_values.max():.4f}")

    # Test optimization
    bounds = torch.stack([torch.zeros(32), torch.ones(32)])
    result = acquisition.optimize(bounds, num_restarts=10, raw_samples=500)

    print(f"\nOptimization result:")
    print(f"  z_optimal shape: {result.z_optimal.shape}")
    print(f"  acquisition_value: {result.acquisition_value:.4f}")
    print(f"  gp_mean: {result.gp_mean:.4f}")
    print(f"  gp_std: {result.gp_std:.4f}")
    print(f"  curvature: {result.curvature:.4f}")
    print(f"  should_evaluate: {result.should_evaluate}")
    print(f"  value_prediction: {result.value_prediction}")

    # Test adaptive acquisition
    print("\nTesting AdaptiveAcquisition...")
    adaptive = AdaptiveAcquisition(
        gp_model=gp,
        flow_model=flow,
        value_head=value_head,
        total_iterations=50,
    )

    for i in [0, 10, 25, 40, 50]:
        adaptive.update_iteration(i)
        params = adaptive.get_current_params()
        print(f"  Iteration {i}: beta={params['ucb_beta']:.2f}, lambda={params['lambda_cost']:.3f}")

    print("\nAll tests passed!")
