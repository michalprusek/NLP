"""LCB-guided flow sampling with CFG-Zero* schedule.

This module implements guided ODE sampling that combines a trained flow model
with a GP surrogate for Bayesian optimization. The guidance steers samples
toward high-scoring regions of SONAR embedding space while maintaining
diversity through the LCB (Lower Confidence Bound) exploration bonus.

Key features:
- CFG-Zero* schedule: Zero guidance for first 4% of steps to prevent
  early trajectory corruption
- LCB gradient guidance: Follows gradient of (mean - alpha * std)
  for principled exploration-exploitation
- Gradient normalization: Prevents gradient explosion
- Denormalization: Handles flow model trained on normalized data

Guided ODE equation: dz/dt = v(z, t) + lambda(t) * grad_LCB(z)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.ecoflow.flow_model import FlowMatchingModel
from src.ecoflow.gp_surrogate import SonarGPSurrogate


def cfg_zero_star_schedule(
    step: int,
    total_steps: int,
    guidance_strength: float,
    zero_init_fraction: float = 0.04,
) -> float:
    """
    CFG-Zero* guidance schedule.

    Returns 0 for first few steps, then full guidance strength.
    This prevents early trajectory corruption from inaccurate
    velocity estimates.

    Args:
        step: Current ODE step (0-indexed)
        total_steps: Total number of ODE steps
        guidance_strength: Maximum guidance strength (lambda)
        zero_init_fraction: Fraction of steps with zero guidance (default 4%)

    Returns:
        Effective guidance strength for this step

    Example:
        >>> cfg_zero_star_schedule(0, 50, 0.5, 0.04)
        0.0
        >>> cfg_zero_star_schedule(2, 50, 0.5, 0.04)
        0.5
    """
    zero_init_steps = max(1, int(zero_init_fraction * total_steps))

    if step < zero_init_steps:
        return 0.0
    else:
        return guidance_strength


class GuidedFlowSampler:
    """
    Samples from flow model with LCB guidance from GP surrogate.

    Implements the guided ODE: dz/dt = v(z, t) + lambda(t) * grad_LCB(z)
    where:
    - v(z, t) is the learned velocity field from the flow model
    - lambda(t) follows CFG-Zero* schedule (zero for first 4% of steps)
    - grad_LCB(z) is the gradient of mean - alpha * std from the GP

    The guidance steers samples toward high-scoring regions while
    maintaining diversity through the exploration bonus (alpha * std).

    Attributes:
        flow_model: Trained FlowMatchingModel
        gp: SonarGPSurrogate for acquisition function
        alpha: LCB exploration weight (higher = more exploration)
        guidance_strength: Maximum guidance strength lambda
        zero_init_fraction: Fraction of steps with zero guidance
        norm_stats: Normalization statistics for denormalization

    Example:
        >>> sampler = GuidedFlowSampler(flow_model, gp, alpha=1.0, guidance_strength=0.3)
        >>> samples = sampler.sample(n_samples=8, device='cuda', num_steps=50)
    """

    def __init__(
        self,
        flow_model: FlowMatchingModel,
        gp_surrogate: SonarGPSurrogate,
        alpha: float = 1.0,
        guidance_strength: float = 0.3,
        zero_init_fraction: float = 0.04,
        norm_stats: Optional[dict] = None,
    ):
        """
        Initialize guided flow sampler.

        Args:
            flow_model: Trained FlowMatchingModel for velocity computation
            gp_surrogate: SonarGPSurrogate for LCB gradient computation
            alpha: LCB exploration weight (default 1.0, use 1.96 for 95% CI)
            guidance_strength: Maximum guidance strength lambda (default 0.3)
            zero_init_fraction: Fraction of steps with zero guidance (default 0.04 = 4%)
            norm_stats: Normalization statistics {'mean': [1024], 'std': [1024]}
                        for denormalizing flow space to SONAR space
        """
        self.flow_model = flow_model
        self.gp = gp_surrogate
        self.alpha = alpha
        self.guidance_strength = guidance_strength
        self.zero_init_fraction = zero_init_fraction

        # Use flow_model's norm_stats if available and not overridden
        if norm_stats is not None:
            self.norm_stats = norm_stats
        elif hasattr(flow_model, 'norm_stats') and flow_model.norm_stats is not None:
            self.norm_stats = flow_model.norm_stats
        else:
            self.norm_stats = None

    def _get_guidance_lambda(self, step: int, total_steps: int) -> float:
        """Get guidance strength for current step using CFG-Zero* schedule."""
        return cfg_zero_star_schedule(
            step, total_steps, self.guidance_strength, self.zero_init_fraction
        )

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert from flow space to SONAR space.

        The flow model is trained on normalized data (mean=0, std=1).
        The GP expects inputs in original SONAR scale.

        Args:
            z: Tensor in flow space [B, 1024]

        Returns:
            Tensor in SONAR space [B, 1024]
        """
        if self.norm_stats is None:
            return z
        mean = self.norm_stats["mean"].to(z.device)
        std = self.norm_stats["std"].to(z.device)
        return z * std + mean

    def _compute_lcb_gradient(
        self, z_sonar: torch.Tensor, scale_to_flow_space: bool = True
    ) -> torch.Tensor:
        """
        Compute gradient of LCB acquisition function, scaled for flow space.

        LCB(z) = mu(z) - alpha * sigma(z)

        For maximization, we follow the positive gradient to find
        points with high predicted value and high uncertainty (exploration).

        The gradient is computed in SONAR space and converted to flow space
        so that it has appropriate magnitude relative to the velocity field.

        Args:
            z_sonar: Points in SONAR space [B, 1024]
            scale_to_flow_space: If True, convert gradient to flow space using
                                 chain rule. The flow model uses normalized coords
                                 where z_sonar = z_flow * std + mean, so
                                 d(LCB)/d(z_flow) = d(LCB)/d(z_sonar) / std

        Returns:
            Gradient [B, 1024] scaled for flow space
        """
        if self.gp.model is None:
            raise RuntimeError("GP must be fitted before computing guidance")

        # Compute gradient with autograd
        with torch.enable_grad():
            z_var = z_sonar.clone().requires_grad_(True)
            posterior = self.gp.model.posterior(z_var)

            mean = posterior.mean.squeeze(-1)  # [B]
            variance = posterior.variance.squeeze(-1)  # [B]
            std = torch.sqrt(variance + 1e-6)  # Numerical stability

            # LCB = mean - alpha * std
            lcb = mean - self.alpha * std

            # Gradient of LCB w.r.t. input
            grad_lcb = torch.autograd.grad(
                lcb.sum(),
                z_var,
                create_graph=False,
                retain_graph=False,
            )[0]

        # Convert gradient from SONAR space to flow space using chain rule
        # z_sonar = z_flow * std + mean
        # => d(LCB)/d(z_flow) = d(LCB)/d(z_sonar) / std
        # This makes the gradient magnitude appropriate for the flow's normalized space
        if scale_to_flow_space and self.norm_stats is not None:
            norm_std = self.norm_stats["std"].to(z_sonar.device)
            grad_lcb = grad_lcb / (norm_std + 1e-8)

        # Clip gradient norm to prevent explosion while preserving direction and scale
        # Max norm of 10.0 is reasonable for flow space (velocities are ~O(1))
        max_grad_norm = 10.0
        grad_norm = grad_lcb.norm(dim=-1, keepdim=True)
        clip_mask = grad_norm > max_grad_norm
        if clip_mask.any():
            grad_lcb = torch.where(
                clip_mask,
                grad_lcb * max_grad_norm / grad_norm,
                grad_lcb
            )

        return grad_lcb

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: str | torch.device = "cuda",
        num_steps: int = 50,
        method: str = "euler",
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate guided samples using ODE integration.

        Integrates the guided ODE from noise (t=0) to data (t=1):
        dz/dt = v(z, t) + lambda(t) * grad_LCB(z)

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            num_steps: Number of ODE integration steps
            method: Integration method ("euler" or "heun")
            return_trajectory: If True, return full trajectory [steps+1, B, 1024]

        Returns:
            Generated samples in SONAR space [n_samples, 1024]
            If return_trajectory=True: trajectory [steps+1, n_samples, 1024]
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.flow_model.velocity_net.eval()

        # Get input dimension from flow model
        input_dim = self.flow_model.input_dim

        # Start from noise at t=0
        z = torch.randn(n_samples, input_dim, device=device)

        trajectory = [z.clone()] if return_trajectory else None
        dt = 1.0 / num_steps

        if method == "euler":
            for i in range(num_steps):
                t = torch.tensor(i * dt, device=device)

                # Base velocity from flow model
                v = self.flow_model.ode_func(t, z)

                # Add guidance if GP is available and not in zero-init phase
                lambda_t = self._get_guidance_lambda(i, num_steps)
                if lambda_t > 0 and self.gp.model is not None:
                    # Denormalize z for GP (GP expects SONAR space)
                    z_sonar = self._denormalize(z)

                    # Compute normalized LCB gradient
                    grad_lcb = self._compute_lcb_gradient(z_sonar)

                    # For MAXIMIZATION: follow positive gradient toward higher LCB
                    v = v + lambda_t * grad_lcb

                # Euler step
                z = z + v * dt

                if return_trajectory:
                    trajectory.append(z.clone())

        elif method == "heun":
            for i in range(num_steps):
                t = torch.tensor(i * dt, device=device)
                t_next = torch.tensor((i + 1) * dt, device=device)

                # Predictor step (Euler)
                v1 = self.flow_model.ode_func(t, z)

                # Add guidance for predictor
                lambda_t = self._get_guidance_lambda(i, num_steps)
                if lambda_t > 0 and self.gp.model is not None:
                    z_sonar = self._denormalize(z)
                    grad_lcb = self._compute_lcb_gradient(z_sonar)
                    v1 = v1 + lambda_t * grad_lcb

                z_pred = z + v1 * dt

                # Corrector step
                v2 = self.flow_model.ode_func(t_next, z_pred)

                # Add guidance for corrector
                lambda_t_next = self._get_guidance_lambda(i + 1, num_steps)
                if lambda_t_next > 0 and self.gp.model is not None:
                    z_pred_sonar = self._denormalize(z_pred)
                    grad_lcb_pred = self._compute_lcb_gradient(z_pred_sonar)
                    v2 = v2 + lambda_t_next * grad_lcb_pred

                # Heun step: average of predictor and corrector velocities
                z = z + 0.5 * (v1 + v2) * dt

                if return_trajectory:
                    trajectory.append(z.clone())

        else:
            raise ValueError(f"Unknown integration method: {method}. Use 'euler' or 'heun'.")

        # Denormalize final samples to SONAR space
        z = self._denormalize(z)

        if return_trajectory:
            # Denormalize full trajectory
            trajectory = torch.stack(trajectory, dim=0)
            if self.norm_stats is not None:
                mean = self.norm_stats["mean"].to(device)
                std = self.norm_stats["std"].to(device)
                trajectory = trajectory * std + mean
            return trajectory

        return z

    def set_guidance_strength(self, guidance_strength: float) -> None:
        """Update guidance strength for ablation studies."""
        self.guidance_strength = guidance_strength

    def set_alpha(self, alpha: float) -> None:
        """Update LCB exploration weight."""
        self.alpha = alpha

    def update_gp(self, gp_surrogate: SonarGPSurrogate) -> None:
        """Update GP surrogate (called after each BO iteration)."""
        self.gp = gp_surrogate
