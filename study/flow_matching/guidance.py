"""CFG-Zero* guidance utilities for flow matching.

Implements the CFG-Zero* schedule from arXiv:2503.18886 which zeros guidance
for the first 4% of ODE integration steps to prevent early trajectory corruption.

This module provides:
- get_guidance_lambda(): CFG-Zero* schedule function
- guided_euler_ode_integrate(): ODE integration with optional guidance

For actual GP-guided sampling, see Phase 8 which integrates with GP surrogate.
"""

import logging
from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_guidance_lambda(
    step: int,
    total_steps: int,
    guidance_strength: float,
    zero_init_fraction: float = 0.04,
) -> float:
    """Get guidance strength using CFG-Zero* schedule.

    CFG-Zero* zeros guidance for the first 4% of steps to prevent
    early trajectory corruption when the flow estimate is inaccurate.

    Args:
        step: Current step (0-indexed)
        total_steps: Total number of ODE steps
        guidance_strength: Maximum guidance strength (lambda)
        zero_init_fraction: Fraction of steps with zero guidance (default 0.04 = 4%)

    Returns:
        Guidance strength for this step (0.0 during zero-init period)
    """
    zero_init_steps = max(1, int(zero_init_fraction * total_steps))
    if step < zero_init_steps:
        return 0.0
    return guidance_strength


@torch.no_grad()
def guided_euler_ode_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    guidance_fn: Optional[Callable[[Tensor], Tensor]] = None,
    guidance_strength: float = 1.0,
    zero_init_fraction: float = 0.04,
    grad_clip_norm: float = 10.0,
    show_progress: bool = True,
    velocity_scale: float = 1.0,
) -> Tensor:
    """
    Integrate ODE from t=0 to t=1 with optional CFG-Zero* guidance.

    The flow ODE is: dx/dt = v(x, t) + lambda(t) * grad_guidance(x)
    where lambda(t) follows the CFG-Zero* schedule (zero for first 4% of steps).

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Number of integration steps.
        device: Computation device.
        guidance_fn: Optional function that computes guidance gradient.
                    Should accept x [N, D] and return gradient [N, D].
                    If None, no guidance is applied.
        guidance_strength: Maximum guidance strength lambda.
        zero_init_fraction: Fraction of steps with zero guidance.
        grad_clip_norm: Maximum norm for guidance gradient clipping.
        show_progress: Whether to show tqdm progress bar.
        velocity_scale: Scale factor for model output (for SI-GVP trained with
                       normalized loss, use pi/2 ≈ 1.57 to un-normalize).

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    dt = 1.0 / n_steps
    x = x0.to(device)
    model.eval()

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Guided ODE integration", leave=False)

    for i in iterator:
        t = i / n_steps
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)

        # Base velocity (un-normalize if model was trained with normalized loss)
        v = model(x, t_batch) * velocity_scale

        # Add guidance (with CFG-Zero* schedule)
        if guidance_fn is not None:
            lambda_t = get_guidance_lambda(i, n_steps, guidance_strength, zero_init_fraction)
            if lambda_t > 0:
                grad = guidance_fn(x)

                # Gradient clipping
                grad_norm = grad.norm(dim=-1, keepdim=True)
                clip_mask = grad_norm > grad_clip_norm
                if clip_mask.any():
                    grad = torch.where(
                        clip_mask,
                        grad * grad_clip_norm / grad_norm,
                        grad
                    )

                v = v + lambda_t * grad

        # Euler step
        x = x + dt * v

    return x


import math as _math  # For SI velocity scale constant


def get_velocity_scale(flow_method: str) -> float:
    """Get velocity scale factor for un-normalizing model output.

    Models trained with SI-GVP (normalized loss) learn velocity / scale.
    At inference, multiply by scale to recover true velocity.

    The scale is chosen so that the target variance matches I-CFM:
    - I-CFM: E[||x1-x0||^2] = 2D
    - SI-GVP: E[||alpha_dot*x0 + sigma_dot*x1||^2] = (pi/2)^2 * D ≈ 2.47D
    - Scale = sqrt(2.47D / 2D) = pi / (2*sqrt(2)) ≈ 1.11

    Args:
        flow_method: Flow method name (e.g., 'icfm', 'si-gvp').

    Returns:
        Scale factor (1.0 for I-CFM/OT-CFM, ~1.11 for SI-GVP).
    """
    if flow_method in ("si-gvp", "si"):
        return _math.pi / (2 * _math.sqrt(2))  # ~1.11
    return 1.0


@torch.no_grad()
def sample_with_guidance(
    model: torch.nn.Module,
    n_samples: int,
    n_steps: int,
    device: str | torch.device,
    guidance_fn: Optional[Callable[[Tensor], Tensor]] = None,
    guidance_strength: float = 1.0,
    zero_init_fraction: float = 0.04,
    velocity_scale: float = 1.0,
    input_dim: int = 1024,
) -> Tensor:
    """
    Generate samples from flow model with optional CFG-Zero* guidance.

    Args:
        model: Velocity network in eval mode.
        n_samples: Number of samples to generate.
        n_steps: Number of ODE integration steps.
        device: Computation device.
        guidance_fn: Optional guidance gradient function.
        guidance_strength: Maximum guidance strength.
        zero_init_fraction: Fraction of steps with zero guidance.
        velocity_scale: Scale factor for model output (use get_velocity_scale(flow_method)).
        input_dim: Input dimension (default 1024 for SONAR).

    Returns:
        Generated samples [n_samples, input_dim] in normalized space.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Start from noise
    x0 = torch.randn(n_samples, input_dim, device=device)

    # Integrate with guidance
    x1 = guided_euler_ode_integrate(
        model=model,
        x0=x0,
        n_steps=n_steps,
        device=device,
        guidance_fn=guidance_fn,
        guidance_strength=guidance_strength,
        zero_init_fraction=zero_init_fraction,
        show_progress=True,
        velocity_scale=velocity_scale,
    )

    return x1


# Example guidance function for testing
def make_random_guidance_fn(target: Tensor) -> Callable[[Tensor], Tensor]:
    """Create a simple guidance function toward a target embedding.

    This is for testing only. Real guidance uses GP-UCB gradient.

    Args:
        target: Target embedding [1, D] or [D] to guide toward.

    Returns:
        Guidance function that returns gradient toward target.
    """
    if target.dim() == 1:
        target = target.unsqueeze(0)

    def guidance_fn(x: Tensor) -> Tensor:
        """Gradient toward target (normalized)."""
        diff = target.to(x.device) - x
        # Normalize to unit length for stable guidance
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return diff / diff_norm

    return guidance_fn
