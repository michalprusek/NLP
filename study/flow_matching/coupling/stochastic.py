"""Stochastic Interpolant coupling for flow matching.

Stochastic Interpolants (Albergo et al., 2023) generalize flow matching by
allowing non-linear interpolation schedules. Unlike I-CFM which uses constant
velocity x1-x0, SI computes velocity as the derivative of the interpolation.

Key difference from I-CFM:
- I-CFM: u_t = x1 - x0 (constant velocity)
- SI: u_t = alpha_dot*x0 + sigma_dot*x1 (time-varying velocity)
"""

import math
from typing import Tuple

import torch

from study.flow_matching.schedules import ScheduleFunction, get_schedule


# Velocity normalization factors to match I-CFM loss scale.
# GVP has (pi/2)^2 velocity variance vs 2 for I-CFM, so we scale by pi/(2*sqrt(2)).
_VELOCITY_SCALE = {
    "linear": 1.0,
    "gvp": math.pi / (2 * math.sqrt(2)),
}


class StochasticInterpolantCoupling:
    """Stochastic Interpolant coupling with configurable schedule.

    Implements stochastic interpolants where:
    - x_t = alpha_t * x0 + sigma_t * x1 (interpolation)
    - u_t = alpha_dot * x0 + sigma_dot * x1 (velocity target)

    Supported schedules:
    - 'linear': Standard CFM (alpha=1-t, sigma=t) - equivalent to I-CFM
    - 'gvp': Variance-preserving (alpha=cos, sigma=sin)

    Attributes:
        schedule_name: Name of the interpolation schedule.
        schedule_fn: Schedule function that returns coefficients.
        velocity_scale: Normalization factor for velocity target.
    """

    def __init__(self, schedule: str = "gvp", normalize_loss: bool = True, **kwargs):
        """Initialize Stochastic Interpolant coupling.

        Args:
            schedule: Interpolation schedule name ('linear' or 'gvp').
            normalize_loss: If True, normalize velocity target so losses are
                           comparable to I-CFM. Default True.
            **kwargs: Ignored (for API compatibility).
        """
        self.schedule_name = schedule
        self.schedule_fn: ScheduleFunction = get_schedule(schedule)
        self.normalize_loss = normalize_loss
        self.velocity_scale = _VELOCITY_SCALE.get(schedule, 1.0) if normalize_loss else 1.0

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t, x_t, u_t for velocity matching.

        The velocity target u_t is the derivative of the interpolation,
        NOT x1 - x0. This is the key difference from I-CFM.

        Args:
            x0: Source samples (noise) [B, D].
            x1: Target samples (data) [B, D].

        Returns:
            t: Uniformly sampled timesteps [B].
            x_t: Interpolated samples [B, D].
            u_t: Target velocity [B, D].
        """
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        alpha_t, sigma_t, alpha_dot, sigma_dot = self.schedule_fn(t)

        # Expand for broadcasting: [B] -> [B, 1]
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
        alpha_dot = alpha_dot.unsqueeze(-1)
        sigma_dot = sigma_dot.unsqueeze(-1)

        x_t = alpha_t * x0 + sigma_t * x1
        u_t = alpha_dot * x0 + sigma_dot * x1

        if self.velocity_scale != 1.0:
            u_t = u_t / self.velocity_scale

        return t, x_t, u_t

    def __repr__(self) -> str:
        return f"StochasticInterpolantCoupling(schedule='{self.schedule_name}')"


__all__ = ["StochasticInterpolantCoupling"]
