"""Stochastic Interpolant coupling for flow matching.

Stochastic Interpolants (Albergo et al., 2023) generalize flow matching by
allowing non-linear interpolation schedules. Unlike I-CFM which uses constant
velocity x1-x0, SI computes velocity as the derivative of the interpolation.

Key difference from I-CFM:
- I-CFM: u_t = x1 - x0 (constant velocity)
- SI: u_t = alpha_dot*x0 + sigma_dot*x1 (time-varying velocity)

This allows for variance-preserving schedules (GVP) that can improve
sample quality, especially for high-dimensional data.
"""

import torch

from study.flow_matching.schedules import get_schedule, ScheduleFunction


class StochasticInterpolantCoupling:
    """Stochastic Interpolant coupling with configurable schedule.

    Implements stochastic interpolants where:
    - x_t = alpha_t * x0 + sigma_t * x1 (interpolation)
    - u_t = alpha_dot * x0 + sigma_dot * x1 (velocity target)

    The velocity target is the time derivative of the interpolation,
    NOT the constant x1-x0 used in I-CFM.

    Supported schedules:
    - 'linear': Standard CFM (alpha=1-t, sigma=t) - equivalent to I-CFM
    - 'gvp': Variance-preserving (alpha=cos, sigma=sin) - better for some tasks

    Attributes:
        schedule_name: Name of the interpolation schedule.
        schedule_fn: Schedule function that returns coefficients.

    Example:
        >>> coupling = StochasticInterpolantCoupling(schedule='gvp')
        >>> t, x_t, u_t = coupling.sample(x0, x1)
    """

    def __init__(self, schedule: str = "gvp"):
        """Initialize Stochastic Interpolant coupling.

        Args:
            schedule: Interpolation schedule name ('linear' or 'gvp').
                     Default is 'gvp' (variance-preserving).
        """
        self.schedule_name = schedule
        self.schedule_fn: ScheduleFunction = get_schedule(schedule)

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t, x_t, u_t for velocity matching.

        Stochastic Interpolant formulation:
        - x_t = alpha_t * x0 + sigma_t * x1
        - u_t = alpha_dot * x0 + sigma_dot * x1

        CRITICAL: The velocity target u_t is the derivative of the interpolation,
        NOT x1 - x0. This is the key difference from I-CFM.

        For GVP schedule:
        - alpha_t = cos(pi*t/2), sigma_t = sin(pi*t/2)
        - alpha_dot = -pi/2*sin(pi*t/2), sigma_dot = pi/2*cos(pi*t/2)

        Args:
            x0: Source samples (noise) of shape [B, D].
            x1: Target samples (data) of shape [B, D].

        Returns:
            Tuple of (t, x_t, u_t):
            - t: Uniformly sampled timesteps of shape [B].
            - x_t: Interpolated samples of shape [B, D].
            - u_t: Target velocity of shape [B, D].
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)

        # Get schedule coefficients
        alpha_t, sigma_t, alpha_dot, sigma_dot = self.schedule_fn(t)

        # Reshape for broadcasting: [B] -> [B, 1]
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
        alpha_dot = alpha_dot.unsqueeze(-1)
        sigma_dot = sigma_dot.unsqueeze(-1)

        # Interpolate: x_t = alpha_t * x0 + sigma_t * x1
        x_t = alpha_t * x0 + sigma_t * x1

        # Velocity target: u_t = alpha_dot * x0 + sigma_dot * x1
        # NOTE: This is NOT x1 - x0!
        u_t = alpha_dot * x0 + sigma_dot * x1

        return t, x_t, u_t

    def __repr__(self) -> str:
        return f"StochasticInterpolantCoupling(schedule='{self.schedule_name}')"


__all__ = ["StochasticInterpolantCoupling"]
