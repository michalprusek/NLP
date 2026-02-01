"""Interpolation schedules for flow matching.

Provides schedule functions for stochastic interpolants and other flow methods.
Each schedule returns (alpha_t, sigma_t, alpha_dot, sigma_dot) where:
- alpha_t: coefficient for x0 (source/noise)
- sigma_t: coefficient for x1 (target/data)
- alpha_dot: d(alpha_t)/dt
- sigma_dot: d(sigma_t)/dt

The interpolation is: x_t = alpha_t * x0 + sigma_t * x1
The velocity target is: u_t = alpha_dot * x0 + sigma_dot * x1

Available schedules:
- linear: Standard CFM schedule (alpha=1-t, sigma=t)
- gvp: Variance-preserving trigonometric schedule (alpha=cos, sigma=sin)
"""

import math
from typing import Callable, Tuple

import torch


def linear_schedule(
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear interpolation schedule (standard CFM).

    x_t = (1-t)*x0 + t*x1
    u_t = -x0 + x1 = x1 - x0

    This is equivalent to I-CFM when used with stochastic interpolants.

    Args:
        t: Timesteps tensor of shape [B] in [0, 1].

    Returns:
        Tuple of (alpha_t, sigma_t, alpha_dot, sigma_dot), each shape [B].
    """
    alpha_t = 1.0 - t
    sigma_t = t
    # Derivatives are constants, broadcast to match t shape
    alpha_dot = torch.full_like(t, -1.0)
    sigma_dot = torch.full_like(t, 1.0)
    return alpha_t, sigma_t, alpha_dot, sigma_dot


def gvp_schedule(
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variance-preserving trigonometric schedule (GVP).

    x_t = cos(pi*t/2)*x0 + sin(pi*t/2)*x1
    u_t = -pi/2*sin(pi*t/2)*x0 + pi/2*cos(pi*t/2)*x1

    This schedule is variance-preserving: alpha_t^2 + sigma_t^2 = 1.
    It has been shown to improve performance on ImageNet when used with
    stochastic interpolants (Albergo et al., 2023).

    The schedule transitions smoothly:
    - t=0: alpha=1, sigma=0 (starts at x0)
    - t=1: alpha=0, sigma=1 (ends at x1)

    Args:
        t: Timesteps tensor of shape [B] in [0, 1].

    Returns:
        Tuple of (alpha_t, sigma_t, alpha_dot, sigma_dot), each shape [B].
    """
    # Scale t to [0, pi/2]
    scaled_t = math.pi * t / 2.0

    # Interpolation coefficients
    alpha_t = torch.cos(scaled_t)
    sigma_t = torch.sin(scaled_t)

    # Derivatives: d/dt[cos(pi*t/2)] = -pi/2 * sin(pi*t/2)
    #              d/dt[sin(pi*t/2)] = pi/2 * cos(pi*t/2)
    alpha_dot = -math.pi / 2.0 * torch.sin(scaled_t)
    sigma_dot = math.pi / 2.0 * torch.cos(scaled_t)

    return alpha_t, sigma_t, alpha_dot, sigma_dot


# Type alias for schedule functions
ScheduleFunction = Callable[
    [torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]

# Registry of available schedules
_SCHEDULES: dict[str, ScheduleFunction] = {
    "linear": linear_schedule,
    "gvp": gvp_schedule,
}


def get_schedule(name: str) -> ScheduleFunction:
    """Get schedule function by name.

    Args:
        name: Schedule name ('linear' or 'gvp').

    Returns:
        Schedule function that takes t and returns (alpha_t, sigma_t, alpha_dot, sigma_dot).

    Raises:
        ValueError: If schedule name is unknown.

    Example:
        >>> schedule_fn = get_schedule('gvp')
        >>> t = torch.tensor([0.0, 0.5, 1.0])
        >>> alpha_t, sigma_t, alpha_dot, sigma_dot = schedule_fn(t)
    """
    if name not in _SCHEDULES:
        available = ", ".join(_SCHEDULES.keys())
        raise ValueError(f"Unknown schedule: {name}. Available: {available}")
    return _SCHEDULES[name]


__all__ = ["linear_schedule", "gvp_schedule", "get_schedule", "ScheduleFunction"]
