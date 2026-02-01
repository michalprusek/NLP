"""Independent Conditional Flow Matching coupling.

ICFM samples x0, x1 independently (random pairing). This is the baseline
CFM method that produces curved flow paths.
"""

from typing import Tuple

import torch


def linear_interpolate(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation and constant velocity for flow matching.

    Args:
        x0: Source samples [B, D].
        x1: Target samples [B, D].
        t: Timesteps [B].

    Returns:
        x_t: Interpolated samples [B, D].
        u_t: Target velocity [B, D].
    """
    t_expanded = t.unsqueeze(-1)
    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    u_t = x1 - x0
    return x_t, u_t


class ICFMCoupling:
    """Independent Conditional Flow Matching coupling.

    Samples x0, x1 independently (random pairing).
    This is the baseline CFM method.

    The ICFM formulation:
    - x_t = (1-t)*x0 + t*x1 (linear interpolation)
    - u_t = x1 - x0 (constant velocity)
    """

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t, x_t, u_t for velocity matching.

        Args:
            x0: Source samples (noise) [B, D].
            x1: Target samples (data) [B, D].

        Returns:
            t: Uniformly sampled timesteps [B].
            x_t: Interpolated samples [B, D].
            u_t: Target velocity [B, D].
        """
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        x_t, u_t = linear_interpolate(x0, x1, t)
        return t, x_t, u_t
