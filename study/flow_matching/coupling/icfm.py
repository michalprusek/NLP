"""Independent Conditional Flow Matching coupling.

ICFM samples x0, x1 independently (random pairing). This is the baseline
CFM method that produces curved flow paths.
"""

from typing import Tuple

import torch


class ICFMCoupling:
    """Independent Conditional Flow Matching coupling.

    Samples x0, x1 independently (random pairing).
    This is the baseline CFM method.

    The ICFM formulation:
    - x_t = (1-t)*x0 + t*x1 (linear interpolation)
    - u_t = x1 - x0 (constant velocity)

    Attributes:
        sigma: Noise level for interpolation (unused in deterministic case).
    """

    def __init__(self, sigma: float = 0.0):
        """Initialize I-CFM coupling.

        Args:
            sigma: Noise level for interpolation (default 0.0 for deterministic).
        """
        self.sigma = sigma

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t, x_t, u_t for velocity matching.

        ICFM formulation: x_t = (1-t)*x0 + t*x1, u_t = x1 - x0

        Args:
            x0: Source samples (noise) [B, D]
            x1: Target samples (data) [B, D]

        Returns:
            t: Uniformly sampled timesteps [B]
            x_t: Interpolated samples [B, D]
            u_t: Target velocity [B, D]
        """
        # Sample time uniformly
        t = torch.rand(x1.shape[0], device=x1.device)

        # Interpolate
        t_unsqueeze = t.unsqueeze(-1)
        x_t = (1 - t_unsqueeze) * x0 + t_unsqueeze * x1

        # Target velocity
        u_t = x1 - x0

        return t, x_t, u_t
