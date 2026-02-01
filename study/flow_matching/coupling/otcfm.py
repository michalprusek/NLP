"""Optimal Transport Conditional Flow Matching coupling.

OT-CFM uses mini-batch Sinkhorn to pair x0, x1 optimally, producing
straighter flow paths than I-CFM. This can improve sampling quality
and convergence.
"""

from typing import Tuple

import torch
from torchcfm.optimal_transport import OTPlanSampler

from study.flow_matching.coupling.icfm import linear_interpolate


class OTCFMCoupling:
    """Optimal Transport Conditional Flow Matching coupling.

    Uses mini-batch Sinkhorn to pair x0, x1 optimally,
    producing straighter flow paths than I-CFM.

    The OT pairing reorders x0 to minimize transport cost,
    which results in straighter, more efficient flow paths.

    Attributes:
        reg: Sinkhorn regularization (higher = more stable in high-D).
        normalize_cost: Whether to normalize cost matrix (prevents overflow).
    """

    def __init__(
        self,
        reg: float = 0.5,
        normalize_cost: bool = True,
        **kwargs,
    ):
        """Initialize OT-CFM coupling.

        Args:
            reg: Sinkhorn regularization (higher = more stable in high-D).
                 For 1024D SONAR embeddings, reg >= 0.5 recommended.
            normalize_cost: Whether to normalize cost matrix (prevents overflow).
                           Recommended True for high-dimensional data.
            **kwargs: Ignored (for API compatibility).
        """
        self.reg = reg
        self.normalize_cost = normalize_cost
        self.ot_sampler = OTPlanSampler(
            method="exact",
            reg=reg,
            normalize_cost=normalize_cost,
        )

    def sample(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample t, x_t, u_t with OT-coupled pairs.

        Uses mini-batch Sinkhorn to compute optimal transport coupling,
        which reorders x0 to minimize transport cost.

        Args:
            x0: Source samples (noise) [B, D].
            x1: Target samples (data) [B, D].

        Returns:
            t: Uniformly sampled timesteps [B].
            x_t: Interpolated samples (with OT pairing) [B, D].
            u_t: Target velocity (with OT pairing) [B, D].
        """
        x0_ot, x1_ot = self.ot_sampler.sample_plan(x0, x1)
        t = torch.rand(x0_ot.shape[0], device=x0_ot.device, dtype=x0_ot.dtype)
        x_t, u_t = linear_interpolate(x0_ot, x1_ot, t)
        return t, x_t, u_t
