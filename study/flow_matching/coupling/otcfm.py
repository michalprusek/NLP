"""Optimal Transport Conditional Flow Matching coupling.

OT-CFM uses mini-batch Sinkhorn to pair x0, x1 optimally, producing
straighter flow paths than I-CFM. This can improve sampling quality
and convergence.
"""

from typing import Tuple

import torch
from torchcfm.optimal_transport import OTPlanSampler


class OTCFMCoupling:
    """Optimal Transport Conditional Flow Matching coupling.

    Uses mini-batch Sinkhorn to pair x0, x1 optimally,
    producing straighter flow paths than I-CFM.

    The OT pairing reorders x0 to minimize transport cost,
    which results in straighter, more efficient flow paths.

    Attributes:
        sigma: Noise level for interpolation.
        reg: Sinkhorn regularization (higher = more stable in high-D).
        normalize_cost: Whether to normalize cost matrix (prevents overflow).
    """

    def __init__(
        self,
        sigma: float = 0.0,
        reg: float = 0.5,
        normalize_cost: bool = True,
    ):
        """Initialize OT-CFM coupling.

        Args:
            sigma: Noise level for interpolation (default 0.0 for deterministic).
            reg: Sinkhorn regularization (higher = more stable in high-D).
                 For 1024D SONAR embeddings, reg >= 0.5 recommended.
            normalize_cost: Whether to normalize cost matrix (prevents overflow).
                           Recommended True for high-dimensional data.
        """
        self.sigma = sigma
        self.reg = reg
        self.normalize_cost = normalize_cost

        # Create OT plan sampler for mini-batch optimal transport
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
            x0: Source samples (noise) [B, D]
            x1: Target samples (data) [B, D]

        Returns:
            t: Uniformly sampled timesteps [B]
            x_t: Interpolated samples (with OT pairing) [B, D]
            u_t: Target velocity (with OT pairing) [B, D]
        """
        # Sample OT-paired indices
        # The OT sampler returns reordered x0 that optimally matches x1
        x0_ot, x1_ot = self.ot_sampler.sample_plan(x0, x1)

        # Sample time uniformly
        t = torch.rand(x1_ot.shape[0], device=x1_ot.device)

        # Interpolate with OT-paired samples
        t_unsqueeze = t.unsqueeze(-1)
        x_t = (1 - t_unsqueeze) * x0_ot + t_unsqueeze * x1_ot

        # Target velocity with OT pairing
        u_t = x1_ot - x0_ot

        return t, x_t, u_t
