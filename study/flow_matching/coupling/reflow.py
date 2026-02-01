"""Reflow coupling for rectified flow training.

Uses pre-generated (x0, x1) pairs from teacher ODE integration.
Unlike ICFM/OT-CFM, reflow ignores the data batch and samples
from its stored pairs, which creates deterministic coupling
between noise and data endpoints.

This deterministic coupling produces straighter ODE trajectories,
enabling faster sampling with fewer integration steps.
"""

import logging
from typing import Tuple, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ReflowCoupling:
    """Coupling using pre-generated reflow pairs.

    Stores synthetic (x0, x1) pairs and samples from them during training.
    Ignores the data batch passed to sample() - uses its own stored pairs.

    The ICFM formulation applies to the stored pairs:
    - x_t = (1-t)*x0_pair + t*x1_pair (linear interpolation)
    - u_t = x1_pair - x0_pair (constant velocity)

    Attributes:
        x0_all: All noise samples [N, D] on CPU.
        x1_all: All ODE endpoints [N, D] on CPU.
        n_pairs: Number of stored pairs.
        current_batch_size: Size of batches returned by sample().

    Example:
        >>> # Generate pairs from teacher
        >>> x0, x1 = generator.generate_pairs(10000, 'cuda')
        >>> coupling = ReflowCoupling(pair_tensors=(x0, x1))
        >>>
        >>> # During training (ignores inputs)
        >>> t, x_t, u_t = coupling.sample(data_noise, data_batch)
    """

    def __init__(
        self,
        pair_tensors: Tuple[Tensor, Tensor],
        batch_size: Optional[int] = None,
        sigma: float = 0.0,
    ):
        """Initialize reflow coupling with pre-generated pairs.

        Args:
            pair_tensors: Tuple of (x0, x1) tensors.
                - x0: Noise samples [N, D]
                - x1: ODE endpoints [N, D]
            batch_size: Number of pairs to sample per call (default: all pairs).
            sigma: Noise level for interpolation (unused, for interface compatibility).

        Note: Pairs are stored on CPU and moved to GPU during sample().
        """
        x0_all, x1_all = pair_tensors

        # Ensure on CPU for storage
        self.x0_all = x0_all.cpu()
        self.x1_all = x1_all.cpu()
        self.n_pairs = x0_all.shape[0]
        self.dim = x0_all.shape[1]
        self.sigma = sigma  # Unused but kept for interface compatibility

        # Batch size defaults to total pairs (one epoch = one pass)
        self.current_batch_size = batch_size or self.n_pairs

        # Shuffle indices for random sampling
        self._indices = torch.randperm(self.n_pairs)
        self._current_idx = 0

        logger.info(
            f"ReflowCoupling initialized with {self.n_pairs} pairs, "
            f"dim={self.dim}, batch_size={self.current_batch_size}"
        )

    def reset(self) -> None:
        """Reshuffle pairs for new epoch.

        Call at the start of each epoch to ensure random sampling.
        """
        self._indices = torch.randperm(self.n_pairs)
        self._current_idx = 0
        logger.debug("ReflowCoupling: shuffled pairs for new epoch")

    def sample(
        self,
        x0: Optional[Tensor],
        x1: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample t, x_t, u_t from stored pairs.

        IMPORTANT: This ignores the x0, x1 inputs and uses stored pairs.
        The inputs are only used to infer device and batch size.

        Args:
            x0: Ignored (noise from data loader). Can be None.
            x1: Ignored (data from data loader). Can be None.

        Returns:
            t: Uniformly sampled timesteps [B]
            x_t: Interpolated samples [B, D]
            u_t: Target velocity [B, D]
        """
        # Determine batch size and device
        if x1 is not None:
            batch_size = x1.shape[0]
            device = x1.device
        elif x0 is not None:
            batch_size = x0.shape[0]
            device = x0.device
        else:
            batch_size = self.current_batch_size
            device = torch.device("cpu")

        # Get random pairs from stored data
        if self._current_idx + batch_size > self.n_pairs:
            # Wrap around and reshuffle if needed
            self.reset()

        idx = self._indices[self._current_idx : self._current_idx + batch_size]
        self._current_idx += batch_size

        # Get pairs and move to device
        x0_pair = self.x0_all[idx].to(device)
        x1_pair = self.x1_all[idx].to(device)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Interpolate (ICFM formulation)
        t_unsqueeze = t.unsqueeze(-1)
        x_t = (1 - t_unsqueeze) * x0_pair + t_unsqueeze * x1_pair

        # Target velocity (constant)
        u_t = x1_pair - x0_pair

        return t, x_t, u_t
