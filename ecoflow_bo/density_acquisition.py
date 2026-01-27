"""
Density-Aware Acquisition Function for Bayesian Optimization.

Combines UCB (Upper Confidence Bound) with prior density weighting
to prefer candidates that lie on the learned manifold.

α_new(z) = α_UCB(z) · p_prior(z)
         = (μ(z) + β·σ(z)) + λ · log p(z)

This prevents GP from proposing candidates far from the training
distribution where the decoder might hallucinate.
"""

import torch
from typing import Optional, Tuple, List
import numpy as np

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from .latent_gp import CoarseToFineGP
from .config import AcquisitionConfig


class DensityAwareAcquisition:
    """
    Density-aware UCB acquisition function.

    α(z) = UCB(z) + density_weight · log p_prior(z)

    Where p_prior(z) = N(0, I), so log p_prior(z) ∝ -0.5 ||z||²

    The density term encourages staying close to the prior,
    which the encoder was regularized to match via KL divergence.
    """

    def __init__(self, config: Optional[AcquisitionConfig] = None):
        if config is None:
            config = AcquisitionConfig()

        self.config = config
        self.beta = config.beta
        self.density_weight = config.density_weight
        self.n_candidates = config.n_candidates
        self.n_restarts = config.n_restarts

    def log_prior_density(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior density under N(0, I).

        log p(z) = -0.5 * d * log(2π) - 0.5 * ||z||²

        We drop the constant and return -0.5 * ||z||²
        """
        return -0.5 * z.pow(2).sum(dim=-1)

    def compute_ucb(
        self, gp: CoarseToFineGP, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute UCB value: μ(z) + β·σ(z)
        """
        mean, var = gp.predict(z)
        std = var.sqrt()
        return mean + self.beta * std

    def compute_acquisition(
        self,
        gp: CoarseToFineGP,
        z: torch.Tensor,
        active_dims: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute density-aware acquisition value.

        Args:
            gp: Fitted GP surrogate
            z: Candidate points [N, latent_dim]
            active_dims: Which dims are active (for density calculation)

        Returns:
            acquisition: Acquisition values [N]
        """
        # UCB
        ucb = self.compute_ucb(gp, z)

        # Prior density (only on active dims)
        if active_dims is None:
            active_dims = gp.active_dims

        z_active = z[:, active_dims]
        log_density = self.log_prior_density(z_active)

        # Combined acquisition
        acquisition = ucb + self.density_weight * log_density

        return acquisition

    def generate_candidates(
        self,
        gp: CoarseToFineGP,
        z_best: torch.Tensor,
        n_candidates: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate candidate points for evaluation.

        Strategy:
        1. Random samples from prior (exploration)
        2. Local perturbations around best point (exploitation)
        3. Latin hypercube samples in search region

        Args:
            gp: Fitted GP
            z_best: Current best latent [latent_dim]
            n_candidates: Number of candidates to generate

        Returns:
            candidates: [n_candidates, latent_dim]
        """
        if n_candidates is None:
            n_candidates = self.n_candidates

        device = z_best.device
        dtype = z_best.dtype
        latent_dim = z_best.shape[0]

        candidates = []

        # 1. Random samples from prior (40% of candidates)
        n_random = int(0.4 * n_candidates)
        z_random = torch.randn(n_random, latent_dim, device=device, dtype=dtype)
        # Zero out inactive dimensions
        inactive_dims = set(range(latent_dim)) - set(gp.active_dims)
        for dim in inactive_dims:
            z_random[:, dim] = 0.0
        candidates.append(z_random)

        # 2. Local perturbations around best (40% of candidates)
        n_local = int(0.4 * n_candidates)
        lower, upper = gp.get_search_bounds(z_best)
        z_local = z_best.unsqueeze(0).repeat(n_local, 1)
        # Add Gaussian noise scaled by stage
        noise_scale = 0.3 * (1 + gp.current_stage)
        noise = torch.randn_like(z_local) * noise_scale
        # Only perturb active dims
        for dim in inactive_dims:
            noise[:, dim] = 0.0
        z_local = z_local + noise
        # Clip to bounds
        z_local = torch.clamp(z_local, lower, upper)
        candidates.append(z_local)

        # 3. Latin hypercube in search bounds (20% of candidates)
        n_lhs = n_candidates - n_random - n_local
        z_lhs = self._latin_hypercube(n_lhs, latent_dim, device, dtype)
        # Scale to search bounds
        z_lhs = lower + z_lhs * (upper - lower)
        candidates.append(z_lhs)

        return torch.cat(candidates, dim=0)

    def _latin_hypercube(
        self, n: int, d: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate Latin hypercube samples in [0, 1]^d."""
        samples = torch.zeros(n, d, device=device, dtype=dtype)
        for i in range(d):
            perm = torch.randperm(n, device=device)
            samples[:, i] = (perm + torch.rand(n, device=device)) / n
        return samples

    def select_best_candidates(
        self,
        gp: CoarseToFineGP,
        candidates: torch.Tensor,
        n_select: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best candidates according to acquisition function.

        Args:
            gp: Fitted GP
            candidates: All candidates [N, latent_dim]
            n_select: Number of candidates to select

        Returns:
            selected_z: Best candidates [n_select, latent_dim]
            acq_values: Acquisition values [n_select]
        """
        acq_values = self.compute_acquisition(gp, candidates)

        # Sort by acquisition value (descending)
        sorted_indices = acq_values.argsort(descending=True)
        selected_indices = sorted_indices[:n_select]

        selected_z = candidates[selected_indices]
        selected_acq = acq_values[selected_indices]

        return selected_z, selected_acq

    def optimize(
        self,
        gp: CoarseToFineGP,
        z_best: torch.Tensor,
        n_select: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full acquisition optimization pipeline.

        Args:
            gp: Fitted GP
            z_best: Current best latent
            n_select: Number of points to return

        Returns:
            best_z: Best candidates [n_select, latent_dim]
            best_acq: Acquisition values [n_select]
        """
        # Generate candidates
        candidates = self.generate_candidates(gp, z_best)

        # Select best
        best_z, best_acq = self.select_best_candidates(
            gp, candidates, n_select=n_select
        )

        return best_z, best_acq


class BatchDensityAwareAcquisition(DensityAwareAcquisition):
    """
    Batch acquisition for parallel evaluation.

    Uses q-UCB (batch UCB) with density weighting and diversity bonus
    to select a batch of candidates for parallel evaluation.
    """

    def __init__(
        self,
        config: Optional[AcquisitionConfig] = None,
        diversity_weight: float = 0.1,
    ):
        super().__init__(config)
        self.diversity_weight = diversity_weight

    def select_diverse_batch(
        self,
        gp: CoarseToFineGP,
        candidates: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select diverse batch using greedy selection with diversity bonus.

        Args:
            gp: Fitted GP
            candidates: All candidates [N, latent_dim]
            batch_size: Batch size

        Returns:
            selected_z: Selected candidates [batch_size, latent_dim]
            acq_values: Acquisition values [batch_size]
        """
        N = candidates.shape[0]
        device = candidates.device

        # Compute base acquisition values
        acq_values = self.compute_acquisition(gp, candidates)

        selected_indices = []
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)

        for _ in range(batch_size):
            # Compute diversity penalty for remaining candidates
            if len(selected_indices) > 0:
                selected_z = candidates[selected_indices]
                remaining_candidates = candidates[remaining_mask]

                # Pairwise distances to selected
                dists = torch.cdist(
                    remaining_candidates, selected_z
                ).min(dim=1).values

                # Diversity bonus (encourage far from selected)
                diversity_bonus = self.diversity_weight * dists

                # Adjusted acquisition
                adjusted_acq = acq_values[remaining_mask] + diversity_bonus
            else:
                adjusted_acq = acq_values[remaining_mask]

            # Select best remaining
            best_local = adjusted_acq.argmax()

            # Map back to global index
            remaining_indices = remaining_mask.nonzero().squeeze(-1)
            best_global = remaining_indices[best_local].item()

            selected_indices.append(best_global)
            remaining_mask[best_global] = False

        selected_indices = torch.tensor(selected_indices, device=device)
        selected_z = candidates[selected_indices]
        selected_acq = acq_values[selected_indices]

        return selected_z, selected_acq
