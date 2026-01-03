"""BoTorch qLogExpectedImprovement for latent space optimization.

This module provides BoTorch-based acquisition function optimization
for the LIPO pipeline. Uses qLogExpectedImprovement which is
numerically stable even when improvement values are extremely small.

Pipeline: z (16D VAE latent) -> Adapter -> z_gp (10D) -> Kumaraswamy Warp -> GP -> qLogEI

After optimization: z_opt (16D) -> VAE decoder -> embedding (768D) -> Vec2Text

Adapted from generation/invbo_decoder/botorch_acq.py for lipo.
"""

import logging
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf

logger = logging.getLogger(__name__)


class CompositeLogEI(AcquisitionFunction):
    """qLogEI that operates on 16D VAE latent space.

    The GP model expects 16D VAE latent input and applies adapter internally:
        z (16D) -> GP (with adapter) -> z_gp (10D) -> Kumaraswamy Warp -> kernel -> qLogEI

    This enables gradient-based optimization in 16D VAE latent space
    with gradients flowing through the adapter and warping.
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        sampler: Optional[torch.Tensor] = None,
    ):
        """Initialize composite acquisition function.

        Args:
            model: GP model that accepts 16D VAE latents and applies adapter internally
            best_f: Best observed negated error rate (max of -error_rates for BoTorch maximization)
            sampler: Optional MC sampler for qLogEI
        """
        super().__init__(model=model)
        self.best_f = best_f
        # qLogEI for numerically stable expected improvement
        self._base_acq = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate acquisition on VAE latent points.

        Args:
            X: VAE latent points with shape (batch, q, d) where d=latent_dim
               For single-point acquisition, shape is (batch, 1, latent_dim)

        Returns:
            Acquisition values with shape (batch,)
        """
        # Pass directly to qLogEI - GP applies adapter and warping internally
        return self._base_acq(X)


class LatentSpaceAcquisition:
    """Optimizes qLogEI in VAE latent space (16D).

    Uses BoTorch's optimize_acqf with multi-start L-BFGS-B optimization.
    This provides proper gradient flow through:
        z (16D) -> GP (with trainable adapter) -> z_gp (10D) -> Kumaraswamy Warp -> kernel -> LogEI

    Key advantages over scipy L-BFGS-B:
    1. Gradient-based refinement with proper autograd
    2. Multi-start with raw_samples avoids poor local optima
    3. Sophisticated initialization from BoTorch
    """

    def __init__(
        self,
        gp_model: Model,
        bounds: torch.Tensor,
        device: torch.device,
    ):
        """Initialize latent space acquisition optimizer.

        Args:
            gp_model: Trained GP model that accepts 16D VAE latents
            bounds: VAE latent space bounds, shape (2, latent_dim)
                    bounds[0] = lower bounds, bounds[1] = upper bounds
            device: Torch device
        """
        self.gp_model = gp_model
        self.bounds = bounds.to(device)
        self.device = device

    def optimize(
        self,
        best_f: float,
        num_restarts: int = 64,
        raw_samples: int = 1024,
        options: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal VAE latent z (16D) maximizing qLogEI.

        Uses BoTorch's optimize_acqf with:
        1. raw_samples random starting points
        2. num_restarts L-BFGS-B optimizations from best raw samples
        3. Returns best result across all restarts

        Args:
            best_f: Best observed negated error rate (max of -error_rates for BoTorch maximization)
            num_restarts: Number of L-BFGS-B restarts
            raw_samples: Number of initial random samples for seeding
            options: Additional options for L-BFGS-B
            seed: Optional random seed for reproducibility

        Returns:
            (candidate, acq_value) tuple where:
            - candidate: Optimal VAE latent tensor, shape (1, latent_dim)
            - acq_value: LogEI value at optimal point
        """
        if options is None:
            options = {"maxiter": 200, "batch_limit": 5}

        # Set seed for reproducible multi-start optimization
        if seed is not None:
            torch.manual_seed(seed)

        # Create composite acquisition function
        acq_fn = CompositeLogEI(
            model=self.gp_model,
            best_f=best_f,
        )

        # Ensure models are in eval mode (including likelihood for GP)
        self.gp_model.eval()
        if hasattr(self.gp_model, 'likelihood'):
            self.gp_model.likelihood.eval()

        # Optimize using BoTorch's optimize_acqf
        # Suppress RuntimeWarnings from L-BFGS-B (common in BO) but log if debug enabled
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.filterwarnings("always", category=RuntimeWarning)
            candidate, acq_value = optimize_acqf(
                acq_function=acq_fn,
                bounds=self.bounds,
                q=1,  # Single point acquisition
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                retry_on_optimization_warning=False,
            )

        # Log warnings at debug level instead of silently suppressing
        if caught_warnings:
            logger.debug(
                f"Optimization completed with {len(caught_warnings)} warnings "
                f"(common in L-BFGS-B optimization)"
            )

        return candidate, acq_value


def get_latent_bounds(
    encoder: nn.Module,
    X_train: torch.Tensor,
    X_min: torch.Tensor,
    X_max: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute bounds for 16D VAE latent space from training data.

    X_train is already stored as 16D VAE latents. We just need to
    normalize and compute bounds with margin for exploration.

    Args:
        encoder: VAEWithAdapter (unused, kept for API compatibility)
        X_train: Training VAE latents (N, latent_dim)
        X_min: Min values for normalization (latent_dim,)
        X_max: Max values for normalization (latent_dim,)
        margin: Fraction to expand bounds (0.2 = 20% each side)

    Returns:
        Bounds tensor, shape (2, latent_dim) for VAE latent space
    """
    with torch.no_grad():
        # Normalize training data (already 16D VAE latents)
        denom = X_max - X_min
        denom[denom == 0] = 1.0
        latents_norm = (X_train - X_min) / denom

        # Compute bounds from normalized latents
        z_min = latents_norm.min(dim=0)[0]
        z_max = latents_norm.max(dim=0)[0]

        # Expand bounds by margin
        z_range = z_max - z_min
        z_min = z_min - margin * z_range
        z_max = z_max + margin * z_range

    return torch.stack([z_min, z_max], dim=0)
