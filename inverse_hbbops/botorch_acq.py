"""BoTorch qLogExpectedImprovement for latent space optimization.

This module provides BoTorch-based acquisition function optimization
for the Inverse HbBoPs pipeline. Uses qLogExpectedImprovement which is
numerically stable even when improvement values are extremely small.

Pipeline: z (10D latent) -> VAE decoder -> embedding (768D) -> GP -> qLogEI

Adapted from generation/invbo_decoder/botorch_acq.py for inverse_hbbops.
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
    """qLogEI that works through VAE decoder transformation.

    Computes LogEI(vae.decode(z)) for latent z, enabling gradient-based
    optimization in the low-dimensional latent space while evaluating
    improvement in the embedding space where the GP operates.

    This is critical for InvBO because:
    1. We optimize in 10D latent space (tractable)
    2. GP operates in 768D embedding space (high-quality predictions)
    3. VAE decoder bridges the two spaces with differentiable transform
    """

    def __init__(
        self,
        model: Model,
        vae: nn.Module,
        best_f: float,
        sampler: Optional[torch.Tensor] = None,
    ):
        """Initialize composite acquisition function.

        Args:
            model: GP model (must implement posterior())
            vae: VAE with decode() method (10D -> 768D)
            best_f: Best observed objective value (error rate to minimize)
            sampler: Optional MC sampler for qLogEI
        """
        super().__init__(model=model)
        self.vae = vae
        self.best_f = best_f
        # qLogEI for numerically stable expected improvement
        self._base_acq = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate acquisition on latent points.

        Args:
            X: Latent points with shape (batch, q, d) where d=10 (latent dim)
               For single-point acquisition, shape is (batch, 1, 10)

        Returns:
            Acquisition values with shape (batch,)
        """
        # Handle shape: (batch, q, d) or (q, d) or (d,)
        if X.dim() == 1:
            X = X.unsqueeze(0).unsqueeze(0)  # (d,) -> (1, 1, d)
        elif X.dim() == 2:
            X = X.unsqueeze(0)  # (q, d) -> (1, q, d)

        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        d = X.shape[-1]

        # Decode latent to embeddings via VAE
        # Flatten for decoder: (batch, q, d) -> (batch*q, d)
        X_flat = X.reshape(-1, d)

        # Ensure VAE is in eval mode but allow gradients
        self.vae.eval()
        embeddings = self.vae.decode(X_flat)  # (batch*q, 768)

        # Reshape for qLogEI: (batch*q, 768) -> (batch, q, 768)
        embedding_dim = embeddings.shape[-1]
        embeddings = embeddings.view(*batch_shape, q, embedding_dim)

        # Evaluate qLogEI on embeddings
        return self._base_acq(embeddings)


class LatentSpaceAcquisition:
    """Optimizes qLogEI in VAE latent space.

    Uses BoTorch's optimize_acqf with multi-start L-BFGS-B optimization.
    This provides proper gradient flow through:
        latent z -> VAE decoder -> embedding -> GP posterior -> LogEI

    Key advantages over scipy L-BFGS-B:
    1. Gradient-based refinement with proper autograd
    2. Multi-start with raw_samples avoids poor local optima
    3. Sophisticated initialization from BoTorch
    """

    def __init__(
        self,
        gp_model: Model,
        vae: nn.Module,
        bounds: torch.Tensor,
        device: torch.device,
    ):
        """Initialize latent space acquisition optimizer.

        Args:
            gp_model: Trained GP model (GPWithEI.gp_model)
            vae: Trained InstructionVAE
            bounds: Latent space bounds, shape (2, latent_dim)
                    bounds[0] = lower bounds, bounds[1] = upper bounds
            device: Torch device
        """
        self.gp_model = gp_model
        self.vae = vae
        self.bounds = bounds.to(device)
        self.device = device

    def optimize(
        self,
        best_f: float,
        num_restarts: int = 20,
        raw_samples: int = 512,
        options: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal latent z maximizing qLogEI.

        Uses BoTorch's optimize_acqf with:
        1. raw_samples random starting points
        2. num_restarts L-BFGS-B optimizations from best raw samples
        3. Returns best result across all restarts

        Args:
            best_f: Best observed objective value
            num_restarts: Number of L-BFGS-B restarts
            raw_samples: Number of initial random samples for seeding
            options: Additional options for L-BFGS-B
            seed: Optional random seed for reproducibility

        Returns:
            (candidate, acq_value) tuple where:
            - candidate: Optimal latent tensor, shape (1, latent_dim)
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
            vae=self.vae,
            best_f=best_f,
        )

        # Ensure models are in eval mode (including likelihood for GP)
        self.gp_model.eval()
        if hasattr(self.gp_model, 'likelihood'):
            self.gp_model.likelihood.eval()
        self.vae.eval()

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
    """Compute latent space bounds from training data.

    Encodes training embeddings to latent space and computes
    bounds with optional margin for exploration.

    Args:
        encoder: Feature extractor (768D -> 10D)
        X_train: Training embeddings (N, 768)
        X_min: Min values for normalization
        X_max: Max values for normalization
        margin: Fraction to expand bounds (0.2 = 20% each side)

    Returns:
        Bounds tensor, shape (2, latent_dim)
    """
    encoder.eval()
    with torch.no_grad():
        # Normalize training data
        denom = X_max - X_min
        denom[denom == 0] = 1.0
        X_norm = (X_train - X_min) / denom

        # Encode to latent
        latents = encoder(X_norm)  # (N, 10)

        # Compute bounds
        z_min = latents.min(dim=0)[0]
        z_max = latents.max(dim=0)[0]

        # Expand bounds by margin
        z_range = z_max - z_min
        z_min = z_min - margin * z_range
        z_max = z_max + margin * z_range

    return torch.stack([z_min, z_max], dim=0)
