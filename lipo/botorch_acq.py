"""BoTorch qLogExpectedImprovement for latent space optimization.

This module provides BoTorch-based acquisition function optimization
for the LIPO pipeline. Uses qLogExpectedImprovement which is
numerically stable even when improvement values are extremely small.

Pipeline: z (32D VAE latent) -> GP (ARD Matern kernel) -> qLogEI

After optimization: z_opt (32D) -> VAE decoder -> embedding (768D) -> Vec2Text

Key design (research-backed):
- Distance penalty prevents GP from exploring empty latent space regions
- ARD kernel learns dimension importance
- qLogEI for numerical stability
- TuRBO optional (disabled by default for global exploration)
"""

import logging
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf

logger = logging.getLogger(__name__)


class DistancePenalizedLogEI(qLogExpectedImprovement):
    """qLogEI with distance penalty for points far from training data.

    Penalizes candidates whose minimum distance to training points exceeds
    a threshold. This prevents GP from proposing points in "empty space"
    where VAE reconstruction quality is poor.

    Penalty formula (in log-space):
        penalty = weight * max(0, min_dist - threshold)
        penalized_logEI = logEI - penalty

    Subtracting in log-space is equivalent to dividing EI by exp(penalty).
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        X_train_normalized: torch.Tensor,
        distance_weight: float = 2.0,
        distance_threshold: float = 0.3,
        **kwargs,
    ):
        """Initialize distance-penalized LogEI.

        Args:
            model: Trained GP model
            best_f: Best observed value (for EI baseline)
            X_train_normalized: Training latents normalized to [0,1]^D, shape (N, D)
            distance_weight: Penalty strength multiplier
            distance_threshold: Min distance before penalty activates
            **kwargs: Additional args for qLogExpectedImprovement
        """
        super().__init__(model=model, best_f=best_f, **kwargs)
        self.register_buffer("X_train", X_train_normalized)
        self.distance_weight = distance_weight
        self.distance_threshold = distance_threshold

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute penalized LogEI.

        Args:
            X: Candidate points, shape (batch, q, D) or (batch, D)

        Returns:
            Penalized LogEI values, shape (batch,)
        """
        # Compute base LogEI
        log_ei = super().forward(X)

        # Handle different input shapes
        if X.dim() == 2:
            X_flat = X  # (batch, D)
        else:
            # (batch, q, D) -> flatten to (batch*q, D) for distance computation
            batch_size, q, dim = X.shape
            X_flat = X.view(-1, dim)

        # Compute pairwise distances to training data
        # X_flat: (M, D), X_train: (N, D) -> dists: (M, N)
        dists = torch.cdist(X_flat, self.X_train)

        # Minimum distance per candidate
        min_dists = dists.min(dim=-1)[0]  # (M,)

        # Reshape back if needed and average over q
        if X.dim() == 3:
            min_dists = min_dists.view(batch_size, q).mean(dim=-1)  # (batch,)

        # Compute penalty (only for distances > threshold)
        excess_dist = torch.clamp(min_dists - self.distance_threshold, min=0.0)
        penalty = self.distance_weight * excess_dist

        # Apply penalty in log-space
        penalized_log_ei = log_ei - penalty

        return penalized_log_ei


class DistancePenalizedUCB(UpperConfidenceBound):
    """UCB with distance penalty for points far from training data.

    UCB formula: μ(x) + β * σ(x)
    where β controls exploration-exploitation trade-off.

    Higher β = more exploration (prefers high uncertainty regions).

    Penalty formula:
        penalty = weight * max(0, min_dist - threshold)
        penalized_UCB = UCB - penalty
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        X_train_normalized: torch.Tensor,
        distance_weight: float = 2.0,
        distance_threshold: float = 0.3,
        **kwargs,
    ):
        """Initialize distance-penalized UCB.

        Args:
            model: Trained GP model
            beta: UCB exploration parameter (higher = more exploration)
            X_train_normalized: Training latents normalized to [0,1]^D, shape (N, D)
            distance_weight: Penalty strength multiplier
            distance_threshold: Min distance before penalty activates
            **kwargs: Additional args for UpperConfidenceBound
        """
        super().__init__(model=model, beta=beta, **kwargs)
        self.register_buffer("X_train", X_train_normalized)
        self.distance_weight = distance_weight
        self.distance_threshold = distance_threshold

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute penalized UCB.

        Args:
            X: Candidate points, shape (batch, 1, D) or (batch, D)

        Returns:
            Penalized UCB values, shape (batch,)
        """
        # Compute base UCB
        ucb = super().forward(X)

        # Handle different input shapes
        if X.dim() == 2:
            X_flat = X  # (batch, D)
        else:
            # (batch, q, D) -> flatten to (batch*q, D) for distance computation
            batch_size, q, dim = X.shape
            X_flat = X.view(-1, dim)

        # Compute pairwise distances to training data
        dists = torch.cdist(X_flat, self.X_train)

        # Minimum distance per candidate
        min_dists = dists.min(dim=-1)[0]  # (M,)

        # Reshape back if needed and average over q
        if X.dim() == 3:
            min_dists = min_dists.view(batch_size, q).mean(dim=-1)  # (batch,)

        # Compute penalty (only for distances > threshold)
        excess_dist = torch.clamp(min_dists - self.distance_threshold, min=0.0)
        penalty = self.distance_weight * excess_dist

        # Apply penalty
        penalized_ucb = ucb - penalty

        return penalized_ucb


class LatentSpaceAcquisition:
    """Optimizes acquisition function in VAE latent space.

    Supports both LogEI (exploitation-focused) and UCB (exploration-focused).

    Uses BoTorch's optimize_acqf with multi-start L-BFGS-B optimization.
    This provides proper gradient flow:
        z (latent_dim) -> GP (ARD Matern kernel) -> Acquisition

    Key advantages:
    1. Gradient-based refinement with proper autograd
    2. Multi-start with raw_samples avoids poor local optima
    3. Sophisticated initialization from BoTorch
    4. Distance penalty prevents exploration in empty latent space
    """

    def __init__(
        self,
        gp_model: Model,
        bounds: torch.Tensor,
        device: torch.device,
        X_train_normalized: Optional[torch.Tensor] = None,
        distance_penalty_enabled: bool = False,
        distance_weight: float = 2.0,
        distance_threshold: float = 0.3,
        acquisition_type: str = "ucb",
        ucb_beta: float = 2.0,
    ):
        """Initialize latent space acquisition optimizer.

        Args:
            gp_model: Trained GP model that operates on VAE latents
            bounds: VAE latent space bounds, shape (2, latent_dim)
                    bounds[0] = lower bounds, bounds[1] = upper bounds
            device: Torch device
            X_train_normalized: Training latents in [0,1]^D (required if penalty enabled)
            distance_penalty_enabled: Whether to apply distance penalty
            distance_weight: Penalty strength multiplier
            distance_threshold: Min distance before penalty activates
            acquisition_type: "ucb" or "logei"
            ucb_beta: UCB exploration parameter (higher = more exploration)
        """
        self.gp_model = gp_model
        self.bounds = bounds.to(device)
        self.device = device
        self.X_train_normalized = X_train_normalized
        self.distance_penalty_enabled = distance_penalty_enabled
        self.distance_weight = distance_weight
        self.distance_threshold = distance_threshold
        self.acquisition_type = acquisition_type.lower()
        self.ucb_beta = ucb_beta

    def optimize(
        self,
        best_f: float,
        num_restarts: int = 64,
        raw_samples: int = 4096,
        options: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal VAE latent z maximizing acquisition function.

        Uses BoTorch's optimize_acqf with:
        1. raw_samples random starting points
        2. num_restarts L-BFGS-B optimizations from best raw samples
        3. Returns best result across all restarts

        Args:
            best_f: Best observed negated error rate (max of -error_rates for BoTorch maximization)
                    Note: Only used for LogEI, ignored for UCB
            num_restarts: Number of L-BFGS-B restarts
            raw_samples: Number of initial random samples for seeding
            options: Additional options for L-BFGS-B
            seed: Optional random seed for reproducibility

        Returns:
            (candidate, acq_value) tuple where:
            - candidate: Optimal VAE latent tensor, shape (1, latent_dim)
            - acq_value: Acquisition value at optimal point
        """
        if options is None:
            options = {"maxiter": 200, "batch_limit": 5}

        # Set seed for reproducible multi-start optimization
        if seed is not None:
            torch.manual_seed(seed)

        # Create acquisition function based on type
        if self.acquisition_type == "ucb":
            acq_fn = self._create_ucb_acquisition()
        else:
            acq_fn = self._create_logei_acquisition(best_f)

        # Ensure models are in eval mode
        self.gp_model.eval()
        if hasattr(self.gp_model, 'likelihood'):
            self.gp_model.likelihood.eval()

        # Optimize using BoTorch's optimize_acqf
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

        # Log warnings at debug level
        if caught_warnings:
            logger.debug(
                f"Optimization completed with {len(caught_warnings)} warnings "
                f"(common in L-BFGS-B optimization)"
            )

        return candidate, acq_value

    def _create_ucb_acquisition(self):
        """Create UCB acquisition function (with or without distance penalty)."""
        if self.distance_penalty_enabled and self.X_train_normalized is not None:
            acq_fn = DistancePenalizedUCB(
                model=self.gp_model,
                beta=self.ucb_beta,
                X_train_normalized=self.X_train_normalized.to(self.device),
                distance_weight=self.distance_weight,
                distance_threshold=self.distance_threshold,
            )
            logger.debug(
                f"Using DistancePenalizedUCB (beta={self.ucb_beta}, "
                f"weight={self.distance_weight}, threshold={self.distance_threshold})"
            )
        else:
            acq_fn = UpperConfidenceBound(
                model=self.gp_model,
                beta=self.ucb_beta,
            )
            logger.debug(f"Using UpperConfidenceBound (beta={self.ucb_beta})")
        return acq_fn

    def _create_logei_acquisition(self, best_f: float):
        """Create LogEI acquisition function (with or without distance penalty)."""
        if self.distance_penalty_enabled and self.X_train_normalized is not None:
            acq_fn = DistancePenalizedLogEI(
                model=self.gp_model,
                best_f=best_f,
                X_train_normalized=self.X_train_normalized.to(self.device),
                distance_weight=self.distance_weight,
                distance_threshold=self.distance_threshold,
            )
            logger.debug(
                f"Using DistancePenalizedLogEI (weight={self.distance_weight}, "
                f"threshold={self.distance_threshold})"
            )
        else:
            acq_fn = qLogExpectedImprovement(
                model=self.gp_model,
                best_f=best_f,
            )
            logger.debug("Using qLogExpectedImprovement")
        return acq_fn


def get_latent_bounds(
    encoder: nn.Module,
    X_train: torch.Tensor,
    X_min: torch.Tensor,
    X_max: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute bounds for VAE latent space from training data.

    X_train is already stored as VAE latents. We just need to
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
        # Normalize training data (VAE latents)
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
