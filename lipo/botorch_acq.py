"""BoTorch qLogExpectedImprovement for latent space optimization.

This module provides BoTorch-based acquisition function optimization
for the LIPO pipeline. Uses qLogExpectedImprovement which is
numerically stable even when improvement values are extremely small.

Pipeline: z (32D VAE latent) -> Adapter -> z_gp (10D) -> Kumaraswamy Warp -> GP -> qLogEI

After optimization: z_opt (32D) -> VAE decoder -> embedding (768D) -> Vec2Text

Adapted from generation/invbo_decoder/botorch_acq.py for lipo.

Key improvements (v2):
- Distance penalty: penalizes latents far from training data (poor round-trip quality)
- Anchor-constrained bounds: limits optimization to near best observed points
"""

import logging
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf

logger = logging.getLogger(__name__)


class CompositeLogEI(AcquisitionFunction):
    """qLogEI with distance penalty for better round-trip quality.

    The GP model expects 32D VAE latent input and applies adapter internally:
        z (32D) -> GP (with adapter) -> z_gp (10D) -> Kumaraswamy Warp -> kernel -> qLogEI

    Distance Penalty:
        Latents far from training data are penalized because they're more likely
        to produce poor Vec2Text round-trip quality (embeddings outside Vec2Text's
        training distribution).

        penalty = distance_weight * max(0, min_distance - threshold)

    This keeps optimization in regions where Vec2Text can reliably reconstruct.
    """

    def __init__(
        self,
        model: Model,
        best_f: float,
        sampler: Optional[torch.Tensor] = None,
        X_train: Optional[torch.Tensor] = None,
        distance_weight: float = 2.0,
        distance_threshold: float = 0.3,
    ):
        """Initialize composite acquisition function.

        Args:
            model: GP model that accepts 32D VAE latents and applies adapter internally
            best_f: Best observed negated error rate (max of -error_rates for BoTorch maximization)
            sampler: Optional MC sampler for qLogEI
            X_train: Training latents (N, latent_dim) for distance penalty computation
            distance_weight: Weight for distance penalty (higher = stronger penalty)
            distance_threshold: Minimum distance before penalty kicks in (in normalized space)
        """
        super().__init__(model=model)
        self.best_f = best_f
        self.distance_weight = distance_weight
        self.distance_threshold = distance_threshold

        # Store training data for distance computation
        if X_train is not None:
            self.register_buffer("X_train", X_train)
        else:
            self.X_train = None

        # qLogEI for numerically stable expected improvement
        self._base_acq = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
        )

    def _compute_distance_penalty(self, X: torch.Tensor) -> torch.Tensor:
        """Compute distance penalty for points far from training data.

        Uses minimum L2 distance to any training point as the penalty signal.
        Points closer than threshold have zero penalty.

        Args:
            X: Query points with shape (batch, q, d) or (batch, d)

        Returns:
            Penalty values with shape (batch,) - negative values (penalty)
        """
        if self.X_train is None or len(self.X_train) == 0:
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        # Reshape X for distance computation
        original_shape = X.shape
        if X.dim() == 3:
            # (batch, q, d) -> (batch * q, d)
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        # Compute pairwise distances to all training points
        # X_flat: (B, d), X_train: (N, d) -> distances: (B, N)
        distances = torch.cdist(X_flat, self.X_train, p=2)

        # Minimum distance to any training point
        min_distances, _ = distances.min(dim=1)  # (B,)

        # Penalty: soft threshold with linear ramp
        # penalty = weight * max(0, distance - threshold)
        penalty = self.distance_weight * F.relu(min_distances - self.distance_threshold)

        # Reshape back if needed
        if len(original_shape) == 3:
            penalty = penalty.reshape(original_shape[0], original_shape[1]).mean(dim=1)

        # Return negative (it's a penalty, reduces acquisition value)
        return -penalty

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate acquisition on VAE latent points.

        Args:
            X: VAE latent points with shape (batch, q, d) where d=latent_dim
               For single-point acquisition, shape is (batch, 1, latent_dim)

        Returns:
            Acquisition values with shape (batch,)
        """
        # Base LogEI acquisition
        base_acq = self._base_acq(X)

        # Add distance penalty (keeps optimization near training data)
        distance_penalty = self._compute_distance_penalty(X)

        return base_acq + distance_penalty


class LatentSpaceAcquisition:
    """Optimizes qLogEI in VAE latent space (32D).

    Uses BoTorch's optimize_acqf with multi-start L-BFGS-B optimization.
    This provides proper gradient flow through:
        z (32D) -> GP (with trainable adapter) -> z_gp (10D) -> Kumaraswamy Warp -> kernel -> LogEI

    Key advantages over scipy L-BFGS-B:
    1. Gradient-based refinement with proper autograd
    2. Multi-start with raw_samples avoids poor local optima
    3. Sophisticated initialization from BoTorch

    Distance penalty (v2):
    4. Penalizes latents far from training data for better round-trip quality
    """

    def __init__(
        self,
        gp_model: Model,
        bounds: torch.Tensor,
        device: torch.device,
        X_train: Optional[torch.Tensor] = None,
        distance_weight: float = 2.0,
        distance_threshold: float = 0.3,
    ):
        """Initialize latent space acquisition optimizer.

        Args:
            gp_model: Trained GP model that accepts 32D VAE latents
            bounds: VAE latent space bounds, shape (2, latent_dim)
                    bounds[0] = lower bounds, bounds[1] = upper bounds
            device: Torch device
            X_train: Training latents for distance penalty (N, latent_dim)
            distance_weight: Weight for distance penalty (default 2.0)
            distance_threshold: Min distance before penalty (default 0.3)
        """
        self.gp_model = gp_model
        self.bounds = bounds.to(device)
        self.device = device
        self.X_train = X_train.to(device) if X_train is not None else None
        self.distance_weight = distance_weight
        self.distance_threshold = distance_threshold

    def optimize(
        self,
        best_f: float,
        num_restarts: int = 64,
        raw_samples: int = 1024,
        options: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find optimal VAE latent z (32D) maximizing qLogEI with distance penalty.

        Uses BoTorch's optimize_acqf with:
        1. raw_samples random starting points
        2. num_restarts L-BFGS-B optimizations from best raw samples
        3. Returns best result across all restarts
        4. Distance penalty keeps optimization near training data

        Args:
            best_f: Best observed negated error rate (max of -error_rates for BoTorch maximization)
            num_restarts: Number of L-BFGS-B restarts
            raw_samples: Number of initial random samples for seeding
            options: Additional options for L-BFGS-B
            seed: Optional random seed for reproducibility

        Returns:
            (candidate, acq_value) tuple where:
            - candidate: Optimal VAE latent tensor, shape (1, latent_dim)
            - acq_value: LogEI value at optimal point (includes distance penalty)
        """
        if options is None:
            options = {"maxiter": 200, "batch_limit": 5}

        # Set seed for reproducible multi-start optimization
        if seed is not None:
            torch.manual_seed(seed)

        # Create composite acquisition function with distance penalty
        acq_fn = CompositeLogEI(
            model=self.gp_model,
            best_f=best_f,
            X_train=self.X_train,
            distance_weight=self.distance_weight,
            distance_threshold=self.distance_threshold,
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
    """Compute bounds for 32D VAE latent space from training data.

    X_train is already stored as 32D VAE latents. We just need to
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
        # Normalize training data (already 32D VAE latents)
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


def get_anchor_constrained_bounds(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_min: torch.Tensor,
    X_max: torch.Tensor,
    top_k: int = 10,
    margin: float = 0.3,
) -> torch.Tensor:
    """Compute bounds constrained to region around best observed points.

    Instead of optimizing over the entire latent space, constrains the search
    to a region around the top-k best observed points. This keeps optimization
    in areas with known good round-trip quality.

    Args:
        X_train: Training VAE latents (N, latent_dim)
        y_train: Training targets (N,) - negative error rates (higher = better)
        X_min: Min values for normalization (latent_dim,)
        X_max: Max values for normalization (latent_dim,)
        top_k: Number of best points to use as anchors
        margin: Fraction to expand bounds around anchors

    Returns:
        Bounds tensor, shape (2, latent_dim) for constrained VAE latent space
    """
    with torch.no_grad():
        # Normalize training data
        denom = X_max - X_min
        denom[denom == 0] = 1.0
        latents_norm = (X_train - X_min) / denom

        # Get top-k best points (highest y = lowest error)
        k = min(top_k, len(y_train))
        _, best_indices = y_train.topk(k)
        anchor_points = latents_norm[best_indices]

        # Compute bounds from anchor points
        z_min = anchor_points.min(dim=0)[0]
        z_max = anchor_points.max(dim=0)[0]

        # Expand bounds by margin (relative to anchor range)
        z_range = z_max - z_min
        z_range = torch.clamp(z_range, min=0.1)  # Ensure minimum range
        z_min = z_min - margin * z_range
        z_max = z_max + margin * z_range

        # Clamp to [0, 1] normalized space
        z_min = torch.clamp(z_min, min=0.0)
        z_max = torch.clamp(z_max, max=1.0)

    return torch.stack([z_min, z_max], dim=0)
