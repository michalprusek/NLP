"""
High-Dimensional GP for FlowPO (256D latent with ~20 training points).

Addresses the curse of dimensionality through:
1. Isotropic kernel - single lengthscale for all 256 dims (safe with few points)
2. SAAS prior - learns which dims matter (better when n >= 30)
3. Adaptive switching - uses isotropic initially, SAAS when enough data

Interface compatible with GPGuidedFlowGenerator:
    gp.predict(z) -> (mean, std)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# SAAS imports (optional - may not be available in all BoTorch versions)
try:
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    SAAS_AVAILABLE = True
except ImportError:
    SAAS_AVAILABLE = False


logger = logging.getLogger(__name__)


def compute_ucb_beta(
    iteration: int,
    total_iterations: int,
    beta_start: float = 4.0,
    beta_end: float = 2.0,
    schedule: str = "linear",
) -> float:
    """
    Compute UCB beta with decay schedule.

    High beta (3-4) is CRITICAL for 256D with few points:
    - More exploration prevents getting stuck
    - Helps discover structure in high-D space

    Args:
        iteration: Current iteration
        total_iterations: Total iterations planned
        beta_start: Initial beta (high exploration)
        beta_end: Final beta (exploitation)
        schedule: "linear" or "sqrt" (slower decay)

    Returns:
        UCB beta value for current iteration
    """
    if total_iterations <= 0:
        return beta_start

    progress = iteration / total_iterations

    if schedule == "sqrt":
        # Slower decay, more exploration
        progress = progress ** 0.5
    elif schedule == "cosine":
        import math
        progress = (1 - math.cos(progress * math.pi)) / 2

    return beta_start + (beta_end - beta_start) * progress


class TrustRegion:
    """
    Trust region management for GP-guided generation.

    Constrains how far guidance can push samples from training data.
    Critical for 256D where GP predictions in empty regions are unreliable.
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        scale: float = 2.0,
        device: str = "cuda",
    ):
        """
        Initialize trust region from training data.

        Args:
            X_train: (N, D) training latents
            scale: Trust radius = scale * max_dist_from_centroid
            device: Torch device
        """
        self.device = device
        X = X_train.to(device)

        self.centroid = X.mean(dim=0)
        dists = (X - self.centroid).norm(dim=-1)
        self.radius = dists.max().item() * scale
        self.max_dist = dists.max().item()

        logger.info(
            f"TrustRegion: centroid norm={self.centroid.norm():.3f}, "
            f"radius={self.radius:.3f}, scale={scale}"
        )

    def is_inside(self, z: torch.Tensor) -> torch.Tensor:
        """Check if points are inside trust region."""
        dist = (z - self.centroid.to(z.device)).norm(dim=-1)
        return dist <= self.radius

    def clip_guidance(
        self,
        z: torch.Tensor,
        grad: torch.Tensor,
        max_step: float = 0.1,
    ) -> torch.Tensor:
        """
        Clip guidance gradient to stay within trust region.

        Args:
            z: (B, D) current positions
            grad: (B, D) guidance gradients
            max_step: Maximum step size as fraction of radius

        Returns:
            (B, D) clipped gradients
        """
        centroid = self.centroid.to(z.device)

        # Distance to boundary
        dist_to_centroid = (z - centroid).norm(dim=-1, keepdim=True)
        dist_to_boundary = self.radius - dist_to_centroid

        # Scale gradient based on distance to boundary
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Max allowed step
        max_allowed = (dist_to_boundary / grad_norm).clamp(min=0, max=max_step * self.radius)

        # Scale gradients
        scale = (max_allowed / (grad_norm + 1e-8)).clamp(max=1.0)

        return grad * scale


class HighDimGPBase(ABC, nn.Module):
    """Abstract base for high-dimensional GP with predict(z) -> (mean, std) interface."""

    def __init__(self, latent_dim: int = 256, device: str = "cuda"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Training data
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.n_train: int = 0

        # Normalization stats
        self.X_mean: Optional[torch.Tensor] = None
        self.X_std: Optional[torch.Tensor] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        # Best observed value (for EI)
        self.best_error_rate: float = 1.0

        # Trust region
        self.trust_region: Optional[TrustRegion] = None

        # Use numerical gradient when analytic is too small
        self.use_numerical_grad: bool = True
        self.numerical_grad_eps: float = 0.01

    def compute_guidance_gradient(
        self,
        z: torch.Tensor,
        ucb_beta: float = 4.0,
    ) -> torch.Tensor:
        """
        Compute guidance gradient towards low-error regions.

        For high-dimensional spaces with few points, analytic GP gradient
        is often numerically zero. This method uses a hybrid approach:
        1. Try analytic gradient first
        2. If too small, use direction towards best training point

        Args:
            z: (B, D) current positions
            ucb_beta: UCB exploration parameter

        Returns:
            (B, D) guidance gradients
        """
        z = z.to(self.device)
        B, D = z.shape

        # Try analytic gradient
        z_grad = z.detach().requires_grad_(True)
        mean, std = self.predict(z_grad)
        reward = -mean + ucb_beta * std  # UCB for minimization
        reward.sum().backward()

        if z_grad.grad is not None:
            analytic_grad = z_grad.grad.detach().clone()
            grad_norm = analytic_grad.norm(dim=-1, keepdim=True)

            # Check if gradient is meaningful
            if grad_norm.mean() > 1e-6:
                return analytic_grad

        # Fallback: Direction towards best training point
        if self.X_train is not None and self.y_train is not None:
            # Find best training point
            best_idx = self.y_train.argmin()
            best_point = self.X_train[best_idx:best_idx+1]  # (1, D)

            # Direction from z to best point
            direction = best_point - z  # (B, D)

            # Normalize to unit length
            dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            direction = direction / dir_norm

            # Scale by distance to best and uncertainty
            with torch.no_grad():
                _, std = self.predict(z)
            scale = std.unsqueeze(-1)  # More gradient when uncertain

            logger.debug(
                f"Using nearest-neighbor fallback gradient, "
                f"best_error={self.y_train[best_idx]:.3f}"
            )

            return direction * scale

        # No training data - return zeros
        return torch.zeros_like(z)

    @abstractmethod
    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and std for latent vectors.

        Args:
            z: (B, D) latent vectors

        Returns:
            mean: (B,) predicted error rates
            std: (B,) prediction uncertainties
        """
        pass

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Fit GP to training data.

        Args:
            X: (N, D) latent vectors
            y: (N,) error rates

        Returns:
            Success flag
        """
        pass

    def add_observation(self, z: torch.Tensor, score: torch.Tensor) -> None:
        """Add new observation and refit GP."""
        z = z.detach().to(self.device)
        score = score.detach().to(self.device)

        if z.dim() == 1:
            z = z.unsqueeze(0)
        if score.dim() == 0:
            score = score.unsqueeze(0)

        if self.X_train is None:
            self.X_train = z
            self.y_train = score
        else:
            self.X_train = torch.cat([self.X_train, z], dim=0)
            self.y_train = torch.cat([self.y_train, score], dim=0)

        self.n_train = self.X_train.shape[0]
        self.best_error_rate = min(self.best_error_rate, score.min().item())

        # Refit
        self.fit(self.X_train, self.y_train)

    def _normalize_X(self, X: torch.Tensor) -> torch.Tensor:
        """Z-score normalize input."""
        if self.X_mean is None:
            return X
        return (X - self.X_mean.to(X.device)) / self.X_std.to(X.device)

    def _fit_normalizer(self, X: torch.Tensor) -> None:
        """Fit normalization stats from training data."""
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0).clamp(min=1e-6)


class IsotropicGPModel(ExactGP):
    """GPyTorch model with isotropic Matern kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # Isotropic kernel - single lengthscale for all dimensions
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=None)
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class IsotropicHighDimGP(HighDimGPBase):
    """
    Isotropic Matern GP for 256D with few training points.

    Safe choice when n << d. Uses single shared lengthscale instead of
    ARD (256 lengthscales), reducing parameters from 258 to 3.

    This is the recommended starting point for high-dimensional BO.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        device: str = "cuda",
        ucb_beta: float = 3.0,
        trust_region_scale: float = 2.0,
        cold_start_threshold: int = 5,
    ):
        """
        Initialize isotropic GP.

        Args:
            latent_dim: Latent dimension (256 for FlowPO)
            device: Torch device
            ucb_beta: UCB exploration parameter (high for 256D)
            trust_region_scale: Trust region radius multiplier
            cold_start_threshold: Return prior below this many points
        """
        super().__init__(latent_dim, device)

        self.ucb_beta = ucb_beta
        self.trust_region_scale = trust_region_scale
        self.cold_start_threshold = cold_start_threshold

        self.gp_model: Optional[IsotropicGPModel] = None
        self.likelihood: Optional[GaussianLikelihood] = None

        logger.info(
            f"IsotropicHighDimGP: latent_dim={latent_dim}, "
            f"ucb_beta={ucb_beta}, trust_region_scale={trust_region_scale}"
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Fit GP to training data.

        Args:
            X: (N, D) latent vectors
            y: (N,) error rates (NOT negated)

        Returns:
            Success flag
        """
        X = X.to(self.device)
        y = y.to(self.device)

        self.X_train = X
        self.y_train = y
        self.n_train = X.shape[0]
        self.best_error_rate = y.min().item()

        # Fit normalizer
        self._fit_normalizer(X)
        X_norm = self._normalize_X(X)

        # Normalize y to zero mean, unit std
        self.y_mean = y.mean().item()
        self.y_std = y.std().item() if y.std() > 1e-6 else 1.0
        y_norm = (y - self.y_mean) / self.y_std

        # Create trust region
        self.trust_region = TrustRegion(X, self.trust_region_scale, self.device)

        # Initialize GP
        self.likelihood = GaussianLikelihood().to(self.device)
        self.gp_model = IsotropicGPModel(X_norm, y_norm, self.likelihood).to(self.device)

        # Fit hyperparameters
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        for i in range(100):
            optimizer.zero_grad()
            output = self.gp_model(X_norm)
            loss = -mll(output, y_norm)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                logger.debug(f"GP fit iter {i}: loss={loss.item():.4f}")

        self.gp_model.eval()
        self.likelihood.eval()

        # Log fitted hyperparameters
        lengthscale = self.gp_model.covar_module.base_kernel.lengthscale.item()
        outputscale = self.gp_model.covar_module.outputscale.item()
        noise = self.likelihood.noise.item()
        logger.info(
            f"GP fitted: lengthscale={lengthscale:.4f}, "
            f"outputscale={outputscale:.4f}, noise={noise:.4f}"
        )

        return True

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and std for latent vectors.

        Returns error rates (higher is worse).
        Supports gradient computation for acquisition optimization.

        Args:
            z: (B, D) latent vectors

        Returns:
            mean: (B,) predicted error rates (differentiable if z.requires_grad)
            std: (B,) prediction uncertainties (differentiable if z.requires_grad)
        """
        z = z.to(self.device)
        batch_size = z.shape[0]

        # Cold start: return prior (with gradients if needed)
        if self.n_train < self.cold_start_threshold or self.gp_model is None:
            # Use z to maintain gradient connection
            mean = torch.full((batch_size,), 0.5, device=self.device)
            std = torch.full((batch_size,), 0.3, device=self.device)
            if z.requires_grad:
                # Add tiny dependency on z to allow gradient flow
                mean = mean + 0.0 * z.sum(dim=-1)
                std = std + 0.0 * z.sum(dim=-1)
            return mean, std

        # Normalize input - keep gradients flowing
        z_norm = self._normalize_X(z)

        # Predict WITH gradients for acquisition optimization
        # GPyTorch supports autograd through posterior mean
        with gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()
            pred = self.gp_model(z_norm)
            mean_norm = pred.mean
            # For variance, we use lazy evaluation
            std_norm = pred.variance.sqrt()

        # Denormalize
        mean = mean_norm * self.y_std + self.y_mean
        std = std_norm * self.y_std

        # Clamp to valid range
        mean = mean.clamp(0, 1)
        std = std.clamp(min=1e-6)

        return mean, std


class SaasHighDimGP(HighDimGPBase):
    """
    SAAS Fully Bayesian GP for high-dimensional optimization.

    Uses Sparse Axis-Aligned Subspaces prior to identify relevant dimensions.
    Better than isotropic when n >= 30, but requires MCMC fitting (slower).

    Reference: Eriksson & Jankowiak (2021) "High-Dimensional Bayesian
    Optimization with Sparse Axis-Aligned Subspaces"
    """

    def __init__(
        self,
        latent_dim: int = 256,
        device: str = "cuda",
        warmup_steps: int = 256,
        num_samples: int = 128,
        thinning: int = 16,
    ):
        """
        Initialize SAAS GP.

        Args:
            latent_dim: Latent dimension
            device: Torch device
            warmup_steps: MCMC warmup steps
            num_samples: MCMC samples
            thinning: MCMC thinning
        """
        super().__init__(latent_dim, device)

        if not SAAS_AVAILABLE:
            raise ImportError(
                "SAAS GP requires BoTorch >= 0.10.0 with SAAS support. "
                "Install with: pip install botorch>=0.10.0"
            )

        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning

        self.gp_model: Optional[SaasFullyBayesianSingleTaskGP] = None

        logger.info(
            f"SaasHighDimGP: latent_dim={latent_dim}, "
            f"warmup={warmup_steps}, samples={num_samples}"
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """Fit SAAS GP via MCMC."""
        X = X.to(self.device)
        y = y.to(self.device)

        self.X_train = X
        self.y_train = y
        self.n_train = X.shape[0]
        self.best_error_rate = y.min().item()

        # Fit normalizer
        self._fit_normalizer(X)
        X_norm = self._normalize_X(X)

        # SAAS expects y in (N, 1) format
        y_2d = y.unsqueeze(-1)

        # Create SAAS GP
        self.gp_model = SaasFullyBayesianSingleTaskGP(
            train_X=X_norm,
            train_Y=y_2d,
            outcome_transform=Standardize(m=1),
        ).to(self.device)

        # Fit via MCMC
        logger.info("Fitting SAAS GP via NUTS MCMC (this may take a minute)...")
        fit_fully_bayesian_model_nuts(
            self.gp_model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=False,
        )

        logger.info("SAAS GP fitting complete")
        return True

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with SAAS GP.

        Note: SAAS posterior has shape (num_mcmc_samples, batch, 1).
        We average over MCMC samples to get final prediction.
        """
        z = z.to(self.device)

        if self.gp_model is None:
            batch_size = z.shape[0]
            return (
                torch.full((batch_size,), 0.5, device=self.device),
                torch.full((batch_size,), 0.3, device=self.device),
            )

        z_norm = self._normalize_X(z)

        with torch.no_grad():
            posterior = self.gp_model.posterior(z_norm)
            # Shape: (num_mcmc_samples, batch, 1) -> average over MCMC samples
            mean = posterior.mean.squeeze(-1).mean(dim=0)  # (batch,)
            # For std, combine MCMC uncertainty (variance of means) and aleatoric (mean of variances)
            var_epistemic = posterior.mean.squeeze(-1).var(dim=0)  # Variance of means
            var_aleatoric = posterior.variance.squeeze(-1).mean(dim=0)  # Mean of variances
            std = (var_epistemic + var_aleatoric).sqrt()  # (batch,)

        mean = mean.clamp(0, 1)
        std = std.clamp(min=1e-6)

        return mean, std

    def compute_guidance_gradient(
        self,
        z: torch.Tensor,
        ucb_beta: float = 4.0,
    ) -> torch.Tensor:
        """
        Weighted SAAS Attractor - anisotropic gradient using learned lengthscales.

        SAAS learns which dimensions are important via ARD lengthscales:
        - Small lengthscale = important dimension (fast change in function)
        - Large lengthscale = noise dimension (slow change)

        We use inverse lengthscales as weights for the attractor direction,
        giving stronger gradient in dimensions that matter.

        Args:
            z: (B, D) current positions
            ucb_beta: UCB exploration parameter

        Returns:
            (B, D) weighted guidance gradients
        """
        z = z.to(self.device)
        B, D = z.shape

        if self.X_train is None or self.y_train is None:
            return torch.zeros(B, D, device=self.device)

        # Find best training point (attractor target)
        best_idx = self.y_train.argmin()
        best_point = self.X_train[best_idx:best_idx+1]  # (1, D)

        # Extract learned lengthscales from SAAS model
        weights = torch.ones(1, D, device=self.device)

        if self.gp_model is not None:
            try:
                # SAAS model stores lengthscales in covar_module.base_kernel.lengthscale
                # Shape: (num_mcmc_samples, 1, D)
                raw_lengthscales = self.gp_model.covar_module.base_kernel.lengthscale

                # Take median across MCMC samples for robust estimate -> (1, D)
                median_lengthscale = raw_lengthscales.median(dim=0).values.squeeze(0)  # (D,)

                # Weights are inverse of lengthscale:
                # - Small lengthscale -> large weight (important dimension)
                # - Large lengthscale -> small weight (noise dimension)
                weights = 1.0 / (median_lengthscale + 1e-6)

                # Normalize so average weight is 1.0 (preserves step magnitude)
                weights = weights / weights.mean()
                weights = weights.unsqueeze(0)  # (1, D)

                # Log dimension importance
                top_k = 10
                top_indices = weights.squeeze().argsort(descending=True)[:top_k]
                top_weights = weights.squeeze()[top_indices]
                logger.debug(
                    f"SAAS top-{top_k} important dims: {top_indices.tolist()}, "
                    f"weights: {top_weights.tolist()[:5]}..."
                )

            except Exception as e:
                logger.warning(f"Could not extract SAAS lengthscales: {e}")
                weights = torch.ones(1, D, device=self.device)

        # Compute direction to best point
        diff = best_point - z  # (B, D)

        # Apply anisotropic weighting
        # Gradient is stronger in important dimensions (small lengthscale)
        weighted_diff = diff * weights  # (B, D)

        # Normalize to unit length
        dir_norm = weighted_diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = weighted_diff / dir_norm

        # Scale by uncertainty
        with torch.no_grad():
            _, std = self.predict(z)
        scale = std.unsqueeze(-1)

        return direction * scale


class AdaptiveHighDimGP(HighDimGPBase):
    """
    Adaptive GP that switches between Isotropic and SAAS based on data size.

    Strategy:
    - n < switch_threshold: Use Isotropic (safe, fast)
    - n >= switch_threshold: Switch to SAAS (better, but needs data)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        device: str = "cuda",
        switch_threshold: int = 30,
        ucb_beta: float = 3.0,
        trust_region_scale: float = 2.0,
    ):
        """
        Initialize adaptive GP.

        Args:
            latent_dim: Latent dimension
            device: Torch device
            switch_threshold: Switch to SAAS when n >= this
            ucb_beta: UCB beta for isotropic GP
            trust_region_scale: Trust region scale for isotropic GP
        """
        super().__init__(latent_dim, device)

        self.switch_threshold = switch_threshold
        self.ucb_beta = ucb_beta
        self.trust_region_scale = trust_region_scale

        # Create both GPs
        self.isotropic_gp = IsotropicHighDimGP(
            latent_dim=latent_dim,
            device=device,
            ucb_beta=ucb_beta,
            trust_region_scale=trust_region_scale,
        )

        self.saas_gp: Optional[SaasHighDimGP] = None
        if SAAS_AVAILABLE:
            self.saas_gp = SaasHighDimGP(latent_dim=latent_dim, device=device)

        self._active_gp: HighDimGPBase = self.isotropic_gp
        self._using_saas = False

        logger.info(
            f"AdaptiveHighDimGP: switch_threshold={switch_threshold}, "
            f"SAAS available={SAAS_AVAILABLE}"
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """Fit appropriate GP based on data size."""
        self.X_train = X
        self.y_train = y
        self.n_train = X.shape[0]
        self.best_error_rate = y.min().item()

        # Decide which GP to use
        should_use_saas = (
            self.n_train >= self.switch_threshold
            and self.saas_gp is not None
        )

        if should_use_saas and not self._using_saas:
            logger.info(
                f"Switching to SAAS GP (n={self.n_train} >= {self.switch_threshold})"
            )
            self._active_gp = self.saas_gp
            self._using_saas = True
        elif not should_use_saas:
            self._active_gp = self.isotropic_gp
            self._using_saas = False

        # Fit active GP
        return self._active_gp.fit(X, y)

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using active GP."""
        return self._active_gp.predict(z)

    @property
    def trust_region(self) -> Optional[TrustRegion]:
        """Get trust region from active GP."""
        return self._active_gp.trust_region


if __name__ == "__main__":
    print("Testing High-Dim GP...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create synthetic data
    latent_dim = 256
    n_train = 20

    X_train = torch.randn(n_train, latent_dim, device=device)
    y_train = torch.rand(n_train, device=device) * 0.5 + 0.2  # Error rates 0.2-0.7

    print(f"\nTraining data: X={X_train.shape}, y={y_train.shape}")
    print(f"y range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    # Test Isotropic GP
    print("\n--- Testing IsotropicHighDimGP ---")
    iso_gp = IsotropicHighDimGP(latent_dim=latent_dim, device=device)
    iso_gp.fit(X_train, y_train)

    X_test = torch.randn(5, latent_dim, device=device)
    mean, std = iso_gp.predict(X_test)
    print(f"Predictions: mean={mean}, std={std}")

    # Test UCB beta schedule
    print("\n--- Testing UCB Beta Schedule ---")
    for i in [0, 10, 25, 50, 100]:
        beta = compute_ucb_beta(i, 100, beta_start=4.0, beta_end=2.0)
        print(f"  iter {i}/100: beta={beta:.2f}")

    # Test trust region
    print("\n--- Testing Trust Region ---")
    tr = TrustRegion(X_train, scale=2.0, device=device)
    inside = tr.is_inside(X_test)
    print(f"Points inside trust region: {inside}")

    print("\n[OK] High-Dim GP tests passed!")
