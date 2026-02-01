"""Riemannian GP surrogates for manifold-aware optimization.

Implements:
- RiemannianGP: GP with geodesic distance kernel
- GeodesicTuRBOGP: TuRBO with geodesic trust region

These methods respect the geometry of the hypersphere where SONAR
embeddings naturally lie (they're normalized to unit length).

References:
- Jaquier et al. (2022) "Geometry-aware Bayesian Optimization in Robotics"
- Wilson & Nickisch (2015) "Kernel Interpolation for Scalable Structured GP"
"""

import logging
import math
import warnings
from typing import Optional, Tuple

import gpytorch
import torch
import torch.nn.functional as F
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate

logger = logging.getLogger(__name__)


class ArcCosineKernel(Kernel):
    """Arc-cosine kernel for normalized vectors.

    k(x, x') = 1 - arccos(x·x') / π

    This kernel measures similarity based on the angle between vectors,
    appropriate for data on the unit hypersphere.
    """

    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **kwargs):
        # Normalize inputs to unit sphere
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)

        # Compute cosine similarity
        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

        # Clamp for numerical stability
        cos_sim = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)

        # Arc-cosine kernel
        return 1.0 - torch.acos(cos_sim) / math.pi


class GeodesicMaternKernel(Kernel):
    """Matern kernel using geodesic distance on the hypersphere.

    k(x, x') = Matern(d_geo(x, x') / ℓ)

    where d_geo is the geodesic distance: arccos(x·x' / ||x|| ||x'||)
    """

    has_lengthscale = True

    def __init__(self, nu: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu

    def _geodesic_distance(self, x1, x2, diag=False):
        """Compute geodesic distance on unit sphere."""
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)

        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

        cos_sim = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)
        return torch.acos(cos_sim)

    def _matern52(self, dist):
        """Matern-5/2 kernel formula."""
        sqrt5 = math.sqrt(5)
        return (1 + sqrt5 * dist + 5 * dist ** 2 / 3) * torch.exp(-sqrt5 * dist)

    def _matern32(self, dist):
        """Matern-3/2 kernel formula."""
        sqrt3 = math.sqrt(3)
        return (1 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)

    def forward(self, x1, x2, diag=False, **kwargs):
        geo_dist = self._geodesic_distance(x1, x2, diag)
        scaled_dist = geo_dist / self.lengthscale

        if self.nu == 2.5:
            return self._matern52(scaled_dist)
        elif self.nu == 1.5:
            return self._matern32(scaled_dist)
        else:
            raise ValueError(f"Unsupported nu: {self.nu}")


class RiemannianGP(BaseGPSurrogate):
    """Riemannian GP with geodesic kernel for hyperspherical data.

    Uses geodesic distance instead of Euclidean distance, which is
    more appropriate for normalized embeddings (like SONAR).

    Key hyperparameters:
        kernel: "arccosine" or "geodesic_matern52"
        normalize_inputs: Whether to project inputs to unit sphere
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.normalize_inputs = config.normalize_inputs
        self.kernel_type = config.kernel
        self.initial_lengthscale = 1.0  # Geodesic distances are in [0, π]

    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Project to unit hypersphere if configured."""
        if self.normalize_inputs:
            return F.normalize(X, p=2, dim=-1)
        return X

    def _create_kernel(self):
        """Create geodesic kernel based on config."""
        if self.kernel_type == "arccosine":
            return ScaleKernel(ArcCosineKernel())
        elif self.kernel_type == "geodesic_matern52":
            kernel = GeodesicMaternKernel(nu=2.5)
            return ScaleKernel(kernel)
        elif self.kernel_type == "geodesic_matern32":
            kernel = GeodesicMaternKernel(nu=1.5)
            return ScaleKernel(kernel)
        else:
            # Fall back to standard kernels with normalization
            from study.gp_ablation.surrogates.standard_gp import create_kernel
            return create_kernel(self.kernel_type, self.D, self.device, use_msr_prior=False)

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model with geodesic kernel."""
        covar_module = self._create_kernel().to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                outcome_transform=Standardize(m=1),
                # No input normalization - we handle normalization ourselves
            )

        # Initialize lengthscale if kernel supports it
        # Note: Check has_lengthscale attribute, not hasattr(), because GPyTorch
        # kernels always have a lengthscale property that throws RuntimeError if not supported
        if (
            hasattr(covar_module, "base_kernel")
            and getattr(covar_module.base_kernel, "has_lengthscale", False)
        ):
            with torch.no_grad():
                covar_module.base_kernel.lengthscale = torch.full(
                    (1,), self.initial_lengthscale, device=self.device
                )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Normalize to unit sphere if configured
        train_X_norm = self._normalize(self._train_X)

        self.model = self._create_model(train_X_norm, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            X_norm = self._normalize(X)
            posterior = self.model.posterior(X_norm)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def get_hyperparameters(self) -> dict:
        """Get current GP hyperparameters.

        Overrides base class to handle kernels without lengthscale (e.g., ArcCosine).
        """
        self._ensure_fitted("getting hyperparameters")

        params = {}

        # Get lengthscale if kernel has one
        base_kernel = None
        if hasattr(self.model.covar_module, "base_kernel"):
            base_kernel = self.model.covar_module.base_kernel

        if base_kernel is not None and hasattr(base_kernel, "lengthscale") and base_kernel.lengthscale is not None:
            ls = base_kernel.lengthscale
            params["lengthscale_mean"] = ls.mean().item()
            params["lengthscale_std"] = ls.std().item() if ls.numel() > 1 else 0.0
        else:
            # Kernel without lengthscale (e.g., ArcCosine)
            params["lengthscale_mean"] = None
            params["lengthscale_std"] = None

        # Get outputscale
        if hasattr(self.model.covar_module, "outputscale"):
            params["outputscale"] = self.model.covar_module.outputscale.item()

        # Get noise
        if hasattr(self.model.likelihood, "noise"):
            params["noise_variance"] = self.model.likelihood.noise.item()

        return params


class GeodesicTuRBOGP(RiemannianGP):
    """TuRBO with geodesic trust region on the hypersphere.

    The trust region is a spherical cap centered on the best point,
    with geodesic radius adapted based on success/failure.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        # Trust region parameters (geodesic radius in radians)
        self.geodesic_radius = config.length_init * math.pi / 2  # Start at ~π/4
        self.radius_min = config.length_min * math.pi / 2
        self.radius_max = config.length_max * math.pi / 2

        self.success_tolerance = config.success_tolerance
        self.failure_tolerance = config.failure_tolerance

        self.success_counter = 0
        self.failure_counter = 0
        self.center: Optional[torch.Tensor] = None
        self.best_value = float("-inf")

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP and update trust region center."""
        super().fit(train_X, train_Y)

        # Update center to best point (normalized)
        best_idx = self._train_Y.argmax()
        self.center = self._normalize(self._train_X[best_idx:best_idx+1]).squeeze(0)
        self.best_value = self._train_Y[best_idx].item()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Update with geodesic trust region adaptation."""
        old_best = self.best_value

        super().update(new_X, new_Y)

        # Update trust region based on improvement
        if self.best_value > old_best + 1e-4 * abs(old_best):
            self.success_counter += 1
            self.failure_counter = 0
            if self.success_counter >= self.success_tolerance:
                self.geodesic_radius = min(self.geodesic_radius * 1.5, self.radius_max)
                self.success_counter = 0
                logger.info(f"Geodesic TR expanded to {self.geodesic_radius:.4f} rad")
        else:
            self.failure_counter += 1
            self.success_counter = 0
            if self.failure_counter >= self.failure_tolerance:
                self.geodesic_radius = max(self.geodesic_radius / 1.5, self.radius_min)
                self.failure_counter = 0
                logger.info(f"Geodesic TR shrunk to {self.geodesic_radius:.4f} rad")

    def _sample_spherical_cap(self, n_samples: int) -> torch.Tensor:
        """Sample uniformly from spherical cap around center.

        Uses the "rotate to pole" method for uniform sampling.
        """
        if self.center is None:
            # No center yet, sample uniformly on sphere
            samples = torch.randn(n_samples, self.D, device=self.device)
            return F.normalize(samples, p=2, dim=-1)

        # Sample from cap using tangent space + exponential map
        # 1. Sample in tangent space (normal to center)
        tangent = torch.randn(n_samples, self.D, device=self.device)

        # 2. Project out component along center
        center = self.center.unsqueeze(0)  # [1, D]
        proj = (tangent * center).sum(dim=-1, keepdim=True) * center
        tangent = tangent - proj

        # 3. Normalize tangent vectors
        tangent = F.normalize(tangent, p=2, dim=-1)

        # 4. Sample geodesic distances uniformly in [0, geodesic_radius]
        # For uniform sampling on cap, use p(θ) ∝ sin^{d-2}(θ)
        # Simplified: uniform in cos range
        cos_radius = math.cos(self.geodesic_radius)
        cos_samples = torch.rand(n_samples, 1, device=self.device) * (1 - cos_radius) + cos_radius
        sin_samples = torch.sqrt(1 - cos_samples ** 2)

        # 5. Exponential map: x = cos(θ) * center + sin(θ) * tangent
        samples = cos_samples * center + sin_samples * tangent

        return F.normalize(samples, p=2, dim=-1)

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest points within geodesic trust region."""
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        # Sample candidates from spherical cap
        candidates = self._sample_spherical_cap(n_samples)

        # Evaluate acquisition function
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:n_candidates]
        return candidates[top_indices]
