"""GP surrogate models for spherical and Euclidean spaces.

Extracted from V2's _fit_gp() and VanillaBO's _fit_gp().
Consolidates the duplicated GP fitting + fallback logic.
"""

from __future__ import annotations

import logging

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from rielbo.core.config import KernelConfig
from rielbo.gp_diagnostics import GPDiagnostics
from rielbo.kernels import create_kernel

logger = logging.getLogger(__name__)


class SphericalGPSurrogate:
    """GP surrogate for data on S^(d-1) with spherical kernels.

    Handles ArcCosine, GeodesicMatern, product sphere, and PCA modes.
    Includes automatic fallback on numerical failures.
    """

    def __init__(
        self,
        kernel_config: KernelConfig,
        subspace_dim: int,
        device: str = "cuda",
        verbose: bool = True,
        pca_mode: bool = False,
    ):
        self.kernel_config = kernel_config
        self.subspace_dim = subspace_dim
        self.device = device
        self.verbose = verbose
        self.pca_mode = pca_mode

        self.gp: SingleTaskGP | None = None
        self.likelihood = None
        self._last_mll_value: float | None = None
        self.fallback_count = 0
        self.gp_diagnostics = GPDiagnostics(verbose=verbose)
        self.diagnostic_history: list[dict] = []

    @property
    def last_mll_value(self) -> float | None:
        return self._last_mll_value

    def _create_kernel(self, dim: int | None = None):
        """Create covariance kernel from config."""
        cfg = self.kernel_config
        d = dim or self.subspace_dim

        if cfg.product_space:
            return create_kernel(
                kernel_type="product",
                kernel_order=cfg.kernel_order,
                n_spheres=cfg.n_spheres,
                use_scale=True,
            )
        else:
            ard_num_dims = d if cfg.kernel_ard else None
            return create_kernel(
                kernel_type=cfg.kernel_type,
                kernel_order=cfg.kernel_order,
                use_scale=True,
                ard_num_dims=ard_num_dims,
            )

    def fit(
        self, X: torch.Tensor, Y: torch.Tensor, iteration: int = 0,
    ) -> None:
        """Fit GP on subspace data.

        Args:
            X: Training inputs [N, d] (on S^(d-1) for spherical, R^d for PCA)
            Y: Training targets [N]
            iteration: Current iteration for diagnostic logging
        """
        X_gp = X.double()
        Y_gp = Y.double().unsqueeze(-1)

        try:
            if self.pca_mode:
                pca_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
                )
                self.gp = SingleTaskGP(
                    X_gp, Y_gp,
                    likelihood=pca_likelihood,
                    input_transform=Normalize(d=X_gp.shape[-1]),
                ).to(self.device)
            else:
                covar_module = self._create_kernel(dim=X.shape[-1])
                self.gp = SingleTaskGP(
                    X_gp, Y_gp, covar_module=covar_module,
                ).to(self.device)

            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            # Store log marginal likelihood
            try:
                with torch.no_grad():
                    output = self.gp(X_gp)
                    self._last_mll_value = mll(output, Y_gp.squeeze(-1)).item()
            except Exception:
                self._last_mll_value = None

            # Diagnostics
            if self.verbose:
                metrics = self.gp_diagnostics.analyze(
                    self.gp, X_gp.float(), Y_gp.squeeze(-1).float(),
                )
                self.gp_diagnostics.log_summary(
                    metrics, prefix=f"[Iter {iteration}]",
                )
                self.diagnostic_history.append(
                    self.gp_diagnostics.get_summary_dict(metrics),
                )

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            self.fallback_count += 1
            logger.error(
                f"GP fit failed (fallback #{self.fallback_count}): {e}"
            )
            self.gp = SingleTaskGP(
                X_gp, Y_gp,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2),
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def fit_quick(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Fit GP quickly and return log marginal likelihood (for LASS scoring).

        Returns float("-inf") on failure.
        """
        X_gp = X.double()
        Y_gp = Y.double().unsqueeze(-1)

        try:
            covar_module = self._create_kernel(dim=X.shape[-1])
            gp = SingleTaskGP(X_gp, Y_gp, covar_module=covar_module).to(self.device)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            with torch.no_grad():
                output = gp(X_gp)
                return mll(output, Y_gp.squeeze(-1)).item()
        except Exception:
            return float("-inf")


class EuclideanGPSurrogate:
    """GP surrogate for full-dimensional Euclidean data with [0,1]^D normalization.

    Used by VanillaBO and TuRBO. Uses BoTorch default Hvarfner priors.
    """

    def __init__(
        self,
        kernel_config: KernelConfig,
        input_dim: int,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.kernel_config = kernel_config
        self.input_dim = input_dim
        self.device = device
        self.verbose = verbose

        self.gp: SingleTaskGP | None = None
        self._last_mll_value: float | None = None
        self.fallback_count = 0
        self.gp_diagnostics = GPDiagnostics(verbose=verbose)
        self.diagnostic_history: list[dict] = []

        # Normalization bounds (set from cold start)
        self._z_min: torch.Tensor | None = None
        self._z_max: torch.Tensor | None = None

        # Z-score normalization (for TuRBO)
        self._z_mean: torch.Tensor | None = None
        self._z_std: torch.Tensor | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    @property
    def last_mll_value(self) -> float | None:
        return self._last_mll_value

    def set_normalization_bounds(self, Z: torch.Tensor) -> None:
        """Set [0,1]^D normalization bounds from data (for Hvarfner/VanillaBO)."""
        self._z_min = Z.min(dim=0).values
        self._z_max = Z.max(dim=0).values
        margin = (self._z_max - self._z_min) * 0.05
        self._z_min = self._z_min - margin
        self._z_max = self._z_max + margin
        zero_range = (self._z_max - self._z_min).abs() < 1e-8
        self._z_max[zero_range] = self._z_min[zero_range] + 1.0

    def update_normalization_bounds(self, z_new: torch.Tensor) -> None:
        """Expand bounds to cover new data."""
        if self._z_min is not None:
            z = z_new.squeeze()
            margin = (self._z_max - self._z_min) * 0.05
            self._z_min = torch.min(self._z_min, z - margin)
            self._z_max = torch.max(self._z_max, z + margin)

    def to_unit_cube(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]^D."""
        return (z - self._z_min) / (self._z_max - self._z_min)

    def from_unit_cube(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]^D."""
        return z_norm * (self._z_max - self._z_min) + self._z_min

    def normalize_z(self, z: torch.Tensor, train_Z: torch.Tensor) -> torch.Tensor:
        """Z-score normalize (for TuRBO)."""
        self._z_mean = train_Z.mean(dim=0)
        self._z_std = train_Z.std(dim=0).clamp(min=1e-6)
        return (z - self._z_mean) / self._z_std

    def denormalize_z(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize from z-score."""
        return z_norm * self._z_std + self._z_mean

    def fit(
        self,
        Z: torch.Tensor,
        Y: torch.Tensor,
        iteration: int = 0,
        mode: str = "hvarfner",
    ) -> None:
        """Fit GP on Euclidean data.

        Args:
            Z: Training inputs [N, D] (raw or pre-normalized)
            Y: Training targets [N]
            iteration: For diagnostic logging
            mode: "hvarfner" (BoTorch defaults) or "turbo" (z-score + Matern)
        """
        if mode == "hvarfner":
            Z_norm = self.to_unit_cube(Z).double()
            Y_gp = Y.double().unsqueeze(-1)
        else:
            # TuRBO mode: z-score + manual Y norm
            Z_norm = self.normalize_z(Z, Z).double()
            self._y_mean = Y.mean().item()
            self._y_std = Y.std().clamp(min=1e-6).item()
            Y_gp = ((Y - self._y_mean) / self._y_std).double().unsqueeze(-1)

        try:
            if mode == "hvarfner":
                self.gp = SingleTaskGP(Z_norm, Y_gp).to(self.device)
            else:
                self.gp = SingleTaskGP(
                    Z_norm, Y_gp,
                    covar_module=ScaleKernel(
                        MaternKernel(nu=2.5, ard_num_dims=self.input_dim),
                    ),
                ).to(self.device)

            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            if self.verbose and iteration % 10 == 0:
                Y_diag = Y_gp.squeeze(-1).float()
                if mode == "hvarfner" and hasattr(self.gp, "outcome_transform"):
                    try:
                        Y_diag = self.gp.outcome_transform(Y_gp)[0].squeeze(-1).float()
                    except Exception:
                        pass
                metrics = self.gp_diagnostics.analyze(
                    self.gp, Z_norm.float(), Y_diag,
                )
                self.gp_diagnostics.log_summary(
                    metrics, prefix=f"[Iter {iteration}]",
                )
                self.diagnostic_history.append(
                    self.gp_diagnostics.get_summary_dict(metrics),
                )

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            self.fallback_count += 1
            logger.error(f"GP fit failed (fallback #{self.fallback_count}): {e}")
            self.gp = SingleTaskGP(
                Z_norm, Y_gp,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2),
                ),
            ).to(self.device)
            self.gp.likelihood.noise = 0.1
            self.gp.eval()
