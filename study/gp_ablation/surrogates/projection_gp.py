"""Projection-based GP surrogates for high-dimensional optimization.

Implements:
- BAxUSGP: Random linear subspace projection (Papenmeier et al. 2022)

References:
- Papenmeier et al. (2022) "Increasing the Scope as You Learn: Adaptive
  Bayesian Optimization in Nested Subspaces"
"""

import math
import warnings
from typing import Optional, Tuple

import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate
from study.gp_ablation.surrogates.standard_gp import create_kernel


class BAxUSGP(BaseGPSurrogate):
    """BAxUS-style GP using random linear subspace projection.

    Projects inputs from D-dimensional space to target_dim subspace
    using a sparse random matrix. This reduces computational cost
    and can improve performance in high-D by focusing on relevant directions.

    Key hyperparameters:
        target_dim: Target dimensionality for projection (default 128)
        seed: Random seed for projection matrix

    The projection matrix S is created deterministically from the seed
    for reproducibility.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.target_dim = config.target_dim
        self.initial_lengthscale = math.sqrt(self.target_dim) / 10

        # Create deterministic projection matrix from seed
        torch.manual_seed(config.seed)
        self.S = self._create_embedding_matrix()
        self._train_X_embedded: Optional[torch.Tensor] = None

    def _create_embedding_matrix(self) -> torch.Tensor:
        """Create sparse random embedding matrix S: D -> target_dim.

        Uses a structured sparse projection where each original dimension
        maps to exactly one target dimension with random sign.
        """
        S = torch.zeros(self.target_dim, self.D, device=self.device)

        for i in range(self.D):
            j = i % self.target_dim
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            S[j, i] = sign

        # Normalize for variance preservation
        return S / math.sqrt(self.D / self.target_dim)

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """Project X from D-dimensional space to target_dim subspace."""
        return X @ self.S.T

    def _transform_for_posterior(self, X: torch.Tensor) -> torch.Tensor:
        """Project to subspace for posterior computation."""
        return self._embed(self._prepare_input(X))

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model in projected space."""
        covar_module = create_kernel(
            self.config.kernel,
            self.target_dim,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.target_dim),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.target_dim,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data (projects to subspace first)."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Project to subspace
        self._train_X_embedded = self._embed(self._train_X)

        self.model = self._create_model(self._train_X_embedded, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and standard deviation."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            X_embedded = self._embed(X)
            posterior = self.model.posterior(X_embedded)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the random embedding matrix S."""
        return self.S

    def lift_to_original(self, X_embedded: torch.Tensor) -> torch.Tensor:
        """Lift embedded points back to original space.

        Uses pseudo-inverse: X = X_embedded @ S (since S is nearly orthogonal).

        Args:
            X_embedded: Points in embedded space [N, target_dim].

        Returns:
            Points in original space [N, D].
        """
        return X_embedded @ self.S


class PCAGP(BaseGPSurrogate):
    """GP in PCA-reduced space.

    Unlike BAxUS, uses data-dependent PCA projection learned from
    the training data. This can be more effective when the data
    lies on a low-dimensional manifold.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.target_dim = config.target_dim
        self.initial_lengthscale = math.sqrt(self.target_dim) / 10

        # PCA components (learned from data)
        self._pca_components: Optional[torch.Tensor] = None
        self._pca_mean: Optional[torch.Tensor] = None
        self._train_X_embedded: Optional[torch.Tensor] = None

    def _learn_pca(self, X: torch.Tensor) -> None:
        """Learn PCA transformation from data."""
        # Center data
        self._pca_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self._pca_mean

        # SVD for PCA
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # Keep top target_dim components
        self._pca_components = Vh[:self.target_dim]  # [target_dim, D]

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """Project X to PCA space."""
        if self._pca_components is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        X_centered = X - self._pca_mean
        return X_centered @ self._pca_components.T

    def _transform_for_posterior(self, X: torch.Tensor) -> torch.Tensor:
        """Project to PCA space for posterior computation."""
        return self._embed(self._prepare_input(X))

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create GP model in PCA space."""
        covar_module = create_kernel(
            self.config.kernel,
            self.target_dim,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.target_dim),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.target_dim,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data (learns PCA and projects first)."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Learn PCA from training data
        self._learn_pca(self._train_X)

        # Project to PCA space
        self._train_X_embedded = self._embed(self._train_X)

        self.model = self._create_model(self._train_X_embedded, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and standard deviation."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            X_embedded = self._embed(X)
            posterior = self.model.posterior(X_embedded)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def get_explained_variance_ratio(self) -> float:
        """Get fraction of variance explained by PCA components."""
        if self._train_X is None or self._pca_components is None:
            return 0.0

        X_centered = self._train_X - self._pca_mean
        total_var = (X_centered ** 2).sum()

        X_projected = self._embed(self._train_X)
        X_reconstructed = X_projected @ self._pca_components
        reconstruction_error = ((X_centered - X_reconstructed) ** 2).sum()

        explained = 1.0 - (reconstruction_error / total_var).item()
        return explained
