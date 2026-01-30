"""GP surrogate for 1024D SONAR embedding optimization.

This module implements a Gaussian Process surrogate model optimized for
high-dimensional SONAR embedding space (1024D). It uses dimension-scaled
initialization following the MSR method (ICLR 2025) to enable effective
GP fitting in high dimensions.

Key features:
- Matern-5/2 kernel with ARD lengthscales
- Dimension-scaled lengthscale initialization (sqrt(D)/10 = 3.2)
- LogNormal prior for lengthscales
- LCB gradient computation via autograd
- Incremental update support
"""

import math
from typing import Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior


class SonarGPSurrogate:
    """
    GP surrogate model for optimization in 1024D SONAR embedding space.

    Uses Matern-5/2 kernel with ARD (Automatic Relevance Determination) and
    dimension-scaled initialization following the MSR method. This enables
    effective GP fitting in high-dimensional spaces where default initialization
    would cause MLE to fail.

    Attributes:
        D: Input dimensionality (default 1024 for SONAR)
        device: Torch device for computation
        model: The underlying BoTorch SingleTaskGP model
        initial_lengthscale: sqrt(D)/10 initialization value

    Example:
        >>> gp = SonarGPSurrogate(D=1024, device='cuda')
        >>> gp.fit(train_X, train_Y)  # [N, 1024], [N]
        >>> mean, std = gp.predict(test_X)  # [M, 1024] -> [M], [M]
        >>> grad = gp.lcb_gradient(X, alpha=1.0)  # [B, 1024] -> [B, 1024]
    """

    def __init__(
        self,
        D: int = 1024,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize GP surrogate.

        Args:
            D: Input dimensionality (1024 for SONAR embeddings)
            device: Device for computation ('cuda' or 'cpu')
        """
        self.D = D
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model: Optional[SingleTaskGP] = None
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None

        # Initial lengthscale: sqrt(D)/10 (MSR method)
        # For D=1024, this is ~3.2
        self.initial_lengthscale = math.sqrt(D) / 10

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """
        Create GP model with proper high-dimensional initialization.

        Args:
            train_X: Training inputs [N, D]
            train_Y: Training targets [N, 1]

        Returns:
            Configured SingleTaskGP model
        """
        # Matern-5/2 with ARD and dimension-scaled prior
        # Prior: LogNormal(loc=sqrt(2)+0.5*log(D), scale=sqrt(3))
        # For D=1024: loc ~ 4.0, encouraging longer lengthscales
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.D,
                lengthscale_prior=LogNormalPrior(
                    loc=math.sqrt(2) + 0.5 * math.log(self.D),
                    scale=math.sqrt(3),
                ),
            )
        )

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=self.D),
            outcome_transform=Standardize(m=1),
        )

        # Critical: Initialize lengthscales to sqrt(D)/10 = 3.2
        # This prevents vanishing gradients during MLE fitting
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """
        Fit GP to training data.

        Args:
            train_X: Training inputs [N, D]
            train_Y: Training targets [N] or [N, 1]

        Note:
            Targets are automatically reshaped to [N, 1] if needed.
            Uses L-BFGS optimizer via BoTorch's fit_gpytorch_mll.
        """
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        # Ensure Y has shape [N, 1] for SingleTaskGP
        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y)

        # Fit using exact marginal log-likelihood
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.model.eval()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """
        Incrementally update GP with new observations.

        Concatenates new data with existing training data and re-fits
        the GP from scratch. For large datasets, consider using sparse
        GP approximations.

        Args:
            new_X: New inputs [M, D]
            new_Y: New targets [M] or [M, 1]
        """
        if self._train_X is None or self._train_Y is None:
            # No existing data, just fit
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)

        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        # Concatenate with existing data
        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        # Re-fit (could use warm-starting for efficiency in future)
        self.model = self._create_model(self._train_X, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.model.eval()

    def predict(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get posterior mean and standard deviation.

        Args:
            X: Test inputs [M, D]

        Returns:
            mean: Posterior mean [M]
            std: Posterior standard deviation [M]

        Raises:
            RuntimeError: If model has not been fitted
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before prediction. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X.to(self.device))
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            # Numerical stability: add small epsilon before sqrt
            std = torch.sqrt(variance + 1e-6)

        return mean, std

    def lcb_gradient(
        self, X: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Compute gradient of LCB (Lower Confidence Bound) acquisition function.

        LCB(x) = mu(x) - alpha * sigma(x)

        For maximization problems, follow +grad(LCB) to find high values
        with exploration bonus. For minimization, follow -grad(LCB).

        Args:
            X: Input points [B, D]
            alpha: Exploration weight (default 1.0, use 1.96 for 95% CI)

        Returns:
            grad_lcb: Gradient of LCB w.r.t. X [B, D]

        Note:
            Must use torch.enable_grad() because BoTorch uses no_grad internally.
            Input is cloned and requires_grad_(True) is called to enable autograd.
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before computing gradients. Call fit() first.")

        self.model.eval()

        # Must enable grad explicitly (BoTorch uses no_grad internally)
        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)

            # Get posterior
            posterior = self.model.posterior(X_var)

            # LCB = mean - alpha * std
            mean = posterior.mean.squeeze(-1)  # [B]
            variance = posterior.variance.squeeze(-1)  # [B]
            std = torch.sqrt(variance + 1e-6)  # Numerical stability

            lcb = mean - alpha * std

            # Compute gradient: sum over batch, get [B, D] gradient
            grad = torch.autograd.grad(
                lcb.sum(),
                X_var,
                create_graph=False,
                retain_graph=False,
            )[0]

        return grad

    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return 0 if self._train_X is None else self._train_X.shape[0]

    def get_training_data(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current training data (X, Y)."""
        return self._train_X, self._train_Y
