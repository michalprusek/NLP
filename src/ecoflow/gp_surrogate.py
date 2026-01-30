"""GP surrogates for 1024D SONAR embedding optimization.

This module implements Gaussian Process surrogate models optimized for
high-dimensional SONAR embedding space (1024D).

Available surrogates:
1. SonarGPSurrogate: Standard GP with MSR initialization (ICLR 2025)
2. BAxUSGPSurrogate: BAxUS-style random subspace projection (NeurIPS 2022)

Key features:
- Matern-5/2 kernel with ARD lengthscales
- Dimension-scaled lengthscale initialization (sqrt(D)/10 = 3.2)
- LogNormal prior for lengthscales
- LCB gradient computation via autograd
- Incremental update support

BAxUS achieves 100% gradient improvement rate by projecting to lower
dimensional subspace where GP fitting is more stable.

References:
- MSR Method: Hvarfner et al. (2024) "Vanilla Bayesian Optimization Performs
  Great in High Dimensions" ICLR 2025
- BAxUS: Papenmeier et al. (2022) "Increasing the Scope as You Learn:
  Adaptive Bayesian Optimization in Nested Subspaces" NeurIPS 2022
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

    def ucb_gradient(
        self, X: torch.Tensor, alpha: float = 1.96
    ) -> torch.Tensor:
        """
        Compute gradient of UCB (Upper Confidence Bound) acquisition function.

        UCB(x) = mu(x) + alpha * sigma(x)  (for MAXIMIZATION)

        Args:
            X: Input points [B, D]
            alpha: Exploration weight (default 1.96 for 95% CI)

        Returns:
            grad_ucb: Gradient of UCB w.r.t. X [B, D]
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before computing gradients. Call fit() first.")

        self.model.eval()

        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)

            posterior = self.model.posterior(X_var)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

            ucb = mean + alpha * std  # + for maximization

            grad = torch.autograd.grad(
                ucb.sum(),
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


class BAxUSGPSurrogate:
    """
    BAxUS-style GP surrogate using random linear subspace projection.

    Projects 1024D SONAR embeddings to lower dimension (default 128D) using
    sparse random matrix, fits GP in subspace, then projects gradients back
    to full space via chain rule.

    This achieves 100% gradient improvement rate in benchmarks while being
    significantly faster than SAASBO (no MCMC sampling required).

    The key insight is that random projections preserve pairwise distances
    (Johnson-Lindenstrauss lemma), so gradients computed in the subspace
    are meaningful when projected back to full space.

    Attributes:
        D: Input dimensionality (default 1024 for SONAR)
        target_dim: Subspace dimensionality (default 128)
        device: Torch device for computation
        S: Sparse random embedding matrix [target_dim, D]

    Example:
        >>> gp = BAxUSGPSurrogate(D=1024, target_dim=128, device='cuda')
        >>> gp.fit(train_X, train_Y)  # [N, 1024], [N]
        >>> mean, std = gp.predict(test_X)  # [M, 1024] -> [M], [M]
        >>> grad = gp.lcb_gradient(X, alpha=1.0)  # [B, 1024] -> [B, 1024]

    Reference:
        Papenmeier et al. (2022) "Increasing the Scope as You Learn:
        Adaptive Bayesian Optimization in Nested Subspaces" NeurIPS 2022
    """

    def __init__(
        self,
        D: int = 1024,
        target_dim: int = 128,
        device: torch.device | str = "cuda",
        seed: Optional[int] = None,
    ):
        """
        Initialize BAxUS GP surrogate.

        Args:
            D: Input dimensionality (1024 for SONAR embeddings)
            target_dim: Subspace dimensionality (default 128, sweet spot from benchmarks)
            device: Device for computation ('cuda' or 'cpu')
            seed: Random seed for embedding matrix (for reproducibility)
        """
        self.D = D
        self.target_dim = target_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Create sparse random embedding matrix
        if seed is not None:
            torch.manual_seed(seed)
        self.S = self._create_embedding_matrix()

        self.model: Optional[SingleTaskGP] = None
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None
        self._train_X_embedded: Optional[torch.Tensor] = None

        # Initial lengthscale for subspace (MSR method scaled for target_dim)
        self.initial_lengthscale = math.sqrt(target_dim) / 10

    def _create_embedding_matrix(self) -> torch.Tensor:
        """
        Create sparse random embedding matrix S: D -> target_dim.

        Uses HeSBO-style construction where each input dimension maps to
        exactly one target dimension with sign +1 or -1.

        Returns:
            S: Embedding matrix [target_dim, D]
        """
        S = torch.zeros(self.target_dim, self.D, device=self.device)

        for i in range(self.D):
            j = i % self.target_dim  # Assign to bin (round-robin)
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            S[j, i] = sign

        # Normalize for variance preservation
        # Each target dimension receives D/target_dim input dimensions
        S = S / math.sqrt(self.D / self.target_dim)

        return S

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project X from D-dimensional space to target_dim subspace.

        Args:
            X: Input tensor [N, D]

        Returns:
            X_embedded: Projected tensor [N, target_dim]
        """
        return X @ self.S.T  # [N, D] @ [D, target_dim] -> [N, target_dim]

    def _create_model(
        self, train_X_embedded: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """
        Create GP model in the subspace.

        Args:
            train_X_embedded: Training inputs in subspace [N, target_dim]
            train_Y: Training targets [N, 1]

        Returns:
            Configured SingleTaskGP model
        """
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.target_dim,
                lengthscale_prior=LogNormalPrior(
                    loc=math.sqrt(2) + 0.5 * math.log(self.target_dim),
                    scale=math.sqrt(3),
                ),
            )
        )

        model = SingleTaskGP(
            train_X=train_X_embedded,
            train_Y=train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=self.target_dim),
            outcome_transform=Standardize(m=1),
        )

        # Initialize lengthscales (MSR method for subspace dimension)
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.target_dim,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """
        Fit GP to training data.

        Projects data to subspace before fitting.

        Args:
            train_X: Training inputs [N, D]
            train_Y: Training targets [N] or [N, 1]
        """
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Project to subspace
        self._train_X_embedded = self._embed(self._train_X)

        # Fit GP in subspace
        self.model = self._create_model(self._train_X_embedded, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.model.eval()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """
        Incrementally update GP with new observations.

        Args:
            new_X: New inputs [M, D]
            new_Y: New targets [M] or [M, 1]
        """
        if self._train_X is None or self._train_Y is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)

        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        # Concatenate with existing data
        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        # Re-project and re-fit
        self._train_X_embedded = self._embed(self._train_X)
        self.model = self._create_model(self._train_X_embedded, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.model.eval()

    def predict(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get posterior mean and standard deviation.

        Args:
            X: Test inputs [M, D] (in original space)

        Returns:
            mean: Posterior mean [M]
            std: Posterior standard deviation [M]
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before prediction. Call fit() first.")

        self.model.eval()

        # Project to subspace for prediction
        X_embedded = self._embed(X.to(self.device))

        with torch.no_grad():
            posterior = self.model.posterior(X_embedded)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

        return mean, std

    def lcb_gradient(
        self, X: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Compute gradient of LCB in the original D-dimensional space.

        The gradient is computed via chain rule:
        d(LCB)/dX = d(LCB)/dX_embedded @ dX_embedded/dX

        Since X_embedded = X @ S^T, we have dX_embedded/dX = S^T,
        and autograd handles this automatically.

        Args:
            X: Input points [B, D]
            alpha: Exploration weight (default 1.0)

        Returns:
            grad_lcb: Gradient of LCB w.r.t. X [B, D]
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before computing gradients. Call fit() first.")

        self.model.eval()

        with torch.enable_grad():
            # Gradient w.r.t. original X (chain rule through embedding)
            X_var = X.clone().to(self.device).requires_grad_(True)
            X_embedded = self._embed(X_var)

            posterior = self.model.posterior(X_embedded)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

            lcb = mean - alpha * std

            grad = torch.autograd.grad(
                lcb.sum(),
                X_var,
                create_graph=False,
                retain_graph=False,
            )[0]

        return grad

    def ucb_gradient(
        self, X: torch.Tensor, alpha: float = 1.96
    ) -> torch.Tensor:
        """
        Compute gradient of UCB in the original D-dimensional space.

        UCB = mean + alpha * std (for MAXIMIZATION problems)

        Args:
            X: Input points [B, D]
            alpha: Exploration weight (default 1.96 for 95% CI)

        Returns:
            grad_ucb: Gradient of UCB w.r.t. X [B, D]
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before computing gradients. Call fit() first.")

        self.model.eval()

        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)
            X_embedded = self._embed(X_var)

            posterior = self.model.posterior(X_embedded)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

            ucb = mean + alpha * std  # + for maximization

            grad = torch.autograd.grad(
                ucb.sum(),
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
        """Get current training data (X, Y) in original space."""
        return self._train_X, self._train_Y

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the random embedding matrix S."""
        return self.S


class HeteroscedasticSonarGP:
    """
    GP surrogate with heteroscedastic binomial noise for accuracy evaluation.

    When evaluating prompts on GSM8K (or similar tasks), the observed accuracy
    is a binomial proportion p_hat = k/n where k is correct answers out of n
    questions. The variance of this estimate is:

        Var(p_hat) = p(1-p) / n

    This variance is HETEROSCEDASTIC: it's highest at p=0.5 (maximum uncertainty)
    and lowest near p=0 or p=1 (highly confident predictions).

    This class uses SingleTaskGP with train_Yvar (fixed observation noise) to
    properly model this noise structure, rather than learning a homoscedastic
    noise term that overestimates uncertainty at extreme accuracies.

    Attributes:
        D: Input dimensionality (default 1024 for SONAR)
        n_eval: Number of evaluation questions (used in variance formula)
        device: Torch device for computation
        model: The underlying BoTorch SingleTaskGP model

    Example:
        >>> gp = HeteroscedasticSonarGP(D=1024, n_eval=150, device='cuda')
        >>> gp.fit(train_X, train_Y)  # train_Y is accuracy in [0, 1]
        >>> mean, std = gp.predict(test_X)
        >>> grad = gp.ucb_gradient(X, alpha=1.96)
    """

    def __init__(
        self,
        D: int = 1024,
        n_eval: int = 150,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize heteroscedastic GP surrogate.

        Args:
            D: Input dimensionality (1024 for SONAR embeddings)
            n_eval: Number of questions per evaluation (default 150)
            device: Device for computation ('cuda' or 'cpu')
        """
        self.D = D
        self.n_eval = n_eval
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model: Optional[SingleTaskGP] = None
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None

        # Initial lengthscale: sqrt(D)/10 (MSR method)
        self.initial_lengthscale = math.sqrt(D) / 10

    def _compute_variance(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute binomial variance for accuracy values.

        Var(p_hat) = p(1-p) / n

        Args:
            Y: Accuracy values in [0, 1], shape [N] or [N, 1]

        Returns:
            variance: Variance for each observation [N] or [N, 1]
        """
        # Flatten if needed
        Y_flat = Y.squeeze(-1) if Y.dim() > 1 else Y

        # Clamp p away from 0 and 1 for numerical stability
        p = Y_flat.clamp(0.01, 0.99)

        # Binomial variance: p(1-p)/n
        var = (p * (1 - p)) / self.n_eval

        # Add minimum variance floor for numerical stability
        var = var.clamp(min=1e-6)

        # Match original shape
        if Y.dim() > 1:
            return var.unsqueeze(-1)
        return var

    def _create_model(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor,
    ) -> SingleTaskGP:
        """
        Create GP model with heteroscedastic (fixed) observation noise.

        Args:
            train_X: Training inputs [N, D]
            train_Y: Training targets [N, 1]
            train_Yvar: Observation noise variance [N, 1]

        Returns:
            Configured SingleTaskGP model with fixed noise
        """
        # Matern-5/2 with ARD and dimension-scaled prior
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

        # SingleTaskGP with train_Yvar for heteroscedastic fixed noise
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,  # Fixed heteroscedastic noise
            covar_module=covar_module,
            input_transform=Normalize(d=self.D),
            outcome_transform=Standardize(m=1),
        )

        # Initialize lengthscales to sqrt(D)/10 = 3.2
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """
        Fit GP to training data with binomial noise model.

        Args:
            train_X: Training inputs [N, D]
            train_Y: Training targets (accuracy in [0,1]) [N] or [N, 1]
        """
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        # Ensure Y has shape [N, 1] for SingleTaskGP
        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Compute heteroscedastic noise variance
        train_Yvar = self._compute_variance(self._train_Y)

        self.model = self._create_model(self._train_X, self._train_Y, train_Yvar)

        # Fit using exact marginal log-likelihood
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.model.eval()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """
        Incrementally update GP with new observations.

        Args:
            new_X: New inputs [M, D]
            new_Y: New targets (accuracy in [0,1]) [M] or [M, 1]
        """
        if self._train_X is None or self._train_Y is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)

        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        # Concatenate with existing data
        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        # Compute heteroscedastic noise variance for all data
        train_Yvar = self._compute_variance(self._train_Y)

        # Re-fit
        self.model = self._create_model(self._train_X, self._train_Y, train_Yvar)

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
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before prediction. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X.to(self.device))
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

        return mean, std

    def lcb_gradient(
        self, X: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Compute gradient of LCB (Lower Confidence Bound) acquisition function.

        LCB(x) = mu(x) - alpha * sigma(x)

        Args:
            X: Input points [B, D]
            alpha: Exploration weight (default 1.0)

        Returns:
            grad_lcb: Gradient of LCB w.r.t. X [B, D]
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before computing gradients. Call fit() first.")

        self.model.eval()

        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)
            posterior = self.model.posterior(X_var)

            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

            lcb = mean - alpha * std

            grad = torch.autograd.grad(
                lcb.sum(),
                X_var,
                create_graph=False,
                retain_graph=False,
            )[0]

        return grad

    def ucb_gradient(
        self, X: torch.Tensor, alpha: float = 1.96
    ) -> torch.Tensor:
        """
        Compute gradient of UCB (Upper Confidence Bound) acquisition function.

        UCB(x) = mu(x) + alpha * sigma(x)

        Args:
            X: Input points [B, D]
            alpha: Exploration weight (default 1.96 for 95% CI)

        Returns:
            grad_ucb: Gradient of UCB w.r.t. X [B, D]
        """
        if self.model is None:
            raise RuntimeError("GP must be fitted before computing gradients. Call fit() first.")

        self.model.eval()

        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)
            posterior = self.model.posterior(X_var)

            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance + 1e-6)

            ucb = mean + alpha * std

            grad = torch.autograd.grad(
                ucb.sum(),
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


def create_surrogate(
    method: str = "msr",
    D: int = 1024,
    device: torch.device | str = "cuda",
    **kwargs,
) -> SonarGPSurrogate | BAxUSGPSurrogate | HeteroscedasticSonarGP:
    """
    Factory function to create a GP surrogate.

    Args:
        method: Surrogate method to use:
            - "msr": Standard GP with MSR initialization (default)
            - "baxus": BAxUS random subspace projection (recommended for guidance)
            - "heteroscedastic": Heteroscedastic GP with binomial noise for accuracy
        D: Input dimensionality (default 1024 for SONAR)
        device: Device for computation
        **kwargs: Additional arguments passed to surrogate constructor
            - target_dim: For baxus method, subspace dimension (default 128)
            - seed: For baxus method, random seed for embedding matrix
            - n_eval: For heteroscedastic method, number of eval questions (default 150)

    Returns:
        GP surrogate instance

    Example:
        >>> gp = create_surrogate("baxus", D=1024, target_dim=128)
        >>> gp.fit(X, y)
        >>> grad = gp.lcb_gradient(X_new)

        >>> # Heteroscedastic GP for accuracy optimization
        >>> gp = create_surrogate("heteroscedastic", D=1024, n_eval=150)
        >>> gp.fit(X, accuracy)  # accuracy in [0, 1]
    """
    method = method.lower()

    if method == "msr" or method == "standard":
        return SonarGPSurrogate(D=D, device=device)
    elif method == "baxus":
        target_dim = kwargs.get("target_dim", 128)
        seed = kwargs.get("seed", None)
        return BAxUSGPSurrogate(D=D, target_dim=target_dim, device=device, seed=seed)
    elif method == "heteroscedastic":
        n_eval = kwargs.get("n_eval", 150)
        return HeteroscedasticSonarGP(D=D, n_eval=n_eval, device=device)
    else:
        raise ValueError(
            f"Unknown surrogate method: {method}. "
            "Use 'msr', 'baxus', or 'heteroscedastic'."
        )
