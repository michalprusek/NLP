"""GP surrogates for 1024D SONAR embedding optimization.

Available surrogates:
1. SonarGPSurrogate: Standard GP with MSR initialization (ICLR 2025)
2. BAxUSGPSurrogate: BAxUS-style random subspace projection (NeurIPS 2022)
3. HeteroscedasticSonarGP: Binomial noise model for accuracy evaluation

References:
- MSR: Hvarfner et al. (2024) "Vanilla Bayesian Optimization in High Dimensions"
- BAxUS: Papenmeier et al. (2022) "Adaptive Bayesian Optimization in Nested Subspaces"
"""

import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import gpytorch
import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior


class BaseGPSurrogate(ABC):
    """Base class for GP surrogate models in high-dimensional embedding space."""

    def __init__(self, D: int, device: torch.device | str):
        self.D = D
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model: Optional[SingleTaskGP] = None
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None

    @abstractmethod
    def _create_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
        """Create and configure the GP model."""
        pass

    @abstractmethod
    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        pass

    def _fit_model(self) -> None:
        """Fit the model using MLL optimization."""
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def _ensure_fitted(self, operation: str = "operation") -> None:
        """Raise error if model is not fitted."""
        if self.model is None:
            raise RuntimeError(f"GP must be fitted before {operation}. Call fit() first.")

    def _prepare_input(self, X: torch.Tensor) -> torch.Tensor:
        """Move input tensor to device."""
        return X.to(self.device)

    def _transform_for_posterior(self, X: torch.Tensor) -> torch.Tensor:
        """Transform input for posterior computation. Override for subspace projection."""
        return self._prepare_input(X)

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and standard deviation."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X_transformed = self._transform_for_posterior(X)
            posterior = self.model.posterior(X_transformed)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def _acquisition_gradient(
        self, X: torch.Tensor, alpha: float, sign: float
    ) -> torch.Tensor:
        """Compute gradient of acquisition function.

        Args:
            X: Input points [B, D]
            alpha: Exploration weight
            sign: +1 for UCB (maximization), -1 for LCB (minimization)
        """
        self._ensure_fitted("computing gradients")
        self.model.eval()

        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)
            X_transformed = self._transform_for_posterior(X_var)

            posterior = self.model.posterior(X_transformed)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

            acquisition = mean + sign * alpha * std

            grad = torch.autograd.grad(
                acquisition.sum(),
                X_var,
                create_graph=False,
                retain_graph=False,
            )[0]

        return grad

    def lcb_gradient(self, X: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Compute gradient of LCB: mu(x) - alpha * sigma(x)."""
        return self._acquisition_gradient(X, alpha, sign=-1.0)

    def ucb_gradient(self, X: torch.Tensor, alpha: float = 1.96) -> torch.Tensor:
        """Compute gradient of UCB: mu(x) + alpha * sigma(x)."""
        return self._acquisition_gradient(X, alpha, sign=+1.0)

    def sample_thompson(
        self, X: torch.Tensor, n_samples: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thompson Sampling: draw posterior samples and return argmax indices."""
        self._ensure_fitted("sampling")
        self.model.eval()

        X_transformed = self._transform_for_posterior(X)

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
            posterior = self.model.posterior(X_transformed)
            samples = posterior.rsample(torch.Size([n_samples])).squeeze(-1)
            best_indices = samples.argmax(dim=-1)

        return samples, best_indices

    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return 0 if self._train_X is None else self._train_X.shape[0]

    def get_training_data(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current training data (X, Y)."""
        return self._train_X, self._train_Y

    def optimize_ucb(
        self,
        alpha: float = 1.96,
        n_restarts: int = 5,
        n_steps: int = 100,
        lr: float = 0.1,
    ) -> tuple[torch.Tensor, float]:
        """Find optimal point by gradient ascent on UCB."""
        self._ensure_fitted("optimization")
        self.model.eval()

        train_X, train_Y = self.get_training_data()
        if train_X is None or train_Y is None:
            raise RuntimeError("No training data available for optimization.")

        start_points = self._get_optimization_start_points(train_X, train_Y, n_restarts)

        best_z = None
        best_ucb = float("-inf")

        for z_init in start_points:
            z = z_init.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([z], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                ucb = self._compute_ucb(z, alpha)
                (-ucb.sum()).backward()
                optimizer.step()

            with torch.no_grad():
                final_ucb = self._compute_ucb(z, alpha).item()

            if final_ucb > best_ucb:
                best_ucb = final_ucb
                best_z = z.detach().clone()

        return best_z, best_ucb

    def _compute_ucb(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        """Compute UCB value for a point."""
        z_transformed = self._transform_for_posterior(z)
        posterior = self.model.posterior(z_transformed)
        mean = posterior.mean.squeeze(-1)
        std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)
        return mean + alpha * std

    def _get_optimization_start_points(
        self, train_X: torch.Tensor, train_Y: torch.Tensor, n_restarts: int
    ) -> list[torch.Tensor]:
        """Get starting points for UCB optimization."""
        n_from_data = min(n_restarts - 1, len(train_Y))
        top_indices = train_Y.squeeze(-1).argsort(descending=True)[:n_from_data]
        start_points = [train_X[idx : idx + 1].clone() for idx in top_indices]

        n_random = n_restarts - len(start_points)
        if n_random > 0:
            data_mean = train_X.mean(dim=0, keepdim=True)
            data_std = train_X.std(dim=0, keepdim=True)
            for _ in range(n_random):
                start_points.append(data_mean + torch.randn_like(data_mean) * data_std)

        return start_points


def _create_matern_covar(
    dim: int, device: torch.device, on_device_prior: bool = False
) -> ScaleKernel:
    """Create Matern-5/2 kernel with ARD and MSR-style prior."""
    if on_device_prior:
        prior = LogNormalPrior(
            loc=torch.tensor(math.sqrt(2) + 0.5 * math.log(dim), device=device),
            scale=torch.tensor(math.sqrt(3), device=device),
        )
    else:
        prior = LogNormalPrior(
            loc=math.sqrt(2) + 0.5 * math.log(dim),
            scale=math.sqrt(3),
        )

    return ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_prior=prior)
    )


class SonarGPSurrogate(BaseGPSurrogate):
    """GP surrogate with MSR initialization for 1024D SONAR embeddings."""

    def __init__(self, D: int = 1024, device: torch.device | str = "cuda"):
        super().__init__(D, device)
        self.initial_lengthscale = math.sqrt(D) / 10

    def _create_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
        covar_module = _create_matern_covar(self.D, self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y)
        self._fit_model()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Incrementally update GP with new observations."""
        if self._train_X is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        self.model = self._create_model(self._train_X, self._train_Y)
        self._fit_model()


class BAxUSGPSurrogate(BaseGPSurrogate):
    """BAxUS-style GP using random linear subspace projection."""

    def __init__(
        self,
        D: int = 1024,
        target_dim: int = 128,
        device: torch.device | str = "cuda",
        seed: Optional[int] = None,
    ):
        super().__init__(D, device)
        self.target_dim = target_dim
        self.initial_lengthscale = math.sqrt(target_dim) / 10

        if seed is not None:
            torch.manual_seed(seed)
        self.S = self._create_embedding_matrix()
        self._train_X_embedded: Optional[torch.Tensor] = None

    def _create_embedding_matrix(self) -> torch.Tensor:
        """Create sparse random embedding matrix S: D -> target_dim."""
        S = torch.zeros(self.target_dim, self.D, device=self.device)

        for i in range(self.D):
            j = i % self.target_dim
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            S[j, i] = sign

        return S / math.sqrt(self.D / self.target_dim)

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """Project X from D-dimensional space to target_dim subspace."""
        return X @ self.S.T

    def _transform_for_posterior(self, X: torch.Tensor) -> torch.Tensor:
        """Project to subspace for posterior computation."""
        return self._embed(self._prepare_input(X))

    def _create_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
        covar_module = _create_matern_covar(
            self.target_dim, self.device, on_device_prior=True
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

        self._train_X_embedded = self._embed(self._train_X)
        self.model = self._create_model(self._train_X_embedded, self._train_Y)
        self._fit_model()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Incrementally update GP with new observations."""
        if self._train_X is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        self._train_X_embedded = self._embed(self._train_X)
        self.model = self._create_model(self._train_X_embedded, self._train_Y)
        self._fit_model()

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the random embedding matrix S."""
        return self.S


class HeteroscedasticSonarGP(BaseGPSurrogate):
    """GP with heteroscedastic binomial noise for accuracy evaluation.

    Uses Var(p_hat) = p(1-p)/n for binomial proportion observations.
    """

    def __init__(
        self,
        D: int = 1024,
        n_eval: int = 150,
        device: torch.device | str = "cuda",
    ):
        super().__init__(D, device)
        self.n_eval = n_eval
        self.initial_lengthscale = math.sqrt(D) / 10

    def _compute_variance(self, Y: torch.Tensor) -> torch.Tensor:
        """Compute binomial variance: p(1-p)/n."""
        Y_flat = Y.squeeze(-1) if Y.dim() > 1 else Y
        p = Y_flat.clamp(0.01, 0.99)
        var = (p * (1 - p)) / self.n_eval
        var = var.clamp(min=1e-6)
        return var.unsqueeze(-1) if Y.dim() > 1 else var

    def _create_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
        raise NotImplementedError("Use _create_model_with_variance instead")

    def _create_model_with_variance(
        self, train_X: torch.Tensor, train_Y: torch.Tensor, train_Yvar: torch.Tensor
    ) -> SingleTaskGP:
        covar_module = _create_matern_covar(self.D, self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                outcome_transform=Standardize(m=1),
            )

        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.full(
                (self.D,), self.initial_lengthscale, device=self.device
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP with binomial noise model."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        train_Yvar = self._compute_variance(self._train_Y)
        self.model = self._create_model_with_variance(
            self._train_X, self._train_Y, train_Yvar
        )
        self._fit_model()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Incrementally update GP with new observations."""
        if self._train_X is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        train_Yvar = self._compute_variance(self._train_Y)
        self.model = self._create_model_with_variance(
            self._train_X, self._train_Y, train_Yvar
        )
        self._fit_model()


def create_surrogate(
    method: str = "msr",
    D: int = 1024,
    device: torch.device | str = "cuda",
    **kwargs,
) -> SonarGPSurrogate | BAxUSGPSurrogate | HeteroscedasticSonarGP:
    """Factory function to create a GP surrogate.

    Args:
        method: "msr" (standard), "baxus" (subspace), or "heteroscedastic" (binomial noise)
        D: Input dimensionality (default 1024 for SONAR)
        device: Device for computation
        **kwargs: target_dim/seed for baxus, n_eval for heteroscedastic
    """
    method = method.lower()

    if method in ("msr", "standard"):
        return SonarGPSurrogate(D=D, device=device)
    elif method == "baxus":
        return BAxUSGPSurrogate(
            D=D,
            target_dim=kwargs.get("target_dim", 128),
            device=device,
            seed=kwargs.get("seed"),
        )
    elif method == "heteroscedastic":
        return HeteroscedasticSonarGP(
            D=D, n_eval=kwargs.get("n_eval", 150), device=device
        )
    else:
        raise ValueError(
            f"Unknown surrogate method: {method}. "
            "Use 'msr', 'baxus', or 'heteroscedastic'."
        )
