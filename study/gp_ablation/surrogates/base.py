"""Base class for GP surrogate models in high-dimensional embedding space.

Provides common interface for all GP surrogates used in the ablation study.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpytorch
import torch
from botorch.models import SingleTaskGP

from study.gp_ablation.config import GPConfig


class BaseGPSurrogate(ABC):
    """Abstract base class for GP surrogate models.

    All GP methods in the ablation study must implement this interface
    to ensure consistent evaluation and comparison.

    Attributes:
        config: GPConfig with method-specific hyperparameters.
        device: Torch device for computation.
        model: The fitted GP model (None until fit() is called).
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        """Initialize surrogate.

        Args:
            config: GP configuration with all hyperparameters.
            device: Device for computation.
        """
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model: Optional[SingleTaskGP] = None
        self._train_X: Optional[torch.Tensor] = None
        self._train_Y: Optional[torch.Tensor] = None

    @property
    def D(self) -> int:
        """Input dimensionality."""
        return self.config.input_dim

    @abstractmethod
    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data.

        Args:
            train_X: Training inputs [N, D].
            train_Y: Training targets [N] or [N, 1].
        """
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and standard deviation.

        Args:
            X: Test inputs [M, D].

        Returns:
            Tuple of (mean [M], std [M]).
        """
        pass

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Incrementally update GP with new observations.

        Default implementation refits from scratch. Subclasses may override
        for more efficient incremental updates.

        Args:
            new_X: New input points [B, D].
            new_Y: New target values [B] or [B, 1].
        """
        if self._train_X is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        self.fit(self._train_X, self._train_Y)

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Suggest next points to evaluate.

        Args:
            n_candidates: Number of candidates to suggest.
            bounds: Optional (lower, upper) bounds for optimization.

        Returns:
            Suggested points [n_candidates, D].
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement suggest(). "
            "Use acquisition function optimization externally."
        )

    def sample_posterior(
        self, X: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Draw samples from the posterior distribution.

        Args:
            X: Test inputs [M, D].
            n_samples: Number of posterior samples.

        Returns:
            Posterior samples [n_samples, M].
        """
        self._ensure_fitted("sampling")
        self.model.eval()

        X = self._prepare_input(X)
        X_transformed = self._transform_for_posterior(X)

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
            posterior = self.model.posterior(X_transformed)
            samples = posterior.rsample(torch.Size([n_samples])).squeeze(-1)

        return samples

    def thompson_sample(
        self, X: torch.Tensor, n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Thompson Sampling: draw posterior samples and return argmax.

        Args:
            X: Candidate points [M, D].
            n_samples: Number of posterior samples.

        Returns:
            Tuple of (samples [n_samples, M], best_indices [n_samples]).
        """
        samples = self.sample_posterior(X, n_samples)
        best_indices = samples.argmax(dim=-1)
        return samples, best_indices

    def compute_acquisition(
        self,
        X: torch.Tensor,
        acquisition: str = "log_ei",
        best_f: Optional[float] = None,
        alpha: float = 1.96,
    ) -> torch.Tensor:
        """Compute acquisition function values.

        Args:
            X: Candidate points [M, D].
            acquisition: Acquisition type (log_ei, ucb, lcb).
            best_f: Best observed value (required for EI).
            alpha: UCB/LCB exploration weight.

        Returns:
            Acquisition values [M].
        """
        self._ensure_fitted("computing acquisition")
        self.model.eval()

        X = self._prepare_input(X)
        X_transformed = self._transform_for_posterior(X)

        with torch.no_grad():
            posterior = self.model.posterior(X_transformed)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        if acquisition == "ucb":
            return mean + alpha * std
        elif acquisition == "lcb":
            return mean - alpha * std
        elif acquisition in ("ei", "log_ei"):
            if best_f is None:
                best_f = self._train_Y.max().item() if self._train_Y is not None else 0.0

            # Compute EI using closed form
            from botorch.acquisition.analytic import ExpectedImprovement
            from botorch.acquisition import LogExpectedImprovement

            if acquisition == "log_ei":
                ei = LogExpectedImprovement(model=self.model, best_f=best_f)
            else:
                ei = ExpectedImprovement(model=self.model, best_f=best_f)

            return ei(X_transformed.unsqueeze(-2)).squeeze(-1)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

    def acquisition_gradient(
        self,
        X: torch.Tensor,
        acquisition: str = "ucb",
        alpha: float = 1.96,
    ) -> torch.Tensor:
        """Compute gradient of acquisition function.

        Args:
            X: Input points [B, D].
            acquisition: Acquisition type (ucb, lcb).
            alpha: Exploration weight.

        Returns:
            Gradient tensor [B, D].
        """
        self._ensure_fitted("computing gradients")
        self.model.eval()

        sign = 1.0 if acquisition == "ucb" else -1.0

        with torch.enable_grad():
            X_var = X.clone().to(self.device).requires_grad_(True)
            X_transformed = self._transform_for_posterior(X_var)

            posterior = self.model.posterior(X_transformed)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

            acq_value = mean + sign * alpha * std

            grad = torch.autograd.grad(
                acq_value.sum(),
                X_var,
                create_graph=False,
                retain_graph=False,
            )[0]

        return grad

    def negative_log_predictive_density(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log predictive density (NLPD).

        Lower is better. Measures calibration of uncertainty estimates.

        Args:
            X: Test inputs [M, D].
            Y: Test targets [M].

        Returns:
            NLPD values [M].
        """
        mean, std = self.predict(X)
        Y = Y.to(self.device).squeeze()

        # Gaussian NLPD: 0.5 * (log(2*pi*sigma^2) + (y - mu)^2 / sigma^2)
        var = std**2 + 1e-6
        nlpd = 0.5 * (torch.log(2 * math.pi * var) + (Y - mean) ** 2 / var)

        return nlpd

    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return 0 if self._train_X is None else self._train_X.shape[0]

    def get_training_data(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current training data (X, Y)."""
        return self._train_X, self._train_Y

    def _ensure_fitted(self, operation: str = "operation") -> None:
        """Raise error if model is not fitted."""
        if self.model is None:
            raise RuntimeError(f"GP must be fitted before {operation}. Call fit() first.")

    def _prepare_input(self, X: torch.Tensor) -> torch.Tensor:
        """Move input tensor to device."""
        return X.to(self.device)

    def _transform_for_posterior(self, X: torch.Tensor) -> torch.Tensor:
        """Transform input for posterior computation.

        Override in subclasses for subspace projection or normalization.
        """
        return X

    def get_hyperparameters(self) -> dict:
        """Get current GP hyperparameters.

        Returns:
            Dict with lengthscale, outputscale, noise_variance.
        """
        self._ensure_fitted("getting hyperparameters")

        params = {}

        # Get lengthscale
        if hasattr(self.model.covar_module, "base_kernel"):
            ls = self.model.covar_module.base_kernel.lengthscale
        else:
            ls = self.model.covar_module.lengthscale
        params["lengthscale_mean"] = ls.mean().item()
        params["lengthscale_std"] = ls.std().item()

        # Get outputscale
        if hasattr(self.model.covar_module, "outputscale"):
            params["outputscale"] = self.model.covar_module.outputscale.item()

        # Get noise
        if hasattr(self.model.likelihood, "noise"):
            params["noise_variance"] = self.model.likelihood.noise.item()

        return params

    def optimize_ucb(
        self,
        alpha: float = 1.96,
        n_restarts: int = 5,
        n_steps: int = 100,
        lr: float = 0.1,
    ) -> Tuple[torch.Tensor, float]:
        """Find optimal point by gradient ascent on UCB.

        Args:
            alpha: UCB exploration weight (default 1.96 for 95% CI)
            n_restarts: Number of random restarts
            n_steps: Gradient steps per restart
            lr: Learning rate for Adam optimizer

        Returns:
            Tuple of (best_z [1, D], best_ucb_value)
        """
        self._ensure_fitted("optimization")
        self.model.eval()

        train_X, train_Y = self.get_training_data()
        if train_X is None or train_Y is None:
            raise RuntimeError("No training data available for optimization.")

        # Get start points from training data (top performers + random)
        start_points = self._get_optimization_start_points(train_X, train_Y, n_restarts)

        best_z = None
        best_ucb = float("-inf")

        for z_init in start_points:
            z = z_init.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([z], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                # Compute UCB = mean + alpha * std
                z_input = self._transform_for_posterior(z.unsqueeze(0))
                posterior = self.model.posterior(z_input)
                mean = posterior.mean.squeeze()
                std = torch.sqrt(posterior.variance.squeeze() + 1e-6)
                ucb = mean + alpha * std
                (-ucb).backward()  # Minimize negative UCB
                optimizer.step()

            with torch.no_grad():
                z_input = self._transform_for_posterior(z.unsqueeze(0))
                posterior = self.model.posterior(z_input)
                mean = posterior.mean.squeeze()
                std = torch.sqrt(posterior.variance.squeeze() + 1e-6)
                final_ucb = (mean + alpha * std).item()

            if final_ucb > best_ucb:
                best_ucb = final_ucb
                best_z = z.detach().clone()

        return best_z.unsqueeze(0), best_ucb

    def _get_optimization_start_points(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        n_restarts: int,
    ) -> list:
        """Get start points for UCB optimization.

        Uses top performers from training data plus random perturbations.
        """
        # Sort by score and take top points
        sorted_indices = train_Y.squeeze().argsort(descending=True)
        n_top = min(n_restarts // 2 + 1, len(sorted_indices))
        top_indices = sorted_indices[:n_top]

        start_points = []

        # Add top performers
        for idx in top_indices[:n_restarts]:
            start_points.append(train_X[idx].clone())

        # Add perturbed versions of best point
        best_idx = sorted_indices[0]
        best_x = train_X[best_idx]
        while len(start_points) < n_restarts:
            perturbed = best_x + 0.1 * torch.randn_like(best_x)
            start_points.append(perturbed)

        return start_points
