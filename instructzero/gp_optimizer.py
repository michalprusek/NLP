"""
Gaussian Process Bayesian Optimization for soft prompt space.

Uses BoTorch for GP modeling and acquisition function optimization.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GPBayesianOptimizer:
    """
    Bayesian Optimizer using Gaussian Process surrogate.

    Uses BoTorch's SingleTaskGP and Expected Improvement acquisition.
    Optimized for low-dimensional soft prompt space (10D).
    """

    def __init__(
        self,
        bounds: Tuple[torch.Tensor, torch.Tensor],
        device: str = "cuda",
        n_candidates: int = 512,
        n_restarts: int = 10,
    ):
        """
        Initialize GP-based Bayesian optimizer.

        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) for each dimension
            device: Torch device
            n_candidates: Number of candidates for acquisition optimization
            n_restarts: Number of restarts for acquisition optimization
        """
        self.lb, self.ub = bounds
        self.lb = self.lb.to(device)
        self.ub = self.ub.to(device)
        self.device = device
        self.n_candidates = n_candidates
        self.n_restarts = n_restarts

        self.dim = len(self.lb)
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        self.model = None
        self.best_value = -float("inf")

    def _normalize_X(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize X to [0, 1]^d for GP."""
        return (X - self.lb) / (self.ub - self.lb)

    def _unnormalize_X(self, X_norm: torch.Tensor) -> torch.Tensor:
        """Unnormalize X from [0, 1]^d."""
        return X_norm * (self.ub - self.lb) + self.lb

    def _standardize_Y(self, Y: torch.Tensor) -> torch.Tensor:
        """Standardize Y for GP (zero mean, unit variance)."""
        if Y.std() < 1e-6:
            return Y - Y.mean()
        return (Y - Y.mean()) / Y.std()

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Update GP model with new observations.

        Args:
            X: Input points, shape (n, dim)
            Y: Objective values, shape (n,) or (n, 1)
        """
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X = X.to(self.device)
        Y = Y.to(self.device)

        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)

        if self.train_X is None:
            self.train_X = X
            self.train_Y = Y
        else:
            self.train_X = torch.cat([self.train_X, X], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y], dim=0)

        self.best_value = self.train_Y.max().item()

        # Normalize inputs and standardize outputs for GP
        X_norm = self._normalize_X(self.train_X)
        Y_std = self._standardize_Y(self.train_Y)

        # Fit GP model
        self.model = SingleTaskGP(X_norm.double(), Y_std.double())
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            logger.warning(f"GP fitting warning: {e}")
            # Continue anyway, model should still work

        logger.debug(f"GP updated with {len(self.train_X)} points, best={self.best_value:.4f}")

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate using Expected Improvement.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of suggested points, shape (n_suggestions, dim)
        """
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf

        if self.model is None:
            # No model yet, return random points
            return self._sample_random(n_suggestions)

        # Use LogExpectedImprovement (numerically stable)
        # Best value in standardized space
        Y_std = self._standardize_Y(self.train_Y)
        best_f = Y_std.max().item()

        ei = LogExpectedImprovement(
            model=self.model,
            best_f=best_f,
        )

        # Optimize acquisition function
        # Bounds are [0, 1]^d after normalization
        bounds_norm = torch.stack([
            torch.zeros(self.dim, device=self.device, dtype=torch.double),
            torch.ones(self.dim, device=self.device, dtype=torch.double)
        ])

        suggestions = []
        for _ in range(n_suggestions):
            try:
                candidate, _ = optimize_acqf(
                    acq_function=ei,
                    bounds=bounds_norm,
                    q=1,
                    num_restarts=self.n_restarts,
                    raw_samples=self.n_candidates,
                )
                suggestions.append(candidate.squeeze(0))
            except Exception as e:
                logger.warning(f"Acquisition optimization failed: {e}, using random")
                suggestions.append(self._sample_random(1).squeeze(0).double())

        X_suggested = torch.stack(suggestions)

        # Unnormalize back to original space
        X_suggested = self._unnormalize_X(X_suggested.float())

        return X_suggested

    def suggest_diverse(self, n_suggestions: int = 4) -> torch.Tensor:
        """
        Suggest diverse points using a mix of EI and random sampling.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of suggested points, shape (n_suggestions, dim)
        """
        if self.model is None or n_suggestions <= 2:
            return self.suggest(n_suggestions)

        # Mix of EI suggestions and random exploration
        n_ei = max(1, n_suggestions // 2)
        n_random = n_suggestions - n_ei

        ei_suggestions = self.suggest(n_ei)
        random_suggestions = self._sample_random(n_random)

        return torch.cat([ei_suggestions, random_suggestions], dim=0)

    def _sample_random(self, n_samples: int) -> torch.Tensor:
        """Sample random points within bounds."""
        return torch.empty(
            n_samples, self.dim, device=self.device
        ).uniform_(-1, 1) * (self.ub - self.lb) / 2 + (self.ub + self.lb) / 2

    def get_best(self) -> Tuple[torch.Tensor, float]:
        """
        Get the best point found so far.

        Returns:
            Tuple of (best_X, best_Y)
        """
        if self.train_X is None:
            raise ValueError("No data yet")

        best_idx = self.train_Y.argmax()
        return self.train_X[best_idx], self.train_Y[best_idx].item()


class TurboGPOptimizer(GPBayesianOptimizer):
    """
    Trust Region Bayesian Optimization (TuRBO) variant.

    Maintains a trust region that expands on success and shrinks on failure.
    Better for higher-dimensional problems.
    """

    def __init__(
        self,
        bounds: Tuple[torch.Tensor, torch.Tensor],
        device: str = "cuda",
        n_candidates: int = 512,
        initial_length: float = 0.8,
        length_min: float = 0.01,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int = 5,
    ):
        super().__init__(bounds, device, n_candidates)

        self.length = initial_length
        self.length_min = length_min
        self.length_max = length_max
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance

        self.success_counter = 0
        self.failure_counter = 0
        self.center: Optional[torch.Tensor] = None

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Update with trust region adaptation."""
        old_best = self.best_value

        super().update(X, Y)

        # Update trust region based on improvement
        if self.best_value > old_best + 1e-4 * abs(old_best):
            self.success_counter += 1
            self.failure_counter = 0
            if self.success_counter >= self.success_tolerance:
                self.length = min(self.length * 2.0, self.length_max)
                self.success_counter = 0
                logger.info(f"TR expanded to {self.length:.4f}")
        else:
            self.failure_counter += len(Y) if Y.dim() > 1 else 1
            self.success_counter = 0
            if self.failure_counter >= self.failure_tolerance:
                self.length = max(self.length / 2.0, self.length_min)
                self.failure_counter = 0
                logger.info(f"TR shrunk to {self.length:.4f}")

        # Update center to best point
        best_idx = self.train_Y.argmax()
        self.center = self.train_X[best_idx].clone()

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """Suggest points within trust region."""
        from botorch.acquisition import LogExpectedImprovement
        from torch.quasirandom import SobolEngine

        if self.model is None or self.center is None:
            return self._sample_random(n_suggestions)

        # Trust region bounds around center
        tr_lb = torch.clamp(self.center - self.length * (self.ub - self.lb) / 2, self.lb, self.ub)
        tr_ub = torch.clamp(self.center + self.length * (self.ub - self.lb) / 2, self.lb, self.ub)

        # Generate Sobol candidates within trust region
        sobol = SobolEngine(self.dim, scramble=True)
        candidates = sobol.draw(self.n_candidates).to(self.device)
        candidates = tr_lb + candidates * (tr_ub - tr_lb)

        # Normalize for GP
        candidates_norm = self._normalize_X(candidates).double()

        # Evaluate LogEI on candidates (numerically stable)
        Y_std = self._standardize_Y(self.train_Y)
        best_f = Y_std.max().item()

        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates_norm.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:n_suggestions]
        return candidates[top_indices]
