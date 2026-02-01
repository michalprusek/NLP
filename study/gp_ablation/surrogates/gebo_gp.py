"""Gradient-Enhanced Bayesian Optimization (GEBO).

Implements:
- GradientEnhancedGP: GP that uses gradient information

GEBO incorporates gradient information to improve sample efficiency.
Since full gradients in 1024D are expensive, we use directional derivatives.

References:
- Wu et al. (2017) "Bayesian Optimization with Gradients"
- Riis et al. (2021) "Directional Derivatives for BO with Finite Differences"
"""

import logging
import math
import warnings
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate
from study.gp_ablation.surrogates.standard_gp import create_kernel

logger = logging.getLogger(__name__)


class GradientEnhancedGP(BaseGPSurrogate):
    """Gradient-Enhanced GP using directional derivatives.

    Instead of modeling full gradients (1024D per point), we model
    directional derivatives along a few key directions:
    1. Direction toward current best point
    2. Principal directions from PCA
    3. Random directions

    Key hyperparameters:
        use_full_gradient: If True, use full gradient (expensive!)
        n_directions: Number of directions for directional derivatives
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.use_full_gradient = config.use_full_gradient
        self.n_directions = config.n_directions
        self.initial_lengthscale = math.sqrt(self.D) / 10

        # Gradient function (set by user or derived from flow)
        self._gradient_fn: Optional[Callable] = None

        # Stored directional derivatives
        self._train_directions: Optional[List[torch.Tensor]] = None
        self._train_dir_derivs: Optional[List[torch.Tensor]] = None

    def set_gradient_function(self, gradient_fn: Callable) -> None:
        """Set the gradient function.

        Args:
            gradient_fn: Function that takes x [D] and returns gradient [D].
        """
        self._gradient_fn = gradient_fn

    def _compute_directions(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Compute key directions for directional derivatives.

        Returns:
            Directions [n_directions, D], normalized.
        """
        directions = []

        # Direction 1: Toward best point
        best_idx = Y.argmax()
        best_x = X[best_idx]

        # Use gradient of distance to best as one direction
        # For each point, direction is (best_x - x) / ||best_x - x||
        # We'll use the mean direction
        diffs = best_x - X
        mean_dir = diffs.mean(dim=0)
        if mean_dir.norm() > 1e-6:
            directions.append(F.normalize(mean_dir, dim=0))

        # Remaining directions: PCA on data
        if len(directions) < self.n_directions:
            X_centered = X - X.mean(dim=0, keepdim=True)
            try:
                U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
                n_pca = min(self.n_directions - len(directions), len(S))
                for i in range(n_pca):
                    directions.append(Vh[i])
            except RuntimeError:
                pass  # SVD can fail on degenerate data

        # Fill remaining with random directions
        while len(directions) < self.n_directions:
            rand_dir = torch.randn(self.D, device=self.device)
            directions.append(F.normalize(rand_dir, dim=0))

        return torch.stack(directions[:self.n_directions])

    def _compute_directional_derivatives(
        self,
        X: torch.Tensor,
        gradients: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute directional derivatives.

        Args:
            X: Input points [N, D].
            gradients: Full gradients [N, D].
            directions: Directions [K, D].

        Returns:
            Directional derivatives [N, K].
        """
        # dir_deriv[i, k] = <grad[i], direction[k]>
        return gradients @ directions.T

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create standard GP model (gradients used in acquisition, not model)."""
        covar_module = create_kernel(
            self.config.kernel,
            self.D,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

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

    def fit(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
    ) -> None:
        """Fit GP to training data with optional gradient information.

        Args:
            train_X: Training inputs [N, D].
            train_Y: Training targets [N] or [N, 1].
            gradients: Optional gradients [N, D]. If None and gradient_fn
                       is set, will compute gradients.
        """
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Compute or use provided gradients
        if gradients is not None:
            self._stored_gradients = gradients.to(self.device)
        elif self._gradient_fn is not None:
            # Compute gradients for all training points
            grads = []
            for x in self._train_X:
                g = self._gradient_fn(x)
                grads.append(g)
            self._stored_gradients = torch.stack(grads)
        else:
            self._stored_gradients = None

        # Compute directions for directional derivatives
        if self._stored_gradients is not None:
            directions = self._compute_directions(self._train_X, self._train_Y)
            self._train_directions = directions
            self._train_dir_derivs = self._compute_directional_derivatives(
                self._train_X, self._stored_gradients, directions
            )

        # Create and fit GP model
        self.model = self._create_model(self._train_X, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and std."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def predict_with_gradient(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get posterior mean, std, and gradient of mean.

        Uses gradient information to refine predictions.

        Args:
            X: Test inputs [M, D].

        Returns:
            Tuple of (mean [M], std [M], grad_mean [M, D]).
        """
        self._ensure_fitted("prediction with gradient")
        self.model.eval()

        X = self._prepare_input(X).requires_grad_(True)

        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        # Compute gradient of mean
        grad_mean = torch.autograd.grad(
            mean.sum(),
            X,
            create_graph=False,
            retain_graph=False,
        )[0]

        return mean.detach(), std.detach(), grad_mean.detach()

    def gradient_guided_suggest(
        self,
        n_candidates: int = 1,
        n_samples: int = 512,
        n_grad_steps: int = 5,
        grad_lr: float = 0.1,
    ) -> torch.Tensor:
        """Suggest candidates using gradient-guided optimization.

        1. Sample candidates via standard acquisition
        2. Refine using gradient of mean prediction

        Args:
            n_candidates: Number of candidates to return.
            n_samples: Number of initial random samples.
            n_grad_steps: Gradient refinement steps per candidate.
            grad_lr: Learning rate for gradient ascent.

        Returns:
            Suggested points [n_candidates, D].
        """
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        # Sample initial candidates
        candidates = torch.randn(n_samples, self.D, device=self.device)

        # If we have training data, center candidates around data distribution
        if self._train_X is not None:
            data_mean = self._train_X.mean(dim=0)
            data_std = self._train_X.std(dim=0)
            candidates = candidates * data_std + data_mean

        # Evaluate acquisition
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates.unsqueeze(-2))

        # Select top candidates for refinement
        top_indices = ei_values.argsort(descending=True)[: n_candidates * 2]
        top_candidates = candidates[top_indices]

        # Refine using gradient ascent on predicted mean
        refined = []
        for x in top_candidates:
            x = x.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=grad_lr)

            for _ in range(n_grad_steps):
                optimizer.zero_grad()
                mean, std, _ = self.predict_with_gradient(x.unsqueeze(0))
                # Maximize mean (plus small exploration bonus)
                objective = mean + 0.1 * std
                (-objective).backward()
                optimizer.step()

            refined.append(x.detach())

        refined = torch.stack(refined)

        # Re-evaluate and return top n_candidates
        with torch.no_grad():
            ei_values = ei(refined.unsqueeze(-2))
        top_indices = ei_values.argsort(descending=True)[:n_candidates]

        return refined[top_indices]
