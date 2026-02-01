"""TuRBO GP implementation for trust region Bayesian optimization.

Implements:
- TuRBOGP: Trust Region Bayesian Optimization (Eriksson et al. 2019)
- TuRBOGradientGP: TuRBO with gradient refinement of candidates

References:
- Eriksson et al. (2019) "Scalable Global Optimization via Local Bayesian Optimization"
"""

import logging
import math
import warnings
from typing import Optional, Tuple

import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate
from study.gp_ablation.surrogates.standard_gp import create_kernel

logger = logging.getLogger(__name__)


class TuRBOGP(BaseGPSurrogate):
    """Trust Region Bayesian Optimization.

    Maintains a trust region that expands on success and shrinks on failure.
    The trust region is centered on the best point found so far.

    Key hyperparameters:
        length_init: Initial trust region length (default 0.8)
        length_min: Minimum length before restart (default 0.01)
        length_max: Maximum length (default 1.6)
        success_tolerance: Successes before expansion (default 3)
        failure_tolerance: Failures before shrinking (default 5)
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        # Trust region parameters from config
        self.length = config.length_init
        self.length_min = config.length_min
        self.length_max = config.length_max
        self.success_tolerance = config.success_tolerance
        self.failure_tolerance = config.failure_tolerance

        # State
        self.success_counter = 0
        self.failure_counter = 0
        self.center: Optional[torch.Tensor] = None
        self.best_value = float("-inf")

        # Initial lengthscale for GP
        self.initial_lengthscale = math.sqrt(self.D) / 10

    def _create_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """Create and configure the GP model."""
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

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit GP to training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

        # Update center and best value
        best_idx = self._train_Y.argmax()
        self.center = self._train_X[best_idx].clone()
        self.best_value = self._train_Y[best_idx].item()

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Update with trust region adaptation."""
        old_best = self.best_value

        # Call parent update (which calls fit)
        super().update(new_X, new_Y)

        # Trust region adaptation
        new_Y = new_Y.to(self.device)
        if new_Y.dim() > 1:
            new_Y = new_Y.squeeze(-1)

        # Check if we improved
        n_new = len(new_Y)
        if self.best_value > old_best + 1e-4 * abs(old_best):
            self.success_counter += n_new
            self.failure_counter = 0
            if self.success_counter >= self.success_tolerance:
                self.length = min(self.length * 2.0, self.length_max)
                self.success_counter = 0
                logger.info(f"TuRBO: Trust region expanded to {self.length:.4f}")
        else:
            self.failure_counter += n_new
            self.success_counter = 0
            if self.failure_counter >= self.failure_tolerance:
                self.length = max(self.length / 2.0, self.length_min)
                self.failure_counter = 0
                logger.info(f"TuRBO: Trust region shrunk to {self.length:.4f}")

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and standard deviation."""
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest points within trust region.

        Args:
            n_candidates: Number of candidates to return.
            bounds: Optional global bounds (lb, ub).
            n_samples: Number of Sobol samples for candidate generation.

        Returns:
            Suggested points [n_candidates, D].
        """
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        if self.center is None:
            # No center yet, sample randomly
            return torch.randn(n_candidates, self.D, device=self.device)

        # Compute trust region bounds
        if bounds is not None:
            lb, ub = bounds[0].to(self.device), bounds[1].to(self.device)
        else:
            # Use data range as global bounds
            data_min = self._train_X.min(dim=0).values
            data_max = self._train_X.max(dim=0).values
            data_range = data_max - data_min
            lb = data_min - 0.1 * data_range
            ub = data_max + 0.1 * data_range

        # Trust region bounds around center
        half_length = self.length * (ub - lb) / 2
        tr_lb = torch.clamp(self.center - half_length, lb, ub)
        tr_ub = torch.clamp(self.center + half_length, lb, ub)

        # Generate Sobol candidates within trust region
        sobol = SobolEngine(self.D, scramble=True)
        unit_samples = sobol.draw(n_samples).to(self.device)
        candidates = tr_lb + unit_samples * (tr_ub - tr_lb)

        # Evaluate acquisition function
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:n_candidates]
        return candidates[top_indices]

    def get_trust_region_length(self) -> float:
        """Get current trust region length."""
        return self.length

    def should_restart(self) -> bool:
        """Check if trust region has collapsed (should restart)."""
        return self.length <= self.length_min


class TuRBOGradientGP(TuRBOGP):
    """TuRBO with gradient refinement of candidates.

    After TuRBO proposes a candidate, performs gradient descent steps
    to refine it using gradients from the SONAR decoder.

    This hybrid approach combines TuRBO's global exploration with
    local gradient-based optimization.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.n_grad_steps = config.n_grad_steps
        self.grad_lr = config.grad_lr
        self._gradient_fn = None  # Set by user

    def set_gradient_function(self, gradient_fn) -> None:
        """Set the gradient function for refinement.

        Args:
            gradient_fn: Function that takes embedding [D] and returns gradient [D].
        """
        self._gradient_fn = gradient_fn

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest points with gradient refinement.

        Args:
            n_candidates: Number of candidates to return.
            bounds: Optional global bounds (lb, ub).
            n_samples: Number of Sobol samples.

        Returns:
            Refined suggested points [n_candidates, D].
        """
        # Get TuRBO suggestions
        candidates = super().suggest(n_candidates, bounds, n_samples)

        # Refine with gradient descent if gradient function available
        if self._gradient_fn is None:
            logger.warning(
                "TuRBOGradientGP: No gradient function set, skipping refinement"
            )
            return candidates

        refined = []
        for i in range(n_candidates):
            x = candidates[i].clone().requires_grad_(True)

            for _ in range(self.n_grad_steps):
                # Get gradient from external function
                grad = self._gradient_fn(x.detach())

                # Gradient ascent (maximizing objective)
                with torch.no_grad():
                    x = x + self.grad_lr * grad

                # Project back to bounds if needed
                if bounds is not None:
                    lb, ub = bounds
                    x = torch.clamp(x, lb.to(self.device), ub.to(self.device))

            refined.append(x.detach())

        return torch.stack(refined)
