"""Curriculum Multi-Fidelity Bayesian Optimization.

Implements:
- CurriculumBO: BO with adaptive fidelity schedule

Starts with cheap, low-fidelity evaluations (subset of test set)
and gradually increases fidelity as the optimization progresses.
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

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate
from study.gp_ablation.surrogates.standard_gp import create_kernel

logger = logging.getLogger(__name__)


class CurriculumBO(BaseGPSurrogate):
    """Multi-fidelity BO with adaptive fidelity schedule.

    Key idea: Start with cheap evaluations (e.g., 10% of GSM8K test set),
    gradually increase fidelity as we approach the budget limit.

    Benefits:
    1. More candidates evaluated early (exploration)
    2. Accurate evaluation of promising candidates late (exploitation)
    3. GP noise model adapts to current fidelity

    Key hyperparameters:
        fidelity_start: Starting fidelity (e.g., 0.1 for 10%)
        fidelity_schedule: Schedule type ("linear", "exponential")
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.fidelity_start = config.fidelity_start
        self.fidelity_schedule = config.fidelity_schedule
        self.initial_lengthscale = math.sqrt(self.D) / 10

        # Current fidelity level
        self.current_fidelity = self.fidelity_start

        # Track iteration for schedule
        self._iteration = 0
        self._total_budget = config.n_iterations

        # Fidelity history for each observation
        self._train_fidelities: Optional[torch.Tensor] = None

    def set_fidelity(self, fidelity: float) -> None:
        """Manually set current fidelity level.

        Args:
            fidelity: Fidelity in [0, 1].
        """
        self.current_fidelity = min(max(fidelity, 0.01), 1.0)

    def step_fidelity(self) -> float:
        """Update fidelity based on schedule and return new value.

        Should be called at each BO iteration.

        Returns:
            New fidelity value.
        """
        self._iteration += 1
        progress = min(self._iteration / max(self._total_budget, 1), 1.0)

        if self.fidelity_schedule == "linear":
            # Linear: fidelity_start -> 1.0
            self.current_fidelity = self.fidelity_start + (
                1.0 - self.fidelity_start
            ) * progress
        elif self.fidelity_schedule == "exponential":
            # Exponential: stays low longer, then increases rapidly
            # f(t) = f_start + (1 - f_start) * (1 - exp(-3*t))
            self.current_fidelity = self.fidelity_start + (
                1.0 - self.fidelity_start
            ) * (1 - math.exp(-3 * progress))
        elif self.fidelity_schedule == "step":
            # Step: discrete jumps
            if progress < 0.33:
                self.current_fidelity = 0.1
            elif progress < 0.66:
                self.current_fidelity = 0.5
            else:
                self.current_fidelity = 1.0
        else:
            # Default to linear
            self.current_fidelity = self.fidelity_start + (
                1.0 - self.fidelity_start
            ) * progress

        logger.info(
            f"CurriculumBO: iteration {self._iteration}/{self._total_budget}, "
            f"fidelity={self.current_fidelity:.2f}"
        )

        return self.current_fidelity

    def _compute_noise_variance(self, fidelity: float, y: float) -> float:
        """Compute observation noise variance based on fidelity.

        Lower fidelity = higher noise (more uncertainty in measurement).

        Uses binomial variance model: Var = p(1-p)/(n*fidelity)
        """
        # Assume n=1000 full evaluations (GSM8K test set size)
        n_full = 1000
        n_samples = int(n_full * fidelity)
        n_samples = max(n_samples, 10)  # Minimum 10 samples

        # Binomial variance
        p = min(max(y, 0.01), 0.99)  # Clamp accuracy
        var = (p * (1 - p)) / n_samples

        return max(var, 1e-6)

    def _create_model(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor,
    ) -> SingleTaskGP:
        """Create GP with heteroscedastic noise based on fidelity."""
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
        """Fit GP with fidelity-aware noise.

        Initial fit assumes all observations are at current fidelity.
        """
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        # Assume all initial data is at current fidelity
        n = len(self._train_Y)
        self._train_fidelities = torch.full(
            (n,), self.current_fidelity, device=self.device
        )

        # Compute variance for each observation
        train_Yvar = torch.tensor(
            [
                self._compute_noise_variance(
                    self._train_fidelities[i].item(),
                    self._train_Y[i].item(),
                )
                for i in range(n)
            ],
            device=self.device,
        ).unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y, train_Yvar)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def update(
        self,
        new_X: torch.Tensor,
        new_Y: torch.Tensor,
        fidelity: Optional[float] = None,
    ) -> None:
        """Update GP with new observations at specified fidelity.

        Args:
            new_X: New input points [B, D].
            new_Y: New target values [B] or [B, 1].
            fidelity: Fidelity level for new observations.
                      If None, uses current_fidelity.
        """
        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() == 1:
            new_Y = new_Y.unsqueeze(-1)

        fidelity = fidelity if fidelity is not None else self.current_fidelity

        if self._train_X is None:
            self._train_X = new_X
            self._train_Y = new_Y
            self._train_fidelities = torch.full(
                (len(new_Y),), fidelity, device=self.device
            )
        else:
            self._train_X = torch.cat([self._train_X, new_X], dim=0)
            self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

            new_fids = torch.full((len(new_Y),), fidelity, device=self.device)
            self._train_fidelities = torch.cat(
                [self._train_fidelities, new_fids], dim=0
            )

        # Recompute variances
        n = len(self._train_Y)
        train_Yvar = torch.tensor(
            [
                self._compute_noise_variance(
                    self._train_fidelities[i].item(),
                    self._train_Y[i].item(),
                )
                for i in range(n)
            ],
            device=self.device,
        ).unsqueeze(-1)

        self.model = self._create_model(self._train_X, self._train_Y, train_Yvar)

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

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest candidates.

        At low fidelity, suggest more candidates (they're cheap).
        At high fidelity, focus on fewer promising candidates.
        """
        from botorch.acquisition import LogExpectedImprovement

        self._ensure_fitted("suggestion")

        # Adjust number of samples based on fidelity
        # Low fidelity = evaluate more candidates
        effective_n = int(n_candidates * (2.0 - self.current_fidelity))
        effective_n = max(effective_n, n_candidates)

        # Sample candidates
        if self._train_X is not None:
            mean = self._train_X.mean(dim=0)
            std = self._train_X.std(dim=0)
            candidates = mean + std * torch.randn(n_samples, self.D, device=self.device)
        else:
            candidates = torch.randn(n_samples, self.D, device=self.device)

        # Evaluate acquisition
        best_f = self._train_Y.max().item()
        ei = LogExpectedImprovement(model=self.model, best_f=best_f)

        with torch.no_grad():
            ei_values = ei(candidates.unsqueeze(-2))

        # Select top candidates
        top_indices = ei_values.argsort(descending=True)[:effective_n]
        selected = candidates[top_indices]

        # Return requested number (might be fewer at high fidelity)
        return selected[:n_candidates]

    def get_fidelity_stats(self) -> dict:
        """Get statistics about fidelity levels used."""
        if self._train_fidelities is None:
            return {"n_observations": 0}

        return {
            "n_observations": len(self._train_fidelities),
            "current_fidelity": self.current_fidelity,
            "mean_fidelity": self._train_fidelities.mean().item(),
            "min_fidelity": self._train_fidelities.min().item(),
            "max_fidelity": self._train_fidelities.max().item(),
            "iteration": self._iteration,
            "total_budget": self._total_budget,
        }
