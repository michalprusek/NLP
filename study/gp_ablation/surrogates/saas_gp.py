"""SAASBO: Fully Bayesian GP with Sparse Axis-Aligned Subspace prior.

Implements:
- SAASGP: SAAS Fully Bayesian GP (Eriksson & Jankowiak 2021)

References:
- Eriksson & Jankowiak (2021) "High-Dimensional Bayesian Optimization with
  Sparse Axis-Aligned Subspaces"
"""

import logging
import math
import warnings
from typing import Optional, Tuple

import torch
from botorch.exceptions.warnings import InputDataWarning

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate

logger = logging.getLogger(__name__)


class SAASGP(BaseGPSurrogate):
    """SAASBO: Fully Bayesian GP with SAAS sparsity prior.

    Uses NUTS MCMC to sample GP hyperparameters with a sparsity-inducing
    prior on lengthscales. This automatically identifies relevant dimensions.

    Key hyperparameters:
        nuts_warmup: MCMC warmup steps (default 256)
        nuts_samples: MCMC samples (default 128)

    Note: SAASBO is more expensive than MAP fitting but provides better
    uncertainty quantification in high-dimensional spaces.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.nuts_warmup = config.nuts_warmup
        self.nuts_samples = config.nuts_samples
        self.initial_lengthscale = math.sqrt(self.D) / 10

    def _create_saas_model(
        self, train_X: torch.Tensor, train_Y: torch.Tensor
    ):
        """Create SAAS Fully Bayesian GP model."""
        try:
            from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        except ImportError:
            raise ImportError(
                "SAASBO requires BoTorch >= 0.8.0. "
                "Install with: pip install botorch>=0.8.0"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            model = SaasFullyBayesianSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
            )

        return model.to(self.device)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit SAAS GP using NUTS MCMC."""
        try:
            from botorch import fit_fully_bayesian_model_nuts
        except ImportError:
            try:
                from botorch.fit import fit_fully_bayesian_model_nuts
            except ImportError:
                raise ImportError(
                    "SAASBO requires BoTorch >= 0.8.0. "
                    "Install with: pip install botorch>=0.8.0"
                )

        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() == 1:
            self._train_Y = self._train_Y.unsqueeze(-1)

        self.model = self._create_saas_model(self._train_X, self._train_Y)

        # Fit using NUTS MCMC
        logger.info(
            f"Fitting SAAS GP with NUTS: warmup={self.nuts_warmup}, "
            f"samples={self.nuts_samples}"
        )

        fit_fully_bayesian_model_nuts(
            self.model,
            warmup_steps=self.nuts_warmup,
            num_samples=self.nuts_samples,
            thinning=2,  # Keep every 2nd sample
            disable_progbar=True,
        )

        self.model.eval()
        logger.info("SAAS GP fitted successfully")

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and standard deviation.

        For fully Bayesian model, this averages predictions over MCMC samples.
        """
        self._ensure_fitted("prediction")
        self.model.eval()

        with torch.no_grad():
            X = self._prepare_input(X)
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = torch.sqrt(posterior.variance.squeeze(-1) + 1e-6)

        return mean, std

    def get_lengthscale_importance(self) -> torch.Tensor:
        """Get importance of each dimension based on lengthscale posterior.

        Returns:
            Importance scores [D] (lower lengthscale = more important).
        """
        self._ensure_fitted("getting importance")

        # Get mean inverse lengthscales from MCMC samples
        # In SAAS, shorter lengthscale means more important dimension
        if hasattr(self.model, "median_lengthscale"):
            ls = self.model.median_lengthscale
        else:
            # Fallback: get from covar module
            ls = self.model.covar_module.base_kernel.lengthscale.mean(dim=0)

        # Importance = 1 / lengthscale (normalized)
        importance = 1.0 / (ls + 1e-6)
        importance = importance / importance.sum()

        return importance

    def get_top_dimensions(self, k: int = 10) -> torch.Tensor:
        """Get indices of top-k most important dimensions.

        Args:
            k: Number of top dimensions.

        Returns:
            Indices of top-k dimensions [k].
        """
        importance = self.get_lengthscale_importance()
        return importance.argsort(descending=True)[:k]
