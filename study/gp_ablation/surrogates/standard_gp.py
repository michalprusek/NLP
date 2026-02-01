"""Standard GP surrogates with MSR initialization.

Implements:
- StandardMSRGP: Standard GP with Matern-5/2 and MSR lengthscale prior (ICLR 2025)
- HeteroscedasticGP: GP with binomial noise model for accuracy evaluation

References:
- Hvarfner et al. (2024) "Vanilla Bayesian Optimization in High Dimensions"
"""

import math
import warnings
from typing import Optional, Tuple

import gpytorch
import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate


def create_kernel(
    kernel_type: str,
    dim: int,
    device: torch.device,
    use_msr_prior: bool = True,
) -> ScaleKernel:
    """Create kernel with optional MSR lengthscale prior.

    Args:
        kernel_type: Kernel type (matern52, matern32, rbf).
        dim: Input dimensionality.
        device: Torch device.
        use_msr_prior: If True, use MSR lengthscale prior from Hvarfner 2024.

    Returns:
        ScaleKernel wrapping the base kernel.
    """
    # MSR prior: LogNormal(sqrt(2) + 0.5*log(D), sqrt(3))
    # This encourages lengthscales around sqrt(D) which is appropriate for high-D
    if use_msr_prior:
        prior = LogNormalPrior(
            loc=torch.tensor(math.sqrt(2) + 0.5 * math.log(dim), device=device),
            scale=torch.tensor(math.sqrt(3), device=device),
        )
    else:
        prior = None

    # Create base kernel
    if kernel_type == "matern52":
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_prior=prior,
        )
    elif kernel_type == "matern32":
        base_kernel = MaternKernel(
            nu=1.5,
            ard_num_dims=dim,
            lengthscale_prior=prior,
        )
    elif kernel_type == "rbf":
        base_kernel = RBFKernel(
            ard_num_dims=dim,
            lengthscale_prior=prior,
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return ScaleKernel(base_kernel)


class StandardMSRGP(BaseGPSurrogate):
    """Standard GP with MSR initialization for high-dimensional optimization.

    Uses Matern-5/2 kernel with ARD and MSR lengthscale prior from
    Hvarfner et al. (2024). Initial lengthscale is sqrt(D)/10.

    This is the recommended baseline for high-D BO (ICLR 2025).
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)
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

        # Initialize lengthscale to MSR value
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

        # Fit hyperparameters via MLL optimization
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

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


class HeteroscedasticGP(BaseGPSurrogate):
    """GP with heteroscedastic binomial noise for accuracy evaluation.

    Uses Var(p_hat) = p(1-p)/n for binomial proportion observations.
    This is appropriate when Y represents accuracy computed from n samples.

    Note: We do NOT use Standardize transform because it's incompatible with
    known observation variance. The variance must be in the same scale as Y.
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)
        self.n_eval = config.n_eval
        self.initial_lengthscale = math.sqrt(self.D) / 10

    def _compute_variance(self, Y: torch.Tensor) -> torch.Tensor:
        """Compute binomial variance: p(1-p)/n."""
        Y_flat = Y.squeeze(-1) if Y.dim() > 1 else Y
        p = Y_flat.clamp(0.01, 0.99)
        var = (p * (1 - p)) / self.n_eval
        var = var.clamp(min=1e-6)
        return var.unsqueeze(-1) if Y.dim() > 1 else var

    def _create_model(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor,
    ) -> SingleTaskGP:
        """Create GP model with known observation noise.

        Note: We don't use Standardize transform because it conflicts with
        the known variance. The variance is specified in the original Y scale.
        """
        covar_module = create_kernel(
            self.config.kernel,
            self.D,
            self.device,
            use_msr_prior=True,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            # No Standardize transform - variance is in original Y scale
            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                covar_module=covar_module,
                input_transform=Normalize(d=self.D),
                # outcome_transform removed - incompatible with train_Yvar
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
        self.model = self._create_model(self._train_X, self._train_Y, train_Yvar)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

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
