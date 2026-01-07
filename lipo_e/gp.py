"""Gaussian Process for joint instruction-exemplar latent space.

Uses structure-aware product kernel:
    k(x, x') = k_inst(x_inst, x'_inst) × k_ex(x_ex, x'_ex)

This allows different smoothness in instruction vs exemplar subspaces.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    ProductKernel,
)
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior
from botorch.models.gpytorch import GPyTorchModel


class JointPromptGP(ExactGP, GPyTorchModel):
    """GP on joint instruction-exemplar latent space.

    Uses structure-aware product kernel with separate lengthscales
    for instruction dims (0:16) and exemplar dims (16:32).
    """

    _num_outputs = 1

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        instruction_dim: int = 16,
        exemplar_dim: int = 16,
    ):
        # Ensure y is 1D
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y

        super().__init__(train_x, train_y, likelihood)

        self.instruction_dim = instruction_dim
        self.exemplar_dim = exemplar_dim
        total_dim = instruction_dim + exemplar_dim

        self.mean_module = ZeroMean()

        # Structure-aware kernel: product of instruction and exemplar kernels
        self.instruction_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=instruction_dim,
            active_dims=list(range(instruction_dim)),
            lengthscale_prior=GammaPrior(4.0, 8.0),
        )

        self.exemplar_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=exemplar_dim,
            active_dims=list(range(instruction_dim, total_dim)),
            lengthscale_prior=GammaPrior(4.0, 8.0),
        )

        # Combined kernel
        self.covar_module = ScaleKernel(
            ProductKernel(
                self.instruction_kernel,
                self.exemplar_kernel,
            ),
            outputscale_prior=GammaPrior(2.0, 2.0),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP."""
        if x.dim() == 3:
            x = x.squeeze(-2) if x.shape[-2] == 1 else x

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPWithEI:
    """GP wrapper with Expected Improvement optimization.

    Handles:
    - Unit cube normalization of inputs
    - Output standardization
    - Heteroscedastic noise from fidelity
    - Training and prediction
    """

    def __init__(
        self,
        instruction_dim: int = 16,
        exemplar_dim: int = 16,
        device: str = "cuda",
    ):
        self.instruction_dim = instruction_dim
        self.exemplar_dim = exemplar_dim
        self.total_dim = instruction_dim + exemplar_dim
        self.device = device

        self.gp_model: Optional[JointPromptGP] = None
        self.likelihood: Optional[gpytorch.likelihoods.Likelihood] = None

        # Normalization parameters
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

        # Best observed value (for EI)
        self.best_f: Optional[float] = None

    def _normalize_X(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [0, 1]."""
        return (X - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _standardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize outputs to zero mean, unit variance."""
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def _compute_heteroscedastic_noise(
        self,
        error_rates: torch.Tensor,
        fidelities: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Compute heteroscedastic noise from Beta posterior variance.

        Var = p(1-p) / (fidelity + α + β + 1)
        """
        p = error_rates
        variance = p * (1 - p) / (fidelities + alpha + beta + 1)
        # Clamp and standardize
        variance = torch.clamp(variance, min=1e-8, max=0.1)
        # Scale to standardized output space
        return variance / (self.y_std ** 2 + 1e-8)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        fidelities: Optional[torch.Tensor] = None,
        epochs: int = 1000,
        lr: float = 0.01,
        patience: int = 50,
    ) -> dict:
        """Train GP on data.

        Args:
            X: Latent representations (N, total_dim)
            y: Error rates (N,) - will be negated internally
            fidelities: Evaluation fidelities (N,)
            epochs: Training epochs
            lr: Learning rate
            patience: Early stopping patience

        Returns:
            Training history
        """
        X = X.to(self.device)
        y = y.to(self.device)

        # Negate error rates (BoTorch maximizes, we want to minimize error)
        y = -y

        # Store normalization parameters
        self.X_min = X.min(dim=0).values
        self.X_max = X.max(dim=0).values
        self.y_mean = y.mean().item()
        self.y_std = y.std().item()
        if self.y_std < 1e-6:
            self.y_std = 1.0

        # Store best observed (most negative error = best)
        self.best_f = y.max().item()

        # Normalize
        X_norm = self._normalize_X(X)
        y_norm = self._standardize_y(y)

        # Compute noise
        if fidelities is not None:
            fidelities = fidelities.to(self.device)
            noise = self._compute_heteroscedastic_noise(-y, fidelities)  # Un-negate for noise calc
            self.likelihood = FixedNoiseGaussianLikelihood(
                noise=noise,
                learn_additional_noise=True,
            ).to(self.device)
        else:
            self.likelihood = GaussianLikelihood(
                noise_constraint=Interval(0.001, 0.1),
            ).to(self.device)

        # Create model
        self.gp_model = JointPromptGP(
            train_x=X_norm,
            train_y=y_norm,
            likelihood=self.likelihood,
            instruction_dim=self.instruction_dim,
            exemplar_dim=self.exemplar_dim,
        ).to(self.device)

        # Training
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0
        history = {"loss": []}

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(X_norm)
            loss = -mll(output, y_norm)
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return history

    def predict(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and std for new points.

        Args:
            X: Test points (N, total_dim)

        Returns:
            mean: Predicted error rates (un-negated)
            std: Standard deviations
        """
        X = X.to(self.device)
        X_norm = self._normalize_X(X)

        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(X_norm))
            mean_norm = pred.mean
            std_norm = pred.stddev

        # Un-standardize and un-negate
        mean = -(mean_norm * self.y_std + self.y_mean)
        std = std_norm * self.y_std

        return mean, std

    def get_bounds(self, margin: float = 0.1) -> torch.Tensor:
        """Get optimization bounds with margin.

        Returns:
            bounds: (2, total_dim) - [[lower], [upper]]
        """
        # In normalized space, bounds are [0, 1]
        # Add margin for exploration
        lower = torch.zeros(self.total_dim, device=self.device) - margin
        upper = torch.ones(self.total_dim, device=self.device) + margin

        return torch.stack([lower, upper])

    def add_observation(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        fidelity_new: Optional[torch.Tensor] = None,
    ):
        """Add new observation and update GP (fast update).

        For full retrain, call fit() again.
        """
        # Get current training data
        X_train = self.gp_model.train_inputs[0]
        y_train = self.gp_model.train_targets

        # Un-normalize and un-standardize current data
        X_orig = X_train * (self.X_max - self.X_min + 1e-8) + self.X_min
        y_orig = y_train * (self.y_std + 1e-8) + self.y_mean

        # Add new observation (negated)
        X_new = X_new.to(self.device)
        y_new = -y_new.to(self.device)  # Negate

        X_combined = torch.cat([X_orig, X_new.unsqueeze(0)], dim=0)
        y_combined = torch.cat([y_orig, y_new.unsqueeze(0)], dim=0)

        # Recompute normalization
        self.X_min = X_combined.min(dim=0).values
        self.X_max = X_combined.max(dim=0).values
        self.y_mean = y_combined.mean().item()
        self.y_std = y_combined.std().item()
        if self.y_std < 1e-6:
            self.y_std = 1.0

        # Update best
        if y_new.item() > self.best_f:
            self.best_f = y_new.item()

        # Normalize
        X_norm = self._normalize_X(X_combined)
        y_norm = self._standardize_y(y_combined)

        # Update model (fantasy model for efficiency)
        self.gp_model.set_train_data(X_norm, y_norm, strict=False)
