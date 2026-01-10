"""Gaussian Process for joint instruction-exemplar latent space.

Uses structure-aware product kernel:
    k(x, x') = k_inst(x_inst, x'_inst) × k_ex(x_ex, x'_ex)

This allows different smoothness in instruction vs exemplar subspaces.

Optional Deep Kernel Learning (DKL):
    z → FeatureExtractor(z) → Kernel → GP
"""

from typing import Optional, Tuple

import gpytorch
import torch
import torch.nn as nn
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ProductKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior


def _create_product_kernel(
    instruction_dim: int,
    exemplar_dim: int,
    lengthscale_prior: Optional[GammaPrior] = None,
) -> Tuple[MaternKernel, MaternKernel, ScaleKernel]:
    """Create product kernel for instruction-exemplar space.

    Args:
        instruction_dim: Dimension of instruction features
        exemplar_dim: Dimension of exemplar features
        lengthscale_prior: Prior for lengthscales (default: GammaPrior(4.0, 8.0))

    Returns:
        Tuple of (instruction_kernel, exemplar_kernel, combined_covar_module)
    """
    if lengthscale_prior is None:
        lengthscale_prior = GammaPrior(4.0, 8.0)

    instruction_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=instruction_dim,
        active_dims=list(range(instruction_dim)),
        lengthscale_prior=lengthscale_prior,
    )

    exemplar_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=exemplar_dim,
        active_dims=list(range(instruction_dim, instruction_dim + exemplar_dim)),
        lengthscale_prior=lengthscale_prior,
    )

    covar_module = ScaleKernel(
        ProductKernel(instruction_kernel, exemplar_kernel),
        outputscale_prior=GammaPrior(2.0, 2.0),
    )

    return instruction_kernel, exemplar_kernel, covar_module


def _squeeze_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """Handle BoTorch's 3D input by squeezing batch dimension if size 1."""
    if x.dim() == 3 and x.shape[-2] == 1:
        return x.squeeze(-2)
    return x


class JointPromptGP(ExactGP, GPyTorchModel):
    """GP on joint instruction-exemplar latent space.

    Uses structure-aware product kernel with separate lengthscales
    for instruction dims (0:16) and exemplar dims (16:24 by default).
    Dimensions are configurable via instruction_dim and exemplar_dim.
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
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y, likelihood)

        self.instruction_dim = instruction_dim
        self.exemplar_dim = exemplar_dim
        self.mean_module = ZeroMean()

        self.instruction_kernel, self.exemplar_kernel, self.covar_module = (
            _create_product_kernel(instruction_dim, exemplar_dim)
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP."""
        x = _squeeze_batch_dim(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepKernelGP(ExactGP, GPyTorchModel):
    """GP with Deep Kernel Learning (DKL).

    Adds a learned feature extractor before the kernel:
        z -> FeatureExtractor(z) -> Kernel -> GP prediction

    The feature extractor is trained jointly with the GP hyperparameters
    by maximizing the marginal log likelihood.
    """

    _num_outputs = 1

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        instruction_dim: int = 16,
        exemplar_dim: int = 16,
        feature_dim: int = 16,
        hidden_dim: int = 32,
    ):
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y, likelihood)

        self.instruction_dim = instruction_dim
        self.exemplar_dim = exemplar_dim
        self.feature_dim = feature_dim
        total_dim = instruction_dim + exemplar_dim

        # Maps total_dim -> feature_dim * 2 (for inst + ex features)
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

        self.mean_module = ZeroMean()

        # Kernels operate on learned features with different prior
        self.instruction_kernel, self.exemplar_kernel, self.covar_module = (
            _create_product_kernel(
                feature_dim, feature_dim, lengthscale_prior=GammaPrior(3.0, 6.0)
            )
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through DKL GP."""
        x = _squeeze_batch_dim(x)
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)

        return MultivariateNormal(mean_x, covar_x)


class GPWithEI:
    """GP wrapper with Expected Improvement optimization.

    Handles:
    - Unit cube normalization of inputs
    - Output standardization
    - Heteroscedastic noise from fidelity
    - Training and prediction
    - Optional Deep Kernel Learning (DKL)
    """

    def __init__(
        self,
        instruction_dim: int = 16,
        exemplar_dim: int = 16,
        device: str = "cuda",
        use_deep_kernel: bool = True,
        dkl_feature_dim: int = 16,
        dkl_hidden_dim: int = 32,
    ):
        self.instruction_dim = instruction_dim
        self.exemplar_dim = exemplar_dim
        self.total_dim = instruction_dim + exemplar_dim
        self.device = device

        # Deep Kernel Learning parameters
        self.use_deep_kernel = use_deep_kernel
        self.dkl_feature_dim = dkl_feature_dim
        self.dkl_hidden_dim = dkl_hidden_dim

        self.gp_model: Optional[ExactGP] = None  # JointPromptGP or DeepKernelGP
        self.likelihood: Optional[gpytorch.likelihoods.Likelihood] = None

        # Normalization parameters
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

        # Best observed value (for EI)
        self.best_f: Optional[float] = None

        # Training stats (populated after fit())
        self.training_stats: dict = {}

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

        Var = p(1-p) / (fidelity + alpha + beta + 1)
        """
        variance = error_rates * (1 - error_rates) / (fidelities + alpha + beta + 1)
        variance = torch.clamp(variance, min=1e-8, max=0.1)
        return variance / (self.y_std**2 + 1e-8)

    def _create_gp_model(
        self, X_norm: torch.Tensor, y_norm: torch.Tensor
    ) -> ExactGP:
        """Create GP model (DeepKernelGP or JointPromptGP based on config)."""
        if self.use_deep_kernel:
            return DeepKernelGP(
                train_x=X_norm,
                train_y=y_norm,
                likelihood=self.likelihood,
                instruction_dim=self.instruction_dim,
                exemplar_dim=self.exemplar_dim,
                feature_dim=self.dkl_feature_dim,
                hidden_dim=self.dkl_hidden_dim,
            ).to(self.device)

        return JointPromptGP(
            train_x=X_norm,
            train_y=y_norm,
            likelihood=self.likelihood,
            instruction_dim=self.instruction_dim,
            exemplar_dim=self.exemplar_dim,
        ).to(self.device)

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

        # Create likelihood (heteroscedastic if fidelities provided)
        if fidelities is not None:
            fidelities = fidelities.to(self.device)
            noise = self._compute_heteroscedastic_noise(-y, fidelities)
            self.likelihood = FixedNoiseGaussianLikelihood(
                noise=noise, learn_additional_noise=True
            ).to(self.device)
        else:
            self.likelihood = GaussianLikelihood(
                noise_constraint=Interval(0.001, 0.1)
            ).to(self.device)

        self.gp_model = self._create_gp_model(X_norm, y_norm)

        # Training
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0
        history = {"loss": []}

        try:
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.gp_model(X_norm)
                loss = -mll(output, y_norm)

                # Check for NaN loss
                if torch.isnan(loss):
                    raise ValueError(
                        f"NaN loss at epoch {epoch}. "
                        f"Check input data for outliers or duplicate points."
                    )

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
        except RuntimeError as e:
            if "cholesky" in str(e).lower():
                raise RuntimeError(
                    f"GP Cholesky decomposition failed at epoch {epoch}. "
                    f"This often indicates duplicate inputs or ill-conditioned kernel. "
                    f"Try adding jitter (noise_constraint) or removing duplicate training points.\n"
                    f"Original error: {e}"
                ) from e
            raise

        # Collect training stats
        self._collect_training_stats(
            epochs_trained=epoch + 1,
            final_loss=best_loss,
            early_stopped=patience_counter >= patience,
            num_samples=len(X),
            loss_history=history["loss"],
        )

        return history

    def _collect_training_stats(
        self,
        epochs_trained: int,
        final_loss: float,
        early_stopped: bool,
        num_samples: int,
        loss_history: list,
    ):
        """Collect training statistics for debugging and analysis."""
        # Get kernel parameters
        if hasattr(self.gp_model, "covar_module"):
            covar = self.gp_model.covar_module
            if hasattr(covar, "base_kernel") and hasattr(covar.base_kernel, "kernels"):
                # Product kernel case
                inst_kernel = covar.base_kernel.kernels[0]
                ex_kernel = covar.base_kernel.kernels[1]

                inst_ls = inst_kernel.lengthscale.detach().cpu().squeeze()
                ex_ls = ex_kernel.lengthscale.detach().cpu().squeeze()

                self.training_stats = {
                    "epochs_trained": epochs_trained,
                    "final_loss": float(final_loss),
                    "early_stopped": early_stopped,
                    "num_samples": num_samples,
                    "instruction_lengthscale_mean": float(inst_ls.mean()),
                    "instruction_lengthscale_min": float(inst_ls.min()),
                    "instruction_lengthscale_max": float(inst_ls.max()),
                    "exemplar_lengthscale_mean": float(ex_ls.mean()),
                    "exemplar_lengthscale_min": float(ex_ls.min()),
                    "exemplar_lengthscale_max": float(ex_ls.max()),
                    "outputscale": float(covar.outputscale.detach().cpu().item()),
                    "loss_history": loss_history[::max(1, len(loss_history) // 100)],  # Sample 100 points
                }

                # Find most relevant dims (smallest lengthscale = most important)
                all_ls = torch.cat([inst_ls, ex_ls])
                sorted_dims = torch.argsort(all_ls)[:5]
                self.training_stats["top_5_relevant_dims"] = sorted_dims.tolist()
                self.training_stats["top_5_lengthscales"] = all_ls[sorted_dims].tolist()
            else:
                # Single kernel case
                self.training_stats = {
                    "epochs_trained": epochs_trained,
                    "final_loss": float(final_loss),
                    "early_stopped": early_stopped,
                    "num_samples": num_samples,
                    "loss_history": loss_history[::max(1, len(loss_history) // 100)],
                }
        else:
            self.training_stats = {
                "epochs_trained": epochs_trained,
                "final_loss": float(final_loss),
                "early_stopped": early_stopped,
                "num_samples": num_samples,
                "loss_history": loss_history[::max(1, len(loss_history) // 100)],
            }

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

        Raises:
            RuntimeError: If GP has not been fit yet
        """
        if self.gp_model is None or self.likelihood is None:
            raise RuntimeError(
                "GP must be fit before predict(). Call fit() first."
            )

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
