"""Deep Kernel Gaussian Process for instruction optimization.

Provides:
- InstructionDeepKernelGP: GP with Matern 5/2 kernel on 10D latent features
- EI computation for acquisition-guided optimization
"""

import torch
import torch.nn as nn
import gpytorch
import math
import numpy as np
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.outcome import Standardize
from scipy.stats import norm
from scipy.special import erfcx, log1p, expm1
from typing import Tuple, Optional



# =============================================================================
# LogEI: Numerically stable log Expected Improvement
# Based on: "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)
# =============================================================================

# Constants for log_h computation
_C1 = math.log(2 * math.pi) / 2  # log(2π)/2
_C2 = math.log(math.pi / 2) / 2  # log(π/2)/2
_EPS = np.finfo(np.float64).eps  # Machine epsilon
_THRESH = -1 / math.sqrt(_EPS)  # Threshold for asymptotic branch


def _log1mexp(x: float) -> float:
    """Numerically stable computation of log(1 - exp(x)) for x < 0.

    Uses different branches based on threshold to avoid numerical issues.
    Based on Mächler (2012) "Accurately computing log(1 - exp(-|a|))".
    """
    if x > -math.log(2):
        return math.log(-expm1(x))
    else:
        return log1p(-math.exp(x))


def log_h(z: float) -> float:
    """Numerically stable computation of log(h(z)) where h(z) = φ(z) + z·Φ(z).

    This is the core of LogEI from the paper. Uses three branches:
    1. Direct computation for z > -1
    2. erfcx-based computation for -1/√ε < z ≤ -1
    3. Asymptotic approximation for z ≤ -1/√ε

    Args:
        z: Standardized improvement (μ - y*) / σ

    Returns:
        log(h(z)) value, can be negative (no clipping!)
    """
    if z > -1:
        # Direct computation is numerically stable for z > -1
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        h_val = phi_z + z * Phi_z
        if h_val > 0:
            return math.log(h_val)
        else:
            # Fallback to middle branch if numerical issues
            z = -1.0 - 1e-10

    if z > _THRESH:
        # Middle branch: -1/√ε < z ≤ -1
        # log_h(z) = -z²/2 - c1 + log1mexp(log(erfcx(-z/√2)·|z|) + c2)
        z_scaled = -z / math.sqrt(2)
        erfcx_val = erfcx(z_scaled)
        inner = math.log(erfcx_val * abs(z)) + _C2
        return -z * z / 2 - _C1 + _log1mexp(inner)
    else:
        # Asymptotic branch: z ≤ -1/√ε
        # log_h(z) ≈ -z²/2 - c1 - 2·log(|z|)
        return -z * z / 2 - _C1 - 2 * math.log(abs(z))


def log_h_tensor(z: torch.Tensor) -> torch.Tensor:
    """Tensor version of log_h that supports autograd.

    Numerically stable computation of log(h(z)) where h(z) = φ(z) + z·Φ(z).
    Uses piecewise approximation with smooth transitions for gradient stability.

    Args:
        z: Standardized improvement tensor (can be batched)

    Returns:
        log(h(z)) tensor with gradients preserved
    """
    # Constants
    LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
    SQRT_2 = math.sqrt(2)

    # Standard normal PDF and CDF
    phi_z = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    # Use error function for CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
    Phi_z = 0.5 * (1 + torch.erf(z / SQRT_2))

    # h(z) = φ(z) + z * Φ(z)
    h_z = phi_z + z * Phi_z

    # Clamp to avoid log(0) - use small positive value
    h_z_clamped = h_z.clamp(min=1e-40)

    # For very negative z, use asymptotic approximation
    # log_h(z) ≈ -z²/2 - log(2π)/2 - 2*log(|z|) for z << -1
    log_h_asymptotic = -0.5 * z * z - LOG_SQRT_2PI - 2 * torch.log(torch.abs(z).clamp(min=1e-10))

    # Use asymptotic for z < -5, direct for z >= -5
    # Smooth transition using sigmoid
    weight = torch.sigmoid(5 * (z + 5))  # 0 for z << -5, 1 for z >> -5
    log_h_direct = torch.log(h_z_clamped)

    return weight * log_h_direct + (1 - weight) * log_h_asymptotic


class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
    """Gaussian Process with deep kernel for instruction optimization.

    Uses ARD Matern 5/2 kernel on 10D latent features from VAEWithAdapter.
    Inherits from GPyTorchModel for BoTorch compatibility (enables EI, etc.).

    Architecture:
        768D GTR embedding -> VAEWithAdapter -> 10D latent
                                                   |
                                   Matern 5/2 kernel (ARD)
                                                   |
                                   GP(mean=0, K(latent))

    The kernel learns per-dimension lengthscales (ARD) to weight
    different latent dimensions based on their importance for prediction.
    """

    _num_outputs = 1  # Required for BoTorch

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        feature_extractor: nn.Module,
    ):
        """Initialize GP.

        Args:
            train_x: Training embeddings (N, 768)
            train_y: Training targets (N,) - error rates
            likelihood: Gaussian likelihood
            feature_extractor: Deep kernel feature extractor
        """
        # Ensure train_y is 1D for ExactGP
        train_y = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,  # Matern 5/2 - smooth but flexible
                ard_num_dims=10,  # Per-dimension lengthscales
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        self.feature_extractor = feature_extractor

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through GP.

        Args:
            x: Instruction embedding (batch, 768) or (batch, n, 768)

        Returns:
            MultivariateNormal distribution over function values
        """
        # Extract latent features
        latent = self.feature_extractor(x)

        # Handle 3D for BoTorch posterior
        if latent.dim() == 3:
            latent = latent.squeeze(-2) if latent.shape[-2] == 1 else latent

        return MultivariateNormal(
            self.mean_module(latent),
            self.covar_module(latent),
        )


class GPWithEI:
    """Wrapper for GP with Expected Improvement acquisition.

    Manages GP training, prediction, and EI computation for
    instruction optimization.
    """

    def __init__(
        self,
        device: str = "cuda",
        latent_dim: int = 10,
    ):
        """Initialize GP wrapper.

        Args:
            device: Device to use
            latent_dim: Latent dimension (10)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        # Components (initialized during training)
        self.feature_extractor: Optional[nn.Module] = None
        self.likelihood: Optional[GaussianLikelihood] = None
        self.gp_model: Optional[InstructionDeepKernelGP] = None

        # Training data
        self.X_train: Optional[torch.Tensor] = None  # (N, 768)
        self.y_train: Optional[torch.Tensor] = None  # (N,)

        # Normalization parameters
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # Best observed value (for EI)
        self.y_best: Optional[float] = None

    def set_training_data(
        self,
        embeddings: torch.Tensor,
        error_rates: torch.Tensor,
    ):
        """Set training data for GP.

        Args:
            embeddings: Instruction embeddings (N, 768)
            error_rates: Error rates (N,)
        """
        self.X_train = embeddings.to(self.device)
        self.y_train = error_rates.to(self.device)
        self.y_best = self.y_train.min().item()

    def train(
        self,
        epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
    ) -> bool:
        """Train GP on stored data.

        Args:
            epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data. Call set_training_data() first.")

        X = self.X_train
        y = self.y_train

        # Unit cube normalization for inputs
        self.X_min = X.min(dim=0)[0]
        self.X_max = X.max(dim=0)[0]
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (X - self.X_min) / denom

        # Standardize output transform (BoTorch standard approach)
        # This replaces manual z-score normalization and ensures correct
        # denormalization with posterior() and acquisition functions
        self.outcome_transform = Standardize(m=1).to(self.device)
        y_transformed, _ = self.outcome_transform(y.unsqueeze(-1))
        y_norm = y_transformed.squeeze(-1)

        # Keep y_mean/y_std for backwards compatibility with LogEI methods
        self.y_mean = self.outcome_transform.means.squeeze()
        self.y_std = self.outcome_transform.stdvs.squeeze()

        # Feature extractor must be pre-set (VAEWithAdapter)
        if self.feature_extractor is None:
            raise RuntimeError(
                "feature_extractor must be set before training. "
                "Use VAEWithAdapter from training.py."
            )
        # Noise constraint - allow GP to learn noise in reasonable range
        # Too low (1e-4) causes overconfidence, too high causes underfitting
        self.likelihood = GaussianLikelihood(
            noise_constraint=Interval(0.001, 0.1)
        ).to(self.device)
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood, self.feature_extractor
        ).to(self.device)

        # Register outcome transform for BoTorch compatibility
        # This allows qLogEI to automatically untransform predictions
        self.gp_model.outcome_transform = self.outcome_transform

        # Training loop
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0

        if verbose:
            print("Training GP...")

        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_norm)
                    loss = -mll(output, y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

                    if verbose and (epoch + 1) % 100 == 0:
                        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

                except RuntimeError as e:
                    if "cholesky" in str(e).lower():
                        if verbose:
                            print(f"  Cholesky error at epoch {epoch + 1}")
                        return False
                    raise

        if verbose:
            print(f"  GP training complete (epochs={epoch + 1}, loss={best_loss:.4f})")

        return True

    def predict(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[float, float]:
        """Predict error rate for embedding.

        Args:
            embedding: Instruction embedding (768,) or (1, 768)

        Returns:
            (mean, std) predictions in original scale
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained.")

        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Normalize
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (embedding - self.X_min) / denom

        # Predict
        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(X_norm))
            mean_norm = pred.mean.item()
            std_norm = pred.stddev.item()

        # Denormalize
        mean = mean_norm * self.y_std.item() + self.y_mean.item()
        std = std_norm * self.y_std.item()

        return mean, std

    def validate_predictions(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        """Validate GP predictions against true values.

        Useful for checking GP fit quality after training.

        Args:
            X: Embeddings to predict on (N, 768)
            y: True error rates (N,)

        Returns:
            Dictionary with MAE, RMSE, and max error metrics
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained.")

        preds = []
        for i in range(X.shape[0]):
            mean, _ = self.predict(X[i])
            preds.append(mean)

        preds = torch.tensor(preds, device=self.device)
        y = y.to(self.device)

        mae = (preds - y).abs().mean()
        rmse = ((preds - y) ** 2).mean().sqrt()
        max_error = (preds - y).abs().max()

        return {
            "mae": mae.item(),
            "rmse": rmse.item(),
            "max_error": max_error.item(),
        }

    def expected_improvement(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> float:
        """Compute Expected Improvement for embedding.

        EI(x) = (y_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
        where z = (y_best - mu(x) - xi) / sigma(x)

        We minimize error, so improvement = y_best - mu(x).

        Note: Does NOT clip to zero - allows negative EI values for better
        gradient-based optimization (see LogEI paper, NeurIPS 2023).

        Args:
            embedding: Instruction embedding (768,)
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            Expected improvement value (can be negative!)
        """
        if self.gp_model is None or self.y_best is None:
            return 0.0

        mean, std = self.predict(embedding)

        if std <= 0:
            # Deterministic case: return raw improvement (no clipping)
            return self.y_best - mean

        z = (self.y_best - mean - xi) / std
        ei = (self.y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        # No clipping to zero - allows gradient flow for better optimization
        return ei

    def log_expected_improvement(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> float:
        """Compute Log Expected Improvement for embedding.

        LogEI(x) = log_h(z) + log(σ(x))
        where z = (y_best - mu(x) - xi) / sigma(x)

        This is numerically stable even when EI values are extremely small,
        enabling better gradient-based optimization. Based on:
        "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)

        Args:
            embedding: Instruction embedding (768,)
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            Log expected improvement value (can be very negative!)
        """
        if self.gp_model is None or self.y_best is None:
            return float("-inf")

        mean, std = self.predict(embedding)

        if std <= 1e-10:
            # Near-deterministic: return -inf if no improvement possible
            if self.y_best > mean:
                return math.log(self.y_best - mean) if self.y_best - mean > 0 else float("-inf")
            return float("-inf")

        z = (self.y_best - mean - xi) / std

        # LogEI = log_h(z) + log(σ)
        return log_h(z) + math.log(std)

    def log_expected_improvement_tensor(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> torch.Tensor:
        """Compute Log Expected Improvement as a differentiable tensor.

        This version maintains gradients for optimization, allowing backprop
        through the decoder -> embedding -> GP -> LogEI chain.

        Args:
            embedding: Instruction embedding (768,) or (1, 768) - requires_grad allowed
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            LogEI as scalar tensor with gradient support
        """
        if self.gp_model is None or self.y_best is None:
            return torch.tensor(float("-inf"), device=self.device)

        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Normalize (preserving gradients)
        denom = self.X_max - self.X_min
        denom = denom.clone()
        denom[denom == 0] = 1.0
        X_norm = (embedding - self.X_min) / denom

        # Get GP prediction with gradients
        self.gp_model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.gp_model(X_norm)
            mean_norm = pred.mean  # Keep as tensor
            var_norm = pred.variance.clamp(min=1e-12)  # Clamp for numerical stability
            std_norm = torch.sqrt(var_norm)

        # Denormalize (keep as tensors)
        mean = mean_norm * self.y_std + self.y_mean
        std = std_norm * self.y_std

        # Compute z-score
        z = (self.y_best - mean - xi) / std.clamp(min=1e-10)

        # LogEI = log_h(z) + log(σ)
        log_ei = log_h_tensor(z) + torch.log(std.clamp(min=1e-10))

        return log_ei.squeeze()

    def get_latent(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get 10D latent for embedding using trained feature extractor.

        Args:
            embedding: Instruction embedding (768,)

        Returns:
            Latent (10,)
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not initialized.")

        embedding = embedding.to(self.device)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Normalize first
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (embedding - self.X_min) / denom

        self.feature_extractor.eval()
        with torch.no_grad():
            latent = self.feature_extractor(X_norm)

        return latent.squeeze(0)

    def add_observation_and_retrain(
        self,
        embedding: torch.Tensor,
        error_rate: float,
        epochs: int = 500,
        patience: int = 10,
        verbose: bool = False,
    ) -> bool:
        """Add new observation and retrain GP with preserved normalization.

        Uses incremental training that:
        - Preserves original normalization statistics (X_min, X_max, y_mean, y_std)
        - Warm-starts feature extractor from previous weights
        - Uses lower learning rate for stability

        Args:
            embedding: New instruction embedding (768,)
            error_rate: Observed error rate for this embedding
            epochs: Training epochs for retraining
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if retraining succeeded
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data. Call set_training_data() first.")

        # Add new observation
        embedding = embedding.to(self.device)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        self.X_train = torch.cat([self.X_train, embedding], dim=0)
        self.y_train = torch.cat([
            self.y_train,
            torch.tensor([error_rate], dtype=torch.float32, device=self.device)
        ])

        # Update best
        if error_rate < self.y_best:
            self.y_best = error_rate

        if verbose:
            print(f"  GP updated (total samples: {len(self.y_train)})")

        # Incremental retrain with preserved normalization
        return self._incremental_retrain(
            epochs=epochs,
            lr=0.001,  # Lower LR for stability (was 0.01)
            patience=patience,
            verbose=verbose,
        )

    def _incremental_retrain(
        self,
        epochs: int = 500,
        lr: float = 0.001,
        patience: int = 10,
        verbose: bool = False,
    ) -> bool:
        """Incremental retrain with updated normalization and frozen feature extractor.

        This method:
        - Recomputes output normalization (y_mean, y_std) for new data
        - Keeps input normalization (X_min, X_max) stable
        - KEEPS existing feature extractor (critical for VAE mode!)
        - Only retrains GP kernel hyperparameters
        """
        X = self.X_train
        y = self.y_train

        # Keep input normalization stable (prevents latent space drift)
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        X_norm = (X - self.X_min) / denom

        # Recompute output normalization for new data
        # This is critical - old y_mean/y_std don't account for new observations
        self.outcome_transform = Standardize(m=1).to(self.device)
        y_transformed, _ = self.outcome_transform(y.unsqueeze(-1))
        y_norm = y_transformed.squeeze(-1)
        self.y_mean = self.outcome_transform.means.squeeze()
        self.y_std = self.outcome_transform.stdvs.squeeze()

        # IMPORTANT: Keep existing feature_extractor!
        # In VAE mode, this is VAEWithAdapter - don't replace it.

        # Save old GP kernel hyperparameters for warm-start
        old_covar_state = None
        old_likelihood_state = None
        if self.gp_model is not None:
            old_covar_state = self.gp_model.covar_module.state_dict()
            old_likelihood_state = self.likelihood.state_dict()

        # Create new likelihood and GP model with existing feature extractor
        self.likelihood = GaussianLikelihood(
            noise_constraint=Interval(0.001, 0.1)
        ).to(self.device)
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood, self.feature_extractor
        ).to(self.device)

        # Warm-start: restore kernel hyperparameters
        if old_covar_state is not None:
            self.gp_model.covar_module.load_state_dict(old_covar_state)
        if old_likelihood_state is not None:
            self.likelihood.load_state_dict(old_likelihood_state)

        # Register outcome transform for BoTorch compatibility
        self.gp_model.outcome_transform = self.outcome_transform

        # Training loop
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0

        if verbose:
            print("Training GP...")

        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_norm)
                    loss = -mll(output, y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

                    if verbose and (epoch + 1) % 100 == 0:
                        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

                except RuntimeError as e:
                    if "cholesky" in str(e).lower():
                        if verbose:
                            print(f"  Cholesky error at epoch {epoch + 1}")
                        return False
                    raise

        if verbose:
            print(f"  GP training complete (epochs={epoch + 1}, loss={best_loss:.4f})")

        return True

    def get_training_size(self) -> int:
        """Get number of training samples."""
        return len(self.y_train) if self.y_train is not None else 0
