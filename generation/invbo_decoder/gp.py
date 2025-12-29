"""Deep Kernel Gaussian Process for instruction optimization.

Provides:
- InstructionDeepKernelGP: GP with Matern 5/2 kernel on 10D latent features
- EI computation for acquisition-guided optimization
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from scipy.stats import norm
from typing import Tuple, Optional

from generation.invbo_decoder.encoder import InstructionFeatureExtractor


class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
    """Gaussian Process with deep kernel for instruction optimization.

    Uses ARD Matern 5/2 kernel on 10D latent features from InstructionFeatureExtractor.
    Inherits from GPyTorchModel for BoTorch compatibility (enables EI, etc.).

    Architecture:
        768D GTR embedding -> InstructionFeatureExtractor -> 10D latent
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
        feature_extractor: InstructionFeatureExtractor,
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
        self.feature_extractor: Optional[InstructionFeatureExtractor] = None
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

        # Z-score standardization for outputs
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Initialize model
        self.feature_extractor = InstructionFeatureExtractor(
            input_dim=768, latent_dim=self.latent_dim
        ).to(self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.gp_model = InstructionDeepKernelGP(
            X_norm, y_norm, self.likelihood, self.feature_extractor
        ).to(self.device)

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

    def expected_improvement(
        self,
        embedding: torch.Tensor,
        xi: float = 0.01,
    ) -> float:
        """Compute Expected Improvement for embedding.

        EI(x) = (y_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
        where z = (y_best - mu(x) - xi) / sigma(x)

        We minimize error, so improvement = y_best - mu(x).

        Args:
            embedding: Instruction embedding (768,)
            xi: Exploration-exploitation trade-off (default 0.01)

        Returns:
            Expected improvement value
        """
        if self.gp_model is None or self.y_best is None:
            return 0.0

        mean, std = self.predict(embedding)

        if std <= 0:
            return max(self.y_best - mean, 0)

        z = (self.y_best - mean - xi) / std
        ei = (self.y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        return max(ei, 0)

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
