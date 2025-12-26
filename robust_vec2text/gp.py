"""Gaussian Process for latent space optimization.

Simplified GP without FeatureExtractor - operates directly
on 32D VAE latent representations.
"""

import torch
import gpytorch
from typing import Optional, Tuple
from scipy.stats import norm
import numpy as np


class LatentGP(gpytorch.models.ExactGP):
    """Gaussian Process directly on VAE latent space.

    Uses ARD Matérn 5/2 kernel on 32D latent features.
    No FeatureExtractor needed - VAE already extracts semantic features.

    This simplified design:
    - Reduces complexity (fewer learnable parameters)
    - Avoids overfitting on small datasets
    - Enables gradient-based optimization through the GP
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        latent_dim: int = 32,
    ):
        """Initialize GP.

        Args:
            train_x: Training inputs of shape (N, 32)
            train_y: Training targets of shape (N,)
            likelihood: GPyTorch likelihood (usually GaussianLikelihood)
            latent_dim: Dimension of latent space (32)
        """
        super().__init__(train_x, train_y, likelihood)

        self.latent_dim = latent_dim

        # Zero mean prior
        self.mean_module = gpytorch.means.ZeroMean()

        # ARD Matérn 5/2 kernel - learns separate lengthscale per dimension
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=latent_dim,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 32) - VAE latent representations

        Returns:
            MultivariateNormal distribution
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPTrainer:
    """Trainer for LatentGP with robust error handling.

    Handles common GP training issues like Cholesky decomposition failures
    by progressively increasing jitter.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        device: str = "cuda",
    ):
        """Initialize trainer.

        Args:
            latent_dim: Dimension of latent space (32)
            device: Device to use
        """
        self.latent_dim = latent_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.gp_model: Optional[LatentGP] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

        # Normalization parameters
        self.X_mean: Optional[torch.Tensor] = None
        self.X_std: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        max_epochs: int = 500,
        lr: float = 0.01,
        patience: int = 20,
        verbose: bool = True,
    ) -> bool:
        """Train GP on latent representations.

        Args:
            X: Training inputs of shape (N, 32) - VAE latent vectors
            y: Training targets of shape (N,) - error rates
            max_epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded, False otherwise
        """
        if len(X) < 4:
            if verbose:
                print("Not enough data for GP training (need at least 4)")
            return False

        # Check for variance in targets
        if y.std() < 1e-6:
            if verbose:
                print("No variance in targets, GP training skipped")
            return False

        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)

        # Normalize inputs (z-score)
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std

        # Normalize targets (z-score)
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Try training with progressively increasing jitter
        for jitter in [1e-4, 1e-3, 5e-3, 1e-2]:
            success = self._train_with_jitter(
                X_norm, y_norm, jitter, max_epochs, lr, patience, verbose
            )
            if success:
                return True
            if verbose:
                print(f"GP training failed with jitter={jitter}, trying higher...")

        if verbose:
            print("GP training failed with all jitter levels")
        return False

    def _train_with_jitter(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        jitter: float,
        max_epochs: int,
        lr: float,
        patience: int,
        verbose: bool,
    ) -> bool:
        """Train GP with specific jitter level."""
        try:
            # Initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.gp_model = LatentGP(
                X, y, self.likelihood, self.latent_dim
            ).to(self.device)

            # Training mode
            self.gp_model.train()
            self.likelihood.train()

            # Optimizer
            optimizer = torch.optim.AdamW(
                self.gp_model.parameters(), lr=lr, weight_decay=1e-4
            )

            # Loss function
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.gp_model
            )

            best_loss = float("inf")
            patience_counter = 0

            with gpytorch.settings.cholesky_jitter(jitter):
                for epoch in range(max_epochs):
                    optimizer.zero_grad()
                    output = self.gp_model(X)
                    loss = -mll(output, y)
                    loss.backward()
                    optimizer.step()

                    # Early stopping
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

            if verbose:
                print(f"GP training successful (jitter={jitter}, epochs={epoch+1})")

            return True

        except RuntimeError as e:
            if "cholesky" in str(e).lower() or "positive" in str(e).lower():
                return False
            raise

    def predict(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with trained GP.

        Args:
            X: Input tensor of shape (batch, 32)

        Returns:
            Tuple of (mean, std) in original scale
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained. Call train() first.")

        X = X.to(self.device)

        # Normalize inputs
        X_norm = (X - self.X_mean) / self.X_std

        # Predict
        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(X_norm))
            mean_norm = pred.mean
            std_norm = pred.stddev

        # Denormalize
        mean = mean_norm * self.y_std + self.y_mean
        std = std_norm * self.y_std

        return mean, std

    def expected_improvement(
        self,
        X: torch.Tensor,
        best_y: float,
        xi: float = 0.01,
    ) -> torch.Tensor:
        """Compute Expected Improvement acquisition function.

        Args:
            X: Candidate points of shape (batch, 32)
            best_y: Best observed value (lower is better)
            xi: Exploration parameter

        Returns:
            EI values of shape (batch,)
        """
        mean, std = self.predict(X)

        # For minimization, EI = E[max(best_y - f(x), 0)]
        improvement = best_y - mean - xi
        Z = improvement / (std + 1e-8)

        # Use scipy for stable computation
        Z_np = Z.cpu().numpy()
        ei_np = improvement.cpu().numpy() * norm.cdf(Z_np) + std.cpu().numpy() * norm.pdf(Z_np)
        ei_np = np.maximum(ei_np, 0)  # EI is non-negative

        return torch.tensor(ei_np, device=X.device, dtype=X.dtype)
