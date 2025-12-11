"""
Deep Kernel GP for HyLO.

Implements:
- FeatureExtractor: Structural-aware feature extraction (instruction + exemplar -> latent)
- HyLOGP: Gaussian Process with Matérn 5/2 ARD kernel on latent space
- GPTrainer: Training and inference including differentiable EI computation
"""
import torch
import torch.nn as nn
import gpytorch
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


class FeatureExtractor(nn.Module):
    """Structural-aware feature extractor for deep kernel GP.

    Architecture from HbBoPs paper (Section 3.2):
    - Instruction encoder: Linear(768, 64) -> ReLU -> Linear(64, 32) -> ReLU
    - Exemplar encoder:   Linear(768, 64) -> ReLU -> Linear(64, 32) -> ReLU
    - Joint encoder:      Linear(64, 32) -> ReLU -> Linear(32, 10)

    The separate encoders capture that instructions and exemplars have
    different semantic roles in prompt composition.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 10,
        use_leaky_relu: bool = False,
        leaky_relu_slope: float = 0.01
    ):
        """
        Args:
            input_dim: Dimension of input embeddings (768 for GTR)
            latent_dim: Dimension of output latent space (default: 10)
            use_leaky_relu: Use LeakyReLU instead of ReLU (helps with gradient flow)
            leaky_relu_slope: Negative slope for LeakyReLU
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Select activation function
        def make_activation():
            if use_leaky_relu:
                return nn.LeakyReLU(negative_slope=leaky_relu_slope)
            return nn.ReLU()

        # Separate encoders for instruction and exemplar
        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            make_activation(),
            nn.Linear(64, 32),
            make_activation()
        )

        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            make_activation(),
            nn.Linear(64, 32),
            make_activation()
        )

        # Joint encoder combines both to latent space
        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * 32, 32),
            make_activation(),
            nn.Linear(32, latent_dim)
        )

    def forward(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor
    ) -> torch.Tensor:
        """Extract latent features from instruction and exemplar embeddings.

        Args:
            instruction_emb: (batch, 768) instruction embeddings
            exemplar_emb: (batch, 768) exemplar embeddings

        Returns:
            (batch, latent_dim) latent features
        """
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)
        combined = torch.cat([inst_features, ex_features], dim=1)
        return self.joint_encoder(combined)


class HyLOGP(gpytorch.models.ExactGP):
    """Gaussian Process with deep kernel for latent space modeling.

    Uses ARD Matérn 5/2 kernel operating on latent features extracted by
    FeatureExtractor. The ARD (Automatic Relevance Determination) learns
    which latent dimensions are most important for predicting error rates.

    Kernel: k(z, z') = outputscale * matern52(||z - z'||_ARD)

    where ||z - z'||_ARD = sqrt(sum_d ((z_d - z'_d) / lengthscale_d)^2)
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        feature_extractor: FeatureExtractor,
        input_dim: int = 768
    ):
        """
        Args:
            train_x: (N, 2*input_dim) concatenated instruction+exemplar embeddings
            train_y: (N,) standardized error rates
            likelihood: GPyTorch likelihood (e.g., GaussianLikelihood)
            feature_extractor: Trained FeatureExtractor module
            input_dim: Dimension of instruction/exemplar embeddings
        """
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        # ARD Matérn 5/2 kernel with priors
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=feature_extractor.latent_dim,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
        )

        self.feature_extractor = feature_extractor
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through GP.

        Args:
            x: (N, 2*input_dim) concatenated embeddings

        Returns:
            MultivariateNormal distribution over function values
        """
        # Split concatenated embeddings
        instruction_emb = x[:, :self.input_dim]
        exemplar_emb = x[:, self.input_dim:]

        # Extract latent features
        latent = self.feature_extractor(instruction_emb, exemplar_emb)

        # GP prior
        mean = self.mean_module(latent)
        covar = self.covar_module(latent)

        return gpytorch.distributions.MultivariateNormal(mean, covar)


@dataclass
class GPParams:
    """Container for trained GP parameters needed for differentiable EI."""
    model: HyLOGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    feature_extractor: FeatureExtractor
    X_min: torch.Tensor
    X_max: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    train_latents: torch.Tensor
    train_y_norm: torch.Tensor
    kernel_lengthscale: torch.Tensor
    kernel_outputscale: torch.Tensor
    noise_var: torch.Tensor
    device: torch.device


class GPTrainer:
    """Manages GP training and inference for HyLO."""

    def __init__(
        self,
        latent_dim: int = 10,
        train_epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        device: torch.device = None,
        use_leaky_relu: bool = False,
        leaky_relu_slope: float = 0.01
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            train_epochs: Maximum training epochs
            lr: Learning rate for Adam optimizer
            patience: Early stopping patience
            device: Torch device
            use_leaky_relu: Use LeakyReLU instead of ReLU in feature extractor
            leaky_relu_slope: Negative slope for LeakyReLU
        """
        self.latent_dim = latent_dim
        self.train_epochs = train_epochs
        self.lr = lr
        self.patience = patience
        self.device = device or torch.device("cpu")
        self.use_leaky_relu = use_leaky_relu
        self.leaky_relu_slope = leaky_relu_slope

        # Will be set during training
        self.gp_params: Optional[GPParams] = None

    def train(
        self,
        instruction_embeddings: np.ndarray,
        exemplar_embeddings: np.ndarray,
        error_rates: np.ndarray,
        verbose: bool = True
    ) -> GPParams:
        """Train GP on provided embeddings and error rates.

        Args:
            instruction_embeddings: (N, 768) instruction embeddings
            exemplar_embeddings: (N, 768) exemplar embeddings
            error_rates: (N,) error rates in [0, 1]
            verbose: Print training progress

        Returns:
            GPParams containing trained model and parameters
        """
        N = len(error_rates)

        # Concatenate embeddings
        X = np.concatenate([instruction_embeddings, exemplar_embeddings], axis=1)
        y = error_rates

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Input normalization to unit cube [0, 1]
        X_min = X_tensor.min(dim=0).values
        X_max = X_tensor.max(dim=0).values
        denom = X_max - X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X_tensor - X_min) / denom

        # Output standardization (zero mean, unit variance)
        y_mean = y_tensor.mean()
        y_std = y_tensor.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device)
        y_norm = (y_tensor - y_mean) / y_std

        # Initialize model
        feature_extractor = FeatureExtractor(
            input_dim=768,
            latent_dim=self.latent_dim,
            use_leaky_relu=self.use_leaky_relu,
            leaky_relu_slope=self.leaky_relu_slope
        ).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        model = HyLOGP(
            train_x=X_norm,
            train_y=y_norm,
            likelihood=likelihood,
            feature_extractor=feature_extractor,
            input_dim=768
        ).to(self.device)

        # Training mode
        model.train()
        likelihood.train()

        # Optimizer
        optimizer = torch.optim.AdamW(
            [{'params': model.parameters()}],
            lr=self.lr
        )

        # Loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.train_epochs):
            optimizer.zero_grad()

            try:
                output = model(X_norm)
                loss = -mll(output, y_norm)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()

                if loss_val < best_loss - 1e-4:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and epoch % 500 == 0:
                    print(f"Epoch {epoch}: loss = {loss_val:.4f}")

            except Exception as e:
                if verbose:
                    print(f"Training error at epoch {epoch}: {e}")
                # Add jitter and continue
                continue

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        # Extract kernel parameters
        kernel_lengthscale = model.covar_module.base_kernel.lengthscale.detach().squeeze()
        kernel_outputscale = model.covar_module.outputscale.detach()
        noise_var = likelihood.noise.detach()

        # Compute training latent features
        with torch.no_grad():
            inst_norm = X_norm[:, :768]
            ex_norm = X_norm[:, 768:]
            train_latents = feature_extractor(inst_norm, ex_norm)

        # Store parameters
        self.gp_params = GPParams(
            model=model,
            likelihood=likelihood,
            feature_extractor=feature_extractor,
            X_min=X_min,
            X_max=X_max,
            y_mean=y_mean,
            y_std=y_std,
            train_latents=train_latents,
            train_y_norm=y_norm,
            kernel_lengthscale=kernel_lengthscale,
            kernel_outputscale=kernel_outputscale,
            noise_var=noise_var,
            device=self.device
        )

        if verbose:
            print(f"GP training complete. Final loss: {best_loss:.4f}")
            print(f"Kernel lengthscale: {kernel_lengthscale.cpu().numpy()}")
            print(f"Kernel outputscale: {kernel_outputscale.item():.4f}")
            print(f"Noise variance: {noise_var.item():.6f}")

        return self.gp_params

    def predict(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and std for new embeddings.

        Args:
            instruction_emb: (768,) or (N, 768) instruction embedding(s)
            exemplar_emb: (768,) or (N, 768) exemplar embedding(s)

        Returns:
            (mean, std) in original scale (de-standardized)
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        # Ensure batch dimension
        if instruction_emb.dim() == 1:
            instruction_emb = instruction_emb.unsqueeze(0)
        if exemplar_emb.dim() == 1:
            exemplar_emb = exemplar_emb.unsqueeze(0)

        # Concatenate and normalize
        X = torch.cat([instruction_emb, exemplar_emb], dim=1)
        denom = p.X_max - p.X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X - p.X_min) / denom

        # Predict
        with torch.no_grad():
            p.model.eval()
            p.likelihood.eval()
            pred = p.likelihood(p.model(X_norm))

            # De-standardize
            mean = pred.mean * p.y_std + p.y_mean
            std = pred.stddev * p.y_std

        return mean.squeeze(), std.squeeze()

    def compute_ei_differentiable(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor,
        vmin_b: float,
        use_log_ei: bool = False,
        ei_epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Compute Expected Improvement in a differentiable way.

        Uses manual GP prediction to enable gradient flow through the
        instruction embedding for optimization.

        EI formula:
            EI(p) = (v_min - mu) * Phi(z) + sigma * phi(z)
            where z = (v_min - mu) / sigma

        Args:
            instruction_emb: (768,) tensor with requires_grad=True
            exemplar_emb: (768,) tensor (can be detached)
            vmin_b: Best observed error rate
            use_log_ei: If True, return log(EI) instead of EI (helps gradient flow)
            ei_epsilon: Epsilon for numerical stability in division

        Returns:
            Scalar EI value (differentiable w.r.t. instruction_emb)
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        # Ensure batch dimension
        if instruction_emb.dim() == 1:
            instruction_emb = instruction_emb.unsqueeze(0)
        if exemplar_emb.dim() == 1:
            exemplar_emb = exemplar_emb.unsqueeze(0)

        # Concatenate embeddings
        embedding = torch.cat([instruction_emb, exemplar_emb], dim=1).squeeze(0)

        # Normalize to unit cube
        denom = p.X_max - p.X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (embedding - p.X_min) / denom

        # Extract latent features (differentiable)
        inst_norm = X_norm[:768].unsqueeze(0)
        ex_norm = X_norm[768:].unsqueeze(0)
        test_latent = p.feature_extractor(inst_norm, ex_norm)  # (1, 10)

        # Compute kernel matrices (Matérn 5/2)
        # k(r) = outputscale * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
        def matern52_kernel(x1, x2, lengthscale, outputscale):
            diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # (N1, N2, D)
            scaled_diff = diff / lengthscale
            dist = torch.sqrt((scaled_diff ** 2).sum(dim=-1) + 1e-8)
            sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device))
            k = outputscale * (1 + sqrt5 * dist + 5.0/3.0 * dist**2) * torch.exp(-sqrt5 * dist)
            return k

        # K_train_train: (N, N)
        K_train = matern52_kernel(
            p.train_latents, p.train_latents,
            p.kernel_lengthscale, p.kernel_outputscale
        )
        K_train = K_train + p.noise_var * torch.eye(K_train.shape[0], device=K_train.device)

        # K_test_train: (1, N)
        K_test_train = matern52_kernel(
            test_latent, p.train_latents,
            p.kernel_lengthscale, p.kernel_outputscale
        )

        # K_test_test: (1, 1)
        K_test = matern52_kernel(
            test_latent, test_latent,
            p.kernel_lengthscale, p.kernel_outputscale
        )

        # GP predictive distribution
        # mean = K_test_train @ K_train^{-1} @ y
        # var = K_test_test - K_test_train @ K_train^{-1} @ K_test_train^T
        L = torch.linalg.cholesky(
            K_train + 1e-4 * torch.eye(K_train.shape[0], device=K_train.device)
        )
        alpha = torch.cholesky_solve(p.train_y_norm.unsqueeze(1), L)
        mean_norm = (K_test_train @ alpha).squeeze()

        v = torch.cholesky_solve(K_test_train.T, L)
        var_norm = (K_test - K_test_train @ v).squeeze()
        std_norm = torch.sqrt(torch.clamp(var_norm, min=1e-8))

        # De-standardize
        mean = mean_norm * p.y_std + p.y_mean
        std = std_norm * p.y_std

        # EI formula
        z = (vmin_b - mean) / (std + ei_epsilon)

        normal = torch.distributions.Normal(
            torch.zeros(1, device=embedding.device),
            torch.ones(1, device=embedding.device)
        )
        cdf_z = normal.cdf(z)
        pdf_z = torch.exp(normal.log_prob(z))

        ei = (vmin_b - mean) * cdf_z + std * pdf_z

        # Log-EI transformation for better gradient flow
        if use_log_ei:
            ei = torch.log(ei + ei_epsilon)

        return ei.squeeze()

    def compute_ei_batch(
        self,
        instruction_embs: torch.Tensor,
        exemplar_embs: torch.Tensor,
        vmin_b: float
    ) -> np.ndarray:
        """Compute EI for batch of embeddings (non-differentiable).

        Useful for scanning all exemplars to find the best one.

        Args:
            instruction_embs: (N, 768) instruction embeddings
            exemplar_embs: (N, 768) exemplar embeddings
            vmin_b: Best observed error rate

        Returns:
            (N,) numpy array of EI values
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        # Concatenate and normalize
        X = torch.cat([instruction_embs, exemplar_embs], dim=1)
        denom = p.X_max - p.X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X - p.X_min) / denom

        with torch.no_grad():
            p.model.eval()
            p.likelihood.eval()
            pred = p.likelihood(p.model(X_norm))

            # De-standardize
            means = pred.mean.cpu().numpy() * p.y_std.item() + p.y_mean.item()
            stds = pred.stddev.cpu().numpy() * p.y_std.item()

        # Compute EI
        from scipy.stats import norm
        ei_values = np.zeros(len(means))
        for i, (m, s) in enumerate(zip(means, stds)):
            if s <= 0:
                ei_values[i] = max(vmin_b - m, 0)
            else:
                z = (vmin_b - m) / s
                ei_values[i] = (vmin_b - m) * norm.cdf(z) + s * norm.pdf(z)

        return ei_values

    def get_latent_features(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor
    ) -> torch.Tensor:
        """Get latent features for embeddings.

        Useful for visualization.

        Args:
            instruction_emb: (768,) or (N, 768)
            exemplar_emb: (768,) or (N, 768)

        Returns:
            (latent_dim,) or (N, latent_dim) latent features
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        # Ensure batch dimension
        if instruction_emb.dim() == 1:
            instruction_emb = instruction_emb.unsqueeze(0)
        if exemplar_emb.dim() == 1:
            exemplar_emb = exemplar_emb.unsqueeze(0)

        # Move to GP device
        instruction_emb = instruction_emb.to(p.device)
        exemplar_emb = exemplar_emb.to(p.device)

        # Concatenate and normalize
        X = torch.cat([instruction_emb, exemplar_emb], dim=1)
        denom = p.X_max - p.X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X - p.X_min) / denom

        with torch.no_grad():
            inst_norm = X_norm[:, :768]
            ex_norm = X_norm[:, 768:]
            latent = p.feature_extractor(inst_norm, ex_norm)

        return latent.squeeze()
