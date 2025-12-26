"""
Deep Kernel GP with LatentProjector for HyLO2.

Extends the original gp_model with:
- LatentProjector: 10D -> 768D projection for Vec2Text
- GPTrainer2: Joint training with reconstruction loss
- GPParams2: Extended parameters including projector

Architecture:
    Encoder: (instruction_768 + exemplar_768) -> FeatureExtractor -> latent_10
    Decoder: latent_10 -> LatentProjector -> instruction_768

Training:
    Phase 1 (warmup): Loss = -MLL
    Phase 2 (joint):  Loss = -MLL + lambda * MSE(original_inst, reconstructed_inst)
"""
import torch
import torch.nn as nn
import gpytorch
import numpy as np
from typing import Tuple, Optional
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


class LatentProjector(nn.Module):
    """Projects 10D latent space back to 768D instruction embedding space.

    This is the decoder half of an autoencoder structure where FeatureExtractor
    is the encoder. The projector learns to produce embeddings in Vec2Text's
    training distribution.

    Architecture:
        Linear(10, 768) - simple linear projection as recommended by research

    Key insight:
        The FeatureExtractor takes BOTH instruction and exemplar to produce latent.
        But we only want to reconstruct instruction. This creates an asymmetric
        autoencoder where the projector learns to extract the instruction-relevant
        part of the 10D latent.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        output_dim: int = 768
    ):
        """
        Args:
            latent_dim: Dimension of input latent space (default: 10)
            output_dim: Dimension of output embedding space (768 for GTR)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Simple linear layer as recommended in Vec2Text alignment research
        self.projection = nn.Linear(latent_dim, output_dim)

        # Initialize with small weights for stable early training
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent features to instruction embedding space.

        Args:
            latent: (batch, 10) or (10,) latent features

        Returns:
            (batch, 768) or (768,) instruction embeddings
        """
        return self.projection(latent)


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
class GPParams2:
    """Container for trained GP parameters including LatentProjector.

    Extended from GPParams to include:
    - latent_projector: Trained LatentProjector for 10D -> 768D
    - latent_mean/std: Statistics for bounding latent space optimization
    - inst_min/max: Normalization bounds for instruction embeddings
    """
    model: HyLOGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    feature_extractor: FeatureExtractor
    latent_projector: LatentProjector
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
    # Latent space statistics for optimization bounds
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    # Instruction normalization bounds (for denormalization after projection)
    inst_min: torch.Tensor
    inst_max: torch.Tensor


class GPTrainer2:
    """GP Trainer with joint LatentProjector training.

    Training proceeds in two phases:
    1. Warmup (epochs 0 to warmup_epochs): GP training only (-MLL loss)
       - FeatureExtractor and GP learn to model error rates
       - LatentProjector weights are updated but reconstruction loss is zero

    2. Joint (epochs warmup_epochs to end): -MLL + lambda * reconstruction_loss
       - Forces LatentProjector to learn to reconstruct instruction embeddings
       - FeatureExtractor continues to refine while accommodating reconstruction

    Loss formula:
        Phase 1: total_loss = -MLL
        Phase 2: total_loss = -MLL + lambda * MSE(original_inst_norm, reconstructed_inst)
    """

    def __init__(
        self,
        latent_dim: int = 10,
        train_epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        device: torch.device = None,
        use_leaky_relu: bool = False,
        leaky_relu_slope: float = 0.01,
        reconstruction_weight: float = 1.0,
        warmup_epochs: int = 500
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
            reconstruction_weight: Lambda for reconstruction loss (default: 0.1)
            warmup_epochs: GP-only epochs before adding reconstruction (default: 500)
        """
        self.latent_dim = latent_dim
        self.train_epochs = train_epochs
        self.lr = lr
        self.patience = patience
        self.device = device or torch.device("cpu")
        self.use_leaky_relu = use_leaky_relu
        self.leaky_relu_slope = leaky_relu_slope
        self.reconstruction_weight = reconstruction_weight
        self.warmup_epochs = warmup_epochs

        # Will be set during training
        self.gp_params: Optional[GPParams2] = None

    def train(
        self,
        instruction_embeddings: np.ndarray,
        exemplar_embeddings: np.ndarray,
        error_rates: np.ndarray,
        verbose: bool = True
    ) -> GPParams2:
        """Train GP + FeatureExtractor + LatentProjector jointly.

        Args:
            instruction_embeddings: (N, 768) instruction embeddings
            exemplar_embeddings: (N, 768) exemplar embeddings
            error_rates: (N,) error rates in [0, 1]
            verbose: Print training progress

        Returns:
            GPParams2 containing trained models and parameters
        """
        N = len(error_rates)

        # Concatenate embeddings for GP input
        X = np.concatenate([instruction_embeddings, exemplar_embeddings], axis=1)
        y = error_rates

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Keep original instruction embeddings for reconstruction loss
        inst_orig = torch.tensor(
            instruction_embeddings, dtype=torch.float32, device=self.device
        )

        # Input normalization to unit cube [0, 1]
        X_min = X_tensor.min(dim=0).values
        X_max = X_tensor.max(dim=0).values
        denom = X_max - X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X_tensor - X_min) / denom

        # Instruction normalization bounds
        inst_min = X_min[:768]
        inst_max = X_max[:768]
        inst_denom = inst_max - inst_min
        inst_denom = torch.where(inst_denom == 0, torch.ones_like(inst_denom), inst_denom)

        # Normalize original instructions for reconstruction comparison
        inst_orig_norm = (inst_orig - inst_min) / inst_denom

        # Output standardization (zero mean, unit variance)
        y_mean = y_tensor.mean()
        y_std = y_tensor.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device)
        y_norm = (y_tensor - y_mean) / y_std

        # Initialize models
        feature_extractor = FeatureExtractor(
            input_dim=768,
            latent_dim=self.latent_dim,
            use_leaky_relu=self.use_leaky_relu,
            leaky_relu_slope=self.leaky_relu_slope
        ).to(self.device)

        latent_projector = LatentProjector(
            latent_dim=self.latent_dim,
            output_dim=768
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

        # Single optimizer for all parameters (GP, FeatureExtractor, LatentProjector)
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
            {'params': latent_projector.parameters()}
        ], lr=self.lr)

        # Loss functions
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        mse_loss = nn.MSELoss()

        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0

        if verbose:
            print(f"Training GP with LatentProjector (warmup: {self.warmup_epochs} epochs)")
            print(f"Reconstruction weight: {self.reconstruction_weight}")

        for epoch in range(self.train_epochs):
            optimizer.zero_grad()

            try:
                # Forward pass through GP
                output = model(X_norm)
                gp_loss = -mll(output, y_norm)

                # Reconstruction loss (after warmup)
                if epoch >= self.warmup_epochs:
                    # Get latent features
                    inst_norm = X_norm[:, :768]
                    ex_norm = X_norm[:, 768:]
                    latent = feature_extractor(inst_norm, ex_norm)

                    # Reconstruct instruction embeddings (in normalized space)
                    reconstructed_inst = latent_projector(latent)

                    # Reconstruction loss
                    recon_loss = mse_loss(reconstructed_inst, inst_orig_norm)

                    total_loss = gp_loss + self.reconstruction_weight * recon_loss
                else:
                    total_loss = gp_loss
                    recon_loss = torch.tensor(0.0, device=self.device)

                total_loss.backward()
                optimizer.step()

                loss_val = total_loss.item()

                # Early stopping - only after warmup phase to ensure reconstruction training
                if epoch >= self.warmup_epochs:
                    if loss_val < best_loss - 1e-4:
                        best_loss = loss_val
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
                else:
                    # During warmup, just track best loss but don't stop
                    if loss_val < best_loss:
                        best_loss = loss_val

                if verbose and epoch % 500 == 0:
                    if epoch >= self.warmup_epochs:
                        print(f"Epoch {epoch}: total={loss_val:.4f}, "
                              f"gp={gp_loss.item():.4f}, recon={recon_loss.item():.6f}")
                    else:
                        print(f"Epoch {epoch}: loss={loss_val:.4f} (warmup)")

            except Exception as e:
                if verbose:
                    print(f"Training error at epoch {epoch}: {e}")
                continue

        # Switch to eval mode
        model.eval()
        likelihood.eval()
        latent_projector.eval()

        # Extract kernel parameters
        kernel_lengthscale = model.covar_module.base_kernel.lengthscale.detach().squeeze()
        kernel_outputscale = model.covar_module.outputscale.detach()
        noise_var = likelihood.noise.detach()

        # Compute training latent features and their statistics
        with torch.no_grad():
            inst_norm = X_norm[:, :768]
            ex_norm = X_norm[:, 768:]
            train_latents = feature_extractor(inst_norm, ex_norm)
            latent_mean = train_latents.mean(dim=0)
            latent_std = train_latents.std(dim=0)
            # Ensure non-zero std for each dimension
            latent_std = torch.where(latent_std < 1e-6, torch.ones_like(latent_std), latent_std)

        # Store parameters
        self.gp_params = GPParams2(
            model=model,
            likelihood=likelihood,
            feature_extractor=feature_extractor,
            latent_projector=latent_projector,
            X_min=X_min,
            X_max=X_max,
            y_mean=y_mean,
            y_std=y_std,
            train_latents=train_latents,
            train_y_norm=y_norm,
            kernel_lengthscale=kernel_lengthscale,
            kernel_outputscale=kernel_outputscale,
            noise_var=noise_var,
            device=self.device,
            latent_mean=latent_mean,
            latent_std=latent_std,
            inst_min=inst_min,
            inst_max=inst_max
        )

        if verbose:
            print(f"\nGP training complete. Final loss: {best_loss:.4f}")
            print(f"Kernel lengthscale: {kernel_lengthscale.cpu().numpy()}")
            print(f"Kernel outputscale: {kernel_outputscale.item():.4f}")
            print(f"Noise variance: {noise_var.item():.6f}")
            print(f"Latent mean: {latent_mean.cpu().numpy()}")
            print(f"Latent std: {latent_std.cpu().numpy()}")

            # Compute final reconstruction error
            with torch.no_grad():
                latent = feature_extractor(X_norm[:, :768], X_norm[:, 768:])
                reconstructed = latent_projector(latent)
                final_recon_mse = mse_loss(reconstructed, inst_orig_norm).item()
                print(f"Final reconstruction MSE: {final_recon_mse:.6f}")

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

    def compute_ei_in_latent_space(
        self,
        latent: torch.Tensor,
        vmin_b: float,
        use_log_ei: bool = False,
        ei_epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Compute Expected Improvement directly from latent point.

        This is the key method for latent space optimization - it computes EI
        without going through the FeatureExtractor (since we already have the latent).

        Args:
            latent: (10,) or (batch, 10) latent point(s)
            vmin_b: Best observed error rate
            use_log_ei: Use log(EI) for better gradient flow
            ei_epsilon: Epsilon for numerical stability

        Returns:
            Scalar or (batch,) EI value(s) (differentiable w.r.t. latent)
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Matern 5/2 kernel computation
        def matern52_kernel(x1, x2, lengthscale, outputscale):
            diff = x1.unsqueeze(1) - x2.unsqueeze(0)
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

        # K_test_train: (batch, N)
        K_test_train = matern52_kernel(
            latent, p.train_latents,
            p.kernel_lengthscale, p.kernel_outputscale
        )

        # K_test_test: (batch, batch)
        K_test = matern52_kernel(
            latent, latent,
            p.kernel_lengthscale, p.kernel_outputscale
        )

        # GP predictive distribution
        L = torch.linalg.cholesky(
            K_train + 1e-4 * torch.eye(K_train.shape[0], device=K_train.device)
        )
        alpha = torch.cholesky_solve(p.train_y_norm.unsqueeze(1), L)
        mean_norm = (K_test_train @ alpha).squeeze(-1)

        v = torch.cholesky_solve(K_test_train.T, L)
        # For single test point, get diagonal of variance
        var_norm = torch.diag(K_test - K_test_train @ v)
        std_norm = torch.sqrt(torch.clamp(var_norm, min=1e-8))

        # De-standardize
        mean = mean_norm * p.y_std + p.y_mean
        std = std_norm * p.y_std

        # EI formula
        z = (vmin_b - mean) / (std + ei_epsilon)

        normal = torch.distributions.Normal(
            torch.zeros(1, device=latent.device),
            torch.ones(1, device=latent.device)
        )
        cdf_z = normal.cdf(z)
        pdf_z = torch.exp(normal.log_prob(z))

        ei = (vmin_b - mean) * cdf_z + std * pdf_z

        if use_log_ei:
            ei = torch.log(ei + ei_epsilon)

        return ei.squeeze()

    def project_latent_to_embedding(
        self,
        latent: torch.Tensor,
        denormalize: bool = True
    ) -> torch.Tensor:
        """Project latent point to instruction embedding space.

        Args:
            latent: (10,) or (batch, 10) latent point(s)
            denormalize: If True, return in original embedding scale

        Returns:
            (768,) or (batch, 768) instruction embedding(s)
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        # Project through LatentProjector
        projected = p.latent_projector(latent)

        if denormalize:
            # Convert from normalized space back to original embedding scale
            inst_denom = p.inst_max - p.inst_min
            inst_denom = torch.where(inst_denom == 0, torch.ones_like(inst_denom), inst_denom)
            projected = projected * inst_denom + p.inst_min

        return projected

    def get_latent_bounds(self, sigma: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bounds for latent space optimization.

        Uses training latent statistics to constrain search to +/- N sigma.

        Args:
            sigma: Number of standard deviations for bounds

        Returns:
            (lower_bound, upper_bound) each of shape (latent_dim,)
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params
        lower = p.latent_mean - sigma * p.latent_std
        upper = p.latent_mean + sigma * p.latent_std
        return lower, upper

    def get_latent_features(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor
    ) -> torch.Tensor:
        """Get latent features for embeddings.

        Useful for visualization and initialization.

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
