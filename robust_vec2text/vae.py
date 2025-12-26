"""Variational Autoencoder for instruction embeddings.

Maps 768D GTR embeddings to/from 32D latent space.
Uses cosine-priority loss for Vec2Text compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class InstructionVAE(nn.Module):
    """Variational Autoencoder for instruction embeddings.

    Architecture:
        Encoder: 768D -> 256 -> 128 -> 32D (mu, logvar)
        Decoder: 32D -> 128 -> 256 -> 768D (L2-normalized)

    The 32D latent dimension provides enough capacity for semantic
    preservation while being small enough for efficient GP optimization.

    Attributes:
        input_dim: Input dimension (768 for GTR embeddings)
        latent_dim: Latent space dimension (32 for GP compatibility)
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
    ):
        """Initialize VAE.

        Args:
            input_dim: Input embedding dimension (768 for GTR)
            latent_dim: Latent space dimension (32 for GP)
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch, 768)

        Returns:
            Tuple of (mu, logvar), each of shape (batch, 32)
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE sampling.

        Args:
            mu: Mean of latent distribution (batch, 32)
            logvar: Log variance of latent distribution (batch, 32)

        Returns:
            Sampled latent vector (batch, 32)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean (deterministic)
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.

        Output is L2-normalized to match GTR embedding format.

        Args:
            z: Latent tensor of shape (batch, 32)

        Returns:
            Reconstruction of shape (batch, 768), L2-normalized
        """
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        x_recon = self.fc5(h)

        # L2 normalize to match GTR embedding format
        x_recon = F.normalize(x_recon, p=2, dim=-1)

        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, sample, decode.

        Args:
            x: Input tensor of shape (batch, 768)

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean) without sampling.

        Args:
            x: Input tensor of shape (batch, 768)

        Returns:
            Latent mean of shape (batch, 32)
        """
        mu, _ = self.encode(x)
        return mu


class VAELoss(nn.Module):
    """Combined loss for VAE with cosine priority.

    Loss = lambda_cosine * L_cosine + lambda_mse * L_mse + lambda_kld * L_kld

    Where:
        L_cosine: 1 - cosine_similarity (priority for Vec2Text)
        L_mse: MSE reconstruction loss (auxiliary)
        L_kld: KL divergence regularization (light smoothing)

    The cosine loss is prioritized because Vec2Text requires
    directional similarity for accurate text reconstruction.
    """

    def __init__(
        self,
        lambda_cosine: float = 1.0,
        lambda_mse: float = 0.1,
        lambda_kld: float = 0.001,
    ):
        """Initialize loss function.

        Args:
            lambda_cosine: Weight for cosine loss (default: 1.0, priority)
            lambda_mse: Weight for MSE loss (default: 0.1, auxiliary)
            lambda_kld: Weight for KL divergence (default: 0.001, light)
        """
        super().__init__()

        self.lambda_cosine = lambda_cosine
        self.lambda_mse = lambda_mse
        self.lambda_kld = lambda_kld

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            x: Original input (batch, 768)
            x_recon: Reconstruction (batch, 768)
            mu: Latent mean (batch, 32)
            logvar: Latent log variance (batch, 32)

        Returns:
            Tuple of (total_loss, components_dict)
        """
        batch_size = x.size(0)

        # Cosine loss: 1 - cosine_similarity
        # This is the priority loss for Vec2Text compatibility
        cosine_sim = F.cosine_similarity(x, x_recon, dim=-1)
        loss_cosine = (1 - cosine_sim).mean()

        # MSE loss (auxiliary)
        loss_mse = F.mse_loss(x_recon, x)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Light regularization for smooth latent space
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = loss_kld / batch_size  # Normalize by batch size

        # Total loss with priority weighting
        total = (
            self.lambda_cosine * loss_cosine
            + self.lambda_mse * loss_mse
            + self.lambda_kld * loss_kld
        )

        components = {
            "cosine": loss_cosine.item(),
            "mse": loss_mse.item(),
            "kld": loss_kld.item(),
            "total": total.item(),
            "cosine_sim_mean": cosine_sim.mean().item(),
        }

        return total, components
