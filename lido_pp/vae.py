"""
Variational Autoencoder for LID-O++.

Standalone VAE implementation for instruction embeddings.
Copied from lipo/encoder.py to avoid botorch dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class InstructionVAE(nn.Module):
    """Variational Autoencoder for instruction embeddings.

    Provides smooth latent space via KL regularization to N(0,1).

    Architecture (32D latent):
        Encoder: 768D -> 512 -> 256 -> 128 -> 2*32 (mu + log_var)
        Decoder: 32D -> 128 -> 256 -> 512 -> 768D (L2 normalized)

    Loss:
        L = (1-mse_weight)*cosine_loss + mse_weight*mse_loss + beta * KL
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        beta: float = 0.005,
        gamma: float = 0.0,
        mse_weight: float = 0.2,
    ):
        """Initialize VAE.

        Args:
            input_dim: Input embedding dimension (768 for GTR/GritLM)
            latent_dim: Latent space dimension (32 by default)
            beta: KL regularization weight
            gamma: Cycle consistency weight (disabled by default)
            mse_weight: Weight for MSE in reconstruction loss (0.2 = 20% MSE + 80% cosine)
        """
        super().__init__()

        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError(f"mse_weight must be in [0, 1], got {mse_weight}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.mse_weight = mse_weight

        # Encoder: 768 -> 512 -> 256 -> 128 -> 2*latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim * 2),
        )

        # Decoder: latent -> 128 -> 256 -> 512 -> 768
        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for GELU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)

        if was_1d:
            mu = mu.squeeze(0)
            log_var = log_var.squeeze(0)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to embedding (L2 normalized)."""
        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)

        x_recon = self.decoder_layers(z)
        x_recon = F.normalize(x_recon, p=2, dim=-1)

        if was_1d:
            x_recon = x_recon.squeeze(0)

        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        mse_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss."""
        if beta is None:
            beta = self.beta
        if gamma is None:
            gamma = self.gamma
        if mse_weight is None:
            mse_weight = self.mse_weight

        # Reconstruction loss
        cosine_sim = F.cosine_similarity(x, x_recon, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        mse_loss = F.mse_loss(x, x_recon)
        recon_loss = (1 - mse_weight) * cosine_loss + mse_weight * mse_loss

        # KL divergence to N(0,1)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        total_loss = recon_loss + beta * kl_loss

        # Cycle consistency (optional)
        cycle_loss_val = 0.0
        cycle_cosine = 0.0
        if gamma > 0 and z is not None:
            z_recon = self.encode_mu(x_recon)
            z_cosine = F.cosine_similarity(z, z_recon, dim=-1)
            cycle_loss = (1 - z_cosine).mean()
            cycle_loss_val = cycle_loss.item()
            cycle_cosine = z_cosine.mean().item()
            total_loss = total_loss + gamma * cycle_loss

        loss_dict = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl": kl_loss.item(),
            "cycle": cycle_loss_val,
            "cosine_mean": cosine_sim.mean().item(),
            "cycle_cosine": cycle_cosine,
        }

        return total_loss, loss_dict

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to mean (deterministic, no sampling)."""
        mu, _ = self.encode(x)
        return mu

    def sample_latent(self, n_samples: int = 1) -> torch.Tensor:
        """Sample from prior N(0,1)."""
        device = next(self.parameters()).device
        return torch.randn(n_samples, self.latent_dim, device=device)

    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Linear interpolation between two latents."""
        alphas = torch.linspace(0, 1, n_steps, device=z1.device)
        return torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
