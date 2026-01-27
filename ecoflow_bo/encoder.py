"""
Matryoshka Probabilistic Encoder: 768D GTR → 8D hierarchical latent.

Key features:
- Probabilistic output (mu, log_sigma) for VAE training
- Matryoshka structure: first 2 dims carry most information
- Dropout-based SimCSE augmentation for contrastive learning
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .config import EncoderConfig


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm and GELU."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualDownBlock(nn.Module):
    """Residual block with dimension reduction and projection shortcut."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection shortcut for dimension change
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        h = self.norm(x)
        h = self.fc1(h)
        h = torch.nn.functional.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # Residual with projection
        return h + self.shortcut(x)


class MatryoshkaEncoder(nn.Module):
    """
    Probabilistic encoder with Matryoshka representation learning.

    Outputs mu and log_sigma for VAE reparameterization. The encoder is
    trained with Matryoshka loss to ensure that prefix dimensions (e.g.,
    first 2 dims) carry disproportionate information.

    This enables coarse-to-fine Bayesian optimization:
    - Start GP with dims [0,1] only (2D is easy with 20 points)
    - Progressively unlock [0:4], then [0:8]
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()

        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.matryoshka_dims = config.matryoshka_dims

        # Build encoder with residual connections at every layer
        blocks = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            # Residual block with projection shortcut for dimension change
            blocks.append(ResidualDownBlock(prev_dim, hidden_dim, config.dropout))
            # Additional same-dim residual block for more capacity
            blocks.append(ResidualBlock(hidden_dim, config.dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*blocks)

        # Output heads for mu and log_sigma
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_log_sigma = nn.Linear(prev_dim, config.latent_dim)

        # Initialize log_sigma to produce small initial variance
        nn.init.constant_(self.fc_log_sigma.weight, 0.0)
        nn.init.constant_(self.fc_log_sigma.bias, -2.0)  # sigma ≈ 0.14

    def forward(
        self, x: torch.Tensor, return_params: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization trick.

        Args:
            x: Input embeddings [B, 768]
            return_params: If True, return (z, mu, log_sigma). If False, return z only.

        Returns:
            z: Sampled latent [B, latent_dim]
            mu: Mean [B, latent_dim]
            log_sigma: Log std [B, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)

        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=2.0)

        # Reparameterization trick
        if self.training:
            sigma = torch.exp(log_sigma)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps
        else:
            z = mu  # Use mean for inference

        if return_params:
            return z, mu, log_sigma
        return z

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling (return mu only)."""
        h = self.encoder(x)
        return self.fc_mu(h)

    def get_matryoshka_embeddings(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get embeddings at each Matryoshka dimension level.

        Used during training to compute hierarchical reconstruction loss.

        Args:
            x: Input embeddings [B, 768]

        Returns:
            List of latent embeddings at each Matryoshka level:
            - z[:, :2] for dim=2
            - z[:, :4] for dim=4
            - z[:, :8] for dim=8
        """
        z, mu, log_sigma = self.forward(x)
        return [z[:, :dim] for dim in self.matryoshka_dims], mu, log_sigma

    def sample_with_prefix(
        self, x: torch.Tensor, active_dims: int, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample z with only first `active_dims` dimensions active.

        Used during coarse-to-fine optimization. Inactive dimensions
        are set to 0 (prior mean).

        Args:
            x: Input embeddings [B, 768]
            active_dims: Number of dimensions to keep active
            n_samples: Number of samples per input

        Returns:
            z: [B * n_samples, latent_dim] with z[:, active_dims:] = 0
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=2.0)
        sigma = torch.exp(log_sigma)

        B = x.shape[0]
        z_samples = []

        for _ in range(n_samples):
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps
            # Zero out inactive dimensions
            z[:, active_dims:] = 0.0
            z_samples.append(z)

        return torch.cat(z_samples, dim=0)


class SimCSEAugmentor:
    """
    SimCSE-style augmentation using dropout.

    Two forward passes through the encoder with different dropout masks
    produce two different z samples for the same input. These form
    positive pairs for contrastive learning.
    """

    def __init__(self, encoder: MatryoshkaEncoder):
        self.encoder = encoder

    def get_positive_pairs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get two augmented views of the same input.

        Args:
            x: Input embeddings [B, 768]

        Returns:
            z1, z2: Two different samples from q(z|x) [B, latent_dim]
        """
        # Ensure encoder is in training mode for dropout
        was_training = self.encoder.training
        self.encoder.train()

        z1, _, _ = self.encoder(x)
        z2, _, _ = self.encoder(x)

        if not was_training:
            self.encoder.eval()

        return z1, z2
