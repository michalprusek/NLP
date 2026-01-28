"""
Matryoshka Probabilistic Encoder: 768D GTR → z_core (16D) + z_detail (32D).

Key features:
- Residual latent decomposition: z_core (Matryoshka, GP-optimized) + z_detail (fixed during BO)
- Probabilistic output (mu, log_sigma) for VAE training
- Matryoshka structure on z_core: first 4 dims carry most information
- Dropout-based SimCSE augmentation for contrastive learning
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass

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


@dataclass
class EncoderOutput:
    """Output from encoder forward pass."""
    z_core: torch.Tensor      # [B, core_dim] - Matryoshka latent (GP-optimized)
    z_detail: torch.Tensor    # [B, detail_dim] - Detail latent (fixed during BO)
    z_full: torch.Tensor      # [B, core_dim + detail_dim] - Concatenated
    mu_core: torch.Tensor     # [B, core_dim]
    log_sigma_core: torch.Tensor
    mu_detail: torch.Tensor   # [B, detail_dim]
    log_sigma_detail: torch.Tensor


class MatryoshkaEncoder(nn.Module):
    """
    Probabilistic encoder with Matryoshka + Residual Latent representation.

    Outputs:
    - z_core (16D): Matryoshka-structured, used for GP optimization
    - z_detail (32D): Detail residual, fixed during BO but helps reconstruction

    The total 48D latent provides high-fidelity reconstruction while
    GP operates in tractable 16D space.
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()

        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim  # z_core dimension
        self.detail_dim = config.detail_dim  # z_detail dimension
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

        # Output heads for z_core (Matryoshka, GP-optimized)
        self.fc_mu_core = nn.Linear(prev_dim, config.latent_dim)
        self.fc_log_sigma_core = nn.Linear(prev_dim, config.latent_dim)

        # Output heads for z_detail (residual, fixed during BO)
        self.fc_mu_detail = nn.Linear(prev_dim, config.detail_dim)
        self.fc_log_sigma_detail = nn.Linear(prev_dim, config.detail_dim)

        # Initialize log_sigma to produce small initial variance
        nn.init.constant_(self.fc_log_sigma_core.weight, 0.0)
        nn.init.constant_(self.fc_log_sigma_core.bias, -2.0)  # sigma ≈ 0.14
        nn.init.constant_(self.fc_log_sigma_detail.weight, 0.0)
        nn.init.constant_(self.fc_log_sigma_detail.bias, -2.0)

        # Backwards compatibility aliases
        self.fc_mu = self.fc_mu_core
        self.fc_log_sigma = self.fc_log_sigma_core

    def forward(
        self, x: torch.Tensor, return_params: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization trick.

        For backwards compatibility, returns (z_core, mu_core, log_sigma_core).
        Use forward_full() for complete residual latent output.

        Args:
            x: Input embeddings [B, 768]
            return_params: If True, return (z, mu, log_sigma). If False, return z only.

        Returns:
            z: Sampled z_core [B, latent_dim]
            mu: Mean of z_core [B, latent_dim]
            log_sigma: Log std of z_core [B, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu_core(h)
        log_sigma = self.fc_log_sigma_core(h)

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

    def forward_full(self, x: torch.Tensor) -> EncoderOutput:
        """
        Full forward pass returning both z_core and z_detail.

        Args:
            x: Input embeddings [B, 768]

        Returns:
            EncoderOutput with z_core, z_detail, z_full, and all parameters
        """
        h = self.encoder(x)

        # z_core (Matryoshka)
        mu_core = self.fc_mu_core(h)
        log_sigma_core = self.fc_log_sigma_core(h)
        log_sigma_core = torch.clamp(log_sigma_core, min=-10.0, max=2.0)

        # z_detail (residual)
        mu_detail = self.fc_mu_detail(h)
        log_sigma_detail = self.fc_log_sigma_detail(h)
        log_sigma_detail = torch.clamp(log_sigma_detail, min=-10.0, max=2.0)

        # Reparameterization trick
        if self.training:
            sigma_core = torch.exp(log_sigma_core)
            eps_core = torch.randn_like(sigma_core)
            z_core = mu_core + sigma_core * eps_core

            sigma_detail = torch.exp(log_sigma_detail)
            eps_detail = torch.randn_like(sigma_detail)
            z_detail = mu_detail + sigma_detail * eps_detail
        else:
            z_core = mu_core
            z_detail = mu_detail

        # Full latent is concatenation
        z_full = torch.cat([z_core, z_detail], dim=-1)

        return EncoderOutput(
            z_core=z_core,
            z_detail=z_detail,
            z_full=z_full,
            mu_core=mu_core,
            log_sigma_core=log_sigma_core,
            mu_detail=mu_detail,
            log_sigma_detail=log_sigma_detail,
        )

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling (return mu_core only). Backwards compatible."""
        h = self.encoder(x)
        return self.fc_mu_core(h)

    def encode_deterministic_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode without sampling, returning both z_core and z_detail.

        Returns:
            z_core: [B, latent_dim]
            z_detail: [B, detail_dim]
        """
        h = self.encoder(x)
        mu_core = self.fc_mu_core(h)
        mu_detail = self.fc_mu_detail(h)
        return mu_core, mu_detail

    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode to full latent (z_core + z_detail concatenated).

        Returns:
            z_full: [B, latent_dim + detail_dim]
        """
        z_core, z_detail = self.encode_deterministic_full(x)
        return torch.cat([z_core, z_detail], dim=-1)

    def get_matryoshka_embeddings(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Get embeddings at each Matryoshka dimension level.

        Used during training to compute hierarchical reconstruction loss.
        Note: Only operates on z_core (Matryoshka-structured).

        Args:
            x: Input embeddings [B, 768]

        Returns:
            embeddings: List of latent embeddings at each Matryoshka level
                (z_core[:, :4], z_core[:, :8], z_core[:, :16] by default)
            mu: Posterior mean of z_core [B, latent_dim]
            log_sigma: Posterior log std of z_core [B, latent_dim]
        """
        z, mu, log_sigma = self.forward(x)
        return [z[:, :dim] for dim in self.matryoshka_dims], mu, log_sigma

    def get_matryoshka_embeddings_full(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], EncoderOutput]:
        """
        Get Matryoshka embeddings along with full encoder output.

        The key difference: z_detail is appended to each Matryoshka level,
        so the decoder always sees the full detail for better reconstruction.

        Args:
            x: Input embeddings [B, 768]

        Returns:
            embeddings: List of (z_core_prefix + z_detail) at each Matryoshka level
            output: Full EncoderOutput
        """
        output = self.forward_full(x)
        embeddings = []

        for dim in self.matryoshka_dims:
            # z_core prefix (Matryoshka masking)
            z_core_masked = output.z_core.clone()
            z_core_masked[:, dim:] = 0.0

            # Concatenate with full z_detail
            z_full_masked = torch.cat([z_core_masked, output.z_detail], dim=-1)
            embeddings.append(z_full_masked)

        return embeddings, output

    def sample_with_prefix(
        self, x: torch.Tensor, active_dims: int, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample z_core with only first `active_dims` dimensions active.

        Used during coarse-to-fine optimization. Inactive dimensions
        are set to 0 (prior mean).

        Args:
            x: Input embeddings [B, 768]
            active_dims: Number of z_core dimensions to keep active
            n_samples: Number of samples per input

        Returns:
            z: [B * n_samples, latent_dim] with z[:, active_dims:] = 0
        """
        h = self.encoder(x)
        mu = self.fc_mu_core(h)
        log_sigma = self.fc_log_sigma_core(h)
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

    def sample_full_with_prefix(
        self, x: torch.Tensor, active_core_dims: int, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample full latent with prefix masking on z_core.

        Args:
            x: Input embeddings [B, 768]
            active_core_dims: Number of z_core dimensions to keep active
            n_samples: Number of samples per input

        Returns:
            z_full: [B * n_samples, latent_dim + detail_dim]
        """
        h = self.encoder(x)

        # z_core
        mu_core = self.fc_mu_core(h)
        log_sigma_core = torch.clamp(self.fc_log_sigma_core(h), min=-10.0, max=2.0)
        sigma_core = torch.exp(log_sigma_core)

        # z_detail
        mu_detail = self.fc_mu_detail(h)
        log_sigma_detail = torch.clamp(self.fc_log_sigma_detail(h), min=-10.0, max=2.0)
        sigma_detail = torch.exp(log_sigma_detail)

        B = x.shape[0]
        z_samples = []

        for _ in range(n_samples):
            # Sample z_core with masking
            eps_core = torch.randn_like(sigma_core)
            z_core = mu_core + sigma_core * eps_core
            z_core[:, active_core_dims:] = 0.0

            # Sample z_detail (full, no masking)
            eps_detail = torch.randn_like(sigma_detail)
            z_detail = mu_detail + sigma_detail * eps_detail

            z_full = torch.cat([z_core, z_detail], dim=-1)
            z_samples.append(z_full)

        return torch.cat(z_samples, dim=0)
