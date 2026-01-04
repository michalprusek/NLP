"""Instruction encoder for LIPO.

Provides:
- GTRInstructionEncoder: GTR-T5-Base encoder wrapper (Vec2Text compatible)
- InstructionVAE: Variational autoencoder with KL regularization for smooth latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from lipo.config import get_device


class GTRInstructionEncoder:
    """GTR-T5-Base encoder for instruction embeddings.

    Produces L2-normalized 768D embeddings compatible with Vec2Text.
    Uses SentenceTransformer internally.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/gtr-t5-base",
        normalize: bool = True,
        device: str = "auto",
    ):
        """Initialize GTR encoder.

        Args:
            model_name: SentenceTransformer model name
            normalize: L2-normalize embeddings (required for Vec2Text)
            device: Device to use
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.embedding_dim = 768
        self.device = get_device(device)

        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to 768D numpy array."""
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Encode text to 768D tensor on device."""
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )
        return embedding.to(self.device)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch encode texts to (N, 768) numpy array."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )

    def encode_batch_tensor(
        self, texts: List[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Batch encode texts to (N, 768) tensor."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        ).to(self.device)


class InstructionVAE(nn.Module):
    """Variational Autoencoder for instruction embeddings.

    Provides smooth latent space via KL regularization to N(0,1).
    Designed for joint use with GP (for EI optimization) and Vec2Text (for inversion).

    Architecture (default 32D latent):
        Encoder: 768D → 384 → 192 → 96 → 2*32 (mu + log_var)
        Decoder: 32D → 96 → 192 → 384 → 768D (L2 normalized)

    Compression ratios per layer: 2× → 2× → 2× → 3× (gradual, max 3×)

    Loss:
        L = recon_loss + beta * KL(q(z|x) || N(0,1))
        (cycle_loss disabled by default with gamma=0)

    The KL regularization ensures:
        - Smooth latent space (gradual transitions)
        - Latents distributed around N(0,1)
        - No holes in the latent space
        - Better generalization to unseen points
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        beta: float = 0.01,
        gamma: float = 0.0,
    ):
        """Initialize VAE.

        Args:
            input_dim: Input embedding dimension (768 for GTR)
            latent_dim: Latent space dimension (32 by default, 24x compression)
            beta: KL regularization weight (0.01 for tight latent space)
            gamma: Cycle consistency weight (disabled by default)
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma

        # Encoder: 768 -> mu, log_var
        # Using LayerNorm instead of BatchNorm for:
        # - Stability with batch_size=1
        # - Consistent behavior in train/eval modes
        # Architecture: 768 → 384 → 192 → 96 → 2*latent_dim
        # Gradual compression: 2× → 2× → 3× (max 3× per layer)
        # GELU activation for smoother gradients (used in transformers)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 384),  # 768 → 384 (2×)
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 192),  # 384 → 192 (2×)
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 96),  # 192 → 96 (2×)
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, latent_dim * 2),  # 96 → 32 (3×) mu + log_var
        )

        # Decoder: latent -> 768 (symmetric architecture)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96),  # 32 → 96 (3×)
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 192),  # 96 → 192 (2×)
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 384),  # 192 → 384 (2×)
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, input_dim),  # 384 → 768 (2×)
        )

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for GELU activations.

        Uses Xavier/Glorot uniform initialization which works well with GELU.
        Prevents posterior collapse in early epochs by ensuring proper
        gradient flow through the network.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input embeddings (batch, 768) or (768,)

        Returns:
            (mu, log_var) tuple, each (batch, latent_dim) or (latent_dim,)
        """
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
        """Reparameterization trick for sampling.

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            Sampled latent z = mu + std * eps
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to embedding.

        Args:
            z: Latent tensor (batch, latent_dim) or (latent_dim,)

        Returns:
            Reconstructed embedding (batch, 768) or (768,), L2-normalized
        """
        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)

        x_recon = self.decoder(z)
        x_recon = F.normalize(x_recon, p=2, dim=-1)

        if was_1d:
            x_recon = x_recon.squeeze(0)

        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input embeddings (batch, 768)

        Returns:
            (x_recon, mu, log_var, z) tuple
        """
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
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss with optional cycle consistency.

        Loss = recon_loss + beta * KL_loss + gamma * cycle_loss

        Cycle consistency ensures that z ≈ encode_mu(decode(z)), preventing
        the latent space from having "holes" where decoded embeddings map
        back to different latent regions.

        Args:
            x: Original embeddings (batch, 768)
            x_recon: Reconstructed embeddings (batch, 768)
            mu: Latent means
            log_var: Latent log variances
            z: Sampled latent (required if gamma > 0)
            beta: Optional override for KL weight
            gamma: Optional override for cycle consistency weight

        Returns:
            (total_loss, loss_dict) tuple
        """
        if beta is None:
            beta = self.beta
        if gamma is None:
            gamma = self.gamma

        # Reconstruction loss (cosine)
        cosine_sim = F.cosine_similarity(x, x_recon, dim=-1)
        recon_loss = (1 - cosine_sim).mean()

        # KL divergence to N(0,1)
        # KL = -0.5 * sum(1 + log_var - mu^2 - var)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        # Total loss starts with recon + KL
        total_loss = recon_loss + beta * kl_loss

        # Cycle consistency loss: z ≈ encode_mu(decode(z))
        # Only compute if gamma > 0 (skip expensive encode_mu call otherwise)
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
            "kl": kl_loss.item(),
            "cycle": cycle_loss_val,
            "cosine_mean": cosine_sim.mean().item(),
            "cycle_cosine": cycle_cosine,
        }

        return total_loss, loss_dict

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to mean (deterministic, for GP).

        Args:
            x: Input embeddings

        Returns:
            Mean of latent distribution (no sampling)
        """
        mu, _ = self.encode(x)
        return mu

    def sample_latent(self, n_samples: int = 1) -> torch.Tensor:
        """Sample from prior N(0,1).

        Args:
            n_samples: Number of samples

        Returns:
            Samples from N(0,1) of shape (n_samples, latent_dim)
        """
        device = next(self.parameters()).device
        return torch.randn(n_samples, self.latent_dim, device=device)

    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Linear interpolation between two latents.

        Smooth latent space ensures meaningful interpolations.

        Args:
            z1: Start latent
            z2: End latent
            n_steps: Number of interpolation steps

        Returns:
            Tensor of shape (n_steps, latent_dim)
        """
        alphas = torch.linspace(0, 1, n_steps, device=z1.device)
        return torch.stack([(1 - a) * z1 + a * z2 for a in alphas])


class VAEWithAdapter(nn.Module):
    """Wrapper for frozen VAE encoder (no adapter compression).

    Architecture for optimization:
        x (768D) → VAE encoder → z (32D) → GP → qLogEI

    GP operates directly on 32D VAE latent space - no adapter bottleneck.
    This simplifies the architecture and avoids overfitting the adapter
    with limited training data.

    For decoding after optimization:
        z (32D) → VAE decoder → embedding (768D)

    Note: Named "VAEWithAdapter" for API compatibility but adapter is removed.
    """

    def __init__(self, vae: nn.Module, vae_latent_dim: int = 32):
        """Initialize frozen VAE wrapper.

        Args:
            vae: Trained InstructionVAE to wrap (will be frozen)
            vae_latent_dim: VAE latent dimension (32 by default, matches config.latent_dim)
        """
        super().__init__()
        # Freeze VAE (don't register as submodule to avoid saving it twice)
        object.__setattr__(self, '_vae', vae)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.vae_latent_dim = vae_latent_dim
        # No adapter - GP works directly on 32D VAE latent

    def encode_vae(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embedding to VAE latent (deterministic mu).

        Args:
            x: Input embedding (batch, 768) or (768,)

        Returns:
            VAE latent (batch, 32) or (32,)
        """
        return self._vae.encode_mu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embedding to VAE latent for GP.

        Pipeline: x (768D) → VAE encoder → z (32D)

        No adapter compression - GP works on full 32D latent.

        Args:
            x: Input embedding (batch, 768)

        Returns:
            VAE latent (batch, 32) for GP training
        """
        return self._vae.encode_mu(x)

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        """Identity function (no adapter).

        For API compatibility. Returns input unchanged since there's no adapter.

        Args:
            z: VAE latent tensor of shape (..., 32)

        Returns:
            Same tensor unchanged (..., 32)
        """
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode VAE latent to 768D embedding.

        Args:
            z: VAE latent tensor of shape (..., 32)

        Returns:
            Embedding tensor of shape (..., 768)
        """
        return self._vae.decode(z)

    @property
    def device(self) -> torch.device:
        """Get device where the underlying VAE is located."""
        return next(self._vae.parameters()).device

    def to(self, device):
        """Move to device, including the internal VAE.

        Overrides nn.Module.to() because _vae is stored via object.__setattr__
        to avoid double registration, so it won't be moved automatically.
        """
        # Move internal VAE first
        self._vae.to(device)
        # Then call parent to() for any other registered modules
        return super().to(device)

    def cpu(self):
        """Move to CPU, including the internal VAE."""
        return self.to(torch.device('cpu'))

    def cuda(self, device=None):
        """Move to CUDA, including the internal VAE."""
        if device is None:
            device = torch.device('cuda')
        return self.to(device)
