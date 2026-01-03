"""Instruction encoder for LIPO.

Provides:
- GTRInstructionEncoder: GTR-T5-Base encoder wrapper (Vec2Text compatible)
- InstructionVAE: Variational autoencoder with KL regularization for smooth latent space

Self-contained - no imports from other modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


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
        self.device = self._get_device(device)

        self.model = SentenceTransformer(model_name, device=self.device)

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

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

    Architecture:
        Encoder: 768D → 384 → 192 → 2*latent_dim (mu + log_var)
        Decoder: latent_dim → 192 → 384 → 768D (L2 normalized)

    Loss:
        L = recon_loss + beta * KL(q(z|x) || N(0,1))

    The KL regularization ensures:
        - Smooth latent space (gradual transitions)
        - Latents distributed around N(0,1)
        - No holes in the latent space
        - Better generalization to unseen points
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        beta: float = 0.1,
        gamma: float = 0.0,
    ):
        """Initialize VAE.

        Args:
            input_dim: Input embedding dimension (768 for GTR)
            latent_dim: Latent space dimension (64 for GP compatibility)
            beta: KL regularization weight (higher = more regularized)
            gamma: Cycle consistency weight (ensures z ≈ encode(decode(z)))
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
        # Architecture: 768 → 384 → 192 → 2*latent_dim
        # Layer before latent (192) is 3× larger than latent (64) for feature mixing
        # GELU activation for smoother gradients (used in transformers)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, latent_dim * 2),  # mu + log_var
        )

        # Decoder: latent -> 768 (symmetric architecture)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, input_dim),
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

        # Cycle consistency loss: z ≈ encode_mu(decode(z))
        # This prevents "holes" in latent space where different z's decode to similar x
        # but encode back to completely different z's
        cycle_loss = torch.tensor(0.0, device=x.device)
        cycle_cosine = 0.0
        if gamma > 0 and z is not None:
            # Re-encode the reconstructed embedding
            z_recon = self.encode_mu(x_recon)
            # Use cosine similarity for consistency with other losses
            z_cosine = F.cosine_similarity(z, z_recon, dim=-1)
            cycle_loss = (1 - z_cosine).mean()
            cycle_cosine = z_cosine.mean().item()

        # Total loss
        total_loss = recon_loss + beta * kl_loss + gamma * cycle_loss

        loss_dict = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "cycle": cycle_loss.item() if gamma > 0 else 0.0,
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
    """Wrapper combining frozen VAE encoder with trainable adapter.

    Architecture for optimization:
        z (64D) → Adapter → z_gp (10D) → GP → qLogEI

    The adapter compresses VAE latent for efficient GP modeling while
    optimization happens in the full 64D space with gradients flowing
    through the adapter.

    For decoding after optimization:
        z (64D) → VAE decoder → embedding (768D)
    """

    def __init__(self, vae: nn.Module, vae_latent_dim: int = 64, gp_latent_dim: int = 10):
        super().__init__()
        # Freeze VAE (don't register as submodule to avoid saving it twice)
        object.__setattr__(self, '_vae', vae)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.vae_latent_dim = vae_latent_dim
        self.gp_latent_dim = gp_latent_dim

        # Trainable adapter: compresses VAE latents (64D) to GP latents (10D)
        self.adapter = nn.Sequential(
            nn.Linear(vae_latent_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, gp_latent_dim),
        )

    def encode_vae(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embedding to 64D VAE latent (deterministic mu).

        Use this for getting latent for optimization bounds.
        """
        return self._vae.encode_mu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embedding to 10D GP latent (through adapter).

        Pipeline: x (768D) → VAE encoder → z (64D) → adapter → z_gp (10D)

        This is used for GP training.
        """
        z = self._vae.encode_mu(x)
        return self.adapter(z)

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        """Apply adapter to VAE latent.

        Args:
            z: VAE latent tensor of shape (..., 64)

        Returns:
            GP latent tensor of shape (..., 10)
        """
        return self.adapter(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode 64D VAE latent to 768D embedding.

        Args:
            z: VAE latent tensor of shape (..., 64)

        Returns:
            Embedding tensor of shape (..., 768)
        """
        return self._vae.decode(z)
