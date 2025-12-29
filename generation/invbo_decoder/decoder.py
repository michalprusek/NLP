"""Latent decoder for GP-to-Vec2Text embedding space.

Provides:
- LatentDecoder: 10D -> 768D decoder with L2 normalization
- DecoderCyclicLoss: Cyclic consistency loss with soft tolerance

This addresses the "misalignment problem" from InvBO (NeurIPS 2024):
The encoder-decoder cycle may not perfectly reconstruct due to
Vec2Text's inherent reconstruction gap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LatentDecoder(nn.Module):
    """Decoder from GP latent space to Vec2Text embedding space.

    Architecture (mirror of encoder):
        10D latent
            |
        Linear(10, 32) + ReLU + BatchNorm
            |
        Linear(32, 64) + ReLU + BatchNorm
            |
        Linear(64, 256) + ReLU + BatchNorm
            |
        Linear(256, 768)
            |
        L2 normalization
            |
        768D Vec2Text-compatible embedding

    The L2 normalization is critical for Vec2Text compatibility,
    as GTR embeddings are L2-normalized.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        output_dim: int = 768,
        normalize: bool = True,
    ):
        """Initialize decoder.

        Args:
            latent_dim: Input latent dimension (10)
            output_dim: Output embedding dimension (768)
            normalize: L2-normalize output (required for Vec2Text)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.normalize = normalize

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to embedding.

        Args:
            z: Latent tensor (batch, 10) or (10,)

        Returns:
            Embedding (batch, 768) or (768,), L2-normalized if enabled

        Note:
            For single samples (1D input), temporarily switches to eval mode
            to avoid BatchNorm instability with batch_size=1.
        """
        # Handle single sample (BatchNorm needs batch dimension > 1 in training mode)
        was_1d = z.dim() == 1
        if was_1d:
            z = z.unsqueeze(0)
            was_training = self.decoder.training
            self.decoder.eval()

        embedding = self.decoder(z)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        if was_1d:
            embedding = embedding.squeeze(0)
            if was_training:
                self.decoder.train()

        return embedding


class DecoderCyclicLoss(nn.Module):
    """Cyclic consistency loss for decoder training.

    Implements soft cyclic loss from InvBO paper:
        L_cycle = ||z - encoder(decoder(z))||^2

    With soft tolerance to handle Vec2Text reconstruction gap:
        L_cycle = max(0, ||z - z_recon|| - tolerance)^2

    Additional auxiliary losses:
        - L_embedding: cosine similarity between decoded and target embeddings
        - L_recon: MSE between decoded and target embeddings

    Total loss:
        L = lambda_cycle * L_cycle + lambda_embedding * L_embedding + lambda_recon * L_recon
    """

    def __init__(
        self,
        lambda_cycle: float = 1.0,
        lambda_embedding: float = 0.5,
        lambda_recon: float = 1.0,
        tolerance: float = 0.1,
    ):
        """Initialize loss module.

        Args:
            lambda_cycle: Weight for cyclic loss
            lambda_embedding: Weight for cosine embedding loss
            lambda_recon: Weight for MSE reconstruction loss
            tolerance: Soft tolerance for cyclic loss (accepts some gap)
        """
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_embedding = lambda_embedding
        self.lambda_recon = lambda_recon
        self.tolerance = tolerance

    def forward(
        self,
        z: torch.Tensor,
        z_recon: torch.Tensor,
        embedding_decoded: torch.Tensor,
        embedding_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss.

        Args:
            z: Original latent (batch, 10)
            z_recon: Reconstructed latent after encode(decode(z)) (batch, 10)
            embedding_decoded: Decoded embedding (batch, 768)
            embedding_target: Target embedding from training data (batch, 768)

        Returns:
            (total_loss, loss_dict) with individual loss components
        """
        # Cyclic loss with soft tolerance
        cycle_dist = torch.norm(z - z_recon, dim=-1)  # (batch,)
        cycle_dist_tolerant = F.relu(cycle_dist - self.tolerance)
        loss_cycle = (cycle_dist_tolerant ** 2).mean()

        # Cosine embedding loss (1 - cosine_sim)
        cosine_sim = F.cosine_similarity(embedding_decoded, embedding_target, dim=-1)
        loss_embedding = (1 - cosine_sim).mean()

        # MSE reconstruction loss
        loss_recon = F.mse_loss(embedding_decoded, embedding_target)

        # Total loss
        total_loss = (
            self.lambda_cycle * loss_cycle
            + self.lambda_embedding * loss_embedding
            + self.lambda_recon * loss_recon
        )

        loss_dict = {
            "total": total_loss.item(),
            "cycle": loss_cycle.item(),
            "embedding": loss_embedding.item(),
            "recon": loss_recon.item(),
            "cycle_dist_mean": cycle_dist.mean().item(),
            "cosine_sim_mean": cosine_sim.mean().item(),
        }

        return total_loss, loss_dict


class CyclicLossSimple(nn.Module):
    """Simplified cyclic loss for decoder training.

    Uses only cyclic consistency without auxiliary losses.
    Useful for ablation studies.

    L = ||z - encoder(decoder(z))||^2
    """

    def __init__(self, tolerance: float = 0.0):
        """Initialize loss.

        Args:
            tolerance: Soft tolerance (0 = strict MSE)
        """
        super().__init__()
        self.tolerance = tolerance

    def forward(
        self,
        z: torch.Tensor,
        z_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cyclic loss.

        Args:
            z: Original latent (batch, 10)
            z_recon: Reconstructed latent (batch, 10)

        Returns:
            Scalar loss
        """
        if self.tolerance > 0:
            cycle_dist = torch.norm(z - z_recon, dim=-1)
            cycle_dist_tolerant = F.relu(cycle_dist - self.tolerance)
            return (cycle_dist_tolerant ** 2).mean()
        else:
            return F.mse_loss(z, z_recon)
