"""
Loss functions for EcoFlow-BO training.

Components:
- CFM Loss: Flow matching loss (handled in decoder)
- KL Loss: KL(q(z|x) || N(0,I))
- Matryoshka Contrastive Loss: SimCSE-style InfoNCE with hierarchical supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .config import EncoderConfig, TrainingConfig


class KLDivergenceLoss(nn.Module):
    """
    KL divergence between encoder posterior q(z|x) and standard normal prior.

    KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log_sigma^2 - mu^2 - sigma^2)

    This regularizes the latent space to be close to N(0,I), which:
    1. Enables sampling new z from prior for exploration
    2. Makes density estimation tractable (log p(z) ≈ -0.5 ||z||^2)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, mu: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence.

        Args:
            mu: Posterior mean [B, latent_dim]
            log_sigma: Posterior log std [B, latent_dim]

        Returns:
            kl: KL divergence loss
        """
        # KL = -0.5 * sum(1 + 2*log_sigma - mu^2 - sigma^2)
        kl = -0.5 * (1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())

        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        elif self.reduction == "none":
            return kl
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning (SimCSE-style).

    Positive pairs: Two dropout-augmented views of the same input
    Negative pairs: All other samples in the batch

    Temperature controls the sharpness of the softmax.
    Lower temperature = harder negatives.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z1: First view [B, latent_dim]
            z2: Second view [B, latent_dim]

        Returns:
            loss: InfoNCE loss
        """
        B = z1.shape[0]
        device = z1.device

        # Normalize embeddings (cosine similarity)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix [B, B]
        # sim[i, j] = cosine(z1[i], z2[j])
        sim = torch.mm(z1, z2.t()) / self.temperature

        # Labels: positive pairs are on diagonal
        labels = torch.arange(B, device=device)

        # Cross entropy loss (both directions)
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.t(), labels)

        return (loss_12 + loss_21) / 2


class MatryoshkaContrastiveLoss(nn.Module):
    """
    Matryoshka-aware contrastive loss.

    Applies InfoNCE at multiple dimension levels to ensure that
    prefix dimensions (e.g., first 2 dims) carry more information.

    This enables coarse-to-fine GP optimization:
    - Train GP on dims [0,1] first (2D is easy)
    - Then expand to [0:4], then [0:8]
    """

    def __init__(
        self,
        matryoshka_dims: List[int],
        matryoshka_weights: List[float],
        temperature: float = 0.05,
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights
        self.temperature = temperature
        self.infonce = InfoNCELoss(temperature)

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Matryoshka contrastive loss.

        Args:
            z1: First view [B, latent_dim]
            z2: Second view [B, latent_dim]

        Returns:
            loss: Weighted sum of InfoNCE at each level
            details: Dict with loss at each level
        """
        total_loss = 0.0
        details = {}

        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            # Get prefix
            z1_prefix = z1[:, :dim]
            z2_prefix = z2[:, :dim]

            # InfoNCE at this level
            loss_level = self.infonce(z1_prefix, z2_prefix)
            total_loss = total_loss + weight * loss_level

            details[f"contrastive_dim{dim}"] = loss_level.item()

        return total_loss, details


class EcoFlowLoss(nn.Module):
    """
    Combined loss for EcoFlow-BO training.

    L_total = L_CFM + λ_KL * L_KL + λ_contrastive * L_contrastive

    With annealing schedule:
    - Epoch 0-20:  CFM only, KL=0.0001
    - Epoch 20-50: Ramp KL to 0.01, add contrastive
    - Epoch 50+:   Full loss
    """

    def __init__(
        self,
        encoder_config: Optional[EncoderConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        super().__init__()
        if encoder_config is None:
            encoder_config = EncoderConfig()
        if training_config is None:
            training_config = TrainingConfig()

        self.encoder_config = encoder_config
        self.training_config = training_config

        self.kl_loss = KLDivergenceLoss()
        self.contrastive_loss = MatryoshkaContrastiveLoss(
            matryoshka_dims=encoder_config.matryoshka_dims,
            matryoshka_weights=encoder_config.matryoshka_weights,
            temperature=training_config.contrastive_temperature,
        )

    def get_loss_weights(self, epoch: int) -> Tuple[float, float, float]:
        """
        Get loss weights for current epoch based on annealing schedule.

        Returns:
            cfm_weight, kl_weight, contrastive_weight
        """
        cfg = self.training_config

        # CFM weight is always 1.0
        cfm_weight = cfg.cfm_weight

        # KL annealing
        if epoch < cfg.kl_anneal_start:
            kl_weight = cfg.kl_weight_start
        elif epoch >= cfg.kl_anneal_end:
            kl_weight = cfg.kl_weight_end
        else:
            # Linear interpolation
            progress = (epoch - cfg.kl_anneal_start) / (cfg.kl_anneal_end - cfg.kl_anneal_start)
            kl_weight = cfg.kl_weight_start + progress * (cfg.kl_weight_end - cfg.kl_weight_start)

        # Contrastive annealing
        if epoch < cfg.contrastive_anneal_start:
            contrastive_weight = cfg.contrastive_weight_start
        elif epoch >= cfg.contrastive_anneal_end:
            contrastive_weight = cfg.contrastive_weight_end
        else:
            progress = (epoch - cfg.contrastive_anneal_start) / (
                cfg.contrastive_anneal_end - cfg.contrastive_anneal_start
            )
            contrastive_weight = (
                cfg.contrastive_weight_start
                + progress * (cfg.contrastive_weight_end - cfg.contrastive_weight_start)
            )

        return cfm_weight, kl_weight, contrastive_weight

    def forward(
        self,
        cfm_loss: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss.

        Args:
            cfm_loss: Flow matching loss from decoder
            mu: Encoder mean [B, latent_dim]
            log_sigma: Encoder log std [B, latent_dim]
            z1, z2: Two views for contrastive learning [B, latent_dim]
            epoch: Current epoch for annealing

        Returns:
            total_loss: Combined loss
            details: Dict with individual loss components
        """
        cfm_w, kl_w, contrastive_w = self.get_loss_weights(epoch)

        # KL loss
        kl_loss = self.kl_loss(mu, log_sigma)

        # Contrastive loss (with Matryoshka weighting)
        contrastive_loss, contrastive_details = self.contrastive_loss(z1, z2)

        # Total loss
        total_loss = cfm_w * cfm_loss + kl_w * kl_loss + contrastive_w * contrastive_loss

        # Detailed logging
        details = {
            "loss_total": total_loss.item(),
            "loss_cfm": cfm_loss.item(),
            "loss_kl": kl_loss.item(),
            "loss_contrastive": contrastive_loss.item(),
            "weight_cfm": cfm_w,
            "weight_kl": kl_w,
            "weight_contrastive": contrastive_w,
            **contrastive_details,
        }

        return total_loss, details


class MatryoshkaReconstructionLoss(nn.Module):
    """
    Matryoshka-aware reconstruction loss.

    Decodes from prefix dimensions and computes reconstruction error.
    Higher weight on smaller prefixes encourages information compression.

    Used during training to ensure prefix dimensions are sufficient
    for reasonable reconstruction.
    """

    def __init__(
        self,
        matryoshka_dims: List[int],
        matryoshka_weights: List[float],
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights

    def forward(
        self,
        decoder,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Matryoshka reconstruction loss.

        Args:
            decoder: RectifiedFlowDecoder
            z: Full latent [B, latent_dim]
            x_target: Target embeddings [B, data_dim]

        Returns:
            loss: Weighted reconstruction loss
            details: Dict with loss at each level
        """
        total_loss = 0.0
        details = {}

        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            # Create masked z with only prefix active
            z_masked = z.clone()
            z_masked[:, dim:] = 0.0

            # Decode with masked z
            x_recon = decoder.decode(z_masked)

            # Cosine distance (1 - cosine_sim) as loss
            cos_sim = F.cosine_similarity(x_recon, x_target, dim=-1).mean()
            loss_level = 1 - cos_sim

            total_loss = total_loss + weight * loss_level
            details[f"recon_dim{dim}"] = loss_level.item()
            details[f"cosine_dim{dim}"] = cos_sim.item()

        return total_loss, details
