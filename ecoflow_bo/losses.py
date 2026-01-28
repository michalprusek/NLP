"""
Loss functions for EcoFlow-BO training.

Components:
- KLDivergenceLoss: KL(q(z|x) || N(0,I)) for VAE regularization
- InfoNCELoss: SimCSE-style contrastive loss
- MatryoshkaCFMLoss: Hierarchical CFM loss at multiple latent dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


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


class MatryoshkaCFMLoss(nn.Module):
    """
    Matryoshka-aware Conditional Flow Matching loss.

    Computes CFM loss at multiple latent dimension levels (4, 8, 16) with
    weighted sum to encourage hierarchical information encoding.

    Loss = 0.2 * CFM(z[:4]) + 0.3 * CFM(z[:8]) + 0.5 * CFM(z[:16])

    This ensures:
    - First 4 dims capture coarse semantic structure
    - Dims 5-8 add finer details
    - Dims 9-16 capture remaining nuances
    """

    def __init__(
        self,
        matryoshka_dims: List[int] = None,
        matryoshka_weights: List[float] = None,
    ):
        super().__init__()
        if matryoshka_dims is None:
            matryoshka_dims = [4, 8, 16]
        if matryoshka_weights is None:
            matryoshka_weights = [0.2, 0.3, 0.5]

        assert len(matryoshka_dims) == len(matryoshka_weights)
        assert abs(sum(matryoshka_weights) - 1.0) < 1e-6

        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights

    def forward(
        self,
        decoder,
        x_target: torch.Tensor,
        z: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Matryoshka CFM loss at multiple latent levels.

        Args:
            decoder: RectifiedFlowDecoder with compute_cfm_loss method
            x_target: Target embeddings [B, data_dim]
            z: Full latent [B, latent_dim]
            t: Optional time values [B] (shared across all levels for consistency)

        Returns:
            loss: Weighted sum of CFM losses
            details: Dict with loss at each level
        """
        B = x_target.shape[0]
        device = x_target.device

        # Sample shared time for all Matryoshka levels (consistency)
        if t is None:
            t = torch.rand(B, device=device)

        total_loss = 0.0
        details = {}

        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            # Mask latent to use only first `dim` dimensions
            z_masked = z.clone()
            z_masked[:, dim:] = 0.0

            # Compute CFM loss with masked latent
            loss_level = decoder.compute_cfm_loss(x_target, z_masked, t=t)

            total_loss = total_loss + weight * loss_level
            details[f"cfm_dim{dim}"] = loss_level.item()

        details["cfm_total"] = total_loss.item()
        return total_loss, details
