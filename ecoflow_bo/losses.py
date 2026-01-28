"""
Loss functions for EcoFlow-BO training.

Components:
- KLDivergenceLoss: KL(q(z|x) || N(0,I)) for VAE regularization
- InfoNCELoss: SimCSE-style contrastive loss
- MatryoshkaCFMLoss: Hierarchical CFM loss at multiple latent dimensions
- ResidualMatryoshkaCFMLoss: Matryoshka on z_core + full z_detail loss (KEY INNOVATION!)
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


class ResidualMatryoshkaCFMLoss(nn.Module):
    """
    Residual Matryoshka CFM loss: The key innovation for high-quality BO.

    Combines Matryoshka structure on z_core with full z_detail for reconstruction:

    z_full = [z_core (16D) | z_detail (32D)] = 48D total

    Training strategy:
    1. loss_4:  z[:4] active, rest=0  → Coarse semantics (weight=1.0)
    2. loss_8:  z[:8] active, rest=0  → Medium detail (weight=0.5)
    3. loss_16: z[:16] active, rest=0 → Fine z_core (weight=0.25)
    4. loss_48: z[:48] active         → Full reconstruction (weight=0.1)

    Why this matters for BO:
    - GP starts with only 4D → needs strong signal from first 4 dims
    - High weight on loss_4 forces encoder to pack meaning into first dims
    - z_detail (32D) learns "residual" for perfect reconstruction

    Analogy: Like JPEG - loss_4 is DC coefficient, loss_48 is full image.
    """

    def __init__(
        self,
        core_dim: int = 16,
        detail_dim: int = 32,
        matryoshka_dims: List[int] = None,
        matryoshka_weights: List[float] = None,
        full_weight: float = 0.1,
    ):
        super().__init__()
        if matryoshka_dims is None:
            matryoshka_dims = [4, 8, 16]
        if matryoshka_weights is None:
            # Higher weight on early dims = more info packed there
            matryoshka_weights = [1.0, 0.5, 0.25]

        self.core_dim = core_dim
        self.detail_dim = detail_dim
        self.full_dim = core_dim + detail_dim
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights
        self.full_weight = full_weight

    def forward(
        self,
        decoder,
        x_target: torch.Tensor,
        z_full: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Residual Matryoshka CFM loss.

        Args:
            decoder: RectifiedFlowDecoder with compute_cfm_loss method
            x_target: Target embeddings [B, 768]
            z_full: Full latent [B, 48] = [z_core (16D), z_detail (32D)]
            t: Optional shared time [B]

        Returns:
            loss: Weighted sum of all CFM losses
            details: Dict with loss at each level
        """
        B = x_target.shape[0]
        device = x_target.device

        # Sample shared time for consistency across all levels
        if t is None:
            t = torch.rand(B, device=device)

        total_loss = 0.0
        details = {}

        # 1. Matryoshka levels on z_core (z_detail = 0)
        # This teaches the decoder to work with partial z_core
        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            # Only first `dim` of z_core active, z_detail = 0
            z_masked = torch.zeros_like(z_full)
            z_masked[:, :dim] = z_full[:, :dim]

            loss_level = decoder.compute_cfm_loss(x_target, z_masked, t=t)
            total_loss = total_loss + weight * loss_level
            details[f"cfm_core_{dim}"] = loss_level.item()

        # 2. Full reconstruction (z_core + z_detail)
        # This teaches z_detail to capture what z_core misses
        loss_full = decoder.compute_cfm_loss(x_target, z_full, t=t)
        total_loss = total_loss + self.full_weight * loss_full
        details["cfm_full_48"] = loss_full.item()

        # Normalize by total weight (optional, for consistent scale)
        total_weight = sum(self.matryoshka_weights) + self.full_weight
        total_loss = total_loss / total_weight

        details["cfm_total"] = total_loss.item()
        return total_loss, details


class ResidualKLLoss(nn.Module):
    """
    KL divergence for residual latent: KL on z_core + KL on z_detail.

    Both z_core and z_detail are regularized to N(0,I), but can have
    different weights to control their behavior.
    """

    def __init__(
        self,
        core_weight: float = 1.0,
        detail_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.core_weight = core_weight
        self.detail_weight = detail_weight
        self.kl = KLDivergenceLoss(reduction=reduction)

    def forward(
        self,
        mu_core: torch.Tensor,
        log_sigma_core: torch.Tensor,
        mu_detail: torch.Tensor,
        log_sigma_detail: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute KL for both z_core and z_detail.

        Returns:
            total_kl: Weighted sum of KL losses
            details: Dict with individual KL values
        """
        kl_core = self.kl(mu_core, log_sigma_core)
        kl_detail = self.kl(mu_detail, log_sigma_detail)

        total = self.core_weight * kl_core + self.detail_weight * kl_detail

        return total, {
            "kl_core": kl_core.item(),
            "kl_detail": kl_detail.item(),
            "kl_total": total.item(),
        }
