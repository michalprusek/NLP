"""Variational Autoencoder with Weighted Loss Support.

Extends the robust_vec2text VAE with per-sample weighting for
weighted retraining (focusing latent space on high-quality regions).

The core architecture is unchanged:
    Encoder: 768D -> 256 -> 128 -> 32D (mu, logvar)
    Decoder: 32D -> 128 -> 256 -> 768D (L2-normalized)

The key addition is sample_weights parameter in the loss function,
enabling weighted retraining where high-performing prompts have
more influence on the latent space structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


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

    def cycle_consistency_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute cycle-consistency loss: ||E(D(z)) - z||^2.

        This loss forces the VAE latent space to be a "fixed point" for
        the Decode->Encode operation, reducing inversion drift when
        optimizing in latent space and inverting via Vec2Text.

        Args:
            z: Latent vectors of shape (batch, 32)

        Returns:
            Scalar loss tensor (MSE between z and re-encoded z)
        """
        decoded = self.decode(z)  # z -> 768D embedding
        re_encoded = self.get_latent(decoded)  # 768D -> z'
        return F.mse_loss(re_encoded, z)

    def invert_decoder(
        self,
        target_embedding: torch.Tensor,
        n_steps: int = 500,
        lr: float = 0.1,
        early_stop_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """Find z_inv such that decode(z_inv) ≈ target_embedding.

        Uses gradient descent to minimize ||decode(z) - target||^2 + cosine loss.
        This implements the InvBO-style inversion for creating aligned GP training
        samples, reducing the prediction gap.

        The key insight (from InvBO paper) is that using the encoder introduces
        misalignment: encoder(x) → z, but decode(z) != x. By inverting the decoder
        directly, we find z_inv where decode(z_inv) ≈ x, creating aligned triplets.

        Args:
            target_embedding: GTR embedding (768,) to reconstruct
            n_steps: Max optimization steps
            lr: Learning rate for Adam optimizer
            early_stop_threshold: Stop if loss change < threshold

        Returns:
            z_inv: Inverted latent (32,) that reconstructs target_embedding
        """
        device = target_embedding.device

        # Initialize from encoder (warm start) - much faster than random init
        with torch.no_grad():
            if target_embedding.dim() == 1:
                z_init = self.get_latent(target_embedding.unsqueeze(0)).squeeze(0)
            else:
                z_init = self.get_latent(target_embedding).squeeze(0)

        # Clone and require gradients
        z = z_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        # Ensure target is 1D for loss computation
        target = target_embedding.squeeze() if target_embedding.dim() > 1 else target_embedding

        prev_loss = float('inf')
        for step in range(n_steps):
            optimizer.zero_grad()

            # Decode current z
            decoded = self.decode(z.unsqueeze(0)).squeeze(0)

            # Combined loss: MSE + weighted cosine (match VAE training priorities)
            mse_loss = F.mse_loss(decoded, target)
            cosine_loss = 1 - F.cosine_similarity(
                decoded.unsqueeze(0), target.unsqueeze(0)
            ).squeeze()
            loss = mse_loss + 10 * cosine_loss  # Weight cosine heavily (like VAE training)

            loss.backward()
            optimizer.step()

            # Early stopping if converged
            if abs(prev_loss - loss.item()) < early_stop_threshold:
                break
            prev_loss = loss.item()

        return z.detach()


class WeightedVAELoss(nn.Module):
    """VAE loss with per-sample weighting and cycle-consistency support.

    Loss = sum_i w_i * (lambda_cosine * L_cosine_i + lambda_mse * L_mse_i)
           + lambda_kld * L_kld
           + lambda_cycle * L_cycle

    Where:
        w_i: Per-sample weight (higher = more influence)
        L_cosine: 1 - cosine_similarity (priority for Vec2Text)
        L_mse: MSE reconstruction loss (auxiliary)
        L_kld: KL divergence regularization (light smoothing)
        L_cycle: Cycle-consistency loss ||E(D(z)) - z||^2 (reduces inversion drift)

    The cosine loss is prioritized because Vec2Text requires
    directional similarity for accurate text reconstruction.

    The cycle-consistency loss forces the VAE latent space to be a
    "fixed point" for the Decode->Encode operation, dramatically
    improving Vec2Text inversion accuracy.

    Sample weights enable weighted retraining where high-performing
    prompts have more influence on the latent space structure.
    This is key for the COWBOYS methodology.
    """

    def __init__(
        self,
        lambda_cosine: float = 20.0,
        lambda_mse: float = 1.0,
        lambda_kld: float = 0.0025,
        lambda_cycle: float = 2.0,
    ):
        """Initialize loss function.

        Args:
            lambda_cosine: Weight for cosine loss (default: 20.0, priority for GTR)
                GTR embeddings are L2-normalized, so direction matters most.
                High weight ensures VAE preserves angular similarity for Vec2Text.
            lambda_mse: Weight for MSE loss (default: 1.0, auxiliary)
            lambda_kld: Weight for KL divergence (default: 0.0025, with annealing)
                Too low (0.0001) allows VAE to "cheat" by clustering far from origin,
                breaking pCN sampler's N(0,I) assumption. Too high (>0.01) causes
                posterior collapse. 0.0025 is sweet spot per "Optimus" paper.
            lambda_cycle: Weight for cycle-consistency loss (default: 2.0)
                Ensures E(D(z)) ≈ z, critical for Trust Region accuracy.
                Weight 2.0 means latent consistency is 2x more important than MSE.
        """
        super().__init__()

        self.lambda_cosine = lambda_cosine
        self.lambda_mse = lambda_mse
        self.lambda_kld = lambda_kld
        self.lambda_cycle = lambda_cycle

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        cycle_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted total loss.

        Args:
            x: Original input (batch, 768)
            x_recon: Reconstruction (batch, 768)
            mu: Latent mean (batch, 32)
            logvar: Latent log variance (batch, 32)
            sample_weights: Optional per-sample weights (batch,)
                           If None, uses uniform weights.
            cycle_loss: Optional pre-computed cycle-consistency loss.
                        Computed externally as vae.cycle_consistency_loss(mu).

        Returns:
            Tuple of (total_loss, components_dict)
        """
        batch_size = x.size(0)

        # Cosine loss: 1 - cosine_similarity (per sample)
        cosine_sim = F.cosine_similarity(x, x_recon, dim=-1)  # (batch,)
        loss_cosine_per_sample = 1 - cosine_sim  # (batch,)

        # MSE loss (per sample)
        loss_mse_per_sample = torch.mean((x - x_recon) ** 2, dim=-1)  # (batch,)

        # Combined reconstruction loss per sample
        recon_loss_per_sample = (
            self.lambda_cosine * loss_cosine_per_sample
            + self.lambda_mse * loss_mse_per_sample
        )

        # Apply sample weights if provided
        if sample_weights is not None:
            # Ensure weights are normalized
            weights = sample_weights / sample_weights.sum()
            weighted_recon_loss = (recon_loss_per_sample * weights * batch_size).mean()
        else:
            weighted_recon_loss = recon_loss_per_sample.mean()

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Not weighted per sample (regularization should be uniform)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = loss_kld / batch_size

        # Total loss
        total = weighted_recon_loss + self.lambda_kld * loss_kld

        # Add cycle-consistency loss if provided and lambda_cycle > 0
        cycle_loss_value = 0.0
        if cycle_loss is not None and self.lambda_cycle > 0:
            total = total + self.lambda_cycle * cycle_loss
            cycle_loss_value = cycle_loss.item()

        components = {
            "cosine": loss_cosine_per_sample.mean().item(),
            "mse": loss_mse_per_sample.mean().item(),
            "kld": loss_kld.item(),
            "cycle": cycle_loss_value,
            "total": total.item(),
            "cosine_sim_mean": cosine_sim.mean().item(),
        }

        return total, components


# For backward compatibility with robust_vec2text
class VAELoss(WeightedVAELoss):
    """Alias for WeightedVAELoss for backward compatibility."""

    pass
