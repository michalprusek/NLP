"""Funnel Flow for GTR embedding compression.

Implements surjective normalizing flows with dimensionality reduction
for Latent Space Bayesian Optimization (LSBO).

Architecture:
    768D GTR embedding → Bijective layers → Funnel (768→64D) → Bijective layers → 64D latent
    64D latent → Inverse bijective → Inverse funnel (64→768D) → Inverse bijective → 768D

Key insight: Funnel layers are surjective (lossy forward) but learn to model
the discarded dimensions p(z_discarded | z_kept), enabling good reconstruction.

References:
    - Funnels: Exact maximum likelihood with dimensionality reduction (NeurIPS 2021)
    - Surjectors library: https://surjectors.readthedocs.io
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)


# =============================================================================
# Building Blocks
# =============================================================================

class MLP(nn.Module):
    """Simple MLP with configurable layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module = nn.GELU(),
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation,
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Bijective Layers (dimension-preserving)
# =============================================================================

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows.

    Splits input into two halves, transforms second half conditioned on first.
    Bijective: f(x) = [x1, x2 * exp(s(x1)) + t(x1)]
    """

    def __init__(self, dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2

        # Conditioner: maps first half to scale and translation for second half
        self.conditioner = MLP(
            input_dim=self.split_dim,
            hidden_dims=hidden_dims,
            output_dim=2 * (dim - self.split_dim),  # s and t
        )

        # Initialize to identity transform
        nn.init.zeros_(self.conditioner.net[-1].weight)
        nn.init.zeros_(self.conditioner.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x -> z, returns (z, log_det)."""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]

        # Get scale and translation
        st = self.conditioner(x1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2  # Bound scale for stability

        # Transform
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([x1, z2], dim=-1)

        # Log determinant
        log_det = s.sum(dim=-1)

        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z -> x."""
        z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]

        st = self.conditioner(z1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2

        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([z1, x2], dim=-1)

        return x


class PermutationLayer(nn.Module):
    """Fixed random permutation layer."""

    def __init__(self, dim: int):
        super().__init__()
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', inv_perm)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.perm], torch.zeros(x.size(0), device=x.device)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.inv_perm]


# =============================================================================
# Funnel Layer (dimension-reducing, surjective)
# =============================================================================

class FunnelLayer(nn.Module):
    """Surjective funnel layer for dimensionality reduction.

    Forward: Keeps first n_keep dimensions, discards rest
    Inverse: Samples discarded dimensions from learned conditional p(z_discard | z_keep)

    The key insight: We learn a decoder that models p(z_discarded | z_kept),
    so reconstruction samples from this distribution.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 512],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_discard = input_dim - output_dim

        assert self.n_discard > 0, "Funnel must reduce dimensions"

        # Decoder: models p(z_discarded | z_kept) as Gaussian
        # Outputs mean and log_std for discarded dimensions
        self.decoder = MLP(
            input_dim=output_dim,
            hidden_dims=hidden_dims,
            output_dim=2 * self.n_discard,  # mean and log_std
        )

        logger.info(f"FunnelLayer: {input_dim}D → {output_dim}D (discarding {self.n_discard})")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: keep first output_dim dimensions.

        Returns:
            z_kept: (batch, output_dim)
            log_det: Log determinant contribution (negative log prob of discarded)
        """
        z_kept = x[:, :self.output_dim]
        z_discarded = x[:, self.output_dim:]

        # Get conditional distribution parameters
        params = self.decoder(z_kept)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        # Log probability of discarded dimensions under learned conditional
        dist = Normal(mean, std)
        log_prob = dist.log_prob(z_discarded).sum(dim=-1)

        # For surjective flows, log_det = -log p(z_discard | z_keep)
        # This comes from the change of variables formula for surjections
        log_det = -log_prob

        return z_kept, log_det

    def inverse(self, z_kept: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Inverse pass: sample discarded dimensions from learned conditional.

        Args:
            z_kept: Kept dimensions (batch, output_dim)
            deterministic: If True, use mean instead of sampling

        Returns:
            x: Reconstructed full vector (batch, input_dim)
        """
        params = self.decoder(z_kept)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-6

        if deterministic:
            z_discarded = mean
        else:
            dist = Normal(mean, std)
            z_discarded = dist.rsample()

        x = torch.cat([z_kept, z_discarded], dim=-1)
        return x


# =============================================================================
# Full Funnel Flow
# =============================================================================

@dataclass
class FunnelFlowOutput:
    """Output from funnel flow forward pass."""
    z: torch.Tensor  # Latent representation
    log_det: torch.Tensor  # Log determinant of Jacobian
    log_prob: torch.Tensor  # Log probability under prior


class GTRFunnelFlow(nn.Module):
    """Funnel Flow for GTR embedding compression.

    Architecture:
        Input: 768D GTR embedding
        → Pre-funnel bijective layers (768D)
        → Funnel layer (768D → latent_dim)
        → Post-funnel bijective layers (latent_dim)
        Output: latent_dim latent vector

    Training objective: Maximize log p(x) = log p(z) + log |det J|
    where J is the Jacobian of the transformation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        n_pre_layers: int = 4,
        n_post_layers: int = 4,
        hidden_dims: List[int] = [512, 512],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Pre-funnel bijective layers (in input_dim space)
        self.pre_layers = nn.ModuleList()
        for i in range(n_pre_layers):
            self.pre_layers.append(AffineCouplingLayer(input_dim, hidden_dims))
            self.pre_layers.append(PermutationLayer(input_dim))

        # Funnel layer (dimension reduction)
        self.funnel = FunnelLayer(input_dim, latent_dim, hidden_dims)

        # Post-funnel bijective layers (in latent_dim space)
        self.post_layers = nn.ModuleList()
        for i in range(n_post_layers):
            self.post_layers.append(AffineCouplingLayer(latent_dim, hidden_dims[:1]))
            self.post_layers.append(PermutationLayer(latent_dim))

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"GTRFunnelFlow: {input_dim}D → {latent_dim}D, "
            f"{n_pre_layers} pre + {n_post_layers} post layers, "
            f"{n_params:,} parameters"
        )

    def forward(self, x: torch.Tensor) -> FunnelFlowOutput:
        """Forward pass: embed -> latent.

        Args:
            x: GTR embeddings (batch, 768)

        Returns:
            FunnelFlowOutput with z, log_det, log_prob
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)

        # Pre-funnel bijective layers
        z = x
        for layer in self.pre_layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

        # Funnel (dimension reduction)
        z, log_det = self.funnel(z)
        log_det_total = log_det_total + log_det

        # Post-funnel bijective layers
        for layer in self.post_layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

        # Log probability under standard normal prior
        log_prob_prior = Normal(0, 1).log_prob(z).sum(dim=-1)

        # Total log probability: log p(x) = log p(z) + log |det J|
        log_prob = log_prob_prior + log_det_total

        return FunnelFlowOutput(z=z, log_det=log_det_total, log_prob=log_prob)

    def inverse(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Inverse pass: latent -> embed.

        Args:
            z: Latent vectors (batch, latent_dim)
            deterministic: If True, use mean for funnel inverse

        Returns:
            Reconstructed embeddings (batch, 768)
        """
        # Inverse post-funnel layers (reverse order)
        x = z
        for layer in reversed(self.post_layers):
            x = layer.inverse(x)

        # Inverse funnel (sample discarded dimensions)
        x = self.funnel.inverse(x, deterministic=deterministic)

        # Inverse pre-funnel layers (reverse order)
        for layer in reversed(self.pre_layers):
            x = layer.inverse(x)

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embeddings to latent (convenience method)."""
        return self.forward(x).z

    def decode(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Decode latent to embeddings (convenience method)."""
        return self.inverse(z, deterministic=deterministic)

    def reconstruct(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Full reconstruction: encode then decode."""
        z = self.encode(x)
        return self.decode(z, deterministic=deterministic)


# =============================================================================
# Loss Function
# =============================================================================

class FunnelFlowLoss(nn.Module):
    """Loss function for Funnel Flow training.

    Combines:
    - Negative log-likelihood (NLL): -log p(x) = -log p(z) - log |det J|
    - Reconstruction loss: cosine similarity between input and reconstruction
    """

    def __init__(
        self,
        nll_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        super().__init__()
        self.nll_weight = nll_weight
        self.recon_weight = recon_weight

    def forward(
        self,
        flow_output: FunnelFlowOutput,
        x_original: torch.Tensor,
        x_reconstructed: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute loss.

        Args:
            flow_output: Output from flow forward pass
            x_original: Original embeddings
            x_reconstructed: Reconstructed embeddings

        Returns:
            total_loss: Combined loss
            metrics: Dict with individual loss components
        """
        # NLL loss: -log p(x)
        nll_loss = -flow_output.log_prob.mean()

        # Reconstruction loss: 1 - cosine similarity
        cos_sim = F.cosine_similarity(x_original, x_reconstructed, dim=-1)
        recon_loss = (1 - cos_sim).mean()

        # Total loss
        total_loss = self.nll_weight * nll_loss + self.recon_weight * recon_loss

        metrics = {
            'loss': total_loss.item(),
            'nll': nll_loss.item(),
            'recon': recon_loss.item(),
            'cos_sim': cos_sim.mean().item(),
            'log_det': flow_output.log_det.mean().item(),
        }

        return total_loss, metrics


# =============================================================================
# Training utilities
# =============================================================================

def train_funnel_flow(
    flow: GTRFunnelFlow,
    embeddings: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cuda",
) -> GTRFunnelFlow:
    """Train funnel flow on GTR embeddings.

    Args:
        flow: FunnelFlow model
        embeddings: GTR embeddings (n_samples, 768)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on

    Returns:
        Trained flow model
    """
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    flow = flow.to(device)
    embeddings = embeddings.to(device)

    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = FunnelFlowLoss(nll_weight=1.0, recon_weight=10.0)

    best_cos_sim = 0.0

    for epoch in range(epochs):
        flow.train()
        epoch_metrics = {'loss': 0, 'nll': 0, 'recon': 0, 'cos_sim': 0}
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for (batch,) in pbar:
            optimizer.zero_grad()

            # Forward pass
            output = flow(batch)

            # Reconstruction
            x_recon = flow.decode(output.z, deterministic=True)

            # Loss
            loss, metrics = loss_fn(output, batch, x_recon)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'cos': f"{metrics['cos_sim']:.4f}",
            })

        scheduler.step()

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        if epoch_metrics['cos_sim'] > best_cos_sim:
            best_cos_sim = epoch_metrics['cos_sim']

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}: loss={epoch_metrics['loss']:.4f}, "
                f"cos_sim={epoch_metrics['cos_sim']:.4f} (best: {best_cos_sim:.4f})"
            )

    logger.info(f"Training complete. Best cos_sim: {best_cos_sim:.4f}")
    return flow
