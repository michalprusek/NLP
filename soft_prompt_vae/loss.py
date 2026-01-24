"""Loss functions for Soft-Prompt VAE training.

Implements:
- ELBO loss with reconstruction and KL terms
- Cyclical KL annealing (from Microsoft's paper)
- Free bits constraint to prevent posterior collapse
"""

import logging
import math
from typing import Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from soft_prompt_vae.config import TrainingConfig
from soft_prompt_vae.model import VAEOutput

logger = logging.getLogger(__name__)


@dataclass
class LossOutput:
    """Detailed loss breakdown."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    kl_raw: torch.Tensor  # Before free bits
    beta: float
    active_dims: int  # KL > free_bits
    bow: Optional[torch.Tensor] = None  # Bag-of-Words auxiliary loss
    contrastive: Optional[torch.Tensor] = None  # InfoNCE contrastive loss


class CyclicalAnnealingSchedule:
    """Cyclical KL annealing schedule.

    Based on "Cyclical Annealing Schedule: A Simple Approach to
    Mitigating KL Vanishing" (Fu et al., 2019)

    Schedule:
    - Divides training into num_cycles cycles
    - Each cycle: ramp up beta from 0 to max over ratio fraction
    - Hold at max for remaining fraction
    """

    def __init__(
        self,
        total_steps: int,
        num_cycles: int = 4,
        ratio: float = 0.5,
        beta_max: float = 1.0,
    ):
        """Initialize schedule.

        Args:
            total_steps: Total training steps
            num_cycles: Number of annealing cycles
            ratio: Fraction of cycle spent ramping (0.5 = 50% ramp, 50% hold)
            beta_max: Maximum beta value
        """
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        self.ratio = ratio
        self.beta_max = beta_max

        self.steps_per_cycle = total_steps // num_cycles
        self.ramp_steps = int(self.steps_per_cycle * ratio)

        logger.info(
            f"Cyclical annealing: {num_cycles} cycles, "
            f"{self.ramp_steps} ramp steps per cycle, "
            f"beta_max={beta_max}"
        )

    def get_beta(self, step: int) -> float:
        """Get beta value for current step.

        Args:
            step: Current training step

        Returns:
            Beta value in [0, beta_max]
        """
        if self.total_steps == 0:
            return self.beta_max

        # Position within current cycle
        cycle_pos = step % self.steps_per_cycle

        if cycle_pos < self.ramp_steps:
            # Ramping up
            return self.beta_max * (cycle_pos / self.ramp_steps)
        else:
            # Holding at max
            return self.beta_max


class FreeBitsKL:
    """Free bits constraint for KL divergence.

    Ensures minimum KL per latent dimension to prevent posterior collapse.
    From "Generating Sentences from a Continuous Space" (Bowman et al., 2016)
    """

    def __init__(self, free_bits: float = 2.0, reduction: str = "mean"):
        """Initialize free bits constraint.

        Args:
            free_bits: Minimum KL per dimension (in nats)
            reduction: How to reduce over dimensions ("mean", "sum")
        """
        self.free_bits = free_bits
        self.reduction = reduction

    def __call__(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute KL with free bits constraint.

        Args:
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)

        Returns:
            kl_loss: KL loss with free bits (scalar)
            kl_raw: Raw KL without free bits (scalar)
            active_dims: Number of dimensions with KL > free_bits
        """
        # KL per dimension: 0.5 * (μ² + σ² - 1 - log(σ²))
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)

        # Mean over batch
        kl_per_dim = kl_per_dim.mean(dim=0)  # (latent_dim,)

        # Raw KL (for monitoring)
        kl_raw = kl_per_dim.sum()

        # Apply free bits (max with threshold)
        kl_clamped = torch.clamp(kl_per_dim, min=self.free_bits)

        # Count active dimensions
        active_dims = (kl_per_dim > self.free_bits).sum().item()

        # Reduce over dimensions
        if self.reduction == "mean":
            kl_loss = kl_clamped.mean()
        else:
            kl_loss = kl_clamped.sum()

        return kl_loss, kl_raw, active_dims


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for latent regularization.

    Forces the encoder to produce distinguishable latent representations by
    maximizing agreement between positive pairs (original and augmented views)
    while pushing apart negatives (other samples in the batch).

    Loss formula:
        L = -log(exp(sim(z, z+)/τ) / Σ_j exp(sim(z, z_j)/τ))

    where z+ is the positive pair (augmented view) and z_j are all samples.

    Based on:
    - "Representation Learning with Contrastive Predictive Coding" (Oord et al., 2018)
    - "A Simple Framework for Contrastive Learning" (Chen et al., 2020)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter τ controlling distribution sharpness.
                        Lower values make the loss more discriminative.
            normalize: Whether to L2-normalize latents before computing similarity.
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        # Cache for eye mask (avoid recreating every forward)
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_batch_size: int = 0

    def _get_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get or create cached diagonal mask."""
        if self._cached_mask is None or self._cached_batch_size != batch_size or self._cached_mask.device != device:
            self._cached_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            self._cached_batch_size = batch_size
        return self._cached_mask

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_queue: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Optimized implementation with cached masks and fused operations.

        Args:
            z_anchor: Anchor latents (batch, latent_dim)
            z_positive: Positive pair latents (batch, latent_dim)
            z_queue: Optional queue of negative latents (queue_size, latent_dim)
                    If None, uses in-batch negatives only.

        Returns:
            Scalar InfoNCE loss
        """
        batch_size = z_anchor.size(0)

        if batch_size < 2:
            # Need at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=z_anchor.device, requires_grad=True)

        # Normalize if requested (fused with temperature scaling below)
        if self.normalize:
            z_anchor = F.normalize(z_anchor, dim=1)
            z_positive = F.normalize(z_positive, dim=1)
            if z_queue is not None:
                z_queue = F.normalize(z_queue, dim=1)

        # Compute all similarities in one matmul, then extract positive
        # Full similarity matrix: (batch, batch)
        sim_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature

        # Positive similarities are on diagonal
        pos_sim = sim_matrix.diag()

        # Get cached mask for negatives
        mask = self._get_mask(batch_size, z_anchor.device)

        # Negative similarities (mask out diagonal)
        if z_queue is not None:
            # Add queue negatives
            neg_sim_queue = torch.mm(z_anchor, z_queue.t()) / self.temperature
            neg_sim_batch = sim_matrix.masked_fill(mask, float("-inf"))
            neg_sim = torch.cat([neg_sim_batch, neg_sim_queue], dim=1)
        else:
            neg_sim = sim_matrix.masked_fill(mask, float("-inf"))

        # Concatenate pos as first element for cross_entropy
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_anchor.device)

        # Cross-entropy loss
        return F.cross_entropy(logits, labels)


class MomentumQueue:
    """MoCo-style momentum queue for larger effective negative batch size.

    Maintains a fixed-size queue of recent latent representations to serve
    as negatives in contrastive learning. This decouples the negative sample
    count from batch size, enabling more negatives without memory cost.

    Based on "Momentum Contrast for Unsupervised Visual Representation Learning"
    (He et al., 2020)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        queue_size: int = 4096,
    ):
        """Initialize momentum queue.

        Args:
            latent_dim: Dimensionality of latent vectors
            queue_size: Maximum queue size (should be much larger than batch_size)
        """
        self.latent_dim = latent_dim
        self.queue_size = queue_size

        # Queue storage (on CPU to save GPU memory)
        self._queue: Optional[torch.Tensor] = None
        self._ptr = 0  # Current insertion pointer
        self._filled = False  # True once queue has been filled at least once

    def enqueue(self, z: torch.Tensor) -> None:
        """Add new latents to the queue.

        Args:
            z: Latent vectors to add (batch, latent_dim)
        """
        batch_size = z.size(0)
        z_detached = z.detach().cpu()

        if self._queue is None:
            # Initialize queue
            self._queue = torch.zeros(self.queue_size, self.latent_dim)

        # Insert with wraparound
        if self._ptr + batch_size <= self.queue_size:
            self._queue[self._ptr : self._ptr + batch_size] = z_detached
        else:
            # Wraparound case
            overflow = (self._ptr + batch_size) - self.queue_size
            self._queue[self._ptr:] = z_detached[: self.queue_size - self._ptr]
            self._queue[:overflow] = z_detached[self.queue_size - self._ptr :]

        new_ptr = self._ptr + batch_size
        if new_ptr >= self.queue_size:
            self._filled = True  # Queue has wrapped around at least once
        self._ptr = new_ptr % self.queue_size

    def get_queue(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get current queue contents.

        Args:
            device: Device to move queue to

        Returns:
            Queue tensor or None if empty
        """
        if self._queue is None:
            return None
        return self._queue.to(device)

    def is_full(self) -> bool:
        """Check if queue has been filled at least once."""
        return self._queue is not None and self._filled


def compute_bow_loss(
    bow_logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute Bag-of-Words auxiliary loss.

    Multi-label classification: predict which words appear in the target sequence.
    This forces the latent z to encode semantic content about the output.

    Fully vectorized implementation using scatter_add.

    Args:
        bow_logits: Predicted logits from z (batch, vocab_size)
        target_ids: Target token IDs (batch, seq_len)
        ignore_index: Token ID to ignore (padding/special tokens)

    Returns:
        Scalar loss value (binary cross-entropy)
    """
    batch_size, vocab_size = bow_logits.shape
    device = bow_logits.device

    # Create binary target: 1 if word appears in target, 0 otherwise
    # Vectorized: replace ignore_index with 0 (will be masked), clamp to valid range
    valid_mask = target_ids != ignore_index
    clamped_ids = target_ids.clamp(0, vocab_size - 1)

    # Create batch indices for scatter
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(target_ids)

    # Flatten for scatter
    flat_batch = batch_idx[valid_mask]
    flat_ids = clamped_ids[valid_mask]

    # Scatter to create bow_target (vectorized)
    bow_target = torch.zeros(batch_size, vocab_size, device=device)
    bow_target[flat_batch, flat_ids] = 1.0

    # Binary cross-entropy with logits (more numerically stable)
    loss = F.binary_cross_entropy_with_logits(
        bow_logits,
        bow_target,
        reduction="mean",
    )

    return loss


class SoftPromptVAELoss(nn.Module):
    """Complete loss function for Soft-Prompt VAE.

    Combines:
    - Reconstruction loss (cross-entropy)
    - KL divergence with free bits
    - Cyclical beta annealing
    - Bag-of-Words auxiliary loss (optional)
    - InfoNCE contrastive loss (optional, for CDP-VAE)
    """

    def __init__(
        self,
        config: TrainingConfig,
        total_steps: int,
        bow_loss_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
    ):
        """Initialize loss function.

        Args:
            config: Training configuration
            total_steps: Total training steps for annealing schedule
            bow_loss_weight: Weight for Bag-of-Words auxiliary loss (0 to disable)
            contrastive_weight: Weight for InfoNCE contrastive loss (0 to disable)
            contrastive_temperature: Temperature for InfoNCE loss
        """
        super().__init__()
        self.config = config
        self.bow_loss_weight = bow_loss_weight
        self.contrastive_weight = contrastive_weight

        # Annealing schedule
        self.annealing = CyclicalAnnealingSchedule(
            total_steps=total_steps,
            num_cycles=config.num_cycles,
            ratio=config.cycle_ratio,
            beta_max=config.beta_max,
        )

        # Free bits constraint
        self.free_bits = FreeBitsKL(
            free_bits=config.free_bits,
            reduction="sum",  # Sum over dimensions, then scale
        )

        # InfoNCE contrastive loss
        self.infonce = None
        if contrastive_weight > 0:
            self.infonce = InfoNCELoss(
                temperature=contrastive_temperature,
                normalize=True,
            )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        step: int,
        bow_logits: Optional[torch.Tensor] = None,
        mu_augmented: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Compute total loss.

        Args:
            logits: Predicted logits (batch, seq_len, vocab_size)
            labels: Target labels (batch, seq_len), -100 for ignored
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
            step: Current training step
            bow_logits: Bag-of-Words logits from z (batch, vocab_size), optional
            mu_augmented: Augmented latent means for contrastive loss (batch, latent_dim), optional

        Returns:
            LossOutput with detailed breakdown
        """
        # Reconstruction loss
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        recon_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # KL loss with free bits
        kl_loss, kl_raw, active_dims = self.free_bits(mu, logvar)

        # Scale KL by latent dimension for comparable magnitude
        latent_dim = mu.size(1)
        kl_loss = kl_loss / latent_dim

        # Get current beta
        beta = self.annealing.get_beta(step)

        # Total loss: reconstruction + beta * KL
        total_loss = recon_loss + beta * kl_loss

        # Bag-of-Words auxiliary loss (if enabled and logits provided)
        bow_loss = None
        if self.bow_loss_weight > 0 and bow_logits is not None:
            bow_loss = compute_bow_loss(bow_logits, labels, ignore_index=-100)
            total_loss = total_loss + self.bow_loss_weight * bow_loss

        # InfoNCE contrastive loss (if enabled and augmented latents provided)
        contrastive_loss = None
        if self.contrastive_weight > 0 and self.infonce is not None and mu_augmented is not None:
            contrastive_loss = self.infonce(mu, mu_augmented)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        return LossOutput(
            total=total_loss,
            reconstruction=recon_loss,
            kl=kl_loss,
            kl_raw=kl_raw,
            beta=beta,
            active_dims=active_dims,
            bow=bow_loss,
            contrastive=contrastive_loss,
        )


def compute_vae_loss(
    output: VAEOutput,
    labels: torch.Tensor,
    loss_fn: SoftPromptVAELoss,
    step: int,
) -> LossOutput:
    """Convenience function to compute loss from VAEOutput.

    Args:
        output: VAE forward output
        labels: Target labels
        loss_fn: Loss function instance
        step: Current training step

    Returns:
        LossOutput with detailed breakdown
    """
    # Get mu_augmented if available in output
    mu_augmented = getattr(output, "mu_augmented", None)

    return loss_fn(
        logits=output.logits,
        labels=labels,
        mu=output.mu,
        logvar=output.logvar,
        step=step,
        bow_logits=output.bow_logits,
        mu_augmented=mu_augmented,
    )
