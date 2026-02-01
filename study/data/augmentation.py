"""Data augmentation for flow matching training.

Provides mixup interpolation, Gaussian noise injection, and dimension dropout for
improving generalization on small (1K-10K) SONAR embedding datasets.

CRITICAL: Only augment x1 (data), never x0 (noise).
Augmentation happens BEFORE coupling.sample().

Order of operations: mixup -> noise -> dropout (as per research).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.

    Attributes:
        mixup_alpha: Mixup alpha parameter for Beta distribution.
            0 = disabled, recommended 0.2.
        noise_std: Standard deviation of Gaussian noise.
            0 = disabled, recommended 0.1.
        dropout_rate: Dimension dropout rate (placeholder for 07-02).
            0 = disabled.
    """

    mixup_alpha: float = 0.0
    noise_std: float = 0.0
    dropout_rate: float = 0.0

    @property
    def enabled(self) -> bool:
        """Check if any augmentation is enabled."""
        return self.mixup_alpha > 0 or self.noise_std > 0 or self.dropout_rate > 0


def mixup_embeddings(embeddings: Tensor, alpha: float) -> Tensor:
    """Apply mixup augmentation to embeddings.

    Performs linear interpolation between pairs of embeddings using
    lambda values sampled from Beta(alpha, alpha) distribution.

    Args:
        embeddings: Input embeddings of shape (batch_size, embed_dim).
        alpha: Mixup alpha parameter for Beta distribution.
            Higher values produce more aggressive mixing.

    Returns:
        Mixed embeddings of same shape as input.
    """
    if alpha <= 0:
        return embeddings

    batch_size = embeddings.size(0)
    device = embeddings.device

    # Sample lambda from Beta(alpha, alpha) - one per sample
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample((batch_size,)).to(device)

    # Expand lambda for broadcasting: (batch_size,) -> (batch_size, 1)
    lam = lam.unsqueeze(1)

    # Create random permutation for pairing
    indices = torch.randperm(batch_size, device=device)

    # Linear interpolation: lam * x + (1-lam) * x[indices]
    mixed = lam * embeddings + (1 - lam) * embeddings[indices]

    return mixed


def add_gaussian_noise(embeddings: Tensor, noise_std: float) -> Tensor:
    """Add zero-mean Gaussian noise to embeddings.

    Args:
        embeddings: Input embeddings of shape (batch_size, embed_dim).
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        Noisy embeddings of same shape as input.
    """
    if noise_std <= 0:
        return embeddings

    noise = torch.randn_like(embeddings) * noise_std
    return embeddings + noise


def dimension_dropout(embeddings: Tensor, dropout_rate: float, training: bool = True) -> Tensor:
    """Apply dimension dropout (masking) to embeddings.

    Uses F.dropout which randomly zeros out dimensions with probability dropout_rate.
    This is stochastic masking - it zeros out (masks) random dimensions.
    F.dropout handles scaling by 1/(1-p) automatically to maintain expected values.

    NOTE: F.dropout IS dimension masking - the "dropout/masking" requirement
    in DATA-07 is satisfied because dropout stochastically zeros dimensions.

    Args:
        embeddings: Input embeddings of shape (batch_size, embed_dim).
        dropout_rate: Probability of zeroing each dimension (0.0 = disabled).
        training: Whether in training mode. If False, returns original.

    Returns:
        Masked embeddings of same shape as input.
    """
    if not training or dropout_rate <= 0:
        return embeddings

    return F.dropout(embeddings, p=dropout_rate, training=training)


def augment_batch(
    embeddings: Tensor,
    config: AugmentationConfig,
    training: bool = True,
) -> Tensor:
    """Apply all configured augmentations to a batch.

    Order of operations: mixup -> noise -> dropout (as per research).

    Args:
        embeddings: Input embeddings of shape (batch_size, embed_dim).
        config: Augmentation configuration.
        training: Whether in training mode. If False, returns original.

    Returns:
        Augmented embeddings of same shape as input.
    """
    if not training:
        return embeddings

    # Order: mixup -> noise -> dropout
    result = embeddings

    if config.mixup_alpha > 0:
        result = mixup_embeddings(result, config.mixup_alpha)

    if config.noise_std > 0:
        result = add_gaussian_noise(result, config.noise_std)

    if config.dropout_rate > 0:
        result = dimension_dropout(result, config.dropout_rate, training=training)

    return result


_AUG_PRESETS: dict[str, AugmentationConfig] = {
    "none": AugmentationConfig(),
    "mixup": AugmentationConfig(mixup_alpha=0.2),
    "noise": AugmentationConfig(noise_std=0.1),
    "dropout": AugmentationConfig(dropout_rate=0.1),
    "mixup+noise": AugmentationConfig(mixup_alpha=0.2, noise_std=0.1),
    "all": AugmentationConfig(mixup_alpha=0.2, noise_std=0.1, dropout_rate=0.1),
}


def parse_aug_string(aug: str) -> AugmentationConfig:
    """Parse augmentation string to config.

    Parses aug string like 'mixup', 'noise', 'mixup+noise', 'all' to
    AugmentationConfig with appropriate defaults.

    Args:
        aug: Augmentation string (e.g., 'none', 'mixup', 'noise',
             'dropout', 'mixup+noise', 'all').

    Returns:
        AugmentationConfig with appropriate parameter values.
    """
    return _AUG_PRESETS.get(aug, AugmentationConfig())
