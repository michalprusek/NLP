"""Configuration dataclasses for EcoFlow-BO."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch


# ============================================================================
# Utility functions
# ============================================================================


def ensure_active_dims_list(
    active_dims: Union[int, List[int], None],
    latent_dim: int = 16
) -> Optional[List[int]]:
    """
    Convert active_dims to List[int] format for consistent handling.

    Args:
        active_dims: Which dimensions are active. Can be:
            - None: returns None (use all dimensions)
            - int: returns list(range(active_dims)) - first N dimensions
            - List[int]: returns as-is
        latent_dim: Maximum latent dimension (for validation)

    Returns:
        List of active dimension indices, or None if all dimensions active

    Example:
        >>> ensure_active_dims_list(4)
        [0, 1, 2, 3]
        >>> ensure_active_dims_list([0, 1, 2, 3])
        [0, 1, 2, 3]
        >>> ensure_active_dims_list(None)
        None
    """
    if active_dims is None:
        return None
    if isinstance(active_dims, int):
        if active_dims > latent_dim:
            raise ValueError(
                f"active_dims={active_dims} exceeds latent_dim={latent_dim}"
            )
        return list(range(active_dims))
    return active_dims


def clamp_to_search_bounds(
    z: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """
    Clamp tensor values to search bounds.

    Args:
        z: Tensor to clamp [B, D] or [D]
        lower: Lower bounds [D]
        upper: Upper bounds [D]

    Returns:
        Clamped tensor with same shape as z
    """
    return torch.clamp(z, lower, upper)


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass
class ResidualLatentConfig:
    """Configuration for residual latent decomposition.

    The key innovation: split latent into z_core (GP-optimized) and z_detail (fixed).
    This allows GP to operate in tractable 16D while decoder has 48D capacity.

    z_full = [z_core, z_detail]
           = [16D   , 32D     ] = 48D total
    """
    core_dim: int = 16          # z_core dimension (GP-optimized, Matryoshka)
    detail_dim: int = 32        # z_detail dimension (fixed during BO)

    # During BO, z_detail can be:
    # - "zero": all zeros (simplest, may hurt reconstruction)
    # - "mean": mean of training set z_details
    # - "sample": fresh sample from N(0,I) each iteration
    # - "nearest": copy z_detail from nearest neighbor by z_core (recommended!)
    #              Uses training set to find sample with most similar z_core
    detail_mode: str = "nearest"

    # For "nearest" mode: number of neighbors to average (1 = single nearest)
    n_neighbors: int = 1

    @property
    def full_dim(self) -> int:
        """Total latent dimension for decoder."""
        return self.core_dim + self.detail_dim


@dataclass
class EncoderConfig:
    """Configuration for MatryoshkaEncoder with residual latent output."""
    input_dim: int = 768  # GTR embedding dimension
    latent_dim: int = 16  # z_core dimension (Matryoshka)
    detail_dim: int = 32  # z_detail dimension (residual)
    hidden_dims: List[int] = field(default_factory=lambda: [768, 512, 256, 128, 64, 32])
    dropout: float = 0.1  # Enables SimCSE-style augmentation

    # Matryoshka settings: which prefix dimensions to supervise
    # e.g., [4, 8, 16] means we supervise reconstructions at dims 4, 8, and 16
    # Note: These apply only to z_core, not z_detail
    matryoshka_dims: List[int] = field(default_factory=lambda: [4, 8, 16])
    # Relative importance weights for each Matryoshka level
    # Higher weight on smaller dims encourages more information in early dims
    matryoshka_weights: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])

    @property
    def full_latent_dim(self) -> int:
        """Total latent dimension (core + detail)."""
        return self.latent_dim + self.detail_dim


@dataclass
class DiTVelocityNetConfig:
    """Configuration for DiT-based VelocityNetwork (CFM/Rectified Flow).

    DiT (Diffusion Transformer) architecture with cross-attention to latent tokens.
    Each latent dimension becomes a separate token, enabling selective influence.

    Default config (~150M params) designed for 2x L40S (96GB VRAM).

    Note: condition_dim = core_dim + detail_dim = 48D for residual latent.
    """
    data_dim: int = 768  # GTR embedding dimension
    condition_dim: int = 48  # Full latent dim: z_core(16) + z_detail(32)
    hidden_dim: int = 512  # Token dimension for transformer (512 → ~70M params)
    n_layers: int = 16  # Number of DiT blocks (more depth for capacity)
    n_heads: int = 8  # Attention heads (hidden_dim must be divisible by n_heads)
    mlp_ratio: int = 4  # MLP hidden = hidden_dim * mlp_ratio
    dropout: float = 0.1

    # Multi-token input representation
    # Split 768D into n_input_tokens tokens for better self-attention
    n_input_tokens: int = 12  # 768 / 12 = 64D per token
    input_token_dim: int = 64  # data_dim / n_input_tokens

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.data_dim != self.n_input_tokens * self.input_token_dim:
            raise ValueError(
                f"data_dim ({self.data_dim}) must equal "
                f"n_input_tokens * input_token_dim "
                f"({self.n_input_tokens} * {self.input_token_dim} = "
                f"{self.n_input_tokens * self.input_token_dim})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class DecoderConfig:
    """Configuration for RectifiedFlowDecoder."""
    sigma: float = 0.01  # Noise scale for flow matching
    n_reflow_steps: int = 1  # Number of reflow iterations (0 = no reflow)
    # After reflow, we can use 1-step Euler for decoding
    euler_steps: int = 1  # 1 for rectified, 20-50 for standard
    solver: str = "euler"  # "euler" for rectified, "dopri5" for standard


@dataclass
class TrainingConfig:
    """Configuration for Phase 1 manifold training."""
    batch_size: int = 2048  # Per GPU
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 10

    # Loss weights (fixed, no annealing - simpler and works well)
    cfm_weight: float = 1.0
    kl_weight: float = 0.001
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.05

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bf16 for L40S

    # Checkpointing
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "results/ecoflow_checkpoints"

    # Reflow training (Phase 1b)
    reflow_batch_size: int = 4096  # Larger batch for straight trajectories
    reflow_epochs: int = 50


@dataclass
class GPConfig:
    """Configuration for LatentSpaceGP."""
    # Coarse-to-fine schedule: which dims to activate at each stage
    # Stage 0: dims [0-3], Stage 1: dims [0-7], Stage 2: all 16 dims
    active_dims_schedule: List[List[int]] = field(
        default_factory=lambda: [
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7],
            list(range(16))
        ]
    )
    # Points to collect before advancing to next stage
    points_per_stage: List[int] = field(default_factory=lambda: [10, 15, 30])

    # GP hyperparameters
    noise_prior_mean: float = 0.1
    lengthscale_prior_mean: float = 1.0
    use_ard: bool = True  # Automatic Relevance Determination


@dataclass
class AcquisitionConfig:
    """Configuration for density-aware acquisition."""
    beta: float = 2.0  # UCB exploration parameter
    density_weight: float = 0.5  # Weight for prior density term
    n_candidates: int = 1000  # Candidates to sample
    n_restarts: int = 10  # For gradient-based optimization


@dataclass
class CycleConfig:
    """Configuration for cycle consistency checking."""
    error_threshold: float = 1.0  # Max ||z - z_reencoded|| for valid sample
    max_retries: int = 5  # Max candidates to try if rejected


@dataclass
class EcoFlowConfig:
    """Master configuration for EcoFlow-BO."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    velocity_net: DiTVelocityNetConfig = field(default_factory=DiTVelocityNetConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    residual_latent: ResidualLatentConfig = field(default_factory=ResidualLatentConfig)

    # Data paths
    embeddings_path: str = "datasets/gtr_embeddings_full.pt"
    texts_path: str = "datasets/combined_texts.json"

    # Device
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration consistency."""
        # Ensure Matryoshka dims match latent dim
        assert self.encoder.matryoshka_dims[-1] == self.encoder.latent_dim, \
            f"Last Matryoshka dim must equal latent_dim"

        # Ensure GP active dims schedule is consistent
        assert self.gp.active_dims_schedule[-1] == list(range(self.encoder.latent_dim)), \
            f"Final GP stage must use all {self.encoder.latent_dim} dims"

        # Ensure all GP stages use contiguous prefixes [0, 1, ..., n-1]
        # This is required for proper cycle consistency checking
        for i, stage_dims in enumerate(self.gp.active_dims_schedule):
            expected = list(range(len(stage_dims)))
            assert stage_dims == expected, \
                f"GP stage {i} must use contiguous prefix [0..{len(stage_dims)-1}], got {stage_dims}"

        # Ensure Matryoshka weights sum to ~1
        weight_sum = sum(self.encoder.matryoshka_weights)
        assert 0.99 <= weight_sum <= 1.01, \
            f"Matryoshka weights should sum to 1.0, got {weight_sum}"

        # Ensure residual latent config matches encoder config
        assert self.residual_latent.core_dim == self.encoder.latent_dim, \
            f"residual_latent.core_dim ({self.residual_latent.core_dim}) must equal " \
            f"encoder.latent_dim ({self.encoder.latent_dim})"
        assert self.residual_latent.detail_dim == self.encoder.detail_dim, \
            f"residual_latent.detail_dim ({self.residual_latent.detail_dim}) must equal " \
            f"encoder.detail_dim ({self.encoder.detail_dim})"

        # Ensure velocity_net condition_dim matches full latent
        expected_cond_dim = self.encoder.latent_dim + self.encoder.detail_dim
        assert self.velocity_net.condition_dim == expected_cond_dim, \
            f"velocity_net.condition_dim ({self.velocity_net.condition_dim}) must equal " \
            f"encoder.latent_dim + encoder.detail_dim ({expected_cond_dim})"
