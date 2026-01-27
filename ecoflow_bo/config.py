"""Configuration dataclasses for EcoFlow-BO."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EncoderConfig:
    """Configuration for MatryoshkaEncoder."""
    input_dim: int = 768  # GTR embedding dimension
    latent_dim: int = 8  # Keep at 8D for efficient GP optimization
    hidden_dims: List[int] = field(default_factory=lambda: [768, 512, 256, 128, 64, 32, 16])
    dropout: float = 0.1  # Enables SimCSE-style augmentation

    # Matryoshka settings: which prefix dimensions to supervise
    # e.g., [2, 4, 8] means we supervise reconstructions at dims 2, 4, and 8
    matryoshka_dims: List[int] = field(default_factory=lambda: [2, 4, 8])
    # Relative importance weights for each Matryoshka level
    # Higher weight on smaller dims encourages more information in early dims
    matryoshka_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])


@dataclass
class VelocityNetConfig:
    """Configuration for VelocityNetwork (CFM/Rectified Flow)."""
    data_dim: int = 768  # GTR embedding dimension
    condition_dim: int = 8  # Latent dimension (must match encoder)
    hidden_dim: int = 2048  # Increased from 1024 for larger capacity
    n_layers: int = 12  # Increased from 6 for deeper network
    time_embed_dim: int = 512  # Increased from 256
    dropout: float = 0.1


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

    # Loss weights and annealing schedule
    # Phase 1 (epoch 0-5): CFM with minimal KL, contrastive starts
    # Phase 2 (epoch 5-30): Ramp contrastive to full, continue KL annealing
    # Phase 3 (epoch 30-50): Continue KL ramp to full
    # Phase 4 (epoch 50+): Full loss (all weights at target)
    cfm_weight: float = 1.0
    kl_weight_start: float = 0.0001
    kl_weight_end: float = 0.01
    kl_anneal_start: int = 0
    kl_anneal_end: int = 50

    contrastive_weight_start: float = 0.005  # Start with small nonzero weight
    contrastive_weight_end: float = 0.1
    contrastive_anneal_start: int = 5  # Start earlier to shape latent structure
    contrastive_anneal_end: int = 30  # Shorter ramp for full effect sooner
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
    # Stage 0: dims [0,1], Stage 1: dims [0-3], Stage 2: all 8 dims
    active_dims_schedule: List[List[int]] = field(
        default_factory=lambda: [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]
    )
    # Points to collect before advancing to next stage
    points_per_stage: List[int] = field(default_factory=lambda: [10, 10, 30])

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
    velocity_net: VelocityNetConfig = field(default_factory=VelocityNetConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)

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
