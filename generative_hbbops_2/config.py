"""
Configuration for HyLO2 optimization with LatentProjector.

Extends HyLOConfig with parameters for:
- Joint training with reconstruction loss
- Latent space optimization
"""
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path


@dataclass
class HyLO2Config:
    """Configuration for HyLO2 optimization pipeline with LatentProjector.

    Extends the base configuration with parameters for:
    - Reconstruction loss training
    - Latent space optimization (10D instead of 768D)

    Key new parameters:
        reconstruction_weight: Lambda for reconstruction loss in joint training
        warmup_epochs: Number of GP-only epochs before adding reconstruction loss
        latent_lr: Learning rate for 10D latent space optimization (higher than 768D)
        latent_bounds_sigma: Constrain optimization to +/- N sigma of training latents
    """
    # Data paths
    data_path: str = "/home/prusek/NLP/datasets/hbbops/full_grid_combined.jsonl"
    instructions_path: str = "/home/prusek/NLP/datasets/hbbops/instructions_25.txt"
    exemplars_path: str = "/home/prusek/NLP/datasets/hbbops/examples_25.txt"
    validation_path: str = "/home/prusek/NLP/hbbops/data/validation.json"
    output_dir: str = "/home/prusek/NLP/generative_hbbops_2/results"

    # GP training
    n_initial_samples: int = 4
    gp_train_epochs: int = 3000
    gp_lr: float = 0.01
    gp_patience: int = 10
    gp_min_observations: int = 4

    # Encoder
    encoder_name: str = "sentence-transformers/gtr-t5-base"
    embedding_dim: int = 768

    # Latent space
    latent_dim: int = 10

    # NEW: Reconstruction training parameters
    reconstruction_weight: float = 1.0     # Lambda in loss = -MLL + lambda * recon_loss (equal weight)
    warmup_epochs: int = 500               # GP-only epochs before adding reconstruction

    # NEW: Latent space optimization parameters
    latent_n_steps: int = 500              # Gradient steps in latent space
    latent_lr: float = 0.1                 # Higher LR for 10D (vs 0.01 for 768D)
    latent_convergence_threshold: float = 1e-6
    latent_max_iterations: int = 10
    latent_n_restarts: int = 5
    latent_bounds_sigma: float = 3.0       # Constrain to +/- 3 sigma

    # Vec2Text inversion
    vec2text_num_steps: int = 50
    vec2text_beam_width: int = 4

    # Visualization
    save_visualizations: bool = True
    visualization_dpi: int = 300

    # Device
    device: str = "auto"
    seed: int = 42

    # Gradient stability
    use_log_ei: bool = False
    gradient_clip_norm: Optional[float] = None
    ei_epsilon: float = 1e-8

    # Basin hopping / perturbation
    perturbation_scale: float = 0.1
    use_latent_bounds: bool = True  # Whether to constrain optimization to training distribution

    # Feature extractor architecture
    use_leaky_relu: bool = False
    leaky_relu_slope: float = 0.01

    def __post_init__(self):
        """Validate configuration after initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # -1 means use all samples
        if self.n_initial_samples != -1 and self.n_initial_samples < 4:
            raise ValueError("n_initial_samples must be at least 4 for GP training (or -1 for all)")

        if self.warmup_epochs >= self.gp_train_epochs:
            raise ValueError("warmup_epochs must be less than gp_train_epochs")

        if self.reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative")

        if self.latent_bounds_sigma <= 0:
            raise ValueError("latent_bounds_sigma must be positive")
