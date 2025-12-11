"""
Configuration for HyLO optimization.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass
class HyLOConfig:
    """Configuration for HyLO optimization pipeline.

    Attributes:
        data_path: Path to full_grid_combined.jsonl
        instructions_path: Path to instructions file
        exemplars_path: Path to exemplars file
        output_dir: Directory for results and visualizations

        n_initial_samples: Number of random samples for GP training

        encoder_name: HuggingFace model name for GTR encoder
        embedding_dim: Dimension of embeddings (768 for GTR-base)

        strategy: Optimization strategy ("coordinate_descent" or "gumbel_softmax")

        cd_n_steps: Gradient steps per coordinate descent iteration
        cd_lr: Learning rate for coordinate descent
        cd_convergence_threshold: EI improvement threshold for convergence
        cd_max_iterations: Maximum coordinate descent iterations

        gs_n_steps: Total steps for Gumbel-Softmax optimization
        gs_lr: Learning rate for Gumbel-Softmax
        gs_initial_temperature: Starting temperature
        gs_final_temperature: Final temperature after annealing
        gs_anneal_rate: Temperature decay rate per step

        vec2text_num_steps: Correction iterations for Vec2Text
        vec2text_beam_width: Beam search width

        gp_train_epochs: Maximum GP training epochs
        gp_lr: Learning rate for GP training
        gp_patience: Early stopping patience

        save_visualizations: Whether to save PNG visualizations
        visualization_dpi: DPI for saved images

        device: Device for computation ("auto", "cuda", "cpu", "mps")
        seed: Random seed for reproducibility
    """
    # Data paths
    data_path: str = "/home/prusek/NLP/datasets/hbbops/full_grid_combined.jsonl"
    instructions_path: str = "/home/prusek/NLP/datasets/hbbops/instructions_25.txt"
    exemplars_path: str = "/home/prusek/NLP/datasets/hbbops/examples_25.txt"
    validation_path: str = "/home/prusek/NLP/hbbops/data/validation.json"
    output_dir: str = "/home/prusek/NLP/generative_hbbops/results"

    # GP training
    n_initial_samples: int = 4
    gp_train_epochs: int = 3000
    gp_lr: float = 0.01
    gp_patience: int = 10
    gp_min_observations: int = 4

    # Encoder
    encoder_name: str = "sentence-transformers/gtr-t5-base"
    embedding_dim: int = 768

    # Optimization strategy
    strategy: Literal["coordinate_descent", "gumbel_softmax"] = "coordinate_descent"

    # Coordinate Descent (Strategy A)
    cd_n_steps: int = 500
    cd_lr: float = 0.01
    cd_convergence_threshold: float = 1e-6
    cd_max_iterations: int = 10

    # Gumbel-Softmax (Strategy B)
    gs_n_steps: int = 1000
    gs_lr: float = 0.01
    gs_initial_temperature: float = 5.0
    gs_final_temperature: float = 0.1
    gs_anneal_rate: float = 0.99

    # Vec2Text inversion
    vec2text_num_steps: int = 50
    vec2text_beam_width: int = 4

    # Latent space
    latent_dim: int = 10

    # Visualization
    save_visualizations: bool = True
    visualization_dpi: int = 300

    # Device
    device: str = "auto"
    seed: int = 42

    # Gradient stability (for fixing vanishing gradients in EI optimization)
    use_log_ei: bool = False                     # Use log(EI) instead of EI
    gradient_clip_norm: Optional[float] = None   # Gradient clipping (None = disabled)
    ei_epsilon: float = 1e-8                     # Epsilon in EI computation

    # Basin hopping / multi-start
    perturbation_scale: float = 0.1              # Noise scale for basin hopping
    cd_n_restarts: int = 5                       # Number of random restarts

    # Feature extractor architecture
    use_leaky_relu: bool = False                 # Use LeakyReLU instead of ReLU
    leaky_relu_slope: float = 0.01               # Negative slope for LeakyReLU

    def __post_init__(self):
        """Validate configuration after initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # -1 means use all samples
        if self.n_initial_samples != -1 and self.n_initial_samples < 4:
            raise ValueError("n_initial_samples must be at least 4 for GP training (or -1 for all)")

        if self.strategy not in ("coordinate_descent", "gumbel_softmax"):
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if self.gs_final_temperature >= self.gs_initial_temperature:
            raise ValueError("gs_final_temperature must be less than gs_initial_temperature")
