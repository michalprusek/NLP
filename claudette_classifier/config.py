"""Configuration for Claudette binary classifier."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for training and evaluation."""

    # Data
    dataset_path: Path = Path("datasets/tos_converted/tos_dataset.json")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Model
    encoder_name: str = "nlpaueb/legal-bert-base-uncased"
    hidden_dims: list[int] = None  # [512, 256, 128] - will be set in __post_init__
    dropout: float = 0.3
    num_residual_blocks: int = 3

    # Training
    batch_size: int = 32
    learning_rate: float = 2e-5
    encoder_lr: float = 1e-5  # Lower LR for pre-trained encoder
    num_epochs: int = 50
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0

    # Class imbalance techniques
    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_oversampling: bool = True

    # Hardware
    device: str = "auto"  # auto, cuda, mps, cpu

    # Output
    output_dir: Path = Path("results/claudette_classifier")
    save_best_model: bool = True
    log_interval: int = 10

    def __post_init__(self):
        """Set default values after initialization."""
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
