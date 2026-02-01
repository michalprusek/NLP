"""Training configuration for flow matching experiments.

Provides TrainingConfig dataclass with all training hyperparameters,
including locked defaults for EMA, gradient clipping, and early stopping.
"""

import os
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for flow matching training.

    Experiment identification fields:
        arch: Architecture name (e.g., 'mlp', 'dit', 'unet')
        flow: Flow matching method (e.g., 'icfm', 'otcfm')
        dataset: Dataset size (e.g., '1k', '5k', '10k')
        aug: Augmentation method (e.g., 'none', 'mixup')
        group: Wandb group for ablation organization

    Training parameters (configurable):
        epochs: Maximum training epochs
        batch_size: Training batch size
        lr: Learning rate
        warmup_steps: Linear warmup steps

    Locked parameters (do not change):
        ema_decay: EMA decay rate (0.9999)
        grad_clip: Gradient clipping max norm (1.0)
        patience: Early stopping patience (20)
        min_delta: Early stopping minimum improvement (0.0)
        val_frequency: Validation frequency in epochs (1)

    Paths:
        checkpoint_dir: Directory for saving checkpoints
        stats_path: Path to normalization statistics file
    """

    # Experiment identification
    arch: str
    flow: str = "icfm"  # Default to I-CFM for backward compatibility
    dataset: str = ""
    aug: str = ""
    group: str = ""
    scale: str = "small"  # Model scale: tiny, small, base
    seed: int = 42  # Random seed for reproducibility

    # Training parameters (configurable)
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    warmup_steps: int = 1000

    # Locked parameters (do not change)
    ema_decay: float = field(default=0.9999, repr=False)
    grad_clip: float = field(default=1.0, repr=False)
    patience: int = field(default=100, repr=False)
    min_delta: float = field(default=0.0, repr=False)
    val_frequency: int = field(default=1, repr=False)

    # OT-CFM specific parameters (only used when flow='otcfm')
    otcfm_sigma: float = field(default=0.0, repr=False)
    otcfm_reg: float = field(default=0.5, repr=False)
    otcfm_normalize_cost: bool = field(default=True, repr=False)

    # Stochastic Interpolant parameters (only used when flow='si*')
    si_schedule: str = field(default="gvp", repr=False)

    # Augmentation parameters (07-01)
    mixup_alpha: float = field(default=0.0, repr=False)
    noise_std: float = field(default=0.0, repr=False)
    dropout_rate: float = field(default=0.0, repr=False)  # Placeholder for 07-02

    # Paths
    checkpoint_dir: str = "study/checkpoints"
    stats_path: str = "study/datasets/normalization_stats.pt"

    @property
    def run_name(self) -> str:
        """Generate run name from config fields."""
        # Include scale in name if not default 'small'
        if self.scale != "small":
            base = f"{self.arch}-{self.scale}-{self.flow}-{self.dataset}-{self.aug}"
        else:
            base = f"{self.arch}-{self.flow}-{self.dataset}-{self.aug}"
        # Include seed in name if not default 42
        if self.seed != 42:
            base = f"{base}-s{self.seed}"
        return base

    def to_dict(self) -> dict:
        """Convert config to dict for Wandb logging."""
        return {
            "arch": self.arch,
            "flow": self.flow,
            "dataset": self.dataset,
            "aug": self.aug,
            "group": self.group,
            "scale": self.scale,
            "seed": self.seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "ema_decay": self.ema_decay,
            "grad_clip": self.grad_clip,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "val_frequency": self.val_frequency,
            "otcfm_sigma": self.otcfm_sigma,
            "otcfm_reg": self.otcfm_reg,
            "otcfm_normalize_cost": self.otcfm_normalize_cost,
            "si_schedule": self.si_schedule,
            "mixup_alpha": self.mixup_alpha,
            "noise_std": self.noise_std,
            "dropout_rate": self.dropout_rate,
            "checkpoint_dir": self.checkpoint_dir,
            "stats_path": self.stats_path,
            "run_name": self.run_name,
        }

    def validate_stats_path(self) -> None:
        """Validate that stats_path file exists.

        Raises:
            ValueError: If stats_path file does not exist.
        """
        if not os.path.exists(self.stats_path):
            raise ValueError(
                f"Normalization stats file not found: {self.stats_path}. "
                "Run the normalization pipeline first (Phase 01-02)."
            )
