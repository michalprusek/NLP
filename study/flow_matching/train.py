"""CLI entry point for flow matching training.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python -m study.flow_matching.train \
        --arch mlp --flow icfm --dataset 5k --group ablation-flow \
        --epochs 100 --batch-size 256 --lr 1e-4

All training runs on GPU 1 (A5000) with CUDA_VISIBLE_DEVICES=1.
"""

import argparse
import logging
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

from study.data.dataset import load_all_splits
from study.flow_matching.config import TrainingConfig
from study.flow_matching.trainer import FlowTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimpleVelocityNet(nn.Module):
    """Simple MLP velocity network for testing.

    This is a placeholder model for testing the training pipeline.
    Real models will be added in Phase 3.

    Input: x_t (batch, 1024) + t (batch,)
    Output: velocity (batch, 1024)
    """

    def __init__(self, dim: int = 1024, hidden: int = 512, num_layers: int = 3):
        """Initialize simple velocity network.

        Args:
            dim: Input/output dimension (1024 for SONAR).
            hidden: Hidden layer dimension.
            num_layers: Number of hidden layers.
        """
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Main network
        layers = [nn.Linear(dim + hidden, hidden), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy input tensor (batch, 1024).
            t: Time tensor (batch,) in [0, 1].

        Returns:
            Predicted velocity (batch, 1024).
        """
        # Embed time
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        t_emb = self.time_embed(t)

        # Concatenate and forward
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def verify_gpu() -> torch.device:
    """Verify GPU assignment and return device.

    Prints GPU information and warns if not on expected GPU.

    Returns:
        torch.device for training.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Training requires GPU. "
            "Check CUDA installation and GPU availability."
        )

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

    logger.info(f"GPU Device: {gpu_name}")
    logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")

    # Check if using expected GPU (A5000 or L40S)
    if "A5000" not in gpu_name and "L40S" not in gpu_name:
        warnings.warn(
            f"Expected GPU A5000 or L40S but got '{gpu_name}'. "
            "Training may have different performance characteristics."
        )

    # Warn if CUDA_VISIBLE_DEVICES is not set to 1
    if cuda_visible != "1":
        warnings.warn(
            f"CUDA_VISIBLE_DEVICES={cuda_visible} (expected '1'). "
            "Make sure you're using the correct GPU."
        )

    return torch.device("cuda")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train flow matching model on SONAR embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Architecture name (e.g., mlp, dit, unet)",
    )
    parser.add_argument(
        "--flow",
        type=str,
        required=True,
        help="Flow matching method (e.g., icfm, otcfm)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["1k", "5k", "10k"],
        help="Dataset size",
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="Wandb group for ablation organization",
    )

    # Optional training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Linear warmup steps",
    )

    # Optional augmentation
    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        help="Augmentation method (e.g., none, mixup)",
    )

    # Resume training (placeholder - implemented in Plan 02)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (not yet implemented)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Set random seeds
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Verify GPU
    device = verify_gpu()

    # Create config
    config = TrainingConfig(
        arch=args.arch,
        flow=args.flow,
        dataset=args.dataset,
        aug=args.aug,
        group=args.group,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
    )

    logger.info(f"Run name: {config.run_name}")
    logger.info(f"Config: {config.to_dict()}")

    # Validate stats path
    config.validate_stats_path()
    logger.info(f"Stats path validated: {config.stats_path}")

    # Load datasets
    train_ds, val_ds, test_ds = load_all_splits(
        size=args.dataset,
        stats_path=config.stats_path,
        return_normalized=True,
    )
    logger.info(
        f"Loaded datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # Create model (placeholder - real models in Phase 3)
    model = SimpleVelocityNet(dim=1024, hidden=512, num_layers=3)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Check for resume (placeholder)
    if args.resume:
        logger.warning(
            "--resume not yet implemented. "
            "Will be added in Plan 02 with Wandb integration."
        )

    # Create trainer and run
    trainer = FlowTrainer(model, config, train_ds, val_ds, device)
    result = trainer.train()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run name: {config.run_name}")
    print(f"Epochs run: {result['epochs_run']}")
    print(f"Best val loss: {result['best_val_loss']:.6f}")
    print(f"Final train loss: {result['final_train_loss']:.6f}")
    print(f"Final val loss: {result['final_val_loss']:.6f}")
    print(f"Early stopped: {result['early_stopped']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
