"""CLI entry point for flow matching training.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python -m study.flow_matching.train \
        --arch mlp --flow icfm --dataset 5k --group ablation-flow \
        --epochs 100 --batch-size 256 --lr 1e-4

Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=1 uv run python -m study.flow_matching.train \
        --arch mlp --flow icfm --dataset 5k --group ablation-flow \
        --epochs 200 --batch-size 256 --lr 1e-4 \
        --resume study/checkpoints/mlp-icfm-5k-none/best.pt

All training runs on GPU 1 (A5000) with CUDA_VISIBLE_DEVICES=1.
"""

import argparse
import logging
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

from study.data import load_all_splits
from study.flow_matching.config import TrainingConfig
from study.flow_matching.models import create_model
from study.flow_matching.trainer import FlowTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        "--scale",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help="Model scale (tiny, small, base)",
    )
    parser.add_argument(
        "--flow",
        type=str,
        required=True,
        help="Flow matching method (icfm, otcfm, spherical, spherical-ot, reflow, si, si-gvp, si-linear)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="gvp",
        choices=["linear", "gvp"],
        help="SI schedule type (only used with --flow=si)",
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
        help="Augmentation method (e.g., none, mixup, noise, mixup+noise)",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Mixup alpha parameter (overrides --aug default, 0=disabled)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std (overrides --aug default, 0=disabled)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="Dimension dropout rate (0.0 = disabled, recommended 0.1)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Wandb configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="flow-matching-study",
        help="Wandb project name",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Mixed precision training
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (FP16) training for faster computation",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    start_time = time.time()

    try:
        args = parse_args()

        # Set random seeds
        set_seed(args.seed)
        logger.info(f"Random seed: {args.seed}")

        # Verify GPU
        device = verify_gpu()

        # Map flow method for SI variants
        # 'si' with schedule='gvp' -> 'si-gvp' for checkpoint naming
        # 'si' with schedule='linear' -> 'si-linear' for checkpoint naming
        flow_method = args.flow
        if args.flow == "si":
            flow_method = f"si-{args.schedule}"
            logger.info(f"SI with {args.schedule} schedule -> flow method: {flow_method}")

        # Create config
        config = TrainingConfig(
            arch=args.arch,
            flow=flow_method,
            dataset=args.dataset,
            aug=args.aug,
            group=args.group,
            scale=args.scale,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            si_schedule=args.schedule if args.flow.startswith("si") else "gvp",
            mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std,
            dropout_rate=args.dropout_rate,
        )

        logger.info(f"Run name: {config.run_name}")
        logger.info(f"Config: {config.to_dict()}")

        # Validate stats path
        config.validate_stats_path()
        logger.info(f"Stats path validated: {config.stats_path}")

        # Check resume path if provided
        resume_path = None
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
            resume_path = args.resume
            logger.info(f"Will resume from: {resume_path}")

        # Load datasets
        train_ds, val_ds, test_ds = load_all_splits(
            size=args.dataset,
            stats_path=config.stats_path,
            return_normalized=True,
        )
        logger.info(
            f"Loaded datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

        # Create model via factory with scale
        model = create_model(args.arch, scale=args.scale)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {args.arch} ({args.scale}), Parameters: {param_count:,}")

        # Create trainer and run
        trainer = FlowTrainer(
            model=model,
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            wandb_project=args.wandb_project,
            resume_path=resume_path,
            test_dataset=test_ds,  # Pass test dataset for final evaluation
            use_amp=args.amp,  # Mixed precision training
        )
        if args.amp:
            logger.info("Mixed precision (AMP) training enabled")
        result = trainer.train()

        # Calculate training time
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)

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
        print(f"Checkpoint: {result['checkpoint_path']}")
        print(f"Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        # Print test metrics if available
        if "test_loss" in result:
            print("-" * 60)
            print("TEST EVALUATION")
            print(f"Test loss: {result['test_loss']:.6f}")
            print(f"Test MSE: {result['test_mse']:.6f} +/- {result['test_mse_std']:.6f}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
