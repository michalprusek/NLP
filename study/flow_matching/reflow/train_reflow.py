"""Train 2-rectified flow using pairs from OT-CFM teacher.

This script:
1. Loads OT-CFM teacher model
2. Generates 10K synthetic pairs via ODE integration
3. Trains new MLP model on these pairs using ReflowCoupling
4. Saves checkpoint to study/checkpoints/mlp-reflow-1k-none/

Usage:
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline uv run python -m study.flow_matching.reflow.train_reflow

The reflow procedure produces straighter ODE paths by training on
deterministically coupled (noise, endpoint) pairs from the teacher.
"""

import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from study.flow_matching.reflow.pair_generator import ReflowPairGenerator
from study.flow_matching.coupling.reflow import ReflowCoupling
from study.flow_matching.evaluate import load_checkpoint
from study.flow_matching.models import create_model
from study.flow_matching.utils import EMAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_or_load_pairs(
    teacher_checkpoint: str,
    n_pairs: int,
    cache_path: str,
    device: torch.device,
    force_regenerate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate or load cached reflow pairs.

    Args:
        teacher_checkpoint: Path to OT-CFM teacher checkpoint.
        n_pairs: Number of pairs to generate.
        cache_path: Path to save/load cached pairs.
        device: Device for generation.
        force_regenerate: If True, regenerate even if cache exists.

    Returns:
        Tuple of (x0, x1) tensors on CPU.
    """
    cache_path = Path(cache_path)

    # Check if cached pairs exist
    if cache_path.exists() and not force_regenerate:
        logger.info(f"Loading cached pairs from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        return cached["x0"], cached["x1"]

    # Load teacher model
    logger.info(f"Loading teacher from {teacher_checkpoint}")
    teacher, _ = load_checkpoint(teacher_checkpoint, "mlp", device)

    # Generate pairs
    logger.info(f"Generating {n_pairs} reflow pairs...")
    generator = ReflowPairGenerator(teacher, n_steps=100)

    # Generate in batches for memory efficiency
    x0, x1 = generator.generate_dataset(
        n_total=n_pairs,
        batch_size=512,
        device=device,
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x0": x0, "x1": x1}, cache_path)
    logger.info(f"Saved pairs to {cache_path}")

    return x0, x1


def train_reflow(
    x0_pairs: torch.Tensor,
    x1_pairs: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda"),
    checkpoint_dir: str = "study/checkpoints/mlp-reflow-1k-none",
    wandb_group: str = "05-reflow",
    wandb_name: str = "mlp-reflow-1k-none",
    stats_path: str = "study/datasets/normalization_stats.pt",
) -> dict:
    """Train 2-rectified flow model on synthetic pairs.

    Args:
        x0_pairs: Noise samples [N, 1024].
        x1_pairs: ODE endpoints [N, 1024].
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        device: Training device.
        checkpoint_dir: Directory for saving checkpoints.
        wandb_group: Wandb group name.
        wandb_name: Wandb run name.
        stats_path: Path to normalization stats.

    Returns:
        Training summary dict.
    """
    # Create model
    model = create_model("mlp")
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Created MLP model with {param_count:,} parameters")

    # Create EMA
    ema = EMAModel(model, decay=0.9999)

    # Create coupling from pairs
    coupling = ReflowCoupling(pair_tensors=(x0_pairs, x1_pairs), batch_size=batch_size)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Calculate steps
    n_pairs = len(x0_pairs)
    steps_per_epoch = (n_pairs + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs

    # Cosine schedule with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Initialize Wandb
    wandb.init(
        project="flow-matching-study",
        group=wandb_group,
        name=wandb_name,
        config={
            "arch": "mlp",
            "flow": "reflow",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_pairs": n_pairs,
            "steps_per_epoch": steps_per_epoch,
        },
    )

    # Training state
    best_train_loss = float("inf")
    checkpoint_path = Path(checkpoint_dir) / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Load stats for checkpoint saving
    stats = None
    if Path(stats_path).exists():
        stats = torch.load(stats_path, weights_only=True)

    logger.info(f"Starting reflow training: {epochs} epochs, {steps_per_epoch} steps/epoch")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Reset coupling for new epoch (reshuffles pairs)
        coupling.reset()

        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step in progress:
            # Get batch from coupling (creates its own x0, x1 from pairs)
            # We pass dummy tensors since coupling ignores them
            dummy = torch.randn(batch_size, 1024, device=device)
            t, x_t, v_target = coupling.sample(dummy, dummy)

            # Move to device (coupling moves internally but let's be explicit)
            x_t = x_t.to(device)
            v_target = v_target.to(device)
            t = t.to(device)

            # Forward
            v_pred = model(x_t, t)

            # Loss
            loss = F.mse_loss(v_pred, v_target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step
            optimizer.step()
            scheduler.step()

            # EMA update
            ema.update(model)

            # Track
            epoch_loss += loss.item()
            global_step += 1

            # Log every 10 steps
            if step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch stats
        avg_loss = epoch_loss / steps_per_epoch

        # Log epoch metrics
        wandb.log(
            {
                "train/epoch_loss": avg_loss,
                "epoch": epoch,
            },
            step=global_step,
        )

        # Save checkpoint if improved
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            logger.info(f"Epoch {epoch}: loss={avg_loss:.6f} (NEW BEST)")

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_train_loss,
                    "normalization_stats": stats,
                    "config": {
                        "arch": "mlp",
                        "flow": "reflow",
                        "n_pairs": n_pairs,
                    },
                },
                checkpoint_path,
            )
        else:
            logger.info(f"Epoch {epoch}: loss={avg_loss:.6f}")

    wandb.finish()

    return {
        "epochs_run": epochs,
        "best_loss": best_train_loss,
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    """Main entry point for reflow training."""
    start_time = time.time()

    # Configuration
    TEACHER_CHECKPOINT = "study/checkpoints/mlp-otcfm-1k-none/best.pt"
    N_PAIRS = 10000  # 10x dataset size per research
    PAIRS_CACHE = "study/datasets/reflow_pairs_1k.pt"
    CHECKPOINT_DIR = "study/checkpoints/mlp-reflow-1k-none"

    # Training hyperparameters (match baselines)
    EPOCHS = 100
    BATCH_SIZE = 256
    LR = 1e-4

    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for reflow training")

    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Generate or load pairs
    x0_pairs, x1_pairs = generate_or_load_pairs(
        teacher_checkpoint=TEACHER_CHECKPOINT,
        n_pairs=N_PAIRS,
        cache_path=PAIRS_CACHE,
        device=device,
    )
    logger.info(f"Pairs ready: x0={x0_pairs.shape}, x1={x1_pairs.shape}")

    # Train
    result = train_reflow(
        x0_pairs=x0_pairs,
        x1_pairs=x1_pairs,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Timing
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("REFLOW TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs: {result['epochs_run']}")
    print(f"Best loss: {result['best_loss']:.6f}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Duration: {minutes}m {seconds}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
