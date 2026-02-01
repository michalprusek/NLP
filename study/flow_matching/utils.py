"""Training utilities for flow matching experiments.

Provides EarlyStopping, EMAModel, cosine schedule with warmup, and checkpoint utilities.
"""

import copy
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Tracks the best validation loss and counts epochs without improvement.
    When the counter reaches patience, training should stop.

    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement required to reset counter.
        counter: Current count of epochs without improvement.
        best_loss: Best validation loss seen so far.
        should_stop: Whether training should stop.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if validation loss improved.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if loss improved (new best), False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # Did not improve

    def reset(self) -> None:
        """Reset early stopping state for reuse."""
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model parameters that are updated
    with exponential moving average at each training step.

    Attributes:
        decay: EMA decay rate (default 0.9999).
        shadow: Dictionary of shadow parameters.
        backup: Backup of original parameters for restore().
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """Initialize EMA model.

        Args:
            model: Model to track (should already be on target device).
            decay: EMA decay rate.
        """
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow parameters from current model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters.

        Args:
            model: Model with updated parameters.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module) -> None:
        """Apply shadow parameters to model.

        Backs up original parameters before applying.

        Args:
            model: Model to apply shadow parameters to.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original parameters from backup.

        Use after apply() to restore original model state.

        Args:
            model: Model to restore parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get shadow state dict for checkpointing.

        Returns:
            Deep copy of shadow parameters.
        """
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load shadow state dict from checkpoint.

        Args:
            state_dict: Shadow parameters to load.
        """
        self.shadow = copy.deepcopy(state_dict)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create schedule with linear warmup and cosine decay.

    Learning rate linearly increases from 0 to base_lr during warmup,
    then decays following a cosine curve to min_lr_ratio * base_lr.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as fraction of base_lr.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Checkpoint Utilities
# =============================================================================


def get_checkpoint_path(run_name: str) -> str:
    """Get checkpoint path for a run.

    Args:
        run_name: Run name (e.g., 'mlp-icfm-5k-none').

    Returns:
        Path to best checkpoint: study/checkpoints/{run_name}/best.pt
    """
    return f"study/checkpoints/{run_name}/best.pt"


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    best_loss: float,
    config: Dict[str, Any],
    stats: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Save training checkpoint.

    Saves all training state required for resumption:
    - Model weights
    - EMA shadow parameters
    - Optimizer state
    - Scheduler state
    - Training progress

    Args:
        path: Path to save checkpoint.
        epoch: Current training epoch (0-indexed).
        model: Model being trained.
        ema: EMA model wrapper.
        optimizer: Optimizer with state.
        scheduler: Learning rate scheduler.
        best_loss: Best validation loss achieved.
        config: Training configuration dict.
        stats: Optional normalization stats for consistency verification.
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "config": config,
    }

    # Include normalization stats if provided
    if stats is not None:
        checkpoint["normalization_stats"] = stats

    # Save checkpoint
    torch.save(checkpoint, path)

    # Verify file exists and has content
    if not os.path.exists(path):
        raise RuntimeError(f"Failed to save checkpoint: {path} does not exist")

    file_size = os.path.getsize(path)
    if file_size == 0:
        raise RuntimeError(f"Failed to save checkpoint: {path} is empty")

    logger.info(f"Saved checkpoint: {path} ({file_size:,} bytes)")


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> Tuple[int, float]:
    """Load training checkpoint.

    CRITICAL: State restoration order:
    1. Load model.state_dict() FIRST (model must exist and be on device)
    2. Load EMA shadow state SECOND (after model loaded)
    3. Load optimizer state THIRD (after model parameters accessible)
    4. Load scheduler state LAST (after optimizer created)

    This order ensures optimizer state tensors match model parameters.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into (must be on device).
        ema: EMA wrapper to load shadow state into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to move tensors to.

    Returns:
        Tuple of (start_epoch, best_loss) where start_epoch is epoch+1
        (the next epoch to train).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint (weights_only=False because we have config dict)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # 1. Load model state FIRST
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.debug("Loaded model state dict")

    # 2. Load EMA shadow state SECOND
    ema.load_state_dict(checkpoint["ema_shadow"])
    logger.debug("Loaded EMA shadow state")

    # 3. Load optimizer state THIRD
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.debug("Loaded optimizer state dict")

    # 4. Load scheduler state LAST
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.debug("Loaded scheduler state dict")

    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

    logger.info(
        f"Loaded checkpoint from epoch {epoch}, best_loss={best_loss:.6f}"
    )

    # Return next epoch to train
    return epoch + 1, best_loss


def load_checkpoint_with_stats_check(
    path: str,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    current_stats_path: Optional[str] = None,
) -> Tuple[int, float]:
    """Load checkpoint with normalization stats consistency check.

    Wrapper around load_checkpoint that verifies normalization stats
    haven't changed since the checkpoint was created. This helps detect
    data pipeline changes that could cause training inconsistencies.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        ema: EMA wrapper to load shadow state into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to move tensors to.
        current_stats_path: Path to current normalization stats file.
            If None, stats check is skipped.

    Returns:
        Tuple of (start_epoch, best_loss).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint once (weights_only=False because we have config dict)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # 1. Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.debug("Loaded model state dict")

    # 2. Load EMA shadow state
    ema.load_state_dict(checkpoint["ema_shadow"])
    logger.debug("Loaded EMA shadow state")

    # 3. Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.debug("Loaded optimizer state dict")

    # 4. Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.debug("Loaded scheduler state dict")

    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    start_epoch = epoch + 1

    logger.info(f"Loaded checkpoint from epoch {epoch}, best_loss={best_loss:.6f}")

    # Skip stats check if no current stats path provided
    if current_stats_path is None:
        logger.debug("Stats consistency check: SKIPPED (no stats path)")
        return start_epoch, best_loss

    saved_stats = checkpoint.get("normalization_stats")
    if saved_stats is None:
        logger.warning("Stats consistency check: SKIPPED (no stats in checkpoint)")
        return start_epoch, best_loss

    # Check if current stats file exists
    if not os.path.exists(current_stats_path):
        logger.warning(
            f"Stats consistency check: SKIPPED (stats file not found: {current_stats_path})"
        )
        return start_epoch, best_loss

    # Load current stats and compare
    current_stats = torch.load(current_stats_path, map_location="cpu", weights_only=True)

    try:
        saved_mean_sum = saved_stats["mean"].sum().item()
        saved_std_sum = saved_stats["std"].sum().item()
        current_mean_sum = current_stats["mean"].sum().item()
        current_std_sum = current_stats["std"].sum().item()

        mean_match = abs(saved_mean_sum - current_mean_sum) < 1e-5
        std_match = abs(saved_std_sum - current_std_sum) < 1e-5

        if mean_match and std_match:
            logger.info("Stats consistency check: PASSED")
        else:
            logger.warning(
                "Stats consistency check: WARNING - stats differ. "
                f"Saved: mean_sum={saved_mean_sum:.6f}, std_sum={saved_std_sum:.6f}. "
                f"Current: mean_sum={current_mean_sum:.6f}, std_sum={current_std_sum:.6f}. "
                "This may indicate data pipeline changes."
            )
    except (KeyError, TypeError) as e:
        logger.warning(f"Stats consistency check: ERROR - {e}")

    return start_epoch, best_loss
