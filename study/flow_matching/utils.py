"""Training utilities for flow matching experiments.

Provides EarlyStopping, EMAModel, and cosine schedule with warmup.
"""

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


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
