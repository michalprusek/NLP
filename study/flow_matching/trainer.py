"""Flow matching trainer with EMA, gradient clipping, and early stopping.

Provides FlowTrainer class that orchestrates training with:
- Flow matching loss (ICFM formulation)
- EMA weight averaging
- Gradient clipping
- Cosine schedule with warmup
- Early stopping based on validation loss
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from study.data.dataset import FlowDataset, create_dataloader
from study.flow_matching.config import TrainingConfig
from study.flow_matching.utils import (
    EarlyStopping,
    EMAModel,
    get_cosine_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


class FlowTrainer:
    """Training orchestrator for flow matching models.

    Handles the complete training loop including:
    - Model optimization with AdamW
    - EMA weight averaging
    - Gradient clipping
    - Cosine learning rate schedule with warmup
    - Validation every epoch
    - Early stopping based on validation loss

    Note: Wandb logging and checkpointing are NOT included here.
    They will be added in Plan 02.

    Attributes:
        model: Velocity network to train.
        config: Training configuration.
        device: Device for training (cuda/cpu).
        global_step: Total training steps taken.
        current_epoch: Current training epoch.
        best_val_loss: Best validation loss seen.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: FlowDataset,
        val_dataset: FlowDataset,
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: Velocity network (will be moved to device).
            config: Training configuration.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            device: Device for training.
        """
        self.config = config
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Move model to device FIRST, then create EMA
        self.model = model.to(device)

        # Initialize training state
        self._global_step = 0
        self._current_epoch = 0
        self._best_val_loss = float("inf")

        # Setup training infrastructure
        self._setup()

    def _setup(self) -> None:
        """Initialize optimizer, scheduler, EMA, early stopping, and data loaders."""
        # Create EMA after model is on device
        self.ema = EMAModel(self.model, decay=self.config.ema_decay)

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=0.01,
        )

        # Calculate total training steps
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=42,
            num_workers=4,
            pin_memory=True,
            drop_last=True,  # Avoid small batches
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            seed=42,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

        logger.info(
            f"Training setup complete: "
            f"{len(self.train_dataset)} train samples, "
            f"{len(self.val_dataset)} val samples, "
            f"{len(self.train_loader)} batches/epoch, "
            f"{total_steps} total steps"
        )

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, x1 in enumerate(self.train_loader):
            x1 = x1.to(self.device)

            # Sample noise (source distribution)
            x0 = torch.randn_like(x1)

            # Sample time uniformly
            t = torch.rand(x1.shape[0], device=self.device)

            # Interpolate: x_t = (1-t)*x0 + t*x1
            t_unsqueeze = t.unsqueeze(-1)
            x_t = (1 - t_unsqueeze) * x0 + t_unsqueeze * x1

            # Target velocity: v = x1 - x0
            v_target = x1 - x0

            # Forward pass
            v_pred = self.model(x_t, t)

            # MSE loss
            loss = F.mse_loss(v_pred, v_target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip,
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # EMA update
            self.ema.update(self.model)

            # Track metrics
            epoch_loss += loss.item()
            n_batches += 1
            self._global_step += 1

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.debug(
                    f"Epoch {self._current_epoch} | "
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Grad: {grad_norm:.4f} | "
                    f"LR: {lr:.2e}"
                )

        avg_loss = epoch_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Compute validation loss.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x1 in self.val_loader:
            x1 = x1.to(self.device)

            # Sample noise
            x0 = torch.randn_like(x1)

            # Sample time
            t = torch.rand(x1.shape[0], device=self.device)

            # Interpolate
            t_unsqueeze = t.unsqueeze(-1)
            x_t = (1 - t_unsqueeze) * x0 + t_unsqueeze * x1

            # Target velocity
            v_target = x1 - x0

            # Forward pass
            v_pred = self.model(x_t, t)

            # Loss
            loss = F.mse_loss(v_pred, v_target)

            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / n_batches

    def train(self) -> Dict:
        """Run full training loop.

        Returns:
            Training summary dict with:
            - epochs_run: Number of epochs completed
            - best_val_loss: Best validation loss achieved
            - final_train_loss: Training loss at last epoch
            - final_val_loss: Validation loss at last epoch
            - early_stopped: Whether training stopped early
        """
        logger.info(
            f"Starting training: "
            f"{self.config.epochs} epochs, "
            f"batch_size={self.config.batch_size}, "
            f"lr={self.config.lr}"
        )

        final_train_loss = float("inf")
        final_val_loss = float("inf")
        early_stopped = False

        for epoch in range(1, self.config.epochs + 1):
            self._current_epoch = epoch

            # Training
            train_loss = self.train_epoch()
            final_train_loss = train_loss

            # Validation (every val_frequency epochs)
            if epoch % self.config.val_frequency == 0:
                val_loss = self.validate()
                final_val_loss = val_loss

                # Check if improved
                improved = self.early_stopping(val_loss)
                if improved:
                    self._best_val_loss = val_loss
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs} | "
                        f"Train: {train_loss:.6f} | "
                        f"Val: {val_loss:.6f} (NEW BEST)"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs} | "
                        f"Train: {train_loss:.6f} | "
                        f"Val: {val_loss:.6f} | "
                        f"Patience: {self.early_stopping.counter}/{self.config.patience}"
                    )

                # Check early stopping
                if self.early_stopping.should_stop:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best val loss: {self._best_val_loss:.6f}"
                    )
                    early_stopped = True
                    break
            else:
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Train: {train_loss:.6f}"
                )

        logger.info(
            f"Training complete. "
            f"Epochs: {self._current_epoch}, "
            f"Best val loss: {self._best_val_loss:.6f}"
        )

        return {
            "epochs_run": self._current_epoch,
            "best_val_loss": self._best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "early_stopped": early_stopped,
        }

    @property
    def global_step(self) -> int:
        """Total training steps taken."""
        return self._global_step

    @property
    def current_epoch(self) -> int:
        """Current training epoch."""
        return self._current_epoch

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen."""
        return self._best_val_loss
