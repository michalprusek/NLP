"""Flow matching trainer with EMA, gradient clipping, early stopping, and Wandb.

Provides FlowTrainer class that orchestrates training with:
- Flow matching loss (ICFM formulation)
- EMA weight averaging
- Gradient clipping
- Cosine schedule with warmup
- Early stopping based on validation loss
- Wandb experiment tracking
- Checkpoint saving/loading with resume support
"""

import logging
import traceback
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW

from study.data.augmentation import AugmentationConfig, augment_batch
from study.data.dataset import FlowDataset, create_dataloader
from study.flow_matching.config import TrainingConfig
from study.flow_matching.coupling import create_coupling
from study.flow_matching.utils import (
    EarlyStopping,
    EMAModel,
    get_checkpoint_path,
    get_cosine_schedule_with_warmup,
    load_checkpoint_with_stats_check,
    save_checkpoint,
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
    - Wandb experiment tracking with proper grouping
    - Checkpoint save/resume with stats consistency verification

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
        wandb_project: str = "flow-matching-study",
        resume_path: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            model: Velocity network (will be moved to device).
            config: Training configuration.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            device: Device for training.
            wandb_project: Wandb project name.
            resume_path: Path to checkpoint to resume from (optional).
        """
        self.config = config
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.wandb_project = wandb_project
        self.resume_path = resume_path

        # Move model to device FIRST, then create EMA
        self.model = model.to(device)

        # Initialize training state
        self._global_step = 0
        self._current_epoch = 0
        self._best_val_loss = float("inf")
        self._start_epoch = 1  # For resume support

        # Checkpoint path for this run
        self._checkpoint_path = get_checkpoint_path(config.run_name)

        # Setup augmentation config from aug string and explicit parameters
        self.aug_config = self._create_aug_config()

        # Setup training infrastructure (includes Wandb init and checkpoint loading)
        self._setup()

    def _create_aug_config(self) -> Optional[AugmentationConfig]:
        """Create augmentation config from config parameters.

        Parses config.aug string to set defaults:
        - "none" -> all 0.0
        - "mixup" -> mixup_alpha=0.2
        - "noise" -> noise_std=0.1
        - "mixup+noise" -> mixup_alpha=0.2, noise_std=0.1

        Explicit config values (mixup_alpha, noise_std) override defaults.

        Returns:
            AugmentationConfig if any augmentation enabled, None otherwise.
        """
        # Parse aug string for defaults
        aug_str = self.config.aug.lower() if self.config.aug else "none"

        # Determine defaults from aug string
        default_mixup = 0.2 if "mixup" in aug_str else 0.0
        default_noise = 0.1 if "noise" in aug_str else 0.0

        # Use explicit config values if provided (non-zero), else use defaults
        mixup_alpha = self.config.mixup_alpha if self.config.mixup_alpha > 0 else default_mixup
        noise_std = self.config.noise_std if self.config.noise_std > 0 else default_noise
        dropout_rate = self.config.dropout_rate  # No default from aug string

        # Only create config if some augmentation is enabled
        if mixup_alpha > 0 or noise_std > 0 or dropout_rate > 0:
            aug_config = AugmentationConfig(
                mixup_alpha=mixup_alpha,
                noise_std=noise_std,
                dropout_rate=dropout_rate,
            )
            logger.info(
                f"Augmentation enabled: mixup_alpha={mixup_alpha}, "
                f"noise_std={noise_std}, dropout_rate={dropout_rate}"
            )
            return aug_config

        logger.info("Augmentation disabled")
        return None

    def _setup(self) -> None:
        """Initialize optimizer, scheduler, EMA, coupling, early stopping, data loaders, and Wandb."""
        # Create EMA after model is on device
        self.ema = EMAModel(self.model, decay=self.config.ema_decay)

        # Create coupling method based on config
        coupling_kwargs = {}
        if self.config.flow == "otcfm":
            coupling_kwargs = {
                "sigma": self.config.otcfm_sigma,
                "reg": self.config.otcfm_reg,
                "normalize_cost": self.config.otcfm_normalize_cost,
            }
        self.coupling = create_coupling(self.config.flow, **coupling_kwargs)
        logger.info(f"Using {self.config.flow} coupling method")

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

        # Initialize Wandb
        wandb.init(
            project=self.wandb_project,
            group=self.config.group,
            name=self.config.run_name,
            config=self.config.to_dict(),
            resume="allow" if self.resume_path else "never",
        )

        # Define metrics with summary aggregations
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("train/loss", summary="min")

        # Load checkpoint if resuming
        # CRITICAL: Load checkpoint AFTER model is on device and optimizer/scheduler created
        # This ensures optimizer state tensors match model parameters
        if self.resume_path:
            self._start_epoch, self._best_val_loss = load_checkpoint_with_stats_check(
                path=self.resume_path,
                model=self.model,
                ema=self.ema,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
                current_stats_path=self.config.stats_path,
            )
            # Update early stopping with loaded best loss
            self.early_stopping.best_loss = self._best_val_loss
            logger.info(
                f"Resumed from epoch {self._start_epoch - 1}, "
                f"best_val_loss={self._best_val_loss:.4f}"
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

            # Apply augmentation (only during training, before coupling.sample())
            if self.aug_config is not None:
                x1 = augment_batch(x1, self.aug_config, training=True)

            # Sample noise (source distribution)
            x0 = torch.randn_like(x1)

            # Get interpolated samples and target velocity from coupling
            t, x_t, v_target = self.coupling.sample(x0, x1)

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

            # Log to Wandb every 10 steps
            if batch_idx % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "train/lr": lr,
                        "epoch": self._current_epoch,
                    },
                    step=self._global_step,
                )
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

            # Get interpolated samples and target velocity from coupling
            t, x_t, v_target = self.coupling.sample(x0, x1)

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

        Handles Wandb logging, checkpointing, and error tagging.

        Returns:
            Training summary dict with:
            - epochs_run: Number of epochs completed
            - best_val_loss: Best validation loss achieved
            - final_train_loss: Training loss at last epoch
            - final_val_loss: Validation loss at last epoch
            - early_stopped: Whether training stopped early
            - checkpoint_path: Path to best checkpoint
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

        try:
            for epoch in range(self._start_epoch, self.config.epochs + 1):
                self._current_epoch = epoch

                # Training
                train_loss = self.train_epoch()
                final_train_loss = train_loss

                # Validation (every val_frequency epochs)
                if epoch % self.config.val_frequency == 0:
                    val_loss = self.validate()
                    final_val_loss = val_loss

                    # Log validation metrics to Wandb
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/best_loss": self._best_val_loss,
                        },
                        step=self._global_step,
                    )

                    # Check if improved
                    improved = self.early_stopping(val_loss)
                    if improved:
                        self._best_val_loss = val_loss
                        logger.info(
                            f"Epoch {epoch}/{self.config.epochs} | "
                            f"Train: {train_loss:.6f} | "
                            f"Val: {val_loss:.6f} (NEW BEST)"
                        )

                        # Save checkpoint on improvement
                        # Load normalization stats if available
                        stats = None
                        if self.config.stats_path:
                            try:
                                stats = torch.load(
                                    self.config.stats_path,
                                    map_location="cpu",
                                    weights_only=True,
                                )
                            except Exception as e:
                                logger.warning(f"Could not load stats for checkpoint: {e}")

                        save_checkpoint(
                            path=self._checkpoint_path,
                            epoch=epoch,
                            model=self.model,
                            ema=self.ema,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            best_loss=self._best_val_loss,
                            config=self.config.to_dict(),
                            stats=stats,
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

            # Training completed successfully
            logger.info(
                f"Training complete. "
                f"Epochs: {self._current_epoch}, "
                f"Best val loss: {self._best_val_loss:.6f}"
            )

            # Update Wandb summary
            wandb.summary["final_epoch"] = self._current_epoch
            wandb.summary["best_val_loss"] = self._best_val_loss
            wandb.summary["checkpoint_path"] = self._checkpoint_path

            # Finish Wandb run
            wandb.finish()

        except Exception as e:
            # Tag run as failed in Wandb
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())

            wandb.run.tags = wandb.run.tags + ("failed",) if wandb.run.tags else ("failed",)
            wandb.run.notes = f"Training failed: {str(e)}"
            wandb.summary["error"] = str(e)
            wandb.finish(exit_code=1)

            raise

        return {
            "epochs_run": self._current_epoch,
            "best_val_loss": self._best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "early_stopped": early_stopped,
            "checkpoint_path": self._checkpoint_path,
        }

    def finish(self) -> None:
        """Finish Wandb run for clean shutdown."""
        wandb.finish()

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
