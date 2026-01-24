"""DDP training script for Soft-Prompt VAE.

Supports multi-GPU training with Accelerate, gradient checkpointing,
mixed precision, and cyclical KL annealing.

Usage:
    # Single GPU
    uv run python -m soft_prompt_vae.train --phase 1

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 -m soft_prompt_vae.train --phase 1
"""

import argparse
import logging
import os
import math
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING

# Suppress NVML warnings (driver/library mismatch doesn't affect training)
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message=".*pynvml.*")

# Pre-configure CUDA/NCCL before importing torch to avoid NVML issues
# NCCL_P2P_DISABLE: Prevents NCCL from using NVML for P2P topology detection
# NCCL_SHM_DISABLE: Forces socket-based communication (slower but more compatible)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from soft_prompt_vae.config import VAEConfig
from soft_prompt_vae.model import LlamaSoftPromptVAE
from soft_prompt_vae.loss import SoftPromptVAELoss, compute_vae_loss
from soft_prompt_vae.metrics import compute_all_metrics, ActiveUnitsCounter
from soft_prompt_vae.data.tokenization import create_tokenizer
from soft_prompt_vae.data.loader import create_dataloader, get_num_batches_estimate
from soft_prompt_vae.collapse_monitor import CollapseMonitor, AdaptiveInterventionScheduler, CollapseMonitorConfig
from soft_prompt_vae.augmentation import TextAugmenter, AugmentationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(
            f"{prefix}GPU Memory: {allocated:.1f}GB allocated, "
            f"{reserved:.1f}GB reserved, {max_allocated:.1f}GB peak"
        )


def train_epoch(
    model: LlamaSoftPromptVAE,
    dataloader,
    loss_fn: SoftPromptVAELoss,
    optimizer,
    scheduler,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    config: VAEConfig,
    augmenter: Optional[TextAugmenter] = None,
    collapse_monitor: Optional[CollapseMonitor] = None,
    intervention_scheduler: Optional[AdaptiveInterventionScheduler] = None,
) -> int:
    """Train for one epoch.

    Args:
        model: VAE model
        dataloader: Training dataloader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler
        accelerator: Accelerator instance
        epoch: Current epoch number
        global_step: Global step counter
        config: VAE configuration
        augmenter: Optional TextAugmenter for contrastive learning (CDP-VAE)
        collapse_monitor: Optional CollapseMonitor for adaptive intervention
        intervention_scheduler: Optional AdaptiveInterventionScheduler

    Returns:
        Updated global step
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_bow = 0.0
    total_contrastive = 0.0
    num_batches = 0

    # Active units tracker
    au_counter = ActiveUnitsCounter()

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    for batch in progress_bar:
        with accelerator.accumulate(model):
            # Forward pass (with augmentation for contrastive learning)
            output = model(
                instruction_ids=batch.instruction_ids,
                instruction_attention_mask=batch.instruction_attention_mask,
                response_ids=batch.response_ids,
                response_attention_mask=batch.response_attention_mask,
                augmenter=augmenter,
            )

            # Monitor for collapse and apply adaptive interventions
            if collapse_monitor is not None and intervention_scheduler is not None:
                collapse_metrics = collapse_monitor.update(
                    output.mu, output.logvar, output.mu_augmented
                )
                if collapse_metrics.is_collapsing:
                    intervention = intervention_scheduler.intervene(collapse_metrics)
                    # Update loss function weights dynamically
                    loss_fn.bow_loss_weight = intervention["bow_weight"]
                    loss_fn.contrastive_weight = intervention["contrastive_weight"]
                    # Update model word dropout rate
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.config.word_dropout_rate = intervention["word_dropout"]
                else:
                    # Gradually relax back to initial values
                    intervention_scheduler.intervene(collapse_metrics)

            # Compute loss with annealing (includes BoW and contrastive if enabled)
            loss_output = loss_fn(
                logits=output.logits,
                labels=batch.labels,
                mu=output.mu,
                logvar=output.logvar,
                step=global_step,
                bow_logits=output.bow_logits,
                mu_augmented=output.mu_augmented,
            )

            # Backward pass
            accelerator.backward(loss_output.total)

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update metrics (use .item() to release tensor references)
        total_loss += loss_output.total.item()
        total_recon += loss_output.reconstruction.item()
        total_kl += loss_output.kl.item()
        if loss_output.bow is not None:
            total_bow += loss_output.bow.item()
        if loss_output.contrastive is not None:
            total_contrastive += loss_output.contrastive.item()
        num_batches += 1

        # Periodic GPU memory cleanup (every 50 batches)
        if num_batches % 50 == 0:
            torch.cuda.empty_cache()

        # Update AU counter
        au_counter.update(output.mu.detach())

        # Update progress bar
        if num_batches % 10 == 0:
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_kl = total_kl / num_batches
            active_units, au_ratio, _ = au_counter.compute()

            postfix = {
                "loss": f"{avg_loss:.4f}",
                "recon": f"{avg_recon:.4f}",
                "kl": f"{avg_kl:.4f}",
                "beta": f"{loss_output.beta:.3f}",
                "AU": f"{active_units}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            # Add BoW to progress bar if enabled
            if loss_output.bow is not None:
                avg_bow = total_bow / num_batches
                postfix["bow"] = f"{avg_bow:.4f}"
            # Add contrastive to progress bar if enabled
            if loss_output.contrastive is not None:
                avg_contrastive = total_contrastive / num_batches
                postfix["nce"] = f"{avg_contrastive:.4f}"

            progress_bar.set_postfix(postfix)

        # Logging
        if global_step % config.training.logging_steps == 0:
            log_dict = {
                "train/loss": loss_output.total.item(),
                "train/recon_loss": loss_output.reconstruction.item(),
                "train/kl_loss": loss_output.kl.item(),
                "train/kl_raw": loss_output.kl_raw.item(),
                "train/beta": loss_output.beta,
                "train/active_dims": loss_output.active_dims,
                "train/lr": scheduler.get_last_lr()[0],
            }
            # Add BoW loss if enabled
            if loss_output.bow is not None:
                log_dict["train/bow_loss"] = loss_output.bow.item()
            # Add contrastive loss if enabled (CDP-VAE)
            if loss_output.contrastive is not None:
                log_dict["train/contrastive_loss"] = loss_output.contrastive.item()
            # Add collapse metrics if monitoring enabled
            if collapse_monitor is not None and collapse_monitor._last_metrics is not None:
                cm = collapse_monitor._last_metrics
                log_dict["collapse/active_units"] = cm.active_units
                log_dict["collapse/au_ratio"] = cm.active_unit_ratio
                log_dict["collapse/mutual_info"] = cm.mutual_info_estimate
                log_dict["collapse/intervention_level"] = cm.intervention_level
            # Add intervention scheduler status
            if intervention_scheduler is not None:
                status = intervention_scheduler.get_status()
                log_dict["intervention/bow_weight"] = status["bow_weight"]
                log_dict["intervention/contrastive_weight"] = status["contrastive_weight"]
                log_dict["intervention/word_dropout"] = status["word_dropout"]

            accelerator.log(log_dict, step=global_step)

        global_step += 1

        # Save checkpoint
        if global_step % config.training.save_steps == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, config, accelerator
            )

    # Epoch summary
    if accelerator.is_local_main_process:
        avg_loss = total_loss / num_batches
        active_units, au_ratio, _ = au_counter.compute()
        logger.info(
            f"Epoch {epoch} complete: loss={avg_loss:.4f}, "
            f"AU={active_units} ({au_ratio:.1%})"
        )
        log_gpu_memory(f"End of epoch {epoch}: ")

    return global_step


def save_checkpoint(
    model: LlamaSoftPromptVAE,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    config: VAEConfig,
    accelerator: Accelerator,
) -> None:
    """Save training checkpoint."""
    if not accelerator.is_local_main_process:
        return

    checkpoint_dir = config.training.output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model
    unwrapped_model = accelerator.unwrap_model(model)

    # Save model state
    torch.save(
        {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        checkpoint_dir / "checkpoint.pt",
    )

    # Save config
    config.save(checkpoint_dir / "config.json")

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def load_checkpoint(
    checkpoint_path: Path,
    model: LlamaSoftPromptVAE,
    optimizer=None,
    scheduler=None,
) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["global_step"]


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Soft-Prompt VAE")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Training phase (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Use preprocessed data if available",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision (use fp32)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (required for deep prefix)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--no-deep-prefix",
        action="store_true",
        help="Disable deep prefix (allows gradient checkpointing)",
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Enable memory-efficient mode: batch_size=4, disable augmentation, disable contrastive",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = VAEConfig.load(Path(args.config))
    else:
        config = VAEConfig()

    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_batch_size = args.batch_size
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.no_deep_prefix:
        config.model.use_deep_prefix = False

    # Memory-efficient mode: reduce memory usage at cost of training quality
    if args.memory_efficient:
        logger.info("Memory-efficient mode enabled:")
        config.training.per_device_batch_size = min(config.training.per_device_batch_size, 4)
        config.model.contrastive_weight = 0.0  # Disable contrastive (saves encoder forward pass)
        config.model.augmentation_probability = 0.0  # Disable augmentation
        config.model.use_deep_prefix = False  # Ensure gradient checkpointing works
        logger.info(f"  - Batch size: {config.training.per_device_batch_size}")
        logger.info(f"  - Contrastive: disabled")
        logger.info(f"  - Augmentation: disabled")
        logger.info(f"  - Deep prefix: disabled (gradient checkpointing enabled)")

    # Handle mixed precision
    mixed_precision = config.training.mixed_precision
    if args.no_mixed_precision:
        mixed_precision = "no"

    # Initialize accelerator
    # dispatch_batches=False prevents accelerate from concatenating custom VAEBatch objects
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(config.training.logging_dir),
        dataloader_config=dataloader_config,
    )

    # Set seed
    set_seed(args.seed)

    if accelerator.is_local_main_process:
        logger.info("=" * 60)
        logger.info("Soft-Prompt VAE Training")
        logger.info(f"Phase: {args.phase}")
        logger.info(f"Epochs: {config.training.num_epochs}")
        logger.info(f"Batch size: {config.training.per_device_batch_size}")
        logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {config.training.effective_batch_size}")
        logger.info(f"Mixed precision: {config.training.mixed_precision}")
        logger.info(f"World size: {accelerator.num_processes}")
        logger.info("=" * 60)

    # Create tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = create_tokenizer(config.model, config.data)

    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_dataloader(
        config=config.data,
        tokenizer=tokenizer,
        batch_size=config.training.per_device_batch_size,
        phase=args.phase,
        use_preprocessed=args.use_preprocessed,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
    )

    # Estimate total steps
    num_batches = get_num_batches_estimate(
        config.data,
        config.training.per_device_batch_size,
        args.phase,
        accelerator.num_processes,
    )
    total_steps = num_batches * config.training.num_epochs
    total_steps = total_steps // config.training.gradient_accumulation_steps

    if accelerator.is_local_main_process:
        logger.info(f"Estimated batches per epoch: {num_batches}")
        logger.info(f"Estimated total steps: {total_steps}")

    # Create model
    logger.info("Creating model...")
    use_ddp = accelerator.num_processes > 1
    model = LlamaSoftPromptVAE(config.model, use_ddp=use_ddp)

    # Enable gradient checkpointing (with deep prefix compatibility check)
    use_grad_ckpt = config.training.gradient_checkpointing and not args.no_gradient_checkpointing
    if config.model.use_deep_prefix and use_grad_ckpt:
        logger.warning(
            "Deep prefix is INCOMPATIBLE with gradient checkpointing (past_key_values disabled). "
            "Automatically disabling gradient checkpointing to preserve deep prefix functionality."
        )
        use_grad_ckpt = False

    if use_grad_ckpt:
        model.llama.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        logger.info("Gradient checkpointing disabled (deep prefix mode)")

    # Create loss function with BoW and contrastive weights from model config
    loss_fn = SoftPromptVAELoss(
        config.training,
        total_steps,
        bow_loss_weight=config.model.bow_loss_weight,
        contrastive_weight=config.model.contrastive_weight,
        contrastive_temperature=config.model.contrastive_temperature,
    )

    # Initialize CDP-VAE components (augmenter, collapse monitor, intervention scheduler)
    augmenter = None
    collapse_monitor = None
    intervention_scheduler = None

    if config.model.contrastive_weight > 0:
        # Initialize text augmenter for contrastive learning
        aug_config = AugmentationConfig(
            span_mask_prob=config.model.augmentation_span_mask_prob,
            word_dropout_rate=config.model.augmentation_word_dropout,
        )
        augmenter = TextAugmenter(aug_config)
        augmenter.configure_from_tokenizer(tokenizer)
        logger.info(
            f"Text augmenter initialized: span_mask={aug_config.span_mask_prob}, "
            f"word_dropout={aug_config.word_dropout_rate}"
        )

    if config.training.enable_collapse_monitoring:
        # Initialize collapse monitoring
        collapse_config = CollapseMonitorConfig(
            au_ratio_warning=config.training.collapse_au_warning,
            au_ratio_critical=config.training.collapse_au_critical,
        )
        collapse_monitor = CollapseMonitor(
            latent_dim=config.model.latent_dim,
            config=collapse_config,
        )
        intervention_scheduler = AdaptiveInterventionScheduler(
            initial_bow_weight=config.model.bow_loss_weight,
            initial_contrastive_weight=config.model.contrastive_weight,
            initial_word_dropout=config.model.word_dropout_rate,
        )
        logger.info(
            f"Collapse monitoring enabled: AU warning={collapse_config.au_ratio_warning}, "
            f"AU critical={collapse_config.au_ratio_critical}"
        )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.training.learning_rate * 0.1,
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, global_step = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )

    # Prepare with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Initialize tracker
    accelerator.init_trackers(
        project_name="soft_prompt_vae",
        config={
            "phase": args.phase,
            "epochs": config.training.num_epochs,
            "batch_size": config.training.per_device_batch_size,
            "learning_rate": config.training.learning_rate,
        },
    )

    # Training loop
    logger.info("Starting training...")
    log_gpu_memory("Before training: ")
    try:
        for epoch in range(start_epoch, config.training.num_epochs):
            # Reset collapse monitor statistics at epoch start (optional)
            if collapse_monitor is not None:
                collapse_monitor.reset()

            global_step = train_epoch(
                model=model,
                dataloader=dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                epoch=epoch,
                global_step=global_step,
                config=config,
                augmenter=augmenter,
                collapse_monitor=collapse_monitor,
                intervention_scheduler=intervention_scheduler,
            )

            # Save epoch checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, config, accelerator
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted")

    finally:
        # Save final checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, config, accelerator
        )
        accelerator.end_training()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
