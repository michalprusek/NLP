"""
Training script for flow matching model with OT-CFM objective.

Trains a VelocityNetwork on SONAR embeddings using Optimal Transport
Conditional Flow Matching from the torchcfm library.
"""

import argparse
import copy
import logging
import math
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

from src.ecoflow.velocity_network import VelocityNetwork
from src.ecoflow.data import get_sonar_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains shadow copies of parameters for stable evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA with model parameters.

        Args:
            model: Model to track
            decay: EMA decay rate (default: 0.9999)
        """
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with current model parameters.

        Args:
            model: Model with updated parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module) -> None:
        """
        Apply shadow parameters to model.

        Args:
            model: Model to update with shadow parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get shadow state dict for checkpointing."""
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load shadow state dict from checkpoint."""
        self.shadow = copy.deepcopy(state_dict)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a schedule with linear warmup and cosine decay.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial (default: 0.1)

    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_flow(args: argparse.Namespace) -> None:
    """
    Main training function for flow matching model.

    Args:
        args: Command line arguments
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading data from: {args.data_path}")
    train_loader = get_sonar_dataloader(
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    logger.info(f"Dataset size: {len(train_loader.dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Batches per epoch: {len(train_loader)}")

    # Create model
    model = VelocityNetwork(
        input_dim=1024,  # SONAR embedding dimension
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Create flow matcher
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    logger.info("Using ExactOptimalTransportConditionalFlowMatcher (sigma=0.0)")

    # Create EMA
    ema = EMAModel(model, decay=args.ema_decay)
    logger.info(f"EMA decay: {args.ema_decay}")

    # Create learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")

    # Training loop
    best_loss = float("inf")
    global_step = 0

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, x1 in enumerate(train_loader):
            x1 = x1.to(device)

            # Sample noise
            x0 = torch.randn_like(x1)

            # Get OT-CFM interpolation
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            # Forward pass
            vt = model(xt, t)

            # Compute loss
            loss = F.mse_loss(vt, ut)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )

            # Update weights
            optimizer.step()
            scheduler.step()

            # Update EMA
            ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log batch progress (every 100 batches)
            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Grad norm: {grad_norm:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch}/{args.epochs} complete | "
            f"Avg loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Validation (placeholder - just log current stats)
        if epoch % args.val_interval == 0:
            logger.info(f"Validation at epoch {epoch}: best loss = {best_loss:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:04d}.pt"
            )
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    checkpoint = {
        "epoch": args.epochs,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "ema_shadow": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, final_path)
    logger.info(f"Saved final checkpoint: {final_path}")
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train flow matching model on SONAR embeddings"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/sonar_embeddings.pt",
        help="Path to SONAR embeddings file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flow_model",
        help="Output directory for checkpoints",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for velocity network",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )

    # EMA and checkpointing
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay rate",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=10,
        help="Epochs between validation",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=20,
        help="Epochs between checkpoints",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    train_flow(args)


if __name__ == "__main__":
    main()
