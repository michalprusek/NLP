"""Train flow matching model with OT-CFM objective on SONAR embeddings."""

import argparse
import copy
import logging
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch_optimizer as optim_extra

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from rielbo.velocity_network import VelocityNetwork
from rielbo.data import get_sonar_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module) -> None:
        """Apply shadow parameters to model."""
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
    """Create schedule with linear warmup and cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    best_loss: float,
    args: argparse.Namespace,
    norm_stats: Dict,
) -> None:
    """Save training checkpoint with verification."""
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "ema_shadow": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "args": vars(args),
        "norm_stats": norm_stats,
    }
    torch.save(checkpoint, path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"Checkpoint file is missing or empty: {path}")
    logger.info(f"Saved checkpoint: {path}")


def train_flow(args: argparse.Namespace) -> None:
    """Main training function for flow matching model."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Output directory: {args.output_dir}, Device: {device}")

    train_loader, norm_stats = get_sonar_dataloader(
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        normalize=True,
    )
    logger.info(
        f"Dataset: {len(train_loader.dataset)} samples, "
        f"batch_size={args.batch_size}, batches={len(train_loader)}"
    )

    model = VelocityNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.optimizer == "lamb":
        optimizer = optim_extra.Lamb(
            model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999)
        )
        logger.info("Using LAMB optimizer")
    else:
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999)
        )
        logger.info("Using AdamW optimizer")

    if args.use_ot:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        logger.info("Using OT-CFM (O(n^3))")
    else:
        FM = ConditionalFlowMatcher(sigma=0.0)
        logger.info("Using standard CFM")

    ema = EMAModel(model, decay=args.ema_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"Total steps: {total_steps}, warmup: {args.warmup_steps}")

    best_loss = float("inf")
    global_step = 0

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, x1 in enumerate(train_loader):
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t, xt, ut = t.to(device), xt.to(device), ut.to(device)

            vt = model(xt, t)
            loss = F.mse_loss(vt, ut)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f} | Grad: {grad_norm:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / num_batches
        logger.info(
            f"Epoch {epoch}/{args.epochs} complete | "
            f"Avg loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % args.val_interval == 0:
            logger.info(f"Validation at epoch {epoch}: best loss = {best_loss:.6f}")

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:04d}.pt"
            )
            try:
                save_checkpoint(
                    checkpoint_path, epoch, global_step, model, ema,
                    optimizer, scheduler, best_loss, args, norm_stats
                )
            except Exception as e:
                logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")

    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    save_checkpoint(
        final_path, args.epochs, global_step, model, ema,
        optimizer, scheduler, best_loss, args, norm_stats
    )
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train flow matching model on SONAR embeddings"
    )

    parser.add_argument(
        "--data-path", type=str, default="datasets/sonar_embeddings.pt",
        help="Path to SONAR embeddings file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/flow_model",
        help="Output directory for checkpoints"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "lamb"],
        help="Optimizer: adamw (default) or lamb (for large batch)"
    )
    parser.add_argument(
        "--use-ot", action="store_true", default=False,
        help="Use OT-CFM (O(n^3), slow for large batches)"
    )
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument(
        "--input-dim", type=int, default=1024,
        help="Input dimension (1024 for SONAR, 768 for GTR)"
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--val-interval", type=int, default=10, help="Epochs between validation")
    parser.add_argument("--save-interval", type=int, default=20, help="Epochs between checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_flow(args)


if __name__ == "__main__":
    main()
