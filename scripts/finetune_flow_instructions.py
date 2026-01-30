#!/usr/bin/env python
"""
Fine-tune flow model on instruction embeddings.

This script fine-tunes an existing flow model checkpoint on task-specific
instruction embeddings to improve generation quality for prompt optimization.

Usage:
    # Fine-tune existing model
    python scripts/finetune_flow_instructions.py \
        --base-checkpoint results/flow_ot_*/checkpoint_final.pt \
        --data datasets/instruction_embeddings_50k.pt \
        --epochs 20 --lr 1e-5

    # Train from scratch on instructions only
    python scripts/finetune_flow_instructions.py \
        --data datasets/instruction_embeddings_50k.pt \
        --epochs 50 --lr 1e-4 --from-scratch
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_instruction_data(path: str, normalize: bool = True):
    """Load instruction embeddings and optionally normalize."""
    logger.info(f"Loading instruction embeddings from {path}...")
    data = torch.load(path, weights_only=False)

    embeddings = data["embeddings"]
    instructions = data.get("instructions", [])
    sources = data.get("sources", {})

    logger.info(f"Loaded {len(embeddings)} embeddings")
    logger.info(f"Sources: {sources}")

    if normalize:
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True) + 1e-6
        embeddings = (embeddings - mean) / std
        logger.info(f"Normalized: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
        return embeddings, {"mean": mean.squeeze(0), "std": std.squeeze(0)}

    return embeddings, None


def create_dataloader(embeddings: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Create DataLoader for training."""
    dataset = TensorDataset(embeddings)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def compute_flow_matching_loss(model, x1: torch.Tensor, sigma_min: float = 1e-4):
    """Compute conditional flow matching loss (OT path)."""
    batch_size = x1.shape[0]
    device = x1.device

    # Sample time uniformly
    t = torch.rand(batch_size, device=device)

    # Sample noise
    x0 = torch.randn_like(x1)

    # Interpolate (OT path)
    t_expand = t.view(-1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1

    # Target velocity is constant for OT
    target_velocity = x1 - x0

    # Predict velocity
    predicted_velocity = model.velocity_net(xt, t)

    # MSE loss
    loss = ((predicted_velocity - target_velocity) ** 2).mean()

    return loss


def train_epoch(model, dataloader, optimizer, device, epoch: int):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        x1 = batch[0].to(device)

        optimizer.zero_grad()
        loss = compute_flow_matching_loss(model, x1)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n_batches


def validate(model, dataloader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x1 = batch[0].to(device)
            loss = compute_flow_matching_loss(model, x1)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Fine-tune flow model on instruction embeddings")

    # Data
    parser.add_argument("--data", type=str, required=True, help="Path to instruction embeddings")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")

    # Model
    parser.add_argument("--base-checkpoint", type=str, default=None,
                       help="Base checkpoint to fine-tune (if None, train from scratch)")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (use 1e-5 for fine-tune, 1e-4 for scratch)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=2)

    # Model architecture (only for from-scratch)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "scratch" if args.from_scratch else "finetune"
        args.output_dir = f"results/flow_inst_{mode}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    embeddings, norm_stats = load_instruction_data(args.data, normalize=True)

    # Split train/val
    n_val = int(len(embeddings) * args.val_split)
    perm = torch.randperm(len(embeddings))
    train_emb = embeddings[perm[n_val:]]
    val_emb = embeddings[perm[:n_val]]
    logger.info(f"Train: {len(train_emb)}, Val: {len(val_emb)}")

    # Create dataloaders
    train_loader = create_dataloader(train_emb, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_emb, args.batch_size, shuffle=False)

    # Load or create model
    if args.from_scratch or args.base_checkpoint is None:
        logger.info("Creating new flow model from scratch...")
        from src.ecoflow.flow_model import FlowMatchingModel
        model = FlowMatchingModel(
            input_dim=1024,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    else:
        logger.info(f"Loading base checkpoint: {args.base_checkpoint}")
        from src.ecoflow.validate import load_model_from_checkpoint
        model = load_model_from_checkpoint(args.base_checkpoint, device="cpu", use_ema=True)

    model = model.to(device)
    model.norm_stats = norm_stats

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float("inf")
    logger.info("Starting training...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Mode: {'from scratch' if args.from_scratch else 'fine-tune'}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.velocity_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "norm_stats": norm_stats,
                "config": {
                    "input_dim": 1024,
                    "hidden_dim": args.hidden_dim if args.from_scratch else model.velocity_net.hidden_dim,
                    "num_layers": args.num_layers if args.from_scratch else len(model.velocity_net.blocks),
                    "num_heads": args.num_heads,  # Use CLI arg (defaults to 8)
                },
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint_best.pt"))
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")

    # Save final model
    checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.velocity_net.state_dict(),
        "norm_stats": norm_stats,
        "config": {
            "input_dim": 1024,
            "hidden_dim": args.hidden_dim if args.from_scratch else model.velocity_net.hidden_dim,
            "num_layers": args.num_layers if args.from_scratch else len(model.velocity_net.blocks),
            "num_heads": args.num_heads,  # Use CLI arg (defaults to 8)
        },
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint_final.pt"))
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
