"""
VAE Training with Distributed Data Parallel (DDP) for Alpaca dataset.

This script trains the Instruction VAE (768D -> 32D compression) using DDP
across multiple GPUs.

Usage:
    torchrun --nproc_per_node=2 --master_port=12355 \
        -m lido_pp.training.train_vae_ddp \
        --data lido_pp/data/alpaca_embeddings.pt \
        --epochs 50000 --batch-size 128 --lr 6e-4

Architecture:
    Encoder: 768 -> 512 -> 256 -> 128 -> 64 (mu + log_var)
    Decoder: 32 -> 128 -> 256 -> 512 -> 768 (L2 normalized)
    ~1.2M parameters

Loss: (1-mse_weight)*cosine_loss + mse_weight*mse_loss + beta * KL
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lido_pp.vae import InstructionVAE
from lido_pp.training.alpaca_dataset import EmbeddingDataset
from lido_pp.training.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    save_checkpoint_ddp,
    reduce_tensor,
    print_rank0,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE with DDP")
    parser.add_argument(
        "--data",
        type=str,
        default="lido_pp/data/alpaca_embeddings.pt",
        help="Path to pre-computed embeddings",
    )
    parser.add_argument("--epochs", type=int, default=50000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=32, help="VAE latent dimension")
    parser.add_argument("--beta", type=float, default=0.005, help="KL weight")
    parser.add_argument("--mse-weight", type=float, default=0.2, help="MSE weight in recon loss")
    parser.add_argument("--warmup-epochs", type=int, default=2500, help="KL annealing epochs")
    parser.add_argument("--patience", type=int, default=1000, help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    current_beta: float,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_cosine = 0.0
    n_batches = 0

    for batch in dataloader:
        embeddings = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, log_var, z = model(embeddings)

        # Compute loss using VAE's loss function
        loss, loss_dict = model.module.loss(
            embeddings, x_recon, mu, log_var, z,
            beta=current_beta,
            mse_weight=args.mse_weight,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict["total"]
        total_recon += loss_dict["recon"]
        total_kl += loss_dict["kl"]
        total_cosine += loss_dict["cosine_mean"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "cosine": total_cosine / n_batches,
    }


@torch.no_grad()
def validate(
    model: DDP,
    dataloader: DataLoader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_cosine = 0.0
    n_batches = 0

    for batch in dataloader:
        embeddings = batch.to(device)

        x_recon, mu, log_var, z = model(embeddings)

        loss, loss_dict = model.module.loss(
            embeddings, x_recon, mu, log_var, z,
            beta=args.beta,
            mse_weight=args.mse_weight,
        )

        total_loss += loss_dict["total"]
        total_recon += loss_dict["recon"]
        total_cosine += loss_dict["cosine_mean"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "cosine": total_cosine / n_batches,
    }


def train_worker(rank: int, world_size: int, args):
    """Main training worker for each GPU."""
    # Setup DDP
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed + rank)

    # Load embeddings
    print_rank0(f"Loading embeddings from {args.data}...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        embeddings = data["embeddings"]
    else:
        embeddings = data

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    print_rank0(f"Embeddings shape: {embeddings.shape}")

    # Train/val split
    n_val = int(len(embeddings) * args.val_ratio)
    indices = torch.randperm(len(embeddings))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_embeddings = embeddings[train_indices]
    val_embeddings = embeddings[val_indices]

    print_rank0(f"Train: {len(train_embeddings)}, Val: {len(val_embeddings)}")

    # Create datasets
    train_dataset = EmbeddingDataset(train_embeddings)
    val_dataset = EmbeddingDataset(val_embeddings)

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    # Create model
    model = InstructionVAE(
        input_dim=768,
        latent_dim=args.latent_dim,
        beta=args.beta,
        mse_weight=args.mse_weight,
    ).to(device)

    # Wrap in DDP
    model = DDP(model, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = datetime.now()

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_rank0(f"\nStarting VAE training:")
    print_rank0(f"  Epochs: {args.epochs}")
    print_rank0(f"  Batch size per GPU: {args.batch_size}")
    print_rank0(f"  Total batch size: {args.batch_size * world_size}")
    print_rank0(f"  Learning rate: {args.lr}")
    print_rank0(f"  Beta (KL weight): {args.beta}")
    print_rank0(f"  Warmup epochs: {args.warmup_epochs}")
    print_rank0(f"  World size: {world_size}")

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # KL annealing
        if epoch < args.warmup_epochs:
            current_beta = args.beta * (epoch / args.warmup_epochs)
        else:
            current_beta = args.beta

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args, current_beta
        )

        # Scheduler step
        scheduler.step()

        # Validate periodically
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            val_metrics = validate(model, val_loader, device, args)

            # Reduce metrics across GPUs
            train_loss = reduce_tensor(
                torch.tensor(train_metrics["loss"], device=device), world_size
            ).item()
            val_loss = reduce_tensor(
                torch.tensor(val_metrics["loss"], device=device), world_size
            ).item()
            val_cosine = reduce_tensor(
                torch.tensor(val_metrics["cosine"], device=device), world_size
            ).item()

            if rank == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"Cosine: {val_cosine:.4f} | "
                    f"Beta: {current_beta:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"Time: {elapsed:.1f}m"
                )

            # Early stopping check (after warmup)
            if epoch >= args.warmup_epochs:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    save_checkpoint_ddp(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_cosine": val_cosine,
                            "args": vars(args),
                        },
                        str(checkpoint_dir / "vae_best.pt"),
                        rank,
                    )
                else:
                    patience_counter += args.log_interval

                if patience_counter >= args.patience:
                    print_rank0(f"Early stopping at epoch {epoch+1}")
                    break

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint_ddp(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                str(checkpoint_dir / f"vae_epoch{epoch+1:05d}.pt"),
                rank,
            )

    # Final save
    save_checkpoint_ddp(
        {
            "epoch": args.epochs,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        str(checkpoint_dir / "vae_alpaca_final.pt"),
        rank,
    )

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print_rank0(f"\nTraining complete! Total time: {total_time:.1f} minutes")
    print_rank0(f"Best validation loss: {best_val_loss:.4f}")
    print_rank0(f"Checkpoints saved to: {checkpoint_dir}")

    cleanup_ddp()


def main():
    args = parse_args()

    # Get world size from environment (set by torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # Running with torchrun
        train_worker(local_rank, world_size, args)
    else:
        # Single GPU training
        print("Running single GPU training...")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
