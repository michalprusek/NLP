"""
FlowDiT Training with Distributed Data Parallel (DDP).

This script trains the FlowDiT velocity field predictor for flow matching
in the 32D VAE latent space using DDP across multiple GPUs.

Usage:
    torchrun --nproc_per_node=2 --master_port=12355 \
        -m lido_pp.training.train_flowdit_ddp \
        --embeddings lido_pp/data/alpaca_embeddings.pt \
        --vae lido_pp/checkpoints/vae_alpaca_final.pt \
        --epochs 10000 --batch-size 64 --oat-weight 0.1

Architecture:
    Input: 32D VAE latent
    Transformer: 6 blocks, hidden_dim=512, 8 heads
    Cross-attention to 768D context embeddings
    Output: 32D velocity vector
    ~35M parameters

Loss: L_CFM + oat_weight * L_OAT
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lido_pp.flow.flow_dit import FlowDiT
from lido_pp.flow.losses import conditional_flow_matching_loss, oat_regularization
from lido_pp.vae import InstructionVAE
from lido_pp.training.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    save_checkpoint_ddp,
    reduce_tensor,
    print_rank0,
    barrier,
)


class FlowMatchingDataset(Dataset):
    """Dataset for Flow Matching with embeddings and latents."""

    def __init__(
        self,
        embeddings: torch.Tensor,  # (N, 768) - context
        latents: torch.Tensor,  # (N, 32) - targets
    ):
        assert len(embeddings) == len(latents)
        self.embeddings = embeddings
        self.latents = latents

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],  # Context (768D)
            "latent": self.latents[idx],  # Target x_1 (32D)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowDiT with DDP")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="lido_pp/data/alpaca_embeddings.pt",
        help="Path to pre-computed embeddings",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="lido_pp/checkpoints/vae_alpaca_final.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=500, help="Warmup epochs")
    parser.add_argument("--oat-weight", type=float, default=0.1, help="OAT regularization weight")
    parser.add_argument("--oat-steps", type=int, default=10, help="OAT integration steps")
    parser.add_argument("--sigma-min", type=float, default=0.001)
    # Model architecture
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--context-dim", type=int, default=768)
    parser.add_argument("--num-context-tokens", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=500, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume from FlowDiT checkpoint")
    return parser.parse_args()


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_cfm = 0.0
    total_oat = 0.0
    total_v_cosine = 0.0
    n_batches = 0

    for batch in dataloader:
        embeddings = batch["embedding"].to(device)  # (B, 768)
        x_1 = batch["latent"].to(device)  # (B, 32)

        # Generate noise (x_0)
        x_0 = torch.randn_like(x_1)

        # Reshape context for cross-attention: (B, num_ctx, context_dim)
        context = embeddings.unsqueeze(1).expand(-1, args.num_context_tokens, -1)

        optimizer.zero_grad()

        # CFM loss
        cfm_loss, cfm_metrics = conditional_flow_matching_loss(
            model.module if hasattr(model, 'module') else model,
            x_0, x_1, context,
            sigma_min=args.sigma_min,
        )

        total_loss_batch = cfm_loss

        # OAT regularization (if enabled)
        oat_loss_val = 0.0
        if args.oat_weight > 0:
            oat_loss, oat_metrics = oat_regularization(
                model.module if hasattr(model, 'module') else model,
                x_0, x_1, context,
                num_steps=args.oat_steps,
            )
            total_loss_batch = total_loss_batch + args.oat_weight * oat_loss
            oat_loss_val = oat_loss.item()

        # Backward
        total_loss_batch.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Accumulate metrics
        total_loss += total_loss_batch.item()
        total_cfm += cfm_metrics["cfm_loss"]
        total_oat += oat_loss_val
        total_v_cosine += cfm_metrics["v_cosine_sim"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "cfm_loss": total_cfm / n_batches,
        "oat_loss": total_oat / n_batches,
        "v_cosine_sim": total_v_cosine / n_batches,
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
    total_cfm = 0.0
    total_v_cosine = 0.0
    n_batches = 0

    for batch in dataloader:
        embeddings = batch["embedding"].to(device)
        x_1 = batch["latent"].to(device)
        x_0 = torch.randn_like(x_1)
        context = embeddings.unsqueeze(1).expand(-1, args.num_context_tokens, -1)

        cfm_loss, cfm_metrics = conditional_flow_matching_loss(
            model.module if hasattr(model, 'module') else model,
            x_0, x_1, context,
            sigma_min=args.sigma_min,
        )

        total_loss += cfm_loss.item()
        total_cfm += cfm_metrics["cfm_loss"]
        total_v_cosine += cfm_metrics["v_cosine_sim"]
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cfm_loss": total_cfm / max(n_batches, 1),
        "v_cosine_sim": total_v_cosine / max(n_batches, 1),
    }


def train_worker(rank: int, world_size: int, args):
    """Main training worker for each GPU."""
    if world_size > 1:
        setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)

    # Load VAE for encoding embeddings to latents
    print_rank0(f"Loading VAE from {args.vae}...")
    vae_ckpt = torch.load(args.vae, map_location=device, weights_only=False)

    vae = InstructionVAE(
        input_dim=768,
        latent_dim=args.latent_dim,
    ).to(device)
    # Support both standalone VAE and translator checkpoint formats
    if "vae_state_dict" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["vae_state_dict"])
        print_rank0("Loaded VAE from translator checkpoint")
    elif "model_state_dict" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["model_state_dict"])
        print_rank0("Loaded VAE from standalone checkpoint")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(vae_ckpt.keys())}")
    vae.eval()

    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False

    print_rank0("VAE loaded and frozen")

    barrier()

    # Load embeddings
    print_rank0(f"Loading embeddings from {args.embeddings}...")
    data = torch.load(args.embeddings, map_location="cpu", weights_only=False)

    embeddings = data["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    print_rank0(f"Embeddings shape: {embeddings.shape}")

    # Encode all embeddings to latents using VAE
    print_rank0("Encoding embeddings to latents...")
    latents_list = []
    batch_size_encode = 512

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size_encode):
            batch = embeddings[i:i+batch_size_encode].to(device)
            z = vae.encode_mu(batch)  # Deterministic encoding
            latents_list.append(z.cpu())

    latents = torch.cat(latents_list, dim=0)
    print_rank0(f"Latents shape: {latents.shape}")

    # Train/val split
    n_val = int(len(embeddings) * args.val_ratio)
    indices = torch.randperm(len(embeddings))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = FlowMatchingDataset(
        embeddings[train_indices],
        latents[train_indices],
    )
    val_dataset = FlowMatchingDataset(
        embeddings[val_indices],
        latents[val_indices],
    )

    print_rank0(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create distributed samplers and loaders
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

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

    # Create FlowDiT model
    model = FlowDiT(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        time_embed_dim=256,
        context_dim=args.context_dim,
        num_context_tokens=args.num_context_tokens,
        mlp_ratio=4.0,
        dropout=args.dropout,
        cross_attention=True,
    ).to(device)

    num_params = model.get_num_params()
    print_rank0(f"FlowDiT parameters: {num_params / 1e6:.2f}M")

    # Wrap in DDP (only for multi-GPU)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        print_rank0(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Load model state
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Resume epoch
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

        # Load scheduler state if available, otherwise manually step
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Fallback for old checkpoints without scheduler state
            for _ in range(start_epoch):
                if _ >= args.warmup_epochs:
                    scheduler.step()

        print_rank0(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    start_time = datetime.now()

    print_rank0(f"\nStarting FlowDiT training:")
    print_rank0(f"  Epochs: {args.epochs}")
    print_rank0(f"  Batch size per GPU: {args.batch_size}")
    print_rank0(f"  Total batch size: {args.batch_size * world_size}")
    print_rank0(f"  Learning rate: {args.lr}")
    print_rank0(f"  OAT weight: {args.oat_weight}")
    print_rank0(f"  World size: {world_size}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args, epoch
        )

        # Warmup learning rate
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_factor
        else:
            scheduler.step()

        # Validate periodically
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            val_metrics = validate(model, val_loader, device, args)

            # Reduce metrics (only in multi-GPU mode)
            if world_size > 1:
                train_loss = reduce_tensor(
                    torch.tensor(train_metrics["loss"], device=device), world_size
                ).item()
                val_loss = reduce_tensor(
                    torch.tensor(val_metrics["loss"], device=device), world_size
                ).item()
                val_cosine = reduce_tensor(
                    torch.tensor(val_metrics["v_cosine_sim"], device=device), world_size
                ).item()
            else:
                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]
                val_cosine = val_metrics["v_cosine_sim"]

            if rank == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Train: {train_loss:.4f} (CFM: {train_metrics['cfm_loss']:.4f}, OAT: {train_metrics['oat_loss']:.4f}) | "
                    f"Val: {val_loss:.4f} | "
                    f"V-Cos: {val_cosine:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Time: {elapsed:.1f}m"
                )

            # Early stopping check (after warmup)
            if epoch >= args.warmup_epochs:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    save_checkpoint_ddp(
                        {
                            "epoch": epoch,
                            "model_state_dict": (model.module if hasattr(model, 'module') else model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_loss": val_loss,
                            "val_cosine": val_cosine,
                            "args": vars(args),
                        },
                        str(checkpoint_dir / "flowdit_best.pt"),
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
                    "model_state_dict": (model.module if hasattr(model, 'module') else model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                },
                str(checkpoint_dir / f"flowdit_epoch{epoch+1:05d}.pt"),
                rank,
            )

    # Final save
    save_checkpoint_ddp(
        {
            "epoch": args.epochs,
            "model_state_dict": (model.module if hasattr(model, 'module') else model).state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        str(checkpoint_dir / "flowdit_alpaca_final.pt"),
        rank,
    )

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print_rank0(f"\nTraining complete! Total time: {total_time:.1f} minutes")
    print_rank0(f"Best validation loss: {best_val_loss:.4f}")

    if world_size > 1:
        cleanup_ddp()


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        train_worker(local_rank, world_size, args)
    else:
        print("Running single GPU training...")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
