#!/usr/bin/env python3
"""
Phase 1 Training: Unsupervised Manifold Learning with Rectified Flow.

Trains the MatryoshkaEncoder and RectifiedFlowDecoder on GTR embeddings.

Phases:
1. Standard CFM training with Matryoshka loss
2. Reflow training for 1-step decoding

Usage:
    # Single GPU
    python ecoflow_bo/train_manifold.py --batch-size 2048 --epochs 100

    # DDP on 2x L40S
    torchrun --nproc_per_node=2 ecoflow_bo/train_manifold.py \
        --batch-size 2048 --epochs 100 --latent-dim 8
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import torch_optimizer as optim_extra  # LAMB optimizer for large batches

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecoflow_bo.config import (
    EcoFlowConfig,
    EncoderConfig,
    VelocityNetConfig,
    DecoderConfig,
    TrainingConfig,
)
from ecoflow_bo.encoder import MatryoshkaEncoder, SimCSEAugmentor
from ecoflow_bo.velocity_network import VelocityNetwork
from ecoflow_bo.cfm_decoder import RectifiedFlowDecoder
from ecoflow_bo.losses import EcoFlowLoss
from ecoflow_bo.data import create_dataloaders, ReflowDataset


def setup_gpu_optimizations():
    """Enable GPU optimizations for Ampere/Hopper/Ada GPUs (A100, H100, L40S)."""
    # TF32 for faster matmuls on A100/H100/L40S
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True


def compute_scaled_lr(base_lr: float, batch_size: int, base_batch_size: int = 256) -> float:
    """
    Compute scaled learning rate for large batch training.

    Uses linear scaling with cap for very large batches.

    Args:
        base_lr: Base learning rate (tuned for base_batch_size)
        batch_size: Actual batch size
        base_batch_size: Reference batch size (default 256)

    Returns:
        Scaled learning rate
    """
    # Linear scaling capped at 64x to prevent instability
    scale_factor = min(batch_size / base_batch_size, 64.0)
    return base_lr * scale_factor


def create_optimizer(
    params,
    optimizer_type: str,
    lr: float,
    weight_decay: float = 0.01,
    batch_size: int = 256,
):
    """
    Create optimizer with proper configuration for batch size.

    Args:
        params: Model parameters
        optimizer_type: 'lamb' for large batches (>4K), 'adamw' for smaller batches
        lr: Base learning rate
        weight_decay: Weight decay coefficient
        batch_size: Training batch size

    Returns:
        Configured optimizer
    """
    # Scale LR for large batches when using LAMB
    if optimizer_type.lower() == 'lamb':
        scaled_lr = compute_scaled_lr(lr, batch_size, base_batch_size=256)
        print(f"LAMB optimizer: base_lr={lr:.2e}, scaled_lr={scaled_lr:.2e} (linear capped at 64x, batch_size={batch_size})")
        return optim_extra.Lamb(
            params,
            lr=scaled_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-6,
        )
    else:
        # For AdamW, use square root scaling (more conservative)
        if batch_size > 1024:
            import math
            scale_factor = math.sqrt(batch_size / 256)
            scaled_lr = lr * scale_factor
            print(f"AdamW optimizer: base_lr={lr:.2e}, scaled_lr={scaled_lr:.2e} (sqrt scaling, batch_size={batch_size})")
        else:
            scaled_lr = lr
            print(f"AdamW optimizer: lr={scaled_lr:.2e} (batch_size={batch_size})")

        return AdamW(params, lr=scaled_lr, weight_decay=weight_decay)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


def save_checkpoint(
    encoder: nn.Module,
    velocity_net: nn.Module,
    decoder: RectifiedFlowDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: EcoFlowConfig,
    checkpoint_dir: str,
    is_best: bool = False,
    is_reflowed: bool = False,
):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    encoder_state = (
        encoder.module.state_dict()
        if hasattr(encoder, "module")
        else encoder.state_dict()
    )
    velocity_net_state = (
        velocity_net.module.state_dict()
        if hasattr(velocity_net, "module")
        else velocity_net.state_dict()
    )

    checkpoint = {
        "encoder": encoder_state,
        "velocity_net": velocity_net_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "is_reflowed": is_reflowed,
    }

    # Only save best.pt (no intermediate checkpoints to save disk space)
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint: {best_path} (epoch {epoch})")


def train_epoch(
    encoder: nn.Module,
    decoder: RectifiedFlowDecoder,
    train_loader,
    optimizer,
    loss_fn: EcoFlowLoss,
    epoch: int,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    rank: int,
):
    """Train for one epoch."""
    encoder.train()
    decoder.velocity_net.train()

    augmentor = SimCSEAugmentor(encoder.module if hasattr(encoder, "module") else encoder)

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process(rank))

    for batch_idx, x in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        B = x.shape[0]

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            # Encode with two views for contrastive learning
            z1, mu, log_sigma = encoder(x)

            # Second forward pass for SimCSE (different dropout mask)
            z2, _, _ = encoder(x)

            # CFM loss
            cfm_loss = decoder.compute_cfm_loss(x, z1)

            # Combined loss with annealing
            loss, details = loss_fn(cfm_loss, mu, log_sigma, z1, z2, epoch)

        # Backward with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.velocity_net.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        if is_main_process(rank) and batch_idx % 50 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cfm": f"{details['loss_cfm']:.4f}",
                "kl": f"{details['loss_kl']:.4f}",
                "contr": f"{details['loss_contrastive']:.4f}",
            })

    return total_loss / n_batches


@torch.no_grad()
def validate(
    encoder: nn.Module,
    decoder: RectifiedFlowDecoder,
    val_loader,
    device: str,
    rank: int,
):
    """Validate reconstruction quality."""
    encoder.eval()
    decoder.velocity_net.eval()

    total_cosine_sim = 0.0
    total_l2_error = 0.0
    n_samples = 0

    for x in tqdm(val_loader, desc="Validating", disable=not is_main_process(rank)):
        x = x.to(device, non_blocking=True)

        # Encode
        enc_module = encoder.module if hasattr(encoder, "module") else encoder
        z = enc_module.encode_deterministic(x)

        # Decode
        x_recon = decoder.decode_deterministic(z)

        # Metrics
        cosine_sim = nn.functional.cosine_similarity(x, x_recon, dim=-1).mean()
        l2_error = torch.norm(x - x_recon, dim=-1).mean()

        total_cosine_sim += cosine_sim.item() * x.shape[0]
        total_l2_error += l2_error.item() * x.shape[0]
        n_samples += x.shape[0]

    # Aggregate across processes
    if dist.is_initialized():
        metrics = torch.tensor(
            [total_cosine_sim, total_l2_error, n_samples], device=device
        )
        dist.all_reduce(metrics)
        total_cosine_sim, total_l2_error, n_samples = metrics.tolist()

    return {
        "cosine_sim": total_cosine_sim / n_samples,
        "l2_error": total_l2_error / n_samples,
    }


def train_reflow(
    encoder: nn.Module,
    decoder: RectifiedFlowDecoder,
    train_loader,
    optimizer,
    epochs: int,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    rank: int,
    checkpoint_dir: str,
    config: EcoFlowConfig,
):
    """Train reflow phase for straight trajectories."""
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("REFLOW TRAINING PHASE")
        print("=" * 60)

    encoder.eval()
    decoder.velocity_net.train()

    # Collect all training embeddings
    if is_main_process(rank):
        print("Collecting training embeddings...")

    all_z = []
    enc_module = encoder.module if hasattr(encoder, "module") else encoder

    with torch.no_grad():
        for x in tqdm(train_loader, desc="Encoding", disable=not is_main_process(rank)):
            x = x.to(device, non_blocking=True)
            z = enc_module.encode_deterministic(x)
            all_z.append(z)

    all_z = torch.cat(all_z, dim=0)

    if is_main_process(rank):
        print(f"Encoded {len(all_z)} samples, generating reflow pairs...")

    # Generate reflow pairs
    x_0, z, x_1 = decoder.generate_reflow_pairs(all_z, n_ode_steps=50, batch_size=1024)

    if is_main_process(rank):
        print(f"Generated {len(x_0)} reflow pairs")
        straightness = decoder.get_trajectory_straightness(z[:100])
        print(f"Pre-reflow straightness: {straightness:.4f}")

    # Create reflow dataloader
    reflow_dataset = ReflowDataset(x_0, z, x_1)
    reflow_loader = torch.utils.data.DataLoader(
        reflow_dataset,
        batch_size=config.training.reflow_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # Reflow training
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            reflow_loader,
            desc=f"Reflow Epoch {epoch + 1}/{epochs}",
            disable=not is_main_process(rank),
        )

        for x_0_batch, z_batch, x_1_batch in pbar:
            x_0_batch = x_0_batch.to(device, non_blocking=True)
            z_batch = z_batch.to(device, non_blocking=True)
            x_1_batch = x_1_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                loss = decoder.compute_reflow_loss(x_0_batch, z_batch, x_1_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.velocity_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            if is_main_process(rank):
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches

        if is_main_process(rank):
            straightness = decoder.get_trajectory_straightness(z[:100])
            print(f"Reflow Epoch {epoch + 1}: loss={avg_loss:.4f}, straightness={straightness:.4f}")

    decoder.mark_as_reflowed()

    # Save reflowed checkpoint
    if is_main_process(rank):
        save_checkpoint(
            encoder,
            decoder.velocity_net,
            decoder,
            optimizer,
            -1,  # Special epoch for reflow
            config,
            checkpoint_dir,
            is_best=True,
            is_reflowed=True,
        )
        print(f"Post-reflow straightness: {straightness:.4f}")
        print(f"Decoder now uses {decoder.config.euler_steps}-step generation")


def main():
    parser = argparse.ArgumentParser(description="Train EcoFlow-BO manifold")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--reflow-epochs", type=int, default=50, help="Reflow epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--embeddings-path", type=str, default="datasets/gtr_embeddings_full.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="results/ecoflow_checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--skip-reflow", action="store_true", help="Skip reflow training")
    parser.add_argument("--optimizer", type=str, default="lamb", choices=["lamb", "adamw"],
                        help="Optimizer type: 'lamb' for large batches (>4K), 'adamw' for smaller")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Warmup epochs (default: 10 for LAMB, 5 for AdamW)")
    args = parser.parse_args()

    # Set default warmup based on optimizer (longer warmup for large batch LAMB)
    if args.warmup_epochs is None:
        if args.optimizer == 'lamb' and args.batch_size >= 4096:
            args.warmup_epochs = 10  # More warmup for large batch LAMB
        else:
            args.warmup_epochs = 5

    # Setup distributed
    rank, world_size, local_rank, distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Enable A100/H100 optimizations (TF32, cuDNN benchmark)
    setup_gpu_optimizations()

    if is_main_process(rank):
        print("=" * 60)
        print("EcoFlow-BO Manifold Training")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Distributed: {distributed} (world_size={world_size})")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"Epochs: {args.epochs} + {args.reflow_epochs} reflow")
        print(f"Latent dim: {args.latent_dim}")
        print(f"Optimizer: {args.optimizer.upper()} (large batch optimized)")
        print(f"Base LR: {args.lr:.2e}")
        print("=" * 60)

    # Configuration
    config = EcoFlowConfig(
        encoder=EncoderConfig(latent_dim=args.latent_dim),
        velocity_net=VelocityNetConfig(condition_dim=args.latent_dim),
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            reflow_epochs=args.reflow_epochs,
        ),
    )

    # Create models
    encoder = MatryoshkaEncoder(config.encoder).to(device)
    velocity_net = VelocityNetwork(config.velocity_net).to(device)
    decoder = RectifiedFlowDecoder(velocity_net, config.decoder)

    if is_main_process(rank):
        enc_params = sum(p.numel() for p in encoder.parameters())
        vel_params = sum(p.numel() for p in velocity_net.parameters())
        print(f"Encoder params: {enc_params:,}")
        print(f"Velocity net params: {vel_params:,}")
        print(f"Total params: {enc_params + vel_params:,}")

    # Wrap with DDP
    if distributed:
        encoder = DDP(encoder, device_ids=[local_rank])
        velocity_net_ddp = DDP(velocity_net, device_ids=[local_rank])
        decoder.velocity_net = velocity_net_ddp

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        embeddings_path=args.embeddings_path,
        batch_size=args.batch_size,
        distributed=distributed,
    )

    # Optimizer with proper LR scaling for large batches
    params = list(encoder.parameters()) + list(velocity_net.parameters())
    if is_main_process(rank):
        print(f"Optimizer: {args.optimizer.upper()}")
    optimizer = create_optimizer(
        params,
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=config.training.weight_decay,
        batch_size=args.batch_size,
    )

    # LR scheduler with warmup (longer warmup for large batches)
    warmup_epochs = args.warmup_epochs
    if is_main_process(rank):
        print(f"Warmup epochs: {warmup_epochs}")

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs  # Start from 1% of LR
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    # Loss and scaler
    loss_fn = EcoFlowLoss(config.encoder, config.training)
    scaler = GradScaler("cuda", enabled=config.training.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        # weights_only=False required for EcoFlowConfig dataclass in checkpoint
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder"])
        velocity_net.load_state_dict(checkpoint["velocity_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if is_main_process(rank):
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_cosine_sim = 0.0

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            encoder,
            decoder,
            train_loader,
            optimizer,
            loss_fn,
            epoch,
            scaler,
            device,
            config.training.use_amp,
            rank,
        )

        scheduler.step()

        # Validate
        val_metrics = validate(encoder, decoder, val_loader, device, rank)

        if is_main_process(rank):
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"cosine_sim={val_metrics['cosine_sim']:.4f}, "
                f"l2_error={val_metrics['l2_error']:.4f}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Save checkpoint
            is_best = val_metrics["cosine_sim"] > best_cosine_sim
            if is_best:
                best_cosine_sim = val_metrics["cosine_sim"]

            # Only save best checkpoint (not intermediate epochs)
            if is_best:
                save_checkpoint(
                    encoder,
                    velocity_net,
                    decoder,
                    optimizer,
                    epoch,
                    config,
                    args.checkpoint_dir,
                    is_best=True,
                )

    # Reflow training
    if not args.skip_reflow:
        # Create new optimizer for reflow (lower LR, use same optimizer type)
        reflow_optimizer = create_optimizer(
            velocity_net.parameters(),
            optimizer_type=args.optimizer,
            lr=args.lr * 0.1,  # Lower base LR for reflow
            weight_decay=config.training.weight_decay,
            batch_size=config.training.reflow_batch_size,
        )

        train_reflow(
            encoder,
            decoder,
            train_loader,
            reflow_optimizer,
            config.training.reflow_epochs,
            scaler,
            device,
            config.training.use_amp,
            rank,
            args.checkpoint_dir,
            config,
        )

    cleanup_distributed()

    if is_main_process(rank):
        print("\nTraining complete!")
        print(f"Best cosine similarity: {best_cosine_sim:.4f}")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
