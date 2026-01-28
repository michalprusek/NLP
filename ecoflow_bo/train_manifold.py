#!/usr/bin/env python3
"""
Joint Training for EcoFlow-BO: Encoder + DiT Decoder together.

Trains encoder and velocity network jointly with:
- Matryoshka CFM loss (hierarchical: 4D, 8D, 16D)
- KL regularization (latent → N(0,I))
- Contrastive loss (SimCSE-style)

Usage:
    # Single GPU
    python ecoflow_bo/train_manifold.py --batch-size 2048 --epochs 200

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 ecoflow_bo/train_manifold.py --batch-size 2048 --epochs 200
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
import torch_optimizer as optim_extra

sys.path.insert(0, str(Path(__file__).parent.parent))

from ecoflow_bo.config import (
    EcoFlowConfig,
    EncoderConfig,
    DiTVelocityNetConfig,
    DecoderConfig,
    TrainingConfig,
)
from ecoflow_bo.encoder import MatryoshkaEncoder
from ecoflow_bo.velocity_network import VelocityNetwork
from ecoflow_bo.cfm_decoder import RectifiedFlowDecoder
from ecoflow_bo.losses import MatryoshkaCFMLoss, KLDivergenceLoss, InfoNCELoss
from ecoflow_bo.data import create_dataloaders


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def save_checkpoint(encoder, velocity_net, optimizer, epoch, config, path, metrics=None):
    """Save checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder": encoder.module.state_dict() if hasattr(encoder, "module") else encoder.state_dict(),
        "velocity_net": velocity_net.module.state_dict() if hasattr(velocity_net, "module") else velocity_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def train_epoch(
    encoder: nn.Module,
    decoder: RectifiedFlowDecoder,
    train_loader,
    optimizer,
    epoch: int,
    scaler: GradScaler,
    device: str,
    rank: int,
    matryoshka_cfm: MatryoshkaCFMLoss,
    kl_loss_fn: KLDivergenceLoss,
    contrastive_loss_fn: InfoNCELoss,
    kl_weight: float,
    contrastive_weight: float,
    matryoshka_dims: list,
):
    """Train one epoch with joint encoder-decoder optimization."""
    encoder.train()
    decoder.velocity_net.train()

    total_loss = 0.0
    total_cfm = 0.0
    total_kl = 0.0
    total_contr = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main(rank))

    for batch_idx, x in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Encode with two views for contrastive
            z1, mu, log_sigma = encoder(x)
            z2, _, _ = encoder(x)

            # Matryoshka CFM loss (hlavní loss)
            cfm_loss, cfm_details = matryoshka_cfm(decoder, x, z1)

            # KL regularization
            kl_loss = kl_loss_fn(mu, log_sigma)

            # Contrastive loss (random Matryoshka dim)
            dim_idx = torch.randint(len(matryoshka_dims), (1,)).item()
            dim = matryoshka_dims[dim_idx]
            contr_loss = contrastive_loss_fn(z1[:, :dim], z2[:, :dim])

            # Total loss
            loss = cfm_loss + kl_weight * kl_loss + contrastive_weight * contr_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.velocity_net.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cfm += cfm_loss.item()
        total_kl += kl_loss.item()
        total_contr += contr_loss.item()
        n_batches += 1

        if is_main(rank) and batch_idx % 50 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cfm": f"{cfm_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "contr": f"{contr_loss.item():.4f}",
            })

    return {
        "loss": total_loss / n_batches,
        "cfm": total_cfm / n_batches,
        "kl": total_kl / n_batches,
        "contrastive": total_contr / n_batches,
    }


@torch.no_grad()
def validate(encoder, decoder, val_loader, device, rank, n_steps=50):
    """Validate reconstruction quality."""
    encoder.eval()
    decoder.velocity_net.eval()

    total_cosine = 0.0
    total_l2 = 0.0
    n_samples = 0

    for x in tqdm(val_loader, desc="Validating", disable=not is_main(rank)):
        x = x.to(device, non_blocking=True)
        enc = encoder.module if hasattr(encoder, "module") else encoder
        z = enc.encode_deterministic(x)
        x_recon = decoder.decode(z, n_steps=n_steps)

        cosine = nn.functional.cosine_similarity(x, x_recon, dim=-1).mean()
        l2 = torch.norm(x - x_recon, dim=-1).mean()

        total_cosine += cosine.item() * x.shape[0]
        total_l2 += l2.item() * x.shape[0]
        n_samples += x.shape[0]

    if dist.is_initialized():
        metrics = torch.tensor([total_cosine, total_l2, n_samples], device=device)
        dist.all_reduce(metrics)
        total_cosine, total_l2, n_samples = metrics.tolist()

    return {"cosine_sim": total_cosine / n_samples, "l2_error": total_l2 / n_samples}


def main():
    parser = argparse.ArgumentParser(description="Joint EcoFlow-BO Training")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--velocity-lr-mult", type=float, default=3.0,
                        help="Learning rate multiplier for velocity network (default: 3x encoder LR)")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--kl-weight", type=float, default=0.001)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", type=str, default="results/ecoflow_checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--embeddings-path", type=str, default="datasets/gtr_embeddings_full.pt")
    args = parser.parse_args()

    # Distributed setup
    rank, world_size, local_rank, distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if is_main(rank):
        print("=" * 60)
        print("EcoFlow-BO Joint Training (Encoder + DiT Decoder)")
        print("=" * 60)
        print(f"Device: {device}, Distributed: {distributed} (world_size={world_size})")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"Epochs: {args.epochs}")
        print(f"Latent dim: {args.latent_dim}")
        print(f"Loss weights: CFM=1.0, KL={args.kl_weight}, Contrastive={args.contrastive_weight}")
        print("=" * 60)

    # Config
    config = EcoFlowConfig(
        encoder=EncoderConfig(latent_dim=args.latent_dim),
        velocity_net=DiTVelocityNetConfig(condition_dim=args.latent_dim),
        training=TrainingConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr),
    )

    # Models
    encoder = MatryoshkaEncoder(config.encoder).to(device)
    velocity_net = VelocityNetwork(config.velocity_net).to(device)
    decoder = RectifiedFlowDecoder(velocity_net, config.decoder)

    if is_main(rank):
        enc_p = sum(p.numel() for p in encoder.parameters())
        vel_p = sum(p.numel() for p in velocity_net.parameters())
        print(f"Encoder: {enc_p:,} params")
        print(f"Velocity net: {vel_p:,} params")
        print(f"Total: {enc_p + vel_p:,} params")

    # DDP
    if distributed:
        encoder = DDP(encoder, device_ids=[local_rank])
        velocity_net = DDP(velocity_net, device_ids=[local_rank])
        decoder.velocity_net = velocity_net

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        embeddings_path=args.embeddings_path,
        batch_size=args.batch_size,
        distributed=distributed,
    )

    # Optimizer (LAMB for large batch) with separate LR for encoder and velocity net
    scale = min(args.batch_size / 256, 64.0)
    encoder_lr = args.lr * scale
    velocity_lr = encoder_lr * args.velocity_lr_mult  # Higher LR for velocity network

    param_groups = [
        {"params": list(encoder.parameters()), "lr": encoder_lr, "name": "encoder"},
        {"params": list(velocity_net.parameters()), "lr": velocity_lr, "name": "velocity_net"},
    ]

    if is_main(rank):
        print(f"LAMB optimizer (separate LRs):")
        print(f"  Encoder LR: {encoder_lr:.2e} (scaled {scale:.1f}x from {args.lr:.2e})")
        print(f"  Velocity LR: {velocity_lr:.2e} ({args.velocity_lr_mult:.1f}x encoder)")

    optimizer = optim_extra.Lamb(param_groups, weight_decay=0.01)

    # Scheduler
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - 10, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    # Loss functions
    matryoshka_dims = config.encoder.matryoshka_dims
    matryoshka_weights = config.encoder.matryoshka_weights
    matryoshka_cfm = MatryoshkaCFMLoss(matryoshka_dims, matryoshka_weights)
    kl_loss_fn = KLDivergenceLoss()
    contrastive_loss_fn = InfoNCELoss(temperature=0.05)

    scaler = GradScaler("cuda", enabled=True)

    # Resume
    start_epoch = 0
    best_cosine = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        velocity_net.load_state_dict(ckpt["velocity_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_cosine = ckpt.get("metrics", {}).get("cosine_sim", 0.0)
        if is_main(rank):
            print(f"Resumed from epoch {start_epoch}, best_cosine={best_cosine:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, epoch, scaler, device, rank,
            matryoshka_cfm, kl_loss_fn, contrastive_loss_fn,
            args.kl_weight, args.contrastive_weight, matryoshka_dims,
        )
        scheduler.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = validate(encoder, decoder, val_loader, device, rank)
        else:
            val_metrics = None

        if is_main(rank):
            lr = scheduler.get_last_lr()[0]
            log = f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, cfm={train_metrics['cfm']:.4f}, kl={train_metrics['kl']:.4f}, contr={train_metrics['contrastive']:.4f}, lr={lr:.2e}"
            if val_metrics:
                log += f" | val_cosine={val_metrics['cosine_sim']:.4f}, l2={val_metrics['l2_error']:.4f}"
            print(log)

            # Save best
            if val_metrics and val_metrics["cosine_sim"] > best_cosine:
                best_cosine = val_metrics["cosine_sim"]
                save_checkpoint(
                    encoder, velocity_net, optimizer, epoch, config,
                    f"{args.checkpoint_dir}/best.pt", val_metrics
                )
                print(f"  New best! cosine_sim={best_cosine:.4f}")

    cleanup_distributed()

    if is_main(rank):
        print(f"\nTraining complete! Best cosine_sim: {best_cosine:.4f}")
        print(f"Checkpoint: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
