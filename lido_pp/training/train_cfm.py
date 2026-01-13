"""
Text Flow Autoencoder (TFA) Training for FlowPO.

This script trains the TFA to map SONAR embeddings (1024D) to compact latent
space (128D) via simulation-free flow matching with Lipschitz regularization.

The TFA is a core component of FlowPO (Novel Contribution #1).

Usage:
    # Pre-compute SONAR embeddings first
    uv run python -m lido_pp.training.precompute_embeddings \
        --encoder sonar --output lido_pp/data/sonar_embeddings.pt

    # Train TFA
    uv run python -m lido_pp.training.train_cfm \
        --data lido_pp/data/sonar_embeddings.pt \
        --epochs 10000 --batch-size 64

Architecture:
    SONAR 1024D → Linear(256) → ODE Flow → Linear(128) → 128D
    Compression ratio: 8:1 (vs 128:1 in old GritLM→32D)
    Invertible via reverse ODE integration.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lido_pp.backbone.cfm_encoder import (
    TextFlowAutoencoder,
    flow_matching_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text Flow Autoencoder (TFA)")
    parser.add_argument(
        "--data",
        type=str,
        default="lido_pp/data/sonar_embeddings.pt",
        help="Path to pre-computed SONAR embeddings (1024D)",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension (128 recommended for SONAR)",
    )
    parser.add_argument("--flow-dim", type=int, default=256, help="Flow space dimension")
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=20,
        help="ODE integration steps (inference only)",
    )
    parser.add_argument("--lambda-gw", type=float, default=0.0, help="GW loss weight (optional)")
    parser.add_argument("--lambda-recon", type=float, default=0.1, help="Reconstruction loss weight")
    parser.add_argument(
        "--lambda-lip",
        type=float,
        default=0.01,
        help="Lipschitz regularization weight (BO-friendly)",
    )
    parser.add_argument(
        "--lipschitz-bound",
        type=float,
        default=10.0,
        help="Maximum Lipschitz constant",
    )
    parser.add_argument("--warmup-epochs", type=int, default=500, help="LR warmup epochs")
    parser.add_argument("--patience", type=int, default=1000, help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_epoch(
    model: TextFlowAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Train for one epoch (simulation-free Flow Matching with Lipschitz reg)."""
    model.train()
    total_loss = 0.0
    total_fm = 0.0
    total_recon = 0.0
    total_gw = 0.0
    total_lip = 0.0
    n_batches = 0

    for batch in dataloader:
        embeddings = batch[0].to(device)

        optimizer.zero_grad()

        # Forward pass and compute loss (SIMULATION-FREE - no ODE solver!)
        losses = flow_matching_loss(
            model,
            embeddings,
            lambda_recon=args.lambda_recon,
            lambda_gw=args.lambda_gw,
            lambda_lip=args.lambda_lip,
        )

        loss = losses["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_fm += losses["fm"]
        total_recon += losses["recon"]
        total_gw += losses.get("gw", 0.0)
        total_lip += losses.get("lip", 0.0)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "fm": total_fm / n_batches,
        "recon": total_recon / n_batches,
        "gw": total_gw / n_batches,
        "lip": total_lip / n_batches,
    }


@torch.no_grad()
def validate(
    model: TextFlowAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Validate model (simulation-free loss + ODE reconstruction check)."""
    model.eval()
    total_loss = 0.0
    total_fm = 0.0
    total_recon = 0.0
    total_gw = 0.0
    total_lip = 0.0
    total_cos_ode = 0.0  # ODE-based reconstruction cosine
    n_batches = 0
    n_ode_batches = 0

    for i, batch in enumerate(dataloader):
        embeddings = batch[0].to(device)

        # Simulation-free loss (same as training)
        losses = flow_matching_loss(
            model,
            embeddings,
            lambda_recon=args.lambda_recon,
            lambda_gw=args.lambda_gw,
            lambda_lip=0.0,  # Don't compute Lipschitz during validation
        )

        total_loss += losses["loss"].item()
        total_fm += losses["fm"]
        total_recon += losses["recon"]
        total_gw += losses.get("gw", 0.0)
        n_batches += 1

        # Every 10 batches, also check actual ODE reconstruction
        if i % 10 == 0:
            z, x_recon = model(embeddings[:8])  # Uses ODE solver
            cos_sim = F.cosine_similarity(embeddings[:8], x_recon, dim=-1).mean()
            total_cos_ode += cos_sim.item()
            n_ode_batches += 1

    return {
        "loss": total_loss / n_batches,
        "fm": total_fm / n_batches,
        "recon": total_recon / n_batches,
        "gw": total_gw / n_batches,
        "cos_ode": total_cos_ode / max(n_ode_batches, 1),  # Actual ODE reconstruction
    }


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading embeddings from {args.data}...")
    data = torch.load(args.data, weights_only=False)

    if isinstance(data, dict):
        embeddings = data["embeddings"]
        metadata = data.get("metadata", {})
        print(f"Metadata: {metadata}")
    else:
        embeddings = data
        metadata = {}

    print(f"Embeddings shape: {embeddings.shape}")
    input_dim = embeddings.shape[1]

    # Validate input dimensions
    if input_dim == 1024:
        print("[OK] SONAR embeddings detected (1024D)")
    elif input_dim == 4096:
        print("[WARNING] GritLM embeddings detected (4096D)")
        print("For FlowPO, use SONAR embeddings with --encoder sonar")
    elif input_dim == 768:
        print("[WARNING] GTR embeddings detected (768D)")
        print("For FlowPO, use SONAR embeddings with --encoder sonar")
    else:
        print(f"[WARNING] Unexpected embedding dimension {input_dim}D")

    # Split data
    dataset = TensorDataset(embeddings)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {train_size}, Val: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model (Text Flow Autoencoder)
    model = TextFlowAutoencoder(
        input_dim=input_dim,
        flow_dim=args.flow_dim,
        latent_dim=args.latent_dim,
        num_ode_steps=args.ode_steps,  # Only used during inference
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Compression ratio
    compression_ratio = input_dim / args.latent_dim
    print(f"Compression ratio: {compression_ratio:.1f}:1 ({input_dim}D → {args.latent_dim}D)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Scheduler with warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )

    # Training state
    best_val_loss = float("inf")
    best_val_cos = 0.0
    epochs_without_improvement = 0
    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("FlowPO: Text Flow Autoencoder (TFA) Training")
    print("Novel Contribution #1: Simulation-free FM for text autoencoding")
    print("=" * 70)
    print(f"Input dim: {input_dim}D (SONAR)")
    print(f"Flow dim: {args.flow_dim}D")
    print(f"Latent dim: {args.latent_dim}D")
    print(f"Compression: {compression_ratio:.1f}:1")
    print(f"ODE steps (inference): {args.ode_steps}")
    print(f"Lambda recon: {args.lambda_recon}")
    print(f"Lambda GW: {args.lambda_gw}")
    print(f"Lambda Lipschitz: {args.lambda_lip} (BO-friendly regularization)")
    print("-" * 70)
    print("Training: Simulation-free (no ODE solver)")
    print("Inference: Euler integration for encode/decode")
    print("=" * 70 + "\n")

    for epoch in range(1, args.epochs + 1):
        # Warmup LR
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)

        # Validate
        val_metrics = validate(model, val_loader, device, args)

        # Step scheduler after warmup
        if epoch > args.warmup_epochs:
            scheduler.step()

        # Check for improvement (use ODE reconstruction cosine as main metric)
        if val_metrics["cos_ode"] > best_val_cos:
            best_val_cos = val_metrics["cos_ode"]
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_cos_ode": best_val_cos,
                "args": vars(args),
                "metadata": metadata,
                "architecture": "TextFlowAutoencoder",
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
            }, checkpoint_dir / "tfa_best.pt")
        else:
            epochs_without_improvement += 1

        # Logging
        if epoch % args.log_interval == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = datetime.now() - start_time
            print(
                f"Epoch {epoch:5d} | "
                f"FM: {train_metrics['fm']:.4f} | "
                f"Recon: {train_metrics['recon']:.4f} | "
                f"Lip: {train_metrics['lip']:.4f} | "
                f"Val CosODE: {val_metrics['cos_ode']:.4f} | "
                f"Best: {best_val_cos:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed}"
            )

        # Periodic checkpoint
        if epoch % args.checkpoint_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_cos_ode": val_metrics["cos_ode"],
                "args": vars(args),
            }, checkpoint_dir / f"tfa_epoch_{epoch}.pt")

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Final summary
    print("\n" + "=" * 70)
    print("TFA Training Complete")
    print("=" * 70)
    print(f"Best validation ODE cosine: {best_val_cos:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total time: {datetime.now() - start_time}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'tfa_best.pt'}")

    # Test invertibility on best model
    print("\n--- Invertibility Test ---")
    checkpoint = torch.load(checkpoint_dir / "tfa_best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_batch = next(iter(val_loader))[0][:8].to(device)
        z, x_recon = model(test_batch)
        cos_sim = F.cosine_similarity(test_batch, x_recon, dim=-1)
        print(f"Test cosine similarities: {cos_sim.tolist()}")
        print(f"Mean: {cos_sim.mean():.4f}, Min: {cos_sim.min():.4f}, Max: {cos_sim.max():.4f}")

        # Also check latent statistics
        print(f"\nLatent statistics:")
        print(f"  Shape: {z.shape}")
        print(f"  Mean: {z.mean():.4f}, Std: {z.std():.4f}")
        print(f"  Min: {z.min():.4f}, Max: {z.max():.4f}")


if __name__ == "__main__":
    main()
