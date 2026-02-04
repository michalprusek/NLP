"""Train flow matching model on SELFIES VAE molecular embeddings.

This script trains a spherical-OT flow model for GuacaMol molecular optimization.
Uses SELFIES VAE from LOLBO with 256D latent space.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.train_flow_guacamol \
        --epochs 1000 --batch-size 256 --n-samples 10000

The trained model can be used with run_guacamol.py for Bayesian optimization.
"""

import argparse
import copy
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)

from rielbo.velocity_network import VelocityNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
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


def encode_molecules(
    codec,
    smiles_list: list[str],
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode molecules in batches."""
    logger.info(f"Encoding {len(smiles_list)} molecules...")

    embeddings_list = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding"):
        batch = smiles_list[i:i + batch_size]
        with torch.no_grad():
            emb = codec.encode(batch)
        embeddings_list.append(emb.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    logger.info(f"Encoded embeddings shape: {embeddings.shape}")
    return embeddings


def compute_embedding_stats(embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute normalization statistics for embeddings."""
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    norm = embeddings.norm(dim=-1).mean()

    logger.info(f"Embedding statistics:")
    logger.info(f"  Mean norm: {norm:.4f}")
    logger.info(f"  Mean (abs): {mean.abs().mean():.4f}")
    logger.info(f"  Std: {std.mean():.4f}")

    return {
        "mean": mean,
        "std": std,
        "embedding_norm": norm,
    }


def train_flow(args: argparse.Namespace) -> None:
    """Main training function."""
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load molecules
    if args.zinc:
        logger.info("Loading ZINC data...")
        from shared.guacamol.data import load_zinc_smiles
        smiles_list = load_zinc_smiles(
            path=args.zinc_path,
            n_samples=args.n_samples,
        )
    else:
        logger.info("Loading GuacaMol data...")
        from shared.guacamol.data import load_guacamol_data
        smiles_list, scores, _ = load_guacamol_data(
            n_samples=args.n_samples,
            task_id=args.task_id,
        )
    logger.info(f"Loaded {len(smiles_list)} molecules")

    # Encode molecules
    logger.info("Loading SELFIES VAE codec...")
    from shared.guacamol.codec import SELFIESVAECodec
    codec = SELFIESVAECodec.from_pretrained(device=str(device))

    embeddings = encode_molecules(codec, smiles_list, batch_size=args.encode_batch_size)
    # Keep on CPU for DataLoader compatibility

    # Compute statistics
    norm_stats = compute_embedding_stats(embeddings)
    embedding_norm = norm_stats["embedding_norm"].item()

    # For spherical flow, normalize to unit sphere
    if args.spherical:
        logger.info("Normalizing embeddings to unit sphere for spherical flow...")
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        norm_stats = None  # No mean/std normalization for spherical
    else:
        # Standard z-score normalization
        embeddings = (embeddings - norm_stats["mean"]) / norm_stats["std"].clamp(min=1e-6)

    # Create dataloader (keep data on CPU, move to GPU in training loop)
    dataset = TensorDataset(embeddings)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid CUDA issues
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"DataLoader: {len(train_loader)} batches")

    # Create model
    model = VelocityNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Flow matcher
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    logger.info("Using OT-CFM flow matching")

    # EMA
    ema = EMAModel(model, decay=args.ema_decay)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"Total steps: {total_steps}, warmup: {args.warmup_steps}")

    # Training loop
    best_loss = float("inf")
    global_step = 0

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x1 = batch[0].to(device)  # Target embeddings, move to GPU

            # For spherical flow, sample noise on unit sphere
            if args.spherical:
                x0 = torch.randn_like(x1)
                x0 = F.normalize(x0, p=2, dim=-1)
            else:
                x0 = torch.randn_like(x1)

            # Sample flow
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            # Already on device since x0, x1 are on device

            # For spherical flow, project xt to sphere
            if args.spherical:
                xt = F.normalize(xt, p=2, dim=-1)

            # Forward pass
            vt = model(xt, t)
            loss = F.mse_loss(vt, ut)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / num_batches
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Avg loss: {avg_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss

            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "norm_stats": {
                    k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in norm_stats.items()
                } if norm_stats else None,
                "embedding_norm": embedding_norm,
                "is_spherical": args.spherical,
                "input_dim": args.input_dim,
            }

            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path} (loss={best_loss:.6f})")

        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:04d}.pt")
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "norm_stats": {
                    k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in norm_stats.items()
                } if norm_stats else None,
                "embedding_norm": embedding_norm,
                "is_spherical": args.spherical,
                "input_dim": args.input_dim,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info(f"Training complete. Best loss: {best_loss:.6f}")


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train flow matching model on SELFIES VAE molecular embeddings"
    )

    # Data arguments
    parser.add_argument(
        "--n-samples", type=int, default=10000,
        help="Number of molecules to use for training"
    )
    parser.add_argument(
        "--task-id", type=str, default="pdop",
        help="GuacaMol task ID for loading data"
    )
    parser.add_argument(
        "--zinc", action="store_true",
        help="Use ZINC dataset instead of GuacaMol"
    )
    parser.add_argument(
        "--zinc-path", type=str, default="datasets/zinc/zinc_all.txt",
        help="Path to ZINC SMILES file"
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=256,
        help="Batch size for encoding molecules"
    )

    # Model arguments
    parser.add_argument(
        "--input-dim", type=int, default=256,
        help="Input dimension (256 for SELFIES VAE)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=6,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of attention heads"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.9999,
        help="EMA decay rate"
    )

    # Flow arguments
    parser.add_argument(
        "--spherical", action="store_true",
        help="Use spherical flow (normalize to unit sphere)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="rielbo/checkpoints/guacamol_flow",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--save-interval", type=int, default=100,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    train_flow(args)


if __name__ == "__main__":
    main()
