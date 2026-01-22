#!/usr/bin/env python3
"""
Train ManifoldKeeper on MEGA instruction dataset (1M+ instructions).

Uses pre-computed SONAR embeddings from mega_instructions_latest.pt.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from flowpo_hd.config import FlowPOHDConfig, get_device
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP
from flowpo_hd.training.train_manifold_keeper import ManifoldKeeperTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class PrecomputedEmbeddingDataset(Dataset):
    """Dataset from pre-computed SONAR embeddings."""

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        logger.info(f"Dataset: {len(self):,} samples, dim={embeddings.shape[1]}")
        logger.info(f"  Norm: mean={embeddings.norm(dim=-1).mean():.4f}, std={embeddings.norm(dim=-1).std():.4f}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)  # Large batches for 2x L40S
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-blocks", type=int, default=4)  # More blocks for larger dataset
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str,
                       default="flowpo_hd/data/mega_instructions_latest.pt")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="flowpo_hd/checkpoints_mega")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    # Auxiliary losses for semantic/norm preservation
    parser.add_argument("--use-aux-losses", action="store_true",
                       help="Enable auxiliary losses for semantic/norm preservation")
    parser.add_argument("--aux-semantic-weight", type=float, default=0.1,
                       help="Weight for semantic preservation loss")
    parser.add_argument("--aux-norm-weight", type=float, default=0.1,
                       help="Weight for norm preservation loss (MSE)")
    parser.add_argument("--aux-proj-steps", type=int, default=10,
                       help="ODE steps for projection in aux losses")
    parser.add_argument("--aux-t-threshold", type=float, default=0.0,
                       help="Only apply aux loss when t > threshold (0=all, 0.8=near manifold)")
    args = parser.parse_args()

    device = torch.device(get_device(args.device))
    logger.info(f"Device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load pre-computed embeddings
    logger.info(f"Loading embeddings from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")
    embeddings = data["embeddings"]
    logger.info(f"Loaded {len(embeddings):,} embeddings, shape={embeddings.shape}")

    if "metadata" in data:
        logger.info(f"Metadata: {data['metadata']}")

    # Create dataset and dataloader
    dataset = PrecomputedEmbeddingDataset(embeddings)

    # Split for validation (1% for val)
    val_size = min(10000, len(dataset) // 100)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset):,} samples, {len(val_loader)} batches")

    # Create model
    logger.info("Creating ManifoldKeeper...")
    model = ManifoldKeeperMLP(
        dim=1024,  # SONAR dim
        hidden_dim=args.hidden_dim,
        time_dim=256,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"Parameters: {model.num_params:,} ({model.num_params/1e6:.1f}M)")

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("global_step", 0)
        logger.info(f"Resumed from step {start_step}")

    # Create trainer
    trainer = ManifoldKeeperTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        use_ot=True,
        timestep_sampling="u_shaped",
        u_shaped_a=4.0,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        # Auxiliary losses
        use_aux_losses=args.use_aux_losses,
        aux_semantic_weight=args.aux_semantic_weight,
        aux_norm_weight=args.aux_norm_weight,
        aux_proj_steps=args.aux_proj_steps,
        aux_t_threshold=args.aux_t_threshold,
    )

    # Set start step if resuming
    if start_step > 0:
        trainer.global_step = start_step

    # Train
    logger.info("=" * 60)
    logger.info("Starting MEGA ManifoldKeeper training")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {len(embeddings):,} instructions")
    logger.info(f"  epochs={args.epochs}")
    logger.info(f"  batch_size={args.batch_size}")
    logger.info(f"  lr={args.lr}")
    logger.info(f"  hidden_dim={args.hidden_dim}")
    logger.info(f"  num_blocks={args.num_blocks}")
    logger.info(f"  warmup_steps={args.warmup_steps}")
    logger.info(f"  patience={args.patience}")
    if args.use_aux_losses:
        logger.info(f"  [AUX LOSSES ENABLED]")
        logger.info(f"    semantic_weight={args.aux_semantic_weight}")
        logger.info(f"    norm_weight={args.aux_norm_weight}")
        logger.info(f"    proj_steps={args.aux_proj_steps}")
        logger.info(f"    t_threshold={args.aux_t_threshold}")
    logger.info("=" * 60)

    trainer.train(
        epochs=args.epochs,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        patience=args.patience,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
