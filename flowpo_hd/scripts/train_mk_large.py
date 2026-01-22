#!/usr/bin/env python3
"""
Train ManifoldKeeper on 1.1M universal instructions.

Uses pre-computed unnormalized SONAR embeddings from sonar_unified_unnorm.pt.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from flowpo_hd.config import FlowPOHDConfig, get_device
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP
from flowpo_hd.training.train_manifold_keeper import (
    compute_flow_matching_loss,
    ManifoldKeeperTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class PrecomputedEmbeddingDataset(Dataset):
    """Dataset from pre-computed SONAR embeddings."""

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        logger.info(f"Dataset: {len(self)} samples, dim={embeddings.shape[1]}")
        logger.info(f"  mean_norm={embeddings.norm(dim=-1).mean():.4f}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)  # Large batches for OT
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str,
                       default="/home/prusek/NLP/lido_pp/data/sonar_unified_unnorm.pt")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="/home/prusek/NLP/flowpo_hd/checkpoints")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device(get_device(args.device))
    logger.info(f"Device: {device}")

    # Load pre-computed embeddings
    logger.info(f"Loading embeddings from {args.data_path}...")
    data = torch.load(args.data_path, map_location="cpu")
    embeddings = data["embeddings"]
    logger.info(f"Loaded {len(embeddings):,} embeddings, shape={embeddings.shape}")

    # Create dataset and dataloader
    dataset = PrecomputedEmbeddingDataset(embeddings)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"DataLoader: {len(train_loader)} batches per epoch")

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

    # Create trainer
    trainer = ManifoldKeeperTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # No validation for now
        lr=args.lr,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        use_ot=True,
        timestep_sampling="u_shaped",
        u_shaped_a=4.0,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"  epochs={args.epochs}")
    logger.info(f"  batch_size={args.batch_size}")
    logger.info(f"  lr={args.lr}")
    logger.info(f"  warmup_steps={args.warmup_steps}")
    logger.info(f"  patience={args.patience}")

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
