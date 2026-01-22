#!/usr/bin/env python3
"""
Train FlowDiT with Auxiliary Losses for semantic preservation.

The problem: Standard flow matching causes mode collapse.
Solution: Add auxiliary losses that enforce:
1. Semantic preservation: cos_sim(original, projected) should be high
2. Norm preservation: ||projected|| should match ||original||

Usage:
    uv run python flowpo_hd/scripts/train_flow_dit_aux.py \
        --data-path flowpo_hd/data/mega_raw_encoded.pt \
        --epochs 50 \
        --aux-semantic-weight 0.5 \
        --aux-norm-weight 0.1
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from flowpo_hd.flow_dit import FlowDiT, integrate_euler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """Dataset from pre-computed SONAR embeddings."""

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def sample_timesteps_u_shaped(batch_size: int, device: torch.device, a: float = 4.0) -> torch.Tensor:
    """U-shaped timestep distribution - more weight at t=0 and t=1."""
    u = torch.rand(batch_size, device=device)
    centered = 2.0 * u - 1.0
    sign = torch.sign(centered)
    abs_centered = torch.abs(centered)
    t = 0.5 + 0.5 * sign * (1.0 - torch.exp(-a * abs_centered))
    return t.clamp(0.001, 0.999)


def compute_ot_pairing(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """Approximate OT pairing using Sinkhorn."""
    C = torch.cdist(x_0, x_1, p=2).pow(2)
    K = torch.exp(-C / 0.05)
    B = x_0.shape[0]
    u = torch.ones(B, device=x_0.device) / B

    for _ in range(20):
        v = 1.0 / (K.T @ u + 1e-8)
        u = 1.0 / (K @ v + 1e-8)

    P = torch.diag(u) @ K @ torch.diag(v)
    return P.argmax(dim=0)


def compute_loss(
    model: FlowDiT,
    x_1: torch.Tensor,
    use_ot: bool = True,
    aux_semantic_weight: float = 0.0,
    aux_norm_weight: float = 0.0,
    aux_proj_steps: int = 10,
) -> dict:
    """
    Compute flow matching loss with auxiliary losses.

    Args:
        model: FlowDiT model
        x_1: Target embeddings (B, D)
        use_ot: Use optimal transport pairing
        aux_semantic_weight: Weight for semantic preservation loss
        aux_norm_weight: Weight for norm preservation loss
        aux_proj_steps: ODE steps for projection in aux loss
    """
    B, D = x_1.shape
    device = x_1.device

    # Sample noise
    x_0 = torch.randn_like(x_1)

    # OT pairing
    if use_ot and B > 1:
        perm = compute_ot_pairing(x_0, x_1)
        x_0 = x_0[perm]

    # Sample timesteps (U-shaped)
    t = sample_timesteps_u_shaped(B, device)

    # Linear interpolation: x_t = t*x_1 + (1-t)*x_0
    t_view = t.view(-1, 1)
    x_t = t_view * x_1 + (1 - t_view) * x_0

    # Target velocity (straight line from noise to data)
    u_target = x_1 - x_0

    # Predicted velocity
    v_pred = model(x_t, t)

    # Flow matching loss
    flow_loss = F.mse_loss(v_pred, u_target)

    # Auxiliary losses
    semantic_loss = torch.tensor(0.0, device=device)
    norm_loss = torch.tensor(0.0, device=device)

    if aux_semantic_weight > 0 or aux_norm_weight > 0:
        # Project x_1 through flow: treat as t=0.5 and integrate to t=1
        # This tests if the model preserves semantics
        with torch.no_grad():
            # Add small noise to x_1 (simulating perturbation)
            noise_scale = 0.05 * x_1.norm(dim=-1, keepdim=True)
            x_noisy = x_1 + torch.randn_like(x_1) * noise_scale

        # Project back (with gradient for training)
        x_proj = integrate_euler(model, x_noisy, num_steps=aux_proj_steps, t_start=0.3, t_end=1.0)

        if aux_semantic_weight > 0:
            # Semantic loss: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(x_1, x_proj, dim=-1)
            semantic_loss = (1 - cos_sim).mean()

        if aux_norm_weight > 0:
            # Norm preservation loss
            orig_norms = x_1.norm(dim=-1)
            proj_norms = x_proj.norm(dim=-1)
            norm_loss = F.mse_loss(proj_norms, orig_norms)

    # Total loss
    total_loss = flow_loss + aux_semantic_weight * semantic_loss + aux_norm_weight * norm_loss

    # Metrics
    with torch.no_grad():
        cos_sim_vel = F.cosine_similarity(v_pred, u_target, dim=-1).mean()
        v_norm = v_pred.norm(dim=-1).mean()
        u_norm = u_target.norm(dim=-1).mean()

    return {
        'loss': total_loss,
        'flow_loss': flow_loss,
        'semantic_loss': semantic_loss,
        'norm_loss': norm_loss,
        'cos_sim_vel': cos_sim_vel,
        'v_norm': v_norm,
        'u_norm': u_norm,
    }


def validate(model: FlowDiT, val_loader: DataLoader, device: torch.device) -> dict:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x_1 = batch.to(device)
            result = compute_loss(model, x_1, use_ot=True)
            total_loss += result['flow_loss'].item()
            total_cos_sim += result['cos_sim_vel'].item()
            n_batches += 1

    model.train()

    return {
        'val_loss': total_loss / n_batches,
        'val_cos_sim': total_cos_sim / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='flowpo_hd/data/mega_raw_encoded.pt')
    parser.add_argument('--checkpoint-dir', type=str, default='flowpo_hd/checkpoints_flow_dit_aux')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--aux-semantic-weight', type=float, default=0.5,
                       help='Weight for semantic preservation loss')
    parser.add_argument('--aux-norm-weight', type=float, default=0.1,
                       help='Weight for norm preservation loss')
    parser.add_argument('--aux-proj-steps', type=int, default=10,
                       help='ODE steps for projection in aux loss')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of training samples (for debugging)')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, map_location='cpu')
    embeddings = data['embeddings']

    if args.max_samples:
        embeddings = embeddings[:args.max_samples]

    logger.info(f"Loaded {len(embeddings):,} embeddings")
    logger.info(f"  Norm: {embeddings.norm(dim=-1).mean():.4f} ± {embeddings.norm(dim=-1).std():.4f}")

    # Split
    val_size = min(10000, len(embeddings) // 100)
    train_size = len(embeddings) - val_size

    indices = torch.randperm(len(embeddings), generator=torch.Generator().manual_seed(42))
    train_emb = embeddings[indices[:train_size]]
    val_emb = embeddings[indices[train_size:]]

    train_dataset = EmbeddingDataset(train_emb)
    val_dataset = EmbeddingDataset(val_emb)

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

    logger.info(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    # Create model
    model = FlowDiT(
        latent_dim=1024,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_embed_dim=256,
        mlp_ratio=2.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Resume
    global_step = 0
    best_val_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"Resumed at step {global_step}")

    # Training
    logger.info("=" * 60)
    logger.info("Training FlowDiT with Auxiliary Losses")
    logger.info("=" * 60)
    logger.info(f"  aux_semantic_weight: {args.aux_semantic_weight}")
    logger.info(f"  aux_norm_weight: {args.aux_norm_weight}")
    logger.info(f"  aux_proj_steps: {args.aux_proj_steps}")
    logger.info("=" * 60)

    model.train()
    steps_without_improvement = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            x_1 = batch.to(device)

            # LR warmup
            if global_step < args.warmup_steps:
                lr_scale = global_step / args.warmup_steps
            else:
                lr_scale = 1.0
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

            # Forward
            result = compute_loss(
                model, x_1,
                use_ot=True,
                aux_semantic_weight=args.aux_semantic_weight,
                aux_norm_weight=args.aux_norm_weight,
                aux_proj_steps=args.aux_proj_steps,
            )

            # Backward
            optimizer.zero_grad()
            result['loss'].backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                logger.info(
                    f"Step {global_step} | loss={result['loss'].item():.4f} "
                    f"flow={result['flow_loss'].item():.4f} "
                    f"sem={result['semantic_loss'].item():.4f} "
                    f"norm={result['norm_loss'].item():.4f} "
                    f"cos={result['cos_sim_vel'].item():.4f} "
                    f"grad={grad_norm:.4f}"
                )

            # Validation
            if global_step % args.val_interval == 0:
                val_metrics = validate(model, val_loader, device)
                logger.info(
                    f"Validation | loss={val_metrics['val_loss']:.4f} "
                    f"cos_sim={val_metrics['val_cos_sim']:.4f}"
                )

                # Checkpointing
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    steps_without_improvement = 0
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, f"{args.checkpoint_dir}/best.pt")
                    logger.info(f"Saved best checkpoint (val_loss={best_val_loss:.4f})")
                else:
                    steps_without_improvement += args.val_interval

            # Periodic save
            if global_step % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, f"{args.checkpoint_dir}/latest.pt")

            # Early stopping
            if args.patience > 0 and steps_without_improvement >= args.patience:
                logger.info(f"Early stopping at step {global_step}")
                return

        logger.info(f"Epoch {epoch+1}/{args.epochs} complete")

    logger.info("Training complete!")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
