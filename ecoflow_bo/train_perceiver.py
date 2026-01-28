"""
EcoFlow-BO Training with Perceiver Decoder (Direct Reconstruction).

This is a simpler training setup that uses:
- MatryoshkaEncoder: 768D → 16D
- PerceiverDecoder: 16D → 768D (direct, no flow matching)

Loss: MSE reconstruction + KL regularization + Contrastive learning

No ODE, no time conditioning, no flow matching - just clean autoencoding.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim_extra
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from typing import Tuple, Optional

from ecoflow_bo.config import EcoFlowConfig, EncoderConfig, TrainingConfig
from ecoflow_bo.encoder import MatryoshkaEncoder
from ecoflow_bo.perceiver_decoder import PerceiverDecoder, PerceiverDecoderConfig
from ecoflow_bo.losses import KLDivergenceLoss, InfoNCELoss
from ecoflow_bo.data import create_dataloaders


def setup_distributed() -> Tuple[int, int, int, bool]:
    """Setup distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        init_process_group("nccl")
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        destroy_process_group()


def is_main(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


class MatryoshkaReconstructionLoss(nn.Module):
    """
    Matryoshka-aware reconstruction loss.

    Computes MSE at multiple z truncation levels (4D, 8D, 16D) with weights.
    This ensures first dimensions carry most information.
    """

    def __init__(
        self,
        matryoshka_dims: list = None,
        matryoshka_weights: list = None,
    ):
        super().__init__()
        if matryoshka_dims is None:
            matryoshka_dims = [4, 8, 16]
        if matryoshka_weights is None:
            matryoshka_weights = [0.4, 0.35, 0.25]

        assert len(matryoshka_dims) == len(matryoshka_weights)
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights

    def forward(
        self,
        decoder: PerceiverDecoder,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Matryoshka reconstruction loss.

        Args:
            decoder: PerceiverDecoder
            z: Full latent [B, latent_dim]
            x_target: Target embeddings [B, 768]

        Returns:
            loss: Weighted reconstruction loss
            details: Dict with loss at each level
        """
        total_loss = 0.0
        details = {}

        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            # Decode with masked z
            x_recon = decoder.forward_with_matryoshka(z, active_dims=dim)

            # MSE loss (normalized by dimension)
            mse = F.mse_loss(x_recon, x_target)

            # Cosine similarity for logging
            cos_sim = F.cosine_similarity(x_recon, x_target, dim=-1).mean()

            total_loss = total_loss + weight * mse
            details[f"mse_dim{dim}"] = mse.item()
            details[f"cosine_dim{dim}"] = cos_sim.item()

        details["recon_total"] = total_loss.item()
        return total_loss, details


def train_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    optimizer,
    scaler: GradScaler,
    device: str,
    recon_loss_fn: MatryoshkaReconstructionLoss,
    kl_loss_fn: KLDivergenceLoss,
    contrastive_loss_fn: InfoNCELoss,
    kl_weight: float,
    contrastive_weight: float,
    matryoshka_dims: list,
) -> dict:
    """Train for one epoch."""
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_contr = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for x in pbar:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.bfloat16):
            # Encoder forward (two views for contrastive)
            z1, mu, log_sigma = encoder(x)
            z2, _, _ = encoder(x)  # Different dropout

            # Reconstruction loss (Matryoshka)
            recon_loss, recon_details = recon_loss_fn(decoder, z1, x)

            # KL loss
            kl_loss = kl_loss_fn(mu, log_sigma)

            # Contrastive loss (random Matryoshka dim)
            dim_idx = torch.randint(len(matryoshka_dims), (1,)).item()
            dim = matryoshka_dims[dim_idx]
            contr_loss = contrastive_loss_fn(z1[:, :dim], z2[:, :dim])

            # Total loss
            loss = recon_loss + kl_weight * kl_loss + contrastive_weight * contr_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_contr += contr_loss.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "recon": f"{recon_loss.item():.4f}",
            "kl": f"{kl_loss.item():.4f}",
            "contr": f"{contr_loss.item():.4f}",
        })

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "contrastive": total_contr / n_batches,
    }


@torch.no_grad()
def validate(
    encoder: nn.Module,
    decoder: nn.Module,
    val_loader,
    device: str,
    n_samples: int = 1000,
) -> dict:
    """Validate reconstruction quality."""
    encoder.eval()
    decoder.eval()

    total_cosine = 0.0
    total_mse = 0.0
    n = 0

    # Unwrap DDP modules for accessing custom methods
    enc = encoder.module if hasattr(encoder, "module") else encoder

    for x in val_loader:
        x = x.to(device)
        z = enc.encode_deterministic(x)
        x_recon = decoder(z)

        cos_sim = F.cosine_similarity(x, x_recon, dim=-1).sum()
        mse = F.mse_loss(x, x_recon, reduction="sum")

        total_cosine += cos_sim.item()
        total_mse += mse.item()
        n += x.shape[0]

        if n >= n_samples:
            break

    if n == 0:
        return {"val_cosine": 0.0, "val_mse": float("inf")}

    return {
        "val_cosine": total_cosine / n,
        "val_mse": total_mse / n / 768,  # Per dimension
    }


def main():
    parser = argparse.ArgumentParser(description="EcoFlow-BO Training (Perceiver)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr-mult", type=float, default=1.0,
                        help="LR multiplier for decoder (default: same as encoder)")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=1024,
                        help="Perceiver hidden size (can be large - only 16 tokens)")
    parser.add_argument("--depth", type=int, default=12,
                        help="Perceiver depth (number of self-attention layers)")
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
        print("EcoFlow-BO Training (Perceiver Decoder)")
        print("=" * 60)
        print(f"Device: {device}, Distributed: {distributed} (world_size={world_size})")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"Epochs: {args.epochs}")
        print(f"Latent dim: {args.latent_dim}")
        print(f"Perceiver: hidden={args.hidden_size}, depth={args.depth}")
        print(f"Loss weights: KL={args.kl_weight}, Contrastive={args.contrastive_weight}")
        print("=" * 60)

    # Config
    encoder_config = EncoderConfig(latent_dim=args.latent_dim)
    decoder_config = PerceiverDecoderConfig(
        latent_dim=args.latent_dim,
        output_dim=768,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=16,
    )

    # Models
    encoder = MatryoshkaEncoder(encoder_config).to(device)
    decoder = PerceiverDecoder(decoder_config).to(device)

    if is_main(rank):
        enc_p = sum(p.numel() for p in encoder.parameters())
        dec_p = sum(p.numel() for p in decoder.parameters())
        print(f"Encoder: {enc_p:,} params")
        print(f"Decoder: {dec_p:,} params")
        print(f"Total: {enc_p + dec_p:,} params")

    # DDP
    if distributed:
        encoder = DDP(encoder, device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        embeddings_path=args.embeddings_path,
        batch_size=args.batch_size,
        distributed=distributed,
    )

    # Optimizer with separate LR for encoder and decoder
    scale = min(args.batch_size / 256, 64.0)
    encoder_lr = args.lr * scale
    decoder_lr = encoder_lr * args.decoder_lr_mult

    param_groups = [
        {"params": list(encoder.parameters()), "lr": encoder_lr, "name": "encoder"},
        {"params": list(decoder.parameters()), "lr": decoder_lr, "name": "decoder"},
    ]

    if is_main(rank):
        print(f"LAMB optimizer:")
        print(f"  Encoder LR: {encoder_lr:.2e} (scaled {scale:.1f}x)")
        print(f"  Decoder LR: {decoder_lr:.2e} ({args.decoder_lr_mult:.1f}x encoder)")

    optimizer = optim_extra.Lamb(param_groups, weight_decay=0.01)

    # Scheduler
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - 10, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    # Loss functions
    matryoshka_dims = encoder_config.matryoshka_dims
    matryoshka_weights = encoder_config.matryoshka_weights
    recon_loss_fn = MatryoshkaReconstructionLoss(matryoshka_dims, matryoshka_weights)
    kl_loss_fn = KLDivergenceLoss()
    contrastive_loss_fn = InfoNCELoss(temperature=0.05)

    scaler = GradScaler("cuda", enabled=True)

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    best_cosine = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_cosine = ckpt.get("best_cosine", 0.0)
        if is_main(rank):
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, scaler, device,
            recon_loss_fn, kl_loss_fn, contrastive_loss_fn,
            args.kl_weight, args.contrastive_weight, matryoshka_dims,
        )

        scheduler.step()

        # Validate
        val_metrics = validate(encoder, decoder, val_loader, device)

        if is_main(rank):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                f"recon={train_metrics['recon']:.4f}, "
                f"kl={train_metrics['kl']:.4f}, "
                f"contr={train_metrics['contrastive']:.4f}, "
                f"val_cos={val_metrics['val_cosine']:.4f}, "
                f"lr={current_lr:.2e}"
            )

            # Save checkpoint
            if val_metrics["val_cosine"] > best_cosine:
                best_cosine = val_metrics["val_cosine"]
                enc_state = encoder.module.state_dict() if distributed else encoder.state_dict()
                dec_state = decoder.module.state_dict() if distributed else decoder.state_dict()
                torch.save({
                    "epoch": epoch,
                    "encoder": enc_state,
                    "decoder": dec_state,
                    "encoder_config": encoder_config,
                    "decoder_config": decoder_config,
                    "optimizer": optimizer.state_dict(),
                    "best_cosine": best_cosine,
                }, os.path.join(args.checkpoint_dir, "best_perceiver.pt"))
                print(f"  ★ New best: cosine={best_cosine:.4f}")

            # Periodic save
            if (epoch + 1) % 10 == 0:
                enc_state = encoder.module.state_dict() if distributed else encoder.state_dict()
                dec_state = decoder.module.state_dict() if distributed else decoder.state_dict()
                torch.save({
                    "epoch": epoch,
                    "encoder": enc_state,
                    "decoder": dec_state,
                    "encoder_config": encoder_config,
                    "decoder_config": decoder_config,
                    "optimizer": optimizer.state_dict(),
                    "best_cosine": best_cosine,
                }, os.path.join(args.checkpoint_dir, f"perceiver_epoch{epoch}.pt"))

    if is_main(rank):
        print(f"\nTraining complete! Best cosine similarity: {best_cosine:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
