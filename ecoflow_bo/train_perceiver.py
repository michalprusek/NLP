"""
EcoFlow-BO Training with Residual Latent Architecture.

Architecture:
- MatryoshkaEncoder: 768D → z_core (16D) + z_detail (32D) = 48D
- PerceiverDecoder: 48D → 768D (direct reconstruction)

Loss: ResidualMatryoshkaCFM (4D, 8D, 16D z_core + full 48D) + KL + Contrastive

Key Innovation: GP optimizes only z_core (16D), z_detail enhances reconstruction.
"""

import argparse
import os
import sys

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
from typing import Tuple

from ecoflow_bo.config import EcoFlowConfig, EncoderConfig
from ecoflow_bo.encoder import MatryoshkaEncoder
from ecoflow_bo.perceiver_decoder import PerceiverDecoder, PerceiverDecoderConfig
from ecoflow_bo.losses import KLDivergenceLoss, InfoNCELoss, ResidualKLLoss
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
    if torch.distributed.is_initialized():
        destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


class ResidualMatryoshkaLoss(nn.Module):
    """
    Residual Matryoshka reconstruction loss for z_full = [z_core, z_detail].

    Training strategy (from NeurIPS feedback):
    - loss_4:  z[:4] active, rest=0   → Coarse semantics (weight=1.0)
    - loss_8:  z[:8] active, rest=0   → Medium detail (weight=0.5)
    - loss_16: z[:16] active, rest=0  → Fine z_core (weight=0.25)
    - loss_48: z[:48] active          → Full reconstruction (weight=0.1)

    High weight on early dims forces encoder to pack meaning into first dims.
    """

    def __init__(
        self,
        core_dim: int = 16,
        detail_dim: int = 32,
        matryoshka_dims: list = None,
        matryoshka_weights: list = None,
        full_weight: float = 0.1,
    ):
        super().__init__()
        if matryoshka_dims is None:
            matryoshka_dims = [4, 8, 16]
        if matryoshka_weights is None:
            matryoshka_weights = [1.0, 0.5, 0.25]

        self.core_dim = core_dim
        self.detail_dim = detail_dim
        self.full_dim = core_dim + detail_dim
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights
        self.full_weight = full_weight

    def forward(
        self,
        decoder: PerceiverDecoder,
        z_full: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Residual Matryoshka reconstruction loss.

        Args:
            decoder: PerceiverDecoder
            z_full: Full latent [B, 48] = [z_core (16D), z_detail (32D)]
            x_target: Target embeddings [B, 768]

        Returns:
            loss: Weighted reconstruction loss
            details: Dict with loss at each level
        """
        total_loss = 0.0
        details = {}
        total_weight = sum(self.matryoshka_weights) + self.full_weight

        # 1. Matryoshka levels on z_core (z_detail = 0)
        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            z_masked = torch.zeros_like(z_full)
            z_masked[:, :dim] = z_full[:, :dim]

            x_recon = decoder(z_masked)
            mse = F.mse_loss(x_recon, x_target)
            cos_sim = F.cosine_similarity(x_recon, x_target, dim=-1).mean()

            total_loss = total_loss + weight * mse
            details[f"mse_core_{dim}"] = mse.item()
            details[f"cos_core_{dim}"] = cos_sim.item()

        # 2. Full reconstruction (z_core + z_detail)
        x_recon_full = decoder(z_full)
        mse_full = F.mse_loss(x_recon_full, x_target)
        cos_full = F.cosine_similarity(x_recon_full, x_target, dim=-1).mean()

        total_loss = total_loss + self.full_weight * mse_full
        details["mse_full_48"] = mse_full.item()
        details["cos_full_48"] = cos_full.item()

        # Normalize
        total_loss = total_loss / total_weight
        details["recon_total"] = total_loss.item()

        return total_loss, details


def train_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader,
    optimizer,
    scaler: GradScaler,
    device: str,
    recon_loss_fn: ResidualMatryoshkaLoss,
    kl_loss_fn: ResidualKLLoss,
    contrastive_loss_fn: InfoNCELoss,
    kl_weight: float,
    contrastive_weight: float,
    core_dim: int,
) -> dict:
    """Train for one epoch with residual latent."""
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_contr = 0.0
    n_batches = 0

    # Unwrap DDP for custom methods
    enc = encoder.module if hasattr(encoder, "module") else encoder

    pbar = tqdm(train_loader, desc="Training")
    for x in pbar:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.bfloat16):
            # Encoder forward - get full residual latent
            out1 = enc.forward_full(x)
            out2 = enc.forward_full(x)  # Different dropout for contrastive

            z_full_1 = out1.z_full  # [B, 48]
            z_full_2 = out2.z_full

            # Reconstruction loss (Residual Matryoshka)
            recon_loss, recon_details = recon_loss_fn(decoder, z_full_1, x)

            # KL loss on both z_core and z_detail
            kl_loss, kl_details = kl_loss_fn(
                out1.mu_core, out1.log_sigma_core,
                out1.mu_detail, out1.log_sigma_detail,
            )

            # Contrastive loss on z_core (what GP will optimize)
            contr_loss = contrastive_loss_fn(out1.z_core, out2.z_core)

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
    """Validate reconstruction quality with residual latent."""
    encoder.eval()
    decoder.eval()

    total_cosine_full = 0.0
    total_cosine_core = 0.0
    total_mse = 0.0
    n = 0

    enc = encoder.module if hasattr(encoder, "module") else encoder

    for x in val_loader:
        x = x.to(device)

        # Get full latent
        z_core, z_detail = enc.encode_deterministic_full(x)
        z_full = torch.cat([z_core, z_detail], dim=-1)

        # Full reconstruction
        x_recon_full = decoder(z_full)

        # Core-only reconstruction (z_detail = 0)
        z_core_only = torch.zeros_like(z_full)
        z_core_only[:, :z_core.shape[-1]] = z_core
        x_recon_core = decoder(z_core_only)

        cos_full = F.cosine_similarity(x, x_recon_full, dim=-1).sum()
        cos_core = F.cosine_similarity(x, x_recon_core, dim=-1).sum()
        mse = F.mse_loss(x, x_recon_full, reduction="sum")

        total_cosine_full += cos_full.item()
        total_cosine_core += cos_core.item()
        total_mse += mse.item()
        n += x.shape[0]

        if n >= n_samples:
            break

    if n == 0:
        return {"val_cos_full": 0.0, "val_cos_core": 0.0, "val_mse": float("inf")}

    return {
        "val_cos_full": total_cosine_full / n,
        "val_cos_core": total_cosine_core / n,
        "val_mse": total_mse / n / 768,
    }


def main():
    parser = argparse.ArgumentParser(description="EcoFlow-BO Residual Latent Training")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--core-dim", type=int, default=16)
    parser.add_argument("--detail-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--kl-weight", type=float, default=0.001)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", type=str, default="results/ecoflow_checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--embeddings-path", type=str, default="datasets/gtr_embeddings_full.pt")
    args = parser.parse_args()

    full_dim = args.core_dim + args.detail_dim

    # Distributed setup
    rank, world_size, local_rank, distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if is_main(rank):
        print("=" * 60)
        print("EcoFlow-BO: Residual Latent Training")
        print("=" * 60)
        print(f"Device: {device}, Distributed: {distributed} (world_size={world_size})")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"Latent: z_core={args.core_dim}D + z_detail={args.detail_dim}D = {full_dim}D")
        print(f"Perceiver: hidden={args.hidden_size}, depth={args.depth}")
        print(f"Loss weights: KL={args.kl_weight}, Contrastive={args.contrastive_weight}")
        print("=" * 60)

    # Config
    encoder_config = EncoderConfig(
        latent_dim=args.core_dim,
        detail_dim=args.detail_dim,
    )
    decoder_config = PerceiverDecoderConfig(
        latent_dim=full_dim,  # 48D input
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

    if distributed:
        encoder = DDP(encoder, device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        embeddings_path=args.embeddings_path,
        batch_size=args.batch_size,
        distributed=distributed,
    )

    # Optimizer
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
        print(f"  Decoder LR: {decoder_lr:.2e}")

    optimizer = optim_extra.Lamb(param_groups, weight_decay=0.01)

    # Scheduler
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - 10, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    # Loss functions
    recon_loss_fn = ResidualMatryoshkaLoss(
        core_dim=args.core_dim,
        detail_dim=args.detail_dim,
    )
    kl_loss_fn = ResidualKLLoss(core_weight=1.0, detail_weight=0.5)
    contrastive_loss_fn = InfoNCELoss(temperature=0.05)

    scaler = GradScaler("cuda", enabled=True)

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

        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, scaler, device,
            recon_loss_fn, kl_loss_fn, contrastive_loss_fn,
            args.kl_weight, args.contrastive_weight, args.core_dim,
        )

        scheduler.step()

        val_metrics = validate(encoder, decoder, val_loader, device)

        if is_main(rank):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                f"recon={train_metrics['recon']:.4f}, "
                f"kl={train_metrics['kl']:.4f}, "
                f"val_cos_full={val_metrics['val_cos_full']:.4f}, "
                f"val_cos_core={val_metrics['val_cos_core']:.4f}, "
                f"lr={current_lr:.2e}"
            )

            # Save best
            if val_metrics["val_cos_full"] > best_cosine:
                best_cosine = val_metrics["val_cos_full"]
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
                }, os.path.join(args.checkpoint_dir, "best_residual.pt"))
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
                }, os.path.join(args.checkpoint_dir, f"residual_epoch{epoch}.pt"))

    if is_main(rank):
        print(f"\nTraining complete! Best cosine: {best_cosine:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
