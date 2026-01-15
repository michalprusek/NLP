"""
Text Flow Autoencoder (TFA) Training for FlowPO.

This script trains the TFA to map SONAR embeddings (1024D) to compact latent
space (128D) via simulation-free flow matching with Lipschitz regularization.

The TFA is a core component of FlowPO (Novel Contribution #1).

Usage:
    # Single GPU
    uv run python -m lido_pp.training.train_cfm \
        --data lido_pp/data/sonar_embeddings.pt \
        --epochs 10000 --batch-size 512

    # Multi-GPU (DDP)
    uv run torchrun --nproc_per_node=2 -m lido_pp.training.train_cfm \
        --data lido_pp/data/sonar_embeddings.pt \
        --epochs 10000 --batch-size 512

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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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


def setup_ddp():
    """Initialize DDP if running with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


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
    parser.add_argument("--flow-dim", type=int, default=512, help="Flow space dimension (increased for capacity)")
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=20,
        help="ODE integration steps for inference (ALIGNED with train for stability)",
    )
    parser.add_argument(
        "--train-ode-steps",
        type=int,
        default=20,
        help="ODE integration steps for training loss (ALIGNED with inference)",
    )
    parser.add_argument(
        "--velocity-layers",
        type=int,
        default=6,
        help="Number of layers in velocity network (deeper = more capacity)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for velocity field (0.1 recommended for large datasets >500k)",
    )
    parser.add_argument(
        "--augment-noise",
        type=float,
        default=0.02,
        help="Gaussian noise std for embedding augmentation (0 to disable)",
    )
    parser.add_argument("--lambda-gw", type=float, default=0.0, help="GW loss weight (optional)")
    parser.add_argument("--lambda-recon", type=float, default=0.5, help="Reconstruction loss weight (increased for stability)")
    parser.add_argument(
        "--lambda-lip",
        type=float,
        default=0.1,
        help="Lipschitz regularization weight (10x increase for stability)",
    )
    parser.add_argument(
        "--lambda-consistency",
        type=float,
        default=0.1,
        help="Forward-backward consistency loss weight (new for RegFlow-style stability)",
    )
    parser.add_argument(
        "--lipschitz-bound",
        type=float,
        default=5.0,
        help="Maximum Lipschitz constant (tighter for stability)",
    )
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        choices=["uniform", "u_shaped"],
        default="u_shaped",
        help="Timestep sampling strategy (u_shaped improves convergence by ~28%%)",
    )
    parser.add_argument(
        "--use-ot",
        action="store_true",
        default=True,
        help="Use Optimal Transport pairing (OT-CFM) for straighter trajectories (default: True)",
    )
    parser.add_argument(
        "--no-ot",
        action="store_true",
        help="Disable Optimal Transport pairing (use standard CFM)",
    )
    parser.add_argument(
        "--lip-penalty-type",
        type=str,
        choices=["hinge", "soft", "quadratic"],
        default="soft",
        help="Lipschitz penalty type: hinge (only penalize above bound), soft (smooth, always active), quadratic (encourage low ratios)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=500, help="LR warmup epochs")
    parser.add_argument("--patience", type=int, default=1000, help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval (1 = every epoch)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader num_workers (default: 8 per CLAUDE.md)",
    )
    return parser.parse_args()


def train_epoch(
    model: TextFlowAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Train for one epoch (OT-CFM with Lipschitz regularization)."""
    model.train()
    total_loss = 0.0
    total_fm = 0.0
    total_recon = 0.0
    total_gw = 0.0
    total_lip = 0.0
    total_lip_ratio = 0.0
    total_consistency = 0.0
    n_batches = 0

    # Resolve use_ot from args (--no-ot overrides --use-ot)
    use_ot = args.use_ot and not getattr(args, 'no_ot', False)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0 and n_batches == 0:
            print(f"[DEBUG] First batch starting, shape: {batch[0].shape}")
            print(f"[DEBUG] OT-CFM enabled: {use_ot}")
            sys.stdout.flush()

        embeddings = batch[0].to(device)

        # Data augmentation: add small Gaussian noise to embeddings
        # This improves generalization and prevents overfitting
        # NOTE: Do NOT normalize - SONAR decoder needs unnormalized embeddings
        if args.augment_noise > 0:
            noise = torch.randn_like(embeddings) * args.augment_noise
            embeddings = embeddings + noise

        optimizer.zero_grad()

        if batch_idx == 0 and n_batches == 0:
            print(f"[DEBUG] Calling flow_matching_loss...")
            sys.stdout.flush()

        # Forward pass and compute loss (OT-CFM for straighter trajectories)
        losses = flow_matching_loss(
            model,
            embeddings,
            lambda_recon=args.lambda_recon,
            lambda_gw=args.lambda_gw,
            lambda_lip=args.lambda_lip,
            lambda_consistency=args.lambda_consistency,
            timestep_sampling=args.timestep_sampling,
            lip_bound=args.lipschitz_bound,
            lip_penalty_type=args.lip_penalty_type,
            use_ot=use_ot,
        )

        if batch_idx == 0 and n_batches == 0:
            print(f"[DEBUG] Loss computed: {losses['loss'].item():.4f}")
            sys.stdout.flush()

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
        total_lip_ratio += losses.get("lip_ratio", 0.0)
        total_consistency += losses.get("consistency", 0.0)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "fm": total_fm / n_batches,
        "recon": total_recon / n_batches,
        "gw": total_gw / n_batches,
        "lip": total_lip / n_batches,
        "lip_ratio": total_lip_ratio / n_batches,  # NEW: actual Lipschitz ratio
        "consistency": total_consistency / n_batches,
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

    # Resolve use_ot from args
    use_ot = args.use_ot and not getattr(args, 'no_ot', False)

    for i, batch in enumerate(dataloader):
        embeddings = batch[0].to(device)

        # Simulation-free loss (same as training, but no Lipschitz)
        losses = flow_matching_loss(
            model,
            embeddings,
            lambda_recon=args.lambda_recon,
            lambda_gw=args.lambda_gw,
            lambda_lip=0.0,  # Don't compute Lipschitz during validation
            use_ot=use_ot,
        )

        total_loss += losses["loss"].item()
        total_fm += losses["fm"]
        total_recon += losses["recon"]
        total_gw += losses.get("gw", 0.0)
        n_batches += 1

        # Every 5 batches, check actual ODE reconstruction with more samples
        # This gives better coverage (~4% of val set) and lower variance
        if i % 5 == 0:
            # Use unwrapped model for forward pass (DDP compatibility)
            unwrapped = model.module if hasattr(model, 'module') else model
            sample_size = min(32, embeddings.shape[0])  # Use up to 32 samples
            z, x_recon = unwrapped(embeddings[:sample_size])  # Uses ODE solver
            cos_sim = F.cosine_similarity(embeddings[:sample_size], x_recon, dim=-1).mean()
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

    # Setup DDP if running with torchrun
    rank, local_rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    # Set seed
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    # Set device based on DDP or single GPU
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Device: {device}" + (f" (DDP: {world_size} GPUs)" if use_ddp else ""))

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data with robust error handling
    if is_main_process():
        print(f"Loading embeddings from {args.data}...")

    try:
        data = torch.load(args.data, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            "Generate SONAR embeddings first using:\n"
            "  python -m lido_pp.training.precompute_embeddings --encoder sonar"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load data from {args.data}: {e}\n"
            "Ensure the file is a valid PyTorch tensor or dict with 'embeddings' key."
        )

    if isinstance(data, dict):
        if "embeddings" not in data:
            raise KeyError(
                f"Data file {args.data} is missing 'embeddings' key.\n"
                f"Available keys: {list(data.keys())}"
            )
        embeddings = data["embeddings"]
        metadata = data.get("metadata", {})
        if is_main_process():
            print(f"Metadata: {metadata}")
    else:
        embeddings = data
        metadata = {}

    if is_main_process():
        print(f"Embeddings shape: {embeddings.shape}")
    input_dim = embeddings.shape[1]

    # Validate input dimensions
    if is_main_process():
        if input_dim == 1024:
            print("✓ SONAR embeddings detected (1024D)")
        elif input_dim == 4096:
            print("WARNING: GritLM embeddings detected (4096D)")
            print("For FlowPO, use SONAR embeddings with --encoder sonar")
        elif input_dim == 768:
            print("WARNING: GTR embeddings detected (768D)")
            print("For FlowPO, use SONAR embeddings with --encoder sonar")
        else:
            print(f"WARNING: Unexpected embedding dimension {input_dim}D")

    # Split data
    dataset = TensorDataset(embeddings)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    if is_main_process():
        print(f"Train: {train_size}, Val: {val_size}")

    # Create dataloaders with DDP sampler if needed
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model (Text Flow Autoencoder) with increased capacity
    model = TextFlowAutoencoder(
        input_dim=input_dim,
        flow_dim=args.flow_dim,
        latent_dim=args.latent_dim,
        num_ode_steps=args.ode_steps,          # Inference: 50 steps for quality
        num_train_ode_steps=args.train_ode_steps,  # Training: 10 steps for speed
        num_velocity_layers=args.velocity_layers,
        dropout=args.dropout,  # Regularization for large datasets
    ).to(device)

    # Wrap in DDP if multi-GPU
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    num_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Compression ratio
    compression_ratio = input_dim / args.latent_dim
    if is_main_process():
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

    # Resolve use_ot from args
    use_ot = args.use_ot and not getattr(args, 'no_ot', False)

    if is_main_process():
        print("\n" + "=" * 70)
        print("FlowPO: Text Flow Autoencoder (TFA) Training")
        print("Novel Contribution #1: Simulation-free FM for text autoencoding")
        print("=" * 70)
        print(f"Input dim: {input_dim}D (SONAR)")
        print(f"Flow dim: {args.flow_dim}D")
        print(f"Latent dim: {args.latent_dim}D")
        print(f"Compression: {compression_ratio:.1f}:1")
        print(f"ODE steps (train): {args.train_ode_steps} (fast)")
        print(f"ODE steps (inference): {args.ode_steps} (quality)")
        print("-" * 70)
        print(f"OT-CFM: {'ENABLED (straighter trajectories)' if use_ot else 'DISABLED'}")
        print(f"Timestep sampling: {args.timestep_sampling}")
        print(f"Augment noise: {args.augment_noise}")
        print("-" * 70)
        print(f"Lambda recon: {args.lambda_recon}")
        print(f"Lambda GW: {args.lambda_gw}")
        print(f"Lambda Lipschitz: {args.lambda_lip} (penalty={args.lip_penalty_type})")
        print(f"Lambda Consistency: {args.lambda_consistency}")
        print(f"Lipschitz bound: {args.lipschitz_bound}")
        print("-" * 70)
        print("Training: OT-CFM + ODE reconstruction loss")
        print("Inference: Euler integration for encode/decode")
        print("=" * 70 + "\n")

    if is_main_process():
        print(f"\n[DEBUG] Starting training loop...")
        print(f"[DEBUG] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"[DEBUG] Batch size: {args.batch_size}, Epochs: {args.epochs}")
        sys.stdout.flush()

    for epoch in range(1, args.epochs + 1):
        if is_main_process() and epoch == 1:
            print(f"[DEBUG] Epoch {epoch} starting...")
            sys.stdout.flush()

        # Set epoch for distributed sampler
        if use_ddp:
            train_sampler.set_epoch(epoch)

        # Warmup LR
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor

        # Train with error context
        try:
            if is_main_process() and epoch == 1:
                print(f"[DEBUG] Calling train_epoch()...")
                sys.stdout.flush()
            train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        except (ValueError, RuntimeError) as e:
            if is_main_process():
                print(f"\n[ERROR] Training failed at epoch {epoch}: {e}")
                print("Saving emergency checkpoint...")
                model_to_save = model.module if use_ddp else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "error": str(e),
                }, checkpoint_dir / "tfa_emergency.pt")
            raise

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

            # Save best model (only on main process)
            if is_main_process():
                model_to_save = model.module if use_ddp else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
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

        # Logging (only on main process)
        if is_main_process() and (epoch % args.log_interval == 0 or epoch == 1):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = datetime.now() - start_time
            lip_ratio = train_metrics.get('lip_ratio', 0.0)
            print(
                f"Epoch {epoch:5d} | "
                f"FM: {train_metrics['fm']:.4f} | "
                f"Recon: {train_metrics['recon']:.4f} | "
                f"Lip: {train_metrics['lip']:.4f} (R={lip_ratio:.1f}) | "
                f"Cons: {train_metrics.get('consistency', 0.0):.4f} | "
                f"Val CosODE: {val_metrics['cos_ode']:.4f} | "
                f"Best: {best_val_cos:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed}"
            )

        # Periodic checkpoint (only on main process)
        if is_main_process() and epoch % args.checkpoint_interval == 0:
            model_to_save = model.module if use_ddp else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_cos_ode": val_metrics["cos_ode"],
                "args": vars(args),
            }, checkpoint_dir / f"tfa_epoch_{epoch}.pt")

        # Early stopping
        if epochs_without_improvement >= args.patience:
            if is_main_process():
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Final summary (only on main process)
    if is_main_process():
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
        test_model = TextFlowAutoencoder(
            input_dim=input_dim,
            flow_dim=args.flow_dim,
            latent_dim=args.latent_dim,
            num_ode_steps=args.ode_steps,          # Use full inference steps for test
            num_train_ode_steps=args.train_ode_steps,
            num_velocity_layers=args.velocity_layers,
            dropout=args.dropout,
        ).to(device)
        test_model.load_state_dict(checkpoint["model_state_dict"])
        test_model.eval()

        with torch.no_grad():
            test_batch = next(iter(val_loader))[0][:8].to(device)
            z, x_recon = test_model(test_batch)
            cos_sim = F.cosine_similarity(test_batch, x_recon, dim=-1)
            print(f"Test cosine similarities: {cos_sim.tolist()}")
            print(f"Mean: {cos_sim.mean():.4f}, Min: {cos_sim.min():.4f}, Max: {cos_sim.max():.4f}")

            # Also check latent statistics
            print(f"\nLatent statistics:")
            print(f"  Shape: {z.shape}")
            print(f"  Mean: {z.mean():.4f}, Std: {z.std():.4f}")
            print(f"  Min: {z.min():.4f}, Max: {z.max():.4f}")

    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
