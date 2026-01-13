"""
Latent Projector Training with Distributed Data Parallel (DDP).

This script trains the Latent Projector (768D -> 4×4096D prefix tokens)
for GritLM text generation using DDP across multiple GPUs.

The projector learns to map embeddings to prefix tokens that enable
GritLM to reconstruct the original text via cross-entropy loss.

Usage:
    torchrun --nproc_per_node=2 --master_port=12355 \
        -m lido_pp.training.train_projector_ddp \
        --data alpaca \
        --epochs 50 --batch-size 4 --gradient-accumulation 4

Architecture:
    MLP: 768D -> 3072D (GELU) -> 3072D (GELU) -> 16384D (4×4096D)
    ~62M parameters

Loss: Cross-entropy for next-token prediction (GritLM frozen)
"""

import argparse
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lido_pp.training.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    save_checkpoint_ddp,
    reduce_tensor,
    print_rank0,
    barrier,
)
from lido_pp.training.alpaca_dataset import load_alpaca_dataset, AlpacaInstructionDataset


class ProjectorDataset(Dataset):
    """Dataset for projector training with texts and embeddings."""

    def __init__(self, texts: List[str], embeddings: torch.Tensor):
        self.texts = texts
        self.embeddings = embeddings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "embedding": self.embeddings[idx],
        }


def collate_fn(batch):
    """Collate batch of samples."""
    texts = [b["text"] for b in batch]
    embeddings = torch.stack([b["embedding"] for b in batch])
    return {"texts": texts, "embeddings": embeddings}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Projector with DDP")
    parser.add_argument(
        "--data",
        type=str,
        default="alpaca",
        help="Dataset name or path to embeddings file",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="lido_pp/data/alpaca_embeddings.pt",
        help="Path to pre-computed embeddings",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-prefix-tokens", type=int, default=4, help="Number of prefix tokens")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--gritlm-model", type=str, default="GritLM/GritLM-7B")
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def compute_loss(
    projector: nn.Module,
    model,  # GritLM
    tokenizer,
    texts: List[str],
    embeddings: torch.Tensor,
    device: torch.device,
    max_length: int = 128,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute cross-entropy loss for text reconstruction.

    Args:
        projector: LatentProjector module
        model: GritLM model (frozen)
        tokenizer: GritLM tokenizer
        texts: Target texts
        embeddings: Input embeddings (B, 768)
        device: Device
        max_length: Maximum sequence length

    Returns:
        loss: Cross-entropy loss
        metrics: Dict with loss components
    """
    batch_size = len(texts)

    # Project to prefix embeddings
    # projector is in float32 for stable gradients
    embeddings_f32 = embeddings.to(dtype=torch.float32)
    prefix_embeds = projector(embeddings_f32)  # (B, num_prefix, hidden_dim)

    # Cast to model dtype
    prefix_embeds = prefix_embeds.to(model.dtype)

    # Tokenize target texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Get token embeddings
    embed_tokens = model.get_input_embeddings()
    target_embeds = embed_tokens(input_ids)  # (B, L, hidden_dim)

    # Concatenate: [prefix | target_tokens]
    inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

    # Extend attention mask for prefix
    prefix_mask = torch.ones(
        batch_size, prefix_embeds.shape[1],
        dtype=attention_mask.dtype,
        device=device,
    )
    extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    # Create labels: -100 for prefix (no loss), then input_ids shifted
    # We want to predict the next token, so labels are shifted
    prefix_labels = torch.full(
        (batch_size, prefix_embeds.shape[1]),
        -100,  # Ignore prefix in loss
        dtype=torch.long,
        device=device,
    )
    labels = torch.cat([prefix_labels, input_ids], dim=1)

    # Shift labels by 1 for causal LM (predict next token)
    # labels[i] should be the target for position i (which is inputs_embeds[i+1])
    shifted_labels = labels[..., 1:].contiguous()

    # Forward through GritLM with autocast for memory efficiency
    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = model(
            inputs_embeds=inputs_embeds[:, :-1],  # Remove last token (no next target)
            attention_mask=extended_attention_mask[:, :-1],
            use_cache=False,
        )

    logits = outputs.logits  # (B, seq_len, vocab_size)

    # Compute cross-entropy loss
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        shifted_labels.view(-1),
        ignore_index=-100,
    )

    # Compute metrics
    with torch.no_grad():
        # Perplexity (only for non-masked positions)
        valid_mask = shifted_labels.view(-1) != -100
        if valid_mask.sum() > 0:
            valid_loss = F.cross_entropy(
                logits.view(-1, vocab_size)[valid_mask],
                shifted_labels.view(-1)[valid_mask],
            )
            perplexity = torch.exp(valid_loss).item()
        else:
            perplexity = float("inf")

    metrics = {
        "loss": loss.item(),
        "perplexity": perplexity,
    }

    return loss, metrics


def train_epoch(
    projector: DDP,
    model,
    tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation."""
    projector.train()
    model.eval()  # GritLM stays in eval mode (frozen)

    total_loss = 0.0
    total_ppl = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        texts = batch["texts"]
        embeddings = batch["embeddings"].to(device)

        # Compute loss
        loss, metrics = compute_loss(
            projector.module if hasattr(projector, 'module') else projector,
            model,
            tokenizer,
            texts,
            embeddings,
            device,
            args.max_length,
        )

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation

        # Backward with gradient scaler
        scaler.scale(loss).backward()

        # Accumulate metrics
        total_loss += metrics["loss"]
        total_ppl += metrics["perplexity"]
        n_batches += 1

        # Gradient step after accumulation
        if (step + 1) % args.gradient_accumulation == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return {
        "loss": total_loss / max(n_batches, 1),
        "perplexity": total_ppl / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    projector: DDP,
    model,
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Validate model."""
    projector.eval()

    total_loss = 0.0
    total_ppl = 0.0
    n_batches = 0

    for batch in dataloader:
        texts = batch["texts"]
        embeddings = batch["embeddings"].to(device)

        loss, metrics = compute_loss(
            projector.module if hasattr(projector, 'module') else projector,
            model,
            tokenizer,
            texts,
            embeddings,
            device,
            args.max_length,
        )

        total_loss += metrics["loss"]
        total_ppl += metrics["perplexity"]
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "perplexity": total_ppl / max(n_batches, 1),
    }


def train_worker(rank: int, world_size: int, args):
    """Main training worker for each GPU."""
    # Setup DDP (only for multi-GPU)
    if world_size > 1:
        setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(args.seed + rank)

    # Load GritLM (one instance per GPU)
    print_rank0(f"Loading GritLM model: {args.gritlm_model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.gritlm_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.gritlm_model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    # Freeze GritLM
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = model.config.hidden_size
    print_rank0(f"GritLM hidden dim: {hidden_dim}")

    barrier()

    # Load data
    print_rank0(f"Loading data from {args.embeddings_path}...")
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)

    embeddings = data["embeddings"]
    texts = data.get("instructions", [])

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    if not texts:
        # Load from dataset if not in embeddings file
        texts = load_alpaca_dataset(max_samples=len(embeddings))

    print_rank0(f"Loaded {len(texts)} samples")

    # Train/val split
    n_val = int(len(embeddings) * args.val_ratio)
    indices = torch.randperm(len(embeddings))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = ProjectorDataset(
        [texts[i] for i in train_indices],
        embeddings[train_indices],
    )
    val_dataset = ProjectorDataset(
        [texts[i] for i in val_indices],
        embeddings[val_indices],
    )

    print_rank0(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create distributed samplers and loaders
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create projector
    from lido_pp.backbone.latent_injection import LatentProjector

    projector = LatentProjector(
        latent_dim=768,
        hidden_dim=hidden_dim,
        num_prefix_tokens=args.num_prefix_tokens,
        use_layer_norm=True,
        dropout=0.1,
    ).to(device)

    # Wrap in DDP
    # Wrap in DDP (only for multi-GPU)
    if world_size > 1:
        projector = DDP(projector, device_ids=[rank], find_unused_parameters=False)

    # Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_val_loss = float("inf")
    start_time = datetime.now()

    print_rank0(f"\nStarting Projector training:")
    print_rank0(f"  Epochs: {args.epochs}")
    print_rank0(f"  Batch size per GPU: {args.batch_size}")
    print_rank0(f"  Gradient accumulation: {args.gradient_accumulation}")
    print_rank0(f"  Effective batch size: {args.batch_size * args.gradient_accumulation * world_size}")
    print_rank0(f"  Learning rate: {args.lr}")
    print_rank0(f"  Prefix tokens: {args.num_prefix_tokens}")
    print_rank0(f"  World size: {world_size}")

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            projector, model, tokenizer, train_loader, optimizer, scaler,
            device, args, epoch
        )

        scheduler.step()

        # Validate
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            val_metrics = validate(projector, model, tokenizer, val_loader, device, args)

            # Reduce metrics
            train_loss = reduce_tensor(
                torch.tensor(train_metrics["loss"], device=device), world_size
            ).item()
            val_loss = reduce_tensor(
                torch.tensor(val_metrics["loss"], device=device), world_size
            ).item()
            val_ppl = reduce_tensor(
                torch.tensor(val_metrics["perplexity"], device=device), world_size
            ).item()

            if rank == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val PPL: {val_ppl:.2f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"Time: {elapsed:.1f}m"
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint_ddp(
                    {
                        "epoch": epoch,
                        "model_state_dict": projector.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "args": vars(args),
                    },
                    str(checkpoint_dir / "projector_best.pt"),
                    rank,
                )

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint_ddp(
                {
                    "epoch": epoch,
                    "model_state_dict": projector.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                str(checkpoint_dir / f"projector_epoch{epoch+1:04d}.pt"),
                rank,
            )

    # Final save
    save_checkpoint_ddp(
        {
            "epoch": args.epochs,
            "model_state_dict": projector.module.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        str(checkpoint_dir / "projector_alpaca_final.pt"),
        rank,
    )

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print_rank0(f"\nTraining complete! Total time: {total_time:.1f} minutes")
    print_rank0(f"Best validation loss: {best_val_loss:.4f}")

    cleanup_ddp()


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        train_worker(local_rank, world_size, args)
    else:
        print("Running single GPU training...")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
