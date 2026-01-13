"""
Phase A: Translator Training (VAE + Projector jointly).

This script trains VAE and Projector together for perfect Text ↔ Latent mapping.

Pipeline:
    Text → GritLM(embed) → VAE_enc → latent → VAE_dec → Projector → GritLM(gen) → Text'

Loss: Text reconstruction (cross-entropy) + VAE KL regularization

NO FlowDiT - that's Phase B.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m lido_pp.training.train_translator \
        --epochs 50 --batch-size 4 --lr 1e-4

This is analogous to how Stable Diffusion trains VAE separately from U-Net.
"""

import argparse
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lido_pp.vae import InstructionVAE
from lido_pp.backbone.latent_injection import LatentProjector
from lido_pp.training.alpaca_dataset import load_alpaca_dataset, load_combined_dataset


class TextDataset(Dataset):
    """Simple dataset of texts."""
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class TextEmbeddingDataset(Dataset):
    """Dataset with pre-computed embeddings."""
    def __init__(self, texts: List[str], embeddings: torch.Tensor):
        assert len(texts) == len(embeddings), f"Mismatch: {len(texts)} texts vs {len(embeddings)} embeddings"
        self.texts = texts
        self.embeddings = embeddings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.embeddings[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Translator (VAE + Projector)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vae-lr", type=float, default=1e-4, help="VAE learning rate")
    parser.add_argument("--proj-lr", type=float, default=1e-4, help="Projector learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-prefix-tokens", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vae-beta", type=float, default=0.001, help="VAE KL weight (lower for reconstruction)")
    parser.add_argument("--gritlm-model", type=str, default="GritLM/GritLM-7B")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit Alpaca dataset size")
    parser.add_argument("--ultrachat-samples", type=int, default=None, help="Add UltraChat samples (None=skip, 0=all)")
    parser.add_argument("--precomputed-embeddings", type=str, default=None, help="Path to precomputed embeddings .pt file")
    parser.add_argument("--checkpoint-dir", type=str, default="lido_pp/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    # Resume from checkpoint
    parser.add_argument("--resume-vae", type=str, default=None, help="Resume VAE from checkpoint")
    parser.add_argument("--resume-proj", type=str, default=None, help="Resume Projector from checkpoint")
    return parser.parse_args()


class TranslatorTrainer:
    """
    Trainer for VAE + Projector (Phase A: Translator).

    The goal is perfect reconstruction:
    Text → encode → latent → decode → project → generate → Text'

    We want Text ≈ Text' so that when FlowDiT manipulates latents,
    the Projector can faithfully reconstruct the intended text.
    """

    def __init__(
        self,
        vae: InstructionVAE,
        projector: LatentProjector,
        gritlm_model,
        tokenizer,
        encoder,  # GritLM encoder for embeddings
        args,
        device: str = "cuda:0",
    ):
        self.vae = vae
        self.projector = projector
        self.model = gritlm_model  # For generation
        self.tokenizer = tokenizer
        self.encoder = encoder  # For encoding text to embeddings
        self.args = args
        self.device = device

        # Freeze GritLM
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding layer
        self.embed_tokens = self.model.get_input_embeddings()
        self.hidden_dim = self.model.config.hidden_size

        # Optimizers - separate for VAE and Projector
        self.vae_optimizer = torch.optim.AdamW(
            vae.parameters(), lr=args.vae_lr, weight_decay=args.weight_decay
        )
        self.proj_optimizer = torch.optim.AdamW(
            projector.parameters(), lr=args.proj_lr, weight_decay=args.weight_decay
        )

        # Schedulers
        self.vae_scheduler = CosineAnnealingLR(self.vae_optimizer, T_max=args.epochs, eta_min=1e-6)
        self.proj_scheduler = CosineAnnealingLR(self.proj_optimizer, T_max=args.epochs, eta_min=1e-6)

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings using GritLM encoder."""
        with torch.no_grad():
            # Use GritLM in embedding mode
            embeddings = self.encoder.encode_batch(texts, batch_size=len(texts), show_progress=False)
            return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

    def compute_loss(self, texts: List[str], precomputed_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss through VAE + Projector.

        Pipeline:
        1. Text → GritLM encoder → embedding (768D) [or use precomputed]
        2. Embedding → VAE encoder → latent (32D)
        3. Latent → VAE decoder → reconstructed embedding (768D)
        4. Reconstructed embedding → Projector → prefix tokens (4 x 4096D)
        5. Prefix tokens + GritLM → generate → compare with original text
        """
        batch_size = len(texts)

        # Step 1: Use precomputed embeddings or encode on-the-fly
        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings.to(self.device)
        else:
            embeddings = self.encode_texts(texts)  # (B, 768)

        # Step 2-3: VAE forward (encode + decode)
        x_recon, mu, log_var, z = self.vae(embeddings)  # All (B, *)

        # VAE losses
        vae_recon_loss = 1 - F.cosine_similarity(embeddings, x_recon, dim=-1).mean()
        vae_kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        # Step 4: Project reconstructed embedding to prefix tokens
        # Projector expects (B, 768), outputs (B, num_prefix, hidden_dim)
        prefix_embeds = self.projector(x_recon)  # (B, 4, 4096)
        prefix_embeds = prefix_embeds.to(self.model.dtype)

        # Step 5: Generate and compute cross-entropy loss
        # Tokenize target texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get token embeddings
        target_embeds = self.embed_tokens(input_ids)  # (B, L, hidden_dim)

        # Concatenate: [prefix | target_tokens]
        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

        # Extend attention mask
        prefix_mask = torch.ones(batch_size, prefix_embeds.shape[1], dtype=attention_mask.dtype, device=self.device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Create labels: -100 for prefix, then input_ids
        prefix_labels = torch.full((batch_size, prefix_embeds.shape[1]), -100, dtype=torch.long, device=self.device)
        labels = torch.cat([prefix_labels, input_ids], dim=1)
        shifted_labels = labels[..., 1:].contiguous()

        # Forward through GritLM
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = self.model(
                inputs_embeds=inputs_embeds[:, :-1],
                attention_mask=extended_mask[:, :-1],
                use_cache=False,
            )

        logits = outputs.logits
        vocab_size = logits.shape[-1]

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            shifted_labels.view(-1),
            ignore_index=-100,
        )

        # Total loss: CE + beta * VAE_KL + small VAE reconstruction term
        total_loss = ce_loss + self.args.vae_beta * vae_kl_loss + 0.1 * vae_recon_loss

        # Metrics
        with torch.no_grad():
            valid_mask = shifted_labels.view(-1) != -100
            if valid_mask.sum() > 0:
                perplexity = torch.exp(F.cross_entropy(
                    logits.view(-1, vocab_size)[valid_mask],
                    shifted_labels.view(-1)[valid_mask],
                )).item()
            else:
                perplexity = float("inf")

        metrics = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "vae_recon": vae_recon_loss.item(),
            "vae_kl": vae_kl_loss.item(),
            "perplexity": perplexity,
            "embedding_cosine": (1 - vae_recon_loss.item()),
        }

        return total_loss, metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int, use_precomputed: bool = False) -> Dict[str, float]:
        """Train for one epoch."""
        self.vae.train()
        self.projector.train()

        total_metrics = {k: 0.0 for k in ["total_loss", "ce_loss", "vae_recon", "vae_kl", "perplexity", "embedding_cosine"]}
        n_batches = 0

        self.vae_optimizer.zero_grad()
        self.proj_optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            if use_precomputed:
                texts, embeddings = batch
                loss, metrics = self.compute_loss(texts, precomputed_embeddings=embeddings)
            else:
                texts = batch
                loss, metrics = self.compute_loss(texts)

            # Scale for gradient accumulation
            loss = loss / self.args.gradient_accumulation

            # Backward
            self.scaler.scale(loss).backward()

            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] += v
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['total_loss']:.4f}",
                "ppl": f"{metrics['perplexity']:.1f}",
                "cos": f"{metrics['embedding_cosine']:.3f}",
            })

            # Gradient step
            if (step + 1) % self.args.gradient_accumulation == 0:
                # Unscale and clip
                self.scaler.unscale_(self.vae_optimizer)
                self.scaler.unscale_(self.proj_optimizer)

                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.projector.parameters(), self.args.grad_clip)

                # Step
                self.scaler.step(self.vae_optimizer)
                self.scaler.step(self.proj_optimizer)
                self.scaler.update()

                self.vae_optimizer.zero_grad()
                self.proj_optimizer.zero_grad()

        # Average metrics
        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, use_precomputed: bool = False) -> Dict[str, float]:
        """Validate model."""
        self.vae.eval()
        self.projector.eval()

        total_metrics = {k: 0.0 for k in ["total_loss", "ce_loss", "vae_recon", "perplexity", "embedding_cosine"]}
        n_batches = 0

        for batch in dataloader:
            if use_precomputed:
                texts, embeddings = batch
                _, metrics = self.compute_loss(texts, precomputed_embeddings=embeddings)
            else:
                texts = batch
                _, metrics = self.compute_loss(texts)
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save both VAE and Projector."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "vae_state_dict": self.vae.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
            "proj_optimizer": self.proj_optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(self.args),
        }, path)
        print(f"[Checkpoint] Saved to {path}")


def main():
    args = parse_args()
    device = args.device

    torch.manual_seed(args.seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load GritLM (single instance for both encoding and generation)
    print(f"Loading GritLM: {args.gritlm_model}...")
    from lido_pp.backbone.gritlm_encoder import GritLMUnifiedEncoder

    encoder = GritLMUnifiedEncoder(
        model_name=args.gritlm_model,
        device=device,
        dtype="float16",
    )

    # Use the same model for generation (access internal model)
    model = encoder.model
    tokenizer = encoder.tokenizer
    hidden_dim = encoder.hidden_dim
    print(f"GritLM hidden dim: {hidden_dim}")

    # Create VAE
    print("Creating VAE...")
    vae = InstructionVAE(
        input_dim=768,
        latent_dim=args.latent_dim,
        beta=args.vae_beta,
    ).to(device)

    if args.resume_vae:
        print(f"Loading VAE from {args.resume_vae}")
        ckpt = torch.load(args.resume_vae, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])

    # Create Projector
    print("Creating Projector...")
    projector = LatentProjector(
        latent_dim=768,  # Takes VAE decoder output (768D)
        hidden_dim=hidden_dim,
        num_prefix_tokens=args.num_prefix_tokens,
    ).to(device)

    if args.resume_proj:
        print(f"Loading Projector from {args.resume_proj}")
        ckpt = torch.load(args.resume_proj, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt["model_state_dict"])

    # Load dataset
    use_precomputed = args.precomputed_embeddings is not None

    if use_precomputed:
        print(f"Loading precomputed embeddings from {args.precomputed_embeddings}...")
        emb_data = torch.load(args.precomputed_embeddings, weights_only=False)
        all_embeddings = emb_data["embeddings"]
        texts = emb_data["instructions"]
        print(f"Loaded {len(texts)} samples with precomputed embeddings")
        print(f"Embeddings shape: {all_embeddings.shape}")
    else:
        if args.ultrachat_samples is not None:
            print("Loading combined Alpaca + UltraChat dataset...")
            texts = load_combined_dataset(
                alpaca_samples=args.max_samples,
                ultrachat_samples=args.ultrachat_samples if args.ultrachat_samples > 0 else None,
            )
        else:
            print("Loading Alpaca dataset...")
            texts = load_alpaca_dataset(max_samples=args.max_samples)
        all_embeddings = None
        print(f"Loaded {len(texts)} samples (embeddings computed on-the-fly)")

    # Train/val split
    n_val = int(len(texts) * args.val_ratio)
    import random
    random.seed(args.seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Create dataloaders
    if use_precomputed:
        train_embeddings = all_embeddings[train_indices]
        val_embeddings = all_embeddings[val_indices]
        train_dataset = TextEmbeddingDataset(train_texts, train_embeddings)
        val_dataset = TextEmbeddingDataset(val_texts, val_embeddings)
    else:
        train_dataset = TextDataset(train_texts)
        val_dataset = TextDataset(val_texts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if use_precomputed else 0,
        drop_last=True,
        pin_memory=use_precomputed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if use_precomputed else 0,
        pin_memory=use_precomputed,
    )

    # Create trainer
    trainer = TranslatorTrainer(
        vae=vae,
        projector=projector,
        gritlm_model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        args=args,
        device=device,
    )

    # Training loop
    print(f"\nStarting Translator training (VAE + Projector):")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"  VAE LR: {args.vae_lr}, Projector LR: {args.proj_lr}")
    print(f"  VAE beta (KL weight): {args.vae_beta}")
    print(f"  Precomputed embeddings: {use_precomputed}")
    print(f"  Num prefix tokens: {args.num_prefix_tokens}")

    best_val_loss = float("inf")
    start_time = datetime.now()

    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, use_precomputed=use_precomputed)

        # Update schedulers
        trainer.vae_scheduler.step()
        trainer.proj_scheduler.step()

        # Validate
        if (epoch + 1) % args.log_interval == 0:
            val_metrics = trainer.validate(val_loader, use_precomputed=use_precomputed)

            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val PPL: {val_metrics['perplexity']:.1f} | "
                f"Emb Cos: {val_metrics['embedding_cosine']:.3f} | "
                f"Time: {elapsed:.1f}m"
            )

            # Save best
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                trainer.save_checkpoint(
                    str(checkpoint_dir / "translator_best.pt"),
                    epoch, val_metrics
                )

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            trainer.save_checkpoint(
                str(checkpoint_dir / f"translator_epoch{epoch+1:04d}.pt"),
                epoch, train_metrics
            )

    # Final save
    trainer.save_checkpoint(
        str(checkpoint_dir / "translator_final.pt"),
        args.epochs, train_metrics
    )

    # Also save VAE and Projector separately for Phase B
    torch.save({
        "model_state_dict": vae.state_dict(),
        "args": {"latent_dim": args.latent_dim, "beta": args.vae_beta},
    }, str(checkpoint_dir / "vae_translator.pt"))

    torch.save({
        "model_state_dict": projector.state_dict(),
        "args": {"num_prefix_tokens": args.num_prefix_tokens, "hidden_dim": hidden_dim},
    }, str(checkpoint_dir / "projector_translator.pt"))

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nTraining complete! Total time: {total_time:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nNext step: Train FlowDiT (Phase B) using vae_translator.pt")


if __name__ == "__main__":
    main()
