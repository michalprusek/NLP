#!/usr/bin/env python3
"""
Train TextFlow on instruction prompts for NFBO.

This adapts the original TextFlow (designed for SELFIES molecules) to work with
natural language instruction prompts.

Usage:
    uv run python -m nfbo_gsm8k.train_textflow --data study/datasets/vs_10k.pt --epochs 100
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import local textflow module (patched without torchtext)
from nfbo_gsm8k.textflow.discreteflow_model import DFModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Tokenizer for Instructions
# ============================================================================

class InstructionTokenizer:
    """
    Simple character-level tokenizer for instruction prompts.

    Special tokens:
        0: <PAD>
        1: <EOS>
        2: <UNK>
        3+: characters
    """

    PAD_TOKEN = 0
    EOS_TOKEN = 1
    UNK_TOKEN = 2

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 3  # PAD, EOS, UNK

    def build_vocab(self, texts: list[str], min_freq: int = 1):
        """Build vocabulary from texts."""
        counter = Counter()
        for text in texts:
            counter.update(text)

        # Add characters with frequency >= min_freq
        for char, freq in sorted(counter.items()):
            if freq >= min_freq:
                self.char_to_idx[char] = self.vocab_size
                self.idx_to_char[self.vocab_size] = char
                self.vocab_size += 1

        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
        return self

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token indices."""
        tokens = []
        for char in text[:self.max_length - 1]:  # Leave room for EOS
            tokens.append(self.char_to_idx.get(char, self.UNK_TOKEN))
        tokens.append(self.EOS_TOKEN)

        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.PAD_TOKEN)

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token indices to text."""
        if tokens.dim() > 1:
            tokens = tokens.squeeze()

        chars = []
        for idx in tokens.tolist():
            if idx == self.EOS_TOKEN or idx == self.PAD_TOKEN:
                break
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])

        return "".join(chars)

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "max_length": self.max_length,
            "char_to_idx": self.char_to_idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "InstructionTokenizer":
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(max_length=data["max_length"])
        tokenizer.char_to_idx = data["char_to_idx"]
        tokenizer.idx_to_char = {int(v): k for k, v in data["char_to_idx"].items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer


# ============================================================================
# Dataset
# ============================================================================

class InstructionDataset(Dataset):
    """Dataset of tokenized instructions."""

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        eps, loss, metrics = model(batch, kl_weight=kl_weight)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += metrics["recon_loss"].item()
        total_kl += metrics["kl_loss"].item()

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def evaluate(model, dataloader, device, kl_weight=1.0):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            eps, loss, metrics = model(batch, kl_weight=kl_weight)

            total_loss += loss.item()
            total_recon += metrics["recon_loss"].item()
            total_kl += metrics["kl_loss"].item()

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def test_reconstruction(model, tokenizer, texts: list[str], device):
    """Test reconstruction quality."""
    model.eval()

    results = []
    for text in texts[:5]:
        try:
            tokens = tokenizer.encode(text).unsqueeze(0).to(device)

            # Encode
            z, _, _ = model.encode_z(tokens)

            # Decode - handle potential shape issues
            with torch.no_grad():
                recon, valid_mask, _ = model.decode_z(z)

            if len(recon) > 0:
                recon_text = tokenizer.decode(recon[0])
            else:
                recon_text = "[empty]"

            results.append({
                "original": text[:100],
                "reconstructed": recon_text[:100],
            })
        except Exception as e:
            results.append({
                "original": text[:100],
                "reconstructed": f"[error: {str(e)[:50]}]",
            })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TextFlow on instructions")

    # Data
    parser.add_argument("--data", type=str, default="study/datasets/vs_10k.pt",
                        help="Path to instruction dataset (.pt with 'instructions' key)")
    parser.add_argument("--json-data", type=str, default=None,
                        help="Path to JSON instruction list (alternative to --data)")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Maximum number of samples to use")

    # Model (original NFBO paper defaults)
    parser.add_argument("--hidden-size", type=int, default=500,
                        help="Hidden size for LSTM (paper: 500)")
    parser.add_argument("--z-size", type=int, default=40,
                        help="Latent dimension per position (paper: 40)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length (paper: 288 for molecules)")
    parser.add_argument("--num-flow-layers", type=int, default=5,
                        help="Number of flow layers in prior (paper: 5)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--kl-warmup", type=int, default=10,
                        help="KL warmup epochs")

    # Output
    parser.add_argument("--output-dir", type=str, default="nfbo_gsm8k/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TextFlow Training for Instructions")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    # Load data
    logger.info("Loading data...")
    if args.json_data:
        with open(args.json_data) as f:
            texts = json.load(f)
        if isinstance(texts[0], dict):
            texts = [t.get("instruction", t.get("text", str(t))) for t in texts]
    else:
        data = torch.load(args.data)
        texts = data["instructions"]

    # Limit samples
    if len(texts) > args.max_samples:
        indices = np.random.choice(len(texts), args.max_samples, replace=False)
        texts = [texts[i] for i in indices]

    logger.info(f"Loaded {len(texts)} instructions")
    logger.info(f"Sample: {texts[0][:100]}...")

    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = InstructionTokenizer(max_length=args.max_length)
    tokenizer.build_vocab(texts)
    tokenizer.save(output_dir / "tokenizer.json")

    # Tokenize all texts
    logger.info("Tokenizing...")
    tokens = torch.stack([tokenizer.encode(t) for t in tqdm(texts)])

    # Split into train/val
    n_val = min(1000, len(tokens) // 10)
    perm = torch.randperm(len(tokens))
    val_tokens = tokens[perm[:n_val]]
    train_tokens = tokens[perm[n_val:]]

    logger.info(f"Train: {len(train_tokens)}, Val: {len(val_tokens)}")

    # Create dataloaders
    train_loader = DataLoader(
        InstructionDataset(train_tokens),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        InstructionDataset(val_tokens),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating model...")
    # Original NFBO paper configuration
    prior_kwargs = {
        "p_rnn_layers": 2,  # paper: 2
        "p_rnn_units": args.hidden_size,  # paper: same as hidden_size
        "p_num_flow_layers": args.num_flow_layers,  # paper: 5
        "transform_function": "nlsq",  # paper: nlsq (not affine!)
        "nohiddenflow": False,  # paper: False (uses MADE hiddenflow)
        "hiddenflow_layers": 2,  # paper: 2
        "hiddenflow_units": 100,  # paper: 100
        "hiddenflow_flow_layers": 10,  # paper: 10
        "hiddenflow_scf_layers": False,  # paper: False
    }

    # Original NFBO paper model configuration
    model = DFModel(
        hidden_size=args.hidden_size,  # paper: 500
        zsize=args.z_size,  # paper: 40
        dropout_p=0.2,  # paper: 0.2
        dropout_locations=["prior_rnn"],  # paper: ['prior_rnn']
        prior_type="AF",  # paper: AF
        gen_bilstm_layers=2,  # paper: 2
        prior_kwargs=prior_kwargs,
        q_rnn_layers=2,  # paper: 2
        tie_weights=True,  # paper: true
        max_T=args.max_length,
        vocab_size=tokenizer.vocab_size,
        canonical=False,  # no SELFIES canonicalization for text
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # KL warmup
        kl_weight = min(1.0, epoch / args.kl_warmup)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, kl_weight)

        # Validate
        val_metrics = evaluate(model, val_loader, device, kl_weight)

        # Log
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: {train_metrics['loss']:.4f} (recon={train_metrics['recon_loss']:.4f}, kl={train_metrics['kl_loss']:.4f}) | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"KL weight: {kl_weight:.2f}"
        )

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": {
                    "hidden_size": args.hidden_size,
                    "z_size": args.z_size,
                    "max_length": args.max_length,
                    "vocab_size": tokenizer.vocab_size,
                    "num_flow_layers": args.num_flow_layers,
                },
            }, output_dir / "best_model.pt")
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Test reconstruction - disabled for now as decode_z is very slow with long sequences
        # if epoch % 10 == 0:
        #     recon_results = test_reconstruction(model, tokenizer, texts[:5], device)
        #     logger.info("Reconstruction samples:")
        #     for r in recon_results[:2]:
        #         logger.info(f"  Original: {r['original']}")
        #         logger.info(f"  Recon:    {r['reconstructed']}")

        scheduler.step()

    # Final save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_size": args.hidden_size,
            "z_size": args.z_size,
            "max_length": args.max_length,
            "vocab_size": tokenizer.vocab_size,
            "num_flow_layers": args.num_flow_layers,
        },
    }, output_dir / "final_model.pt")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
