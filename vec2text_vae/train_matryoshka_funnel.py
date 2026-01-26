#!/usr/bin/env python3
"""Train Cascading Matryoshka Funnel Flow with 128D latent space.

Architecture: 768D → 128D with cascading decoders at each 16D step
    z[:16] → z[16:32] → z[32:48] → ... → z[112:128] → z[128:768]

Each cascade decoder predicts only 16 dimensions, making the task simple.
The final decoder predicts 640 dimensions (768-128).

Benefits of 128D over 64D:
- More capacity for semantic nuances
- Smaller final decoder jump (640D vs 704D)
- Same GP complexity (still 16D per stage)
- 8 stages instead of 4 for finer-grained optimization

Usage:
    # Quick test
    python vec2text_vae/train_matryoshka_funnel.py --epochs 10 --n-samples 10000

    # Full training on A100 (recommended)
    tmux new-session -d -s cascade_train \
        "CUDA_VISIBLE_DEVICES=0 python vec2text_vae/train_matryoshka_funnel.py \
        --dataset combined --epochs 100 --batch-size 16384 --latent-dim 128 \
        2>&1 | tee results/cascade_128d_$(date +%Y%m%d_%H%M%S).log"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text_vae.matryoshka_funnel import (
    CascadingMatryoshkaFunnelLoss,
    CascadingMatryoshkaGTRFunnelFlow,
    MatryoshkaFunnelLoss,
    MatryoshkaGTRFunnelFlow,
    ProgressiveMatryoshkaScheduler,
    evaluate_matryoshka_reconstruction,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Embedding extraction (vec2text compatible)
# =============================================================================


def get_vec2text_embeddings(
    texts: list,
    device: str = "cuda",
    batch_size: int = 2048
) -> torch.Tensor:
    """Get GTR embeddings using vec2text-compatible method."""
    import vec2text.models.model_utils as model_utils
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading GTR-T5-Base encoder for vec2text-compatible embeddings...")

    model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base")
    encoder = model.encoder.to(device)
    encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            model_output = encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            embeddings = model_utils.mean_pool(
                model_output.last_hidden_state,
                inputs['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        all_embeddings.append(embeddings.cpu())

    del encoder, model
    torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0)


# =============================================================================
# Evaluation utilities
# =============================================================================


def evaluate_cascading_reconstruction(
    flow: CascadingMatryoshkaGTRFunnelFlow,
    embeddings: torch.Tensor,
    matryoshka_dims: Tuple[int, ...],
) -> Dict[str, float]:
    """Evaluate reconstruction quality at each Matryoshka level for cascading flow."""
    flow.eval()
    results = {}

    with torch.no_grad():
        z = flow.encode(embeddings)

        for level in matryoshka_dims:
            # Zero out dimensions beyond level
            z_masked = z.clone()
            z_masked[:, level:] = 0.0

            # Decode with cascade awareness
            x_recon = flow.decode(z_masked, active_dim=level, deterministic=True)
            cos_sim = F.cosine_similarity(embeddings, x_recon, dim=-1).mean().item()
            results[f'{level}D'] = cos_sim

    return results


# =============================================================================
# Training loop for Cascading architecture
# =============================================================================


def train_cascading_matryoshka_funnel(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor,
    matryoshka_dims: Tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 128),
    latent_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 8192,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: str = "vec2text_vae/checkpoints",
    encoder_hidden_dims: List[int] = None,
    decoder_hidden_dims: List[int] = None,
    predictor_hidden_dims: List[int] = None,
) -> CascadingMatryoshkaGTRFunnelFlow:
    """Train Cascading Matryoshka Funnel Flow.

    Args:
        train_embeddings: Training GTR embeddings
        val_embeddings: Validation GTR embeddings
        test_embeddings: Test GTR embeddings
        matryoshka_dims: Nested dimensions (e.g., 16, 32, ..., 128)
        latent_dim: Final latent dimension (should match last matryoshka_dim)
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        checkpoint_dir: Checkpoint directory
        encoder_hidden_dims: Hidden dims for encoder MLPs
        decoder_hidden_dims: Hidden dims for final decoder
        predictor_hidden_dims: Hidden dims for level predictors

    Returns:
        Trained CascadingMatryoshkaGTRFunnelFlow
    """
    # Defaults for hidden dimensions
    if encoder_hidden_dims is None:
        encoder_hidden_dims = [512, 256]
    if decoder_hidden_dims is None:
        decoder_hidden_dims = [256, 512]
    if predictor_hidden_dims is None:
        predictor_hidden_dims = [256, 256]
    logger.info("=" * 70)
    logger.info(f"Training CASCADING Matryoshka Funnel Flow")
    logger.info(f"Architecture: 768D → {latent_dim}D")
    logger.info(f"Cascade levels: {matryoshka_dims}")
    logger.info(f"Train: {train_embeddings.shape[0]:,}, Val: {val_embeddings.shape[0]:,}, Test: {test_embeddings.shape[0]:,}")
    logger.info("=" * 70)

    # Create cascading model
    flow = CascadingMatryoshkaGTRFunnelFlow(
        input_dim=768,
        latent_dim=latent_dim,
        matryoshka_dims=matryoshka_dims,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
    ).to(device)

    n_params = sum(p.numel() for p in flow.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Keep embeddings on GPU - we have plenty of VRAM on A100
    train_embeddings = train_embeddings.to(device)
    val_embeddings = val_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)

    train_dataset = TensorDataset(train_embeddings)
    # Note: pin_memory and num_workers not needed since data is already on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Cascading loss function
    loss_fn = CascadingMatryoshkaFunnelLoss(
        matryoshka_dims=matryoshka_dims,
        cascade_weight=1.0,
        recon_weight=1.0,
        nll_weight=0.01,
        progressive_cascade=True,
    )

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule with warmup
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_cos_sim_avg = 0.0
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'cascade_loss': [],
        'recon_loss': [],
        'val_cos_sim': {str(d): [] for d in matryoshka_dims},
        'lr': [],
    }

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Training
        flow.train()
        train_metrics = {
            'total_loss': 0,
            'cascade_loss': 0,
            'recon_loss': 0,
        }
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (lr={current_lr:.2e})")
        for (batch,) in pbar:
            optimizer.zero_grad()

            loss, metrics = loss_fn(flow, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()

            train_metrics['total_loss'] += metrics['total_loss']
            train_metrics['cascade_loss'] += metrics['cascade_loss']
            train_metrics['recon_loss'] += metrics['recon_loss']
            n_batches += 1

            # Show first and last level cos_sim
            first_level = matryoshka_dims[0]
            last_level = matryoshka_dims[-1]
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                f'{first_level}D': f"{metrics.get(f'cos_sim_{first_level}D', 0):.3f}",
                f'{last_level}D': f"{metrics.get(f'cos_sim_{last_level}D', 0):.3f}",
            })

        lr_scheduler.step()

        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= n_batches

        history['train_loss'].append(train_metrics['total_loss'])
        history['cascade_loss'].append(train_metrics['cascade_loss'])
        history['recon_loss'].append(train_metrics['recon_loss'])

        # Validation
        flow.eval()
        val_results = evaluate_cascading_reconstruction(
            flow, val_embeddings, matryoshka_dims
        )
        for dim, cos_sim in val_results.items():
            dim_key = dim.replace('D', '')
            if dim_key in history['val_cos_sim']:
                history['val_cos_sim'][dim_key].append(cos_sim)

        val_cos_sim_avg = np.mean(list(val_results.values()))

        # Save best model
        if val_cos_sim_avg > best_val_cos_sim_avg:
            best_val_cos_sim_avg = val_cos_sim_avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_results': val_results,
                'val_cos_sim_avg': best_val_cos_sim_avg,
                'matryoshka_dims': matryoshka_dims,
                'latent_dim': latent_dim,
                'architecture': 'cascading',
            }, checkpoint_path / "cascading_matryoshka_funnel_best.pt")
            logger.info(f"Epoch {epoch+1}: New best val_cos_sim_avg={best_val_cos_sim_avg:.4f} - saved!")

        # Log progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            first_level = matryoshka_dims[0]
            mid_level = matryoshka_dims[len(matryoshka_dims)//2]
            last_level = matryoshka_dims[-1]
            logger.info(
                f"Epoch {epoch+1}: loss={train_metrics['total_loss']:.4f} "
                f"(cascade={train_metrics['cascade_loss']:.4f}, recon={train_metrics['recon_loss']:.4f}), "
                f"val: {first_level}D={val_results.get(f'{first_level}D', 0):.3f}, "
                f"{mid_level}D={val_results.get(f'{mid_level}D', 0):.3f}, "
                f"{last_level}D={val_results.get(f'{last_level}D', 0):.3f}"
            )

    # Load best checkpoint
    best_ckpt = torch.load(
        checkpoint_path / "cascading_matryoshka_funnel_best.pt",
        weights_only=False  # Our checkpoint contains numpy scalars in val_results
    )
    flow.load_state_dict(best_ckpt['model_state_dict'])

    # Final test evaluation
    flow.eval()
    test_results = evaluate_cascading_reconstruction(flow, test_embeddings, matryoshka_dims)

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Best validation cos_sim_avg: {best_val_cos_sim_avg:.4f}")
    logger.info("Test results per cascade level:")
    for dim, cos_sim in test_results.items():
        logger.info(f"  {dim}: {cos_sim:.4f}")
    logger.info("=" * 70)

    # Save history
    with open(checkpoint_path / "cascading_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return flow


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Cascading Matryoshka Funnel Flow"
    )
    parser.add_argument("--latent-dim", type=int, default=128,
                        help="Latent dimension (default: 128)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="Batch size (16384 for A100 80GB)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples (None=all)")
    parser.add_argument("--dataset", type=str, default="combined",
                        choices=["finetome", "combined"],
                        help="Dataset to use")
    parser.add_argument("--step-size", type=int, default=16,
                        help="Step size for cascade levels (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="vec2text_vae/checkpoints")
    args = parser.parse_args()

    # Generate matryoshka dims from step size
    matryoshka_dims = tuple(range(args.step_size, args.latent_dim + 1, args.step_size))
    if matryoshka_dims[-1] != args.latent_dim:
        matryoshka_dims = matryoshka_dims + (args.latent_dim,)

    logger.info(f"Cascade levels: {matryoshka_dims}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {gpu_mem:.1f} GB")
        if gpu_mem >= 70:
            logger.info("Detected A100 80GB - using large batch size")
            if args.batch_size < 16384:
                logger.info(f"Consider using --batch-size 16384 for optimal throughput")

    # Load embeddings (prefer full cached, fallback to small)
    embeddings_cache_full = Path("vec2text_vae/cache/gtr_embeddings_full.pt")
    embeddings_cache_small = Path("vec2text_vae/cache/gtr_embeddings.pt")

    if embeddings_cache_full.exists():
        logger.info(f"Loading FULL cached embeddings from {embeddings_cache_full}...")
        embeddings = torch.load(embeddings_cache_full, map_location='cpu')
        if args.n_samples:
            embeddings = embeddings[:args.n_samples]
        logger.info(f"Loaded full cached embeddings: {embeddings.shape}")
    elif embeddings_cache_small.exists():
        logger.info(f"Loading small cached embeddings from {embeddings_cache_small}...")
        embeddings = torch.load(embeddings_cache_small, map_location='cpu')
        if args.n_samples:
            embeddings = embeddings[:args.n_samples]
        logger.info(f"Loaded small cached embeddings: {embeddings.shape}")
    else:
        # Fallback: encode from texts
        if args.dataset == "combined":
            cache_path = Path("vec2text_vae/cache/combined_texts.json")
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Combined dataset not found at {cache_path}. "
                    "Run: python vec2text_vae/prepare_datasets.py first"
                )
            logger.info(f"Loading combined dataset from {cache_path}...")
            with open(cache_path) as f:
                texts = json.load(f)
            if args.n_samples:
                texts = texts[:args.n_samples]
        else:
            from datasets import load_dataset
            logger.info("Loading FineTome-100k dataset...")
            dataset = load_dataset("mlabonne/FineTome-100k", split="train")
            n_samples = args.n_samples or len(dataset)
            texts = []
            for ex in dataset.select(range(min(n_samples, len(dataset)))):
                conv_text = " ".join([turn['value'] for turn in ex['conversations']])
                texts.append(conv_text[:1024])

        logger.info(f"Loaded {len(texts):,} texts")
        embeddings = get_vec2text_embeddings(texts, device)

        # Cache for next time - use full cache path
        try:
            embeddings_cache_full.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching embeddings to {embeddings_cache_full}...")
            torch.save(embeddings, embeddings_cache_full)
            logger.info(f"Successfully cached {embeddings.shape[0]:,} embeddings")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not cache embeddings: {e}. Continuing without caching.")

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # 80/10/10 split
    n_total = embeddings.shape[0]
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_embeddings = embeddings[train_idx]
    val_embeddings = embeddings[val_idx]
    test_embeddings = embeddings[test_idx]

    logger.info(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    # Train cascading model
    flow = train_cascading_matryoshka_funnel(
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        matryoshka_dims=matryoshka_dims,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
