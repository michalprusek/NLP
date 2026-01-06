"""Prepare task-agnostic training data with noise augmentation.

Fine-tune Vec2Text to be robust to embedding noise/perturbations.
This works for any downstream task, not just specific VAE reconstructions.

The key insight: VAE/AE reconstruction introduces specific types of noise:
1. Small angular deviation (cosine sim ~0.95-0.99)
2. Slight norm changes
3. Some information loss in high-frequency components

We simulate this with controlled noise augmentation.

Usage:
    # Use existing universal dataset
    uv run python -m lipo.vec2text_finetune.prepare_noisy_data

    # Or with custom input
    uv run python -m lipo.vec2text_finetune.prepare_noisy_data \
        --input lipo/vec2text_finetune/data/universal_train.json
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def add_reconstruction_noise(
    embeddings: torch.Tensor,
    target_cosine_range: Tuple[float, float] = (0.92, 0.99),
    norm_scale_range: Tuple[float, float] = (0.95, 1.05),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add noise that simulates VAE/AE reconstruction characteristics.

    Args:
        embeddings: [N, 768] clean GTR embeddings
        target_cosine_range: Range of cosine similarities to target
        norm_scale_range: Range of norm scaling factors

    Returns:
        noisy_embeddings: [N, 768] perturbed embeddings
        cosine_sims: [N] actual cosine similarities achieved
    """
    device = embeddings.device
    N, D = embeddings.shape

    # Sample target cosine for each embedding
    target_cosines = torch.FloatTensor(N).uniform_(*target_cosine_range).to(device)

    # Calculate noise scale needed to achieve target cosine
    # cos(theta) = dot(a, a+noise) / (|a| * |a+noise|)
    # For unit vectors: cos(theta) ≈ 1 - noise_var/2
    # So: noise_scale ≈ sqrt(2 * (1 - target_cosine))
    noise_scales = torch.sqrt(2 * (1 - target_cosines)).unsqueeze(1)

    # Generate noise orthogonal-ish to original embedding
    # This better simulates reconstruction error
    noise = torch.randn_like(embeddings)

    # Make noise more orthogonal to original (partial Gram-Schmidt)
    proj = (noise * embeddings).sum(dim=1, keepdim=True) * embeddings / (
        embeddings.norm(dim=1, keepdim=True) ** 2 + 1e-8
    )
    noise = noise - 0.5 * proj  # Partial orthogonalization
    noise = noise / (noise.norm(dim=1, keepdim=True) + 1e-8)

    # Apply scaled noise
    original_norms = embeddings.norm(dim=1, keepdim=True)
    noisy = embeddings + noise_scales * original_norms * noise

    # Apply random norm scaling
    norm_scales = torch.FloatTensor(N, 1).uniform_(*norm_scale_range).to(device)
    noisy = noisy * norm_scales

    # Re-normalize to similar magnitude as originals (important for Vec2Text)
    noisy_norms = noisy.norm(dim=1, keepdim=True)
    noisy = noisy * (original_norms / (noisy_norms + 1e-8))

    # Calculate actual cosine similarities
    cosine_sims = torch.nn.functional.cosine_similarity(embeddings, noisy, dim=1)

    return noisy, cosine_sims


def process_with_noise_augmentation(
    input_path: str,
    output_dir: str,
    num_augmentations: int = 3,
    target_cosine_range: Tuple[float, float] = (0.90, 0.98),
    batch_size: int = 1000,
) -> Dict:
    """Process dataset with noise augmentation.

    For each (text, embedding) pair, create multiple noisy versions.

    Args:
        input_path: Path to JSON with [{text, embedding}, ...]
        output_dir: Output directory
        num_augmentations: Number of noisy versions per original
        target_cosine_range: Range of cosine similarities
        batch_size: Processing batch size

    Returns:
        Statistics dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    if len(data) == 0:
        logger.error(f"No data found in {input_path}")
        return {"error": "empty dataset", "total": 0}

    logger.info(f"  Loaded {len(data)} examples")
    logger.info(f"  Creating {num_augmentations} augmented versions each")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = []
    all_cosines = []

    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Augmenting"):
        batch = data[i:i + batch_size]

        # Extract embeddings
        embeddings = torch.tensor(
            [item["embedding"] for item in batch],
            dtype=torch.float32,
            device=device
        )
        texts = [item["text"] for item in batch]

        # Create multiple augmented versions
        for aug_idx in range(num_augmentations):
            # Different noise levels for diversity
            cosine_min = target_cosine_range[0] + aug_idx * 0.02
            cosine_max = min(target_cosine_range[1] + aug_idx * 0.01, 0.995)

            noisy, cosines = add_reconstruction_noise(
                embeddings,
                target_cosine_range=(cosine_min, cosine_max),
            )

            all_cosines.extend(cosines.cpu().tolist())

            # Store results
            for j, text in enumerate(texts):
                all_results.append({
                    "text": text,
                    "embedding": noisy[j].cpu().tolist(),
                })

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_results)

    # Stats
    cosines = np.array(all_cosines)
    stats = {
        "total": len(all_results),
        "original_count": len(data),
        "augmentations": num_augmentations,
        "cosine_mean": float(cosines.mean()),
        "cosine_std": float(cosines.std()),
        "cosine_min": float(cosines.min()),
        "cosine_max": float(cosines.max()),
        "cosine_p10": float(np.percentile(cosines, 10)),
        "cosine_p50": float(np.percentile(cosines, 50)),
        "cosine_p90": float(np.percentile(cosines, 90)),
    }

    logger.info("Augmentation statistics:")
    logger.info(f"  Total examples: {stats['total']}")
    logger.info(f"  Cosine sim: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
    logger.info(f"  Range: [{stats['cosine_min']:.4f}, {stats['cosine_max']:.4f}]")
    logger.info(f"  P10/P50/P90: {stats['cosine_p10']:.4f}/{stats['cosine_p50']:.4f}/{stats['cosine_p90']:.4f}")

    # Split train/eval
    n_eval = min(10000, len(all_results) // 10)
    eval_data = all_results[:n_eval]
    train_data = all_results[n_eval:]

    # Save
    train_path = output_dir / "noisy_train.json"
    eval_path = output_dir / "noisy_eval.json"
    stats_path = output_dir / "augmentation_stats.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f)

    with open(eval_path, "w") as f:
        json.dump(eval_data, f)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved {len(train_data)} train examples to {train_path}")
    logger.info(f"Saved {len(eval_data)} eval examples to {eval_path}")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="lipo/vec2text_finetune/data/universal_train.json",
                        help="Input JSON with (text, embedding) pairs")
    parser.add_argument("--output-dir", type=str,
                        default="lipo/vec2text_finetune/data_noisy",
                        help="Output directory")
    parser.add_argument("--augmentations", type=int, default=3,
                        help="Number of noisy versions per original")
    parser.add_argument("--cosine-min", type=float, default=0.88,
                        help="Minimum target cosine similarity")
    parser.add_argument("--cosine-max", type=float, default=0.97,
                        help="Maximum target cosine similarity")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    stats = process_with_noise_augmentation(
        input_path=args.input,
        output_dir=args.output_dir,
        num_augmentations=args.augmentations,
        target_cosine_range=(args.cosine_min, args.cosine_max),
        batch_size=args.batch_size,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
