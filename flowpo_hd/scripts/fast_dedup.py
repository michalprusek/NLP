#!/usr/bin/env python3
"""
Fast semantic deduplication using FAISS.

Takes pre-encoded SONAR embeddings and removes duplicates
based on cosine similarity threshold.
"""

import argparse
import logging
import os
from datetime import datetime

import faiss
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def fast_dedup_faiss(
    embeddings: torch.Tensor,
    threshold: float = 0.92,
    batch_size: int = 10000,
) -> np.ndarray:
    """
    Fast deduplication using FAISS IndexFlatIP (inner product = cosine for normalized vectors).

    Returns indices of items to keep.
    """
    n_samples, dim = embeddings.shape
    logger.info(f"Fast FAISS dedup: {n_samples:,} samples, dim={dim}, threshold={threshold}")

    # Normalize for cosine similarity
    embeddings_np = embeddings.numpy().astype(np.float32)
    faiss.normalize_L2(embeddings_np)

    # Build index incrementally
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized

    keep_indices = []

    for start in tqdm(range(0, n_samples, batch_size), desc="Dedup"):
        end = min(start + batch_size, n_samples)
        batch = embeddings_np[start:end]

        if index.ntotal == 0:
            # First batch - keep all
            keep_mask = np.ones(end - start, dtype=bool)
        else:
            # Search against existing index
            similarities, _ = index.search(batch, 1)  # Find most similar
            max_sims = similarities[:, 0]
            keep_mask = max_sims < threshold

        # Also check within batch (sequential)
        for i in range(len(keep_mask)):
            if not keep_mask[i]:
                continue
            # Check against previously kept items in this batch
            for j in range(i):
                if keep_mask[j]:
                    sim = np.dot(batch[i], batch[j])
                    if sim >= threshold:
                        keep_mask[i] = False
                        break

        # Add kept items to index and record indices
        for i in range(len(keep_mask)):
            if keep_mask[i]:
                global_idx = start + i
                keep_indices.append(global_idx)
                index.add(batch[i:i+1])

    logger.info(f"Kept {len(keep_indices):,} / {n_samples:,} ({len(keep_indices)/n_samples*100:.1f}%)")
    return np.array(keep_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                       help="Input .pt file with embeddings")
    parser.add_argument("--output", type=str, default=None,
                       help="Output .pt file (default: input_deduped.pt)")
    parser.add_argument("--threshold", type=float, default=0.92,
                       help="Cosine similarity threshold")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading {args.input}...")
    data = torch.load(args.input, map_location="cpu")

    embeddings = data["embeddings"]
    instructions = data.get("instructions", None)
    metadata = data.get("metadata", {})

    logger.info(f"Embeddings: {embeddings.shape}")
    logger.info(f"Instructions: {len(instructions) if instructions else 'N/A'}")

    # Deduplicate
    keep_indices = fast_dedup_faiss(
        embeddings,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    # Filter
    kept_embeddings = embeddings[keep_indices]
    kept_instructions = [instructions[i] for i in keep_indices] if instructions else None

    logger.info(f"After dedup: {len(kept_embeddings):,} samples")
    logger.info(f"Norm: mean={kept_embeddings.norm(dim=-1).mean():.4f}")

    # Update metadata
    metadata["dedup_threshold"] = args.threshold
    metadata["n_samples_before_dedup"] = len(embeddings)
    metadata["n_samples"] = len(kept_embeddings)
    metadata["dedup_timestamp"] = datetime.now().isoformat()

    # Save
    output_path = args.output or args.input.replace(".pt", "_deduped.pt")
    torch.save({
        "embeddings": kept_embeddings,
        "instructions": kept_instructions,
        "metadata": metadata,
    }, output_path)
    logger.info(f"Saved to {output_path}")

    # Show samples
    if kept_instructions:
        logger.info("\nSample instructions:")
        import random
        for i in random.sample(range(len(kept_instructions)), min(5, len(kept_instructions))):
            logger.info(f"  [{i}] {kept_instructions[i][:80]}...")


if __name__ == "__main__":
    main()
