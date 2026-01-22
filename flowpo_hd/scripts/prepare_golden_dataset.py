#!/usr/bin/env python3
"""
Prepare Golden Dataset Mix for ManifoldKeeper training.

Composition:
- 40% Complex Reasoning (MathInstruct)
- 30% Hard Constraints & Code (OpenHermes-2.5 filtered)
- 20% General Instructions (alpaca-cleaned)
- 10% Diverse/Adversarial (Evol-Instruct)

Pre-processing:
1. Length filtering: 50-400 characters
2. Semantic deduplication: cosine > 0.95 → remove
3. NO normalization (raw SONAR values)

Target: 100k-200k unique instructions
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def filter_by_length(texts: List[str], min_len: int = 50, max_len: int = 400) -> List[str]:
    """Filter texts by character length."""
    filtered = [t for t in texts if min_len <= len(t.strip()) <= max_len]
    logger.info(f"  Length filter: {len(texts)} → {len(filtered)} ({len(filtered)/len(texts)*100:.1f}%)")
    return filtered


def deduplicate_exact(texts: List[str]) -> List[str]:
    """Remove exact duplicates."""
    unique = list(dict.fromkeys(texts))  # Preserves order
    logger.info(f"  Exact dedup: {len(texts)} → {len(unique)}")
    return unique


def load_mathinstruct(target_count: int) -> List[str]:
    """Load MathInstruct dataset (Complex Reasoning)."""
    logger.info("Loading MathInstruct...")

    try:
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
        instructions = [item['instruction'] for item in ds if item.get('instruction')]
        logger.info(f"  Raw: {len(instructions)} instructions")

        # Filter and dedupe
        instructions = filter_by_length(instructions)
        instructions = deduplicate_exact(instructions)

        # Sample if too many
        if len(instructions) > target_count * 2:
            import random
            random.seed(42)
            instructions = random.sample(instructions, target_count * 2)
            logger.info(f"  Sampled to {len(instructions)}")

        return instructions[:target_count * 2]  # Return extra for dedup later
    except Exception as e:
        logger.warning(f"Failed to load MathInstruct: {e}")
        return []


def load_openhermes_code(target_count: int) -> List[str]:
    """Load OpenHermes-2.5 filtered for code/json (Hard Constraints)."""
    logger.info("Loading OpenHermes-2.5 (code/json filtered)...")

    try:
        # Streaming to handle large dataset
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

        instructions = []
        code_keywords = ['code', 'json', 'python', 'javascript', 'function', 'class',
                        'api', 'sql', 'html', 'css', 'format', 'syntax', 'parse']

        for item in tqdm(ds, desc="Scanning OpenHermes", total=1000000):
            if len(instructions) >= target_count * 3:
                break

            # Get instruction text
            convs = item.get('conversations', [])
            if not convs:
                continue

            # Get first human message
            for conv in convs:
                if conv.get('from') == 'human':
                    text = conv.get('value', '')

                    # Filter for code/json related
                    text_lower = text.lower()
                    if any(kw in text_lower for kw in code_keywords):
                        if 50 <= len(text.strip()) <= 400:
                            instructions.append(text.strip())
                    break

        logger.info(f"  Filtered: {len(instructions)} code/json instructions")
        instructions = deduplicate_exact(instructions)

        return instructions[:target_count * 2]
    except Exception as e:
        logger.warning(f"Failed to load OpenHermes: {e}")
        return []


def load_alpaca_cleaned(target_count: int) -> List[str]:
    """Load alpaca-cleaned (General Instructions)."""
    logger.info("Loading alpaca-cleaned...")

    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")

        instructions = []
        for item in ds:
            # Combine instruction and input if present
            instr = item.get('instruction', '')
            inp = item.get('input', '')

            if inp:
                text = f"{instr}\n{inp}"
            else:
                text = instr

            if text.strip():
                instructions.append(text.strip())

        logger.info(f"  Raw: {len(instructions)} instructions")
        instructions = filter_by_length(instructions)
        instructions = deduplicate_exact(instructions)

        return instructions[:target_count * 2]
    except Exception as e:
        logger.warning(f"Failed to load alpaca-cleaned: {e}")
        return []


def load_evol_instruct(target_count: int) -> List[str]:
    """Load Evol-Instruct (Diverse/Adversarial)."""
    logger.info("Loading Evol-Instruct...")

    try:
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")

        instructions = [item['instruction'] for item in ds if item.get('instruction')]
        logger.info(f"  Raw: {len(instructions)} instructions")

        instructions = filter_by_length(instructions)
        instructions = deduplicate_exact(instructions)

        return instructions[:target_count * 2]
    except Exception as e:
        logger.warning(f"Failed to load Evol-Instruct: {e}")
        return []


def semantic_deduplication(
    instructions: List[str],
    threshold: float = 0.95,
    batch_size: int = 512,
    device: str = "cuda",
) -> List[str]:
    """
    Remove semantically similar instructions using SONAR embeddings.

    Uses batched processing and approximate nearest neighbor search.
    """
    logger.info(f"Semantic deduplication (threshold={threshold})...")
    logger.info(f"  Input: {len(instructions)} instructions")

    # Import SONAR helper
    import sys
    sys.path.insert(0, '/home/prusek/NLP')
    from flowpo_hd.utils import SONARHelper

    sonar = SONARHelper(device=device, normalize=False)

    # Encode all instructions in batches
    logger.info("  Encoding with SONAR...")
    embeddings = []
    for i in tqdm(range(0, len(instructions), batch_size), desc="Encoding"):
        batch = instructions[i:i+batch_size]
        emb = sonar.encode(batch)
        embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    # Normalize for cosine similarity
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)

    # Greedy deduplication (keep first, remove similar)
    logger.info("  Finding duplicates...")
    keep_mask = torch.ones(len(instructions), dtype=torch.bool)

    # Process in chunks to avoid OOM
    chunk_size = 5000
    for start_idx in tqdm(range(0, len(instructions), chunk_size), desc="Dedup chunks"):
        end_idx = min(start_idx + chunk_size, len(instructions))

        # Compute similarity of this chunk against all kept embeddings
        chunk_emb = embeddings_norm[start_idx:end_idx].to(device)

        for i in range(start_idx, end_idx):
            if not keep_mask[i]:
                continue

            # Compare with all subsequent embeddings
            if i + 1 < len(instructions):
                query = embeddings_norm[i:i+1].to(device)
                candidates = embeddings_norm[i+1:].to(device)

                # Compute similarities
                sims = torch.mm(query, candidates.T).squeeze(0)

                # Mark similar ones for removal
                similar_mask = sims > threshold
                if similar_mask.any():
                    similar_indices = torch.where(similar_mask)[0] + i + 1
                    keep_mask[similar_indices.cpu()] = False

    # Filter instructions
    kept_instructions = [inst for inst, keep in zip(instructions, keep_mask) if keep]
    logger.info(f"  After dedup: {len(kept_instructions)} ({len(kept_instructions)/len(instructions)*100:.1f}%)")

    return kept_instructions, embeddings[keep_mask]


def main():
    parser = argparse.ArgumentParser(description="Prepare Golden Dataset Mix")
    parser.add_argument("--target-size", type=int, default=300000,
                       help="Target dataset size (default: 300k)")
    parser.add_argument("--output-dir", type=str, default="flowpo_hd/data",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for SONAR encoding")
    parser.add_argument("--dedup-threshold", type=float, default=0.95,
                       help="Cosine similarity threshold for deduplication")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, use cached data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate target counts per category
    target_counts = {
        'mathinstruct': int(args.target_size * 0.40),
        'openhermes': int(args.target_size * 0.30),
        'alpaca': int(args.target_size * 0.20),
        'evol': int(args.target_size * 0.10),
    }

    logger.info("=" * 60)
    logger.info("Golden Dataset Mix Preparation")
    logger.info("=" * 60)
    logger.info(f"Target size: {args.target_size:,}")
    logger.info(f"Composition:")
    for name, count in target_counts.items():
        logger.info(f"  {name}: {count:,}")
    logger.info("=" * 60)

    # Load datasets
    all_instructions = []
    sources = []

    # 1. MathInstruct (40%)
    math_instr = load_mathinstruct(target_counts['mathinstruct'])
    all_instructions.extend(math_instr[:target_counts['mathinstruct']])
    sources.extend(['mathinstruct'] * len(math_instr[:target_counts['mathinstruct']]))

    # 2. OpenHermes code/json (30%)
    hermes_instr = load_openhermes_code(target_counts['openhermes'])
    all_instructions.extend(hermes_instr[:target_counts['openhermes']])
    sources.extend(['openhermes'] * len(hermes_instr[:target_counts['openhermes']]))

    # 3. Alpaca-cleaned (20%)
    alpaca_instr = load_alpaca_cleaned(target_counts['alpaca'])
    all_instructions.extend(alpaca_instr[:target_counts['alpaca']])
    sources.extend(['alpaca'] * len(alpaca_instr[:target_counts['alpaca']]))

    # 4. Evol-Instruct (10%)
    evol_instr = load_evol_instruct(target_counts['evol'])
    all_instructions.extend(evol_instr[:target_counts['evol']])
    sources.extend(['evol'] * len(evol_instr[:target_counts['evol']]))

    logger.info(f"\nTotal before dedup: {len(all_instructions):,}")

    # Shuffle
    import random
    random.seed(42)
    combined = list(zip(all_instructions, sources))
    random.shuffle(combined)
    all_instructions, sources = zip(*combined)
    all_instructions = list(all_instructions)
    sources = list(sources)

    # Semantic deduplication
    kept_instructions, embeddings = semantic_deduplication(
        all_instructions,
        threshold=args.dedup_threshold,
        device=args.device,
    )

    # Final stats
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL DATASET STATS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total instructions: {len(kept_instructions):,}")
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Norm stats: mean={embeddings.norm(dim=-1).mean():.4f}, std={embeddings.norm(dim=-1).std():.4f}")

    # Sample examples
    logger.info(f"\nSample instructions:")
    for i in [0, len(kept_instructions)//4, len(kept_instructions)//2, -1]:
        logger.info(f"  [{i}] {kept_instructions[i][:100]}...")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"golden_mix_{timestamp}.pt")

    torch.save({
        'embeddings': embeddings,
        'instructions': kept_instructions,
        'metadata': {
            'model': 'sonar-text-encoder',
            'encoder_type': 'sonar',
            'dataset': 'golden_mix',
            'composition': {
                'mathinstruct': 0.40,
                'openhermes_code': 0.30,
                'alpaca': 0.20,
                'evol_instruct': 0.10,
            },
            'n_samples': len(kept_instructions),
            'embedding_dim': 1024,
            'normalized': False,
            'dedup_threshold': args.dedup_threshold,
            'length_filter': '50-400 chars',
            'source_lang': 'eng_Latn',
            'timestamp': datetime.now().isoformat(),
        }
    }, output_path)

    logger.info(f"\nSaved to {output_path}")

    # Also save a symlink as latest
    latest_path = os.path.join(args.output_dir, "golden_mix_latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(output_path), latest_path)
    logger.info(f"Symlinked as {latest_path}")


if __name__ == "__main__":
    main()
