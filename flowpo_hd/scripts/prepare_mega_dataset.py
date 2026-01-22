#!/usr/bin/env python3
"""
Prepare MEGA Dataset for ManifoldKeeper training.

Target: 1-2 MILLION high-quality instructions

Sources:
- MathInstruct (262k) - Complex reasoning
- OpenHermes-2.5 (1M) - Diverse high-quality
- Alpaca-cleaned (52k) - General instructions
- Evol-Instruct (70k) - Evolved instructions
- Dolly (15k) - Databricks instructions
- FLAN (subset) - Google's instruction collection
- SlimOrca (500k) - Cleaned Orca dataset

Pre-processing:
1. Length filtering: 50-500 characters
2. Semantic deduplication: cosine > 0.92 → remove
3. NO normalization (raw SONAR values)
"""

import argparse
import logging
import os
import random
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def filter_by_length(texts: List[str], min_len: int = 50, max_len: int = 500) -> List[str]:
    """Filter texts by character length."""
    filtered = [t.strip() for t in texts if min_len <= len(t.strip()) <= max_len]
    logger.info(f"  Length filter: {len(texts)} → {len(filtered)} ({len(filtered)/max(len(texts),1)*100:.1f}%)")
    return filtered


def deduplicate_exact(texts: List[str]) -> List[str]:
    """Remove exact duplicates."""
    unique = list(dict.fromkeys(texts))
    logger.info(f"  Exact dedup: {len(texts)} → {len(unique)}")
    return unique


def load_mathinstruct() -> List[str]:
    """Load MathInstruct dataset."""
    logger.info("Loading MathInstruct...")
    try:
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
        instructions = [item['instruction'] for item in ds if item.get('instruction')]
        logger.info(f"  Raw: {len(instructions)}")
        instructions = filter_by_length(instructions)
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load MathInstruct: {e}")
        return []


def load_openhermes_full() -> List[str]:
    """Load full OpenHermes-2.5 dataset."""
    logger.info("Loading OpenHermes-2.5 (FULL)...")
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")

        instructions = []
        for item in tqdm(ds, desc="Processing OpenHermes"):
            convs = item.get('conversations', [])
            for conv in convs:
                if conv.get('from') == 'human':
                    text = conv.get('value', '').strip()
                    if 50 <= len(text) <= 500:
                        instructions.append(text)
                    break

        logger.info(f"  Filtered: {len(instructions)}")
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load OpenHermes: {e}")
        return []


def load_alpaca() -> List[str]:
    """Load alpaca-cleaned."""
    logger.info("Loading alpaca-cleaned...")
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        instructions = []
        for item in ds:
            instr = item.get('instruction', '')
            inp = item.get('input', '')
            text = f"{instr}\n{inp}" if inp else instr
            if text.strip():
                instructions.append(text.strip())
        logger.info(f"  Raw: {len(instructions)}")
        instructions = filter_by_length(instructions)
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load alpaca: {e}")
        return []


def load_evol_instruct() -> List[str]:
    """Load Evol-Instruct."""
    logger.info("Loading Evol-Instruct...")
    try:
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
        instructions = [item['instruction'] for item in ds if item.get('instruction')]
        logger.info(f"  Raw: {len(instructions)}")
        instructions = filter_by_length(instructions)
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load Evol-Instruct: {e}")
        return []


def load_dolly() -> List[str]:
    """Load Dolly-15k."""
    logger.info("Loading Dolly-15k...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        instructions = []
        for item in ds:
            instr = item.get('instruction', '')
            ctx = item.get('context', '')
            text = f"{instr}\n{ctx}" if ctx else instr
            if text.strip():
                instructions.append(text.strip())
        logger.info(f"  Raw: {len(instructions)}")
        instructions = filter_by_length(instructions)
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load Dolly: {e}")
        return []


def load_slimorca() -> List[str]:
    """Load SlimOrca (cleaned Orca)."""
    logger.info("Loading SlimOrca...")
    try:
        ds = load_dataset("Open-Orca/SlimOrca", split="train")

        instructions = []
        for item in tqdm(ds, desc="Processing SlimOrca"):
            convs = item.get('conversations', [])
            for conv in convs:
                if conv.get('from') == 'human':
                    text = conv.get('value', '').strip()
                    if 50 <= len(text) <= 500:
                        instructions.append(text)
                    break

        logger.info(f"  Filtered: {len(instructions)}")
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load SlimOrca: {e}")
        return []


def load_ultrachat() -> List[str]:
    """Load UltraChat (full)."""
    logger.info("Loading UltraChat...")
    try:
        ds = load_dataset("stingning/ultrachat", split="train", streaming=True)

        instructions = []
        for i, item in enumerate(tqdm(ds, desc="Processing UltraChat", total=2000000)):
            if i >= 2000000:  # Limit to 2M
                break
            data = item.get('data', [])
            if data and len(data) > 0:
                text = data[0].strip() if isinstance(data[0], str) else str(data[0])
                if 50 <= len(text) <= 500:
                    instructions.append(text)

        logger.info(f"  Filtered: {len(instructions)}")
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load UltraChat: {e}")
        return []


def load_sharegpt() -> List[str]:
    """Load ShareGPT dataset."""
    logger.info("Loading ShareGPT...")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")

        instructions = []
        for item in tqdm(ds, desc="Processing ShareGPT"):
            convs = item.get('conversations', [])
            for conv in convs:
                if conv.get('from') == 'human':
                    text = conv.get('value', '').strip()
                    if 50 <= len(text) <= 500:
                        instructions.append(text)
                    break

        logger.info(f"  Filtered: {len(instructions)}")
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load ShareGPT: {e}")
        return []


def load_wizard_vicuna() -> List[str]:
    """Load WizardLM Vicuna ShareGPT."""
    logger.info("Loading WizardLM Vicuna...")
    try:
        ds = load_dataset("cognitivecomputations/wizard_vicuna_70k_unfiltered", split="train")

        instructions = []
        for item in tqdm(ds, desc="Processing WizardVicuna"):
            convs = item.get('conversations', [])
            for conv in convs:
                if conv.get('from') == 'human':
                    text = conv.get('value', '').strip()
                    if 50 <= len(text) <= 500:
                        instructions.append(text)
                    break

        logger.info(f"  Filtered: {len(instructions)}")
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load WizardVicuna: {e}")
        return []


def load_oasst() -> List[str]:
    """Load OpenAssistant dataset."""
    logger.info("Loading OpenAssistant...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")

        instructions = []
        for item in ds:
            if item.get('role') == 'prompter':
                text = item.get('text', '').strip()
                if 50 <= len(text) <= 500:
                    instructions.append(text)

        logger.info(f"  Filtered: {len(instructions)}")
        return deduplicate_exact(instructions)
    except Exception as e:
        logger.warning(f"Failed to load OpenAssistant: {e}")
        return []


def encode_with_sonar(
    instructions: List[str],
    batch_size: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode instructions with SONAR."""
    import sys
    sys.path.insert(0, '/home/prusek/NLP')
    from flowpo_hd.utils import SONARHelper

    sonar = SONARHelper(device=device, normalize=False)

    logger.info(f"Encoding {len(instructions):,} instructions with SONAR...")
    all_embeddings = []
    for i in tqdm(range(0, len(instructions), batch_size), desc="Encoding"):
        batch = instructions[i:i+batch_size]
        emb = sonar.encode(batch)
        all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info(f"Embeddings: {all_embeddings.shape}")
    return all_embeddings


def fast_dedup_faiss(
    embeddings: torch.Tensor,
    threshold: float = 0.92,
    batch_size: int = 10000,
) -> List[int]:
    """
    Fast deduplication using FAISS IndexFlatIP.
    Returns indices of items to keep.
    """
    import faiss
    import numpy as np

    n_samples, dim = embeddings.shape
    logger.info(f"Fast FAISS dedup: {n_samples:,} samples, threshold={threshold}")

    # Normalize for cosine similarity
    embeddings_np = embeddings.numpy().astype(np.float32)
    faiss.normalize_L2(embeddings_np)

    # Build index incrementally
    index = faiss.IndexFlatIP(dim)
    keep_indices = []

    for start in tqdm(range(0, n_samples, batch_size), desc="Dedup"):
        end = min(start + batch_size, n_samples)
        batch = embeddings_np[start:end]

        if index.ntotal == 0:
            keep_mask = np.ones(end - start, dtype=bool)
        else:
            similarities, _ = index.search(batch, 1)
            max_sims = similarities[:, 0]
            keep_mask = max_sims < threshold

        # Check within batch
        for i in range(len(keep_mask)):
            if not keep_mask[i]:
                continue
            for j in range(i):
                if keep_mask[j]:
                    sim = np.dot(batch[i], batch[j])
                    if sim >= threshold:
                        keep_mask[i] = False
                        break

        # Add kept items
        for i in range(len(keep_mask)):
            if keep_mask[i]:
                keep_indices.append(start + i)
                index.add(batch[i:i+1])

    logger.info(f"Kept {len(keep_indices):,} / {n_samples:,} ({len(keep_indices)/n_samples*100:.1f}%)")
    return keep_indices


def main():
    parser = argparse.ArgumentParser(description="Prepare MEGA Dataset")
    parser.add_argument("--output-dir", type=str, default="flowpo_hd/data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dedup-threshold", type=float, default=0.92)
    parser.add_argument("--max-per-source", type=int, default=0,
                       help="Max instructions per source (0=unlimited)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MEGA Dataset Preparation (Target: 1-2M instructions)")
    logger.info("=" * 70)

    all_instructions = []

    # Load all sources
    sources = [
        ("MathInstruct", load_mathinstruct),
        ("OpenHermes", load_openhermes_full),
        ("Alpaca", load_alpaca),
        ("EvolInstruct", load_evol_instruct),
        ("Dolly", load_dolly),
        ("SlimOrca", load_slimorca),
        ("UltraChat", load_ultrachat),
        ("ShareGPT", load_sharegpt),
        ("WizardVicuna", load_wizard_vicuna),
        ("OpenAssistant", load_oasst),
    ]

    source_counts = {}
    for name, loader in sources:
        try:
            instructions = loader()
            # Limit per source (if specified)
            if args.max_per_source > 0 and len(instructions) > args.max_per_source:
                random.seed(42)
                instructions = random.sample(instructions, args.max_per_source)
                logger.info(f"  Sampled to {args.max_per_source}")

            source_counts[name] = len(instructions)
            all_instructions.extend(instructions)
            logger.info(f"  {name}: {len(instructions):,} instructions")
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}")
            source_counts[name] = 0

    logger.info(f"\nTotal before dedup: {len(all_instructions):,}")

    # Shuffle
    random.seed(42)
    random.shuffle(all_instructions)

    # Exact dedup again (cross-source)
    all_instructions = deduplicate_exact(all_instructions)

    # Encode with SONAR
    embeddings = encode_with_sonar(all_instructions, device=args.device)

    # Save intermediate (in case dedup fails)
    intermediate_path = os.path.join(args.output_dir, "mega_raw_encoded.pt")
    torch.save({
        "embeddings": embeddings,
        "instructions": all_instructions,
        "metadata": {"n_samples": len(all_instructions), "sources": source_counts},
    }, intermediate_path)
    logger.info(f"Saved intermediate to {intermediate_path}")

    # Fast FAISS dedup
    keep_indices = fast_dedup_faiss(embeddings, threshold=args.dedup_threshold)
    kept_instructions = [all_instructions[i] for i in keep_indices]
    embeddings = embeddings[keep_indices]

    # Stats
    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL STATS")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total: {len(kept_instructions):,}")
    logger.info(f"Embeddings: {embeddings.shape}")
    logger.info(f"Norm: mean={embeddings.norm(dim=-1).mean():.4f}, std={embeddings.norm(dim=-1).std():.4f}")
    logger.info(f"\nSource breakdown:")
    for name, count in source_counts.items():
        logger.info(f"  {name}: {count:,}")

    # Samples
    logger.info(f"\nSample instructions:")
    for i in random.sample(range(len(kept_instructions)), min(5, len(kept_instructions))):
        logger.info(f"  [{i}] {kept_instructions[i][:100]}...")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"mega_instructions_{timestamp}.pt")

    torch.save({
        'embeddings': embeddings,
        'instructions': kept_instructions,
        'metadata': {
            'model': 'sonar-text-encoder',
            'dataset': 'mega_instructions',
            'sources': source_counts,
            'n_samples': len(kept_instructions),
            'embedding_dim': 1024,
            'normalized': False,
            'dedup_threshold': args.dedup_threshold,
            'length_filter': '50-500 chars',
            'timestamp': datetime.now().isoformat(),
        }
    }, output_path)

    logger.info(f"\nSaved to {output_path}")

    # Symlink
    latest_path = os.path.join(args.output_dir, "mega_instructions_latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(output_path), latest_path)
    logger.info(f"Symlinked to {latest_path}")


if __name__ == "__main__":
    main()
