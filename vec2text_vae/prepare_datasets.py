"""Prepare instruction datasets optimized for prompt optimization / LSBO.

Focus on datasets with:
- Chain-of-thought reasoning traces
- Multiple prompt variants for same tasks
- High-quality instruction-response pairs

Datasets included:
- OpenOrca (GPT-4 CoT reasoning)
- WizardLM (evolved instructions)
- UltraChat (diverse dialogues)
- P3/PromptSource (multi-prompt per task)
- FLAN (diverse NLP tasks)
- MetaMathQA (math reasoning)
- GSM8K (math CoT)
"""

import logging
from pathlib import Path
from typing import List, Optional
import json
import random

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_openorca(max_samples: int = 200000) -> List[str]:
    """OpenOrca - GPT-4 generated CoT reasoning traces."""
    logger.info("Loading OpenOrca...")
    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

        texts = []
        for ex in tqdm(ds, desc="OpenOrca", total=max_samples):
            # Combine system, question, and response with CoT
            parts = []
            if ex.get('system_prompt'):
                parts.append(f"System: {ex['system_prompt']}")
            if ex.get('question'):
                parts.append(f"Question: {ex['question']}")
            if ex.get('response'):
                parts.append(f"Response: {ex['response']}")

            text = "\n".join(parts)
            if len(text) > 100:
                texts.append(text[:2048])

            if len(texts) >= max_samples:
                break

        logger.info(f"OpenOrca: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load OpenOrca: {e}")
        return []


def extract_wizardlm(max_samples: int = 100000) -> List[str]:
    """WizardLM - Evolved complex instructions."""
    logger.info("Loading WizardLM Evol-Instruct...")
    try:
        ds = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train")

        texts = []
        for ex in tqdm(ds, desc="WizardLM"):
            convs = ex.get('conversations', [])
            if convs:
                conv_text = "\n".join([
                    f"{c.get('from', 'user')}: {c.get('value', '')}"
                    for c in convs
                ])
                if len(conv_text) > 100:
                    texts.append(conv_text[:2048])

            if max_samples and len(texts) >= max_samples:
                break

        logger.info(f"WizardLM: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load WizardLM: {e}")
        return []


def extract_ultrachat(max_samples: int = 100000) -> List[str]:
    """UltraChat - Diverse multi-turn dialogues."""
    logger.info("Loading UltraChat...")
    try:
        ds = load_dataset("stingning/ultrachat", split="train", streaming=True)

        texts = []
        for ex in tqdm(ds, desc="UltraChat", total=max_samples):
            data = ex.get('data', [])
            if data:
                # Alternate user/assistant
                conv_parts = []
                for i, msg in enumerate(data):
                    role = "User" if i % 2 == 0 else "Assistant"
                    conv_parts.append(f"{role}: {msg}")

                conv_text = "\n".join(conv_parts)
                if len(conv_text) > 100:
                    texts.append(conv_text[:2048])

            if len(texts) >= max_samples:
                break

        logger.info(f"UltraChat: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load UltraChat: {e}")
        return []


def extract_flan(max_samples: int = 100000) -> List[str]:
    """FLAN Collection - Diverse NLP tasks with instructions."""
    logger.info("Loading FLAN Collection...")
    try:
        ds = load_dataset("Muennighoff/flan", split="train", streaming=True)

        texts = []
        for ex in tqdm(ds, desc="FLAN", total=max_samples):
            # Combine inputs and targets with task info
            parts = []
            if ex.get('inputs'):
                parts.append(ex['inputs'])
            if ex.get('targets'):
                parts.append(f"Answer: {ex['targets']}")

            text = "\n".join(parts)
            if len(text) > 50:
                texts.append(text[:2048])

            if len(texts) >= max_samples:
                break

        logger.info(f"FLAN: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load FLAN: {e}")
        return []


def extract_metamath(max_samples: int = 100000) -> List[str]:
    """MetaMathQA - Math reasoning with multiple solution styles."""
    logger.info("Loading MetaMathQA...")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train")

        texts = []
        for ex in tqdm(ds, desc="MetaMathQA"):
            # Query + response with CoT
            text = f"Question: {ex.get('query', '')}\n\nSolution: {ex.get('response', '')}"
            if len(text) > 100:
                texts.append(text[:2048])

            if max_samples and len(texts) >= max_samples:
                break

        logger.info(f"MetaMathQA: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load MetaMathQA: {e}")
        return []


def extract_gsm8k_cot(max_samples: int = None) -> List[str]:
    """GSM8K with Chain-of-Thought solutions."""
    logger.info("Loading GSM8K...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")

        texts = []
        for ex in tqdm(ds, desc="GSM8K"):
            text = f"Question: {ex.get('question', '')}\n\nStep-by-step solution: {ex.get('answer', '')}"
            if len(text) > 50:
                texts.append(text[:2048])

            if max_samples and len(texts) >= max_samples:
                break

        logger.info(f"GSM8K: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load GSM8K: {e}")
        return []


def extract_code_alpaca(max_samples: int = 50000) -> List[str]:
    """Code Alpaca - Programming instructions."""
    logger.info("Loading Code Alpaca...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

        texts = []
        for ex in tqdm(ds, desc="CodeAlpaca"):
            parts = []
            if ex.get('instruction'):
                parts.append(f"Task: {ex['instruction']}")
            if ex.get('input'):
                parts.append(f"Input: {ex['input']}")
            if ex.get('output'):
                parts.append(f"Code:\n{ex['output']}")

            text = "\n".join(parts)
            if len(text) > 50:
                texts.append(text[:2048])

            if max_samples and len(texts) >= max_samples:
                break

        logger.info(f"CodeAlpaca: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load CodeAlpaca: {e}")
        return []


def extract_sharegpt(max_samples: int = 100000) -> List[str]:
    """ShareGPT - Real user conversations with GPT."""
    logger.info("Loading ShareGPT...")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")

        texts = []
        for ex in tqdm(ds, desc="ShareGPT"):
            convs = ex.get('conversations', [])
            if convs:
                conv_parts = [f"{c.get('from', 'human')}: {c.get('value', '')}" for c in convs]
                conv_text = "\n".join(conv_parts)
                if len(conv_text) > 100:
                    texts.append(conv_text[:2048])

            if max_samples and len(texts) >= max_samples:
                break

        logger.info(f"ShareGPT: {len(texts)} texts")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load ShareGPT: {e}")
        return []


def prepare_combined_dataset(
    output_path: str = "vec2text_vae/cache/combined_texts.json",
    target_size: int = 2000000,
) -> List[str]:
    """Download and combine all datasets for prompt optimization. Target: 2M instructions."""

    all_texts = []

    # Priority datasets - maximize CoT and reasoning
    all_texts.extend(extract_openorca(500000))          # CoT reasoning (largest)
    all_texts.extend(extract_wizardlm(196000))          # Complex instructions (full dataset)
    all_texts.extend(extract_metamath(395000))          # Math reasoning (full dataset)
    all_texts.extend(extract_flan(500000))              # Diverse NLP tasks

    # Secondary datasets
    all_texts.extend(extract_ultrachat(200000))         # Multi-turn dialogue
    all_texts.extend(extract_sharegpt(90000))           # Real conversations (full)
    all_texts.extend(extract_gsm8k_cot())               # Math CoT (~7.5k)
    all_texts.extend(extract_code_alpaca(20000))        # Code instructions (full)

    logger.info(f"Total before dedup: {len(all_texts)} texts")

    # Fast exact dedup on prefix (skip slow MinHash)
    logger.info("Exact deduplication...")
    seen_prefix = set()
    unique_texts = []
    for text in tqdm(all_texts, desc="Exact dedup"):
        prefix = text[:300].lower().strip()
        h = hash(prefix)
        if h not in seen_prefix:
            seen_prefix.add(h)
            unique_texts.append(text)

    logger.info(f"After exact dedup: {len(unique_texts)} texts")

    # Shuffle
    random.shuffle(unique_texts)

    # Limit to target size
    if len(unique_texts) > target_size:
        unique_texts = unique_texts[:target_size]
        logger.info(f"Limited to {target_size} texts")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(unique_texts, f)

    logger.info(f"Saved to {output_path}")

    # Print stats
    avg_len = sum(len(t) for t in unique_texts) / len(unique_texts)
    logger.info(f"Average text length: {avg_len:.0f} chars")

    return unique_texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=2000000,
                        help="Target dataset size after dedup (default: 2M)")
    parser.add_argument("--output", type=str, default="vec2text_vae/cache/combined_texts.json")
    args = parser.parse_args()

    texts = prepare_combined_dataset(
        output_path=args.output,
        target_size=args.target_size,
    )

    print(f"\nDataset ready: {len(texts)} texts")
    print(f"Saved to: {args.output}")
