"""Download universal instruction datasets from HuggingFace.

Selected high-quality instruction datasets:
- WizardLM Evol-Instruct (196k evolved complex instructions)
- Alpaca (52k LLM-generated instructions)
- Dolly (15k human-written instructions)
- Super-Natural Instructions (task definitions)

Usage:
    uv run python -m lipo.vec2text_finetune.download_instructions
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("lipo/vec2text_finetune/data")


def is_clean_instruction(text: str, min_len: int = 15, max_len: int = 500) -> bool:
    """Filter out garbage instructions."""
    if not text or not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < min_len or len(text) > max_len:
        return False

    # Must be mostly ASCII (English)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.9:
        return False

    # No excessive special characters
    special_ratio = sum(1 for c in text if not c.isalnum() and c not in " .,!?'-:;()\"") / len(text)
    if special_ratio > 0.3:
        return False

    # Must contain actual words
    words = re.findall(r'[a-zA-Z]+', text)
    if len(words) < 3:
        return False

    # No garbage patterns
    garbage_patterns = [
        r'^[^a-zA-Z]*$',
        r'(.)\1{5,}',
        r'[\u4e00-\u9fff]',  # Chinese
        r'[\u0400-\u04ff]',  # Cyrillic
        r'[\u0600-\u06ff]',  # Arabic
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, text):
            return False

    return True


def download_wizardlm() -> Tuple[List[str], List[str]]:
    """Download WizardLM Evol-Instruct (196k evolved instructions)."""
    from datasets import load_dataset

    logger.info("Downloading WizardLM Evol-Instruct (196k)...")
    instructions = []

    try:
        dataset = load_dataset(
            "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
            split="train",
        )

        for example in dataset:
            convs = example.get("conversations", [])
            if convs and len(convs) > 0:
                # First turn is the instruction
                text = convs[0].get("value", "")
                if text and is_clean_instruction(text):
                    instructions.append(text.strip())

        logger.info(f"  WizardLM: {len(instructions)} clean instructions")
        return instructions, instructions[:5]

    except Exception as e:
        logger.warning(f"  Failed to download WizardLM: {e}")
        return [], []


def download_alpaca() -> Tuple[List[str], List[str]]:
    """Download Stanford Alpaca dataset (52k)."""
    from datasets import load_dataset

    logger.info("Downloading Alpaca dataset (52k)...")
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        instructions = []

        for example in dataset:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            if input_text:
                text = f"{instruction}\n\nInput: {input_text}"
            else:
                text = instruction

            if text and is_clean_instruction(text):
                instructions.append(text.strip())

        logger.info(f"  Alpaca: {len(instructions)} clean instructions")
        return instructions, instructions[:5]

    except Exception as e:
        logger.warning(f"  Failed to download Alpaca: {e}")
        return [], []


def download_dolly() -> Tuple[List[str], List[str]]:
    """Download Databricks Dolly dataset (15k human-written)."""
    from datasets import load_dataset

    logger.info("Downloading Dolly dataset (15k)...")
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        instructions = []

        for example in dataset:
            instruction = example.get("instruction", "")
            context = example.get("context", "")
            if context:
                text = f"{instruction}\n\nContext: {context}"
            else:
                text = instruction

            if text and is_clean_instruction(text):
                instructions.append(text.strip())

        logger.info(f"  Dolly: {len(instructions)} clean instructions")
        return instructions, instructions[:5]

    except Exception as e:
        logger.warning(f"  Failed to download Dolly: {e}")
        return [], []


def download_natural_instructions() -> Tuple[List[str], List[str]]:
    """Download Super-Natural Instructions (task definitions)."""
    from datasets import load_dataset

    logger.info("Downloading Super-Natural Instructions...")
    instructions = []

    try:
        dataset = load_dataset(
            "Muennighoff/natural-instructions",
            split="train",
            streaming=True,
        )

        seen_definitions = set()
        count = 0
        for example in dataset:
            # Get task definition (unique per task)
            definition = example.get("definition", "")
            if definition and definition not in seen_definitions:
                seen_definitions.add(definition)
                if is_clean_instruction(definition):
                    instructions.append(definition.strip())

            count += 1
            if count % 100000 == 0:
                logger.info(f"  Processed {count} examples, {len(instructions)} unique task definitions")

        logger.info(f"  Natural Instructions: {len(instructions)} unique task definitions")
        return instructions, instructions[:5]

    except Exception as e:
        logger.warning(f"  Failed to download Natural Instructions: {e}")
        return [], []


def download_sharegpt() -> Tuple[List[str], List[str]]:
    """Download ShareGPT (real user conversations)."""
    from datasets import load_dataset

    logger.info("Downloading ShareGPT...")
    instructions = []

    try:
        dataset = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
        )

        for example in dataset:
            convs = example.get("conversations", [])
            if convs and len(convs) > 0:
                # First human turn is the instruction
                first = convs[0]
                if first.get("from") == "human":
                    text = first.get("value", "")
                    if text and is_clean_instruction(text):
                        instructions.append(text.strip())

        logger.info(f"  ShareGPT: {len(instructions)} clean instructions")
        return instructions, instructions[:5]

    except Exception as e:
        logger.warning(f"  Failed to download ShareGPT: {e}")
        return [], []


def deduplicate(instructions: List[str]) -> List[str]:
    """Remove near-duplicate instructions."""
    seen = set()
    unique = []

    for inst in instructions:
        normalized = inst.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(inst)

    return unique


def show_samples(samples_by_dataset: Dict[str, List[str]]):
    """Display sample instructions from each dataset."""
    print("\n" + "=" * 80)
    print("SAMPLE INSTRUCTIONS FROM EACH DATASET")
    print("=" * 80)

    for dataset_name, samples in samples_by_dataset.items():
        if not samples:
            continue

        print(f"\n{'─' * 40}")
        print(f"{dataset_name.upper()}")
        print(f"{'─' * 40}")

        for i, sample in enumerate(samples[:5], 1):
            display = sample[:250] + "..." if len(sample) > 250 else sample
            display = display.replace('\n', ' | ')
            print(f"\n{i}. {display}")

    print("\n" + "=" * 80)


def main():
    """Download and combine all instruction datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Download instruction datasets")
    parser.add_argument("--preview", action="store_true", help="Show samples only, don't save")
    args = parser.parse_args()

    logger.info("=== Downloading Universal Instruction Datasets ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_instructions = []
    samples_by_dataset = {}
    stats = {}

    # 1. WizardLM Evol-Instruct (196k evolved complex instructions)
    wizardlm_insts, wizardlm_samples = download_wizardlm()
    all_instructions.extend(wizardlm_insts)
    samples_by_dataset["WizardLM Evol-Instruct (196k)"] = wizardlm_samples
    stats["wizardlm"] = len(wizardlm_insts)

    # 2. Alpaca (52k)
    alpaca_insts, alpaca_samples = download_alpaca()
    all_instructions.extend(alpaca_insts)
    samples_by_dataset["Alpaca (52k)"] = alpaca_samples
    stats["alpaca"] = len(alpaca_insts)

    # 3. Dolly (15k human-written)
    dolly_insts, dolly_samples = download_dolly()
    all_instructions.extend(dolly_insts)
    samples_by_dataset["Dolly (15k)"] = dolly_samples
    stats["dolly"] = len(dolly_insts)

    # 4. Super-Natural Instructions (task definitions)
    ni_insts, ni_samples = download_natural_instructions()
    all_instructions.extend(ni_insts)
    samples_by_dataset["Natural Instructions"] = ni_samples
    stats["natural_instructions"] = len(ni_insts)

    # 5. ShareGPT (real user instructions)
    sharegpt_insts, sharegpt_samples = download_sharegpt()
    all_instructions.extend(sharegpt_insts)
    samples_by_dataset["ShareGPT"] = sharegpt_samples
    stats["sharegpt"] = len(sharegpt_insts)

    # Show samples from each dataset
    show_samples(samples_by_dataset)

    if args.preview:
        logger.info("\nPreview mode - not saving data")
        return

    # Process and save
    logger.info(f"\nTotal before dedup: {len(all_instructions):,}")

    unique_instructions = deduplicate(all_instructions)
    logger.info(f"Total after dedup: {len(unique_instructions):,}")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(unique_instructions)

    # Save
    output_path = DATA_DIR / "universal_instructions.json"
    stats["total_unique"] = len(unique_instructions)

    with open(output_path, "w") as f:
        json.dump({
            "instructions": unique_instructions,
            "sources": list(stats.keys())[:-1],
            "stats": stats,
        }, f, indent=2)

    logger.info(f"\nSaved to {output_path}")

    # Summary table
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    for name, count in stats.items():
        print(f"  {name:25s}: {count:>10,}")
    print("=" * 50)


if __name__ == "__main__":
    main()
