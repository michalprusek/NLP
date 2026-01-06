"""
Prepare APE instructions for Vec2Text corrector fine-tuning.

Usage:
    uv run python -m lipo.vec2text_finetune.prepare_data

Output:
    lipo/vec2text_finetune/data/train.json
    lipo/vec2text_finetune/data/eval.json
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example for corrector fine-tuning."""
    text: str  # Original instruction (target)
    embedding: List[float]  # GTR embedding (768D)
    hypothesis: str  # Inversion model output (input to corrector)


def is_clean_instruction(text: str, min_length: int = 20, max_length: int = 500) -> bool:
    """Filter garbage instructions.

    Returns True if instruction is clean and suitable for training.
    """
    if not text or not isinstance(text, str):
        return False

    text = text.strip()

    # Length constraints
    if len(text) < min_length or len(text) > max_length:
        return False

    # Non-ASCII heavy (Chinese, Russian, etc.)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.85:
        return False

    # Unicode garbage patterns
    garbage_pattern = r'[»«â€¢™®©†‡°±²³µ¶·¹º¼½¾¿×÷]'
    if re.search(garbage_pattern, text):
        return False

    # Broken/incomplete sentences (too many ...)
    if text.count('...') > 5:
        return False

    # Too many brackets (malformed)
    if text.count('[') > 3 or text.count(']') > 3:
        return False

    # Starts with code-like patterns
    if re.match(r'^def\s+\w+\(', text) or re.match(r'^class\s+\w+', text):
        return False

    # Mostly uppercase (COMMANDER style)
    upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    if upper_ratio > 0.5:
        return False

    # Check for actual sentence structure
    # Should contain at least some words
    words = text.split()
    if len(words) < 5:
        return False

    # Should not be just punctuation/symbols
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.5:
        return False

    return True


def load_and_filter_instructions(
    ape_path: str = "lipo/data/ape_instructions.json",
) -> List[str]:
    """Load APE instructions and filter out garbage."""
    with open(ape_path) as f:
        data = json.load(f)

    instructions = data.get("instructions", [])
    logger.info(f"Loaded {len(instructions)} raw instructions")

    clean = [inst for inst in instructions if is_clean_instruction(inst)]
    logger.info(f"After filtering: {len(clean)} clean instructions")

    return clean


def generate_hypotheses(
    instructions: List[str],
    batch_size: int = 16,
    device: str = "cuda",
) -> List[TrainingExample]:
    """Generate embeddings and hypotheses for all instructions.

    For each instruction:
    1. Encode with GTR -> embedding
    2. Generate hypothesis with InversionModel -> hypothesis
    3. Return (text, embedding, hypothesis) tuple
    """
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from vec2text.models.config import InversionConfig
    from vec2text.models.inversion import InversionModel

    logger.info("Loading GTR encoder...")
    gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device=device)

    logger.info("Loading InversionModel...")
    # Load inversion model (same as in lipo/inference.py)
    inv_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
    )
    inv_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    )
    inversion_model = InversionModel(inv_config)
    inversion_model.load_state_dict(load_file(inv_weights), strict=False)
    inversion_model = inversion_model.to(device)
    inversion_model.eval()

    examples = []

    logger.info(f"Processing {len(instructions)} instructions...")
    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_texts = instructions[i:i + batch_size]

        # Encode with GTR
        embeddings = gtr.encode(
            batch_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Generate hypotheses with InversionModel
        with torch.no_grad():
            # Prepare input for InversionModel
            hypotheses = []
            for emb in embeddings:
                # InversionModel expects (batch, dim) tensor
                emb_input = emb.unsqueeze(0)

                # Generate with beam search
                gen_kwargs = {
                    "num_beams": 4,
                    "max_length": 64,
                    "no_repeat_ngram_size": 3,
                }

                output_ids = inversion_model.generate(
                    inputs={"frozen_embeddings": emb_input},
                    generation_kwargs=gen_kwargs,
                )

                hypothesis = inversion_model.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                hypotheses.append(hypothesis.strip())

        # Create training examples
        for text, emb, hyp in zip(batch_texts, embeddings, hypotheses):
            examples.append(TrainingExample(
                text=text,
                embedding=emb.cpu().tolist(),
                hypothesis=hyp,
            ))

    return examples


def split_data(
    examples: List[TrainingExample],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Split examples into train/eval sets."""
    import random
    random.seed(seed)

    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    return train, eval_data


def save_data(
    train: List[TrainingExample],
    eval_data: List[TrainingExample],
    output_dir: str = "lipo/vec2text_finetune/data",
):
    """Save training and evaluation data to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.json"
    eval_path = output_path / "eval.json"

    with open(train_path, "w") as f:
        json.dump([asdict(ex) for ex in train], f, indent=2)

    with open(eval_path, "w") as f:
        json.dump([asdict(ex) for ex in eval_data], f, indent=2)

    logger.info(f"Saved {len(train)} training examples to {train_path}")
    logger.info(f"Saved {len(eval_data)} evaluation examples to {eval_path}")


def main():
    """Main data preparation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Vec2Text training data")
    parser.add_argument(
        "--ape-path",
        type=str,
        default="lipo/data/ape_instructions.json",
        help="Path to APE instructions JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lipo/vec2text_finetune/data",
        help="Output directory for train/eval data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference",
    )

    args = parser.parse_args()

    # Step 1: Load and filter instructions
    logger.info("=== Step 1: Loading and filtering instructions ===")
    instructions = load_and_filter_instructions(args.ape_path)

    if len(instructions) == 0:
        logger.error("No clean instructions found!")
        return

    # Step 2: Generate embeddings and hypotheses
    logger.info("=== Step 2: Generating embeddings and hypotheses ===")
    examples = generate_hypotheses(
        instructions,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Step 3: Split data
    logger.info("=== Step 3: Splitting data ===")
    train, eval_data = split_data(examples, train_ratio=args.train_ratio)

    # Step 4: Save data
    logger.info("=== Step 4: Saving data ===")
    save_data(train, eval_data, args.output_dir)

    # Summary
    logger.info("=== Summary ===")
    logger.info(f"Total instructions: {len(instructions)}")
    logger.info(f"Training examples: {len(train)}")
    logger.info(f"Evaluation examples: {len(eval_data)}")

    # Show sample
    if train:
        sample = train[0]
        logger.info(f"\nSample training example:")
        logger.info(f"  Text:\n{sample.text}")
        logger.info(f"  Hypothesis:\n{sample.hypothesis}")


if __name__ == "__main__":
    main()
