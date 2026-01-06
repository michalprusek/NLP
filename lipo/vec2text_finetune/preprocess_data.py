"""Pre-process data into efficient Arrow format for fast training.

This converts JSON to HuggingFace Dataset with pre-tokenization.

Usage:
    uv run python -m lipo.vec2text_finetune.preprocess_data
"""

import json
import logging
from pathlib import Path
from transformers import T5Tokenizer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess_and_save(
    input_path: str,
    output_dir: str,
    max_length: int = 128,
    prefix: str = "Correct: ",
):
    """Convert JSON to pre-tokenized Arrow dataset."""
    logger.info(f"Loading {input_path}...")
    with open(input_path) as f:
        examples = json.load(f)

    logger.info(f"Loaded {len(examples)} examples")

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)

    # Prepare data for HF Dataset
    sources = [prefix + ex["hypothesis"] for ex in examples]
    targets = [ex["text"] for ex in examples]

    logger.info("Creating HuggingFace Dataset...")
    dataset = Dataset.from_dict({
        "source": sources,
        "target": targets,
    })

    def tokenize_fn(batch):
        # Tokenize source
        model_inputs = tokenizer(
            batch["source"],
            max_length=max_length,
            truncation=True,
            # No padding - will be done dynamically
        )

        # Tokenize targets
        labels = tokenizer(
            batch["target"],
            max_length=max_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        # Store lengths for efficient batching
        model_inputs["length"] = [len(ids) for ids in model_inputs["input_ids"]]

        return model_inputs

    logger.info("Tokenizing (batched)...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,  # Parallel tokenization
        remove_columns=["source", "target"],
        desc="Tokenizing",
    )

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_path}...")
    tokenized.save_to_disk(str(output_path))

    logger.info(f"Done! Saved {len(tokenized)} examples")
    logger.info(f"Columns: {tokenized.column_names}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="lipo/vec2text_finetune/data/universal_train.json")
    parser.add_argument("--eval-input", default="lipo/vec2text_finetune/data/universal_eval.json")
    parser.add_argument("--output-dir", default="lipo/vec2text_finetune/data/tokenized")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # Process train
    logger.info("=== Processing Training Data ===")
    preprocess_and_save(
        args.train_input,
        f"{args.output_dir}/train",
        args.max_length,
    )

    # Process eval
    logger.info("\n=== Processing Eval Data ===")
    preprocess_and_save(
        args.eval_input,
        f"{args.output_dir}/eval",
        args.max_length,
    )


if __name__ == "__main__":
    main()
