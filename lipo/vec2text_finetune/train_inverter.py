"""Fine-tune Vec2Text Inverter on universal instructions.

The Inverter is the key component: embedding → text
Fine-tuning it on instruction data should significantly improve reconstruction.

Usage:
    # Multi-GPU (recommended)
    uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
        -m lipo.vec2text_finetune.train_inverter

    # Single GPU
    uv run python -m lipo.vec2text_finetune.train_inverter
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Data
    train_data: str = "lipo/vec2text_finetune/data/universal_train.json"
    eval_data: str = "lipo/vec2text_finetune/data/universal_eval.json"
    output_dir: str = "lipo/vec2text_finetune/checkpoints_inverter"

    # Training
    epochs: int = 5
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Sequence
    max_length: int = 128

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 20
    early_stopping_patience: int = 5


class InverterDataset(Dataset):
    """Dataset for Inverter fine-tuning: embedding → text."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
    ):
        logger.info(f"Loading {data_path}...")
        with open(data_path) as f:
            self.examples = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create dummy embedder input (single PAD token)
        # This is needed because InversionModel.forward() requires these
        # even when frozen_embeddings are provided
        self.dummy_input_ids = torch.tensor([tokenizer.pad_token_id], dtype=torch.long)
        self.dummy_attention_mask = torch.tensor([1], dtype=torch.long)

        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Input: embedding (768D)
        embedding = torch.tensor(example["embedding"], dtype=torch.float32)

        # Target: original text
        target = example["text"]

        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "embedder_input_ids": self.dummy_input_ids.clone(),
            "embedder_attention_mask": self.dummy_attention_mask.clone(),
            "frozen_embeddings": embedding,
            "labels": labels,
        }


class InverterDataCollator:
    """Collate embeddings and labels for Inverter training."""

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        embedder_input_ids = torch.stack([f["embedder_input_ids"] for f in features])
        embedder_attention_mask = torch.stack([f["embedder_attention_mask"] for f in features])
        embeddings = torch.stack([f["frozen_embeddings"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        return {
            "embedder_input_ids": embedder_input_ids,
            "embedder_attention_mask": embedder_attention_mask,
            "frozen_embeddings": embeddings,
            "labels": labels,
        }


class InverterTrainer(Trainer):
    """Custom trainer for InversionModel."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        embedder_input_ids = inputs["embedder_input_ids"]
        embedder_attention_mask = inputs["embedder_attention_mask"]
        embeddings = inputs["frozen_embeddings"]
        labels = inputs["labels"]

        # InversionModel forward pass
        outputs = model(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=embeddings,
            labels=labels,
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def load_inverter(device: str = "cuda"):
    """Load pre-trained InversionModel from ielabgroup."""
    from vec2text.models.config import InversionConfig
    from vec2text.models.inversion import InversionModel

    logger.info("Loading pre-trained Inverter from ielabgroup...")

    inv_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
    )
    inv_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    )

    model = InversionModel(inv_config)
    model.load_state_dict(load_file(inv_weights), strict=False)

    logger.info("Inverter loaded successfully")
    return model


def train(config: Config):
    """Main training function."""
    logger.info("=== Vec2Text Inverter Fine-tuning ===")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"Rank {local_rank}/{world_size}, GPUs: {torch.cuda.device_count()}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    # Load tokenizer (T5)
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)

    # Load model
    model = load_inverter()

    if config.gradient_checkpointing:
        if hasattr(model.encoder_decoder, 'gradient_checkpointing_enable'):
            model.encoder_decoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = InverterDataset(
        config.train_data,
        tokenizer,
        config.max_length,
    )
    eval_dataset = InverterDataset(
        config.eval_data,
        tokenizer,
        config.max_length,
    )

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Data collator
    data_collator = InverterDataCollator()

    # Calculate steps
    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps * world_size
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * config.epochs

    logger.info(f"Effective batch: {effective_batch}")
    logger.info(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False,  # Important for custom inputs
    )

    # Trainer
    trainer = InverterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save
    if local_rank == 0:
        logger.info("Saving final model...")
        final_path = output_dir / "final"
        final_path.mkdir(exist_ok=True)

        # Save model state dict
        torch.save(model.state_dict(), final_path / "pytorch_model.bin")

        # Save config
        model.config.save_pretrained(str(final_path))

        # Save tokenizer
        tokenizer.save_pretrained(str(final_path))

        # Metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        logger.info(f"Done! Model saved to {final_path}")
        logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="lipo/vec2text_finetune/data/universal_train.json")
    parser.add_argument("--eval-data", default="lipo/vec2text_finetune/data/universal_eval.json")
    parser.add_argument("--output-dir", default="lipo/vec2text_finetune/checkpoints_inverter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=200)
    args = parser.parse_args()

    config = Config(
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
    )

    train(config)


if __name__ == "__main__":
    main()
