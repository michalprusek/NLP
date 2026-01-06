"""Fine-tune Vec2Text 512-token InversionModel on instruction data.

The 512-token model uses InversionModel architecture (not CorrectorEncoderModel).
Fine-tuning should fix the Unicode garbage character issue.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python -m lipo.vec2text_finetune.train_512
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Data - using noisy embeddings for VAE/WAE compatibility
    train_data: str = "lipo/vec2text_finetune/data_noisy/noisy_train.json"
    eval_data: str = "lipo/vec2text_finetune/data_noisy/noisy_eval.json"
    output_dir: str = "lipo/vec2text_finetune/checkpoints_512_noisy"

    # Training
    epochs: int = 3
    per_device_batch_size: int = 8  # Smaller batch - no gradient checkpointing
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5  # Conservative LR for fine-tuning
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Sequence - 512 tokens!
    max_length: int = 512

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = False  # InversionModel doesn't support this

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 20
    early_stopping_patience: int = 3


class InversionDataset(Dataset):
    """Dataset for InversionModel fine-tuning: embedding → text.

    InversionModel expects:
    - embedder_input_ids: tokenized input for embedder (GTR)
    - embedder_attention_mask: attention mask for embedder
    - labels: tokenized target text (decoder output)
    - frozen_embeddings: pre-computed embeddings (optional, we use this)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        embedder_tokenizer,
        max_length: int = 512,
    ):
        logger.info(f"Loading {data_path}...")
        with open(data_path) as f:
            self.examples = json.load(f)

        self.tokenizer = tokenizer  # T5 tokenizer for decoder
        self.embedder_tokenizer = embedder_tokenizer  # GTR tokenizer for embedder
        self.max_length = max_length
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["text"]

        # Tokenize for embedder (GTR uses T5 tokenizer internally)
        embedder_inputs = self.embedder_tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target text for decoder
        labels = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Pre-computed embedding
        embedding = torch.tensor(ex["embedding"], dtype=torch.float32)

        return {
            "embedder_input_ids": embedder_inputs["input_ids"].squeeze(0),
            "embedder_attention_mask": embedder_inputs["attention_mask"].squeeze(0),
            "frozen_embeddings": embedding,
            "labels": labels["input_ids"].squeeze(0),
        }


def load_pretrained_model(device: str = "cuda"):
    """Load pre-trained Vec2Text 512-token InversionModel."""
    logger.info("Downloading pre-trained 512-token model...")
    model_dir = snapshot_download("vec2text/gtr-512-noise-0.00001")

    logger.info("Loading config...")
    config = InversionConfig.from_pretrained(model_dir)
    # Ensure max_seq_length is set for 512 tokens
    config.max_seq_length = 512

    logger.info(f"Config: max_seq_length={config.max_seq_length}")

    logger.info("Loading model...")
    model = InversionModel(config)

    # Load weights - handle sharded safetensors
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        # Sharded model - load each shard
        logger.info("Loading sharded safetensors weights...")
        with open(index_path) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(model_dir, shard_file)
            logger.info(f"  Loading {shard_file}...")
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
    elif os.path.exists(single_path):
        state_dict = load_file(single_path)
    else:
        # Try pytorch format
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"Weight mismatch - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    logger.info(f"Model loaded on {device}")

    return model, config


class InversionTrainer(Trainer):
    """Custom trainer for InversionModel that handles all required inputs."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for InversionModel.

        InversionModel expects:
        - embedder_input_ids: tokenized input for embedder
        - embedder_attention_mask: attention mask for embedder
        - labels: tokenized target text (decoder output)
        - frozen_embeddings: pre-computed embeddings
        """
        # InversionModel forward pass with all required inputs
        outputs = model(
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
            frozen_embeddings=inputs["frozen_embeddings"],
            labels=inputs["labels"],
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    model, model_config = load_pretrained_model(device)

    # Load tokenizers
    # T5 tokenizer for decoder (output text)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # GTR uses T5 tokenizer internally for embedder
    embedder_tokenizer = T5Tokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    # Note: InversionModel doesn't support gradient checkpointing
    # Use smaller batch size instead

    # Create datasets
    train_dataset = InversionDataset(
        config.train_data,
        tokenizer,
        embedder_tokenizer,
        max_length=config.max_length,
    )
    eval_dataset = InversionDataset(
        config.eval_data,
        tokenizer,
        embedder_tokenizer,
        max_length=config.max_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important: keep frozen_embeddings
    )

    # Initialize trainer
    trainer = InversionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"  Train examples: {len(train_dataset)}")
    logger.info(f"  Eval examples: {len(eval_dataset)}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.per_device_batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.per_device_batch_size * config.gradient_accumulation_steps}")

    trainer.train()

    # Save final model
    final_path = Path(config.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
