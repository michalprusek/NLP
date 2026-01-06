"""FAST Vec2Text corrector training with accelerate.

Optimizations:
- Pre-tokenized Arrow dataset (no on-the-fly tokenization)
- DistributedDataParallel via accelerate
- Dynamic padding (no wasted compute on padding)
- bf16 mixed precision
- torch.compile for faster training

Usage:
    # Multi-GPU (recommended)
    uv run accelerate launch --multi_gpu --num_processes=2 -m lipo.vec2text_finetune.train_fast

    # Single GPU
    uv run python -m lipo.vec2text_finetune.train_fast
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Data (pre-tokenized Arrow format)
    train_data: str = "lipo/vec2text_finetune/data/tokenized/train"
    eval_data: str = "lipo/vec2text_finetune/data/tokenized/eval"
    output_dir: str = "lipo/vec2text_finetune/checkpoints"

    # Training
    epochs: int = 3
    per_device_batch_size: int = 64  # Larger with dynamic padding
    gradient_accumulation_steps: int = 1  # Less needed with larger batch
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Precision & Speed
    bf16: bool = True
    gradient_checkpointing: bool = True
    torch_compile: bool = False

    # Evaluation
    eval_steps: int = 200
    save_steps: int = 400
    logging_steps: int = 20
    early_stopping_patience: int = 3


def load_corrector():
    """Load ielabgroup corrector's T5 encoder-decoder."""
    logger.info("Loading pre-trained corrector from ielabgroup...")

    try:
        from vec2text.models.config import InversionConfig
        from vec2text.models.corrector_encoder import CorrectorEncoderModel

        corr_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_corrector", "model.safetensors"
        )
        corr_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_corrector"
        )
        corrector = CorrectorEncoderModel(corr_config)
        corrector.load_state_dict(load_file(corr_weights), strict=False)

        model = corrector.encoder_decoder
        logger.info("Extracted T5 from corrector")
        return model

    except Exception as e:
        logger.warning(f"Could not load corrector: {e}")
        logger.info("Falling back to t5-base")
        return T5ForConditionalGeneration.from_pretrained("t5-base")


def train(config: Config):
    """Main training with all optimizations."""
    logger.info("=== Vec2Text FAST Training ===")

    # Detect distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"Rank {local_rank}/{world_size}, GPUs: {torch.cuda.device_count()}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)

    # Load model
    model = load_corrector()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Compile model for speed (PyTorch 2.0+)
    if config.torch_compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Load pre-tokenized datasets
    logger.info("Loading pre-tokenized datasets...")
    train_dataset = load_from_disk(config.train_data)
    eval_dataset = load_from_disk(config.eval_data)

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Data collator with dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if config.bf16 else None,
    )

    # Calculate steps
    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps * world_size
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * config.epochs

    logger.info(f"Effective batch: {effective_batch}")
    logger.info(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=False,  # Faster eval without generation
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        length_column_name="length",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        # Speed optimizations
        dataloader_prefetch_factor=2,
        remove_unused_columns=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save
    if local_rank == 0:
        logger.info("Saving final model...")
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

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
    parser.add_argument("--train-data", default="lipo/vec2text_finetune/data/tokenized/train")
    parser.add_argument("--eval-data", default="lipo/vec2text_finetune/data/tokenized/eval")
    parser.add_argument("--output-dir", default="lipo/vec2text_finetune/checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    config = Config(
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        torch_compile=args.torch_compile,
    )

    train(config)


if __name__ == "__main__":
    main()
