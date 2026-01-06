"""Fine-tune Vec2Text Corrector on noisy embeddings.

The Corrector refines initial hypotheses from the Inverter.
Fine-tuning it on noisy embeddings should improve reconstruction
when VAE-decoded embeddings are used.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m lipo.vec2text_finetune.train_corrector
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
    train_data: str = "lipo/vec2text_finetune/data_noisy/noisy_train.json"
    eval_data: str = "lipo/vec2text_finetune/data_noisy/noisy_eval.json"
    output_dir: str = "lipo/vec2text_finetune/checkpoints_corrector_noisy"

    # Training
    epochs: int = 3
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
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


class CorrectorDataset(Dataset):
    """Dataset for Corrector fine-tuning using pre-computed hypotheses."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        embedder_tokenizer,
        inversion_model,
        corrector_embedder,
        max_length: int = 128,
        device: str = "cuda",
    ):
        logger.info(f"Loading {data_path}...")
        with open(data_path) as f:
            self.examples = json.load(f)

        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.inversion_model = inversion_model
        self.corrector_embedder = corrector_embedder
        self.max_length = max_length
        self.device = device

        # Pre-compute all hypotheses (slow but necessary)
        logger.info("Pre-computing hypotheses (this may take a while)...")
        self._precompute_hypotheses()
        logger.info(f"Loaded {len(self.examples)} examples with hypotheses")

    def _precompute_hypotheses(self):
        """Pre-compute hypotheses for all examples."""
        self.hypotheses = []
        self.hypothesis_embeddings = []

        batch_size = 32
        for i in range(0, len(self.examples), batch_size):
            batch = self.examples[i:i + batch_size]
            embeddings = torch.tensor(
                [ex["embedding"] for ex in batch],
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                # Generate hypotheses
                hypothesis_ids = self.inversion_model.generate(
                    inputs={"frozen_embeddings": embeddings},
                    generation_kwargs={
                        "num_beams": 1,
                        "max_length": self.max_length,
                        "do_sample": False,
                    },
                )

                # Embed hypotheses using SentenceTransformer's encode method
                hypothesis_texts = self.tokenizer.batch_decode(
                    hypothesis_ids, skip_special_tokens=True
                )

                # SentenceTransformer uses .encode() method, not forward()
                hypothesis_embs = self.corrector_embedder.encode(
                    hypothesis_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )

            for j, text in enumerate(hypothesis_texts):
                self.hypotheses.append({
                    "text": text,
                    "input_ids": hypothesis_ids[j].cpu(),
                })
                self.hypothesis_embeddings.append(hypothesis_embs[j].cpu())

            if (i + batch_size) % 1000 == 0:
                logger.info(f"  Processed {i + batch_size}/{len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        hypothesis = self.hypotheses[idx]
        hypothesis_embedding = self.hypothesis_embeddings[idx]

        # Noisy embedding
        embedding = torch.tensor(example["embedding"], dtype=torch.float32)

        # Target text
        target = example["text"]

        # Tokenize hypothesis for corrector input
        hypothesis_tokens = self.embedder_tokenizer(
            hypothesis["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

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
            "frozen_embeddings": embedding,
            "hypothesis_input_ids": hypothesis_tokens["input_ids"].squeeze(),
            "hypothesis_attention_mask": hypothesis_tokens["attention_mask"].squeeze(),
            "hypothesis_embedding": hypothesis_embedding,
            "labels": labels,
        }


class CorrectorDataCollator:
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "frozen_embeddings": torch.stack([f["frozen_embeddings"] for f in features]),
            "hypothesis_input_ids": torch.stack([f["hypothesis_input_ids"] for f in features]),
            "hypothesis_attention_mask": torch.stack([f["hypothesis_attention_mask"] for f in features]),
            "hypothesis_embedding": torch.stack([f["hypothesis_embedding"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


class CorrectorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        frozen_embeddings = inputs["frozen_embeddings"]
        hypothesis_input_ids = inputs["hypothesis_input_ids"]
        hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
        hypothesis_embedding = inputs["hypothesis_embedding"]
        labels = inputs["labels"]

        inputs_embeds, attention_mask = model.get_encoder_embedding(
            embedding=frozen_embeddings,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
            hypothesis_embedding=hypothesis_embedding,
        )

        outputs = model.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss


def load_models(device: str = "cuda"):
    """Load pre-trained models."""
    import vec2text
    from vec2text.models.config import InversionConfig
    from vec2text.models.inversion import InversionModel
    from vec2text.models.corrector_encoder import CorrectorEncoderModel

    logger.info("Loading models...")

    # Inversion model
    inv_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
    )
    inv_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    )
    inversion_model = InversionModel(inv_config)
    inversion_model.load_state_dict(load_file(inv_weights), strict=False)
    inversion_model = inversion_model.to(device).eval()

    # Corrector model
    corr_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_corrector", "model.safetensors"
    )
    corr_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_corrector"
    )
    corrector_model = CorrectorEncoderModel(corr_config)
    corrector_model.load_state_dict(load_file(corr_weights), strict=False)
    corrector_model = corrector_model.to(device)

    # Load corrector for embedder access
    corrector = vec2text.load_corrector(inversion_model, corrector_model)

    logger.info("Models loaded")
    return inversion_model, corrector_model, corrector


def train(config: Config):
    logger.info("=== Vec2Text Corrector Fine-tuning ===")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    # Load models
    inversion_model, corrector_model, corrector = load_models()

    tokenizer = corrector.tokenizer
    embedder_tokenizer = corrector.embedder_tokenizer
    corrector_embedder = corrector.embedder

    if config.gradient_checkpointing:
        if hasattr(corrector_model.encoder_decoder, 'gradient_checkpointing_enable'):
            corrector_model.encoder_decoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Datasets
    logger.info("Creating datasets...")
    train_dataset = CorrectorDataset(
        config.train_data,
        tokenizer,
        embedder_tokenizer,
        inversion_model,
        corrector_embedder,
        config.max_length,
    )
    eval_dataset = CorrectorDataset(
        config.eval_data,
        tokenizer,
        embedder_tokenizer,
        inversion_model,
        corrector_embedder,
        config.max_length,
    )

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    data_collator = CorrectorDataCollator()

    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch
    logger.info(f"Effective batch: {effective_batch}, Steps/epoch: {steps_per_epoch}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
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
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False,
    )

    trainer = CorrectorTrainer(
        model=corrector_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Saving final model...")
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)

    torch.save(corrector_model.state_dict(), final_path / "pytorch_model.bin")
    corrector_model.config.save_pretrained(str(final_path))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"Done! Saved to {final_path}")
    logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="lipo/vec2text_finetune/data_noisy/noisy_train.json")
    parser.add_argument("--eval-data", default="lipo/vec2text_finetune/data_noisy/noisy_eval.json")
    parser.add_argument("--output-dir", default="lipo/vec2text_finetune/checkpoints_corrector_noisy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-steps", type=int, default=500)
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
