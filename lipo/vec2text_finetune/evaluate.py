"""
Evaluate Vec2Text round-trip quality.

This script evaluates the fine-tuned corrector model by measuring
the cosine similarity between original and reconstructed embeddings.

Usage:
    uv run python -m lipo.vec2text_finetune.evaluate \
        --model-path lipo/vec2text_finetune/checkpoints/final \
        --test-data lipo/vec2text_finetune/data/eval.json
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a single reconstruction evaluation."""
    original_text: str
    hypothesis: str
    reconstructed: str
    cosine_similarity: float
    exact_match: bool


def load_finetuned_model(
    model_path: str,
    device: str = "cuda",
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Load fine-tuned T5 model and tokenizer."""
    logger.info(f"Loading fine-tuned model from {model_path}")

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    return model, tokenizer


def load_original_vec2text(device: str = "cuda"):
    """Load original Vec2Text corrector for comparison."""
    import vec2text
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from vec2text.models.config import InversionConfig
    from vec2text.models.inversion import InversionModel
    from vec2text.models.corrector_encoder import CorrectorEncoderModel

    logger.info("Loading original Vec2Text corrector...")

    # Load inversion model
    inv_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
    )
    inv_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    )
    inversion_model = InversionModel(inv_config)
    inversion_model.load_state_dict(load_file(inv_weights), strict=False)

    # Load corrector model
    corr_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_corrector", "model.safetensors"
    )
    corr_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_corrector"
    )
    corrector_model = CorrectorEncoderModel(corr_config)
    corrector_model.load_state_dict(load_file(corr_weights), strict=False)

    # Create corrector
    corrector = vec2text.load_corrector(inversion_model, corrector_model)
    corrector.inversion_trainer.model = corrector.inversion_trainer.model.to(device)
    corrector.model = corrector.model.to(device)

    return corrector


def generate_with_finetuned(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    hypothesis: str,
    max_length: int = 128,
    num_beams: int = 4,
) -> str:
    """Generate corrected text using fine-tuned model."""
    # Prepare input
    input_text = f"Correct: {hypothesis}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()


def evaluate_model(
    model_path: str,
    test_data_path: str,
    device: str = "cuda",
    num_samples: Optional[int] = None,
    compare_original: bool = True,
) -> Tuple[Dict, List[EvaluationResult], List[EvaluationResult]]:
    """Evaluate fine-tuned model on test data.

    Returns dict with metrics for fine-tuned and optionally original model.
    """
    # Load test data
    with open(test_data_path) as f:
        test_data = json.load(f)

    if num_samples:
        test_data = test_data[:num_samples]

    logger.info(f"Evaluating on {len(test_data)} samples")

    # Load GTR encoder for computing embeddings
    logger.info("Loading GTR encoder...")
    gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device=device)

    # Load fine-tuned model
    finetuned_model, tokenizer = load_finetuned_model(model_path, device)

    # Load original Vec2Text if comparing
    original_corrector = None
    if compare_original:
        try:
            original_corrector = load_original_vec2text(device)
        except Exception as e:
            logger.warning(f"Could not load original Vec2Text: {e}")
            compare_original = False

    # Evaluate
    finetuned_results = []
    original_results = []

    for example in tqdm(test_data, desc="Evaluating"):
        original_text = example["text"]
        hypothesis = example["hypothesis"]
        original_embedding = torch.tensor(example["embedding"], device=device)

        # Generate with fine-tuned model
        reconstructed_finetuned = generate_with_finetuned(
            finetuned_model, tokenizer, hypothesis
        )

        # Compute embedding of reconstruction
        recon_embedding_finetuned = torch.tensor(
            gtr.encode(reconstructed_finetuned, normalize_embeddings=True),
            device=device,
        )

        # Compute cosine similarity
        cosine_finetuned = F.cosine_similarity(
            original_embedding.unsqueeze(0),
            recon_embedding_finetuned.unsqueeze(0),
        ).item()

        finetuned_results.append(EvaluationResult(
            original_text=original_text,
            hypothesis=hypothesis,
            reconstructed=reconstructed_finetuned,
            cosine_similarity=cosine_finetuned,
            exact_match=reconstructed_finetuned.strip() == original_text.strip(),
        ))

        # Evaluate original Vec2Text if available
        if compare_original and original_corrector:
            import vec2text
            reconstructed_original = vec2text.invert_embeddings(
                embeddings=original_embedding.unsqueeze(0),
                corrector=original_corrector,
                num_steps=20,
            )[0]

            recon_embedding_original = torch.tensor(
                gtr.encode(reconstructed_original, normalize_embeddings=True),
                device=device,
            )

            cosine_original = F.cosine_similarity(
                original_embedding.unsqueeze(0),
                recon_embedding_original.unsqueeze(0),
            ).item()

            original_results.append(EvaluationResult(
                original_text=original_text,
                hypothesis=hypothesis,
                reconstructed=reconstructed_original,
                cosine_similarity=cosine_original,
                exact_match=reconstructed_original.strip() == original_text.strip(),
            ))

    # Compute aggregate metrics
    def compute_metrics(results: List[EvaluationResult]) -> Dict:
        if not results:
            return {}

        cosines = [r.cosine_similarity for r in results]
        return {
            "mean_cosine": sum(cosines) / len(cosines),
            "min_cosine": min(cosines),
            "max_cosine": max(cosines),
            "below_90": sum(1 for c in cosines if c < 0.90) / len(cosines),
            "below_95": sum(1 for c in cosines if c < 0.95) / len(cosines),
            "exact_match_rate": sum(1 for r in results if r.exact_match) / len(results),
            "n_samples": len(results),
        }

    metrics = {
        "finetuned": compute_metrics(finetuned_results),
    }
    if compare_original and original_results:
        metrics["original"] = compute_metrics(original_results)
        metrics["improvement"] = {
            "cosine_delta": metrics["finetuned"]["mean_cosine"] - metrics["original"]["mean_cosine"],
            "below_90_reduction": metrics["original"]["below_90"] - metrics["finetuned"]["below_90"],
        }

    return metrics, finetuned_results, original_results


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Vec2Text fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="lipo/vec2text_finetune/checkpoints/final",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="lipo/vec2text_finetune/data/eval.json",
        help="Path to test data JSON",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--no-compare-original",
        action="store_true",
        help="Don't compare with original Vec2Text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    args = parser.parse_args()

    # Run evaluation
    metrics, finetuned_results, original_results = evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        device=args.device,
        num_samples=args.num_samples,
        compare_original=not args.no_compare_original,
    )

    # Print results
    logger.info("\n=== Evaluation Results ===")
    logger.info("\nFine-tuned model:")
    for k, v in metrics["finetuned"].items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if "original" in metrics:
        logger.info("\nOriginal Vec2Text:")
        for k, v in metrics["original"].items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        logger.info("\nImprovement:")
        for k, v in metrics["improvement"].items():
            logger.info(f"  {k}: {v:+.4f}")

    # Show worst reconstructions
    logger.info("\n=== Worst Reconstructions (Fine-tuned) ===")
    worst = sorted(finetuned_results, key=lambda x: x.cosine_similarity)[:3]
    for i, r in enumerate(worst):
        logger.info(f"\n{i+1}. Cosine: {r.cosine_similarity:.4f}")
        logger.info(f"   Original:\n{r.original_text}")
        logger.info(f"   Reconstructed:\n{r.reconstructed}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "finetuned_samples": [
                    {
                        "original": r.original_text,
                        "reconstructed": r.reconstructed,
                        "cosine": r.cosine_similarity,
                    }
                    for r in finetuned_results
                ],
            }, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
