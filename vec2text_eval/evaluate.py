#!/usr/bin/env python3
"""Vec2Text inversion quality evaluation.

Evaluates Vec2Text reconstruction quality on 100 diverse test strings.
Computes metrics: exact match, BLEU, cosine similarity, CER, token accuracy.

Usage:
    uv run python -m vec2text_eval.evaluate
    uv run python -m vec2text_eval.evaluate --num-samples 50 --num-steps 20
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text_eval.sample_texts import SAMPLE_TEXTS, get_category, CATEGORIES
from vec2text_eval.metrics import (
    compute_all_metrics,
    aggregate_metrics,
    cosine_similarity,
)


def load_encoder():
    """Load GTR encoder from generative_hbbops (now uses SentenceTransformer internally)."""
    from generative_hbbops.encoder import GTREncoder
    print("Loading GTR encoder...")
    encoder = GTREncoder()
    print(f"  Model: {encoder.model_name}")
    print(f"  Device: {encoder.device}")
    return encoder


def load_inverter(num_steps: int = 50, beam_width: int = 4):
    """Load Vec2Text inverter from generative_hbbops."""
    from generative_hbbops.inverter import Vec2TextInverter
    print(f"Loading Vec2Text inverter (steps={num_steps}, beam={beam_width})...")
    inverter = Vec2TextInverter(
        num_steps=num_steps,
        beam_width=beam_width
    )
    return inverter


def evaluate_sample(
    text: str,
    encoder,
    inverter,
    verbose: bool = False
) -> dict:
    """Evaluate Vec2Text on a single sample.

    Args:
        text: Original text to embed and reconstruct
        encoder: GTREncoder instance
        inverter: Vec2TextInverter instance
        verbose: Print progress

    Returns:
        Dictionary with original, reconstructed, and all metrics
    """
    # Embed original text using GTREncoder
    original_embedding = encoder.encode_tensor(text)

    # Invert embedding back to text
    reconstructed = inverter.invert(original_embedding)

    # Re-embed reconstructed text for cosine similarity
    reconstructed_embedding = encoder.encode_tensor(reconstructed)

    # Convert to numpy for metrics
    orig_np = original_embedding.cpu().numpy()
    recon_np = reconstructed_embedding.cpu().numpy()

    # Compute all metrics
    metrics = compute_all_metrics(
        original=text,
        reconstructed=reconstructed,
        original_embedding=orig_np,
        reconstructed_embedding=recon_np
    )

    result = {
        "original": text,
        "reconstructed": reconstructed,
        **metrics
    }

    if verbose:
        print(f"  Original:      {text[:60]}...")
        print(f"  Reconstructed: {reconstructed[:60]}...")
        print(f"  Exact match: {metrics['exact_match']:.0f}, "
              f"BLEU: {metrics['bleu']:.3f}, "
              f"Cosine: {metrics['cosine_similarity']:.3f}")

    return result


def run_evaluation(
    num_samples: int = 100,
    num_steps: int = 50,
    beam_width: int = 4,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """Run full evaluation on test samples.

    Args:
        num_samples: Number of samples to evaluate (max 100)
        num_steps: Vec2Text correction steps
        beam_width: Vec2Text beam search width
        output_path: Path to save JSON results
        verbose: Print progress

    Returns:
        Dictionary with all results and aggregated metrics
    """
    # Limit to available samples
    num_samples = min(num_samples, len(SAMPLE_TEXTS))
    samples = SAMPLE_TEXTS[:num_samples]

    print(f"\n{'='*60}")
    print("Vec2Text Inversion Evaluation")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}")
    print(f"Steps: {num_steps}, Beam width: {beam_width}")
    print(f"{'='*60}\n")

    # Load models
    encoder = load_encoder()
    inverter = load_inverter(num_steps=num_steps, beam_width=beam_width)

    # Evaluate each sample
    results = []
    start_time = time.time()

    for i, text in enumerate(samples):
        if verbose:
            category = get_category(i)
            print(f"\n[{i+1}/{num_samples}] Category: {category}")

        result = evaluate_sample(text, encoder, inverter, verbose=verbose)
        result["index"] = i
        result["category"] = get_category(i)
        results.append(result)

    elapsed = time.time() - start_time

    # Aggregate metrics
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")

    aggregated = aggregate_metrics(results)
    for metric, stats in aggregated.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f} +/- {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Per-category breakdown
    print(f"\n{'='*60}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*60}")

    for cat_name, (start, end) in CATEGORIES.items():
        cat_results = [r for r in results if r["category"] == cat_name]
        if cat_results:
            cat_agg = aggregate_metrics(cat_results)
            em = cat_agg.get("exact_match", {}).get("mean", 0)
            bleu = cat_agg.get("bleu", {}).get("mean", 0)
            cos = cat_agg.get("cosine_similarity", {}).get("mean", 0)
            print(f"{cat_name:12} | EM: {em:.2f} | BLEU: {bleu:.3f} | Cos: {cos:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    exact_matches = sum(1 for r in results if r["exact_match"] == 1.0)
    print(f"Exact matches: {exact_matches}/{num_samples} ({100*exact_matches/num_samples:.1f}%)")
    print(f"Mean BLEU: {aggregated['bleu']['mean']:.4f}")
    print(f"Mean cosine similarity: {aggregated['cosine_similarity']['mean']:.4f}")
    print(f"Mean CER: {aggregated['cer']['mean']:.4f}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/num_samples:.2f}s per sample)")

    # Prepare output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "num_steps": num_steps,
            "beam_width": beam_width,
            "elapsed_seconds": elapsed,
        },
        "aggregated": aggregated,
        "per_category": {
            cat: aggregate_metrics([r for r in results if r["category"] == cat])
            for cat in CATEGORIES.keys()
        },
        "samples": results,
    }

    # Save results
    if output_path:
        output_file = Path(output_path)
    else:
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Vec2Text inversion quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Number of samples to evaluate (max 100)"
    )
    parser.add_argument(
        "--num-steps", "-s",
        type=int,
        default=50,
        help="Vec2Text correction steps"
    )
    parser.add_argument(
        "--beam-width", "-b",
        type=int,
        default=4,
        help="Beam search width"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

    run_evaluation(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        beam_width=args.beam_width,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
