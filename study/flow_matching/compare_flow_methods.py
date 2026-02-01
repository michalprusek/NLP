"""Compare all flow matching methods.

Loads checkpoints for I-CFM, OT-CFM, Reflow, and SI-GVP models and computes:
- Distribution MSE (how close generated samples are to target distribution)
- Path straightness (deviation from straight-line trajectories)
- Text generation quality (sample coherence)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m study.flow_matching.compare_flow_methods

    # Or with custom checkpoint directory
    CUDA_VISIBLE_DEVICES=0 uv run python -m study.flow_matching.compare_flow_methods \
        --checkpoint-dir study/checkpoints
"""

import argparse
import logging
import os
import pickle
from typing import Optional

import torch

from study.flow_matching.evaluate import (
    load_checkpoint,
    compute_distribution_mse,
    compute_path_straightness,
    generate_and_decode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default checkpoint paths for each method
CHECKPOINTS = {
    "I-CFM": "study/checkpoints/mlp-icfm-1k-none/best.pt",
    "OT-CFM": "study/checkpoints/mlp-otcfm-1k-none/best.pt",
    "Reflow": "study/checkpoints/mlp-reflow-1k-none/best.pt",
    "SI-GVP": "study/checkpoints/mlp-si-gvp-1k-none/best.pt",
}


def compare_methods(
    checkpoint_dir: str = "study/checkpoints",
    n_samples: int = 100,
    n_steps: int = 100,
    device: str = "cuda:0",
    generate_text: bool = True,
    text_samples: int = 2,
) -> dict:
    """Compare all flow matching methods.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        n_samples: Number of samples for metrics.
        n_steps: ODE integration steps.
        device: Device for computation.
        generate_text: Whether to generate text samples.
        text_samples: Number of text samples to generate per method.

    Returns:
        Dictionary with results for each method.
    """
    # Load test data
    test_path = "study/datasets/splits/1k/test.pt"
    try:
        test_data = torch.load(test_path, weights_only=False)
        test_embeddings = test_data["embeddings"]
    except FileNotFoundError:
        logger.error(f"Test data not found: {test_path}. Run data preparation first.")
        raise
    except KeyError as e:
        logger.error(f"Test data missing required key {e}. Check data format in {test_path}")
        raise
    except (RuntimeError, pickle.UnpicklingError) as e:
        logger.error(f"Failed to load test data (corrupted or incompatible): {e}")
        raise
    logger.info(f"Loaded {len(test_embeddings)} test embeddings")

    # Initialize decoder if generating text
    decoder = None
    if generate_text:
        try:
            from ecoflow.decoder import SonarDecoder
            decoder = SonarDecoder(device=device)
        except ImportError as e:
            logger.error(f"SonarDecoder unavailable - missing dependencies: {e}")
            generate_text = False
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"Failed to initialize decoder on {device} - GPU/CUDA error: {e}")
            generate_text = False
        except Exception as e:
            logger.error(f"Unexpected error initializing decoder: {type(e).__name__}: {e}")
            raise  # Re-raise unexpected errors for debugging

    results = {}
    for name, checkpoint_path in CHECKPOINTS.items():
        full_path = checkpoint_path if os.path.exists(checkpoint_path) else os.path.join(
            checkpoint_dir, os.path.basename(os.path.dirname(checkpoint_path)), "best.pt"
        )

        if not os.path.exists(full_path):
            logger.warning(f"Checkpoint not found for {name}: {full_path}")
            continue

        try:
            logger.info(f"Evaluating {name}...")
            model, stats = load_checkpoint(full_path, "mlp", device)

            # Distribution MSE
            mse_result = compute_distribution_mse(
                model, test_embeddings, n_samples=n_samples, n_steps=n_steps, device=device
            )

            # Path straightness
            straight_result = compute_path_straightness(
                model, test_embeddings, n_samples=n_samples, n_steps=n_steps, device=device
            )

            results[name] = {
                "mse": mse_result["mse"],
                "mse_std": mse_result["std"],
                "path_dev": straight_result["mean_path_deviation"],
                "path_dev_max": straight_result["max_path_deviation"],
            }

            # Generate text samples
            if generate_text and decoder is not None:
                try:
                    texts = generate_and_decode(
                        model, stats, decoder, n_samples=text_samples, n_steps=n_steps, device=device
                    )
                    results[name]["texts"] = texts
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    logger.warning(f"Text generation failed for {name} (GPU error): {e}")
                    results[name]["texts"] = []
                except Exception as e:
                    logger.warning(f"Text generation failed for {name}: {type(e).__name__}: {e}")
                    results[name]["texts"] = []

            logger.info(f"{name}: MSE={results[name]['mse']:.4f}, PathDev={results[name]['path_dev']:.4f}")

        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to load checkpoint for {name}: {e}")
            continue
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Out of GPU memory evaluating {name}: {e}. Try reducing n_samples.")
            continue
        except Exception as e:
            import traceback
            logger.error(f"Unexpected error evaluating {name}: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise  # Re-raise unexpected errors for debugging

    return results


def print_comparison_table(results: dict) -> None:
    """Print formatted comparison table."""
    print()
    print("=" * 60)
    print("Flow Method Comparison (Phase 5)")
    print("=" * 60)
    print(f"{'Method':<10} {'Dist MSE':<12} {'Path Dev':<12} {'Path Max':<12}")
    print("-" * 46)

    for name, r in results.items():
        print(f"{name:<10} {r['mse']:<12.4f} {r['path_dev']:<12.6f} {r['path_dev_max']:<12.6f}")

    print("=" * 60)

    # Print text samples
    print("\nSample Texts:")
    print("-" * 60)
    for name, r in results.items():
        if "texts" in r and r["texts"]:
            print(f"\n{name}:")
            for i, text in enumerate(r["texts"], 1):
                print(f"  [{i}] {text}")

    print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare flow matching methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="study/checkpoints",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for metrics",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="ODE integration steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for computation",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip text generation",
    )
    parser.add_argument(
        "--text-samples",
        type=int,
        default=2,
        help="Number of text samples per method",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    results = compare_methods(
        checkpoint_dir=args.checkpoint_dir,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        device=args.device,
        generate_text=not args.no_text,
        text_samples=args.text_samples,
    )

    print_comparison_table(results)

    # Summary
    if results:
        best_mse = min(results.items(), key=lambda x: x[1]["mse"])
        best_path = min(results.items(), key=lambda x: x[1]["path_dev"])
        print(f"Best MSE: {best_mse[0]} ({best_mse[1]['mse']:.4f})")
        print(f"Straightest paths: {best_path[0]} ({best_path[1]['path_dev']:.6f})")


if __name__ == "__main__":
    main()
