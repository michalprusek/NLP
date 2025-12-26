#!/usr/bin/env python3
"""Run Vec2Text-integrated HbBoPs on GSM8K.

This script runs HbBoPs with GTR encoder and autoencoder for Vec2Text inversion.
By default uses instruction-only mode (Vec2Text works well on short instructions).

Usage:
    # Default: Instruction-only mode (recommended)
    uv run python -m vec2text_hbbops.run_vec2text_hbbops

    # Joint optimization (original mode, Vec2Text often fails on long exemplars)
    uv run python -m vec2text_hbbops.run_vec2text_hbbops --joint-optimization

Examples:
    # Default: instruction-only, load from grid, optimize instruction latent
    uv run python -m vec2text_hbbops.run_vec2text_hbbops

    # Custom top-k from grid
    uv run python -m vec2text_hbbops.run_vec2text_hbbops --top-k 50

    # Full HbBoPs with LLM evaluation
    uv run python -m vec2text_hbbops.run_vec2text_hbbops --run-hyperband --model qwen

    # Joint optimization (instruction + exemplar, original mode)
    uv run python -m vec2text_hbbops.run_vec2text_hbbops --joint-optimization

    # Just train and evaluate autoencoder (no Vec2Text)
    uv run python -m vec2text_hbbops.run_vec2text_hbbops --ae-only
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text_hbbops.hbbops_vec2text import HbBoPsVec2Text, Prompt
from vec2text_hbbops.inference import Vec2TextHbBoPsInference
from vec2text_hbbops.training import TrainingConfig


class TeeLogger:
    """Write to both console and file simultaneously."""

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Simple number pattern for answer extraction
NUMBER_PATTERN = r"[-+]?\d+(?:[.,]\d+)?"


def extract_answer(text: str) -> Optional[str]:
    """Extract last number from model output."""
    if not text:
        return None
    numbers = re.findall(NUMBER_PATTERN, text)
    return numbers[-1] if numbers else None


def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """Compare two numbers with tolerance."""
    if predicted == ground_truth:
        return True
    try:
        pred_clean = predicted.replace(",", "")
        gt_clean = ground_truth.replace(",", "")
        return abs(float(pred_clean) - float(gt_clean)) <= tolerance
    except (ValueError, TypeError):
        return False


class GSM8KEvaluator:
    """Evaluator for GSM8K prompts."""

    def __init__(self, llm_client, debug: bool = False):
        self.llm_client = llm_client
        self.debug = debug

    def __call__(self, prompt: Prompt, validation_data: list) -> float:
        """Evaluate prompt on validation data, returns error rate."""
        errors = 0
        prompts = [
            f"Question: {ex['question']}\n\n{str(prompt)}\n\nAnswer:"
            for ex in validation_data
        ]

        try:
            responses = self.llm_client.generate_batch(prompts, max_tokens=1024)
        except Exception as e:
            if self.debug:
                print(f"LLM error: {e}")
            return 1.0

        for i, response in enumerate(responses):
            predicted = extract_answer(response)
            ground_truth = validation_data[i]["answer"]
            if not predicted or not compare_numbers(predicted, ground_truth):
                errors += 1

        return errors / len(validation_data)


def load_data(
    instructions_path: Path,
    exemplars_path: Path,
    validation_path: Path,
) -> tuple:
    """Load instructions, exemplars, and validation data."""
    # Load instructions
    with open(instructions_path, "r", encoding="utf-8") as f:
        instructions = [line.strip() for line in f if line.strip()]

    # Load exemplars
    with open(exemplars_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Split by double newline (exemplars are separated by blank lines)
        exemplars = [ex.strip() for ex in content.split("\n\n") if ex.strip()]

    # Load validation data
    with open(validation_path, "r", encoding="utf-8") as f:
        validation_data = json.load(f)

    return instructions, exemplars, validation_data


def load_from_grid(
    grid_path: Path,
    instructions: List[str],
    exemplars: List[str],
    top_k: int = 25,
) -> List[Dict]:
    """Load top-k prompts from pre-evaluated grid.

    Args:
        grid_path: Path to full_grid_combined.jsonl
        instructions: List of instruction strings
        exemplars: List of exemplar strings
        top_k: Number of top prompts to load

    Returns:
        List of dicts with instruction_id, exemplar_id, error_rate
    """
    grid_data = []
    with open(grid_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                grid_data.append(json.loads(line))

    # Sort by error rate ascending
    grid_data.sort(key=lambda x: x.get("error_rate", 1.0))

    # Take top k
    return grid_data[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Run Vec2Text-integrated HbBoPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for task evaluation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "deepinfra", "auto"],
        help="LLM backend",
    )

    # Data paths
    parser.add_argument(
        "--instructions",
        type=str,
        default="datasets/hbbops/instructions_25.txt",
        help="Path to instructions file",
    )
    parser.add_argument(
        "--exemplars",
        type=str,
        default="datasets/hbbops/examples_25.txt",
        help="Path to exemplars file",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="hbbops_improved_2/data/validation.json",
        help="Path to validation data",
    )

    # Grid mode (default)
    parser.add_argument(
        "--grid-path",
        type=str,
        default="datasets/hbbops/full_grid_combined.jsonl",
        help="Path to pre-evaluated grid",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top prompts to load from grid",
    )

    # Optimization control
    parser.add_argument(
        "--run-hyperband",
        action="store_true",
        help="Run full HbBoPs optimization instead of loading from grid (requires LLM)",
    )
    parser.add_argument(
        "--skip-latent-opt",
        action="store_true",
        help="Skip latent space optimization",
    )
    parser.add_argument(
        "--ae-only",
        action="store_true",
        help="Only train and evaluate autoencoder, no Vec2Text inversion",
    )
    parser.add_argument(
        "--instruction-only",
        action="store_true",
        default=True,
        help="Optimize only instructions (default), keep exemplars fixed from grid",
    )
    parser.add_argument(
        "--joint-optimization",
        action="store_true",
        help="Use joint instruction+exemplar optimization (original mode, Vec2Text often fails on long exemplars)",
    )

    # Autoencoder parameters
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=3000,
        help="Max autoencoder training epochs",
    )
    parser.add_argument(
        "--ae-patience",
        type=int,
        default=20,
        help="Autoencoder early stopping patience",
    )
    parser.add_argument(
        "--ae-lr",
        type=float,
        default=1e-3,
        help="Autoencoder learning rate",
    )
    parser.add_argument(
        "--ae-dropout",
        type=float,
        default=0.3,
        help="Autoencoder dropout rate",
    )
    parser.add_argument(
        "--ae-noise",
        type=float,
        default=0.1,
        help="Denoising noise std",
    )

    # Vec2Text parameters
    parser.add_argument(
        "--v2t-steps",
        type=int,
        default=50,
        help="Vec2Text correction steps",
    )
    parser.add_argument(
        "--v2t-beam",
        type=int,
        default=8,
        help="Vec2Text sequence beam width",
    )

    # HbBoPs parameters
    parser.add_argument("--bmin", type=int, default=10, help="Minimum fidelity")
    parser.add_argument("--eta", type=float, default=2.0, help="Halving ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vec2text_hbbops/results",
        help="Output directory",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_{timestamp}.log"
    result_path = output_dir / f"result_{timestamp}.json"

    # Setup logging
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60)
    print("Vec2Text-HbBoPs Optimization")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Arguments: {vars(args)}")

    try:
        # Load data
        root = Path(__file__).parent.parent
        instructions, exemplars, validation_data = load_data(
            root / args.instructions,
            root / args.exemplars,
            root / args.validation,
        )

        print(f"\nLoaded data:")
        print(f"  Instructions: {len(instructions)}")
        print(f"  Exemplars: {len(exemplars)}")
        print(f"  Validation: {len(validation_data)} samples")

        # Create LLM client (only if running full HbBoPs)
        llm_client = None
        evaluator = None

        if args.run_hyperband and not args.ae_only:
            from src.llm_client import create_llm_client

            print(f"\nInitializing LLM client: {args.model}")
            llm_client = create_llm_client(args.model, backend=args.backend)
            evaluator = GSM8KEvaluator(llm_client, debug=args.debug)

        # Create HbBoPs instance
        hbbops = HbBoPsVec2Text(
            instructions=instructions,
            exemplars=exemplars,
            validation_data=validation_data,
            llm_evaluator=evaluator,
            bmin=args.bmin,
            eta=args.eta,
            seed=args.seed,
            ae_dropout=args.ae_dropout,
            ae_noise_std=args.ae_noise,
        )

        # Training config
        ae_config = TrainingConfig(
            learning_rate=args.ae_lr,
            max_epochs=args.ae_epochs,
            patience=args.ae_patience,
        )

        # Determine optimization mode
        use_instruction_only = args.instruction_only and not args.joint_optimization

        if args.ae_only:
            # Train AE on all embeddings (cartesian product)
            hbbops.train_autoencoder(config=ae_config, verbose=True)
            print("\nAutoencoder-only mode complete.")
            result = {"mode": "ae_only", "timestamp": timestamp}
        else:
            # Load from grid or run HbBoPs
            if args.run_hyperband:
                # Full HbBoPs optimization (requires LLM)
                # Train AE on all embeddings first
                hbbops.train_autoencoder(config=ae_config, verbose=True)
                best_prompt, best_error = hbbops.run_hyperband(verbose=True)
            else:
                # Default: Load from pre-evaluated grid (fast, no LLM)
                # Train joint AE on full grid (625 prompts), GP on top-k only
                hbbops.train_autoencoder_from_grid(
                    grid_path=args.grid_path,
                    config=ae_config,
                    verbose=True,
                )
                best_prompt, best_error = hbbops.load_from_grid(
                    grid_path=args.grid_path,
                    top_k=args.top_k,
                    verbose=True,
                )

            # For instruction-only mode, also train instruction autoencoder
            if use_instruction_only:
                hbbops.train_instruction_autoencoder_from_grid(
                    grid_path=args.grid_path,
                    config=ae_config,
                    verbose=True,
                )

            # Create LLM client for novel prompt evaluation
            from src.llm_client import create_llm_client

            print(f"\nInitializing LLM client for evaluation: {args.model}")
            llm_client = create_llm_client(args.model, backend=args.backend)
            evaluator = GSM8KEvaluator(llm_client, debug=args.debug)

            # Create inference pipeline for Vec2Text inversion
            inference = Vec2TextHbBoPsInference(
                hbbops,
                vec2text_steps=args.v2t_steps,
                vec2text_beam=args.v2t_beam,
                llm_evaluator=evaluator,
            )

            # Run optimization and inversion
            if use_instruction_only:
                # Instruction-only: optimize instruction latent, keep exemplar fixed
                print("\n" + "=" * 60)
                print("Using Instruction-Only Mode")
                print("=" * 60)
                print("Vec2Text works well on short instructions (~30 tokens)")
                print("Exemplar is fixed from grid (Vec2Text fails on long text)")
                result = inference.run_instruction_only_pipeline(
                    n_latent_candidates=100,
                    perturbation_std=0.5,
                    verbose=True,
                )
            else:
                # Joint optimization (original mode)
                result = inference.run_full_pipeline(
                    run_hyperband=False,  # Already loaded/ran above
                    optimize_latent=not args.skip_latent_opt,
                    verbose=True,
                )

            # Prepare result for JSON serialization
            result_dict = {
                "timestamp": timestamp,
                "args": vars(args),
                "best_grid_error": result.best_grid_error,
                "num_evaluations": result.num_evaluations,
                "design_data_size": result.design_data_size,
            }

            if result.best_from_grid:
                result_dict["best_from_grid"] = {
                    "instruction_id": result.best_from_grid.instruction_id,
                    "exemplar_id": result.best_from_grid.exemplar_id,
                    "instruction": result.best_from_grid.instruction,
                    "exemplar": result.best_from_grid.exemplar[:200] + "...",
                }

            if result.best_reconstructed:
                result_dict["best_reconstructed"] = {
                    "instruction_text": result.best_reconstructed.instruction_text,
                    "exemplar_text": result.best_reconstructed.exemplar_text[:200] + "...",
                    "instruction_cosine": result.best_reconstructed.instruction_cosine,
                    "exemplar_cosine": result.best_reconstructed.exemplar_cosine,
                }

            if result.novel_from_latent:
                result_dict["novel_from_latent"] = {
                    "instruction_text": result.novel_from_latent.instruction_text,
                    "exemplar_text": result.novel_from_latent.exemplar_text[:200] + "...",
                    "instruction_cosine": result.novel_from_latent.instruction_cosine,
                    "exemplar_cosine": result.novel_from_latent.exemplar_cosine,
                    "evaluated_error": result.novel_from_latent.evaluated_error,
                }

            result = result_dict

        # Save results
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print("Run Complete")
        print("=" * 60)
        print(f"Log: {log_path}")
        print(f"Results: {result_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
