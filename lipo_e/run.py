"""CLI entry point for LIPO-E.

Usage:
    # Full pipeline with APE + Hyperband + InvBO
    uv run python -m lipo_e.run --iterations 50

    # Skip APE, use cached instructions
    uv run python -m lipo_e.run --no-use-ape --instructions lipo_e/data/ape_instructions.json

    # Hyperband only (no InvBO inference)
    uv run python -m lipo_e.run --hyperband-only
"""

import argparse
import json
import os
import random
import re
import traceback
import torch
import numpy as np
from datetime import datetime

from lipo_e.config import LIPOEConfig, get_device
from lipo_e.encoder import GTREncoder, StructureAwareVAE
from lipo_e.gp import GPWithEI
from lipo_e.training import (
    load_qa_pool_from_json,
    load_instructions,
    generate_ape_instructions,
    HbBoPsEvaluator,
)
from lipo_e.hyperband import LIPOEHyperband
from lipo_e.inference import LIPOEInference


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="LIPO-E: Joint Instruction-Exemplar Optimization")

    # Mode flags
    parser.add_argument("--hyperband-only", action="store_true",
                        help="Run only Hyperband (no InvBO inference)")

    # Core parameters
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of InvBO iterations (after Hyperband)")

    # Model parameters
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="LLM model for evaluation")
    parser.add_argument("--backend", type=str, default="vllm",
                        help="LLM backend")

    # Q/A Pool options
    parser.add_argument("--train-data-path", type=str,
                        default="hbbops_improved_2/data/train.json",
                        help="Path to training JSON for Q/A pool")
    parser.add_argument("--qa-pool-size", type=int, default=6154,
                        help="Number of Q/A pairs to sample (default: all)")

    # APE options
    parser.add_argument("--use-ape", action="store_true", default=True,
                        help="Generate instructions using APE (default: True)")
    parser.add_argument("--no-use-ape", action="store_false", dest="use_ape",
                        help="Load instructions from file instead of APE")
    parser.add_argument("--ape-num-instructions", type=int, default=2000,
                        help="Number of instructions to generate with APE")
    parser.add_argument("--ape-cache-path", type=str,
                        default="lipo_e/data/ape_instructions.json",
                        help="Path to cache APE instructions")
    parser.add_argument("--force-regenerate-ape", action="store_true",
                        help="Force regenerate APE instructions even if cached")
    parser.add_argument("--instructions", type=str,
                        default="datasets/hbbops/instructions_25.txt",
                        help="Path to instructions (when --no-use-ape)")

    # Validation data
    parser.add_argument("--validation-path", type=str,
                        default="hbbops_improved_2/data/validation.json",
                        help="Path to validation data")

    # Hyperband parameters
    parser.add_argument("--bmin", type=int, default=10,
                        help="Minimum fidelity for Hyperband")
    parser.add_argument("--eta", type=float, default=2.0,
                        help="Hyperband halving rate")

    # VAE parameters
    parser.add_argument("--vae-epochs", type=int, default=50000,
                        help="VAE training epochs")
    parser.add_argument("--vae-beta", type=float, default=0.005,
                        help="KL weight")
    parser.add_argument("--instruction-latent-dim", type=int, default=16,
                        help="Instruction latent dimension")
    parser.add_argument("--exemplar-latent-dim", type=int, default=16,
                        help="Exemplar latent dimension")
    parser.add_argument("--num-slots", type=int, default=8,
                        help="Maximum exemplars to select")

    # Acquisition parameters
    parser.add_argument("--acquisition", type=str, default="ucb",
                        choices=["ucb", "logei"],
                        help="Acquisition function for InvBO")
    parser.add_argument("--ucb-beta", type=float, default=8.0,
                        help="UCB exploration parameter")

    # Other
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, cpu, auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="lipo_e/results",
                        help="Output directory")

    args = parser.parse_args()

    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set seed
    set_seed(args.seed)

    # Create config
    config = LIPOEConfig(
        device=device,
        seed=args.seed,
        iterations=args.iterations,
        eval_model=args.model,
        eval_backend=args.backend,
        train_data_path=args.train_data_path,
        qa_pool_size=args.qa_pool_size,
        validation_path=args.validation_path,
        bmin=args.bmin,
        eta=args.eta,
        vae_epochs=args.vae_epochs,
        vae_beta=args.vae_beta,
        instruction_latent_dim=args.instruction_latent_dim,
        exemplar_latent_dim=args.exemplar_latent_dim,
        num_slots=args.num_slots,
        acquisition_type=args.acquisition,
        ucb_beta=args.ucb_beta,
        ape_num_instructions=args.ape_num_instructions,
        ape_cache_path=args.ape_cache_path,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("LIPO-E: Latent Instruction Prompt Optimization with Exemplars")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Seed: {config.seed}")

    # =====================================================
    # Step 1: Load Q/A pool
    # =====================================================
    print("\n[1/5] Loading Q/A pool...")

    qa_pool = load_qa_pool_from_json(
        file_path=config.train_data_path,
        max_samples=config.qa_pool_size,
        shuffle=True,
        seed=config.seed,
    )
    if not qa_pool:
        raise ValueError(f"No Q/A pairs loaded from {config.train_data_path}")
    print(f"  Loaded {len(qa_pool)} Q/A pairs from {config.train_data_path}")

    # Initialize GTR encoder
    gtr_encoder = GTREncoder(device=device)

    # Pre-compute pool embeddings
    print("  Computing pool embeddings...")
    pool_texts = [qa.format() for qa in qa_pool]
    pool_embeddings = gtr_encoder.encode(pool_texts)
    print(f"  Pool embeddings shape: {pool_embeddings.shape}")

    # =====================================================
    # Step 2: Load or generate instructions
    # =====================================================
    print("\n[2/5] Preparing instructions...")

    # Load validation data for APE
    try:
        with open(config.validation_path, "r", encoding="utf-8") as f:
            validation_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Validation file not found: {config.validation_path}"
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in validation file: {config.validation_path}\n"
            f"Parse error at line {e.lineno}: {e.msg}"
        ) from e

    if not validation_data:
        raise ValueError(f"Validation file is empty: {config.validation_path}")
    print(f"  Loaded {len(validation_data)} validation samples")

    if args.use_ape:
        # Generate instructions using APE
        instructions = generate_ape_instructions(
            model=config.ape_model,
            backend=config.ape_backend,
            validation_data=validation_data,
            num_instructions=config.ape_num_instructions,
            cache_path=config.ape_cache_path,
            force_regenerate=args.force_regenerate_ape,
            verbose=True,
        )
        print(f"  Using {len(instructions)} APE-generated instructions")
    else:
        # Load instructions from file
        if args.instructions.endswith('.json'):
            try:
                with open(args.instructions, 'r', encoding='utf-8') as f:
                    instructions = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Instructions file not found: {args.instructions}\n"
                    f"Use --use-ape to generate instructions automatically."
                ) from None
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in instructions file: {args.instructions}\n"
                    f"Parse error at line {e.lineno}: {e.msg}"
                ) from e
            if not isinstance(instructions, list):
                raise ValueError(
                    f"Instructions file must contain a JSON array, got {type(instructions).__name__}"
                )
        else:
            instructions = load_instructions(args.instructions)

        if not instructions:
            raise ValueError(f"No instructions loaded from {args.instructions}")
        print(f"  Loaded {len(instructions)} instructions from {args.instructions}")

    # Pre-compute instruction embeddings
    print("  Computing instruction embeddings...")
    instruction_embeddings = gtr_encoder.encode(instructions)
    print(f"  Instruction embeddings shape: {instruction_embeddings.shape}")

    # =====================================================
    # Step 3: Initialize VAE (for Hyperband proposals)
    # =====================================================
    print("\n[3/5] Initializing VAE...")

    vae = StructureAwareVAE(
        embedding_dim=config.embedding_dim,
        instruction_latent_dim=config.instruction_latent_dim,
        exemplar_latent_dim=config.exemplar_latent_dim,
        num_slots=config.num_slots,
        num_inducing=config.num_inducing_points,
        set_transformer_hidden=config.set_transformer_hidden,
        set_transformer_heads=config.set_transformer_heads,
        beta=config.vae_beta,
        mse_weight=config.vae_mse_weight,
        selection_weight=config.selection_weight,
        num_exemplars_weight=config.num_exemplars_weight,
    ).to(device)
    print(f"  VAE initialized: {config.instruction_latent_dim}D + {config.exemplar_latent_dim}D = {config.total_latent_dim}D")

    # =====================================================
    # Step 4: Run Hyperband
    # =====================================================
    print("\n[4/5] Running Hyperband optimization...")

    # Create evaluator once (not inside closure for efficiency)
    hyperband_evaluator = HbBoPsEvaluator(
        model=config.eval_model,
        backend=config.eval_backend,
        validation_path=config.validation_path,
        device=device,
    )

    # Number pattern for answer extraction
    NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'

    def llm_evaluator(instruction: str, exemplar_text: str, samples: list) -> float:
        """Wrapper for HbBoPsEvaluator."""
        if not samples:
            return 1.0

        # Build prompts and evaluate
        prompts = []
        for ex in samples:
            if exemplar_text:
                prompt = f"{exemplar_text}\n\nQuestion: {ex['question']}\n\n{instruction}\n\nAnswer:"
            else:
                prompt = f"Question: {ex['question']}\n\n{instruction}\n\nAnswer:"
            prompts.append(prompt)

        # LLM call with error handling
        try:
            responses = hyperband_evaluator.llm_client.generate_batch(prompts, max_tokens=1024)
        except (ConnectionError, TimeoutError) as e:
            raise RuntimeError(
                f"Network error during LLM evaluation: {e}\n"
                f"Check network connection and retry."
            ) from e
        except Exception as e:
            print(f"LLM error during evaluation:\n{traceback.format_exc()}")
            raise RuntimeError(f"LLM evaluation failed: {e}") from e

        # Validate response count
        if len(responses) != len(samples):
            print(f"  [WARNING] Response count mismatch: got {len(responses)}, expected {len(samples)}")

        # Score - use zip for safe iteration
        errors = 0
        for i, (ex, response) in enumerate(zip(samples, responses)):
            gold_nums = re.findall(NUMBER_PATTERN, ex['answer'])
            gold = gold_nums[-1] if gold_nums else None

            pred_nums = re.findall(NUMBER_PATTERN, response) if response else []
            pred = pred_nums[-1] if pred_nums else None

            if gold is None or pred is None:
                errors += 1
            else:
                try:
                    pred_float = float(pred.replace(',', ''))
                    gold_float = float(gold.replace(',', ''))
                    if abs(pred_float - gold_float) > 1e-6:
                        errors += 1
                except ValueError as e:
                    # Log failed comparison for debugging
                    print(f"  [DEBUG] Number parse failed: pred='{pred}', gold='{gold}': {e}")
                    errors += 1

        return errors / len(samples)

    hyperband = LIPOEHyperband(
        instructions=instructions,
        qa_pool=qa_pool,
        validation_data=validation_data,
        vae=vae,
        gtr_encoder=gtr_encoder,
        pool_embeddings=pool_embeddings,
        instruction_embeddings=instruction_embeddings,
        llm_evaluator=llm_evaluator,
        config=config,
        device=device,
    )

    best_prompt, best_error = hyperband.run_hyperband(verbose=True)

    # Save Hyperband results
    hyperband_path = os.path.join(output_dir, "hyperband_results.json")
    hyperband.save_results(hyperband_path)
    print(f"  Saved Hyperband results to {hyperband_path}")

    if args.hyperband_only:
        print("\n[DONE] Hyperband complete. Exiting (--hyperband-only).")
        return

    # =====================================================
    # Step 5: Run InvBO inference
    # =====================================================
    print("\n[5/5] Running InvBO inference...")

    # Get GP from Hyperband
    gp = hyperband.gp

    if gp is None:
        print("  Training GP on Hyperband data...")
        hyperband._train_gp()
        gp = hyperband.gp

    # Create evaluator for inference
    evaluator = HbBoPsEvaluator(
        model=config.eval_model,
        backend=config.eval_backend,
        validation_path=config.validation_path,
        device=device,
    )

    inference = LIPOEInference(
        vae=vae,
        gp=gp,
        gtr_encoder=gtr_encoder,
        qa_pool=qa_pool,
        pool_embeddings=pool_embeddings,
        instructions=instructions,
        evaluator=evaluator,
        config=config,
    )

    history = inference.run(
        num_iterations=config.iterations,
        output_dir=output_dir,
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
