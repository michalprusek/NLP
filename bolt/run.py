"""CLI entry point for BOLT.

Usage:
    # Full pipeline with APE + Hyperband + BO
    uv run python -m bolt.run --iterations 50

    # Skip APE, use cached instructions
    uv run python -m bolt.run --no-use-ape --instructions bolt/data/ape_instructions.json

    # Hyperband only (no BO inference)
    uv run python -m bolt.run --hyperband-only

    # Load saved Hyperband results and run inference
    uv run python -m bolt.run --load-hyperband bolt/results/20260107_215457/hyperband_results.json
"""

import argparse
import gc
import json
import os
import random
import re
import torch
import numpy as np
from datetime import datetime

from bolt.config import BOLTConfig, get_device
from bolt.encoder import GTREncoder, StructureAwareVAE
from bolt.gp import GPWithEI
from bolt.training import (
    load_qa_pool_from_json,
    load_instructions,
    generate_ape_instructions,
    HbBoPsEvaluator,
)
from bolt.hyperband import BOLTHyperband
from bolt.inference import BOLTInference


# Pattern for extracting numbers from answers
NUMBER_PATTERN = re.compile(r'[-+]?\d+(?:[.,]\d+)?')


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_last_number(text: str) -> str | None:
    """Extract the last number from text."""
    if not text:
        return None
    matches = NUMBER_PATTERN.findall(text)
    return matches[-1] if matches else None


def numbers_match(pred: str | None, gold: str | None) -> bool:
    """Check if two number strings match (with 1e-6 tolerance)."""
    if pred is None or gold is None:
        return False
    try:
        pred_float = float(pred.replace(',', ''))
        gold_float = float(gold.replace(',', ''))
        return abs(pred_float - gold_float) <= 1e-6
    except ValueError:
        return False


def load_json_file(path: str, description: str = "file") -> list | dict:
    """Load and validate a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {description}: {path}\n"
            f"Parse error at line {e.lineno}: {e.msg}"
        ) from e


def load_hyperband_results(path: str) -> dict:
    """Load saved Hyperband results from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    best_error = results.get('best_error', 'N/A')
    best_any = results.get('best_error_any_fidelity', best_error)

    print(f"  Loaded {len(results['design_data'])} design points")
    print(f"  Best error (full fidelity): {best_error}")
    print(f"  Best error (any fidelity): {best_any}")

    return results


def train_vae_from_design_data(
    vae,
    design_data: list,
    instructions: list,
    gtr_encoder,
    qa_pool: list,
    config,
):
    """Train VAE on design data from Hyperband results.

    Returns:
        VAETrainer: The trained trainer (access .training_stats for stats)
    """
    from bolt.training import VAETrainer, TrainingSample

    samples = [
        TrainingSample(
            instruction_id=entry['instruction_id'],
            instruction_text=entry['instruction'],
            exemplar_ids=entry['exemplar_ids'],
            num_exemplars=entry['num_exemplars'],
            error_rate=entry['error_rate'],
            fidelity=entry['fidelity'],
        )
        for entry in design_data
    ]

    print(f"  Training VAE on {len(samples)} samples...")

    trainer = VAETrainer(
        vae=vae,
        gtr_encoder=gtr_encoder,
        qa_pool=qa_pool,
        instructions=instructions,
        config=config,
    )
    trainer.train(samples=samples)

    print("  VAE training complete")
    return trainer


def train_gp_from_design_data(
    vae,
    design_data: list,
    instruction_embeddings: torch.Tensor,
    pool_embeddings: torch.Tensor,
    config,
    device: str,
) -> GPWithEI:
    """Train GP on design data from Hyperband results."""
    latents = []
    errors = []
    fidelities = []

    vae.eval()
    with torch.no_grad():
        for entry in design_data:
            inst_emb = instruction_embeddings[entry['instruction_id']].unsqueeze(0)
            exemplar_ids = entry['exemplar_ids']

            if exemplar_ids:
                ex_embs = torch.stack([pool_embeddings[i] for i in exemplar_ids]).unsqueeze(0)
                ex_mask = torch.ones(1, len(exemplar_ids), dtype=torch.bool, device=device)
            else:
                ex_embs = torch.zeros(1, 1, 768, device=device)
                ex_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

            z = vae.encode_joint(inst_emb, ex_embs, ex_mask)
            latents.append(z.squeeze())
            errors.append(entry['error_rate'])
            fidelities.append(entry['fidelity'])

    X = torch.stack(latents)
    y = torch.tensor(errors, device=device)
    fidelities_tensor = torch.tensor(fidelities, device=device, dtype=torch.float32)

    print(f"  Training GP on {len(latents)} latent points...")

    gp = GPWithEI(
        instruction_dim=config.instruction_latent_dim,
        exemplar_dim=config.exemplar_latent_dim,
        device=device,
        use_deep_kernel=config.use_deep_kernel,
        dkl_feature_dim=config.dkl_feature_dim,
        dkl_hidden_dim=config.dkl_hidden_dim,
    )
    gp.fit(
        X=X,
        y=y,
        fidelities=fidelities_tensor,
        epochs=config.gp_epochs,
        lr=config.gp_lr,
        patience=config.gp_patience,
    )

    print("  GP training complete")
    return gp


def main():
    parser = argparse.ArgumentParser(description="BOLT: Joint Instruction-Exemplar Optimization")

    # Mode flags
    parser.add_argument("--hyperband-only", action="store_true",
                        help="Run only Hyperband (no BO inference)")
    parser.add_argument("--load-hyperband", type=str, default="",
                        help="Load Hyperband results from JSON and skip to inference")

    # Core parameters
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of BO iterations (after Hyperband)")

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
                        default="bolt/data/ape_instructions.json",
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
    parser.add_argument("--vae-beta", type=float, default=0.02,
                        help="KL weight (default: 0.02 for better regularization)")
    parser.add_argument("--selection-weight", type=float, default=0.2,
                        help="Exemplar selection loss weight (default: 0.2 to not dominate)")
    parser.add_argument("--instruction-latent-dim", type=int, default=16,
                        help="Instruction latent dimension")
    parser.add_argument("--exemplar-latent-dim", type=int, default=8,
                        help="Exemplar latent dimension (default: 8, smaller than instruction)")
    parser.add_argument("--num-exemplars", type=int, default=8,
                        help="Number of exemplars to select (fixed K=8)")

    # Acquisition parameters
    parser.add_argument("--acquisition", type=str, default="ucb",
                        choices=["ucb", "logei"],
                        help="Acquisition function for BO")
    parser.add_argument("--ucb-beta", type=float, default=8.0,
                        help="UCB exploration parameter")

    # Other
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, cpu, auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="bolt/results",
                        help="Output directory")

    args = parser.parse_args()

    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set seed
    set_seed(args.seed)

    # Create config
    config = BOLTConfig(
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
        selection_weight=args.selection_weight,
        instruction_latent_dim=args.instruction_latent_dim,
        exemplar_latent_dim=args.exemplar_latent_dim,
        num_exemplars=args.num_exemplars,
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
    print("BOLT: Latent Instruction Prompt Optimization with Exemplars")
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

    validation_data = load_json_file(config.validation_path, "Validation file")
    if not validation_data:
        raise ValueError(f"Validation file is empty: {config.validation_path}")
    print(f"  Loaded {len(validation_data)} validation samples")

    if args.use_ape:
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
        if args.instructions.endswith('.json'):
            instructions = load_json_file(args.instructions, "Instructions file")
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
        num_exemplars=config.num_exemplars,
        num_inducing=config.num_inducing_points,
        set_transformer_hidden=config.set_transformer_hidden,
        set_transformer_heads=config.set_transformer_heads,
        scorer_hidden_dim=config.scorer_hidden_dim,
        beta=config.vae_beta,
        mse_weight=config.vae_mse_weight,
        selection_weight=config.selection_weight,
        cross_attn_heads=config.cross_attn_heads,
        ranking_loss_type=config.ranking_loss_type,
    ).to(device)
    print(f"  VAE initialized: {config.instruction_latent_dim}D + {config.exemplar_latent_dim}D = {config.total_latent_dim}D (K={config.num_exemplars})")

    # =====================================================
    # Step 4: Run Hyperband OR Load saved results
    # =====================================================
    # Track training stats (for debugging/analysis)
    vae_training_stats = None
    gp_training_stats = None

    if args.load_hyperband:
        print(f"\n[4/5] Loading saved Hyperband results from {args.load_hyperband}...")
        hyperband_results = load_hyperband_results(args.load_hyperband)

        print("\n  Training VAE on loaded design data...")
        vae_trainer = train_vae_from_design_data(
            vae=vae,
            design_data=hyperband_results['design_data'],
            instructions=instructions,
            gtr_encoder=gtr_encoder,
            qa_pool=qa_pool,
            config=config,
        )
        vae_training_stats = vae_trainer.training_stats

        print("\n  Training GP on loaded design data...")
        gp = train_gp_from_design_data(
            vae=vae,
            design_data=hyperband_results['design_data'],
            instruction_embeddings=instruction_embeddings,
            pool_embeddings=pool_embeddings,
            config=config,
            device=device,
        )
        gp_training_stats = gp.training_stats

        print(f"\n  Best error from loaded results: {hyperband_results.get('best_error', 1.0):.4f}")

    else:
        print("\n[4/5] Running Hyperband optimization...")

        hyperband_evaluator = HbBoPsEvaluator(
            model=config.eval_model,
            backend=config.eval_backend,
            validation_path=config.validation_path,
            device=device,
        )

        def build_prompt(instruction: str, exemplar_text: str, question: str) -> str:
            """Build evaluation prompt."""
            if exemplar_text:
                return f"{exemplar_text}\n\nQuestion: {question}\n\n{instruction}\n\nAnswer:"
            return f"Question: {question}\n\n{instruction}\n\nAnswer:"

        def count_errors(samples: list, responses: list) -> int:
            """Count errors between samples and LLM responses."""
            # Validate length match to avoid silent truncation by zip()
            if len(responses) != len(samples):
                raise ValueError(
                    f"Length mismatch: got {len(responses)} responses but {len(samples)} samples. "
                    f"This may indicate LLM API failures or batch processing issues."
                )
            errors = 0
            for ex, response in zip(samples, responses):
                gold = extract_last_number(ex['answer'])
                pred = extract_last_number(response)
                if not numbers_match(pred, gold):
                    errors += 1
            return errors

        def llm_evaluator(instruction: str, exemplar_text: str, samples: list) -> float:
            """Evaluate single candidate."""
            if not samples:
                return 1.0

            prompts = [build_prompt(instruction, exemplar_text, ex['question']) for ex in samples]

            try:
                responses = hyperband_evaluator.llm_client.generate_batch(prompts, max_tokens=1024)
            except (ConnectionError, TimeoutError) as e:
                raise RuntimeError(f"Network error during LLM evaluation: {e}") from e

            # count_errors will validate length and raise ValueError if mismatch
            return count_errors(samples, responses) / len(samples)

        def batch_llm_evaluator(candidates: list, samples: list) -> list:
            """Batch evaluate multiple candidates at once."""
            if not samples or not candidates:
                return [1.0] * len(candidates)

            all_prompts = []
            prompt_counts = []

            for instruction, exemplar_text in candidates:
                candidate_prompts = [
                    build_prompt(instruction, exemplar_text, ex['question'])
                    for ex in samples
                ]
                all_prompts.extend(candidate_prompts)
                prompt_counts.append(len(candidate_prompts))

            try:
                all_responses = hyperband_evaluator.llm_client.generate_batch(all_prompts, max_tokens=1024)
            except (ConnectionError, TimeoutError) as e:
                raise RuntimeError(f"Network error during batch LLM evaluation: {e}") from e

            error_rates = []
            offset = 0
            for count in prompt_counts:
                responses = all_responses[offset:offset + count]
                offset += count
                errors = count_errors(samples[:count], responses)
                error_rates.append(errors / count if count > 0 else 1.0)

            return error_rates

        hyperband = BOLTHyperband(
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
            batch_llm_evaluator=batch_llm_evaluator,
        )

        hyperband.run_hyperband(verbose=True)

        # Save Hyperband results
        hyperband_path = os.path.join(output_dir, "hyperband_results.json")
        hyperband.save_results(hyperband_path)
        print(f"  Saved Hyperband results to {hyperband_path}")

        gp = hyperband.gp

        # Capture GP training stats (VAE is not retrained in Hyperband mode)
        gp_training_stats = gp.training_stats if gp else None

        # Clean up vLLM to free GPU memory before inference
        if hyperband_evaluator._llm_client is not None:
            del hyperband_evaluator._llm_client
            hyperband_evaluator._llm_client = None
        del hyperband_evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print("  Cleaned up Hyperband LLM client to free GPU memory")

        if args.hyperband_only:
            # Save training stats even for hyperband-only mode
            training_stats = {
                "vae": vae_training_stats,
                "gp": gp_training_stats,
            }
            stats_path = os.path.join(output_dir, "training_stats.json")
            with open(stats_path, "w") as f:
                json.dump(training_stats, f, indent=2)
            print(f"  Saved training stats to {stats_path}")
            print("\n[DONE] Hyperband complete. Exiting (--hyperband-only).")
            return

    # =====================================================
    # Step 5: Run BO inference
    # =====================================================
    print("\n[5/5] Running BO inference...")

    # GP should already be set (either from Hyperband or from loading)
    if gp is None:
        raise RuntimeError("GP not initialized. This should not happen.")

    # Create evaluator for inference
    evaluator = HbBoPsEvaluator(
        model=config.eval_model,
        backend=config.eval_backend,
        validation_path=config.validation_path,
        device=device,
    )

    inference = BOLTInference(
        vae=vae,
        gp=gp,
        gtr_encoder=gtr_encoder,
        qa_pool=qa_pool,
        pool_embeddings=pool_embeddings,
        instructions=instructions,
        evaluator=evaluator,
        config=config,
    )

    inference.run(
        num_iterations=config.iterations,
        output_dir=output_dir,
    )

    # Save training stats for debugging/analysis
    training_stats = {
        "vae": vae_training_stats,
        "gp": gp_training_stats,
    }
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    print(f"  Saved training stats to {stats_path}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
