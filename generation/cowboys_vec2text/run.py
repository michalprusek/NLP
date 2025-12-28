#!/usr/bin/env python3
"""COWBOYS Vec2Text CLI.

pCN MCMC optimization with trust regions and weighted retraining.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch

from .optimizer import CowboysOptimizer
from .inference import CowboysInference
from .mcmc import MCMCConfig
from .trust_region import TRConfig
from .training import RetrainConfig


def load_instructions(path: str) -> List[str]:
    """Load instructions from text file.

    Format: Lines starting with 'N. ' where N is instruction number.
    """
    instructions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                instructions.append(match.group(1))
    return instructions


def load_exemplars(path: str) -> List[str]:
    """Load exemplars from text file.

    Format: Blocks separated by '# Exemplar N' comments.
    """
    with open(path, "r") as f:
        content = f.read()

    blocks = []
    current_block = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Exemplar"):
            if current_block:
                blocks.append("\n".join(current_block))
            current_block = []
        elif line.startswith("#"):
            continue
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def load_validation(path: str) -> List[Dict]:
    """Load validation data from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def evaluate_prompt(
    prompt: str,
    validation_data: List[Dict],
    model: str,
    backend: str = "vllm",
) -> float:
    """Evaluate prompt on validation set. Returns error rate."""
    from src.llm_client import create_llm_client
    from src.gsm8k_evaluator import extract_answer, compare_numbers

    client = create_llm_client(model, backend)

    prompts = []
    expected_answers = []

    for item in validation_data:
        question = item["question"]
        expected = item["answer"]
        full_prompt = f"{prompt}\n\nQ: {question}\nA:"
        prompts.append(full_prompt)
        expected_answers.append(expected)

    responses = client.generate_batch(
        prompts,
        max_tokens=512,
        temperature=0.0,
    )

    correct = 0
    for response, expected in zip(responses, expected_answers):
        predicted = extract_answer(response)
        expected_num = extract_answer(expected)
        if predicted and expected_num and compare_numbers(predicted, expected_num):
            correct += 1

    return 1.0 - (correct / len(validation_data))


def main():
    parser = argparse.ArgumentParser(description="COWBOYS Vec2Text Optimization")

    # Data paths
    parser.add_argument("--instructions", type=str, default="datasets/hbbops/instructions_25.txt")
    parser.add_argument("--exemplars", type=str, default="datasets/hbbops/examples_25.txt")
    parser.add_argument("--grid-path", type=str, default="datasets/hbbops/full_grid_combined.jsonl")
    parser.add_argument("--validation", type=str, default="hbbops_improved_2/data/validation.json")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "openai", "deepinfra"])

    # MCMC parameters (NEW)
    parser.add_argument("--mcmc-steps", type=int, default=500, help="MCMC steps per chain")
    parser.add_argument("--mcmc-beta", type=float, default=0.1, help="pCN step size")
    parser.add_argument("--mcmc-chains", type=int, default=5, help="Number of MCMC chains")
    parser.add_argument("--mcmc-warmup", type=int, default=50, help="MCMC warmup steps")
    parser.add_argument("--mcmc-thinning", type=int, default=10, help="MCMC thinning factor")
    parser.add_argument("--mcmc-adapt-beta", action="store_true", default=True, help="Adapt beta during warmup")

    # Trust region parameters (NEW)
    parser.add_argument("--tr-initial", type=float, default=1.0, help="Initial trust region radius")
    parser.add_argument("--tr-min", type=float, default=0.1, help="Minimum trust region radius")
    parser.add_argument("--tr-max", type=float, default=5.0, help="Maximum trust region radius")
    parser.add_argument("--tr-expand", type=float, default=2.0, help="Trust region expand factor")
    parser.add_argument("--tr-contract", type=float, default=0.5, help="Trust region contract factor")
    parser.add_argument("--no-trust-region", action="store_true", help="Disable trust region")

    # Weighted retraining parameters (NEW)
    parser.add_argument("--retrain-interval", type=int, default=10, help="Retrain VAE every N iterations")
    parser.add_argument("--retrain-method", type=str, default="rank", choices=["rank", "exponential"])
    parser.add_argument("--retrain-epochs", type=int, default=50, help="Retraining epochs")
    parser.add_argument("--no-retrain", action="store_true", help="Disable weighted retraining")

    # VAE, GP, Vec2Text parameters (same as robust_vec2text)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--vae-lr", type=float, default=0.001)
    parser.add_argument("--vae-patience", type=int, default=30)
    parser.add_argument("--gp-epochs", type=int, default=500)
    parser.add_argument("--v2t-beam", type=int, default=8)
    parser.add_argument("--v2t-max-length", type=int, default=512)
    parser.add_argument("--v2t-no-repeat-ngram", type=int, default=3)
    parser.add_argument("--v2t-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-decode", type=int, default=20, help="Max candidates to decode per iteration")

    # Iterations
    parser.add_argument("--iterations", type=int, default=10, help="Number of optimization iterations")

    # APE Data Augmentation
    parser.add_argument("--ape-instructions", type=int, default=1000)
    parser.add_argument("--ape-cache", type=str, default="datasets/hbbops/ape_instructions_1000.json")
    parser.add_argument("--skip-ape", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="generation/cowboys_vec2text/results")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Build configurations
    mcmc_config = MCMCConfig(
        n_steps=args.mcmc_steps,
        beta=args.mcmc_beta,
        n_chains=args.mcmc_chains,
        warmup_steps=args.mcmc_warmup,
        thinning=args.mcmc_thinning,
        adapt_beta=args.mcmc_adapt_beta,
    )

    tr_config = TRConfig(
        initial_radius=args.tr_initial,
        min_radius=args.tr_min,
        max_radius=args.tr_max,
        expand_factor=args.tr_expand,
        contract_factor=args.tr_contract,
    )

    retrain_config = RetrainConfig(
        retrain_interval=args.retrain_interval,
        weight_method=args.retrain_method,
        epochs=args.retrain_epochs,
    ) if not args.no_retrain else None

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_{timestamp}.log"
    result_path = output_dir / f"result_{timestamp}.json"

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("COWBOYS Vec2Text Optimization")
    log("=" * 60)
    log(f"Timestamp: {timestamp}")
    log(f"Key settings:")
    log(f"  MCMC: {args.mcmc_chains} chains x {args.mcmc_steps} steps, beta={args.mcmc_beta}")
    log(f"  Trust Region: {'disabled' if args.no_trust_region else f'radius={args.tr_initial}'}")
    log(f"  Retraining: {'disabled' if args.no_retrain else f'every {args.retrain_interval} iters, method={args.retrain_method}'}")

    # Load data
    instructions = load_instructions(args.instructions)
    exemplars = load_exemplars(args.exemplars)
    validation_data = load_validation(args.validation)

    log(f"\nLoaded data:")
    log(f"  Original instructions: {len(instructions)}")
    log(f"  Exemplars: {len(exemplars)}")
    log(f"  Validation: {len(validation_data)} samples")

    # APE Data Augmentation
    if not args.skip_ape:
        log("\n" + "=" * 60)
        log("Loading/Generating Instructions via APE")
        log("=" * 60)

        from robust_vec2text.ape_generator import APEInstructionGenerator

        ape_generator = APEInstructionGenerator(
            model=args.model,
            backend=args.backend,
        )

        ape_instructions = ape_generator.generate_or_load(
            cache_path=args.ape_cache,
            validation_data=validation_data,
            num_instructions=args.ape_instructions,
            verbose=True,
        )

        all_instructions = list(set(instructions + ape_instructions))
        log(f"Total unique instructions: {len(all_instructions)}")
    else:
        log("\nSkipping APE generation (--skip-ape)")
        all_instructions = instructions

    # Initialize optimizer
    log("\n" + "=" * 60)
    log("Initializing COWBOYS Optimizer")
    log("=" * 60)

    optimizer = CowboysOptimizer(
        instructions=all_instructions,
        exemplars=exemplars,
        device="cuda",
        latent_dim=args.latent_dim,
    )

    # Load grid
    log("\n" + "=" * 60)
    log("Loading Pre-evaluated Grid")
    log("=" * 60)

    grid_prompts = optimizer.load_grid(
        args.grid_path,
        top_k=args.top_k,
        train_exemplar_gp=False,
    )

    best_prompt = optimizer.best_grid_prompt
    log(f"\nBest from grid:")
    log(f"  Instruction ID: {best_prompt.instruction_id}")
    log(f"  Exemplar ID: {best_prompt.exemplar_id}")
    log(f"  Error rate: {best_prompt.error_rate:.4f}")

    # Train VAE
    log("\n" + "=" * 60)
    log("Training VAE on Instruction Embeddings")
    log("=" * 60)

    vae_history = optimizer.train_vae(
        epochs=args.vae_epochs,
        lr=args.vae_lr,
        patience=args.vae_patience,
        verbose=True,
    )

    # Train GP on decoded embeddings (COWBOYS fix)
    log("\n" + "=" * 60)
    log("Training GP on VAE-Decoded Embeddings (COWBOYS)")
    log("=" * 60)

    optimizer.train_exemplar_gp_on_decoded(
        grid_path=args.grid_path,
        top_k=args.top_k,
        epochs=args.gp_epochs,
        patience=10,
        verbose=True,
    )

    # Initialize inference
    log("\n" + "=" * 60)
    log("Initializing COWBOYS Inference Pipeline")
    log("=" * 60)

    best_exemplar_emb = optimizer.exemplar_embeddings[best_prompt.exemplar_id]

    inference = CowboysInference(
        vae=optimizer.get_vae(),
        exemplar_selector=optimizer.get_exemplar_selector(),
        exemplar_emb=best_exemplar_emb,
        gtr=optimizer.gtr,
        device="cuda",
    )

    # Initialize trust region
    trust_region = None
    if not args.no_trust_region:
        log("Initializing Trust Region...")
        trust_region = optimizer.initialize_trust_region(config=tr_config)
        log(f"  Initial radius: {trust_region.radius:.4f}")

    # Get best latent
    best_latent = optimizer.get_best_latent()
    log(f"Best latent norm: {best_latent.norm().item():.4f}")

    # Iterative optimization loop
    log("\n" + "=" * 60)
    log(f"Starting COWBOYS Optimization ({args.iterations} iterations)")
    log("=" * 60)

    best_error = best_prompt.error_rate
    best_instruction = best_prompt.instruction
    best_exemplar_id = best_prompt.exemplar_id
    current_latent = best_latent.clone()

    iteration_history = []

    for iteration in range(args.iterations):
        log(f"\n{'='*60}")
        log(f"Iteration {iteration + 1}/{args.iterations}")
        log(f"{'='*60}")
        log(f"Current best error: {best_error:.4f}")

        if trust_region:
            log(f"Trust region radius: {trust_region.radius:.4f}")

        # 1. Check if VAE should be retrained
        if retrain_config and optimizer.should_retrain_vae(iteration, retrain_config):
            log("\n--- Weighted VAE Retraining ---")
            optimizer.retrain_vae(retrain_config, verbose=True)

        # 2. Run COWBOYS optimization pipeline
        log("\n--- pCN MCMC Optimization & Vec2Text Inversion ---")
        result = inference.full_pipeline(
            initial_latent=current_latent,
            best_y=best_error,
            mcmc_config=mcmc_config,
            trust_region=trust_region,
            v2t_beams=args.v2t_beam,
            v2t_max_length=args.v2t_max_length,
            v2t_no_repeat_ngram_size=args.v2t_no_repeat_ngram,
            v2t_repetition_penalty=args.v2t_repetition_penalty,
            max_decode=args.max_decode,
            verbose=True,
        )

        best_result = result.best_result
        log(f"\n  Generated:\n{best_result.text}")
        log(f"  Cosine: {best_result.cosine_similarity:.4f}, LogEI: {best_result.log_ei:.4f}")
        log(f"  MCMC samples: {len(result.mcmc_result.candidates)}, accept_rate: {result.mcmc_result.accept_rate:.3f}")

        # 3. Evaluate novel prompt
        log("\n--- Evaluating ---")
        novel_prompt = f"{best_result.text}\n\n{exemplars[best_exemplar_id]}"
        novel_error = evaluate_prompt(
            novel_prompt,
            validation_data,
            args.model,
            args.backend,
        )

        log(f"  Error rate: {novel_error:.4f} (accuracy: {(1-novel_error)*100:.2f}%)")

        # 4. Update GP with new observation
        log("\n--- Updating GP ---")
        decoded_emb = optimizer.get_decoded_embedding(best_result.text)
        inst_emb = optimizer.gtr.encode_tensor(best_result.text).to(optimizer.device)

        optimizer.exemplar_selector.add_observation_and_retrain(
            decoded_inst_emb=decoded_emb,
            exemplar_emb=best_exemplar_emb,
            error_rate=novel_error,
            epochs=100,
            verbose=False,
        )

        # Also add to optimizer for VAE retraining
        optimizer.add_observation(best_result.text, inst_emb, novel_error)

        log(f"  GP updated (total samples: {len(optimizer.exemplar_selector.y_train)})")

        # 5. Update trust region
        if trust_region:
            modified = trust_region.update(
                best_result.optimized_latent,
                novel_error,
                best_error,
                verbose=True,
            )

        # 6. Update best if improved
        improved = novel_error < best_error
        if improved:
            improvement = best_error - novel_error
            best_error = novel_error
            best_instruction = best_result.text
            log(f"  New best! Improvement: {improvement:.4f}")

        # 7. Store iteration history
        iteration_history.append({
            "iteration": iteration + 1,
            "instruction": best_result.text,
            "cosine_similarity": best_result.cosine_similarity,
            "log_ei": best_result.log_ei,
            "error_rate": novel_error,
            "improved": improved,
            "best_error_so_far": best_error,
            "mcmc_accept_rate": result.mcmc_result.accept_rate,
            "trust_region_radius": trust_region.radius if trust_region else None,
        })

        # 8. Update latent for next iteration
        with torch.no_grad():
            new_inst_emb = optimizer.gtr.encode_tensor(best_result.text).to(optimizer.device)
            current_latent = optimizer.get_vae().get_latent(new_inst_emb.unsqueeze(0)).squeeze(0)

        # Update trust region anchor to best latent if improved
        if improved and trust_region:
            trust_region.set_anchor(current_latent)

    # Final summary
    log("\n" + "=" * 60)
    log(f"COWBOYS Optimization Complete ({args.iterations} iterations)")
    log("=" * 60)
    log(f"Initial best error (grid): {best_prompt.error_rate:.4f}")
    log(f"Final best error: {best_error:.4f}")
    log(f"Total improvement: {best_prompt.error_rate - best_error:.4f}")
    log(f"Best instruction:\n{best_instruction}")

    # Save results
    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "method": "COWBOYS",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "grid_best": {
            "instruction_id": best_prompt.instruction_id,
            "exemplar_id": best_prompt.exemplar_id,
            "instruction": best_prompt.instruction,
            "exemplar": best_prompt.exemplar,
            "error_rate": best_prompt.error_rate,
        },
        "optimized": {
            "instruction": best_instruction,
            "full_prompt": f"{best_instruction}\n\n{exemplars[best_exemplar_id]}",
            "error_rate": best_error,
            "exemplar_id": best_exemplar_id,
        },
        "iteration_history": iteration_history,
        "improvement": best_prompt.error_rate - best_error,
        "vae_best_cosine": optimizer.vae_trainer.best_cosine,
    }

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 60)
    log("Run Complete")
    log("=" * 60)
    log(f"Log: {log_path}")
    log(f"Results: {result_path}")


if __name__ == "__main__":
    main()
