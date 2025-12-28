#!/usr/bin/env python3
"""COWBOYS Vec2Text CLI.

pCN MCMC optimization with trust regions and weighted retraining.
This is the instruction-only version (no exemplars).
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable

import torch
import torch.nn.functional as F
import numpy as np

from .optimizer import CowboysOptimizer
from .inference import CowboysInference
from .mcmc import MCMCConfig
from .trust_region import TRConfig
from .training import RetrainConfig


def evaluate_vae_quality(
    vae,
    embeddings: torch.Tensor,
    device: str = "cuda",
    log_fn: Callable[[str], None] = print,
) -> Dict[str, float]:
    """Evaluate VAE reconstruction quality with detailed metrics.

    Args:
        vae: Trained InstructionVAE model
        embeddings: Input embeddings to evaluate (N, 768)
        device: Device to use
        log_fn: Logging function

    Returns:
        Dictionary of quality metrics
    """
    vae.eval()
    embeddings = embeddings.to(device)

    with torch.no_grad():
        # Forward pass
        x_recon, mu, logvar = vae(embeddings)

        # Compute latent codes
        z = vae.get_latent(embeddings)

        # 1. Reconstruction quality metrics
        # Cosine similarity (most important for Vec2Text)
        cos_sim = F.cosine_similarity(embeddings, x_recon, dim=1)
        cos_mean = cos_sim.mean().item()
        cos_std = cos_sim.std().item()
        cos_min = cos_sim.min().item()
        cos_max = cos_sim.max().item()

        # MSE reconstruction error
        mse = F.mse_loss(x_recon, embeddings, reduction='none').mean(dim=1)
        mse_mean = mse.mean().item()
        mse_std = mse.std().item()

        # L2 reconstruction error (normalized)
        l2_error = torch.norm(x_recon - embeddings, dim=1) / torch.norm(embeddings, dim=1)
        l2_mean = l2_error.mean().item()

        # 2. Latent space statistics
        latent_norm = torch.norm(z, dim=1)
        latent_norm_mean = latent_norm.mean().item()
        latent_norm_std = latent_norm.std().item()

        # Latent dimension variances
        latent_var = z.var(dim=0)
        latent_var_mean = latent_var.mean().item()
        latent_var_min = latent_var.min().item()
        latent_var_max = latent_var.max().item()

        # Active dimensions (variance > 0.01)
        active_dims = (latent_var > 0.01).sum().item()

        # 3. KL divergence (regularization term)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_mean = kld.mean().item()
        kld_std = kld.std().item()

        # 4. Posterior collapse check
        # If KL is very low and variance is near zero, posterior may have collapsed
        posterior_collapsed = (kld_mean < 0.1) and (latent_var_mean < 0.1)

    metrics = {
        "cosine_mean": cos_mean,
        "cosine_std": cos_std,
        "cosine_min": cos_min,
        "cosine_max": cos_max,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "l2_relative_error": l2_mean,
        "latent_norm_mean": latent_norm_mean,
        "latent_norm_std": latent_norm_std,
        "latent_var_mean": latent_var_mean,
        "latent_var_min": latent_var_min,
        "latent_var_max": latent_var_max,
        "active_dims": active_dims,
        "kld_mean": kld_mean,
        "kld_std": kld_std,
        "posterior_collapsed": posterior_collapsed,
    }

    # Log results
    log_fn("\n--- VAE Quality Metrics ---")
    log_fn(f"Reconstruction (Cosine Similarity):")
    log_fn(f"  Mean: {cos_mean:.4f} | Std: {cos_std:.4f} | Min: {cos_min:.4f} | Max: {cos_max:.4f}")
    log_fn(f"Reconstruction (MSE):")
    log_fn(f"  Mean: {mse_mean:.6f} | Std: {mse_std:.6f}")
    log_fn(f"  Relative L2 Error: {l2_mean:.4f}")
    log_fn(f"Latent Space:")
    log_fn(f"  Norm: mean={latent_norm_mean:.4f}, std={latent_norm_std:.4f}")
    log_fn(f"  Variance per dim: mean={latent_var_mean:.4f}, min={latent_var_min:.4f}, max={latent_var_max:.4f}")
    log_fn(f"  Active dimensions (var>0.01): {int(active_dims)}/{z.shape[1]}")
    log_fn(f"KL Divergence:")
    log_fn(f"  Mean: {kld_mean:.4f} | Std: {kld_std:.4f}")
    if posterior_collapsed:
        log_fn("  Warning: Possible posterior collapse detected!")
    log_fn("----------------------------")

    return metrics


def load_instructions(path: str) -> List[str]:
    """Load instructions from text file.

    Format: Lines starting with 'N. ' where N is instruction number.
    """
    instructions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                instructions.append(match.group(1))
    return instructions


def load_validation(path: str) -> List[Dict]:
    """Load validation data from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def evaluate_prompt(
    prompt: str,
    validation_data: List[Dict],
    model: str,
    backend: str = "vllm",
    client=None,
) -> float:
    """Evaluate prompt on validation set. Returns error rate.

    For instruction-only mode, prompt is just the instruction.

    Args:
        client: Optional pre-initialized LLM client to reuse (saves GPU memory).
    """
    from src.llm_client import create_llm_client
    from src.gsm8k_evaluator import extract_answer, compare_numbers

    if client is None:
        client = create_llm_client(model, backend)

    prompts = []
    expected_answers = []

    for item in validation_data:
        question = item["question"]
        expected = item["answer"]
        # Instruction-only format: just instruction + Q/A
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
    parser = argparse.ArgumentParser(description="COWBOYS Vec2Text Optimization (Instruction-Only)")

    # Data paths (instruction-only, no exemplars)
    parser.add_argument("--instructions", type=str, default="datasets/cowboys/instructions_100.txt")
    parser.add_argument("--grid-path", type=str, default="/home/prusek/NLP/datasets/cowboys/grid_100_qend.jsonl")
    parser.add_argument("--validation", type=str, default="hbbops_improved_2/data/validation.json")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "openai", "deepinfra"])

    # MCMC parameters
    parser.add_argument("--mcmc-steps", type=int, default=500, help="MCMC steps per chain")
    parser.add_argument("--mcmc-beta", type=float, default=0.1, help="pCN step size")
    parser.add_argument("--mcmc-chains", type=int, default=5, help="Number of MCMC chains")
    parser.add_argument("--mcmc-warmup", type=int, default=50, help="MCMC warmup steps")
    parser.add_argument("--mcmc-thinning", type=int, default=10, help="MCMC thinning factor")
    parser.add_argument("--mcmc-adapt-beta", action="store_true", default=True, help="Adapt beta during warmup")

    # Trust region parameters (disabled by default)
    parser.add_argument("--tr-initial", type=float, default=1.0, help="Initial trust region radius")
    parser.add_argument("--tr-min", type=float, default=0.1, help="Minimum trust region radius")
    parser.add_argument("--tr-max", type=float, default=5.0, help="Maximum trust region radius")
    parser.add_argument("--tr-expand", type=float, default=2.0, help="Trust region expand factor")
    parser.add_argument("--tr-contract", type=float, default=0.5, help="Trust region contract factor")
    parser.add_argument("--trust-region", action="store_true", help="Enable trust region (disabled by default)")

    # Weighted retraining parameters
    parser.add_argument("--retrain-interval", type=int, default=10, help="Retrain VAE every N iterations")
    parser.add_argument("--retrain-method", type=str, default="rank", choices=["rank", "exponential"])
    parser.add_argument("--retrain-epochs", type=int, default=50, help="Retraining epochs")
    parser.add_argument("--no-retrain", action="store_true", help="Disable weighted retraining")

    # VAE, GP, Vec2Text parameters
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--vae-epochs", type=int, default=3000)
    parser.add_argument("--vae-lr", type=float, default=0.001)
    parser.add_argument("--vae-patience", type=int, default=30)
    parser.add_argument("--vae-cycle-weight", type=float, default=2.0,
                        help="Weight for cycle-consistency loss ||E(D(z)) - z||^2 (default: 2.0)")
    parser.add_argument("--gp-epochs", type=int, default=3000)
    parser.add_argument("--v2t-beam", type=int, default=8)
    parser.add_argument("--v2t-max-length", type=int, default=512)
    parser.add_argument("--v2t-no-repeat-ngram", type=int, default=3)
    parser.add_argument("--v2t-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-decode", type=int, default=20, help="Max candidates to decode per iteration")

    # Iterations
    parser.add_argument("--iterations", type=int, default=10, help="Number of optimization iterations")

    # APE Data Augmentation
    parser.add_argument("--ape-instructions", type=int, default=1000)
    parser.add_argument("--ape-cache", type=str, default="/home/prusek/NLP/datasets/cowboys/ape_instructions_1000.json")
    parser.add_argument("--skip-ape", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="generation/cowboys_vec2text/results")
    parser.add_argument("--debug", action="store_true")

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Generate EI landscape visualization after each iteration")

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
    log("COWBOYS Vec2Text Optimization (Instruction-Only)")
    log("=" * 60)
    log(f"Timestamp: {timestamp}")
    log(f"Key settings:")
    log(f"  MCMC: {args.mcmc_chains} chains x {args.mcmc_steps} steps, beta={args.mcmc_beta}")
    log(f"  Trust Region: {'enabled, radius=' + str(args.tr_initial) if args.trust_region else 'disabled'}")
    log(f"  Retraining: {'disabled' if args.no_retrain else f'every {args.retrain_interval} iters, method={args.retrain_method}'}")
    log(f"  VAE losses: cosine=20, mse=1, kld=0.0025 (annealed), cycle={args.vae_cycle_weight}")

    # Load data (instruction-only, no exemplars)
    instructions = load_instructions(args.instructions)
    validation_data = load_validation(args.validation)

    log(f"\nLoaded data:")
    log(f"  Instructions: {len(instructions)}")
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

        # Release APE generator to free GPU memory before evaluation
        del ape_generator
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        log("Released APE generator GPU memory")
    else:
        log("\nSkipping APE generation (--skip-ape)")
        all_instructions = instructions

    # Initialize optimizer (instruction-only, no exemplars)
    log("\n" + "=" * 60)
    log("Initializing COWBOYS Optimizer (Instruction-Only)")
    log("=" * 60)

    optimizer = CowboysOptimizer(
        instructions=all_instructions,
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
        train_instruction_gp=False,
    )

    best_prompt = optimizer.best_grid_prompt
    log(f"\nBest from grid:")
    log(f"  Instruction ID: {best_prompt.instruction_id}")
    log(f"  Instruction: {best_prompt.instruction}")
    log(f"  Error rate: {best_prompt.error_rate:.4f}")

    # Train VAE
    log("\n" + "=" * 60)
    log("Training VAE on Instruction Embeddings")
    log("=" * 60)

    vae_history = optimizer.train_vae(
        epochs=args.vae_epochs,
        lr=args.vae_lr,
        patience=args.vae_patience,
        lambda_cycle=args.vae_cycle_weight,
        verbose=True,
    )

    # Evaluate VAE quality
    log("\n" + "=" * 60)
    log("Evaluating VAE Quality")
    log("=" * 60)

    # Get all instruction embeddings for evaluation
    all_inst_embs = torch.stack(list(optimizer.instruction_embeddings.values()))
    vae_metrics = evaluate_vae_quality(
        vae=optimizer.get_vae(),
        embeddings=all_inst_embs,
        device="cuda",
        log_fn=log,
    )

    # Train GP on decoded embeddings (COWBOYS fix)
    log("\n" + "=" * 60)
    log("Training Instruction GP on VAE-Decoded Embeddings")
    log("=" * 60)

    optimizer.train_instruction_gp_on_decoded(
        grid_path=args.grid_path,
        top_k=args.top_k,
        epochs=args.gp_epochs,
        patience=10,
        verbose=True,
    )

    # Initialize inference (instruction-only, no exemplar_emb)
    log("\n" + "=" * 60)
    log("Initializing COWBOYS Inference Pipeline")
    log("=" * 60)

    inference = CowboysInference(
        vae=optimizer.get_vae(),
        instruction_selector=optimizer.get_instruction_selector(),
        gtr=optimizer.gtr,
        device="cuda",
    )

    # Initialize trust region (disabled by default)
    trust_region = None
    if args.trust_region:
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

    # Create evaluation client once (reused for all iterations)
    from src.llm_client import create_llm_client
    eval_client = create_llm_client(args.model, args.backend)
    log("Initialized evaluation LLM client")

    best_error = best_prompt.error_rate
    best_instruction = best_prompt.instruction
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

        # EI Landscape Visualization
        viz_metrics = None
        if args.visualize:
            from .visualize import visualize_ei_landscape
            viz_path = output_dir / f"ei_landscape_iter_{iteration + 1}.png"
            try:
                viz_metrics = visualize_ei_landscape(
                    inference=inference,
                    center_latent=best_result.optimized_latent,
                    realized_text=best_result.text,
                    best_y=best_error,
                    trajectory_latents=result.mcmc_result.candidates,
                    trust_region=trust_region,
                    save_path=str(viz_path),
                )
                log(f"  Visualization saved: {viz_path}")
                log(f"    Inversion gap (32D): {viz_metrics['inversion_gap_32d']:.4f}")
                log(f"    LogEI at z_opt: {viz_metrics['log_ei_at_opt']:.4f}")
                log(f"    LogEI at z_realized: {viz_metrics['log_ei_at_realized']:.4f}")
            except Exception as e:
                log(f"  Warning: Visualization failed: {e}")

        # 3. Evaluate novel instruction (instruction-only, no exemplar)
        log("\n--- Evaluating ---")
        novel_instruction = best_result.text

        # DEBUG: Log the exact prompt being evaluated
        log(f"  Evaluating instruction:")
        log(f"    {novel_instruction}")

        novel_error = evaluate_prompt(
            novel_instruction,
            validation_data,
            args.model,
            args.backend,
            client=eval_client,
        )

        log(f"  Error rate: {novel_error:.4f} (accuracy: {(1-novel_error)*100:.2f}%)")

        # 4. Update GP with new observation
        log("\n--- Updating GP ---")
        decoded_emb = optimizer.get_decoded_embedding(best_result.text)
        inst_emb = optimizer.gtr.encode_tensor(best_result.text).to(optimizer.device)

        optimizer.instruction_selector.add_observation_and_retrain(
            decoded_inst_emb=decoded_emb,
            error_rate=novel_error,
            epochs=3000,
            patience=10,
            verbose=False,
        )

        # Also add to optimizer for VAE retraining
        optimizer.add_observation(best_result.text, inst_emb, novel_error)

        log(f"  GP updated (total samples: {len(optimizer.instruction_selector.y_train)})")

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
        iter_record = {
            "iteration": iteration + 1,
            "instruction": best_result.text,
            "cosine_similarity": best_result.cosine_similarity,
            "log_ei": best_result.log_ei,
            "error_rate": novel_error,
            "improved": improved,
            "best_error_so_far": best_error,
            "mcmc_accept_rate": result.mcmc_result.accept_rate,
            "trust_region_radius": trust_region.radius if trust_region else None,
        }
        # Add visualization metrics if available
        if viz_metrics is not None:
            iter_record["inversion_gap_32d"] = viz_metrics.get("inversion_gap_32d")
            iter_record["log_ei_at_opt"] = viz_metrics.get("log_ei_at_opt")
            iter_record["log_ei_at_realized"] = viz_metrics.get("log_ei_at_realized")
        iteration_history.append(iter_record)

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

    # Save results (instruction-only format)
    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "method": "COWBOYS (instruction-only)",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "grid_best": {
            "instruction_id": best_prompt.instruction_id,
            "instruction": best_prompt.instruction,
            "error_rate": best_prompt.error_rate,
        },
        "optimized": {
            "instruction": best_instruction,
            "error_rate": best_error,
        },
        "iteration_history": iteration_history,
        "improvement": best_prompt.error_rate - best_error,
        "vae_best_cosine": optimizer.vae_trainer.best_cosine,
        "vae_quality_metrics": vae_metrics,
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
