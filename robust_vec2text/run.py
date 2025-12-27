#!/usr/bin/env python3
"""Robust VAE-HbBoPs CLI.

Instruction-only optimization with VAE latent space and gradient-based EI.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch

from robust_vec2text.optimizer import RobustHbBoPs
from robust_vec2text.inference import RobustInference
from robust_vec2text.visualize_ei import visualize_ei_landscape, get_training_latents


def load_instructions(path: str) -> list:
    """Load instructions from text file.

    Format: Lines starting with 'N. ' where N is instruction number.
    Skips comments (#) and empty lines.
    """
    import re
    instructions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Match lines starting with number and period: "1. Answer:"
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                instructions.append(match.group(1))
    return instructions


def load_exemplars(path: str) -> list:
    """Load exemplars from text file.

    Format: Blocks separated by '# Exemplar N' comments.
    Each block contains multiple Q/A pairs.
    """
    with open(path, "r") as f:
        content = f.read()

    # Split by exemplar headers
    blocks = []
    current_block = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Exemplar"):
            # Save previous block if exists
            if current_block:
                blocks.append("\n".join(current_block))
            current_block = []
        elif line.startswith("#"):
            # Skip other comments
            continue
        else:
            current_block.append(line)

    # Don't forget last block
    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def load_validation(path: str) -> list:
    """Load validation data from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def evaluate_prompt(
    prompt: str,
    validation_data: list,
    model: str,
    backend: str = "vllm",
) -> float:
    """Evaluate prompt on validation set.

    Returns error rate (lower is better).
    """
    from src.llm_client import create_llm_client
    from src.gsm8k_evaluator import extract_answer, compare_numbers

    client = create_llm_client(model, backend)

    correct = 0
    total = len(validation_data)

    # Batch generate for efficiency
    prompts = []
    expected_answers = []

    for item in validation_data:
        question = item["question"]
        expected = item["answer"]
        full_prompt = f"{prompt}\n\nQ: {question}\nA:"
        prompts.append(full_prompt)
        expected_answers.append(expected)

    # Generate all at once
    responses = client.generate_batch(
        prompts,
        max_tokens=512,
        temperature=0.0,
    )

    for response, expected in zip(responses, expected_answers):
        predicted = extract_answer(response)
        expected_num = extract_answer(expected)  # Extract number from expected answer
        if predicted and expected_num and compare_numbers(predicted, expected_num):
            correct += 1

    return 1.0 - (correct / total)


def main():
    parser = argparse.ArgumentParser(description="Robust VAE-HbBoPs Optimization")

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
        "--grid-path",
        type=str,
        default="datasets/hbbops/full_grid_combined.jsonl",
        help="Path to pre-evaluated grid",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="hbbops_improved_2/data/validation.json",
        help="Path to validation data",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for evaluation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "deepinfra"],
        help="Backend for LLM",
    )

    # Optimization parameters
    parser.add_argument("--top-k", type=int, default=25, help="Top prompts from grid")
    parser.add_argument("--latent-dim", type=int, default=32, help="VAE latent dimension")

    # VAE training
    parser.add_argument("--vae-epochs", type=int, default=200, help="VAE training epochs")
    parser.add_argument("--vae-lr", type=float, default=0.001, help="VAE learning rate")
    parser.add_argument("--vae-patience", type=int, default=30, help="VAE early stopping")

    # GP training
    parser.add_argument("--gp-epochs", type=int, default=500, help="GP training epochs")
    parser.add_argument("--gp-lr", type=float, default=0.01, help="GP learning rate")

    # Gradient optimization
    parser.add_argument("--opt-steps", type=int, default=500, help="Gradient optimization steps")
    parser.add_argument("--opt-lr", type=float, default=0.1, help="Gradient optimization LR")
    parser.add_argument("--opt-patience", type=int, default=10, help="Gradient optimization early stopping patience")

    # Vec2Text
    parser.add_argument("--v2t-beam", type=int, default=8, help="Vec2Text beam width")
    parser.add_argument("--v2t-max-length", type=int, default=512, help="Vec2Text max output length")
    parser.add_argument("--v2t-no-repeat-ngram", type=int, default=3, help="Block repeating n-grams (0 to disable)")
    parser.add_argument("--v2t-repetition-penalty", type=float, default=1.2, help="Repetition penalty (1.0 = no penalty)")

    # Iterative optimization
    parser.add_argument("--iterations", type=int, default=1, help="Number of optimization iterations")

    # APE Data Augmentation
    parser.add_argument("--ape-instructions", type=int, default=1000, help="Number of APE-generated instructions")
    parser.add_argument("--ape-cache", type=str, default="datasets/hbbops/ape_instructions_1000.json", help="Path to cache APE instructions")
    parser.add_argument("--skip-ape", action="store_true", help="Skip APE generation (use original instructions only)")

    # Exemplar Selection
    parser.add_argument("--select-exemplar", action="store_true", help="Use GP to select optimal exemplar")
    parser.add_argument("--exemplar-top-k", type=int, default=5, help="Number of top exemplars to consider")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="robust_vec2text/results",
        help="Output directory",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log file
    log_path = output_dir / f"run_{timestamp}.log"
    result_path = output_dir / f"result_{timestamp}.json"

    # Log function
    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("Robust VAE-HbBoPs Optimization")
    log("=" * 60)
    log(f"Timestamp: {timestamp}")
    log(f"Arguments: {vars(args)}")

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
        log("Loading/Generating Instructions via APE Forward Pass")
        log("=" * 60)
        log(f"APE cache path: {args.ape_cache}")

        from robust_vec2text.ape_generator import APEInstructionGenerator

        ape_generator = APEInstructionGenerator(
            model=args.model,
            backend=args.backend,
        )

        # Use generate_or_load to cache/load instructions
        ape_instructions = ape_generator.generate_or_load(
            cache_path=args.ape_cache,
            validation_data=validation_data,
            num_instructions=args.ape_instructions,
            verbose=True,
        )

        # Combine with original instructions
        all_instructions = list(set(instructions + ape_instructions))
        log(f"Total unique instructions: {len(all_instructions)}")
    else:
        log("\nSkipping APE generation (--skip-ape)")
        all_instructions = instructions

    # Initialize optimizer
    log("\n" + "=" * 60)
    log("Initializing Optimizer")
    log("=" * 60)

    optimizer = RobustHbBoPs(
        instructions=all_instructions,
        exemplars=exemplars,
        device="cuda",
        latent_dim=args.latent_dim,
    )

    # Load grid (without training GP yet - we need VAE first)
    log("\n" + "=" * 60)
    log("Loading Pre-evaluated Grid")
    log("=" * 60)

    grid_prompts = optimizer.load_grid(
        args.grid_path,
        top_k=args.top_k,
        train_exemplar_gp=False,  # Don't train GP on original embeddings
    )

    best_prompt = optimizer.best_grid_prompt
    log(f"\nBest from grid:")
    log(f"  Instruction ID: {best_prompt.instruction_id}")
    log(f"  Exemplar ID: {best_prompt.exemplar_id}")
    log(f"  Error rate: {best_prompt.error_rate:.4f}")

    # Train VAE FIRST (needed before GP training on decoded embeddings)
    log("\n" + "=" * 60)
    log("Training VAE on Instruction Embeddings")
    log("=" * 60)

    vae_history = optimizer.train_vae(
        epochs=args.vae_epochs,
        lr=args.vae_lr,
        patience=args.vae_patience,
        verbose=True,
    )

    # Train GP on VAE-decoded embeddings (COWBOYS approach)
    # This fixes EI=0 by ensuring GP operates in same distribution as EI optimization
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

    # Initialize inference (uses HbBoPs GP from load_grid, not separate LatentGP)
    log("\n" + "=" * 60)
    log("Initializing Inference Pipeline")
    log("=" * 60)

    # Get best exemplar embedding for EI optimization
    best_exemplar_emb = optimizer.exemplar_embeddings[best_prompt.exemplar_id]

    inference = RobustInference(
        vae=optimizer.get_vae(),
        exemplar_selector=optimizer.get_exemplar_selector(),
        exemplar_emb=best_exemplar_emb,
        gtr=optimizer.gtr,
        device="cuda",
    )

    # Get best latent
    best_latent = optimizer.get_best_latent()
    log(f"Best latent norm: {best_latent.norm().item():.4f}")

    # Iterative optimization loop
    log("\n" + "=" * 60)
    log(f"Starting Iterative Optimization ({args.iterations} iterations)")
    log("=" * 60)

    # Track best across iterations
    best_error = best_prompt.error_rate
    best_instruction = best_prompt.instruction
    best_exemplar_id = best_prompt.exemplar_id
    current_latent = best_latent.clone()

    # Store iteration history
    iteration_history = []

    for iteration in range(args.iterations):
        log(f"\n{'='*60}")
        log(f"Iteration {iteration + 1}/{args.iterations}")
        log(f"{'='*60}")
        log(f"Current best error: {best_error:.4f}")

        # 1. Run optimization pipeline
        log("\n--- EI Optimization & Vec2Text Inversion ---")
        result = inference.full_pipeline(
            initial_latent=current_latent,
            best_y=best_error,
            n_opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            opt_patience=args.opt_patience,
            v2t_beams=args.v2t_beam,
            v2t_max_length=args.v2t_max_length,
            v2t_no_repeat_ngram_size=args.v2t_no_repeat_ngram,
            v2t_repetition_penalty=args.v2t_repetition_penalty,
            verbose=False,
        )

        log(f"  Generated: {result.text[:80]}...")
        log(f"  Cosine similarity: {result.cosine_similarity:.4f}")

        # 2. Visualize EI landscape BEFORE evaluation
        #    (shows the optimization target and re-embedded point)
        log("\n--- Visualizing EI Landscape ---")
        try:
            training_latents, training_errors = get_training_latents(
                vae=optimizer.get_vae(),
                exemplar_selector=optimizer.exemplar_selector,
                device=optimizer.device,
            )
            visualize_ei_landscape(
                vae=optimizer.get_vae(),
                exemplar_selector=optimizer.exemplar_selector,
                exemplar_emb=optimizer.exemplar_embeddings[best_exemplar_id],
                training_latents=training_latents,
                training_errors=training_errors,
                optimized_latent=result.optimized_latent,
                reembedded_latent=result.reembedded_latent,
                best_y=best_error,
                iteration=iteration + 1,
                output_dir=str(Path(__file__).parent / "visualizations"),
                device=optimizer.device,
            )
        except Exception as e:
            log(f"  Warning: Visualization failed: {e}")

        # 3. Evaluate novel prompt
        log("\n--- Evaluating ---")
        novel_prompt = f"{result.text}\n\n{exemplars[best_exemplar_id]}"
        novel_error = evaluate_prompt(
            novel_prompt,
            validation_data,
            args.model,
            args.backend,
        )

        log(f"  Error rate: {novel_error:.4f} (accuracy: {(1-novel_error)*100:.2f}%)")

        # 4. Get decoded embedding for GP
        decoded_emb = optimizer.get_decoded_embedding(result.text)
        exemplar_emb = optimizer.exemplar_embeddings[best_exemplar_id]

        # 5. Add to GP and retrain
        log("\n--- Updating GP ---")
        optimizer.exemplar_selector.add_observation_and_retrain(
            decoded_inst_emb=decoded_emb,
            exemplar_emb=exemplar_emb,
            error_rate=novel_error,
            epochs=100,
            verbose=False,
        )
        log(f"  GP updated with new observation (total samples: {len(optimizer.exemplar_selector.y_train)})")

        # 6. Update best if improved
        improved = novel_error < best_error
        if improved:
            improvement = best_error - novel_error
            best_error = novel_error
            best_instruction = result.text
            log(f"  ★ New best! Improvement: {improvement:.4f}")

        # 7. Store iteration history
        iteration_history.append({
            "iteration": iteration + 1,
            "instruction": result.text,
            "cosine_similarity": result.cosine_similarity,
            "error_rate": novel_error,
            "improved": improved,
            "best_error_so_far": best_error,
        })

        # 8. Update latent for next iteration (use VAE encoding of new instruction)
        with torch.no_grad():
            new_inst_emb = optimizer.gtr.encode_tensor(result.text).to(optimizer.device)
            current_latent = optimizer.get_vae().get_latent(new_inst_emb.unsqueeze(0)).squeeze(0)

    # Final summary
    log("\n" + "=" * 60)
    log(f"Optimization Complete ({args.iterations} iterations)")
    log("=" * 60)
    log(f"Initial best error (grid): {best_prompt.error_rate:.4f}")
    log(f"Final best error: {best_error:.4f}")
    log(f"Total improvement: {best_prompt.error_rate - best_error:.4f}")
    log(f"Best instruction: {best_instruction[:100]}...")

    # Use best instruction for final result
    selected_exemplar = exemplars[best_exemplar_id]
    selected_exemplar_id = best_exemplar_id
    novel_prompt = f"{best_instruction}\n\n{selected_exemplar}"
    novel_error = best_error

    # Save results
    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "args": vars(args),
        "grid_best": {
            "instruction_id": best_prompt.instruction_id,
            "exemplar_id": best_prompt.exemplar_id,
            "instruction": best_prompt.instruction,
            "exemplar": best_prompt.exemplar,
            "error_rate": best_prompt.error_rate,
        },
        "optimized": {
            "instruction": best_instruction,
            "full_prompt": novel_prompt,
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
