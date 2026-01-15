#!/usr/bin/env python3
"""
FlowPO Bayesian Optimization Loop with Full LLM Evaluation.

Runs GP-guided flow inference with actual GSM8K evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m lido_pp.scripts.run_flowpo_bo \
        --iterations 20 \
        --candidates-per-iter 8 \
        --eval-samples 1319
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_tfa_checkpoint(checkpoint_path: str, device: str = "cuda:0") -> nn.Module:
    """Load TFA model from checkpoint."""
    from lido_pp.backbone.cfm_encoder import TextFlowAutoencoder

    logger.info(f"Loading TFA from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    input_dim = ckpt.get("input_dim", 1024)
    latent_dim = ckpt.get("latent_dim", 256)
    args = ckpt.get("args", {})
    if isinstance(args, dict):
        flow_dim = args.get("flow_dim", 512)
        velocity_layers = args.get("velocity_layers", 6)
        dropout = args.get("dropout", 0.0)
    else:
        flow_dim = 512
        velocity_layers = 6
        dropout = 0.0

    tfa = TextFlowAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        flow_dim=flow_dim,
        num_velocity_layers=velocity_layers,
        dropout=dropout,
    ).to(device)
    tfa.load_state_dict(ckpt["model_state_dict"])
    tfa.eval()

    logger.info(f"TFA loaded: latent_dim={latent_dim}, flow_dim={flow_dim}, val_cos={ckpt.get('val_cos_ode', 'N/A'):.4f}")
    return tfa


def load_hbbops_evaluations(hbbops_path: str, min_fidelity: int = 600) -> List[Dict]:
    """Load existing evaluations."""
    logger.info(f"Loading evaluations from {hbbops_path}")
    with open(hbbops_path) as f:
        data = json.load(f)

    evaluations = []
    for idx, item in data["results"].items():
        if item["fidelity"] >= min_fidelity:
            evaluations.append({
                "instruction": item["instruction"],
                "accuracy": item["accuracy"],
                "error_rate": item["error_rate"],
                "fidelity": item["fidelity"],
            })

    evaluations.sort(key=lambda x: x["error_rate"])
    logger.info(f"Loaded {len(evaluations)} evaluations (fidelity >= {min_fidelity})")
    return evaluations


def encode_instructions(
    instructions: List[str],
    sonar_encoder,
    tfa: nn.Module,
    device: str,
) -> torch.Tensor:
    """Encode instructions to latent space."""
    embeddings = sonar_encoder.encode(instructions)
    embeddings = torch.tensor(embeddings, device=device, dtype=torch.float32)
    with torch.no_grad():
        latents = tfa.encode(embeddings)
    return latents


def decode_latents_to_text(
    latents: torch.Tensor,
    tfa: nn.Module,
    sonar_decoder,
    typical_sonar_norm: float = 0.18,
) -> List[str]:
    """Decode latents back to text instructions.

    CRITICAL: TFA outputs normalized embeddings (unit vectors), but SONAR
    decoder expects unnormalized embeddings with typical norm ~0.18.
    We scale the output to typical SONAR norm before decoding.
    """
    with torch.no_grad():
        # TFA decode to normalized embedding
        embeddings = tfa.decode(latents, normalize=True)
        # Scale to typical SONAR norm for proper decoding
        embeddings = embeddings * typical_sonar_norm
    texts = sonar_decoder.decode(embeddings)
    return texts


def evaluate_instruction(
    instruction: str,
    llm_client,
    dataset,
    num_samples: int = 1319,
) -> Tuple[float, float]:
    """
    Evaluate instruction on GSM8K.

    Returns:
        (accuracy, error_rate)
    """
    from src.gsm8k_evaluator import extract_answer, extract_ground_truth

    correct = 0
    total = min(num_samples, len(dataset))

    for i in range(total):
        example = dataset[i]
        question = example["question"]
        answer = example["answer"]

        # Format prompt (Q_end style from OPRO paper)
        prompt = f"Q: {question}\n{instruction}\nA:"

        # Get model response
        response = llm_client.generate(prompt, max_tokens=512, temperature=0.0)

        # Extract and compare answers
        pred = extract_answer(response)
        gt = extract_ground_truth(answer)

        if pred is not None and gt is not None:
            try:
                if abs(float(pred) - float(gt)) < 1e-6:
                    correct += 1
            except ValueError:
                if pred.strip() == gt.strip():
                    correct += 1

    accuracy = correct / total
    error_rate = 1.0 - accuracy
    return accuracy, error_rate


def generate_candidates(
    tfa: nn.Module,
    gp,
    train_latents: torch.Tensor,
    num_candidates: int,
    guidance_scale: float,
    exploration_noise: float,
    device: str,
) -> torch.Tensor:
    """Generate candidate latents via GP-guided flow."""
    from lido_pp.flow.gp_guided_flow import GPGuidedFlowGenerator
    from lido_pp.scripts.run_gp_guided_inference import TFAVelocityWrapper, LatentGPWrapper

    # Compute training distribution
    train_mean = train_latents.mean(dim=0)
    train_std = train_latents.std(dim=0).clamp(min=1e-6)

    # Sample initial latents from training distribution
    noise = torch.randn(num_candidates, train_latents.shape[1], device=device)
    z_init = train_mean.unsqueeze(0) + exploration_noise * train_std.unsqueeze(0) * noise

    # Project to flow space
    with torch.no_grad():
        x0_flow = tfa.from_latent(z_init)

    # Create wrappers
    velocity_wrapper = TFAVelocityWrapper(tfa).to(device)
    gp_wrapper = LatentGPWrapper(gp, tfa.to_latent, tfa.from_latent).to(device)

    # Create generator
    generator = GPGuidedFlowGenerator(
        flowdit=velocity_wrapper,
        latent_dim=tfa.flow_dim,
        guidance_scale=guidance_scale,
        schedule="linear",
        ucb_beta=4.0,
    )
    generator.set_gp_model(gp_wrapper)
    generator = generator.to(device)

    # Generate
    result = generator.generate(
        batch_size=num_candidates,
        num_steps=20,
        acquisition="ucb",
        initial_noise=x0_flow,
    )

    # Project to latent
    with torch.no_grad():
        latents = tfa.to_latent(result.latents)

    return latents


def main():
    parser = argparse.ArgumentParser(description="FlowPO BO Loop with LLM Evaluation")

    # Paths
    parser.add_argument("--tfa-checkpoint", default="lido_pp/checkpoints/tfa_best.pt")
    parser.add_argument("--hbbops-path", default="lipo/data/hbbops_results_20260102.json")
    parser.add_argument("--output-dir", default="lido_pp/results/bo_runs")

    # BO parameters
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--candidates-per-iter", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=1319, help="GSM8K samples for evaluation")

    # GP parameters
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--exploration-noise", type=float, default=0.5)
    parser.add_argument("--min-fidelity", type=int, default=600)

    # LLM
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", default="vllm")

    # Device
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=" * 60)
    logger.info("FlowPO Bayesian Optimization")
    logger.info("=" * 60)
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Candidates per iter: {args.candidates_per_iter}")
    logger.info(f"Eval samples: {args.eval_samples}")
    logger.info(f"Output: {run_dir}")

    # Load components
    logger.info("\n[1/5] Loading TFA...")
    tfa = load_tfa_checkpoint(args.tfa_checkpoint, args.device)

    logger.info("\n[2/5] Loading SONAR encoder/decoder...")
    from lido_pp.backbone.sonar_encoder import SONAREncoder, SONARTextDecoder
    # TFA was trained on normalized embeddings, so use normalize=True
    sonar_encoder = SONAREncoder(device=args.device, normalize=True)
    sonar_decoder = SONARTextDecoder(device=args.device)

    # Compute typical SONAR norm for decoding (before normalization)
    # SONAR decoder expects unnormalized embeddings with norm ~0.18
    sonar_encoder_unnorm = SONAREncoder(device=args.device, normalize=False)
    sample_texts = ["Let's think step by step.", "Solve this problem carefully."]
    typical_sonar_norm = sonar_encoder_unnorm.encode(sample_texts).norm(dim=-1).mean().item()
    logger.info(f"Typical SONAR norm: {typical_sonar_norm:.4f}")
    del sonar_encoder_unnorm  # Free memory

    logger.info("\n[3/5] Loading LLM...")
    from src.llm_client import create_llm_client
    # Limit GPU memory to 70% to leave room for SONAR decoder (~8GB)
    llm_client = create_llm_client(
        args.model,
        backend=args.backend,
        gpu_memory_utilization=0.70,
    )

    logger.info("\n[4/5] Loading GSM8K dataset...")
    from datasets import load_from_disk
    dataset = load_from_disk("datasets/gsm8k/test")
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("\n[5/5] Loading initial evaluations...")
    evaluations = load_hbbops_evaluations(args.hbbops_path, args.min_fidelity)

    # Encode initial instructions
    instructions = [e["instruction"] for e in evaluations]
    error_rates = torch.tensor([e["error_rate"] for e in evaluations], device=args.device)
    latents = encode_instructions(instructions, sonar_encoder, tfa, args.device)

    # Initialize GP
    logger.info("\n[6/6] Initializing GP...")
    from lido_pp.gp.high_dim_gp import IsotropicHighDimGP
    gp = IsotropicHighDimGP(
        latent_dim=tfa.latent_dim,
        device=args.device,
        ucb_beta=4.0,
    )
    gp.fit(latents, error_rates)

    # Track results
    all_results = []
    best_error = error_rates.min().item()
    best_instruction = evaluations[error_rates.argmin().item()]["instruction"]

    logger.info(f"\nInitial best error: {best_error:.4f}")
    logger.info(f"Initial best instruction: {best_instruction[:100]}...")

    # BO Loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting BO Loop")
    logger.info("=" * 60)

    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n--- Iteration {iteration}/{args.iterations} ---")

        # Generate candidates
        logger.info("Generating candidates...")
        candidate_latents = generate_candidates(
            tfa=tfa,
            gp=gp,
            train_latents=latents,
            num_candidates=args.candidates_per_iter,
            guidance_scale=args.guidance_scale,
            exploration_noise=args.exploration_noise,
            device=args.device,
        )

        # Decode to text
        logger.info("Decoding to text...")
        candidate_texts = decode_latents_to_text(
            candidate_latents, tfa, sonar_decoder, typical_sonar_norm
        )

        # Evaluate each candidate
        iter_results = []
        for i, text in enumerate(candidate_texts):
            logger.info(f"  Evaluating candidate {i+1}/{len(candidate_texts)}...")
            logger.info(f"    Instruction: {text[:80]}...")

            accuracy, error_rate = evaluate_instruction(
                text, llm_client, dataset, args.eval_samples
            )

            logger.info(f"    Accuracy: {accuracy:.4f}, Error: {error_rate:.4f}")

            iter_results.append({
                "iteration": iteration,
                "candidate_idx": i,
                "instruction": text,
                "accuracy": accuracy,
                "error_rate": error_rate,
            })

            # Update best
            if error_rate < best_error:
                best_error = error_rate
                best_instruction = text
                logger.info(f"    *** NEW BEST! Error: {best_error:.4f} ***")

        all_results.extend(iter_results)

        # Update GP with new data
        new_latents = candidate_latents
        new_errors = torch.tensor(
            [r["error_rate"] for r in iter_results],
            device=args.device
        )

        latents = torch.cat([latents, new_latents], dim=0)
        error_rates = torch.cat([error_rates, new_errors], dim=0)

        gp.fit(latents, error_rates)
        logger.info(f"GP updated: {len(latents)} total points")

        # Save intermediate results
        torch.save({
            "iteration": iteration,
            "latents": latents.cpu(),
            "error_rates": error_rates.cpu(),
            "results": all_results,
            "best_error": best_error,
            "best_instruction": best_instruction,
        }, os.path.join(run_dir, f"checkpoint_iter_{iteration:03d}.pt"))

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("BO Complete!")
    logger.info("=" * 60)
    logger.info(f"Total evaluations: {len(all_results)}")
    logger.info(f"Best error rate: {best_error:.4f}")
    logger.info(f"Best instruction:\n{best_instruction}")

    # Save final results
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
            "best_error": best_error,
            "best_instruction": best_instruction,
        }, f, indent=2)

    logger.info(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
