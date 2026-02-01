#!/usr/bin/env python
"""Run EcoFlow BO with best configuration on full GSM8K test set.

Uses:
- Flow model: U-Net + Spherical-OT (geodesic SLERP interpolation)
- GP: RiemannianGP with ArcCosine kernel (best calibration)
- Guidance: Riemannian (spherical projection + flow-relative + cutoff)
- Evaluation: Full 1319 GSM8K test examples
- Budget: 50k LLM evaluations (~38 iterations)
- Warm start: Top 10 instructions from pre-evaluated set
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_top_instructions(json_path: str, top_k: int = 10) -> tuple:
    """Load top K instructions from JSON and encode to SONAR embeddings.

    Returns:
        Tuple of (embeddings [K, 1024], accuracies [K], instructions [K])
    """
    logger.info(f"Loading top {top_k} instructions from {json_path}...")

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]

    # Sort by accuracy descending
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    top_results = sorted_results[:top_k]

    instructions = [r["instruction"] for r in top_results]
    accuracies = [r["accuracy"] for r in top_results]

    logger.info(f"  Top {top_k} accuracies: {[f'{a:.4f}' for a in accuracies]}")

    # Encode to SONAR embeddings
    logger.info("  Encoding instructions to SONAR embeddings...")
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
    )

    embeddings = encoder.predict(instructions, source_lang="eng_Latn")
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    accuracies_tensor = torch.tensor(accuracies, dtype=torch.float32)

    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Best instruction (acc={accuracies[0]:.4f}):\n{instructions[0]}")

    return embeddings, accuracies_tensor, instructions


def main():
    parser = argparse.ArgumentParser(description="EcoFlow BO with best GP on full GSM8K")
    parser.add_argument("--flow-checkpoint", type=str,
                        default="study/checkpoints/unet-spherical-ot-10k-none/best.pt")
    parser.add_argument("--warm-start-json", type=str,
                        default="datasets/evaluated_instructions/gsm8k_100_instructions.json",
                        help="JSON file with pre-evaluated instructions")
    parser.add_argument("--warm-start-k", type=int, default=10,
                        help="Number of top instructions for warm start")
    parser.add_argument("--gp-kernel", type=str, default="arccosine",
                        choices=["arccosine", "geodesic_matern52"])
    parser.add_argument("--llm-budget", type=int, default=50000,
                        help="Total LLM evaluation budget")
    parser.add_argument("--eval-size", type=int, default=1319,
                        help="GSM8K test set size for evaluation (1319 = full test)")
    parser.add_argument("--model", type=str, default="qwen",
                        help="LLM model for evaluation")
    parser.add_argument("--results-dir", type=str, default="ecoflow/results")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # Calculate iterations from budget
    max_iterations = args.llm_budget // args.eval_size
    logger.info(f"LLM budget: {args.llm_budget}, eval size: {args.eval_size}")
    logger.info(f"Max iterations: {max_iterations}")

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"bo_{args.gp_kernel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load flow model
    logger.info(f"Loading flow model from {args.flow_checkpoint}...")
    from study.flow_matching.models import create_model
    from ecoflow.flow_model import FlowMatchingModel

    ckpt = torch.load(args.flow_checkpoint, map_location=device)
    flow_config = ckpt["config"]

    velocity_net = create_model(
        arch=flow_config["arch"],
        scale="small",
    )
    velocity_net.load_state_dict(ckpt["model_state_dict"])
    velocity_net = velocity_net.to(device).eval()

    flow_model = FlowMatchingModel(
        velocity_net=velocity_net,
        norm_stats=ckpt.get("normalization_stats"),
    )
    logger.info(f"  Flow model loaded: {flow_config['arch']}")

    # Create GP surrogate
    logger.info(f"Creating RiemannianGP with {args.gp_kernel} kernel...")
    from ecoflow.guided_flow import create_optimal_gp_for_guided_flow, GuidedFlowSampler

    gp = create_optimal_gp_for_guided_flow(
        input_dim=1024,
        kernel=args.gp_kernel,
        device=device,
    )
    logger.info(f"  GP created: method={gp.config.method}, kernel={gp.config.kernel}")

    # Create guided sampler with full Riemannian guidance
    sampler = GuidedFlowSampler(
        flow_model=flow_model,
        gp_surrogate=gp,
        alpha=1.96,
        guidance_strength=1.0,
        zero_init_fraction=0.04,
        guidance_schedule="riemannian",  # Full Riemannian: spherical + cutoff + flow-relative
        uncertainty_beta=2.0,            # Higher = more exploitation in confident regions
        guidance_cutoff=0.8,             # Stop guidance at 80%, let flow smooth last 20%
        spherical_projection=True,       # Project gradient onto tangent plane
        flow_relative_scaling=True,      # Scale gradient relative to flow velocity
    )
    logger.info(f"  GuidedFlowSampler created: schedule={sampler.guidance_schedule.value}, "
                f"cutoff={sampler.guidance_cutoff}, spherical={sampler.spherical_projection}")

    # Load SONAR decoder
    logger.info("Loading SONAR decoder...")
    from ecoflow.decoder import SonarDecoder
    decoder = SonarDecoder(device=device)
    logger.info("  SONAR decoder loaded")

    # Create evaluator
    logger.info("Creating GSM8K evaluator...")
    from shared.gsm8k_evaluator import GSM8KEvaluator
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test",
    )
    logger.info(f"  Evaluator created: {len(evaluator)} examples")

    # Create LLM client
    logger.info(f"Creating LLM client: {args.model}...")
    from shared.llm_client import create_llm_client
    llm_client = create_llm_client(args.model, backend="vllm")
    logger.info("  LLM client ready")

    # Create optimization loop
    from ecoflow.optimization_loop import BOOptimizationLoop

    loop = BOOptimizationLoop(
        flow_model=flow_model,
        gp=gp,
        sampler=sampler,
        decoder=decoder,
        evaluator=evaluator,
        llm_client=llm_client,
        n_initial=10,
        eval_subset_size=args.eval_size,  # Full 1319 examples
        device=device,
        l2r_filter_enabled=False,  # Disable L2-r filtering for speed
    )

    # Initialize or resume
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        loop.load_checkpoint(args.resume)
        start_iter = loop.iteration
    else:
        # Warm start from top K pre-evaluated instructions
        embeddings, accuracies, instructions = load_top_instructions(
            args.warm_start_json, top_k=args.warm_start_k
        )

        # Move to device and fit GP
        embeddings = embeddings.to(device)
        accuracies = accuracies.to(device)

        loop.train_X = embeddings
        loop.train_Y = accuracies
        loop.best_score = accuracies.max().item()
        best_idx = accuracies.argmax().item()
        loop.best_prompt = instructions[best_idx]
        loop.best_so_far_list = [loop.best_score]

        # Store prompts for reference
        for i, instr in enumerate(instructions):
            loop._prompts[i] = instr

        # Fit GP on warm start data
        gp.fit(embeddings, accuracies)
        sampler.update_gp(gp)

        # Log initial metrics
        loop.metrics.log_iteration(
            iteration=0,
            score=loop.best_score,
            best_so_far=loop.best_score,
            n_observations=len(instructions),
        )

        logger.info(f"Warm start complete: {len(instructions)} samples, best={loop.best_score:.4f}")
        logger.info(f"Best prompt: {loop.best_prompt}")
        start_iter = 0

    # Run optimization loop
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting BO optimization: {max_iterations} iterations")
    logger.info(f"Evaluating on {args.eval_size} examples per iteration")
    logger.info(f"{'='*60}\n")

    checkpoint_path = results_dir / f"{run_name}_checkpoint.pt"

    try:
        for i in range(start_iter, max_iterations):
            result = loop.step(
                ucb_alpha=1.96,
                n_restarts=5,
                n_opt_steps=100,
                lr=0.1,
                n_candidates=512,
            )

            logger.info(
                f"Iteration {result['iteration']}/{max_iterations}: "
                f"score={result['score']:.4f}, best={result['best_so_far']:.4f}, "
                f"n_obs={result['n_observations']}"
            )

            # Checkpoint every 5 iterations
            if (i + 1) % 5 == 0:
                loop.save_checkpoint(str(checkpoint_path))
                logger.info(f"  Checkpoint saved to {checkpoint_path}")

            # Log prompt (never truncate)
            logger.info(f"  Prompt: {result['prompt']}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving checkpoint...")
        loop.save_checkpoint(str(checkpoint_path))

    # Save final results
    final_path = results_dir / f"{run_name}_final.pt"
    loop.save_checkpoint(str(final_path))

    metrics_path = results_dir / f"{run_name}_metrics.json"
    loop.metrics.save(str(metrics_path))

    logger.info(f"\n{'='*60}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total iterations: {loop.iteration}")
    logger.info(f"Total observations: {loop.n_observations}")
    logger.info(f"Best score: {loop.best_score:.4f}")
    logger.info(f"Best prompt:\n{loop.best_prompt}")
    logger.info(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
