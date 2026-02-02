#!/usr/bin/env python
"""Run Latent Space Bayesian Optimization.

Latent BO operates in flow's noise space z ~ N(0,I) instead of embedding space x.
This gives better GP behavior since z-space is Gaussian.

Usage:
    # Pure Latent BO (simple, fast)
    uv run python -m ecoflow.run_latent_bo --llm-budget 50000

    # Hybrid: Latent BO + Guided Flow refinement
    uv run python -m ecoflow.run_latent_bo --llm-budget 50000 --use-guided-refinement
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_warm_start_data(json_path: str, top_k: int = 10) -> tuple:
    """Load top K instructions and encode to SONAR embeddings."""
    logger.info(f"Loading top {top_k} instructions from {json_path}...")

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
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
    parser = argparse.ArgumentParser(description="Latent Space BO for prompt optimization")
    parser.add_argument("--flow-checkpoint", type=str,
                        default="study/checkpoints/unet-spherical-ot-10k-none/best.pt",
                        help="Path to flow model checkpoint")
    parser.add_argument("--warm-start-json", type=str,
                        default="datasets/evaluated_instructions/gsm8k_100_instructions.json",
                        help="JSON file with pre-evaluated instructions")
    parser.add_argument("--warm-start-k", type=int, default=10,
                        help="Number of top instructions for warm start")
    parser.add_argument("--llm-budget", type=int, default=50000,
                        help="Total LLM evaluation budget")
    parser.add_argument("--eval-size", type=int, default=1319,
                        help="GSM8K test set size for evaluation")
    parser.add_argument("--model", type=str, default="qwen",
                        help="LLM model for evaluation")
    parser.add_argument("--use-guided-refinement", action="store_true",
                        help="Use guided flow refinement after z-space optimization")
    parser.add_argument("--gp-kernel", type=str, default="arccosine",
                        choices=["arccosine", "geodesic_matern52", "matern52"],
                        help="GP kernel for x-space (only used with --use-guided-refinement)")
    parser.add_argument("--results-dir", type=str, default="ecoflow/results")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--inversion-steps", type=int, default=100,
                        help="ODE steps for initial embedding inversion")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Start from random z instead of warm start embeddings")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of iterations (overrides llm-budget calculation)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # Calculate iterations
    if args.iterations is not None:
        max_iterations = args.iterations
        logger.info(f"Iterations: {max_iterations} (specified directly)")
    else:
        max_iterations = args.llm_budget // args.eval_size
        logger.info(f"LLM budget: {args.llm_budget}, eval size: {args.eval_size}")
        logger.info(f"Max iterations: {max_iterations}")

    # Results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = "guided" if args.use_guided_refinement else "pure"
    run_name = f"latent_bo_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    flow_method = flow_config.get("flow", "")
    is_spherical = "spherical" in flow_method.lower()

    flow_model = FlowMatchingModel(
        velocity_net=velocity_net,
        norm_stats=ckpt.get("normalization_stats"),
        is_spherical=is_spherical,
    )
    logger.info(f"  Flow model: {flow_config['arch']}, flow={flow_method}, spherical={is_spherical}")

    # Load SONAR decoder
    logger.info("Loading SONAR decoder...")
    from ecoflow.decoder import SonarDecoder
    decoder = SonarDecoder(device=device)

    # Create evaluator
    logger.info("Creating GSM8K evaluator...")
    from shared.gsm8k_evaluator import GSM8KEvaluator
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split="test",
    )
    logger.info(f"  Evaluator: {len(evaluator)} examples")

    # Create LLM client
    logger.info(f"Creating LLM client: {args.model}...")
    from shared.llm_client import create_llm_client
    llm_client = create_llm_client(args.model, backend="vllm")

    # Optional: x-space GP and guided sampler for hybrid mode
    gp_x = None
    guided_sampler = None

    if args.use_guided_refinement:
        logger.info("Setting up guided refinement components...")
        from ecoflow.guided_flow import create_optimal_gp_for_guided_flow, GuidedFlowSampler

        gp_x = create_optimal_gp_for_guided_flow(
            input_dim=1024,
            kernel=args.gp_kernel,
            device=device,
        )

        guided_sampler = GuidedFlowSampler(
            flow_model=flow_model,
            gp_surrogate=gp_x,
            alpha=1.96,
            guidance_strength=1.0,
            guidance_schedule="riemannian",
            guidance_cutoff=0.8,
            spherical_projection=True,
        )
        logger.info(f"  Guided sampler: kernel={args.gp_kernel}")

    # Create Latent BO
    logger.info("Creating Latent Space BO...")
    from ecoflow.latent_bo import LatentSpaceBO

    latent_bo = LatentSpaceBO(
        flow_model=flow_model,
        decoder=decoder,
        evaluator=evaluator,
        llm_client=llm_client,
        gp_x_space=gp_x,
        guided_sampler=guided_sampler,
        eval_subset_size=args.eval_size,
        device=device,
        use_guided_refinement=args.use_guided_refinement,
    )

    # Initialize or resume
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        latent_bo.load_checkpoint(args.resume)
        start_iter = latent_bo.iteration
    elif args.no_warm_start:
        # Cold start: generate random z, decode, evaluate first prompt
        logger.info("Cold start: initializing from random z...")
        init_result = latent_bo.initialize_from_random(n_initial=1)
        logger.info(f"Cold start complete: {init_result['n_samples']} samples")
        logger.info(f"Initial score: {init_result['best_score']:.4f}")
        logger.info(f"Initial prompt: {init_result['best_prompt']}")
        start_iter = 0
    else:
        # Warm start
        embeddings, accuracies, instructions = load_warm_start_data(
            args.warm_start_json, top_k=args.warm_start_k
        )

        init_result = latent_bo.initialize_from_embeddings(
            embeddings=embeddings.to(device),
            scores=accuracies.to(device),
            instructions=instructions,
            inversion_steps=args.inversion_steps,
        )

        logger.info(f"Warm start complete: {init_result['n_samples']} samples")
        logger.info(f"Mean inversion error: {init_result['mean_inversion_error']:.4f}")
        start_iter = 0

    # Run optimization
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Latent BO: {max_iterations} iterations")
    logger.info(f"Mode: {'Hybrid (z-space GP + guided flow)' if args.use_guided_refinement else 'Pure (z-space GP only)'}")
    logger.info(f"{'='*60}\n")

    checkpoint_path = results_dir / f"{run_name}_checkpoint.pt"

    try:
        for i in range(start_iter, max_iterations):
            result = latent_bo.step(
                alpha=1.96,
                n_candidates=512,
                n_restarts=10,
            )

            logger.info(
                f"Iteration {result['iteration']}/{max_iterations}: "
                f"score={result['score']:.4f}, best={result['best_so_far']:.4f}, "
                f"z_norm={result['z_norm']:.2f}, x_norm={result['x_norm']:.4f}"
            )

            # Checkpoint every 5 iterations
            if (i + 1) % 5 == 0:
                latent_bo.save_checkpoint(str(checkpoint_path))

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Saving checkpoint...")
        latent_bo.save_checkpoint(str(checkpoint_path))

    # Save final results
    final_path = results_dir / f"{run_name}_final.pt"
    latent_bo.save_checkpoint(str(final_path))

    logger.info(f"\n{'='*60}")
    logger.info("LATENT BO COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total iterations: {latent_bo.iteration}")
    logger.info(f"Total observations: {len(latent_bo.train_Z)}")
    logger.info(f"Best score: {latent_bo.best_score:.4f}")
    logger.info(f"Best prompt:\n{latent_bo.best_prompt}")
    logger.info(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
