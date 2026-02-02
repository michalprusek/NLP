#!/usr/bin/env python
"""
Bayesian Optimization for prompt discovery using GP-UCB guided flow matching.

This script runs the simple BO optimization loop:
1. GP-UCB optimization to find optimal embedding
2. Flow projection to stay on-manifold
3. Decode SONAR embedding to text prompt
4. Evaluate prompt on GSM8K via LLM
5. Update GP surrogate with observation
6. Repeat

Example usage:
    python -m rielbo.run --iterations 100

For long runs in tmux:
    tmux new-session -d -s bo_run "python -m rielbo.run --iterations 100 2>&1 | tee ecoflow/results/bo_$(date +%Y%m%d_%H%M%S).log; exec bash"

Resume from checkpoint:
    python -m rielbo.run --resume ecoflow/results/bo_checkpoints/checkpoint_iter50.pt --iterations 100
"""

import argparse
import logging
import os
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with timestamp and level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization for prompt discovery using GP-UCB guided flow matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (5 iterations)
    python -m rielbo.run --iterations 5 --n-initial 5

    # Full run in tmux
    tmux new-session -d -s bo_run "python -m rielbo.run --iterations 100 2>&1 | tee ecoflow/results/bo_$(date +%Y%m%d_%H%M%S).log"

    # Resume from checkpoint
    python -m rielbo.run --resume ecoflow/results/bo_checkpoints/checkpoint_iter50.pt
        """,
    )

    # Optimization parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of BO iterations to run (default: 100)",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=10,
        help="Number of initial random samples before BO (default: 10)",
    )
    parser.add_argument(
        "--eval-subset-size",
        type=int,
        default=150,
        help="Number of GSM8K examples for prompt evaluation (default: 150)",
    )

    # UCB optimization parameters
    parser.add_argument(
        "--ucb-alpha",
        type=float,
        default=1.96,
        help="UCB exploration weight (default: 1.96 for 95%% CI)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=5,
        help="Number of GP optimization restarts (default: 5)",
    )
    parser.add_argument(
        "--n-opt-steps",
        type=int,
        default=100,
        help="Gradient steps per GP optimization restart (default: 100)",
    )
    parser.add_argument(
        "--opt-lr",
        type=float,
        default=0.1,
        help="Learning rate for GP optimization (default: 0.1)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="ecoflow/results/bo_checkpoints",
        help="Directory to save checkpoints (default: ecoflow/results/bo_checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional)",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to pre-evaluated embeddings .pt file for warm-start initialization",
    )
    parser.add_argument(
        "--warm-start-top-k",
        type=int,
        default=100,
        help="Use top K embeddings from warm-start file (default: 100)",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="LLM model for evaluation (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=str,
        default="results/flow_ot_20260130_003427/checkpoint_final.pt",
        help="Path to trained flow model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for computation (default: cuda:0)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "openai", "deepinfra"],
        help="LLM backend to use (default: vllm)",
    )

    # Guidance parameters
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=1.0,
        help="Flow guidance strength lambda (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LCB exploration weight for flow guidance (default: 1.0)",
    )

    # GP surrogate parameters
    parser.add_argument(
        "--subspace-dim",
        type=int,
        default=16,
        help="BAxUS subspace dimension (default: 16)",
    )
    parser.add_argument(
        "--use-heteroscedastic-gp",
        action="store_true",
        help="Use heteroscedastic GP with binomial noise model",
    )

    # L2-r filtering parameters
    parser.add_argument(
        "--disable-l2r-filter",
        action="store_true",
        help="Disable round-trip fidelity (L2-r) filtering",
    )
    parser.add_argument(
        "--l2r-threshold",
        type=float,
        default=0.5,
        help="L2-r threshold for on-manifold filtering (default: 0.5)",
    )

    # Misc
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for BO optimization."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Lazy imports to avoid loading heavy modules during --help
    import torch
    from rielbo.decoder import SonarDecoder
    from rielbo.gp_surrogate import create_surrogate
    from rielbo.guided_flow import GuidedFlowSampler
    from rielbo.optimization_loop import BOOptimizationLoop
    from rielbo.validate import load_model_from_checkpoint
    from shared.gsm8k_evaluator import GSM8KEvaluator
    from shared.llm_client import create_llm_client

    # Log configuration
    logger.info("=" * 60)
    logger.info("BAYESIAN OPTIMIZATION FOR PROMPT DISCOVERY")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Initial samples: {args.n_initial}")
    logger.info(f"Eval subset size: {args.eval_subset_size}")
    logger.info(f"UCB alpha: {args.ucb_alpha}")
    logger.info(f"GP optimization: {args.n_restarts} restarts x {args.n_opt_steps} steps")
    logger.info(f"Flow checkpoint: {args.flow_checkpoint}")
    logger.info(f"LLM model: {args.model}")
    logger.info(f"L2-r filter: {not args.disable_l2r_filter}")
    logger.info("=" * 60)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load flow model
    logger.info("Loading flow model...")
    flow_model = load_model_from_checkpoint(
        args.flow_checkpoint,
        device=args.device,
        use_ema=True,
    )

    # 2. Initialize SONAR decoder and encoder
    logger.info("Initializing SONAR decoder...")
    decoder = SonarDecoder(device=args.device)

    encoder = None
    if not args.disable_l2r_filter:
        logger.info("Initializing SONAR encoder for L2-r filtering...")
        try:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device(args.device),
            )
        except Exception as e:
            logger.error(f"Failed to load SONAR encoder: {e}")
            raise RuntimeError(f"SONAR encoder required for L2-r filtering: {e}")

    # 3. Create LLM client
    logger.info(f"Creating LLM client: {args.model}")
    llm_client = create_llm_client(
        args.model,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 4. Create GSM8K evaluator
    logger.info("Creating GSM8K evaluator...")
    evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")
    logger.info(f"GSM8K evaluator: {len(evaluator)} test examples")

    # 5. Create GP surrogate
    if args.use_heteroscedastic_gp:
        logger.info("Creating Heteroscedastic GP surrogate...")
        gp = create_surrogate(
            method="heteroscedastic",
            D=1024,
            device=args.device,
            n_eval=args.eval_subset_size,
        )
    else:
        logger.info(f"Creating BAxUS GP surrogate ({args.subspace_dim}D)...")
        gp = create_surrogate(
            method="baxus",
            D=1024,
            device=args.device,
            target_dim=args.subspace_dim,
        )

    # 6. Create guided flow sampler
    logger.info("Creating guided flow sampler...")
    sampler = GuidedFlowSampler(
        flow_model=flow_model,
        gp_surrogate=gp,
        alpha=args.alpha,
        guidance_strength=args.guidance_strength,
        norm_stats=flow_model.norm_stats,
    )

    # 7. Create BO optimization loop
    logger.info("Creating BO optimization loop...")
    optimizer = BOOptimizationLoop(
        flow_model=flow_model,
        gp=gp,
        sampler=sampler,
        decoder=decoder,
        evaluator=evaluator,
        llm_client=llm_client,
        n_initial=args.n_initial,
        eval_subset_size=args.eval_subset_size,
        device=args.device,
        encoder=encoder,
        l2r_threshold=args.l2r_threshold,
        l2r_filter_enabled=not args.disable_l2r_filter,
    )

    # 8. Resume, warm-start, or random initialize
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer.load_checkpoint(args.resume)
        start_iteration = optimizer.iteration + 1
    elif args.warm_start:
        logger.info(f"Warm-starting from: {args.warm_start}")
        init_result = optimizer.warm_start(args.warm_start, top_k=args.warm_start_top_k)
        logger.info(f"Warm-start: {init_result['n_samples']} samples, best={init_result['best_score']:.4f}")
        start_iteration = 1
    else:
        logger.info("Initializing with random samples...")
        init_result = optimizer.initialize()
        logger.info(f"Initialization: {init_result['n_samples']} samples, best={init_result['best_score']:.4f}")
        start_iteration = 1

    # 9. Main optimization loop
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

    for iteration in range(start_iteration, args.iterations + 1):
        result = optimizer.step(
            ucb_alpha=args.ucb_alpha,
            n_restarts=args.n_restarts,
            n_opt_steps=args.n_opt_steps,
            lr=args.opt_lr,
        )

        logger.info(
            f"Iter {iteration}/{args.iterations}: "
            f"score={result['score']:.4f}, UCB={result['ucb_value']:.4f}, "
            f"L2_proj={result['l2_projection']:.4f}, best={result['best_so_far']:.4f}"
        )

        if iteration % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_iter{iteration:04d}.pt"
            )
            optimizer.save_checkpoint(checkpoint_path)

    # 10. Final save and summary
    final_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    optimizer.save_checkpoint(final_checkpoint)

    metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
    optimizer.metrics.save(metrics_path)

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {args.iterations}")
    logger.info(f"Total observations: {optimizer.n_observations}")
    logger.info(f"Best score: {optimizer.best_score:.4f}")
    logger.info(f"Best prompt:\n{optimizer.best_prompt}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if hasattr(llm_client, "cleanup"):
        llm_client.cleanup()


if __name__ == "__main__":
    main()
