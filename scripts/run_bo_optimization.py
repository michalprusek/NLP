#!/usr/bin/env python
"""
Bayesian Optimization for prompt discovery using LCB-guided flow matching.

This script runs the full BO optimization loop:
1. Generate guided samples from flow model using LCB acquisition
2. Decode SONAR embeddings to text prompts
3. Evaluate prompts on GSM8K via LLM
4. Update GP surrogate with observations
5. Repeat for specified number of iterations

Example usage:
    python scripts/run_bo_optimization.py --iterations 100 --batch-size 4

For long runs in tmux (per CLAUDE.md):
    tmux new-session -d -s bo_run "python scripts/run_bo_optimization.py --iterations 100 2>&1 | tee results/bo_$(date +%Y%m%d_%H%M%S).log; exec bash"

Resume from checkpoint:
    python scripts/run_bo_optimization.py --resume results/bo_checkpoints/checkpoint_iter50.pt --iterations 100

Dependencies:
    - Trained flow model checkpoint (--flow-checkpoint)
    - SONAR decoder (fairseq2, sonar)
    - vLLM for fast LLM inference
    - GSM8K dataset in datasets/gsm8k/
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
        description="Run Bayesian Optimization for prompt discovery using LCB-guided flow matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (5 iterations)
    python scripts/run_bo_optimization.py --iterations 5 --n-initial 5 --batch-size 2

    # Full run in tmux
    tmux new-session -d -s bo_run "python scripts/run_bo_optimization.py --iterations 100 2>&1 | tee results/bo_$(date +%Y%m%d_%H%M%S).log"

    # Resume from checkpoint
    python scripts/run_bo_optimization.py --resume results/bo_checkpoints/checkpoint_iter50.pt
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
        "--batch-size",
        type=int,
        default=4,
        help="DEPRECATED: Use --n-candidates instead",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=64,
        help="Number of candidates to generate, best selected by UCB (default: 64)",
    )
    parser.add_argument(
        "--eval-subset-size",
        type=int,
        default=150,
        help="Number of GSM8K examples for prompt evaluation (default: 150)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/bo_checkpoints",
        help="Directory to save checkpoints (default: results/bo_checkpoints)",
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
        help="Use top K embeddings from warm-start file (default: 100 = all)",
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

    # Guidance parameters
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=1.0,
        help="LCB guidance strength lambda (default: 1.0, optimal range 1.0-2.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LCB exploration weight (default: 1.0, use 1.96 for 95%% CI)",
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

    return parser.parse_args()


def main():
    """Main entry point for BO optimization."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Lazy imports to avoid loading heavy modules during --help
    import torch
    from src.ecoflow.decoder import SonarDecoder
    from src.ecoflow.gp_surrogate import create_surrogate
    from src.ecoflow.guided_flow import GuidedFlowSampler
    from src.ecoflow.optimization_loop import BOOptimizationLoop
    from src.ecoflow.validate import load_model_from_checkpoint
    from src.gsm8k_evaluator import GSM8KEvaluator
    from src.llm_client import create_llm_client

    # Log configuration
    logger.info("=" * 60)
    logger.info("BAYESIAN OPTIMIZATION FOR PROMPT DISCOVERY")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Initial samples: {args.n_initial}")
    logger.info(f"Candidates per iteration: {args.n_candidates}")
    logger.info(f"Eval subset size: {args.eval_subset_size}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Checkpoint freq: {args.checkpoint_freq}")
    logger.info(f"Resume from: {args.resume}")
    logger.info(f"LLM model: {args.model}")
    logger.info(f"Flow checkpoint: {args.flow_checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Guidance strength: {args.guidance_strength}")
    logger.info(f"LCB alpha: {args.alpha}")
    logger.info(f"L2-r filter enabled: {not args.disable_l2r_filter}")
    logger.info(f"L2-r threshold: {args.l2r_threshold}")
    logger.info("=" * 60)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")

    # 1. Load flow model
    logger.info("Loading flow model from checkpoint...")
    flow_model = load_model_from_checkpoint(
        args.flow_checkpoint,
        device=args.device,
        use_ema=True,
    )
    logger.info("Flow model loaded successfully")

    # 2. Initialize SONAR decoder and encoder
    logger.info("Initializing SONAR decoder...")
    decoder = SonarDecoder(device=args.device)
    logger.info("SONAR decoder initialized")

    # Initialize encoder for L2-r filtering (if enabled)
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
            logger.info("SONAR encoder initialized")
        except Exception as e:
            logger.warning(f"Failed to load SONAR encoder: {e}. L2-r filtering disabled.")
            encoder = None

    # 3. Create LLM client (vLLM)
    logger.info(f"Creating LLM client: {args.model}")
    llm_client = create_llm_client(
        args.model,
        backend="vllm",
        tensor_parallel_size=args.tensor_parallel_size,
    )
    logger.info("LLM client created")

    # 4. Create GSM8K evaluator
    logger.info("Creating GSM8K evaluator...")
    evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")
    logger.info(f"GSM8K evaluator created with {len(evaluator)} test examples")

    # 5. Create GP surrogate (BAxUS with 128D subspace - best gradient quality)
    logger.info("Creating BAxUS GP surrogate (128D subspace)...")
    gp = create_surrogate(method="baxus", D=1024, device=args.device, target_dim=128)
    logger.info("BAxUS GP surrogate created")

    # 6. Create guided flow sampler
    logger.info("Creating guided flow sampler...")
    sampler = GuidedFlowSampler(
        flow_model=flow_model,
        gp_surrogate=gp,
        alpha=args.alpha,
        guidance_strength=args.guidance_strength,
        norm_stats=flow_model.norm_stats,
    )
    logger.info("Guided flow sampler created")

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
        batch_size=args.batch_size,
        eval_subset_size=args.eval_subset_size,
        device=args.device,
        encoder=encoder,
        l2r_threshold=args.l2r_threshold,
        l2r_filter_enabled=not args.disable_l2r_filter,
    )
    logger.info("BO optimization loop created")

    # 8. Resume, warm-start, or random initialize
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer.load_checkpoint(args.resume)
        start_iteration = optimizer.iteration + 1
        logger.info(f"Resumed at iteration {start_iteration}")
    elif args.warm_start:
        logger.info(f"Warm-starting from: {args.warm_start}")
        init_result = optimizer.warm_start(
            args.warm_start,
            top_k=args.warm_start_top_k
        )
        logger.info(f"Warm-start complete:")
        logger.info(f"  - Pre-evaluated samples: {init_result['n_samples']}")
        logger.info(f"  - Score range: {init_result['score_range'][0]:.4f} - {init_result['score_range'][1]:.4f}")
        logger.info(f"  - Best score: {init_result['best_score']:.4f}")
        start_iteration = 1
    else:
        logger.info("Initializing with random samples...")
        init_result = optimizer.initialize()
        logger.info(f"Initialization complete:")
        logger.info(f"  - Samples: {init_result['n_samples']}")
        logger.info(f"  - Best initial score: {init_result['best_score']:.4f}")
        logger.info(f"  - Best initial prompt:\n{init_result['best_prompt']}")
        start_iteration = 1

    # 9. Main optimization loop
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info(f"Generating {args.n_candidates} candidates per iteration, selecting best by UCB")
    logger.info("=" * 60)

    for iteration in range(start_iteration, args.iterations + 1):
        # Run one BO step with UCB-based candidate selection
        result = optimizer.step(n_candidates=args.n_candidates)

        # Log progress
        logger.info(
            f"Iteration {iteration}/{args.iterations}: "
            f"score={result['score']:.4f}, UCB={result['ucb_value']:.4f}, "
            f"best_so_far={result['best_so_far']:.4f}, "
            f"n_obs={result['n_observations']}"
        )

        # Checkpoint
        if iteration % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_iter{iteration:04d}.pt"
            )
            optimizer.save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # 10. Final save and summary
    final_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    optimizer.save_checkpoint(final_checkpoint)
    logger.info(f"Saved final checkpoint: {final_checkpoint}")

    # Save metrics
    metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
    optimizer.metrics.save(metrics_path)
    logger.info(f"Saved metrics: {metrics_path}")

    # Print final summary
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {args.iterations}")
    logger.info(f"Total observations: {optimizer.n_observations}")
    logger.info(f"Best score: {optimizer.best_score:.4f}")
    logger.info(f"Best prompt:\n{optimizer.best_prompt}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Cleanup
    if hasattr(llm_client, "cleanup"):
        llm_client.cleanup()


if __name__ == "__main__":
    main()
