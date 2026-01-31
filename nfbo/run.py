#!/usr/bin/env python
"""
NF-BO Optimization for prompt discovery (Adaptation of Lee et al. 2025).

This script runs the NF-BO loop:
1. Initialize with samples from a pre-trained Flow Matching model (EcoFlow) to get valid seeds.
2. Train a RealNVP flow on the highest-scoring samples found so far.
3. Sample new candidates from the RealNVP flow (which approximates p(x|high_score)).
4. Evaluate and repeat.

Example usage:
    python -m nfbo.run --iterations 50 --n-initial 20
"""

import argparse
import logging
import os
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NF-BO for prompt discovery.")

    # Optimization parameters
    parser.add_argument("--iterations", type=int, default=100, help="Number of BO iterations")
    parser.add_argument("--n-initial", type=int, default=20, help="Number of initial samples")
    parser.add_argument("--n-candidates", type=int, default=64, help="Number of candidates per step")
    parser.add_argument("--eval-subset-size", type=int, default=150, help="GSM8K subset size")

    # NF-BO specific
    parser.add_argument("--top-k-percentile", type=float, default=20.0, help="Train flow on top K%% samples")
    parser.add_argument("--flow-lr", type=float, default=1e-3, help="Learning rate for RealNVP")
    parser.add_argument("--flow-epochs", type=int, default=50, help="Epochs to train RealNVP per step")
    parser.add_argument("--flow-layers", type=int, default=6, help="Number of coupling layers")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dim for coupling layers")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="nfbo/results", help="Output directory")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Save frequency")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--warm-start", type=str, default=None, help="Warm start .pt file")

    # Model/Env
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model")
    # We still need the base flow for initialization
    parser.add_argument("--flow-checkpoint", type=str, default="results/flow_ot_20260130_003427/checkpoint_final.pt")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers", "openai", "deepinfra"], help="LLM backend")

    # Misc
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    # L2-r filtering (still useful for NF-BO to ensure we don't eval garbage)
    parser.add_argument("--disable-l2r-filter", action="store_true")
    parser.add_argument("--l2r-threshold", type=float, default=0.5)

    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Imports
    import torch
    from ecoflow.decoder import SonarDecoder
    from ecoflow.validate import load_model_from_checkpoint
    from shared.gsm8k_evaluator import GSM8KEvaluator
    from shared.llm_client import create_llm_client
    # Needed for loop instantiation (expects gp/sampler args even if unused)
    from ecoflow.gp_surrogate import SonarGPSurrogate
    from ecoflow.guided_flow import GuidedFlowSampler

    from nfbo.sampler import NFBoSampler
    from nfbo.loop import NFBoLoop

    logger.info("=" * 60)
    logger.info("NF-BO: NORMALIZING FLOW BAYESIAN OPTIMIZATION")
    logger.info("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load EcoFlow model (for initialization of valid embeddings)
    logger.info("Loading pre-trained EcoFlow model for initialization...")
    try:
        flow_model = load_model_from_checkpoint(
            args.flow_checkpoint,
            device=args.device,
            use_ema=True,
        )
    except Exception as e:
        logger.warning(f"Could not load flow checkpoint: {e}")
        logger.warning("Initialization will be pure noise (likely poor performance).")
        # specific handling if needed, but load_model usually raises
        flow_model = None

    # 2. Decoder
    decoder = SonarDecoder(device=args.device)

    # L2-r encoder
    encoder = None
    if not args.disable_l2r_filter:
        try:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device(args.device),
            )
        except Exception as e:
            logger.warning(f"L2-r encoder failed: {e}. disabling filter.")

    # 3. LLM Client
    llm_client = create_llm_client(args.model, backend=args.backend, tensor_parallel_size=args.tensor_parallel_size)

    # 4. Evaluator
    evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")

    # 5. NF-BO Components
    logger.info("Initializing NF-BO Sampler...")
    nfbo_sampler = NFBoSampler(
        dim=1024,
        device=args.device,
        # top_k_percentile removed as we use all data for manifold learning in Latent BO
        flow_lr=args.flow_lr,
        flow_epochs=args.flow_epochs,
        n_flow_layers=args.flow_layers,
        hidden_dim=args.hidden_dim,
    )

    # Dummy GP and Sampler for base class compatibility (if needed)
    # The BOOptimizationLoop.__init__ requires them, so we create minimal ones
    dummy_gp = SonarGPSurrogate(D=1024, device=args.device)
    # dummy_gp.fit(...) is called in initialize, so it should work even if empty

    # We pass the real flow_model so the base initialize() works
    class DummyGuidedSampler:
         def update_gp(self, gp): pass

    dummy_sampler = DummyGuidedSampler()

    logger.info("Creating NFBoLoop...")
    loop = NFBoLoop(
        nfbo_sampler=nfbo_sampler,
        flow_model=flow_model,
        gp=dummy_gp, # Base class uses this for metrics/tracking
        sampler=dummy_sampler, # Base class calls update_gp on this
        decoder=decoder,
        evaluator=evaluator,
        llm_client=llm_client,
        n_initial=args.n_initial,
        eval_subset_size=args.eval_subset_size,
        device=args.device,
        encoder=encoder,
        l2r_threshold=args.l2r_threshold,
        l2r_filter_enabled=(encoder is not None),
    )

    # 6. Run
    start_iter = 1
    if args.resume:
        loop.load_checkpoint(args.resume)
        start_iter = loop.iteration + 1
    elif args.warm_start:
        loop.warm_start(args.warm_start)
    else:
        loop.initialize()

    for i in range(start_iter, args.iterations + 1):
        result = loop.step(n_candidates=args.n_candidates)

        if i % args.checkpoint_freq == 0:
            loop.save_checkpoint(f"{args.checkpoint_dir}/checkpoint_iter{i:04d}.pt")

    loop.save_checkpoint(f"{args.checkpoint_dir}/checkpoint_final.pt")
    loop.metrics.save(f"{args.checkpoint_dir}/metrics.json")

    logger.info("Optimization Complete.")

    # =========================================================================
    # TEST SET EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)

    best_prompt = loop.best_prompt
    logger.info(f"Best prompt from optimization:\n{best_prompt}")
    logger.info(f"Best validation score: {loop.best_score:.4f}")

    # Evaluate on full test set
    test_evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")
    logger.info(f"Test set: {len(test_evaluator)} examples")

    # Get all test examples
    test_batch = test_evaluator.get_batch(0, len(test_evaluator))
    test_questions = [ex['question'] for ex in test_batch]
    test_prompts = [f"Q: {q}\n{best_prompt}\nA:" for q in test_questions]

    logger.info("Evaluating best prompt on full test set...")
    test_outputs = llm_client.generate_batch(
        test_prompts, temperature=0.0, max_new_tokens=2048
    )

    test_indices = [ex['idx'] for ex in test_batch]
    test_results = test_evaluator.evaluate_batch(test_outputs, test_indices)
    test_accuracy = test_results['accuracy']
    test_error = 1.0 - test_accuracy

    logger.info(f"Test accuracy: {test_accuracy:.2%}")
    logger.info(f"Test error: {test_error:.2%}")
    logger.info("=" * 60)

    # Save final results with test accuracy
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "method": "NFBO",
        "model": args.model,
        "timestamp": timestamp,
        "config": {
            "iterations": args.iterations,
            "n_initial": args.n_initial,
            "eval_subset_size": args.eval_subset_size,
            "flow_epochs": args.flow_epochs,
            "flow_layers": args.flow_layers,
        },
        "best_prompt": best_prompt,
        "validation_accuracy": loop.best_score,
        "test_accuracy": test_accuracy,
        "test_error": test_error,
    }

    results_file = f"{args.checkpoint_dir}/nfbo_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    # Save best prompt to text file
    prompt_file = f"{args.checkpoint_dir}/nfbo_{timestamp}.txt"
    with open(prompt_file, "w", encoding='utf-8') as f:
        f.write(f"# NF-BO Best Prompt\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Test accuracy: {test_accuracy:.2%}\n\n")
        f.write(best_prompt)
    logger.info(f"Best prompt saved to: {prompt_file}")

    if hasattr(llm_client, "cleanup"):
        llm_client.cleanup()

if __name__ == "__main__":
    main()
