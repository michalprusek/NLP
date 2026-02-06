"""RieLBO for GSM8K: Subspace BO on SONAR embeddings for prompt optimization.

Uses the same SphericalSubspaceBOv2 optimizer as molecular optimization,
but with SONAR text embeddings (1024D) instead of SELFIES VAE (256D).

Usage:
    # Pilot run (quick test)
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m rielbo_gsm8k.run \
        --preset geodesic --n-cold-start 10 --iterations 10

    # Full benchmark
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m rielbo_gsm8k.run \
        --preset geodesic --subspace-dim 16 \
        --n-cold-start 30 --iterations 70 --seed 42 \
        --split test --incremental-json rielbo_gsm8k/results/rielbo_s42.json
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="RieLBO for GSM8K prompt optimization (SONAR embeddings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Task config
    parser.add_argument("--n-cold-start", type=int, default=30,
                        help="Number of seed prompts for cold start")
    parser.add_argument("--iterations", type=int, default=70,
                        help="Number of BO iterations after cold start")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="GSM8K split for evaluation")

    # V2 config
    parser.add_argument("--preset", type=str, default="geodesic",
                        choices=["baseline", "order2", "whitening", "geodesic",
                                 "adaptive", "prob_norm", "smooth", "geometric", "full"],
                        help="V2 preset configuration")
    parser.add_argument("--subspace-dim", type=int, default=16,
                        help="Subspace dimension d")
    parser.add_argument("--acqf", type=str, default="ts",
                        choices=["ts", "ei", "ucb"],
                        help="Acquisition function")
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--trust-region", type=float, default=0.8)

    # Evaluation
    parser.add_argument("--eval-size", type=int, default=1319,
                        help="Fixed eval set size for GSM8K scoring (default: full test set)")
    parser.add_argument("--task-model", type=str, default="qwen",
                        help="Task model for GSM8K evaluation")
    parser.add_argument("--task-backend", type=str, default="vllm",
                        help="Backend for task model")

    # SONAR
    parser.add_argument("--sonar-device", type=str, default="cpu",
                        help="Device for SONAR codec (cpu recommended to save GPU for vLLM)")

    # Output
    parser.add_argument("--incremental-json", type=str, default=None,
                        help="Path for incremental JSON saving")
    parser.add_argument("--results-dir", type=str, default="rielbo_gsm8k/results",
                        help="Results directory")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RieLBO for GSM8K (SONAR Embeddings)")
    logger.info("=" * 60)
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Subspace dim: {args.subspace_dim}")
    logger.info(f"Cold start: {args.n_cold_start}, Iterations: {args.iterations}")
    logger.info(f"Total prompts: {args.n_cold_start + args.iterations}")
    logger.info(f"Eval size: {args.eval_size}, Split: {args.split}")
    logger.info(f"SONAR device: {args.sonar_device}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)

    # 1. Initialize SONAR codec on GPU 1
    logger.info("Loading SONAR codec...")
    from rielbo_gsm8k.sonar_codec import SonarCodec
    codec = SonarCodec(device=args.sonar_device)
    input_dim = codec.EMBEDDING_DIM  # 1024

    # 2. Initialize vLLM task model on GPU 0
    logger.info(f"Loading task model ({args.task_model})...")
    from shared.llm_client import create_llm_client
    llm_client = create_llm_client(
        args.task_model,
        backend=args.task_backend,
    )

    # 3. Initialize GSM8K oracle
    logger.info("Loading GSM8K evaluator...")
    from shared.gsm8k_evaluator import GSM8KEvaluator
    evaluator = GSM8KEvaluator(
        dataset_path="datasets/gsm8k",
        split=args.split,
    )
    logger.info(f"Dataset: {len(evaluator)} {args.split} examples")

    from rielbo_gsm8k.gsm8k_oracle import GSM8KOracle
    oracle = GSM8KOracle(
        llm_client=llm_client,
        evaluator=evaluator,
        eval_set_size=args.eval_size,
        seed=args.seed,
    )

    # 4. Initialize incremental saver
    incremental_saver = None
    if args.incremental_json:
        from shared.incremental_saver import IncrementalPromptSaver
        config = {
            "preset": args.preset,
            "subspace_dim": args.subspace_dim,
            "n_cold_start": args.n_cold_start,
            "iterations": args.iterations,
            "eval_size": args.eval_size,
            "split": args.split,
            "acqf": args.acqf,
            "seed": args.seed,
        }
        incremental_saver = IncrementalPromptSaver(
            output_path=args.incremental_json,
            method="rielbo_gsm8k",
            model=args.task_model,
            config=config,
        )

    # 5. Load seed prompts
    logger.info("Loading seed prompts...")
    from rielbo_gsm8k.seed_prompts import get_seed_prompts
    all_seeds = get_seed_prompts()

    # Use first n_cold_start prompts
    seed_prompts = all_seeds[:args.n_cold_start]
    logger.info(f"Using {len(seed_prompts)} seed prompts")

    # 6. Score seed prompts
    logger.info("Scoring seed prompts...")
    seed_scores = []
    for i, prompt in enumerate(seed_prompts):
        score = oracle.score(prompt)
        seed_scores.append(score)
        logger.info(f"  Seed {i+1}/{len(seed_prompts)}: {score:.4f} | {prompt[:80]}")

        # Save incrementally
        if incremental_saver is not None:
            incremental_saver.save_prompt(
                prompt=prompt,
                score=score,
                iteration=0,
            )

    seed_scores_tensor = torch.tensor(seed_scores, dtype=torch.float32)
    logger.info(f"Seed scoring done. Best: {max(seed_scores):.4f}")

    # 7. Create SphericalSubspaceBOv2
    logger.info("Initializing SphericalSubspaceBOv2...")
    from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config

    config = V2Config.from_preset(args.preset)

    # Use CPU for GP — 16D subspace is lightweight, keeps GPU free for vLLM
    gp_device = "cpu"
    optimizer = SphericalSubspaceBOv2(
        codec=codec,
        oracle=oracle,
        input_dim=input_dim,
        subspace_dim=args.subspace_dim,
        config=config,
        device=gp_device,
        n_candidates=args.n_candidates,
        acqf=args.acqf,
        trust_region=args.trust_region,
        seed=args.seed,
    )

    # 8. Cold start
    optimizer.cold_start(seed_prompts, seed_scores_tensor)

    # 9. Optimize
    logger.info(f"Starting BO: {args.iterations} iterations")

    from tqdm import tqdm
    pbar = tqdm(range(args.iterations), desc="RieLBO-GSM8K")
    n_dup = 0
    n_decode_fail = 0

    for i in pbar:
        result = optimizer.step()

        # Track history
        optimizer.history["iteration"].append(i)
        optimizer.history["best_score"].append(optimizer.best_score)
        optimizer.history["current_score"].append(result["score"])
        optimizer.history["n_evaluated"].append(len(optimizer.smiles_observed))
        optimizer.history["gp_mean"].append(result.get("gp_mean", 0))
        optimizer.history["gp_std"].append(result.get("gp_std", 0))
        optimizer.history["nearest_train_cos"].append(result.get("nearest_train_cos", 0))
        optimizer.history["embedding_norm"].append(result.get("embedding_norm", optimizer.mean_norm))
        optimizer.history["subspace_dim"].append(result.get("subspace_dim", optimizer._current_dim))
        optimizer.history["tr_length"].append(
            optimizer.tr_length if optimizer.tr_length is not None else optimizer.trust_region
        )
        optimizer.history["n_restarts"].append(optimizer.n_restarts)

        if result.get("is_decode_failure"):
            n_decode_fail += 1
        if result["is_duplicate"]:
            n_dup += 1

        # Save incrementally
        if incremental_saver is not None and not result["is_duplicate"]:
            prompt = result.get("smiles", "")
            if prompt:
                incremental_saver.save_prompt(
                    prompt=prompt,
                    score=result["score"],
                    iteration=i + 1,
                )

        postfix = {
            "best": f"{optimizer.best_score:.4f}",
            "curr": f"{result['score']:.4f}",
            "dup": n_dup,
            "fail": n_decode_fail,
        }
        if config.adaptive_tr:
            postfix["tr"] = f"{optimizer.tr_length:.3f}"
            postfix["rst"] = optimizer.n_restarts
        pbar.set_postfix(postfix)

    # 10. Report results
    logger.info("\n" + "=" * 60)
    logger.info("RIELBO-GSM8K RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best score: {optimizer.best_score:.4f}")
    logger.info(f"Best prompt:\n{optimizer.best_smiles}")
    logger.info(f"Prompts evaluated: {len(optimizer.smiles_observed)}")
    logger.info(f"Duplicates: {n_dup}, Decode failures: {n_decode_fail}")
    logger.info(f"GP restarts: {optimizer.n_restarts}")
    logger.info("=" * 60)

    # 11. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        results_dir,
        f"rielbo_s{args.seed}_{timestamp}.json",
    )

    results = {
        "method": "rielbo_gsm8k",
        "task": "gsm8k",
        "best_score": optimizer.best_score,
        "best_prompt": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "n_duplicates": n_dup,
        "n_decode_failures": n_decode_fail,
        "n_restarts": optimizer.n_restarts,
        "history": optimizer.history,
        "args": vars(args),
        "config": {
            "preset": args.preset,
            "input_dim": input_dim,
            "subspace_dim": args.subspace_dim,
            "kernel_order": config.kernel_order,
            "geodesic_tr": config.geodesic_tr,
            "adaptive_tr": config.adaptive_tr,
        },
        "mean_norm": optimizer.mean_norm,
        "seed_scores": seed_scores,
        "timestamp": timestamp,
    }

    # Also save as incremental format for convergence plot
    if incremental_saver is not None:
        incremental_saver.finalize(
            optimizer.best_smiles, optimizer.best_score
        )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # 12. Evaluate best prompt on full test set (skip if eval_size already covers it)
    if args.split == "test" and args.eval_size < len(evaluator):
        logger.info("\nEvaluating best prompt on FULL test set...")
        full_evaluator = GSM8KEvaluator(
            dataset_path="datasets/gsm8k",
            split="test",
        )
        all_indices = list(range(len(full_evaluator)))
        questions = [full_evaluator.dataset[i]["question"] for i in all_indices]

        formatted = [
            f"Q: {q}\n{optimizer.best_smiles}\nA:" for q in questions
        ]

        from tqdm import tqdm
        all_outputs = []
        batch_size = 100
        for j in tqdm(range(0, len(formatted), batch_size), desc="Full eval"):
            batch = formatted[j:j + batch_size]
            outputs = llm_client.generate_batch(
                batch, temperature=0.0, max_new_tokens=512
            )
            all_outputs.extend(outputs)

        full_results = full_evaluator.evaluate_batch(all_outputs, all_indices)
        test_accuracy = full_results["accuracy"]

        logger.info(f"Full test accuracy: {test_accuracy:.4f} ({full_results['correct']}/{full_results['total']})")

        results["test_accuracy"] = test_accuracy
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    return optimizer.best_score


if __name__ == "__main__":
    main()
