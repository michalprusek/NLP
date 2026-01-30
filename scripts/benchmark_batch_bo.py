#!/usr/bin/env python3
"""
Benchmark batch BO vs sequential BO.

Compares:
1. Wall-clock time per iteration
2. Sample efficiency (best score vs total evaluations)
3. Batch diversity metrics
4. Throughput (evaluations per hour)

Usage:
    uv run python scripts/benchmark_batch_bo.py \
        --flow-checkpoint results/flow_checkpoints/best_flow.pt \
        --n-iterations 10 \
        --batch-sizes 1,4,8 \
        --output-dir results/batch_bo_benchmark

    # Dry run (no LLM evaluation, uses mock scores)
    uv run python scripts/benchmark_batch_bo.py \
        --dry-run \
        --n-iterations 5 \
        --batch-sizes 1,4
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark batch BO vs sequential BO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model paths
    parser.add_argument(
        "--flow-checkpoint",
        type=str,
        default="results/flow_checkpoints/best_flow.pt",
        help="Path to trained flow model checkpoint",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default="datasets/evaluated_instructions/instruction_embeddings_with_scores.pt",
        help="Path to pre-evaluated embeddings for warm start",
    )

    # Benchmark parameters
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=10,
        help="Number of BO iterations per batch size (default: 10)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8",
        help="Comma-separated batch sizes to benchmark (default: 1,4,8)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=64,
        help="Number of candidates per iteration (default: 64)",
    )

    # Filtering options
    parser.add_argument(
        "--use-density-filter",
        action="store_true",
        help="Enable flow density filtering",
    )
    parser.add_argument(
        "--density-percentile",
        type=float,
        default=25.0,
        help="Density filter percentile (default: 25.0)",
    )
    parser.add_argument(
        "--disable-l2r-filter",
        action="store_true",
        help="Disable L2-r filtering",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/batch_bo_benchmark",
        help="Directory for benchmark results",
    )

    # Testing
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without actual LLM evaluation (uses mock scores)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (default: cuda)",
    )

    return parser.parse_args()


class MockLLMClient:
    """Mock LLM client for dry-run benchmarking."""

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> List[str]:
        """Return mock answers with random 'correctness'."""
        # Simulate some processing time
        time.sleep(0.01 * len(prompts))
        return ["The answer is 42." for _ in prompts]


class MockEvaluator:
    """Mock evaluator for dry-run benchmarking."""

    def __init__(self, n_questions: int = 100):
        self._n_questions = n_questions
        self.dataset = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(n_questions)]

    def __len__(self):
        return self._n_questions

    def evaluate_batch(self, outputs: List[str], indices: List[int]) -> Dict:
        """Return mock accuracy based on random scores."""
        import random
        # Return accuracy between 0.3 and 0.9
        accuracy = 0.3 + 0.6 * random.random()
        return {"accuracy": accuracy}


def run_benchmark(
    batch_sizes: List[int],
    n_iterations: int,
    n_candidates: int,
    flow_checkpoint: str,
    warm_start_path: str,
    use_density_filter: bool,
    density_percentile: float,
    l2r_filter_enabled: bool,
    device: str,
    dry_run: bool,
) -> Dict:
    """
    Run benchmark for multiple batch sizes.

    Returns:
        Dict with benchmark results for each batch size
    """
    # Lazy imports for fast --help
    from src.ecoflow.flow_model import FlowMatchingModel
    from src.ecoflow.gp_surrogate import SonarGPSurrogate
    from src.ecoflow.guided_flow import GuidedFlowSampler
    from src.ecoflow.optimization_loop import BOOptimizationLoop
    from src.ecoflow.validate import load_model_from_checkpoint

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking batch_size={batch_size}")
        logger.info(f"{'='*60}")

        # Load flow model (fresh for each batch size)
        logger.info(f"Loading flow model from {flow_checkpoint}...")
        flow_model = load_model_from_checkpoint(flow_checkpoint, device=device)

        # Create GP surrogate
        gp = SonarGPSurrogate(D=1024, device=device)

        # Create guided sampler
        sampler = GuidedFlowSampler(
            flow_model=flow_model,
            gp=gp,
            guidance_strength=1.0,
            alpha=1.0,
        )

        # Setup LLM and evaluator
        if dry_run:
            logger.info("DRY RUN mode: using mock LLM and evaluator")
            llm_client = MockLLMClient()
            evaluator = MockEvaluator(n_questions=100)
            decoder = None  # Will be handled separately
        else:
            # Import real components
            from src.ecoflow.decoder import SonarDecoder
            from src.gsm8k_evaluator import GSM8KEvaluator
            from src.llm_client import create_llm_client

            logger.info("Loading SONAR decoder...")
            decoder = SonarDecoder(device=device)

            logger.info("Loading GSM8K evaluator...")
            evaluator = GSM8KEvaluator()

            logger.info("Creating LLM client...")
            llm_client = create_llm_client("qwen", "vllm")

        # For dry run, create mock decoder
        if dry_run:
            class MockDecoder:
                def decode(self, embeddings):
                    return [f"Mock prompt {i}" for i in range(len(embeddings))]
            decoder = MockDecoder()

        # Create optimization loop
        loop = BOOptimizationLoop(
            flow_model=flow_model,
            gp=gp,
            sampler=sampler,
            decoder=decoder,
            evaluator=evaluator,
            llm_client=llm_client,
            batch_size=batch_size,
            device=device,
            l2r_filter_enabled=l2r_filter_enabled,
        )

        # Warm start from pre-evaluated embeddings
        if Path(warm_start_path).exists() and not dry_run:
            logger.info(f"Warm starting from {warm_start_path}...")
            loop.warm_start(warm_start_path, top_k=50)
        else:
            # Initialize with mock data for dry run
            logger.info("Initializing with random data (dry run)...")
            # Create some dummy training data
            loop.train_X = torch.randn(20, 1024, device=device)
            loop.train_Y = torch.rand(20, device=device) * 0.5 + 0.3
            loop.best_score = loop.train_Y.max().item()
            loop.best_prompt = "Mock best prompt"
            loop.best_so_far_list = [loop.best_score]
            loop.gp.fit(loop.train_X, loop.train_Y)
            loop.sampler.update_gp(loop.gp)

        # Track metrics
        iteration_times = []
        scores_per_iteration = []
        best_so_far_curve = []
        diversity_metrics = []
        total_evaluations = 0

        # Run iterations
        start_time = time.time()

        for i in range(n_iterations):
            iter_start = time.time()

            if batch_size == 1:
                # Sequential step
                result = loop.step(n_candidates=n_candidates)
                scores = [result["score"]]
                diversity = 0.0  # N/A for single candidate
            else:
                # Batch step
                result = loop.batch_step(
                    batch_size=batch_size,
                    n_candidates=n_candidates,
                    use_local_penalization=True,
                    use_density_filter=use_density_filter,
                    density_percentile=density_percentile,
                )
                scores = result["scores"]
                diversity = result["stats"].get("batch_diversity", 0.0)

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            scores_per_iteration.append(scores)
            best_so_far_curve.append(result["best_so_far"])
            diversity_metrics.append(diversity)
            total_evaluations += len(scores)

            logger.info(
                f"Iteration {i+1}/{n_iterations}: "
                f"time={iter_time:.2f}s, "
                f"scores={[f'{s:.3f}' for s in scores]}, "
                f"best={result['best_so_far']:.3f}, "
                f"diversity={diversity:.4f}"
            )

        total_time = time.time() - start_time

        # Compute summary statistics
        avg_time_per_iter = sum(iteration_times) / len(iteration_times)
        avg_time_per_eval = total_time / total_evaluations
        throughput = total_evaluations / (total_time / 3600)  # evals per hour

        results[batch_size] = {
            "batch_size": batch_size,
            "n_iterations": n_iterations,
            "total_evaluations": total_evaluations,
            "total_time_seconds": total_time,
            "avg_time_per_iteration": avg_time_per_iter,
            "avg_time_per_evaluation": avg_time_per_eval,
            "throughput_evals_per_hour": throughput,
            "final_best_score": best_so_far_curve[-1],
            "best_so_far_curve": best_so_far_curve,
            "iteration_times": iteration_times,
            "scores_per_iteration": scores_per_iteration,
            "diversity_metrics": diversity_metrics,
            "avg_batch_diversity": sum(diversity_metrics) / len(diversity_metrics) if diversity_metrics else 0,
        }

        logger.info(f"\nBatch size {batch_size} summary:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Avg time/iteration: {avg_time_per_iter:.2f}s")
        logger.info(f"  Avg time/evaluation: {avg_time_per_eval:.2f}s")
        logger.info(f"  Throughput: {throughput:.1f} evals/hour")
        logger.info(f"  Final best: {best_so_far_curve[-1]:.4f}")

        # Clean up
        del loop, gp, sampler, flow_model
        torch.cuda.empty_cache()

    return results


def save_results(results: Dict, output_dir: str, dry_run: bool):
    """Save benchmark results to JSON and CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "benchmark_dry" if dry_run else "benchmark"

    # Save full JSON
    json_path = output_path / f"{prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")

    # Save CSV summary
    csv_path = output_path / f"{prefix}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("batch_size,n_iterations,total_evals,total_time_s,avg_iter_time,avg_eval_time,throughput_per_hour,final_best,avg_diversity\n")
        for bs, data in results.items():
            f.write(
                f"{bs},{data['n_iterations']},{data['total_evaluations']},"
                f"{data['total_time_seconds']:.2f},{data['avg_time_per_iteration']:.2f},"
                f"{data['avg_time_per_evaluation']:.2f},{data['throughput_evals_per_hour']:.1f},"
                f"{data['final_best_score']:.4f},{data['avg_batch_diversity']:.4f}\n"
            )
    logger.info(f"Saved CSV summary to {csv_path}")


def print_comparison(results: Dict):
    """Print comparison table of batch sizes."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(f"{'Batch':>6} {'Iters':>6} {'Evals':>6} {'Time(s)':>8} {'Iter(s)':>8} "
          f"{'Eval(s)':>8} {'E/hour':>8} {'Best':>6} {'Div':>6}")
    print("-" * 80)

    # Get baseline (batch_size=1) for speedup calculation
    baseline_time_per_eval = results.get(1, {}).get("avg_time_per_evaluation", 1.0)

    for bs in sorted(results.keys()):
        data = results[bs]
        speedup = baseline_time_per_eval / data["avg_time_per_evaluation"] if bs > 1 else 1.0

        print(
            f"{bs:>6} {data['n_iterations']:>6} {data['total_evaluations']:>6} "
            f"{data['total_time_seconds']:>8.1f} {data['avg_time_per_iteration']:>8.2f} "
            f"{data['avg_time_per_evaluation']:>8.2f} {data['throughput_evals_per_hour']:>8.0f} "
            f"{data['final_best_score']:>6.3f} {data['avg_batch_diversity']:>6.3f}"
        )
        if bs > 1:
            print(f"       (speedup vs bs=1: {speedup:.2f}x)")

    print("=" * 80)


def main():
    args = parse_args()

    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]

    logger.info("Batch BO Benchmark")
    logger.info(f"  Batch sizes: {batch_sizes}")
    logger.info(f"  Iterations per batch size: {args.n_iterations}")
    logger.info(f"  Candidates per iteration: {args.n_candidates}")
    logger.info(f"  Density filter: {args.use_density_filter}")
    logger.info(f"  L2-r filter: {not args.disable_l2r_filter}")
    logger.info(f"  Dry run: {args.dry_run}")

    # Run benchmark
    results = run_benchmark(
        batch_sizes=batch_sizes,
        n_iterations=args.n_iterations,
        n_candidates=args.n_candidates,
        flow_checkpoint=args.flow_checkpoint,
        warm_start_path=args.warm_start,
        use_density_filter=args.use_density_filter,
        density_percentile=args.density_percentile,
        l2r_filter_enabled=not args.disable_l2r_filter,
        device=args.device,
        dry_run=args.dry_run,
    )

    # Save results
    save_results(results, args.output_dir, args.dry_run)

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
