"""Benchmark runner for comparing BO methods on GuacaMol.

Usage:
    # Single benchmark run
    uv run python -m rielbo.benchmark.runner \
        --methods subspace --tasks adip --seeds 42 --iterations 500

    # Multiple methods and tasks
    uv run python -m rielbo.benchmark.runner \
        --methods subspace,turbo,lolbo --tasks adip,pdop --seeds 42-46

    # Full benchmark (all tasks, all methods, 10 seeds)
    uv run python -m rielbo.benchmark.runner \
        --methods subspace,turbo,lolbo --tasks all --seeds 42-51
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from shared.guacamol.constants import GUACAMOL_TASKS
from rielbo.benchmark.methods import METHODS, get_method

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seed specification.

    Examples:
        "42" -> [42]
        "42,43,44" -> [42, 43, 44]
        "42-51" -> [42, 43, ..., 51]
    """
    if "-" in seeds_str and "," not in seeds_str:
        start, end = map(int, seeds_str.split("-"))
        return list(range(start, end + 1))
    elif "," in seeds_str:
        return [int(s.strip()) for s in seeds_str.split(",")]
    else:
        return [int(seeds_str)]


def parse_tasks(tasks_str: str) -> list[str]:
    """Parse task specification.

    Examples:
        "adip" -> ["adip"]
        "adip,pdop" -> ["adip", "pdop"]
        "all" -> all 12 GuacaMol tasks
    """
    if tasks_str.lower() == "all":
        return GUACAMOL_TASKS.copy()
    return [t.strip() for t in tasks_str.split(",")]


def parse_methods(methods_str: str) -> list[str]:
    """Parse method specification."""
    if methods_str.lower() == "all":
        return list(METHODS.keys())
    return [m.strip() for m in methods_str.split(",")]


def run_single_benchmark(
    method_name: str,
    task_id: str,
    seed: int,
    n_cold_start: int = 100,
    n_iterations: int = 500,
    output_dir: str = "rielbo/results/benchmark",
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """Run a single benchmark configuration.

    Args:
        method_name: Name of BO method ("subspace", "turbo", "lolbo")
        task_id: GuacaMol task ID
        seed: Random seed
        n_cold_start: Number of cold start molecules
        n_iterations: Number of optimization iterations
        output_dir: Directory to save results
        device: Device for computation
        verbose: Whether to print progress

    Returns:
        Results dictionary
    """
    logger.info(f"Running {method_name} on {task_id} with seed {seed}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load shared components
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=device)
    oracle = GuacaMolOracle(task_id=task_id)

    # Load cold start data
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=n_cold_start,
        task_id=task_id,
    )
    # Convert scores to tensor (may already be a numpy array or tensor)
    if isinstance(scores, torch.Tensor):
        scores_tensor = scores.detach().clone().float()
    else:
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

    # Create method
    method_class = get_method(method_name)
    method = method_class(
        codec=codec,
        oracle=oracle,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    # Run optimization
    start_time = time.time()
    method.cold_start(smiles_list, scores_tensor)
    method.optimize(n_iterations=n_iterations, log_interval=50)
    elapsed_time = time.time() - start_time

    # Collect results
    history = method.get_history()
    config = method.get_config()

    results = {
        "method": method_name,
        "task_id": task_id,
        "seed": seed,
        "config": {
            "n_cold_start": n_cold_start,
            "n_iterations": n_iterations,
            **config,
        },
        "best_score": method.best_score,
        "best_smiles": method.best_smiles,
        "n_evaluated": method.n_evaluated,
        "wall_time_seconds": elapsed_time,
        "history": history.to_dict(),
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{method_name}_{task_id}_seed{seed}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Completed {method_name}/{task_id}/seed{seed}: "
        f"best={method.best_score:.4f}, time={elapsed_time:.1f}s"
    )
    logger.info(f"Results saved to {filepath}")

    return results


def run_benchmark_suite(
    methods: list[str],
    tasks: list[str],
    seeds: list[int],
    n_cold_start: int = 100,
    n_iterations: int = 500,
    output_dir: str = "rielbo/results/benchmark",
    device: str = "cuda",
    verbose: bool = False,
    skip_existing: bool = True,
) -> list[dict]:
    """Run full benchmark suite.

    Args:
        methods: List of method names
        tasks: List of task IDs
        seeds: List of random seeds
        n_cold_start: Number of cold start molecules
        n_iterations: Number of optimization iterations
        output_dir: Directory to save results
        device: Device for computation
        verbose: Whether to print progress
        skip_existing: Skip runs that already have results

    Returns:
        List of all results
    """
    total_runs = len(methods) * len(tasks) * len(seeds)
    logger.info(f"Starting benchmark suite: {total_runs} runs")
    logger.info(f"Methods: {methods}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Seeds: {seeds}")

    all_results = []
    failures = []
    completed = 0

    for method_name in methods:
        for task_id in tasks:
            for seed in seeds:
                # Check if already exists
                filename = f"{method_name}_{task_id}_seed{seed}.json"
                filepath = os.path.join(output_dir, filename)

                if skip_existing and os.path.exists(filepath):
                    logger.info(f"Skipping {filename} (already exists)")
                    completed += 1
                    continue

                try:
                    result = run_single_benchmark(
                        method_name=method_name,
                        task_id=task_id,
                        seed=seed,
                        n_cold_start=n_cold_start,
                        n_iterations=n_iterations,
                        output_dir=output_dir,
                        device=device,
                        verbose=verbose,
                    )
                    all_results.append(result)
                except KeyboardInterrupt:
                    logger.info("Benchmark interrupted by user")
                    raise
                except Exception as e:
                    logger.error(f"Failed {method_name}/{task_id}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    failures.append({"method": method_name, "task": task_id, "seed": seed, "error": str(e)})

                completed += 1
                logger.info(f"Progress: {completed}/{total_runs}")

    logger.info(f"Benchmark suite complete: {len(all_results)} successful, {len(failures)} failed")
    if failures:
        for f in failures:
            logger.error(f"  FAILED: {f['method']}/{f['task']}/seed{f['seed']}: {f['error']}")
    return all_results


def generate_summary_table(
    results_dir: str,
    methods: list[str] | None = None,
    tasks: list[str] | None = None,
) -> str:
    """Generate summary table from benchmark results.

    Args:
        results_dir: Directory containing result JSON files
        methods: Methods to include (None = all)
        tasks: Tasks to include (None = all)

    Returns:
        Markdown table string
    """
    from collections import defaultdict

    # Load all results
    results_by_method_task = defaultdict(list)
    for filepath in Path(results_dir).glob("*.json"):
        try:
            with open(filepath) as f:
                result = json.load(f)
            method = result["method"]
            task = result["task_id"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping malformed result file {filepath}: {e}")
            continue

        if methods and method not in methods:
            continue
        if tasks and task not in tasks:
            continue

        results_by_method_task[(method, task)].append(result["best_score"])

    if not results_by_method_task:
        return "No results found"

    # Get unique methods and tasks
    all_methods = sorted(set(m for m, _ in results_by_method_task.keys()))
    all_tasks = sorted(set(t for _, t in results_by_method_task.keys()))

    # Build table
    header = "| Task |" + " | ".join(all_methods) + " | Best |"
    separator = "|------|" + " | ".join(["------"] * len(all_methods)) + " | ---- |"
    rows = [header, separator]

    for task in all_tasks:
        row = [f"| {task} "]
        best_mean = -float("inf")
        best_method = ""

        for method in all_methods:
            scores = results_by_method_task.get((method, task), [])
            if scores:
                mean = np.mean(scores)
                std = np.std(scores)
                row.append(f"| {mean:.4f} ± {std:.4f} ")
                if mean > best_mean:
                    best_mean = mean
                    best_method = method
            else:
                row.append("| - ")

        row.append(f"| **{best_method}** |")
        rows.append("".join(row))

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark comparison of BO methods on GuacaMol"
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="subspace,turbo,lolbo",
        help="Methods to benchmark (comma-separated or 'all')",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="adip",
        help="Tasks to benchmark (comma-separated or 'all')",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Random seeds (single, comma-separated, or range like '42-51')",
    )
    parser.add_argument(
        "--n-cold-start",
        type=int,
        default=100,
        help="Number of cold start molecules",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rielbo/results/benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run benchmarks even if results exist",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary table from existing results",
    )

    args = parser.parse_args()

    if args.summary:
        # Just print summary of existing results
        methods = parse_methods(args.methods) if args.methods != "all" else None
        tasks = parse_tasks(args.tasks) if args.tasks != "all" else None
        table = generate_summary_table(args.output_dir, methods, tasks)
        print(table)
        return

    # Parse arguments
    methods = parse_methods(args.methods)
    tasks = parse_tasks(args.tasks)
    seeds = parse_seeds(args.seeds)

    # Validate
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    for task in tasks:
        if task not in GUACAMOL_TASKS:
            raise ValueError(f"Unknown task: {task}. Available: {GUACAMOL_TASKS}")

    # Run benchmark
    run_benchmark_suite(
        methods=methods,
        tasks=tasks,
        seeds=seeds,
        n_cold_start=args.n_cold_start,
        n_iterations=args.iterations,
        output_dir=args.output_dir,
        device=args.device,
        verbose=args.verbose,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
