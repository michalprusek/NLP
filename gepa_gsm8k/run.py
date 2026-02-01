#!/usr/bin/env python3
"""
GEPA: Genetic-Pareto Prompt Optimization for GSM8K

Uses the official GEPA package for reflective prompt evolution.
Wraps GSM8K dataset and vLLM client for local optimization.

Usage:
    uv run python -m gepa.run --max-calls 50000
    uv run python -m gepa.run --model qwen --reflection-model qwen
"""

import argparse
import logging
import time
import json
import re
from pathlib import Path
from typing import Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class GSM8KDataInst:
    """GSM8K data instance for GEPA."""
    input: str
    additional_context: dict[str, str]
    answer: str

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return ["input", "additional_context", "answer"]


class GSM8KEvaluator:
    """Custom evaluator for GSM8K that extracts and compares numeric answers."""

    # Patterns for answer extraction
    STRICT_PATTERN = r'####\s*(-?[\d,]+\.?\d*)'
    BOXED_PATTERN = r'\\boxed\{(-?[\d,]+\.?\d*)\}'
    FLEXIBLE_PATTERN = r'(-?\$?[\d,]+\.?\d*)'

    def __init__(self, failure_score: float = 0.0):
        self.failure_score = failure_score
        # Import EvaluationResult at init to avoid import issues
        import importlib
        gepa_pkg = importlib.import_module("gepa")
        adapter_module = importlib.import_module("gepa.adapters.default_adapter.default_adapter")
        self.EvaluationResult = adapter_module.EvaluationResult

    def _normalize(self, answer: str) -> str:
        if answer is None:
            return None
        answer = answer.replace(',', '').replace('$', '')
        answer = re.sub(r'\.(?!\d)', '', answer)
        return answer.strip()

    def _extract_answer(self, text: str) -> str | None:
        if not text:
            return None

        # Try #### format
        match = re.search(self.STRICT_PATTERN, text)
        if match:
            return self._normalize(match.group(1))

        # Try \boxed{} format
        match = re.search(self.BOXED_PATTERN, text)
        if match:
            return self._normalize(match.group(1))

        # Fallback: last number
        numbers = re.findall(self.FLEXIBLE_PATTERN, text)
        numbers = [n for n in numbers if n and re.search(r'\d', n)]
        if numbers:
            return self._normalize(numbers[-1])

        return None

    def _compare(self, pred: str, gt: str) -> bool:
        if pred is None or gt is None:
            return False
        if pred == gt:
            return True
        try:
            return abs(float(pred) - float(gt)) < 1e-5
        except (ValueError, TypeError):
            return False

    def __call__(self, data: GSM8KDataInst, response: str):
        """Evaluate response against ground truth."""

        gt = data.answer
        pred = self._extract_answer(response)
        is_correct = self._compare(pred, gt)

        score = 1.0 if is_correct else self.failure_score

        if is_correct:
            feedback = f"Correct! The answer {pred} matches the expected answer {gt}."
        else:
            feedback = (
                f"Incorrect. Expected answer: {gt}, but got: {pred}. "
                "Make sure to show step-by-step work and clearly state the final numeric answer."
            )

        return self.EvaluationResult(score=score, feedback=feedback, objective_scores=None)


class VLLMChatWrapper:
    """Wrapper to make vLLM client compatible with GEPA's ChatCompletionCallable."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.call_count = 0

    def __call__(self, messages: list[dict]) -> str:
        self.call_count += 1

        # Combine system and user messages into a single prompt
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(msg["content"])
            elif msg["role"] == "user":
                prompt_parts.append(msg["content"])

        prompt = "\n\n".join(prompt_parts)

        # Log every 10th call for debugging
        if self.call_count <= 10 or self.call_count % 10 == 0:
            logger.info(f"LLM call #{self.call_count}, prompt length: {len(prompt)}")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

        try:
            response = self.llm_client.generate(prompt, max_new_tokens=512, temperature=0.0)
        except Exception as e:
            logger.error(f"LLM call #{self.call_count} failed: {e}")
            raise

        if self.call_count <= 5:
            logger.info(f"Response #{self.call_count}:\n{response}")

        return response


def load_gsm8k_dataset(dataset_path: str = "datasets/gsm8k", split: str = "train") -> list[GSM8KDataInst]:
    """Load GSM8K dataset as GEPA-compatible data instances."""
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    data = ds[split]

    instances = []
    for item in data:
        # Extract ground truth answer
        answer_text = item["answer"]
        # Get number after ####
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
        if match:
            gt = match.group(1).replace(',', '')
        else:
            # Fallback
            numbers = re.findall(r'(-?\d+\.?\d*)', answer_text)
            gt = numbers[-1] if numbers else "0"

        inst = GSM8KDataInst(
            input=item["question"],
            additional_context={},
            answer=gt
        )
        instances.append(inst)

    return instances


def main():
    parser = argparse.ArgumentParser(
        description="GEPA: Genetic-Pareto Prompt Optimization for GSM8K"
    )

    # Model configuration
    parser.add_argument(
        "--model", type=str, default="qwen",
        help="Task model (qwen, llama)"
    )
    parser.add_argument(
        "--reflection-model", type=str, default=None,
        help="Reflection model (default: same as task model)"
    )
    parser.add_argument(
        "--backend", type=str, default="vllm",
        choices=["vllm", "openai", "litellm"],
        help="LLM backend"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallel size for vLLM"
    )

    # Optimization parameters
    parser.add_argument(
        "--max-calls", type=int, default=50000,
        help="Maximum LLM calls budget"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=150,
        help="Evaluation minibatch size"
    )

    # GEPA parameters
    parser.add_argument(
        "--frontier-type", type=str, default="instance",
        choices=["instance", "objective", "hybrid"],
        help="Pareto frontier type"
    )
    parser.add_argument(
        "--use-merge", action="store_true",
        help="Enable prompt merging"
    )

    # Other options
    parser.add_argument(
        "--results-dir", type=str, default="gepa/results",
        help="Results directory"
    )
    parser.add_argument(
        "--skip-test-eval", action="store_true",
        help="Skip final test set evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    if args.reflection_model is None:
        args.reflection_model = args.model

    # Calculate max_metric_calls from max_calls
    # GEPA counts metric_calls as datapoints, not LLM calls
    # With minibatch_size examples per evaluation, we need:
    max_metric_calls = args.max_calls // args.minibatch_size

    logger.info("=" * 60)
    logger.info("GEPA: Genetic-Pareto Prompt Optimization")
    logger.info("=" * 60)
    logger.info(f"Task model: {args.model}")
    logger.info(f"Reflection model: {args.reflection_model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Budget: {args.max_calls} LLM calls")
    logger.info(f"Max metric calls: {max_metric_calls}")
    logger.info(f"Minibatch size: {args.minibatch_size}")
    logger.info(f"Frontier type: {args.frontier_type}")
    logger.info("=" * 60)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading GSM8K dataset...")
    trainset = load_gsm8k_dataset("datasets/gsm8k", "train")
    logger.info(f"Loaded {len(trainset)} training examples")

    # Initialize LLM client
    if args.backend == "vllm":
        from shared.llm_client import create_llm_client

        logger.info(f"Initializing {args.model} vLLM client...")
        llm_client = create_llm_client(
            args.model,
            backend="vllm",
            tensor_parallel_size=args.tensor_parallel_size
        )
        task_lm = VLLMChatWrapper(llm_client)

        # For reflection, use same client
        if args.reflection_model == args.model:
            reflection_lm = task_lm
        else:
            logger.info(f"Initializing {args.reflection_model} for reflection...")
            reflection_client = create_llm_client(
                args.reflection_model,
                backend="vllm",
                tensor_parallel_size=args.tensor_parallel_size
            )
            reflection_lm = VLLMChatWrapper(reflection_client)
    else:
        # Use LiteLLM string format
        model_map = {
            "qwen": "huggingface/Qwen/Qwen2.5-7B-Instruct",
            "llama": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
        }
        task_lm = model_map.get(args.model, args.model)
        reflection_lm = model_map.get(args.reflection_model, args.reflection_model)

    # Seed prompt
    seed_candidate = {
        "system_prompt": "Solve this math problem step by step. Show your work and clearly state the final numeric answer."
    }

    # Create evaluator
    evaluator = GSM8KEvaluator()

    # Import GEPA package (not local module)
    import importlib
    gepa_pkg = importlib.import_module("gepa")

    logger.info("Starting GEPA optimization...")
    start_time = time.time()

    # Run GEPA
    result = gepa_pkg.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=None,
        task_lm=task_lm,
        evaluator=evaluator,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        frontier_type=args.frontier_type,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=min(20, args.minibatch_size),
        use_merge=args.use_merge,
        display_progress_bar=True,
        seed=args.seed,
        run_dir=str(results_dir / "gepa_run"),
    )

    elapsed = time.time() - start_time

    # Extract best prompt
    best_candidate = result.best_candidate
    best_prompt = best_candidate.get("system_prompt", str(best_candidate))
    best_score = result.best_score

    logger.info("\n" + "=" * 60)
    logger.info("GEPA OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best validation score: {best_score:.4f}")
    logger.info(f"Best prompt:\n{best_prompt}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 60)

    # Save results
    results = {
        "method": "gepa",
        "best_prompt": best_prompt,
        "best_score": best_score,
        "total_time": elapsed,
        "config": {
            "model": args.model,
            "reflection_model": args.reflection_model,
            "max_calls": args.max_calls,
            "minibatch_size": args.minibatch_size,
            "frontier_type": args.frontier_type,
        }
    }

    results_path = results_dir / f"gepa_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Evaluate on test set
    if not args.skip_test_eval:
        logger.info("\nEvaluating best prompt on test set...")

        from datasets import load_from_disk
        ds = load_from_disk("datasets/gsm8k")
        test_data = ds["test"]

        # Create prompts
        prompts = []
        for item in test_data:
            prompt = f"{best_prompt}\n\nQ: {item['question']}\nA:"
            prompts.append(prompt)

        # Generate responses
        all_responses = []
        batch_size = 100
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            if args.backend == "vllm":
                responses = llm_client.generate_batch(batch)
            else:
                responses = [task_lm([{"role": "user", "content": p}]) for p in batch]
            all_responses.extend(responses)
            logger.info(f"Evaluated {len(all_responses)}/{len(prompts)} test examples")

        # Calculate accuracy
        correct = 0
        for item, response in zip(test_data, all_responses):
            gt_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', item["answer"])
            gt = gt_match.group(1).replace(',', '') if gt_match else "0"

            pred = evaluator._extract_answer(response)
            if evaluator._compare(pred, gt):
                correct += 1

        test_accuracy = correct / len(test_data)
        logger.info(f"Test accuracy: {test_accuracy:.4f} ({correct}/{len(test_data)})")

        results["test_accuracy"] = test_accuracy

        # Save final results
        final_path = results_dir / f"gepa_final_{int(time.time())}.json"
        with open(final_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Final results saved to {final_path}")

    return results


if __name__ == "__main__":
    main()
