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
from typing import Any, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class MaxPromptsReachedException(Exception):
    """Raised when max_prompts limit is reached during GEPA optimization."""
    pass


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
        self.EvaluationResult = None  # Lazy import

    def _ensure_evaluation_result(self):
        """Lazily import EvaluationResult from gepa package."""
        if self.EvaluationResult is None:
            import importlib
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
        self._ensure_evaluation_result()

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


class TrackingEvaluatorWrapper:
    """
    Wrapper around GSM8KEvaluator that tracks unique prompts evaluated.

    This works by intercepting the evaluation calls and extracting the
    current prompt from the context. GEPA calls the evaluator per data point,
    so we accumulate results per prompt and save when a new prompt is seen.
    """

    def __init__(
        self,
        base_evaluator: GSM8KEvaluator,
        max_prompts: Optional[int] = None,
        incremental_saver: Optional[Any] = None,
    ):
        self.base_evaluator = base_evaluator
        self.max_prompts = max_prompts
        self.incremental_saver = incremental_saver

        # Copy attributes from base (EvaluationResult is lazily loaded)
        self.failure_score = base_evaluator.failure_score

        # Tracking state
        self.prompts_evaluated = 0
        self.seen_prompts = {}  # prompt -> (total_correct, total_count, iteration)
        self._current_iteration = 0

        # Per-prompt batch accumulator
        self._current_prompt = None
        self._current_correct = 0
        self._current_count = 0

    @property
    def EvaluationResult(self):
        """Delegate to base evaluator's EvaluationResult (lazily loaded)."""
        self.base_evaluator._ensure_evaluation_result()
        return self.base_evaluator.EvaluationResult

    def set_current_prompt(self, prompt: str):
        """Set the current prompt being evaluated. Called before batch evaluation."""
        # If switching to a new prompt, finalize the previous one
        if self._current_prompt is not None and self._current_prompt != prompt:
            self._finalize_prompt()
        # If re-evaluating the same prompt but we have accumulated results, finalize first
        # This handles the case where GEPA re-evaluates prompts in multiple batches
        elif self._current_prompt == prompt and self._current_count > 0:
            logger.info(f"[PromptTracker] Re-evaluation of same prompt detected, finalizing batch of {self._current_count} examples")
            self._finalize_prompt()

        self._current_prompt = prompt
        self._current_correct = 0
        self._current_count = 0

    def _finalize_prompt(self):
        """Finalize tracking for the current prompt."""
        if self._current_prompt is None or self._current_count == 0:
            return

        prompt = self._current_prompt
        score = self._current_correct / self._current_count

        # Only track new prompts
        if prompt not in self.seen_prompts:
            self.prompts_evaluated += 1
            self.seen_prompts[prompt] = (self._current_correct, self._current_count, self._current_iteration)

            # Save to incremental saver
            if self.incremental_saver is not None:
                self.incremental_saver.save_prompt(
                    prompt=prompt,
                    score=score,
                    iteration=self._current_iteration,
                )

            logger.info(f"[PromptTracker] Prompt #{self.prompts_evaluated}: score={score:.4f} ({self._current_correct}/{self._current_count})")

            # Check max_prompts limit
            if self.max_prompts is not None and self.prompts_evaluated >= self.max_prompts:
                logger.info(f"Max prompts reached ({self.prompts_evaluated}/{self.max_prompts}). Stopping.")
                # Finalize saver before raising exception
                if self.incremental_saver is not None:
                    best_prompt, best_score = self._get_best_prompt()
                    self.incremental_saver.finalize(best_prompt, best_score)
                raise MaxPromptsReachedException(
                    f"Reached max_prompts limit: {self.prompts_evaluated}/{self.max_prompts}"
                )
        else:
            # Update existing prompt stats (for re-evaluations)
            old_correct, old_count, iteration = self.seen_prompts[prompt]
            self.seen_prompts[prompt] = (old_correct + self._current_correct, old_count + self._current_count, iteration)

    def __call__(self, data: GSM8KDataInst, response: str):
        """Delegate to base evaluator and track results."""
        result = self.base_evaluator(data, response)

        # Track this evaluation
        self._current_count += 1
        if result.score > 0.5:  # Correct answer
            self._current_correct += 1

        # Auto-finalize after a reasonable batch size (GEPA uses varying batch sizes)
        # We use 100 as threshold - this captures partial batches but still groups evaluations
        # A full test set is 1319, minibatch default is 150
        if self._current_count >= 100 and self._current_count % 100 == 0:
            logger.info(f"[PromptTracker] Auto-checkpoint at {self._current_count} evaluations")

        return result

    def _get_best_prompt(self):
        """Get the best prompt seen so far."""
        if not self.seen_prompts:
            return "", 0.0
        best_prompt = max(self.seen_prompts.items(), key=lambda x: x[1][0] / x[1][1])
        prompt, (total_correct, total_count, _) = best_prompt
        return prompt, total_correct / total_count

    def finalize(self):
        """Finalize tracking - call after optimization completes."""
        # Finalize current prompt if any
        if self._current_prompt is not None:
            self._finalize_prompt()

        # Finalize incremental saver
        if self.incremental_saver is not None:
            best_prompt, best_score = self._get_best_prompt()
            self.incremental_saver.finalize(best_prompt, best_score)


class VLLMChatWrapper:
    """Wrapper to make vLLM client compatible with GEPA's ChatCompletionCallable."""

    def __init__(
        self,
        llm_client,
        tracking_evaluator: Optional['TrackingEvaluatorWrapper'] = None,
    ):
        self.llm_client = llm_client
        self.call_count = 0
        self.tracking_evaluator = tracking_evaluator
        self._last_system_prompt = None

    def __call__(self, messages: list[dict]) -> str:
        self.call_count += 1

        # Combine system and user messages into a single prompt
        prompt_parts = []
        system_prompt = None
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(msg["content"])
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                prompt_parts.append(msg["content"])

        prompt = "\n\n".join(prompt_parts)

        # Track the system prompt - notify evaluator when prompt changes
        if system_prompt and system_prompt != self._last_system_prompt:
            logger.info(f"[VLLMChat] NEW PROMPT detected, tracking_evaluator={self.tracking_evaluator is not None}")
            logger.info(f"[VLLMChat] Prompt preview: {system_prompt[:100]}...")
            if self.tracking_evaluator is not None:
                self.tracking_evaluator.set_current_prompt(system_prompt)
            self._last_system_prompt = system_prompt
        elif self.call_count <= 3:
            logger.info(f"[VLLMChat] Same prompt (call #{self.call_count}), system_prompt={system_prompt is not None}")

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


class BatchingVLLMWrapper:
    """Batches sequential GEPA task_lm calls for massive speedup.

    GEPA's default adapter calls task_lm(messages) once per example
    sequentially: `[self.model(m) for m in requests]`. With N=7473
    trainset examples and ~6s per call, one prompt evaluation takes ~12h.

    This wrapper detects when a new candidate prompt is being evaluated
    (system_prompt changes), pre-generates responses for ALL trainset
    examples in one generate_batch() call, and serves from cache.

    Expected speedup: ~100-200x (batch of N in ~30s vs N×6s sequential).
    """

    def __init__(
        self,
        llm_client,
        trainset: list,
        tracking_evaluator: Optional['TrackingEvaluatorWrapper'] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        gen_batch_size: int = 256,
    ):
        self.llm_client = llm_client
        self.trainset = trainset
        self.tracking_evaluator = tracking_evaluator
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gen_batch_size = gen_batch_size

        self.call_count = 0
        self._current_system_prompt = None
        self._response_cache: dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._prefetch_count = 0

    def _build_prompt(self, system_prompt: str | None, user_content: str) -> str:
        """Join system + user into single prompt (same logic as VLLMChatWrapper)."""
        parts = [p for p in [system_prompt, user_content] if p]
        return "\n\n".join(parts)

    def _prefetch(self, system_prompt: str):
        """Batch-generate responses for all trainset examples with this prompt.

        GEPA sends messages as:
            [{"role": "system", "content": candidate_prompt},
             {"role": "user", "content": data["input"]}]

        We pre-compute: system_prompt + "\\n\\n" + inst.input for each instance.
        """
        self._prefetch_count += 1
        self._response_cache.clear()

        all_prompts = []
        for inst in self.trainset:
            full_prompt = self._build_prompt(system_prompt, inst.input)
            all_prompts.append(full_prompt)

        logger.info(
            f"[BatchWrapper] Prefetching {len(all_prompts)} responses "
            f"(prefetch #{self._prefetch_count})..."
        )

        t0 = time.time()
        all_responses = []
        for i in range(0, len(all_prompts), self.gen_batch_size):
            batch = all_prompts[i:i + self.gen_batch_size]
            responses = self.llm_client.generate_batch(
                batch,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            all_responses.extend(responses)
            if len(all_prompts) > self.gen_batch_size:
                logger.info(
                    f"[BatchWrapper] Sub-batch {i // self.gen_batch_size + 1}: "
                    f"{len(all_responses)}/{len(all_prompts)}"
                )

        for prompt, response in zip(all_prompts, all_responses):
            self._response_cache[prompt] = response

        elapsed = time.time() - t0
        logger.info(
            f"[BatchWrapper] Prefetched {len(all_responses)} responses in {elapsed:.1f}s "
            f"({len(all_responses) / max(elapsed, 0.1):.0f} req/s)"
        )

    def __call__(self, messages: list[dict]) -> str:
        self.call_count += 1

        # Parse messages (same format as VLLMChatWrapper)
        system_prompt = None
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_parts.append(msg["content"])

        user_content = "\n\n".join(user_parts) if user_parts else ""
        full_prompt = self._build_prompt(system_prompt, user_content)

        # Detect new candidate prompt → prefetch all trainset responses
        if system_prompt and system_prompt != self._current_system_prompt:
            self._current_system_prompt = system_prompt

            if self.tracking_evaluator is not None:
                self.tracking_evaluator.set_current_prompt(system_prompt)

            logger.info(
                f"[BatchWrapper] New prompt detected (call #{self.call_count}): "
                f"{system_prompt[:100]}..."
            )
            self._prefetch(system_prompt)

        # Serve from cache
        if full_prompt in self._response_cache:
            self._cache_hits += 1
            if self._cache_hits <= 3 or self._cache_hits % 200 == 0:
                logger.info(
                    f"[BatchWrapper] Cache: {self._cache_hits} hits, "
                    f"{self._cache_misses} misses"
                )
            return self._response_cache.pop(full_prompt)

        # Cache miss → direct call (reflection calls, unexpected formats)
        self._cache_misses += 1
        if self._cache_misses <= 5 or self._cache_misses % 50 == 0:
            logger.info(
                f"[BatchWrapper] Cache miss #{self._cache_misses} "
                f"(prompt_len={len(full_prompt)}, hits={self._cache_hits})"
            )

        try:
            response = self.llm_client.generate(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"[BatchWrapper] Call #{self.call_count} failed: {e}")
            raise

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

    # Benchmarking parameters
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Max prompts to evaluate (for benchmarking). Stops after evaluating this many prompts."
    )
    parser.add_argument(
        "--incremental-json", type=str, default=None,
        help="Path to save incremental JSON with evaluated prompts (for benchmarking)."
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Dataset split to use for evaluation (default: train). Use 'test' for final benchmarking."
    )

    # Other options
    parser.add_argument(
        "--results-dir", type=str, default="gepa_gsm8k/results",
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
    parser.add_argument(
        "--no-batch", action="store_true",
        help="Disable batching wrapper (use sequential calls like original)"
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
    logger.info(f"Split: {args.split}")
    logger.info(f"Max prompts: {args.max_prompts if args.max_prompts else 'unlimited'}")
    logger.info(f"Batching: {'disabled' if args.no_batch else 'enabled (prefetch)'}")
    if args.incremental_json:
        logger.info(f"Incremental JSON: {args.incremental_json}")
    logger.info("=" * 60)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading GSM8K dataset...")
    trainset = load_gsm8k_dataset("datasets/gsm8k", args.split)
    logger.info(f"Loaded {len(trainset)} {args.split} examples")

    # Initialize incremental saver if requested
    incremental_saver = None
    if args.incremental_json:
        from shared.incremental_saver import IncrementalPromptSaver
        config = {
            "max_calls": args.max_calls,
            "minibatch_size": args.minibatch_size,
            "frontier_type": args.frontier_type,
            "max_prompts": args.max_prompts,
            "split": args.split,
        }
        incremental_saver = IncrementalPromptSaver(
            output_path=args.incremental_json,
            method="gepa",
            model=args.model,
            config=config,
        )
        logger.info(f"Incremental saver initialized: {args.incremental_json}")

    # Create evaluator with optional tracking wrapper
    # (Must be created before LLM client so we can pass tracking_evaluator to VLLMChatWrapper)
    base_evaluator = GSM8KEvaluator()
    if args.max_prompts is not None or incremental_saver is not None:
        tracking_evaluator = TrackingEvaluatorWrapper(
            base_evaluator=base_evaluator,
            max_prompts=args.max_prompts,
            incremental_saver=incremental_saver,
        )
        evaluator = tracking_evaluator
    else:
        evaluator = base_evaluator
        tracking_evaluator = None

    # Initialize LLM client
    if args.backend == "vllm":
        from shared.llm_client import create_llm_client

        logger.info(f"Initializing {args.model} vLLM client...")
        llm_client = create_llm_client(
            args.model,
            backend="vllm",
            tensor_parallel_size=args.tensor_parallel_size
        )

        if args.no_batch:
            # Original sequential mode
            task_lm = VLLMChatWrapper(llm_client, tracking_evaluator=tracking_evaluator)
        else:
            # Batching mode: prefetch all trainset responses per prompt
            task_lm = BatchingVLLMWrapper(
                llm_client,
                trainset=trainset,
                tracking_evaluator=tracking_evaluator,
            )
            logger.info(
                f"[BatchWrapper] Initialized with {len(trainset)} trainset examples "
                f"(prefetch on prompt change)"
            )

        # Reflection always uses non-batching wrapper (different message patterns)
        if args.reflection_model == args.model:
            reflection_lm = VLLMChatWrapper(llm_client)
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

    # Import GEPA package (not local module)
    import importlib
    gepa_pkg = importlib.import_module("gepa")

    logger.info("Starting GEPA optimization...")
    start_time = time.time()

    # Run GEPA with handling for max_prompts exception
    result = None
    best_prompt = None
    best_score = 0.0
    stopped_early = False

    try:
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
        # Extract best prompt from result
        best_candidate = result.best_candidate
        best_prompt = best_candidate.get("system_prompt", str(best_candidate))
        best_score = result.best_score
    except MaxPromptsReachedException:
        logger.info("Optimization stopped early due to max_prompts limit.")
        stopped_early = True
        # Get best from tracking evaluator
        if tracking_evaluator is not None:
            best_prompt, best_score = tracking_evaluator._get_best_prompt()
        else:
            best_prompt = seed_candidate["system_prompt"]
            best_score = 0.0

    elapsed = time.time() - start_time

    # Finalize tracking if not stopped early (early stop already finalizes)
    if tracking_evaluator is not None and not stopped_early:
        tracking_evaluator.finalize()

    logger.info("\n" + "=" * 60)
    logger.info("GEPA OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best validation score: {best_score:.4f}")
    logger.info(f"Best prompt:\n{best_prompt}")
    logger.info(f"Total time: {elapsed:.1f}s")
    # Log batching stats if available
    if isinstance(task_lm, BatchingVLLMWrapper):
        logger.info(
            f"Batching stats: {task_lm._prefetch_count} prefetches, "
            f"{task_lm._cache_hits} cache hits, "
            f"{task_lm._cache_misses} cache misses "
            f"({100 * task_lm._cache_hits / max(task_lm._cache_hits + task_lm._cache_misses, 1):.1f}% hit rate)"
        )
    logger.info("=" * 60)

    # Save results
    batch_stats = None
    if isinstance(task_lm, BatchingVLLMWrapper):
        batch_stats = {
            "prefetch_count": task_lm._prefetch_count,
            "cache_hits": task_lm._cache_hits,
            "cache_misses": task_lm._cache_misses,
            "total_calls": task_lm.call_count,
        }

    results = {
        "method": "gepa",
        "best_prompt": best_prompt,
        "best_score": best_score,
        "total_time": elapsed,
        "stopped_early": stopped_early,
        "prompts_evaluated": tracking_evaluator.prompts_evaluated if tracking_evaluator else "N/A",
        "batch_stats": batch_stats,
        "config": {
            "model": args.model,
            "reflection_model": args.reflection_model,
            "max_calls": args.max_calls,
            "minibatch_size": args.minibatch_size,
            "frontier_type": args.frontier_type,
            "max_prompts": args.max_prompts,
            "split": args.split,
            "batching": not args.no_batch,
        }
    }
    if args.incremental_json:
        results["incremental_json"] = args.incremental_json

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
