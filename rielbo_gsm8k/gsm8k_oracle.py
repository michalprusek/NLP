"""GSM8K oracle for prompt optimization.

Evaluates a candidate prompt on a fixed GSM8K eval set and returns
accuracy as a scalar score, suitable for Bayesian optimization.

Uses the same evaluation pipeline as OPRO/ProTeGi for fair comparison.
"""

import hashlib
import logging
from typing import Optional

import torch

from shared.gsm8k_evaluator import GSM8KEvaluator

logger = logging.getLogger(__name__)


class GSM8KOracle:
    """Oracle that scores a text prompt by evaluating it on GSM8K.

    Maintains a fixed evaluation set (same indices every run) and
    caches results for duplicate prompts.

    Args:
        llm_client: LLMClient for generating task model responses
        evaluator: GSM8KEvaluator with loaded dataset
        eval_set_size: Number of examples in the fixed eval set
        seed: Random seed for selecting the fixed eval set
    """

    def __init__(
        self,
        llm_client,
        evaluator: GSM8KEvaluator,
        eval_set_size: int = 261,  # ~20% sample for quick iteration; CLI overrides to 1319 for benchmarking
        seed: int = 42,
    ):
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.eval_set_size = min(eval_set_size, len(evaluator))
        self.seed = seed

        # Select fixed eval set indices
        import random
        rng = random.Random(seed)
        all_indices = list(range(len(evaluator)))
        rng.shuffle(all_indices)
        self.eval_indices = sorted(all_indices[:self.eval_set_size])

        # Pre-load questions
        self.eval_questions = []
        for idx in self.eval_indices:
            example = evaluator.dataset[idx]
            self.eval_questions.append(example["question"])

        # Cache: prompt_hash -> score
        self._cache: dict[str, float] = {}
        self.num_calls = 0

        logger.info(
            f"GSM8KOracle: {self.eval_set_size} eval examples "
            f"(seed={seed}, total={len(evaluator)})"
        )

    def score(self, prompt: str) -> float:
        """Score a single prompt by evaluating on the fixed eval set.

        Args:
            prompt: The instruction/system prompt to evaluate

        Returns:
            Accuracy as a float in [0, 1]
        """
        # Check cache
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        if prompt_hash in self._cache:
            return self._cache[prompt_hash]

        self.num_calls += 1

        # Format prompts: Q: {question}\n{instruction}\nA:
        formatted = [
            f"Q: {q}\n{prompt}\nA:" for q in self.eval_questions
        ]

        # Batch generate
        try:
            outputs = self.llm_client.generate_batch(
                formatted, temperature=0.0, max_new_tokens=512
            )
        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"CUDA OOM during LLM generation (oracle call #{self.num_calls}, "
                f"batch size {len(formatted)}). Try reducing --eval-size."
            )
            raise
        except Exception as e:
            logger.error(f"LLM generation failed during oracle call #{self.num_calls}: {e}")
            raise RuntimeError(
                f"LLM generation failed during oracle call #{self.num_calls}. "
                f"Check vLLM server status. Original error: {e}"
            ) from e

        # Evaluate
        try:
            results = self.evaluator.evaluate_batch(outputs, self.eval_indices)
        except Exception as e:
            logger.error(f"Evaluation failed during oracle call #{self.num_calls}: {e}")
            raise RuntimeError(
                f"Evaluation failed during oracle call #{self.num_calls}: {e}"
            ) from e
        accuracy = results["accuracy"]

        # Cache result
        self._cache[prompt_hash] = accuracy

        logger.info(
            f"Oracle call #{self.num_calls}: "
            f"{accuracy:.4f} ({results['correct']}/{results['total']})"
        )

        return accuracy

    def score_batch(self, prompts: list[str]) -> torch.Tensor:
        """Score multiple prompts.

        Args:
            prompts: List of prompt strings to evaluate

        Returns:
            Tensor of accuracy scores [N]
        """
        scores = [self.score(p) for p in prompts]
        return torch.tensor(scores, dtype=torch.float32)

    @property
    def cache_size(self) -> int:
        return len(self._cache)
