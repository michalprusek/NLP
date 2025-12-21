"""
Batch per-model evaluation with Hoeffding bounds.

This module implements batch evaluation where:
1. All candidates are evaluated on one model before switching to the next
2. Each candidate uses Hoeffding bounds for early stopping
3. Minimizes GPU memory operations by reducing model switches

Strategy:
    Load Model A -> Evaluate all candidates with Hoeffding -> Unload
    Load Model B -> Evaluate all candidates with Hoeffding -> Unload
    Load Model C -> Evaluate all candidates with Hoeffding -> Unload
"""
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_model_hybrid.config import (
    MultiModelHybridConfig,
    MultiModelHybridCandidate,
)
from multi_model_optimizer.evaluator_pool import (
    SingleGPUModelManager,
    extract_answer,
    compare_answers,
)


class EvaluationError(Exception):
    """Raised when LLM evaluation fails (distinct from candidate performing poorly)."""
    pass


class HoeffdingDecision(Enum):
    """Decision outcome from Hoeffding-based sequential test."""
    DROP = "drop"
    """Upper bound < best_accuracy: candidate can't beat champion."""

    PROMOTE = "promote"
    """Lower bound > best_accuracy: candidate definitely better."""

    CONTINUE = "continue"
    """Need more samples to decide."""

    SKIPPED = "skipped"
    """Evaluation failed due to LLM error (not a valid result)."""


class BatchPerModelEvaluator:
    """
    Evaluates candidates in batch per model with Hoeffding early stopping.

    This class implements the Phase 4 evaluation strategy:
    1. Load Model A
    2. For each candidate in union:
       - Evaluate with Hoeffding bounds (exponential schedule: 10, 20, 40, ...)
       - Stop early if DROP or PROMOTE
    3. Unload Model A
    4. Repeat for Model B, C, ...

    Example:
        >>> evaluator = BatchPerModelEvaluator(config, validation_data)
        >>> candidates, budget = evaluator.evaluate_candidates_batch(
        ...     candidates=union_candidates,
        ...     best_aggregated_accuracy=0.75,
        ...     budget_remaining=10000,
        ... )
    """

    def __init__(
        self,
        config: MultiModelHybridConfig,
        validation_data: List[Dict],
        single_gpu_manager: Optional[SingleGPUModelManager] = None,
    ):
        """
        Initialize batch evaluator.

        Args:
            config: Multi-model hybrid configuration
            validation_data: List of {"question": str, "answer": str}
            single_gpu_manager: GPU model manager (created if None)
        """
        self.config = config
        self.validation_data = validation_data
        self.nvalid = len(validation_data)

        # Hoeffding parameters
        self.confidence = config.hoeffding_confidence
        self.delta = 1 - self.confidence
        self.min_samples = config.hoeffding_min_samples
        self.min_promote_samples = config.hoeffding_min_promote_samples

        # Model manager
        if single_gpu_manager is not None:
            self.model_manager = single_gpu_manager
        else:
            self.model_manager = SingleGPUModelManager(
                gpu_id=config.single_gpu_id,
                gpu_memory_utilization=0.85,
            )

        # Budget tracking
        self.budget_used = 0

    def evaluate_candidates_batch(
        self,
        candidates: List[MultiModelHybridCandidate],
        best_aggregated_accuracy: float,
        budget_remaining: int,
        verbose: bool = True,
    ) -> Tuple[List[MultiModelHybridCandidate], int]:
        """
        Evaluate all candidates on all models with batch-per-model strategy.

        Args:
            candidates: Union of per-model selected candidates
            best_aggregated_accuracy: Current champion (for Hoeffding comparison)
            budget_remaining: Maximum additional LLM calls allowed
            verbose: Print progress

        Returns:
            Tuple of:
                - Updated candidates with actual_errors and decisions
                - Budget consumed
        """
        initial_budget = self.budget_used
        local_budget = budget_remaining

        for model_name in self.config.target_models:
            if verbose:
                model_short = model_name.split("/")[-1]
                print(f"\n  Loading model: {model_short}")

            # Load model ONCE
            client = self.model_manager.load_model(
                model_name,
                self.config.task_max_tokens
            )

            # Evaluate all candidates on this model
            for i, candidate in enumerate(candidates):
                if local_budget <= 0:
                    if verbose:
                        print("    Budget exhausted")
                    break

                # Get best accuracy for comparison
                # Use per-model best if available, otherwise aggregated
                best_model_acc = best_aggregated_accuracy

                # Evaluate with Hoeffding bounds
                error_rate, fidelity, decision = self._evaluate_with_hoeffding(
                    candidate=candidate,
                    model_name=model_name,
                    client=client,
                    best_accuracy=best_model_acc,
                    budget_limit=local_budget,
                    verbose=verbose,
                )

                # Store results
                candidate.actual_errors[model_name] = error_rate
                candidate.actual_fidelities[model_name] = fidelity
                candidate.decisions[model_name] = decision.value

                # Update budget
                local_budget -= fidelity
                self.budget_used += fidelity

                if verbose:
                    acc = 1 - error_rate
                    print(
                        f"    Candidate {i+1}/{len(candidates)}: "
                        f"acc={acc:.2%}, n={fidelity}, decision={decision.value}"
                    )

            # Unload model to free GPU memory before loading next model
            if verbose:
                print(f"  Unloading model: {model_short}")
            self.model_manager.unload()

        budget_consumed = self.budget_used - initial_budget
        return candidates, budget_consumed

    def _evaluate_with_hoeffding(
        self,
        candidate: MultiModelHybridCandidate,
        model_name: str,
        client: Any,
        best_accuracy: float,
        budget_limit: int,
        verbose: bool,
    ) -> Tuple[float, int, HoeffdingDecision]:
        """
        Evaluate candidate on one model with Hoeffding early stopping.

        Uses exponential sample schedule: 10, 20, 40, 80, ...

        Args:
            candidate: Candidate to evaluate
            model_name: Model name (for logging)
            client: vLLM client
            best_accuracy: Champion accuracy for comparison
            budget_limit: Maximum samples allowed
            verbose: Print progress

        Returns:
            Tuple of (error_rate, samples_used, decision)
            If evaluation fails, returns (1.0, 0, SKIPPED).
        """
        n_samples = 0
        successes = 0
        decision = HoeffdingDecision.CONTINUE

        try:
            while decision == HoeffdingDecision.CONTINUE:
                # Get next step
                next_n = self._get_next_step(n_samples)
                next_n = min(next_n, self.nvalid, n_samples + budget_limit)

                if next_n <= n_samples:
                    break

                # Evaluate additional samples
                new_successes = self._evaluate_samples(
                    candidate,
                    model_name,
                    client,
                    self.validation_data[n_samples:next_n]
                )

                successes += new_successes
                n_samples = next_n

                # Make decision
                decision = self._hoeffding_decide(
                    successes, n_samples, best_accuracy
                )

                if n_samples >= self.nvalid:
                    break
        except EvaluationError as e:
            if verbose:
                print(f"      [SKIPPED] {e}")
            return 1.0, 0, HoeffdingDecision.SKIPPED

        error_rate = 1 - (successes / n_samples) if n_samples > 0 else 1.0
        return error_rate, n_samples, decision

    def _evaluate_samples(
        self,
        candidate: MultiModelHybridCandidate,
        model_name: str,
        client: Any,
        data: List[Dict],
    ) -> int:
        """
        Evaluate candidate on specific samples, return number of successes.

        Args:
            candidate: Candidate to evaluate
            model_name: Model name (unused, for future extensibility)
            client: vLLM client
            data: List of {"question": str, "answer": str}

        Returns:
            Number of correct answers (successes)
        """
        if not data:
            return 0

        # Format prompts
        prompts = [
            f"Question: {d['question']}\n\n{candidate.instruction}\n\n"
            f"{candidate.exemplar}\n\nAnswer:"
            for d in data
        ]

        # Generate responses
        try:
            responses = client.generate_batch(
                prompts,
                max_new_tokens=self.config.task_max_tokens,
                temperature=0.0,
            )
        except KeyboardInterrupt:
            raise  # Never swallow keyboard interrupt
        except Exception as e:
            raise EvaluationError(
                f"LLM generation failed for {len(prompts)} prompts: {e}"
            ) from e

        # Count successes
        successes = 0
        for response, d in zip(responses, data):
            extracted = extract_answer(response)
            if compare_answers(extracted, d["answer"]):
                successes += 1

        return successes

    def _hoeffding_decide(
        self,
        successes: int,
        n_samples: int,
        best_accuracy: float,
    ) -> HoeffdingDecision:
        """
        Make Hoeffding-based decision.

        Hoeffding's inequality for n Bernoulli samples:
            P(|p̂ - p| ≥ ε) ≤ 2·exp(-2nε²)

        For confidence level (1-δ):
            ε = √(ln(2/δ) / (2n))

        Args:
            successes: Number of correct predictions
            n_samples: Total samples evaluated
            best_accuracy: Current champion's accuracy

        Returns:
            Decision: DROP, PROMOTE, or CONTINUE
        """
        if n_samples < self.min_samples:
            return HoeffdingDecision.CONTINUE

        accuracy = successes / n_samples
        epsilon = self._hoeffding_bound(n_samples)

        upper_bound = min(1.0, accuracy + epsilon)
        lower_bound = max(0.0, accuracy - epsilon)

        # DROP: Even optimistic estimate can't beat champion
        if upper_bound < best_accuracy:
            return HoeffdingDecision.DROP

        # PROMOTE: Even pessimistic estimate is better (with enough samples)
        if lower_bound > best_accuracy and n_samples >= self.min_promote_samples:
            return HoeffdingDecision.PROMOTE

        return HoeffdingDecision.CONTINUE

    def _hoeffding_bound(self, n: int) -> float:
        """
        Calculate Hoeffding error bound ε for n samples.

        For 95% confidence: ε ≈ √(1.84/n)

        Args:
            n: Number of samples

        Returns:
            Error bound ε such that P(|p̂ - p| ≥ ε) ≤ δ
        """
        if n <= 0:
            return 1.0
        return math.sqrt(math.log(2 / self.delta) / (2 * n))

    def _get_next_step(self, current_n: int) -> int:
        """
        Get next sample size from exponential schedule.

        Schedule: 10, 20, 40, 80, 160, 320, ...

        Args:
            current_n: Current number of samples

        Returns:
            Next sample size (10 * 2^k pattern)
        """
        n = self.min_samples
        while n <= current_n:
            n *= 2
        return n

    def get_sample_schedule(self, max_n: int) -> List[int]:
        """
        Generate full exponential sample schedule.

        Args:
            max_n: Maximum samples (full fidelity)

        Returns:
            List of sample sizes: [10, 20, 40, 80, ..., max_n]
        """
        schedule = []
        n = self.min_samples

        while n < max_n:
            schedule.append(n)
            n *= 2

        schedule.append(max_n)
        return schedule
