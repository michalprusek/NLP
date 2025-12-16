"""
Multi-Model Sequential Testing with Hoeffding Bounds and Bonferroni Correction.

Extends sequential testing to handle multiple models with joint confidence
guarantees using union bound (Bonferroni) correction.
"""
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_model_optimizer.aggregation import aggregate_scores, compute_bounds_aggregated


class Decision(Enum):
    """Decision outcome from sequential testing."""

    DROP = "drop"  # Aggregated upper bound < best → can't beat champion
    PROMOTE = "promote"  # Aggregated lower bound > best → definitely better
    CONTINUE = "continue"  # Need more samples


class MultiModelSequentialTester:
    """
    Sequential testing with Hoeffding bounds for multiple models.

    Uses Bonferroni correction for joint confidence:
        For k models with target confidence (1-α), use per-model confidence (1-α/k)

    This ensures that the joint probability of all bounds holding is at least (1-α).

    Example:
        For 4 models with 95% joint confidence:
        - Per-model confidence = 1 - 0.05/4 = 0.9875
        - Per-model δ = 0.0125
    """

    def __init__(
        self,
        model_names: List[str],
        confidence: float = 0.95,
        min_samples: int = 10,
        min_promote_samples: int = 30,
        aggregation: str = "weighted_softmin",
        temperature: float = 0.1,
    ):
        """
        Initialize multi-model sequential tester.

        Args:
            model_names: List of model names (for tracking)
            confidence: Joint confidence level (default 0.95)
            min_samples: Minimum samples before any decision
            min_promote_samples: Minimum samples before PROMOTE decision
            aggregation: Aggregation strategy for combining model scores
            temperature: Temperature for weighted_softmin
        """
        self.model_names = model_names
        self.num_models = len(model_names)

        # Bonferroni correction: divide alpha by number of models
        alpha = 1 - confidence
        per_model_alpha = alpha / self.num_models
        self.per_model_confidence = 1 - per_model_alpha
        self.per_model_delta = per_model_alpha

        self.min_samples = min_samples
        self.min_promote_samples = min_promote_samples
        self.aggregation = aggregation
        self.temperature = temperature

    def hoeffding_bound(self, n: int) -> float:
        """
        Calculate Hoeffding error bound ε for n samples with Bonferroni correction.

        Uses per-model δ (corrected for multiple comparisons).

        Args:
            n: Number of samples

        Returns:
            Error bound ε such that P(|p̂ - p| ≥ ε) ≤ δ_per_model
        """
        if n <= 0:
            return 1.0
        return math.sqrt(math.log(2 / self.per_model_delta) / (2 * n))

    def compute_per_model_bounds(
        self,
        model_successes: Dict[str, int],
        n_samples: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-model accuracy bounds.

        Args:
            model_successes: Dict mapping model_name -> number of correct predictions
            n_samples: Total samples evaluated

        Returns:
            Dict mapping model_name -> {"accuracy": float, "lower": float, "upper": float}
        """
        epsilon = self.hoeffding_bound(n_samples)
        bounds = {}

        for model_name in self.model_names:
            successes = model_successes.get(model_name, 0)
            accuracy = successes / n_samples if n_samples > 0 else 0.0

            bounds[model_name] = {
                "accuracy": accuracy,
                "lower": max(0.0, accuracy - epsilon),
                "upper": min(1.0, accuracy + epsilon),
            }

        return bounds

    def decide(
        self,
        model_successes: Dict[str, int],
        n_samples: int,
        best_aggregated_accuracy: float,
    ) -> Tuple[Decision, Dict[str, Dict[str, float]], float]:
        """
        Decide whether to DROP, PROMOTE, or CONTINUE based on aggregated bounds.

        Strategy:
        1. Compute Hoeffding bounds for each model (with Bonferroni correction)
        2. Aggregate lower and upper bounds across models
        3. Compare aggregated bounds to champion

        Args:
            model_successes: Dict mapping model_name -> number of correct predictions
            n_samples: Total samples evaluated
            best_aggregated_accuracy: Current champion's aggregated accuracy

        Returns:
            Tuple of:
                - Decision: DROP, PROMOTE, or CONTINUE
                - model_bounds: Per-model accuracy bounds
                - aggregated_accuracy: Current aggregated accuracy estimate
        """
        if n_samples < self.min_samples:
            # Not enough samples yet
            bounds = self.compute_per_model_bounds(model_successes, n_samples)
            accuracies = {m: b["accuracy"] for m, b in bounds.items()}
            agg_acc = 1.0 - aggregate_scores(
                {m: 1.0 - a for m, a in accuracies.items()},
                self.aggregation,
                self.temperature,
            )
            return Decision.CONTINUE, bounds, agg_acc

        # Compute per-model bounds
        bounds = self.compute_per_model_bounds(model_successes, n_samples)

        # Aggregate bounds
        agg_lower_acc, agg_upper_acc = compute_bounds_aggregated(
            bounds,
            self.aggregation,
            self.temperature,
        )

        # Current point estimate
        accuracies = {m: b["accuracy"] for m, b in bounds.items()}
        agg_accuracy = 1.0 - aggregate_scores(
            {m: 1.0 - a for m, a in accuracies.items()},
            self.aggregation,
            self.temperature,
        )

        # Decision logic
        # DROP: Even optimistic estimate can't beat champion
        if agg_upper_acc < best_aggregated_accuracy:
            return Decision.DROP, bounds, agg_accuracy

        # PROMOTE: Even pessimistic estimate is better (with enough samples)
        if agg_lower_acc > best_aggregated_accuracy and n_samples >= self.min_promote_samples:
            return Decision.PROMOTE, bounds, agg_accuracy

        return Decision.CONTINUE, bounds, agg_accuracy

    def get_sample_schedule(self, max_n: int) -> List[int]:
        """
        Generate exponential sample schedule: 10 * 2^k.

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

    def get_next_step(self, current_n: int, max_n: int) -> int:
        """
        Get next sample size from exponential schedule.

        Args:
            current_n: Current number of samples
            max_n: Maximum samples (full fidelity)

        Returns:
            Next sample size (10 * 2^k pattern)
        """
        if current_n >= max_n:
            return max_n

        n = self.min_samples
        while n <= current_n:
            n *= 2

        return min(n, max_n)

    def format_decision_report(
        self,
        decision: Decision,
        bounds: Dict[str, Dict[str, float]],
        agg_accuracy: float,
        n_samples: int,
        best_accuracy: float,
    ) -> str:
        """
        Format a human-readable decision report.

        Args:
            decision: The decision made
            bounds: Per-model bounds
            agg_accuracy: Aggregated accuracy
            n_samples: Samples used
            best_accuracy: Champion accuracy

        Returns:
            Formatted string report
        """
        lines = [
            f"Decision: {decision.value.upper()}",
            f"Samples: {n_samples}",
            f"Aggregated Accuracy: {agg_accuracy:.2%}",
            f"Champion Accuracy: {best_accuracy:.2%}",
            "Per-model results:",
        ]

        for model_name in self.model_names:
            b = bounds[model_name]
            lines.append(
                f"  {model_name}: {b['accuracy']:.2%} "
                f"[{b['lower']:.2%}, {b['upper']:.2%}]"
            )

        return "\n".join(lines)
