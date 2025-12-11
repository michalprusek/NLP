"""
Sequential Testing with Hoeffding Bounds.

Implements dynamic sample sizing with early stopping based on
Hoeffding's inequality for efficient prompt evaluation.
"""
import math
from enum import Enum
from typing import Tuple, List


class Decision(Enum):
    """Decision outcome from sequential testing."""
    DROP = "drop"        # Upper bound < best_accuracy → can't beat champion
    PROMOTE = "promote"  # Lower bound > best_accuracy → definitely better
    CONTINUE = "continue"  # Need more samples


class SequentialTester:
    """
    Sequential testing with Hoeffding bounds for Bernoulli outcomes.

    Uses exponential sample progression: 10 * 2^k (10, 20, 40, 80, ...)
    until full fidelity is reached.

    Hoeffding's inequality for n Bernoulli samples:
        P(|p̂ - p| ≥ ε) ≤ 2·exp(-2nε²)

    For confidence level (1-δ):
        ε = √(ln(2/δ) / (2n))
    """

    def __init__(
        self,
        confidence: float = 0.95,
        min_samples: int = 10,
        min_promote_samples: int = 30,
    ):
        """
        Initialize sequential tester.

        Args:
            confidence: Confidence level (default 0.95 = 95%)
            min_samples: Minimum samples before any decision (default 10)
            min_promote_samples: Minimum samples before PROMOTE decision (default 30)
        """
        self.delta = 1 - confidence  # 0.05 for 95% confidence
        self.min_samples = min_samples
        self.min_promote_samples = min_promote_samples

    def hoeffding_bound(self, n: int) -> float:
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

    def decide(
        self,
        successes: int,
        n_samples: int,
        best_accuracy: float,
    ) -> Tuple[Decision, float, float, float]:
        """
        Decide whether to DROP, PROMOTE, or CONTINUE testing.

        Args:
            successes: Number of correct predictions
            n_samples: Total samples evaluated
            best_accuracy: Current champion's accuracy

        Returns:
            Tuple of (decision, accuracy, lower_bound, upper_bound)
        """
        if n_samples < self.min_samples:
            # Not enough samples yet
            accuracy = successes / n_samples if n_samples > 0 else 0.0
            return Decision.CONTINUE, accuracy, 0.0, 1.0

        accuracy = successes / n_samples
        epsilon = self.hoeffding_bound(n_samples)

        upper_bound = min(1.0, accuracy + epsilon)
        lower_bound = max(0.0, accuracy - epsilon)

        # DROP: Even optimistic estimate can't beat champion
        if upper_bound < best_accuracy:
            return Decision.DROP, accuracy, lower_bound, upper_bound

        # PROMOTE: Even pessimistic estimate is better (with enough samples)
        if lower_bound > best_accuracy and n_samples >= self.min_promote_samples:
            return Decision.PROMOTE, accuracy, lower_bound, upper_bound

        return Decision.CONTINUE, accuracy, lower_bound, upper_bound

    def get_sample_schedule(self, max_n: int) -> List[int]:
        """
        Generate exponential sample schedule: 10 * 2^k.

        Args:
            max_n: Maximum samples (full fidelity)

        Returns:
            List of sample sizes: [10, 20, 40, 80, ..., max_n]
        """
        schedule = []
        n = self.min_samples  # Start at 10

        while n < max_n:
            schedule.append(n)
            n *= 2  # Exponential: 10, 20, 40, 80, 160, 320, 640, 1280...

        schedule.append(max_n)  # Always include full fidelity
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

        # Find next power: 10 * 2^k > current_n
        n = self.min_samples
        while n <= current_n:
            n *= 2

        return min(n, max_n)
