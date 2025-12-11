"""
Dynamic exemplar sampling from GSM8K train set.

Each exemplar consists of k Q/A pairs sampled randomly from the training data.
"""
import random
import re
from typing import List, Dict, Set, FrozenSet


class ExemplarSampler:
    """
    Dynamic exemplar sampling from GSM8K train set.

    Each exemplar consists of k Q/A pairs sampled randomly.
    Ensures no duplicates across samples within the optimization run.
    """

    def __init__(self, gsm8k_train_data: List[Dict], seed: int = 42):
        """
        Args:
            gsm8k_train_data: List of {'question': str, 'answer': str}
            seed: Random seed for reproducibility
        """
        self.train_data = gsm8k_train_data
        self.rng = random.Random(seed)
        self.used_combinations: Set[FrozenSet[int]] = set()

    def sample(self, n: int = 25, k: int = 5) -> List[str]:
        """
        Sample n new exemplars, each containing k Q/A pairs.

        Args:
            n: Number of exemplars to sample
            k: Number of Q/A pairs per exemplar

        Returns:
            List of exemplar strings (each is k Q/A pairs joined with newlines)
        """
        exemplars = []
        attempts = 0
        max_attempts = n * 100

        while len(exemplars) < n and attempts < max_attempts:
            attempts += 1

            # Sample k random indices
            indices = tuple(sorted(self.rng.sample(range(len(self.train_data)), k)))
            indices_set = frozenset(indices)

            # Check for duplicate combination
            if indices_set in self.used_combinations:
                continue

            self.used_combinations.add(indices_set)

            # Build exemplar string
            qa_pairs = []
            for idx in indices:
                ex = self.train_data[idx]
                answer = self._format_answer(ex["answer"])
                qa_pairs.append(f"Q: {ex['question']}\nA: {answer}")

            exemplars.append("\n\n".join(qa_pairs))

        return exemplars

    def _format_answer(self, answer: str) -> str:
        """
        Format GSM8K answer for exemplar.

        GSM8K answers contain step-by-step reasoning with #### final_answer.
        We keep the full reasoning for few-shot learning.

        Args:
            answer: Raw GSM8K answer string

        Returns:
            Formatted answer string
        """
        # Keep full answer with reasoning, just clean up whitespace
        answer = answer.strip()

        # If answer has #### marker, ensure it's properly formatted
        if "####" in answer:
            parts = answer.split("####")
            reasoning = parts[0].strip()
            final = parts[1].strip() if len(parts) > 1 else ""
            # Format with clear final answer
            if reasoning:
                return f"{reasoning}\n#### {final}"
            return f"#### {final}"

        return answer

    def sample_single(self, k: int = 5) -> str:
        """
        Sample a single exemplar with k Q/A pairs.

        Args:
            k: Number of Q/A pairs

        Returns:
            Exemplar string
        """
        result = self.sample(n=1, k=k)
        return result[0] if result else ""

    def reset(self) -> None:
        """Reset used combinations (for new optimization run)."""
        self.used_combinations.clear()

    @property
    def num_available(self) -> int:
        """Number of training examples available."""
        return len(self.train_data)

    @property
    def num_sampled(self) -> int:
        """Number of unique exemplar combinations sampled so far."""
        return len(self.used_combinations)
