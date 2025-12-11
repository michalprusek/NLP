"""
OPRO-style instruction generation adapter.

Uses a meta-LLM to generate new instructions based on previously scored ones.
Adapted for hybrid OPRO + HbBoPs where we generate instructions (not full prompts).
"""
from typing import List, Set
import re

from .config import ScoredInstruction


# Meta-prompt template for instruction generation
META_PROMPT_TEMPLATE = """I have some instruction texts along with their corresponding accuracy scores.
The instructions are arranged in ascending order based on their scores, where higher scores indicate better quality.

{scored_instructions}

These instructions are used to guide a language model in solving math word problems.
A good instruction should:
1. Be clear and specific about the expected reasoning process
2. Encourage step-by-step problem solving
3. Specify the expected answer format (e.g., "#### number")

Write a new instruction that is different from the old ones and achieves a higher accuracy score.
Write the instruction in square brackets.
"""


def bucket_score(score: float, num_buckets: int = 20) -> int:
    """
    Bucketize score to nearest 5% (returns integer 0-100).

    Args:
        score: Accuracy score in [0, 1]
        num_buckets: Number of buckets (default 20 = 5% increments)

    Returns:
        Integer score 0-100
    """
    bucket_size = 1.0 / num_buckets
    return int(round(score / bucket_size) * bucket_size * 100)


class OPROInstructionGenerator:
    """
    Adapter for OPRO-style instruction generation.

    Uses a meta-model to generate new instructions based on
    previously scored instructions.
    """

    def __init__(
        self,
        meta_llm,
        num_candidates: int = 8,
        temperature: float = 1.0,
        max_tokens: int = 500,
    ):
        """
        Args:
            meta_llm: LLM client for generation (vLLM or API)
            num_candidates: Number of candidates to generate per call
            temperature: Sampling temperature for diversity
            max_tokens: Maximum tokens per generation
        """
        self.meta_llm = meta_llm
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_candidates(
        self,
        scored_instructions: List[ScoredInstruction],
        existing_instructions: Set[str] = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Generate new instruction candidates using OPRO meta-prompting.

        Args:
            scored_instructions: List of ScoredInstruction with accuracy scores
            existing_instructions: Set of existing instructions to avoid duplicates
            verbose: Print progress

        Returns:
            List of new unique instruction strings
        """
        if existing_instructions is None:
            existing_instructions = set()

        # Format scored instructions context (ascending by score - worst to best)
        sorted_instructions = sorted(
            scored_instructions, key=lambda x: x.best_accuracy
        )

        scored_text = "\n".join(
            [
                f"text: {si.instruction}\nscore: {bucket_score(si.best_accuracy)}"
                for si in sorted_instructions
            ]
        )

        candidates = []
        seen = set(existing_instructions)
        seen.update(si.instruction for si in scored_instructions)

        attempts = 0
        max_attempts = self.num_candidates * 3  # Allow some retries for duplicates

        while len(candidates) < self.num_candidates and attempts < max_attempts:
            attempts += 1

            # Format meta-prompt
            meta_prompt = META_PROMPT_TEMPLATE.format(scored_instructions=scored_text)

            # Generate
            try:
                response = self.meta_llm.generate(
                    meta_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                )
            except Exception as e:
                if verbose:
                    print(f"  Generation error: {e}")
                continue

            # Extract from brackets
            candidate = self._extract_from_brackets(response)

            if not candidate:
                if verbose:
                    print(f"  Attempt {attempts}: No valid instruction extracted")
                continue

            # Add if unique
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
                if verbose:
                    preview = candidate[:60] + "..." if len(candidate) > 60 else candidate
                    print(f"  Generated {len(candidates)}/{self.num_candidates}: {preview}")
            else:
                if verbose:
                    print(f"  Attempt {attempts}: Duplicate instruction, skipping")

        if verbose and len(candidates) < self.num_candidates:
            print(
                f"  Warning: Only generated {len(candidates)}/{self.num_candidates} unique instructions"
            )

        return candidates

    def _extract_from_brackets(self, text: str) -> str:
        """
        Extract instruction from [...] brackets.

        Args:
            text: Raw LLM output

        Returns:
            Extracted instruction or empty string
        """
        if not text:
            return ""

        text = text.strip()

        # Try to find content in square brackets
        bracket_pattern = r'\[([^\[\]]+)\]'
        matches = re.findall(bracket_pattern, text)

        if matches:
            # Return the last match (usually the actual instruction)
            return matches[-1].strip()

        # Fallback: if no brackets, check if response is short enough to be an instruction
        if len(text) < 500 and '\n' not in text:
            return text

        return ""

    def generate_single(
        self,
        scored_instructions: List[ScoredInstruction],
        existing_instructions: Set[str] = None,
    ) -> str:
        """
        Generate a single new instruction.

        Args:
            scored_instructions: Context of scored instructions
            existing_instructions: Set to avoid duplicates

        Returns:
            New instruction string or empty string if failed
        """
        results = self.generate_candidates(
            scored_instructions,
            existing_instructions,
            verbose=False,
        )
        return results[0] if results else ""
