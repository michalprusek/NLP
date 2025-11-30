"""
Method B: Component Recombination Generator

Genetic crossover of high-performing components.
ZERO LLM cost - pure combinatorial.

Key insight: If instruction A and exemplar B both have high scores
independently, their combination (A, B) likely performs well too.
"""

import random
import re
from typing import List, Dict, Set, Tuple
from hype.data_types import (
    Instruction, Exemplar, EvaluationRecord,
    ComponentScore, GenerationResult, ComponentSource
)


class RecombinationGenerator:
    """
    Generate new prompts via recombination of top components.

    Two strategies:
    1. Priority combinations: Identify top instructions x top exemplars
       that haven't been tried yet
    2. Instruction crossover: Combine clauses from high-scoring instructions
    """

    def __init__(
        self,
        top_k_fraction: float = 0.2,  # Use top 20% components
        crossover_probability: float = 0.3,  # Probability of clause crossover
        max_new_instructions: int = 5,
        max_new_exemplars: int = 0,  # Recombination mainly generates instruction variants
        seed: int = None
    ):
        self.top_k_fraction = top_k_fraction
        self.crossover_probability = crossover_probability
        self.max_new_instructions = max_new_instructions
        self.max_new_exemplars = max_new_exemplars
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "recombination"

    @property
    def requires_llm(self) -> bool:
        return False

    def generate(
        self,
        instructions: List[Instruction],
        exemplars: List[Exemplar],
        instruction_scores: Dict[int, ComponentScore],
        exemplar_scores: Dict[int, ComponentScore],
        evaluation_records: List[EvaluationRecord],
        generation: int,
        **kwargs
    ) -> GenerationResult:
        """
        Generate new components via recombination.

        1. Identify top-K instructions and exemplars
        2. Find untried combinations (priority pairs)
        3. Generate instruction variants via clause crossover
        """
        # Get top-K components
        top_k = max(2, int(len(instructions) * self.top_k_fraction))
        top_instructions = self._get_top_k(instruction_scores, top_k)
        top_exemplars = self._get_top_k(exemplar_scores, top_k)

        # Find which pairs have been tried
        tried_pairs: Set[Tuple[int, int]] = set()
        for record in evaluation_records:
            tried_pairs.add((record.instruction_id, record.exemplar_id))

        # Identify priority combinations (top x top, not yet tried)
        priority_pairs = []
        for inst_id in top_instructions:
            for ex_id in top_exemplars:
                if (inst_id, ex_id) not in tried_pairs:
                    priority_pairs.append((inst_id, ex_id))

        # Generate new instructions via crossover
        new_instructions = self._generate_crossover_instructions(
            instructions,
            instruction_scores,
            generation
        )

        metadata = {
            "top_instructions": top_instructions,
            "top_exemplars": top_exemplars,
            "priority_pairs": priority_pairs[:20],  # Top 20 untried pairs
            "num_crossover_instructions": len(new_instructions),
        }

        return GenerationResult(
            new_instructions=new_instructions,
            new_exemplars=[],  # Recombination doesn't generate new exemplars
            metadata=metadata
        )

    def _get_top_k(
        self, scores: Dict[int, ComponentScore], k: int
    ) -> List[int]:
        """Get IDs of top-K components by score"""
        sorted_ids = sorted(
            scores.keys(),
            key=lambda x: scores[x].score,
            reverse=True
        )
        return sorted_ids[:k]

    def _generate_crossover_instructions(
        self,
        instructions: List[Instruction],
        scores: Dict[int, ComponentScore],
        generation: int
    ) -> List[Instruction]:
        """
        Generate new instructions by crossing over clauses from parents.

        Strategy:
        1. Select two high-scoring parents
        2. Split each into clauses (by sentence boundaries)
        3. Combine clauses from both parents
        4. Apply minor mutations
        """
        if len(instructions) < 2:
            return []

        # Get parent pool (top 50%)
        sorted_by_score = sorted(
            instructions,
            key=lambda i: scores.get(i.id, ComponentScore(i.id, 0)).score,
            reverse=True
        )
        parent_pool = sorted_by_score[:max(2, len(sorted_by_score) // 2)]

        new_instructions = []
        existing_texts = {i.text.lower().strip() for i in instructions}
        next_id = max(i.id for i in instructions) + 1

        attempts = 0
        max_attempts = self.max_new_instructions * 3

        while len(new_instructions) < self.max_new_instructions and attempts < max_attempts:
            attempts += 1

            if self.rng.random() < self.crossover_probability and len(parent_pool) >= 2:
                # Crossover: combine clauses from two parents
                p1, p2 = self.rng.sample(parent_pool, 2)
                child_text = self._crossover(p1.text, p2.text)
                parent_ids = [p1.id, p2.id]
            else:
                # Mutation: slightly modify a single parent
                parent = self.rng.choice(parent_pool)
                child_text = self._mutate(parent.text)
                parent_ids = [parent.id]

            # Clean up the result
            child_text = self._clean_text(child_text)

            # Check for duplicates
            if child_text.lower().strip() in existing_texts:
                continue
            if len(child_text) < 20:  # Too short
                continue

            new_inst = Instruction(
                id=next_id,
                text=child_text,
                source=ComponentSource.RECOMBINATION,
                generation=generation,
                parent_ids=parent_ids
            )
            new_instructions.append(new_inst)
            existing_texts.add(child_text.lower().strip())
            next_id += 1

        return new_instructions

    def _crossover(self, text1: str, text2: str) -> str:
        """
        Single-point crossover of instruction clauses.

        Split both texts into clauses, take first half from one
        and second half from the other.
        """
        clauses1 = self._split_into_clauses(text1)
        clauses2 = self._split_into_clauses(text2)

        if len(clauses1) < 2 and len(clauses2) < 2:
            # Can't do meaningful crossover, just concatenate
            return f"{text1.rstrip('.')}. {clauses2[-1]}" if clauses2 else text1

        # Choose crossover point
        point1 = self.rng.randint(1, max(1, len(clauses1) - 1))
        point2 = self.rng.randint(0, max(0, len(clauses2) - 1))

        # Combine
        child_clauses = clauses1[:point1] + clauses2[point2:]

        return '. '.join(c.strip() for c in child_clauses if c.strip()) + '.'

    def _mutate(self, text: str) -> str:
        """
        Apply minor mutations to instruction text.

        Mutations:
        - Swap order of two clauses
        - Remove a clause
        - Duplicate a clause with slight variation
        """
        clauses = self._split_into_clauses(text)

        if len(clauses) < 2:
            return text

        mutation_type = self.rng.choice(['swap', 'remove', 'duplicate'])

        if mutation_type == 'swap' and len(clauses) >= 2:
            # Swap two adjacent clauses
            idx = self.rng.randint(0, len(clauses) - 2)
            clauses[idx], clauses[idx + 1] = clauses[idx + 1], clauses[idx]

        elif mutation_type == 'remove' and len(clauses) > 2:
            # Remove a random clause (keep at least 2)
            idx = self.rng.randint(0, len(clauses) - 1)
            clauses.pop(idx)

        elif mutation_type == 'duplicate' and len(clauses) >= 1:
            # Add emphasis word to a clause
            idx = self.rng.randint(0, len(clauses) - 1)
            emphasis = self.rng.choice(['Make sure to', 'Remember to', 'Always'])
            clauses[idx] = f"{emphasis} {clauses[idx].lower()}"

        return '. '.join(c.strip() for c in clauses if c.strip()) + '.'

    def _split_into_clauses(self, text: str) -> List[str]:
        """Split text into clauses by sentence boundaries"""
        # Split by period, semicolon, or explicit line breaks
        clauses = re.split(r'[.;]\s*|\n', text)
        return [c.strip() for c in clauses if c.strip()]

    def _clean_text(self, text: str) -> str:
        """Clean up generated text"""
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        # Ensure proper sentence ending
        text = text.strip()
        if text and not text.endswith('.'):
            text += '.'
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        return text
