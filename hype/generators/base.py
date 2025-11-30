"""
Base Generator Protocol for HYPE

All generators must implement this interface.
"""

from typing import List, Dict, Protocol, runtime_checkable
from hype.data_types import (
    Instruction, Exemplar, EvaluationRecord,
    ComponentScore, GenerationResult
)


@runtime_checkable
class Generator(Protocol):
    """
    Protocol for component generation methods.

    Each generator takes current components, their scores, and evaluation
    history to produce new candidate components.
    """

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
        Generate new component candidates.

        Args:
            instructions: Current instruction pool
            exemplars: Current exemplar pool
            instruction_scores: S_I scores for instructions
            exemplar_scores: S_E scores for exemplars
            evaluation_records: Full evaluation history
            generation: Current evolution iteration number
            **kwargs: Method-specific parameters

        Returns:
            GenerationResult with new instructions and/or exemplars
        """
        ...

    @property
    def name(self) -> str:
        """Generator method name (for logging)"""
        ...

    @property
    def requires_llm(self) -> bool:
        """Whether this generator requires LLM calls"""
        ...
