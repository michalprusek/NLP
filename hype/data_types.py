"""
Core data types for HYPE

Instruction and Exemplar are the atomic components of prompts.
EvaluationRecord tracks performance across Hyperband runs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ComponentSource(Enum):
    """Origin of a component"""
    INITIAL = "initial"              # From original pool
    SEMANTIC_GRADIENT = "semantic_gradient"  # Method A: LLM-improved
    BOOTSTRAP = "bootstrap"          # Method C: Synthetic


@dataclass
class Instruction:
    """An instruction component for prompts"""
    id: int
    text: str
    source: ComponentSource = ComponentSource.INITIAL
    generation: int = 0  # Which evolution iteration created this
    parent_ids: List[int] = field(default_factory=list)  # For tracking lineage

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if isinstance(other, Instruction):
            return self.text == other.text
        return False


@dataclass
class Exemplar:
    """An exemplar (few-shot examples) component for prompts"""
    id: int
    text: str
    source: ComponentSource = ComponentSource.INITIAL
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if isinstance(other, Exemplar):
            return self.text == other.text
        return False


@dataclass
class EvaluationRecord:
    """Record of a single prompt evaluation"""
    instruction_id: int
    exemplar_id: int
    budget: int  # Fidelity level (number of validation instances)
    error_rate: float  # Fraction of incorrect answers
    bracket: int = 0  # Hyperband bracket
    round: int = 0    # Round within bracket
    generation: int = 0  # Evolution iteration
    timestamp: Optional[str] = None

    # Optional: track specific failures for error analysis
    failed_indices: List[int] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Accuracy = 1 - error_rate"""
        return 1.0 - self.error_rate


@dataclass
class ComponentScore:
    """Score for a single component (instruction or exemplar)"""
    component_id: int
    score: float
    variance: float = 0.0
    num_evaluations: int = 0
    max_budget_seen: int = 0  # Highest fidelity level reached


@dataclass
class GenerationResult:
    """Result of a generation method"""
    new_instructions: List[Instruction] = field(default_factory=list)
    new_exemplars: List[Exemplar] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
