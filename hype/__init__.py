"""
HYPE - Hyperband Prompt Evolution

Evolutionary extension to HbBoPs that adds:
1. Component Scoring (S_I, S_E)
2. Three generation methods (Semantic Gradient, Recombination, Bootstrap)
3. Evolutionary loop for iterative prompt improvement
"""

from hype.data_types import Instruction, Exemplar, EvaluationRecord
from hype.scoring import ComponentScorer
from hype.evolution import HYPE, HYPEConfig

__all__ = [
    'Instruction',
    'Exemplar',
    'EvaluationRecord',
    'ComponentScorer',
    'HYPE',
    'HYPEConfig',
]
