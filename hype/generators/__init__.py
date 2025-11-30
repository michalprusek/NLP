"""
HYPE Generators - Three methods for creating new prompt components

Method A: SemanticGradientGenerator - LLM analyzes errors and improves instructions
Method B: RecombinationGenerator - Genetic crossover of top components (zero LLM cost)
Method C: BootstrapGenerator - Generate synthetic exemplars from hard examples
"""

from hype.generators.base import Generator
from hype.generators.recombination import RecombinationGenerator
from hype.generators.semantic_gradient import SemanticGradientGenerator
from hype.generators.bootstrap import BootstrapGenerator

__all__ = [
    'Generator',
    'RecombinationGenerator',
    'SemanticGradientGenerator',
    'BootstrapGenerator',
]
