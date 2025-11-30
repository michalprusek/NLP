"""
HYPE Generators - Methods for creating new prompt components

Method A: SemanticGradientGenerator - LLM analyzes errors and improves instructions
Method C: BootstrapGenerator - Generate synthetic exemplars from hard examples
"""

from hype.generators.base import Generator
from hype.generators.semantic_gradient import SemanticGradientGenerator
from hype.generators.bootstrap import BootstrapGenerator

__all__ = [
    'Generator',
    'SemanticGradientGenerator',
    'BootstrapGenerator',
]
