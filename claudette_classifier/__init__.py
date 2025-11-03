"""Claudette binary classifier package."""

from .config import Config
from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP
from .train import train
from .evaluate import evaluate, evaluate_detailed

__all__ = [
    'Config',
    'LegalBERTEncoder',
    'DeepResidualMLP',
    'train',
    'evaluate',
    'evaluate_detailed'
]
