"""Shared GuacaMol molecular optimization module.

This module provides utilities for molecular optimization on GuacaMol benchmark tasks,
designed to be used by RieLBO, NFBO, and other optimization methods.

Key components:
- MolecularCodec: Encode SMILES to embeddings and decode back
- GuacaMolOracle: Score molecules on GuacaMol tasks
- Data loaders for GuacaMol CSV files
"""

from shared.guacamol.constants import (
    GUACAMOL_TASKS,
    GUACAMOL_TASK_DESCRIPTIONS,
    DEFAULT_DATA_PATH,
)
from shared.guacamol.oracle import GuacaMolOracle, smiles_to_scores
from shared.guacamol.data import load_guacamol_data, GuacaMolDataset

__all__ = [
    # Constants
    "GUACAMOL_TASKS",
    "GUACAMOL_TASK_DESCRIPTIONS",
    "DEFAULT_DATA_PATH",
    # Oracle
    "GuacaMolOracle",
    "smiles_to_scores",
    # Data
    "load_guacamol_data",
    "GuacaMolDataset",
]
