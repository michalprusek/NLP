"""Data loading utilities for GuacaMol molecular optimization."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from shared.guacamol.constants import (
    DEFAULT_DATA_PATH,
    GUACAMOL_TASKS,
    CSV_COLUMNS,
)

logger = logging.getLogger(__name__)


def load_guacamol_data(
    csv_path: Optional[str] = None,
    n_samples: Optional[int] = None,
    task_id: str = "pdop",
    return_selfies: bool = False,
) -> tuple[list[str], torch.Tensor, Optional[list[str]]]:
    """Load molecules and scores from GuacaMol CSV file.

    Args:
        csv_path: Path to CSV file. Defaults to nfbo_original/data/guacamol/oracle/guacamol_train_data_first_20k.csv
        n_samples: Number of samples to load. None for all.
        task_id: Task to load scores for (column name in CSV)
        return_selfies: Whether to also return SELFIES strings

    Returns:
        Tuple of:
            - smiles_list: List of SMILES strings
            - scores: Tensor of task scores [N]
            - selfies_list: List of SELFIES strings (if return_selfies=True, else None)

    Example:
        >>> smiles, scores, _ = load_guacamol_data(n_samples=1000, task_id="pdop")
        >>> print(f"Loaded {len(smiles)} molecules, max score: {scores.max():.4f}")
    """
    if csv_path is None:
        csv_path = DEFAULT_DATA_PATH

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GuacaMol data file not found: {csv_path}")

    logger.info(f"Loading GuacaMol data from {csv_path}")

    df = pd.read_csv(csv_path)

    if n_samples is not None:
        df = df.head(n_samples)

    # Extract SMILES
    smiles_col = CSV_COLUMNS["smiles"]
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in CSV")
    smiles_list = df[smiles_col].tolist()

    # Extract scores
    if task_id not in df.columns:
        available = [c for c in df.columns if c not in [smiles_col, CSV_COLUMNS["selfies"]]]
        raise ValueError(
            f"Task '{task_id}' not found in CSV. Available tasks: {available}"
        )
    scores = torch.tensor(df[task_id].values, dtype=torch.float32)

    # Extract SELFIES if requested
    selfies_list = None
    if return_selfies:
        selfies_col = CSV_COLUMNS["selfies"]
        if selfies_col in df.columns:
            selfies_list = df[selfies_col].tolist()
        else:
            logger.warning(f"SELFIES column '{selfies_col}' not found, returning None")

    logger.info(
        f"Loaded {len(smiles_list)} molecules for task '{task_id}', "
        f"score range: [{scores.min():.4f}, {scores.max():.4f}]"
    )

    return smiles_list, scores, selfies_list


def load_top_k_molecules(
    csv_path: Optional[str] = None,
    task_id: str = "pdop",
    k: int = 100,
) -> tuple[list[str], torch.Tensor]:
    """Load top-k molecules by score for a task.

    Args:
        csv_path: Path to CSV file
        task_id: Task to sort by
        k: Number of top molecules to return

    Returns:
        Tuple of (smiles_list, scores) for top-k molecules
    """
    smiles_list, scores, _ = load_guacamol_data(
        csv_path=csv_path, task_id=task_id
    )

    # Sort by score descending
    indices = scores.argsort(descending=True)[:k]
    top_smiles = [smiles_list[i] for i in indices]
    top_scores = scores[indices]

    logger.info(
        f"Top-{k} molecules for '{task_id}': "
        f"score range [{top_scores[-1]:.4f}, {top_scores[0]:.4f}]"
    )

    return top_smiles, top_scores


def load_zinc_smiles(
    path: str = "datasets/zinc/zinc_all.txt",
    n_samples: Optional[int] = None,
) -> list[str]:
    """Load SMILES from ZINC dataset.

    Args:
        path: Path to ZINC file (one SMILES per line)
        n_samples: Number of samples to load. None for all.

    Returns:
        List of SMILES strings
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ZINC data file not found: {path}")

    logger.info(f"Loading ZINC data from {path}")

    with open(path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    if n_samples is not None and n_samples < len(smiles_list):
        smiles_list = smiles_list[:n_samples]

    logger.info(f"Loaded {len(smiles_list)} SMILES from ZINC")
    return smiles_list


class GuacaMolDataset(Dataset):
    """PyTorch Dataset for GuacaMol molecules.

    Provides SMILES strings and scores for use in training.

    Example:
        >>> dataset = GuacaMolDataset(task_id="pdop", n_samples=10000)
        >>> smiles, score = dataset[0]
        >>> print(f"SMILES: {smiles}, Score: {score:.4f}")
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        task_id: str = "pdop",
        n_samples: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            csv_path: Path to CSV file
            task_id: Task to load scores for
            n_samples: Number of samples to load
        """
        self.smiles_list, self.scores, _ = load_guacamol_data(
            csv_path=csv_path, task_id=task_id, n_samples=n_samples
        )
        self.task_id = task_id

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.smiles_list[idx], self.scores[idx].item()

    def get_all_smiles(self) -> list[str]:
        """Get all SMILES strings."""
        return self.smiles_list

    def get_all_scores(self) -> torch.Tensor:
        """Get all scores."""
        return self.scores


class GuacaMolEmbeddingDataset(Dataset):
    """PyTorch Dataset for pre-encoded molecular embeddings.

    Used for training flow models on MegaMolBART embeddings.

    Example:
        >>> dataset = GuacaMolEmbeddingDataset(embeddings, smiles, scores)
        >>> embedding, score = dataset[0]
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        smiles_list: Optional[list[str]] = None,
        scores: Optional[torch.Tensor] = None,
    ):
        """Initialize dataset.

        Args:
            embeddings: Pre-computed embeddings [N, D]
            smiles_list: Original SMILES strings (optional)
            scores: Task scores (optional)
        """
        self.embeddings = embeddings
        self.smiles_list = smiles_list
        self.scores = scores

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get embedding for training (no score needed for unsupervised flow)."""
        return self.embeddings[idx]

    def get_embedding(self, idx: int) -> torch.Tensor:
        """Get single embedding."""
        return self.embeddings[idx]

    def get_smiles(self, idx: int) -> Optional[str]:
        """Get SMILES string for an index."""
        if self.smiles_list is None:
            return None
        return self.smiles_list[idx]

    def get_score(self, idx: int) -> Optional[float]:
        """Get score for an index."""
        if self.scores is None:
            return None
        return self.scores[idx].item()
