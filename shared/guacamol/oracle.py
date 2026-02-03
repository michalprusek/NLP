"""GuacaMol oracle for molecular scoring.

This module provides a clean interface to score molecules on GuacaMol benchmark tasks.
It wraps the guacamol library's standard benchmarks.
"""

import logging
import math
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem.QED import qed

# Patch scipy for guacamol compatibility (scipy.histogram was removed in newer versions)
import scipy
if not hasattr(scipy, 'histogram'):
    scipy.histogram = np.histogram

from shared.guacamol.constants import GUACAMOL_TASKS, EXTRA_TASKS

logger = logging.getLogger(__name__)

# Lazy-loaded GuacaMol benchmark objects
_BENCHMARK_CACHE: dict = {}


def _get_benchmark(task_id: str):
    """Get or create GuacaMol benchmark object (lazy loading)."""
    if task_id in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE[task_id]

    from guacamol import standard_benchmarks

    benchmarks = {
        "med1": standard_benchmarks.median_camphor_menthol,
        "med2": standard_benchmarks.median_tadalafil_sildenafil,
        "pdop": standard_benchmarks.perindopril_rings,
        "osmb": standard_benchmarks.hard_osimertinib,
        "adip": standard_benchmarks.amlodipine_rings,
        "siga": standard_benchmarks.sitagliptin_replacement,
        "zale": standard_benchmarks.zaleplon_with_other_formula,
        "valt": standard_benchmarks.valsartan_smarts,
        "dhop": standard_benchmarks.decoration_hop,
        "shop": standard_benchmarks.scaffold_hop,
        "rano": standard_benchmarks.ranolazine_mpo,
        "fexo": standard_benchmarks.hard_fexofenadine,
    }

    if task_id not in benchmarks:
        raise ValueError(
            f"Unknown GuacaMol task: {task_id}. "
            f"Available: {list(benchmarks.keys())} + {EXTRA_TASKS}"
        )

    _BENCHMARK_CACHE[task_id] = benchmarks[task_id]()
    return _BENCHMARK_CACHE[task_id]


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string represents a valid molecule."""
    if smiles is None or len(smiles) == 0:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def score_smiles(smiles: str, task_id: str) -> float:
    """Score a single SMILES string on a GuacaMol task.

    Args:
        smiles: SMILES string to score
        task_id: GuacaMol task identifier (e.g., "pdop", "siga")

    Returns:
        Score in [0, 1] for GuacaMol tasks, or task-specific range for logp/qed.
        Returns 0.0 for invalid molecules.
    """
    if smiles is None or len(smiles) == 0:
        return 0.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    try:
        if task_id == "logp":
            return _penalized_logp(mol)
        elif task_id == "qed":
            return qed(mol)
        elif task_id in GUACAMOL_TASKS:
            benchmark = _get_benchmark(task_id)
            score = benchmark.objective.score(smiles)
            if score is None or score < 0:
                return 0.0
            return score
        else:
            raise ValueError(f"Unknown task: {task_id}")
    except Exception as e:
        logger.info(f"Scoring failed for '{smiles}' on task '{task_id}': {e}")
        return 0.0


def _penalized_logp(mol) -> float:
    """Calculate penalized logP (from NFBO implementation).

    This includes SA (synthetic accessibility) and ring penalties.
    """
    from nfbo_original.objective.guacamol.utils.mol_utils.moses_metrics.SA_Score import (
        sascorer,
    )

    try:
        logp = Crippen.MolLogP(mol)
        sa = sascorer.calculateScore(mol)
        cycle_length = _cycle_score(mol)

        # Normalize using empirical means and stds
        score = (
            (logp - 2.45777691) / 1.43341767
            + (-sa + 3.05352042) / 0.83460587
            + (-cycle_length - -0.04861121) / 0.28746695
        )
        return max(score, -float("inf"))
    except Exception as e:
        logger.debug(f"Penalized logP calculation failed: {e}")
        return 0.0


def _cycle_score(mol) -> int:
    """Calculate cycle penalty for penalized logP."""
    import networkx as nx
    from rdkit.Chem import rdmolops

    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def smiles_to_scores(smiles_list: list[str], task_id: str) -> np.ndarray:
    """Score a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        task_id: GuacaMol task identifier

    Returns:
        Array of scores [N]
    """
    scores = []
    for smiles in smiles_list:
        score = score_smiles(smiles, task_id)
        if score is not None and math.isfinite(score):
            scores.append(score)
        else:
            scores.append(0.0)
    return np.array(scores)


class GuacaMolOracle:
    """Oracle for GuacaMol benchmark tasks.

    Provides a clean interface for scoring molecules, with caching and statistics.

    Example:
        >>> oracle = GuacaMolOracle(task_id="pdop")
        >>> score = oracle.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
        >>> print(f"Score: {score:.4f}")
    """

    def __init__(self, task_id: str = "pdop"):
        """Initialize oracle for a specific task.

        Args:
            task_id: GuacaMol task identifier. One of:
                - "med1", "med2": Median molecules tasks
                - "pdop", "adip", "osmb", "siga", "zale", "rano", "fexo": MPO tasks
                - "valt": Valsartan SMARTS task
                - "dhop", "shop": Scaffold/decoration hop tasks
                - "logp", "qed": Additional tasks
        """
        if task_id not in GUACAMOL_TASKS and task_id not in EXTRA_TASKS:
            raise ValueError(
                f"Unknown task: {task_id}. "
                f"Available: {GUACAMOL_TASKS + EXTRA_TASKS}"
            )

        self.task_id = task_id
        self.num_calls = 0
        self._cache: dict[str, float] = {}

        # Pre-load benchmark to catch errors early
        if task_id in GUACAMOL_TASKS:
            _get_benchmark(task_id)

        logger.info(f"Initialized GuacaMolOracle for task '{task_id}'")

    def score(self, smiles: str, use_cache: bool = True) -> float:
        """Score a single SMILES string.

        Args:
            smiles: SMILES string to score
            use_cache: Whether to use cached scores

        Returns:
            Score value (0.0 for invalid molecules)
        """
        if use_cache and smiles in self._cache:
            return self._cache[smiles]

        score = score_smiles(smiles, self.task_id)
        self.num_calls += 1

        if use_cache:
            self._cache[smiles] = score

        return score

    def score_batch(
        self, smiles_list: list[str], use_cache: bool = True
    ) -> np.ndarray:
        """Score a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            use_cache: Whether to use cached scores

        Returns:
            Array of scores [N]
        """
        scores = []
        for smiles in smiles_list:
            scores.append(self.score(smiles, use_cache=use_cache))
        return np.array(scores)

    def reset_cache(self) -> None:
        """Clear the score cache."""
        self._cache.clear()

    def reset_stats(self) -> None:
        """Reset call counter."""
        self.num_calls = 0

    @property
    def cache_size(self) -> int:
        """Number of cached scores."""
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"GuacaMolOracle(task_id='{self.task_id}', "
            f"num_calls={self.num_calls}, cache_size={self.cache_size})"
        )
