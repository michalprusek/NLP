"""Base class for benchmark methods.

Defines the common interface that all BO methods must implement
for fair comparison in the benchmark framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class StepResult:
    """Result from a single optimization step."""

    score: float  # Score of the evaluated molecule
    best_score: float  # Best score seen so far
    smiles: str  # Generated SMILES string
    is_duplicate: bool = False  # Whether molecule was already evaluated
    is_valid: bool = True  # Whether molecule is valid

    # Optional diagnostics
    gp_mean: Optional[float] = None
    gp_std: Optional[float] = None
    trust_region_length: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkHistory:
    """Stores optimization history for plotting."""

    iteration: list[int] = field(default_factory=list)
    best_score: list[float] = field(default_factory=list)
    current_score: list[float] = field(default_factory=list)
    n_evaluated: list[int] = field(default_factory=list)

    # Optional diagnostics (may not be available for all methods)
    gp_mean: list[float] = field(default_factory=list)
    gp_std: list[float] = field(default_factory=list)
    trust_region_length: list[float] = field(default_factory=list)

    def append(self, result: StepResult, n_eval: int):
        """Append a step result to history."""
        self.iteration.append(len(self.iteration))
        self.best_score.append(result.best_score)
        self.current_score.append(result.score)
        self.n_evaluated.append(n_eval)

        if result.gp_mean is not None:
            self.gp_mean.append(result.gp_mean)
        if result.gp_std is not None:
            self.gp_std.append(result.gp_std)
        if result.trust_region_length is not None:
            self.trust_region_length.append(result.trust_region_length)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "iteration": self.iteration,
            "best_score": self.best_score,
            "current_score": self.current_score,
            "n_evaluated": self.n_evaluated,
            "gp_mean": self.gp_mean if self.gp_mean else None,
            "gp_std": self.gp_std if self.gp_std else None,
            "trust_region_length": self.trust_region_length if self.trust_region_length else None,
        }


class BaseBenchmarkMethod(ABC):
    """Abstract base class for BO methods in the benchmark.

    All methods must implement:
    - cold_start(): Initialize with labeled data
    - step(): Single optimization iteration
    - get_history(): Return optimization history

    Standard interface allows fair comparison of:
    - Sample efficiency (best score vs iterations)
    - Convergence behavior
    - Duplicate handling
    """

    # Class-level method identifier
    method_name: str = "base"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """Initialize benchmark method.

        Args:
            codec: SELFIESVAECodec for encoding/decoding molecules
            oracle: GuacaMolOracle for scoring molecules
            seed: Random seed for reproducibility
            device: Device for computation ("cuda" or "cpu")
            verbose: Whether to print progress logs
        """
        self.codec = codec
        self.oracle = oracle
        self.seed = seed
        self.device = device
        self.verbose = verbose

        # Track state
        self.best_score: float = float("-inf")
        self.best_smiles: str = ""
        self.n_evaluated: int = 0
        self.smiles_set: set[str] = set()

        # History for plotting
        self.history = BenchmarkHistory()

        # Seed RNG
        torch.manual_seed(seed)

    @abstractmethod
    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data.

        Args:
            smiles_list: List of SMILES strings (pre-evaluated)
            scores: Tensor of corresponding scores
        """
        pass

    @abstractmethod
    def step(self) -> StepResult:
        """Execute a single optimization step.

        Returns:
            StepResult with score, best_score, smiles, and diagnostics
        """
        pass

    def get_history(self) -> BenchmarkHistory:
        """Return optimization history."""
        return self.history

    def get_config(self) -> dict:
        """Return method-specific configuration for logging.

        Override in subclasses to add method-specific parameters.
        """
        return {
            "method": self.method_name,
            "seed": self.seed,
            "device": self.device,
        }

    def optimize(self, n_iterations: int, log_interval: int = 50) -> None:
        """Run optimization loop.

        This is the main entry point for running the benchmark.

        Args:
            n_iterations: Number of optimization steps
            log_interval: How often to log progress
        """
        from tqdm import tqdm

        pbar = tqdm(range(n_iterations), desc=f"{self.method_name}", disable=not self.verbose)
        n_duplicates = 0

        for i in pbar:
            result = self.step()
            self.history.append(result, self.n_evaluated)

            if result.is_duplicate:
                n_duplicates += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result.score:.3f}",
                "dup": n_duplicates,
            })

            if self.verbose and (i + 1) % log_interval == 0:
                print(
                    f"[{self.method_name}] Iter {i+1}/{n_iterations} | "
                    f"Best: {self.best_score:.4f} | Curr: {result.score:.4f} | "
                    f"Eval: {self.n_evaluated} | Dup: {n_duplicates}"
                )
