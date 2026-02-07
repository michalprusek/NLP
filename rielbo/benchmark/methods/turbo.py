"""TuRBO wrapper for benchmark framework.

Wraps TuRBOBaseline to the BaseBenchmarkMethod interface.
"""

import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult
from rielbo.turbo_baseline import TuRBOBaseline


class TuRBOBenchmark(BaseBenchmarkMethod):
    """TuRBO benchmark wrapper.

    Trust Region Bayesian Optimization in full R^256 latent space.

    Key characteristics:
    - GP with Matern kernel in full 256D
    - Adaptive trust region (expands on success, shrinks on failure)
    - Expected Improvement acquisition
    - Standard BO baseline for high-dimensional spaces
    """

    method_name = "turbo"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        # Method-specific parameters
        n_candidates: int = 2000,
        trust_region: float = 0.8,
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.n_candidates = n_candidates
        self.trust_region = trust_region

        self.optimizer = TuRBOBaseline(
            codec=codec,
            oracle=oracle,
            input_dim=256,  # SELFIES VAE latent dim
            device=device,
            n_candidates=n_candidates,
            trust_region=trust_region,
            seed=seed,
            verbose=verbose,
        )

        self._iteration = 0

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data."""
        scores_list = scores.tolist()  # TuRBO expects list, not tensor
        self.optimizer.cold_start(smiles_list, scores_list)

        self.best_score = self.optimizer.best_score
        self.best_smiles = self.optimizer.best_smiles
        self.n_evaluated = len(self.optimizer.smiles_observed)
        self.smiles_set = self.optimizer.smiles_observed.copy()

    def step(self) -> StepResult:
        """Execute a single optimization step."""
        self._iteration += 1
        score, smiles = self.optimizer._step(iteration=self._iteration)

        if self.optimizer.turbo_state.restart_triggered:
            from rielbo.turbo_baseline import TurboState
            self.optimizer.turbo_state = TurboState(
                dim=self.optimizer.input_dim,
                length=self.optimizer.trust_region,
                failure_tolerance=max(5, self.optimizer.input_dim // 20),
            )
            self.optimizer.turbo_state.best_value = self.optimizer.best_score

        self.best_score = self.optimizer.best_score
        self.best_smiles = self.optimizer.best_smiles

        is_duplicate = score is None

        if not is_duplicate:
            self.n_evaluated = len(self.optimizer.smiles_observed)
            self.smiles_set = self.optimizer.smiles_observed.copy()

        return StepResult(
            score=score if score is not None else 0.0,
            best_score=self.optimizer.best_score,
            smiles=smiles,
            is_duplicate=is_duplicate,
            is_valid=not is_duplicate,
            trust_region_length=self.optimizer.turbo_state.length,
        )

    def get_config(self) -> dict:
        """Return method-specific configuration."""
        config = super().get_config()
        config.update({
            "n_candidates": self.n_candidates,
            "trust_region": self.trust_region,
            "input_dim": 256,
        })
        return config
