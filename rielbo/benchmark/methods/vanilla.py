"""Vanilla BO (Hvarfner) wrapper for benchmark framework.

Wraps VanillaBO to the BaseBenchmarkMethod interface.
"""

import math

import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult
from rielbo.vanilla_bo import VanillaBO


class VanillaBOBenchmark(BaseBenchmarkMethod):
    """Vanilla BO with BoTorch's Hvarfner dimension-scaled priors.

    GP in full 256D with RBF + LogNormal(√2 + log(D)/2, √3) + ARD.
    No subspace projection — relies on the prior to handle high-D.
    Uses [0,1]^D min-max normalization.
    """

    method_name = "vanilla"

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
        acqf: str = "ts",
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.n_candidates = n_candidates
        self.trust_region = trust_region
        self.acqf = acqf

        self.optimizer = VanillaBO(
            codec=codec,
            oracle=oracle,
            input_dim=256,
            device=device,
            n_candidates=n_candidates,
            acqf=acqf,
            trust_region=trust_region,
            seed=seed,
            verbose=verbose,
        )

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data."""
        self.optimizer.cold_start(smiles_list, scores)

        self.best_score = self.optimizer.best_score
        self.best_smiles = self.optimizer.best_smiles
        self.n_evaluated = len(self.optimizer.smiles_observed)
        self.smiles_set = set(self.optimizer.smiles_observed)

    def step(self) -> StepResult:
        """Execute a single optimization step."""
        result = self.optimizer.step()

        self.best_score = self.optimizer.best_score
        self.best_smiles = self.optimizer.best_smiles
        self.n_evaluated = len(self.optimizer.smiles_observed)
        self.smiles_set = set(self.optimizer.smiles_observed)

        return StepResult(
            score=result["score"],
            best_score=result["best_score"],
            smiles=result["smiles"],
            is_duplicate=result["is_duplicate"],
            is_valid=not result["is_duplicate"],
            gp_mean=result.get("gp_mean"),
            gp_std=result.get("gp_std"),
            trust_region_length=result.get("tr_length"),
        )

    def get_config(self) -> dict:
        """Return method-specific configuration."""
        config = super().get_config()
        ls_mu = math.sqrt(2) + math.log(256) / 2
        config.update({
            "n_candidates": self.n_candidates,
            "trust_region": self.trust_region,
            "acqf": self.acqf,
            "input_dim": 256,
            "kernel": "RBF (BoTorch default)",
            "ls_prior": f"LogNormal({ls_mu:.2f}, {math.sqrt(3):.2f})",
        })
        return config
