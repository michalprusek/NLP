"""Subspace BO wrapper for benchmark framework.

Wraps SphericalSubspaceBO to the BaseBenchmarkMethod interface.
"""

import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult
from rielbo.subspace_bo import SphericalSubspaceBO


class SubspaceBOBenchmark(BaseBenchmarkMethod):
    """Subspace BO benchmark wrapper.

    Projects S^255 → S^15 for tractable GP with ArcCosine kernel.

    Key characteristics:
    - Orthonormal random projection preserves angular structure
    - ArcCosine kernel for Riemannian geometry
    - Thompson Sampling acquisition (default)
    - Trust region around best point
    """

    method_name = "subspace"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        # Method-specific parameters
        subspace_dim: int = 16,
        n_candidates: int = 2000,
        acqf: str = "ts",
        trust_region: float = 0.8,
        kernel: str = "arccosine",
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.subspace_dim = subspace_dim
        self.n_candidates = n_candidates
        self.acqf = acqf
        self.trust_region = trust_region
        self.kernel = kernel

        self.optimizer = SphericalSubspaceBO(
            codec=codec,
            oracle=oracle,
            input_dim=256,  # SELFIES VAE latent dim
            subspace_dim=subspace_dim,
            device=device,
            n_candidates=n_candidates,
            acqf=acqf,
            trust_region=trust_region,
            seed=seed,
            verbose=verbose,
            kernel=kernel,
        )

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        """Initialize with cold start data."""
        self.optimizer.cold_start(smiles_list, scores)

        self.best_score = self.optimizer.best_score
        self.best_smiles = self.optimizer.best_smiles
        self.n_evaluated = len(smiles_list)
        self.smiles_set = set(smiles_list)

    def step(self) -> StepResult:
        """Execute a single optimization step."""
        result = self.optimizer.step()

        self.best_score = result["best_score"]
        self.best_smiles = self.optimizer.best_smiles
        if not result["is_duplicate"] and result["smiles"]:
            self.n_evaluated += 1
            self.smiles_set.add(result["smiles"])

        return StepResult(
            score=result["score"],
            best_score=result["best_score"],
            smiles=result["smiles"],
            is_duplicate=result["is_duplicate"],
            is_valid=bool(result["smiles"]),
            gp_mean=result.get("gp_mean"),
            gp_std=result.get("gp_std"),
            extra={
                "nearest_train_cos": result.get("nearest_train_cos"),
                "embedding_norm": result.get("embedding_norm"),
            },
        )

    def get_config(self) -> dict:
        """Return method-specific configuration."""
        config = super().get_config()
        config.update({
            "subspace_dim": self.subspace_dim,
            "n_candidates": self.n_candidates,
            "acqf": self.acqf,
            "trust_region": self.trust_region,
            "kernel": self.kernel,
        })
        return config
