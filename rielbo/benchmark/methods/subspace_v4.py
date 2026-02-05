"""Subspace BO v4 wrapper for benchmark framework.

Wraps SphericalSubspaceBOv4 to the BaseBenchmarkMethod interface.
v4 = v3 + geodesic novelty bonus in acquisition function.
"""

import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult
from rielbo.subspace_bo_v4 import SphericalSubspaceBOv4


class SubspaceBOv4Benchmark(BaseBenchmarkMethod):
    """Subspace BO v4 benchmark wrapper.

    Key improvements over v3:
    - Geodesic novelty bonus in acquisition (reduces duplicates)
    - Global novelty against ALL observed points
    """

    method_name = "subspace_v4"

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
        n_projections: int = 3,
        window_local: int = 50,
        window_random: int = 30,
        novelty_weight: float = 0.1,
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.subspace_dim = subspace_dim
        self.n_candidates = n_candidates
        self.acqf = acqf
        self.trust_region = trust_region
        self.kernel = kernel
        self.n_projections = n_projections
        self.window_local = window_local
        self.window_random = window_random
        self.novelty_weight = novelty_weight

        self.optimizer = SphericalSubspaceBOv4(
            codec=codec,
            oracle=oracle,
            input_dim=256,
            subspace_dim=subspace_dim,
            device=device,
            n_candidates=n_candidates,
            acqf=acqf,
            trust_region=trust_region,
            seed=seed,
            verbose=verbose,
            kernel=kernel,
            n_projections=n_projections,
            window_local=window_local,
            window_random=window_random,
            novelty_weight=novelty_weight,
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
                "projection_idx": result.get("projection_idx"),
                "window_size": result.get("window_size"),
                "novelty_mean": result.get("novelty_mean"),
                "novelty_selected": result.get("novelty_selected"),
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
            "n_projections": self.n_projections,
            "window_local": self.window_local,
            "window_random": self.window_random,
            "novelty_weight": self.novelty_weight,
        })
        return config
