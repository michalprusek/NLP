"""Subspace BO v5 wrapper for benchmark framework.

v5 = geodesic TR + windowed GP + whitening + adaptive TR + subspace restart.
"""

import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult
from rielbo.subspace_bo_v5 import SphericalSubspaceBOv5


class SubspaceBOv5Benchmark(BaseBenchmarkMethod):
    """Subspace BO v5 benchmark wrapper."""

    method_name = "subspace_v5"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        subspace_dim: int = 16,
        n_candidates: int = 2000,
        acqf: str = "ts",
        kernel: str = "arccosine",
        window_local: int = 50,
        window_random: int = 30,
        geodesic_max_angle: float = 0.5,
        geodesic_global_fraction: float = 0.2,
        tr_init: float = 0.4,
        tr_min: float = 0.02,
        tr_max: float = 0.8,
        tr_success_tol: int = 3,
        tr_fail_tol: int = 10,
        max_restarts: int = 5,
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.subspace_dim = subspace_dim
        self.n_candidates = n_candidates
        self.acqf = acqf
        self.kernel = kernel
        self.window_local = window_local
        self.window_random = window_random
        self.geodesic_max_angle = geodesic_max_angle
        self.tr_init = tr_init

        self.optimizer = SphericalSubspaceBOv5(
            codec=codec,
            oracle=oracle,
            input_dim=256,
            subspace_dim=subspace_dim,
            device=device,
            n_candidates=n_candidates,
            acqf=acqf,
            seed=seed,
            verbose=verbose,
            kernel=kernel,
            window_local=window_local,
            window_random=window_random,
            geodesic_max_angle=geodesic_max_angle,
            geodesic_global_fraction=geodesic_global_fraction,
            tr_init=tr_init,
            tr_min=tr_min,
            tr_max=tr_max,
            tr_success_tol=tr_success_tol,
            tr_fail_tol=tr_fail_tol,
            max_restarts=max_restarts,
        )

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor) -> None:
        self.optimizer.cold_start(smiles_list, scores)
        self.best_score = self.optimizer.best_score
        self.best_smiles = self.optimizer.best_smiles
        self.n_evaluated = len(smiles_list)
        self.smiles_set = set(smiles_list)

    def step(self) -> StepResult:
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
                "tr_length": result.get("tr_length"),
                "n_restarts": result.get("n_restarts"),
            },
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "subspace_dim": self.subspace_dim,
            "n_candidates": self.n_candidates,
            "acqf": self.acqf,
            "kernel": self.kernel,
            "window_local": self.window_local,
            "window_random": self.window_random,
            "geodesic_max_angle": self.geodesic_max_angle,
            "tr_init": self.tr_init,
        })
        return config
