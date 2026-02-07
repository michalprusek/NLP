"""AdaS-BO benchmark adapters.

Two variants:
- AdaSBOTRBenchmark: Geodesic TR + adaptive TR + ID estimation
- AdaSBOStagBenchmark: Stagnation-based restarts + ID estimation
"""

import torch

from rielbo.benchmark.base import BaseBenchmarkMethod, StepResult
from rielbo.subspace_bo_adas import AdaptiveSubspaceBO, AdaSConfig


class _AdaSBOBase(BaseBenchmarkMethod):
    """Shared base for AdaS-BO benchmark variants."""

    method_name = "adas"

    def __init__(
        self,
        codec,
        oracle,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = False,
        preset: str = "adas_tr",
        n_candidates: int = 2000,
        acqf: str = "ts",
    ):
        super().__init__(codec, oracle, seed, device, verbose)

        self.preset = preset
        self.n_candidates = n_candidates
        self.acqf = acqf

        self.adas_config = AdaSConfig.from_preset(preset)

        self.optimizer = AdaptiveSubspaceBO(
            codec=codec,
            oracle=oracle,
            input_dim=256,
            subspace_dim=16,  # placeholder, overridden by ID estimation
            config=self.adas_config,
            device=device,
            n_candidates=n_candidates,
            acqf=acqf,
            seed=seed,
            verbose=verbose,
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

        extra = {
            "nearest_train_cos": result.get("nearest_train_cos"),
            "embedding_norm": result.get("embedding_norm"),
            "current_dim": self.optimizer._current_dim,
            "restart_strategy": self.adas_config.restart_strategy,
        }

        return StepResult(
            score=result["score"],
            best_score=result["best_score"],
            smiles=result["smiles"],
            is_duplicate=result["is_duplicate"],
            is_valid=bool(result["smiles"]),
            gp_mean=result.get("gp_mean"),
            gp_std=result.get("gp_std"),
            trust_region_length=self.optimizer.tr_length,
            extra=extra,
        )

    def get_history(self):
        history = super().get_history()

        original_to_dict = history.to_dict
        opt_hist = self.optimizer.history

        def extended_to_dict():
            d = original_to_dict()
            d["d_target"] = opt_hist.get("d_target", [])
            d["n_restarts"] = opt_hist.get("n_restarts", [])
            d["subspace_dim"] = opt_hist.get("subspace_dim", [])
            d["id_diagnostics"] = self.optimizer._id_diagnostics
            return d

        history.to_dict = extended_to_dict
        return history

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "preset": self.preset,
            "n_candidates": self.n_candidates,
            "acqf": self.acqf,
            "adaptive_subspace": self.adas_config.adaptive_subspace,
            "restart_strategy": self.adas_config.restart_strategy,
            "d_min": self.adas_config.d_min,
            "d_max": self.adas_config.d_max,
        })
        return config


class AdaSBOTRBenchmark(_AdaSBOBase):
    """AdaS-BO with geodesic TR + ID estimation."""

    method_name = "adas_tr"

    def __init__(self, codec, oracle, seed=42, device="cuda", verbose=False,
                 n_candidates=2000, acqf="ts"):
        super().__init__(codec, oracle, seed, device, verbose,
                         preset="adas_tr", n_candidates=n_candidates, acqf=acqf)


class AdaSBOStagBenchmark(_AdaSBOBase):
    """AdaS-BO with stagnation-based restarts + ID estimation."""

    method_name = "adas_stag"

    def __init__(self, codec, oracle, seed=42, device="cuda", verbose=False,
                 n_candidates=2000, acqf="ts"):
        super().__init__(codec, oracle, seed, device, verbose,
                         preset="adas_stag", n_candidates=n_candidates, acqf=acqf)
