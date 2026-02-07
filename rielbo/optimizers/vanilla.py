"""VanillaBO: convenience wrapper around BaseOptimizer for Hvarfner baseline.

Full-dimensional GP with BoTorch default Hvarfner priors (RBF + LogNormal + ARD).
Uses [0,1]^D min-max normalization (z-score BREAKS Hvarfner priors).
"""

from __future__ import annotations

from rielbo.core.config import OptimizerConfig
from rielbo.core.optimizer import BaseOptimizer


class VanillaBO(BaseOptimizer):
    """Vanilla BO with Hvarfner priors — full-dimensional RBF + ARD.

    Usage:
        bo = VanillaBO(codec, oracle, input_dim=256, seed=42)
        bo.cold_start(smiles_list, scores)
        bo.optimize(n_iterations=500)
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        n_candidates: int = 2000,
        acqf: str = "ts",
        ucb_beta: float = 2.0,
        trust_region: float = 0.8,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = True,
    ):
        config = OptimizerConfig.from_preset("vanilla")
        config.seed = seed
        config.device = device
        config.verbose = verbose

        super().__init__(
            codec=codec,
            oracle=oracle,
            config=config,
            input_dim=input_dim,
            n_candidates=n_candidates,
            acqf=acqf,
            ucb_beta=ucb_beta,
            trust_region=trust_region,
        )
