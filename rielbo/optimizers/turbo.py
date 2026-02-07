"""TuRBO: convenience wrapper around BaseOptimizer for TuRBO baseline.

Full-dimensional GP with Matern-5/2, z-score normalization, TuRBO-style TR.
"""

from __future__ import annotations

from rielbo.core.config import OptimizerConfig
from rielbo.core.optimizer import BaseOptimizer


class TuRBO(BaseOptimizer):
    """TuRBO baseline — full-dimensional GP with Matern-5/2 + ARD.

    Usage:
        bo = TuRBO(codec, oracle, input_dim=256, seed=42)
        bo.cold_start(smiles_list, scores)
        bo.optimize(n_iterations=500)
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        n_candidates: int = 2000,
        acqf: str = "ei",
        ucb_beta: float = 2.0,
        trust_region: float = 0.8,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = True,
    ):
        config = OptimizerConfig.from_preset("turbo")
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
