"""SubspaceBO: convenience wrapper around BaseOptimizer for subspace presets.

Supports all V2 presets: baseline, geodesic, explore, lass_ur, etc.
This is a thin factory — all logic lives in BaseOptimizer + components.
"""

from __future__ import annotations

from rielbo.core.config import OptimizerConfig
from rielbo.core.optimizer import BaseOptimizer


class SubspaceBO(BaseOptimizer):
    """Subspace Bayesian Optimization on S^(D-1) → S^(d-1).

    Usage:
        bo = SubspaceBO(codec, oracle, preset="explore")
        bo.cold_start(smiles_list, scores)
        bo.optimize(n_iterations=500)

    Available presets: baseline, geodesic, ur_tr, lass, lass_ur, explore,
    order2, whitening, geometric, full, portfolio.
    """

    def __init__(
        self,
        codec,
        oracle,
        preset: str = "geodesic",
        input_dim: int = 256,
        subspace_dim: int = 16,
        n_candidates: int = 2000,
        acqf: str = "ts",
        ucb_beta: float = 2.0,
        trust_region: float = 0.8,
        seed: int = 42,
        device: str = "cuda",
        verbose: bool = True,
        **config_overrides,
    ):
        config = OptimizerConfig.from_preset(preset)
        config.seed = seed
        config.device = device
        config.verbose = verbose

        # Apply any extra overrides to sub-configs
        for key, value in config_overrides.items():
            _apply_nested_override(config, key, value)

        super().__init__(
            codec=codec,
            oracle=oracle,
            config=config,
            input_dim=input_dim,
            subspace_dim=subspace_dim,
            n_candidates=n_candidates,
            acqf=acqf,
            ucb_beta=ucb_beta,
            trust_region=trust_region,
        )


def _apply_nested_override(config: OptimizerConfig, key: str, value) -> None:
    """Apply an override to the appropriate nested config."""
    # Map common CLI-style keys to nested config fields
    nested_maps = {
        # Trust region
        "ur_std_low": ("trust_region", "ur_std_low"),
        "ur_std_high": ("trust_region", "ur_std_high"),
        "ur_std_collapse": ("trust_region", "ur_std_collapse"),
        "ur_relative": ("trust_region", "ur_relative"),
        "tr_init": ("trust_region", "initial_radius"),
        "tr_min": ("trust_region", "tr_min"),
        "tr_max": ("trust_region", "tr_max"),
        "max_restarts": ("trust_region", "max_restarts"),
        "geodesic_max_angle": ("trust_region", "geodesic_max_angle"),
        # Kernel
        "kernel_type": ("kernel", "kernel_type"),
        "kernel_order": ("kernel", "kernel_order"),
        "kernel_ard": ("kernel", "kernel_ard"),
        # Projection
        "lass": ("projection", "lass"),
        "lass_n_candidates": ("projection", "lass_n_candidates"),
        "whitening": ("projection", "whitening"),
        "adaptive_dim": ("projection", "adaptive_dim"),
        # Acquisition
        "acqf_schedule": ("acquisition", "schedule"),
        "acqf_ucb_beta_high": ("acquisition", "acqf_ucb_beta_high"),
        "acqf_ucb_beta_low": ("acquisition", "acqf_ucb_beta_low"),
        # Norm
        "prob_norm": ("norm_reconstruction", "method"),
    }

    if key in nested_maps:
        section, attr = nested_maps[key]
        sub_config = getattr(config, section)
        if key == "prob_norm" and value:
            setattr(sub_config, attr, "probabilistic")
        else:
            setattr(sub_config, attr, value)
    elif hasattr(config, key):
        setattr(config, key, value)
