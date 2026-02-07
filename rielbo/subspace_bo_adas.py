"""Adaptive Subspace BO (AdaS-BO): data-driven subspace dimension selection.

Extends SphericalSubspaceBOv2 with intrinsic dimensionality estimation from
cold-start data. The dimension is set once and fixed for the entire run.
Two restart strategies: "tr" (TuRBO-style adaptive TR) and "stagnation".
"""

import logging
from dataclasses import dataclass

import torch

from rielbo.adaptive_dim import assess_gp_health, estimate_intrinsic_dim
from rielbo.subspace_bo_v2 import SphericalSubspaceBOv2, V2Config

logger = logging.getLogger(__name__)


@dataclass
class AdaSConfig(V2Config):
    """V2Config extended with ID estimation and restart strategy parameters."""

    adaptive_subspace: bool = True
    restart_strategy: str = "tr"
    stagnation_patience: int = 40
    d_min: int = 8
    d_max: int = 48
    kernel_type: str = "geodesic_matern"
    geodesic_nu: float = 1.5

    @classmethod
    def from_preset(cls, preset: str) -> "AdaSConfig":
        """Create AdaS config from preset name."""
        presets = {
            "adas_tr": cls(
                adaptive_subspace=True,
                geodesic_tr=True,
                adaptive_tr=True,
                restart_strategy="tr",
                kernel_type="geodesic_matern",
            ),
            "adas_stag": cls(
                adaptive_subspace=True,
                geodesic_tr=True,
                adaptive_tr=False,
                restart_strategy="stagnation",
                kernel_type="geodesic_matern",
            ),
        }
        if preset in presets:
            return presets[preset]
        # Fall back to V2 presets, wrapping in AdaSConfig
        v2 = V2Config.from_preset(preset)
        return cls(**{f: getattr(v2, f) for f in V2Config.__dataclass_fields__})


class AdaptiveSubspaceBO(SphericalSubspaceBOv2):
    """Spherical Subspace BO with data-driven dimension selection.

    Estimates intrinsic dimensionality from cold-start data using TwoNN + MLE,
    then runs standard V2 geodesic optimization at that fixed dimension.
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        subspace_dim: int = 16,
        config: AdaSConfig | None = None,
        device: str = "cuda",
        n_candidates: int = 2000,
        ucb_beta: float = 2.0,
        acqf: str = "ts",
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
    ):
        if config is None:
            config = AdaSConfig.from_preset("adas_tr")

        super().__init__(
            codec=codec,
            oracle=oracle,
            input_dim=input_dim,
            subspace_dim=subspace_dim,
            config=config,
            device=device,
            n_candidates=n_candidates,
            ucb_beta=ucb_beta,
            acqf=acqf,
            trust_region=trust_region,
            seed=seed,
            verbose=verbose,
        )

        self._adas_config = config
        self._iters_without_improvement = 0
        self._last_restart_iter = 0
        self._best_score_history: list[float] = []
        self._id_diagnostics: dict = {}
        self.history["d_target"] = []
        self.history["gp_health_signals"] = []

    def _create_kernel(self):
        """Create kernel based on AdaS config (geodesic Matérn by default)."""
        cfg = self._adas_config
        if cfg.kernel_type == "geodesic_matern":
            from rielbo.kernels import create_kernel

            nu_to_order = {0.5: 0, 1.5: 1, 2.5: 2}
            order = nu_to_order.get(cfg.geodesic_nu, 1)
            return create_kernel(
                kernel_type="geodesic_matern",
                kernel_order=order,
                use_scale=True,
            )
        return super()._create_kernel()

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules and estimate intrinsic dim.

        Calls parent cold_start, then estimates intrinsic dimensionality
        and re-initializes the projection if the dimension differs.
        """
        super().cold_start(smiles_list, scores)

        if not self._adas_config.adaptive_subspace:
            return

        directions_np = self.train_U.cpu().numpy()
        n_points = len(directions_np)

        d_estimate, diag = estimate_intrinsic_dim(
            directions_np,
            n_points,
            d_min=self._adas_config.d_min,
            d_max=self._adas_config.d_max,
        )
        self._id_diagnostics = {"trigger": "cold_start", **diag}

        old_dim = self._current_dim
        if d_estimate != old_dim:
            self._current_dim = d_estimate
            self.subspace_dim = d_estimate

            torch.manual_seed(self.seed)
            self._init_projection()

            self.train_V = self.project_to_subspace(self.train_U)
            self._fit_gp()

            logger.info(
                f"AdaS cold start: dim {old_dim} -> {d_estimate} "
                f"(TwoNN={diag['d_twonn']:.1f}, MLE={diag['d_mle']:.1f})"
            )
        else:
            logger.info(f"AdaS cold start: dim confirmed at {d_estimate}")

        self._best_score_history = [self.best_score]

    def _restart_subspace(self):
        """Restart with fresh QR basis at the same dimension."""
        if self.n_restarts >= self.config.max_restarts:
            self.tr_length = self.config.tr_init
            logger.info(f"Max restarts ({self.config.max_restarts}) reached, resetting TR")
            return

        health = assess_gp_health(
            self.diagnostic_history,
            self.iteration,
            self._last_restart_iter,
            self._best_score_history,
        )

        self.n_restarts += 1
        if self.config.adaptive_tr:
            self.tr_length = self.config.tr_init
        self._tr_success_count = 0
        self._tr_fail_count = 0
        self._last_restart_iter = self.iteration
        self._iters_without_improvement = 0

        torch.manual_seed(self.seed + self.n_restarts * 1000)
        self._init_projection()

        if self.train_U is not None:
            self.train_V = self.project_to_subspace(self.train_U)
            self._fit_gp()

        logger.info(
            f"AdaS restart #{self.n_restarts}: fresh projection (d={self._current_dim}), "
            f"health={health['details']}"
        )

    def _check_stagnation(self, improved: bool):
        """Check for stagnation and trigger restart (stagnation strategy only)."""
        if self._adas_config.restart_strategy != "stagnation":
            return

        if improved:
            self._iters_without_improvement = 0
        else:
            self._iters_without_improvement += 1

        if self._iters_without_improvement >= self._adas_config.stagnation_patience:
            logger.info(
                f"Stagnation: {self._iters_without_improvement} iters without improvement"
            )
            self._restart_subspace()

    def _update_trust_region(self, improved: bool):
        """No-op for stagnation strategy; delegates to V2 for TR strategy."""
        if self._adas_config.restart_strategy != "stagnation":
            super()._update_trust_region(improved)

    def step(self) -> dict:
        """One BO iteration with health monitoring."""
        result = super().step()

        prev_best = self._best_score_history[-1] if self._best_score_history else float("-inf")
        self._best_score_history.append(self.best_score)
        best_improved = self.best_score > prev_best

        self._check_stagnation(best_improved)

        health = assess_gp_health(
            self.diagnostic_history,
            self.iteration,
            self._last_restart_iter,
            self._best_score_history,
        )

        if health["dead_gp"] and (self.iteration - self._last_restart_iter) > 30:
            logger.info(
                f"Dead GP at iter {self.iteration}: "
                f"std_ratios={health['details'].get('recent_std_ratios', [])}"
            )
            self._restart_subspace()

        self.history["d_target"].append(self._current_dim)
        self.history["gp_health_signals"].append(health)

        return result
