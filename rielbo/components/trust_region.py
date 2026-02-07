"""Trust region strategies: Static, Adaptive (TuRBO), UR-TR.

Extracted from V2's _update_trust_region(), _update_ur_tr(),
VanillaBO's _update_trust_region(), and TuRBO's TurboState.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rielbo.core.config import TrustRegionConfig

logger = logging.getLogger(__name__)


@dataclass
class TRState:
    """Shared trust region state."""

    radius: float
    success_count: int = 0
    fail_count: int = 0
    n_restarts: int = 0
    # UR-TR specific
    ur_collapse_count: int = 0
    ur_n_rotations: int = 0
    prev_gp_std: float = 1.0
    # Flags
    _needs_restart: bool = False
    _needs_rotation: bool = False

    def clear_flags(self) -> None:
        self._needs_restart = False
        self._needs_rotation = False


class StaticTR:
    """Static trust region — constant radius, no adaptation."""

    def __init__(self, config: TrustRegionConfig, trust_region: float = 0.8):
        self._radius = config.initial_radius if config.geodesic else trust_region
        self.config = config

    @property
    def radius(self) -> float:
        return self._radius

    def update(self, improved: bool, gp_std: float | None = None) -> None:
        pass

    @property
    def needs_restart(self) -> bool:
        return False

    @property
    def needs_rotation(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def get_state(self) -> TRState:
        return TRState(radius=self._radius)


class AdaptiveTR:
    """TuRBO-style adaptive trust region with grow/shrink/restart.

    Grows on consecutive successes, shrinks on consecutive failures,
    restarts with a fresh subspace when radius collapses below tr_min.
    """

    def __init__(self, config: TrustRegionConfig):
        self.config = config
        self._state = TRState(radius=config.initial_radius)

    @property
    def radius(self) -> float:
        return self._state.radius

    @property
    def n_restarts(self) -> int:
        return self._state.n_restarts

    def update(self, improved: bool, gp_std: float | None = None) -> None:
        cfg = self.config
        self._state.clear_flags()

        if improved:
            self._state.success_count += 1
            self._state.fail_count = 0
        else:
            self._state.success_count = 0
            self._state.fail_count += 1

        if self._state.success_count >= cfg.tr_success_tol:
            self._state.radius = min(
                self._state.radius * cfg.tr_grow_factor, cfg.tr_max,
            )
            self._state.success_count = 0
            logger.debug(f"TR grow -> {self._state.radius:.4f}")

        elif self._state.fail_count >= cfg.tr_fail_tol:
            self._state.radius *= cfg.tr_shrink_factor
            self._state.fail_count = 0

            if self._state.radius < cfg.tr_min:
                if self._state.n_restarts >= cfg.max_restarts:
                    self._state.radius = cfg.initial_radius
                    logger.info(
                        f"Max restarts ({cfg.max_restarts}) reached, "
                        f"resetting TR to {cfg.initial_radius}"
                    )
                else:
                    self._state._needs_restart = True
            else:
                logger.debug(f"TR shrink -> {self._state.radius:.4f}")

    @property
    def needs_restart(self) -> bool:
        return self._state._needs_restart

    @property
    def needs_rotation(self) -> bool:
        return False

    def reset(self) -> None:
        """Called after subspace restart is performed."""
        self._state.n_restarts += 1
        self._state.radius = self.config.initial_radius
        self._state.success_count = 0
        self._state.fail_count = 0
        self._state.clear_flags()

    def get_state(self) -> TRState:
        return TRState(
            radius=self._state.radius,
            success_count=self._state.success_count,
            fail_count=self._state.fail_count,
            n_restarts=self._state.n_restarts,
        )


class URTR:
    """Uncertainty-Responsive Trust Region.

    Counter-intuitive: EXPAND when GP std drops (collapsing),
    SHRINK when GP std is high (informative). Rotates subspace
    on sustained GP collapse.
    """

    def __init__(
        self,
        config: TrustRegionConfig,
        initial_radius: float,
        gp_getter=None,
    ):
        self.config = config
        self._state = TRState(radius=initial_radius)
        self._initial_radius = initial_radius
        self._gp_getter = gp_getter  # callable returning current GP

    @property
    def radius(self) -> float:
        return self._state.radius

    @property
    def n_rotations(self) -> int:
        return self._state.ur_n_rotations

    def update(self, improved: bool, gp_std: float | None = None) -> None:
        if gp_std is None:
            return
        cfg = self.config
        self._state.clear_flags()
        self._state.prev_gp_std = gp_std

        # Compute noise-relative thresholds
        noise_std = 1.0
        if cfg.ur_relative and self._gp_getter is not None:
            gp = self._gp_getter()
            if gp is not None:
                try:
                    noise_std = gp.likelihood.noise.item() ** 0.5
                except Exception as e:
                    logger.debug(f"Failed to get GP noise std for UR-TR: {e}")
                    noise_std = 1.0

        eff_std_high = cfg.ur_std_high * noise_std
        eff_std_low = cfg.ur_std_low * noise_std

        if gp_std > eff_std_high:
            # GP informative -> shrink for local exploitation
            self._state.radius = max(
                self._state.radius * cfg.ur_shrink_factor, cfg.ur_tr_min,
            )
            self._state.ur_collapse_count = 0

        elif gp_std < eff_std_low:
            # GP collapsing -> expand for broader exploration
            self._state.radius = min(
                self._state.radius * cfg.ur_expand_factor, cfg.ur_tr_max,
            )

            # Track sustained collapse
            eff_std_collapse = cfg.ur_std_collapse * noise_std
            if gp_std < eff_std_collapse:
                self._state.ur_collapse_count += 1
            else:
                self._state.ur_collapse_count = 0

            # Trigger rotation on sustained collapse
            if self._state.ur_collapse_count >= cfg.ur_collapse_patience:
                if self._state.ur_n_rotations < cfg.ur_max_rotations:
                    self._state._needs_rotation = True
                else:
                    # Max rotations: just reset radius
                    self._state.radius = self._initial_radius
                    self._state.ur_collapse_count = 0
        else:
            # Normal range — keep radius
            self._state.ur_collapse_count = 0

    @property
    def needs_restart(self) -> bool:
        return False

    @property
    def needs_rotation(self) -> bool:
        return self._state._needs_rotation

    def reset(self) -> None:
        """Called after rotation is performed."""
        self._state.ur_n_rotations += 1
        self._state.ur_collapse_count = 0
        self._state.radius = self._initial_radius
        self._state.clear_flags()

    def get_state(self) -> TRState:
        return TRState(
            radius=self._state.radius,
            ur_collapse_count=self._state.ur_collapse_count,
            ur_n_rotations=self._state.ur_n_rotations,
            prev_gp_std=self._state.prev_gp_std,
        )


class TuRBOTR:
    """TuRBO trust region for full-dimensional BO.

    Consolidates TurboState and VanillaBO's TR logic.
    """

    def __init__(self, config: TrustRegionConfig, dim: int):
        self.config = config
        self.dim = dim
        self._radius = config.initial_radius
        self._success_count = 0
        self._failure_count = 0
        self._restart_count = 0

        # Derive failure tolerance
        self._failure_tol = config.turbo_failure_tolerance or max(5, dim // 20)
        self._success_tol = config.vanilla_success_tolerance
        self._length_min = config.turbo_length_min
        self._length_max = config.turbo_length_max
        self._best_value = float("-inf")

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def n_restarts(self) -> int:
        return self._restart_count

    def set_best_value(self, val: float) -> None:
        self._best_value = val

    def update(self, improved: bool, gp_std: float | None = None) -> None:
        if improved:
            self._success_count += 1
            self._failure_count = 0
        else:
            self._failure_count += 1
            self._success_count = 0

        if self._success_count >= self._success_tol:
            self._radius = min(2.0 * self._radius, self._length_max)
            self._success_count = 0

        if self._failure_count >= self._failure_tol:
            self._radius = self._radius / 2.0
            self._failure_count = 0

        if self._radius < self._length_min:
            self._restart_count += 1
            logger.info(
                f"TuRBO restart #{self._restart_count} (was {self._radius:.6f})"
            )
            self._radius = self.config.initial_radius
            self._success_count = 0
            self._failure_count = 0

    @property
    def needs_restart(self) -> bool:
        return False  # TuRBOTR handles restart internally

    @property
    def needs_rotation(self) -> bool:
        return False

    def reset(self) -> None:
        self._radius = self.config.initial_radius
        self._success_count = 0
        self._failure_count = 0

    def get_state(self) -> TRState:
        return TRState(
            radius=self._radius,
            success_count=self._success_count,
            fail_count=self._failure_count,
            n_restarts=self._restart_count,
        )


def create_trust_region(
    config: TrustRegionConfig,
    trust_region: float = 0.8,
    dim: int = 256,
    gp_getter=None,
) -> StaticTR | AdaptiveTR | URTR | TuRBOTR:
    """Factory for trust region strategies."""
    strategy = config.strategy

    if strategy == "static":
        return StaticTR(config, trust_region=trust_region)
    elif strategy == "adaptive":
        return AdaptiveTR(config)
    elif strategy == "ur":
        init_radius = config.geodesic_max_angle * trust_region
        return URTR(config, initial_radius=init_radius, gp_getter=gp_getter)
    elif strategy == "turbo":
        return TuRBOTR(config, dim=dim)
    else:
        raise ValueError(f"Unknown TR strategy: {strategy}")
