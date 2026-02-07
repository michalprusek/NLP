"""Nested dataclass configuration system with preset registry.

Each component has its own config dataclass. OptimizerConfig composes them.
Presets are factory functions returning fully configured OptimizerConfig.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class KernelConfig:
    """Kernel configuration."""

    kernel_type: str = "arccosine"  # "arccosine", "geodesic_matern", "matern", "hvarfner"
    kernel_order: int = 0  # 0 or 2 for arccosine; maps to nu for geodesic_matern
    kernel_ard: bool = False
    product_space: bool = False
    n_spheres: int = 4


@dataclass
class ProjectionConfig:
    """Subspace projection configuration."""

    projection_type: str = "random"  # "random" (QR), "pca", "identity"
    subspace_dim: int = 16
    input_dim: int = 256

    # LASS (Look-Ahead Subspace Selection)
    lass: bool = False
    lass_n_candidates: int = 50

    # Adaptive dimension
    adaptive_dim: bool = False
    adaptive_start_dim: int = 8
    adaptive_end_dim: int = 16
    adaptive_switch_frac: float = 0.5

    # Whitening
    whitening: bool = False


@dataclass
class TrustRegionConfig:
    """Trust region configuration — covers all TR strategies."""

    strategy: str = "adaptive"  # "static", "adaptive", "ur", "turbo"

    # Geodesic TR parameters
    geodesic: bool = False
    geodesic_max_angle: float = 0.5
    geodesic_global_fraction: float = 0.2

    # Adaptive TR (TuRBO-style grow/shrink)
    initial_radius: float = 0.4
    tr_min: float = 0.02
    tr_max: float = 0.8
    tr_success_tol: int = 3
    tr_fail_tol: int = 10
    tr_grow_factor: float = 1.5
    tr_shrink_factor: float = 0.5
    max_restarts: int = 5

    # UR-TR (Uncertainty-Responsive)
    ur_tr: bool = False
    ur_relative: bool = True
    ur_std_high: float = 0.15
    ur_std_low: float = 0.05
    ur_std_collapse: float = 0.005
    ur_expand_factor: float = 1.5
    ur_shrink_factor: float = 0.8
    ur_collapse_patience: int = 15
    ur_tr_min: float = 0.1
    ur_tr_max: float = 1.2
    ur_max_rotations: int = 10

    # TuRBO-specific (full-dim)
    turbo_failure_tolerance: int | None = None  # None → dim // 20
    turbo_length_min: float = 0.5**7
    turbo_length_max: float = 1.6

    # VanillaBO (full-dim Euclidean)
    vanilla_success_tolerance: int = 3
    vanilla_length_min: float = 0.5**7
    vanilla_length_max: float = 1.6

    def __post_init__(self) -> None:
        if self.tr_min >= self.tr_max:
            raise ValueError(f"tr_min ({self.tr_min}) must be < tr_max ({self.tr_max})")
        if self.ur_std_low >= self.ur_std_high:
            raise ValueError(
                f"ur_std_low ({self.ur_std_low}) must be < ur_std_high ({self.ur_std_high})"
            )
        if self.ur_tr_min >= self.ur_tr_max:
            raise ValueError(
                f"ur_tr_min ({self.ur_tr_min}) must be < ur_tr_max ({self.ur_tr_max})"
            )


@dataclass
class AcquisitionConfig:
    """Acquisition function configuration."""

    acqf: str = "ts"  # "ts", "ei", "ucb"
    ucb_beta: float = 2.0
    n_candidates: int = 2000

    # Acquisition schedule (switches to UCB on GP collapse)
    schedule: bool = False
    acqf_ucb_beta_high: float = 4.0
    acqf_ucb_beta_low: float = 0.5


@dataclass
class CandidateGenConfig:
    """Candidate generation strategy."""

    strategy: str = "geodesic"  # "geodesic", "sobol_box"


@dataclass
class NormReconstructionConfig:
    """Norm reconstruction configuration."""

    method: str = "mean"  # "mean", "probabilistic"
    prob_method: str = "gaussian"  # "gaussian", "histogram", "gmm"
    n_candidates: int = 5


@dataclass
class OptimizerConfig:
    """Top-level optimizer configuration composing all components."""

    kernel: KernelConfig = field(default_factory=KernelConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    trust_region: TrustRegionConfig = field(default_factory=TrustRegionConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    candidate_gen: CandidateGenConfig = field(default_factory=CandidateGenConfig)
    norm_reconstruction: NormReconstructionConfig = field(
        default_factory=NormReconstructionConfig
    )

    # General
    seed: int = 42
    device: str = "cuda"
    verbose: bool = True

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> OptimizerConfig:
        """Create config from preset name with optional overrides."""
        factory = _PRESETS.get(preset)
        if factory is None:
            available = list(_PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset}. Available: {available}")
        config = factory()
        # Apply overrides to top-level fields
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    @classmethod
    def available_presets(cls) -> list[str]:
        return list(_PRESETS.keys())


# ── Preset factory functions ─────────────────────────────────────────


def _preset_baseline() -> OptimizerConfig:
    """No improvements — matches V1 behavior."""
    return OptimizerConfig(
        trust_region=TrustRegionConfig(strategy="static", geodesic=False),
        candidate_gen=CandidateGenConfig(strategy="sobol_box"),
    )


def _preset_geodesic() -> OptimizerConfig:
    """Geodesic TR + adaptive TR with restart (V2 recommended)."""
    return OptimizerConfig(
        trust_region=TrustRegionConfig(
            strategy="adaptive",
            geodesic=True,
        ),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_ur_tr() -> OptimizerConfig:
    """Geodesic TR + uncertainty-responsive TR."""
    return OptimizerConfig(
        trust_region=TrustRegionConfig(
            strategy="ur",
            geodesic=True,
            ur_tr=True,
        ),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_lass() -> OptimizerConfig:
    """Geodesic TR + LASS projection selection."""
    return OptimizerConfig(
        projection=ProjectionConfig(lass=True),
        trust_region=TrustRegionConfig(
            strategy="adaptive",
            geodesic=True,
        ),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_lass_ur() -> OptimizerConfig:
    """Geodesic TR + LASS + UR-TR."""
    return OptimizerConfig(
        projection=ProjectionConfig(lass=True),
        trust_region=TrustRegionConfig(
            strategy="ur",
            geodesic=True,
            ur_tr=True,
        ),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_explore() -> OptimizerConfig:
    """LASS + UR-TR + acquisition schedule (SOTA, 0.5555±0.013)."""
    return OptimizerConfig(
        projection=ProjectionConfig(lass=True),
        trust_region=TrustRegionConfig(
            strategy="ur",
            geodesic=True,
            ur_tr=True,
        ),
        acquisition=AcquisitionConfig(schedule=True),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_turbo() -> OptimizerConfig:
    """TuRBO baseline — full-dim GP with Matern-5/2."""
    return OptimizerConfig(
        kernel=KernelConfig(kernel_type="matern", kernel_ard=True),
        projection=ProjectionConfig(projection_type="identity"),
        trust_region=TrustRegionConfig(
            strategy="turbo",
            geodesic=False,
            initial_radius=0.8,
        ),
        candidate_gen=CandidateGenConfig(strategy="sobol_box"),
    )


def _preset_vanilla() -> OptimizerConfig:
    """Vanilla BO with Hvarfner priors — full-dim GP with RBF+ARD."""
    return OptimizerConfig(
        kernel=KernelConfig(kernel_type="hvarfner"),
        projection=ProjectionConfig(projection_type="identity"),
        trust_region=TrustRegionConfig(
            strategy="turbo",
            geodesic=False,
            initial_radius=0.8,
        ),
        candidate_gen=CandidateGenConfig(strategy="sobol_box"),
    )


def _preset_order2() -> OptimizerConfig:
    return OptimizerConfig(
        kernel=KernelConfig(kernel_order=2),
        trust_region=TrustRegionConfig(strategy="static", geodesic=False),
        candidate_gen=CandidateGenConfig(strategy="sobol_box"),
    )


def _preset_whitening() -> OptimizerConfig:
    return OptimizerConfig(
        projection=ProjectionConfig(whitening=True),
        trust_region=TrustRegionConfig(strategy="static", geodesic=False),
        candidate_gen=CandidateGenConfig(strategy="sobol_box"),
    )


def _preset_geometric() -> OptimizerConfig:
    return OptimizerConfig(
        projection=ProjectionConfig(whitening=True),
        trust_region=TrustRegionConfig(
            strategy="adaptive", geodesic=True,
        ),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_full() -> OptimizerConfig:
    return OptimizerConfig(
        kernel=KernelConfig(kernel_order=2),
        projection=ProjectionConfig(whitening=True, adaptive_dim=True),
        trust_region=TrustRegionConfig(
            strategy="adaptive", geodesic=True,
        ),
        norm_reconstruction=NormReconstructionConfig(method="probabilistic"),
        candidate_gen=CandidateGenConfig(strategy="geodesic"),
    )


def _preset_portfolio() -> OptimizerConfig:
    """Multi-subspace portfolio (experimental). Same config as explore."""
    return _preset_explore()


# ── Preset registry ──────────────────────────────────────────────────

_PRESETS: dict[str, Callable[[], OptimizerConfig]] = {
    "baseline": _preset_baseline,
    "geodesic": _preset_geodesic,
    "ur_tr": _preset_ur_tr,
    "lass": _preset_lass,
    "lass_ur": _preset_lass_ur,
    "explore": _preset_explore,
    "turbo": _preset_turbo,
    "vanilla": _preset_vanilla,
    "order2": _preset_order2,
    "whitening": _preset_whitening,
    "geometric": _preset_geometric,
    "full": _preset_full,
    "portfolio": _preset_portfolio,
}
