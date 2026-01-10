"""
Hyperparameter Space Definitions for BOLT Tuning

Three-tier prioritization:
- Tier 1 (CRITICAL): Tune first, highest impact
- Tier 2 (IMPORTANT): Tune after Tier 1 stable
- Tier 3 (FINETUNE): Final polish

Supports:
- Continuous parameters (log/linear scale)
- Categorical parameters
- Conditional parameters (e.g., DKL params only if use_deep_kernel=True)
- Phase-specific subsets
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of hyperparameters."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class ScaleType(Enum):
    """Scale for continuous parameters."""
    LINEAR = "linear"
    LOG = "log"


class TuningTier(Enum):
    """Priority tier for tuning."""
    CRITICAL = 1    # Tune first
    IMPORTANT = 2   # Tune second
    FINETUNE = 3    # Final polish


class TuningPhase(Enum):
    """Phases of Coordinate Descent."""
    VAE = "vae"
    SCORER = "scorer"
    GP = "gp"
    INFERENCE = "inference"


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    param_type: ParameterType
    tier: TuningTier
    phases: List[TuningPhase]  # Which phases this param affects
    description: str = ""

    # For continuous
    low: Optional[float] = None
    high: Optional[float] = None
    scale: ScaleType = ScaleType.LINEAR

    # For discrete
    values: Optional[List[Any]] = None

    # For categorical
    choices: Optional[List[Any]] = None

    # Conditional on another parameter
    condition: Optional[Tuple[str, Any]] = None  # (param_name, required_value)

    # Default value
    default: Optional[Any] = None

    def sample(self, rng: Optional[np.random.Generator] = None) -> Any:
        """Sample a random value from this parameter's space."""
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == ParameterType.CONTINUOUS:
            if self.scale == ScaleType.LOG:
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return float(math.exp(rng.uniform(log_low, log_high)))
            else:
                return float(rng.uniform(self.low, self.high))

        elif self.param_type == ParameterType.DISCRETE:
            return int(rng.choice(self.values))

        elif self.param_type == ParameterType.CATEGORICAL:
            return rng.choice(self.choices)

    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this parameter."""
        if self.param_type == ParameterType.CONTINUOUS:
            return self.low <= value <= self.high

        elif self.param_type == ParameterType.DISCRETE:
            return value in self.values

        elif self.param_type == ParameterType.CATEGORICAL:
            return value in self.choices

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "type": self.param_type.value,
            "tier": self.tier.value,
            "phases": [p.value for p in self.phases],
            "description": self.description,
            "low": self.low,
            "high": self.high,
            "scale": self.scale.value if self.scale else None,
            "values": self.values,
            "choices": self.choices,
            "condition": self.condition,
            "default": self.default,
        }


# =============================================================================
# TIER 1: CRITICAL PARAMETERS (Tune First)
# =============================================================================

CRITICAL_PARAMS: List[ParameterSpec] = [
    # VAE Regularization
    ParameterSpec(
        name="vae_beta",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.VAE],
        description="KL divergence weight (higher = more regularization, smoother latent)",
        low=0.005,
        high=0.1,
        scale=ScaleType.LOG,
        default=0.02,
    ),
    ParameterSpec(
        name="vae_mse_weight",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.VAE],
        description="MSE weight in reconstruction loss (rest is cosine)",
        low=0.1,
        high=0.5,
        scale=ScaleType.LINEAR,
        default=0.2,
    ),

    # Latent Dimensions
    ParameterSpec(
        name="instruction_latent_dim",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.VAE, TuningPhase.GP],
        description="Instruction VAE latent dimension",
        values=[12, 16, 24, 32],
        default=16,
    ),
    ParameterSpec(
        name="exemplar_latent_dim",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.VAE, TuningPhase.SCORER],
        description="Exemplar VAE latent dimension",
        values=[8, 12, 16, 24],
        default=8,
    ),

    # Exploration vs Exploitation
    ParameterSpec(
        name="ucb_beta",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.INFERENCE],
        description="Initial UCB exploration parameter",
        low=4.0,
        high=16.0,
        scale=ScaleType.LINEAR,
        default=8.0,
    ),
    ParameterSpec(
        name="ucb_beta_final",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.INFERENCE],
        description="Final UCB exploration parameter (after decay)",
        low=1.0,
        high=4.0,
        scale=ScaleType.LINEAR,
        default=2.0,
    ),

    # Exemplar Selection (BOLT-specific)
    ParameterSpec(
        name="mmr_lambda",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.SCORER],
        description="MMR balance: 1.0=relevance only, 0.0=diversity only",
        low=0.3,
        high=0.9,
        scale=ScaleType.LINEAR,
        default=0.7,
    ),
    ParameterSpec(
        name="selection_weight",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.CRITICAL,
        phases=[TuningPhase.VAE, TuningPhase.SCORER],
        description="Weight of exemplar selection loss in VAE training",
        low=0.05,
        high=0.4,
        scale=ScaleType.LINEAR,
        default=0.2,
    ),
]


# =============================================================================
# TIER 2: IMPORTANT PARAMETERS (Tune After Tier 1)
# =============================================================================

IMPORTANT_PARAMS: List[ParameterSpec] = [
    # Set Transformer Architecture
    ParameterSpec(
        name="set_transformer_heads",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.SCORER],
        description="Number of attention heads in Set Transformer",
        values=[4, 8],
        default=4,
    ),
    ParameterSpec(
        name="set_transformer_hidden",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.SCORER],
        description="Hidden dimension in Set Transformer",
        values=[64, 128, 256],
        default=128,
    ),
    ParameterSpec(
        name="num_inducing_points",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.SCORER],
        description="Number of inducing points in ISAB",
        values=[4, 8, 16],
        default=4,
    ),

    # GP Configuration
    ParameterSpec(
        name="gp_lr",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.GP],
        description="GP training learning rate",
        low=0.001,
        high=0.01,
        scale=ScaleType.LOG,
        default=0.0025,
    ),
    ParameterSpec(
        name="use_deep_kernel",
        param_type=ParameterType.CATEGORICAL,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.GP],
        description="Use Deep Kernel Learning (feature extractor before kernel)",
        choices=[True, False],
        default=True,
    ),
    ParameterSpec(
        name="dkl_output_dim",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.GP],
        description="DKL joint feature extractor output dimension (HbBoPs-style)",
        values=[6, 10, 16],
        default=10,
        condition=("use_deep_kernel", True),
    ),
    ParameterSpec(
        name="use_product_kernel",
        param_type=ParameterType.CATEGORICAL,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.GP],
        description="Use product kernel (legacy) vs single kernel (HbBoPs)",
        choices=[True, False],
        default=False,
        condition=("use_deep_kernel", True),
    ),
    ParameterSpec(
        name="dkl_hidden_dim",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.GP],
        description="DKL feature extractor hidden dimension",
        values=[16, 32, 64],
        default=32,
        condition=("use_deep_kernel", True),
    ),

    # VAE Training
    ParameterSpec(
        name="vae_lr",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.VAE],
        description="VAE training learning rate",
        low=0.0001,
        high=0.001,
        scale=ScaleType.LOG,
        default=0.0006,
    ),
    ParameterSpec(
        name="vae_epochs",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.VAE],
        description="Maximum VAE training epochs",
        values=[20000, 50000, 100000],
        default=50000,
    ),
    ParameterSpec(
        name="vae_annealing_epochs",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.VAE],
        description="KL annealing warmup epochs",
        values=[1000, 2500, 5000],
        default=2500,
    ),

    # Cross-Attention Scorer
    ParameterSpec(
        name="cross_attn_heads",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.SCORER],
        description="Number of heads in CrossAttention Scorer",
        values=[2, 4, 8],
        default=4,
    ),
    ParameterSpec(
        name="scorer_hidden_dim",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.SCORER],
        description="Hidden dimension in scorer MLP",
        values=[64, 128, 256],
        default=128,
    ),

    # Number of Exemplars
    ParameterSpec(
        name="num_exemplars",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.IMPORTANT,
        phases=[TuningPhase.SCORER, TuningPhase.INFERENCE],
        description="Number of exemplars to select (K)",
        values=[4, 6, 8, 10, 12],
        default=8,
    ),
]


# =============================================================================
# TIER 3: FINE-TUNING PARAMETERS
# =============================================================================

FINETUNE_PARAMS: List[ParameterSpec] = [
    # Inference Optimization
    ParameterSpec(
        name="num_restarts",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="L-BFGS-B restarts for acquisition optimization",
        values=[32, 64, 128],
        default=64,
    ),
    ParameterSpec(
        name="raw_samples",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Raw samples for acquisition initialization",
        values=[2048, 4096, 8192],
        default=4096,
    ),
    ParameterSpec(
        name="cosine_sim_threshold",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Threshold for accepting reconstructed instructions",
        low=0.85,
        high=0.95,
        scale=ScaleType.LINEAR,
        default=0.90,
    ),

    # Distance Penalty
    ParameterSpec(
        name="distance_weight",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Weight of distance penalty in acquisition",
        low=1.0,
        high=4.0,
        scale=ScaleType.LINEAR,
        default=2.0,
    ),
    ParameterSpec(
        name="distance_threshold",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Threshold for distance penalty activation",
        low=0.2,
        high=0.5,
        scale=ScaleType.LINEAR,
        default=0.3,
    ),

    # Latent Space
    ParameterSpec(
        name="latent_noise_scale",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Noise added to latent during optimization",
        low=0.01,
        high=0.1,
        scale=ScaleType.LOG,
        default=0.05,
    ),
    ParameterSpec(
        name="latent_margin",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Margin for latent bounds expansion",
        low=0.05,
        high=0.2,
        scale=ScaleType.LINEAR,
        default=0.1,
    ),

    # Early Stopping
    ParameterSpec(
        name="vae_patience",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.VAE],
        description="Early stopping patience for VAE",
        values=[500, 1000, 2000],
        default=1000,
    ),
    ParameterSpec(
        name="gp_patience",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.GP],
        description="Early stopping patience for GP",
        values=[50, 100, 200],
        default=100,
    ),
    ParameterSpec(
        name="gp_epochs",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.GP],
        description="Maximum GP training epochs",
        values=[5000, 10000, 20000],
        default=10000,
    ),

    # Gradient Clipping
    ParameterSpec(
        name="vae_grad_clip",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.VAE],
        description="Gradient clipping value for VAE",
        low=0.5,
        high=2.0,
        scale=ScaleType.LINEAR,
        default=1.0,
    ),

    # Vec2Text
    ParameterSpec(
        name="vec2text_beam",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Beam width for Vec2Text decoding",
        values=[4, 8, 16],
        default=8,
    ),
    ParameterSpec(
        name="vec2text_max_length",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Maximum token length for Vec2Text",
        values=[64, 128, 256],
        default=128,
    ),

    # Hyperband
    ParameterSpec(
        name="bmin",
        param_type=ParameterType.DISCRETE,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.GP],  # Affects GP training data
        description="Minimum fidelity for Hyperband",
        values=[5, 10, 20],
        default=10,
    ),
    ParameterSpec(
        name="eta",
        param_type=ParameterType.CONTINUOUS,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.GP],
        description="Successive halving rate",
        low=2.0,
        high=3.0,
        scale=ScaleType.LINEAR,
        default=2.0,
    ),

    # Acquisition Type
    ParameterSpec(
        name="acquisition_type",
        param_type=ParameterType.CATEGORICAL,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.INFERENCE],
        description="Acquisition function type",
        choices=["ucb", "logei"],
        default="ucb",
    ),

    # Ranking Loss
    ParameterSpec(
        name="ranking_loss_type",
        param_type=ParameterType.CATEGORICAL,
        tier=TuningTier.FINETUNE,
        phases=[TuningPhase.SCORER],
        description="Ranking loss type for scorer",
        choices=["listmle", "bce"],
        default="listmle",
    ),
]


# =============================================================================
# HYPERPARAMETER SPACE CLASS
# =============================================================================

@dataclass
class HyperparameterConfig:
    """A specific configuration of hyperparameters."""
    values: Dict[str, Any]
    tier: TuningTier
    phase: Optional[TuningPhase] = None
    trial_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "tier": self.tier.value,
            "phase": self.phase.value if self.phase else None,
            "trial_id": self.trial_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HyperparameterConfig:
        return cls(
            values=d["values"],
            tier=TuningTier(d["tier"]),
            phase=TuningPhase(d["phase"]) if d.get("phase") else None,
            trial_id=d.get("trial_id"),
        )


class HyperparameterSpace:
    """
    Complete hyperparameter space with sampling and filtering.

    Features:
    - Tier-based prioritization
    - Phase-specific filtering
    - Conditional parameter handling
    - Sampling strategies (random, grid, sobol)
    - Serialization for checkpointing
    """

    def __init__(self):
        self._params: Dict[str, ParameterSpec] = {}
        self._build_space()

    def _build_space(self):
        """Build the complete parameter space."""
        for param in CRITICAL_PARAMS + IMPORTANT_PARAMS + FINETUNE_PARAMS:
            self._params[param.name] = param

    def get_param(self, name: str) -> Optional[ParameterSpec]:
        """Get a parameter by name."""
        return self._params.get(name)

    def get_all_params(self) -> List[ParameterSpec]:
        """Get all parameters."""
        return list(self._params.values())

    def get_params_by_tier(self, tier: TuningTier) -> List[ParameterSpec]:
        """Get parameters by tier."""
        return [p for p in self._params.values() if p.tier == tier]

    def get_params_by_phase(self, phase: TuningPhase) -> List[ParameterSpec]:
        """Get parameters relevant to a phase."""
        return [p for p in self._params.values() if phase in p.phases]

    def get_params_by_tier_and_phase(
        self,
        tier: TuningTier,
        phase: TuningPhase,
    ) -> List[ParameterSpec]:
        """Get parameters by both tier and phase."""
        return [
            p for p in self._params.values()
            if p.tier == tier and phase in p.phases
        ]

    def sample_random(
        self,
        tier: Optional[TuningTier] = None,
        phase: Optional[TuningPhase] = None,
        base_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> HyperparameterConfig:
        """
        Sample a random configuration.

        Args:
            tier: Only sample params from this tier (None = all)
            phase: Only sample params for this phase (None = all)
            base_config: Base configuration to extend
            seed: Random seed
        """
        rng = np.random.default_rng(seed)

        # Get relevant parameters
        params = self.get_all_params()
        if tier:
            params = [p for p in params if p.tier == tier]
        if phase:
            params = [p for p in params if phase in p.phases]

        # Start from base or defaults
        config = dict(base_config) if base_config else {}
        for p in self._params.values():
            if p.name not in config and p.default is not None:
                config[p.name] = p.default

        # Sample each parameter
        for param in params:
            # Check condition
            if param.condition:
                cond_param, cond_value = param.condition
                if config.get(cond_param) != cond_value:
                    continue

            config[param.name] = param.sample(rng)

        return HyperparameterConfig(
            values=config,
            tier=tier or TuningTier.CRITICAL,
            phase=phase,
        )

    def sample_grid(
        self,
        tier: Optional[TuningTier] = None,
        phase: Optional[TuningPhase] = None,
        n_points_continuous: int = 5,
    ) -> List[HyperparameterConfig]:
        """
        Generate a grid of configurations.

        Args:
            tier: Only include params from this tier
            phase: Only include params for this phase
            n_points_continuous: Number of points for continuous params
        """
        params = self.get_all_params()
        if tier:
            params = [p for p in params if p.tier == tier]
        if phase:
            params = [p for p in params if phase in p.phases]

        # Build grid for each parameter
        grids = {}
        for param in params:
            if param.param_type == ParameterType.CONTINUOUS:
                if param.scale == ScaleType.LOG:
                    grids[param.name] = np.logspace(
                        np.log10(param.low),
                        np.log10(param.high),
                        n_points_continuous,
                    ).tolist()
                else:
                    grids[param.name] = np.linspace(
                        param.low,
                        param.high,
                        n_points_continuous,
                    ).tolist()
            elif param.param_type == ParameterType.DISCRETE:
                grids[param.name] = param.values
            elif param.param_type == ParameterType.CATEGORICAL:
                grids[param.name] = param.choices

        # Generate all combinations (be careful with explosion!)
        if not grids:
            return []

        # Use itertools.product
        import itertools
        keys = list(grids.keys())
        combinations = list(itertools.product(*[grids[k] for k in keys]))

        configs = []
        for combo in combinations:
            values = dict(zip(keys, combo))

            # Add defaults for missing params
            for p in self._params.values():
                if p.name not in values and p.default is not None:
                    values[p.name] = p.default

            # Handle conditions
            valid = True
            for param in params:
                if param.condition:
                    cond_param, cond_value = param.condition
                    if values.get(cond_param) != cond_value:
                        # Remove conditional param if condition not met
                        values.pop(param.name, None)

            configs.append(HyperparameterConfig(
                values=values,
                tier=tier or TuningTier.CRITICAL,
                phase=phase,
            ))

        return configs

    def sample_sobol(
        self,
        n_samples: int,
        tier: Optional[TuningTier] = None,
        phase: Optional[TuningPhase] = None,
        seed: Optional[int] = None,
    ) -> List[HyperparameterConfig]:
        """
        Sample using Sobol quasi-random sequence.

        Better space coverage than pure random.
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            # Fall back to random sampling
            logger.warning(
                "scipy.stats.qmc not available - falling back to random sampling. "
                "Sobol sequences provide better space coverage. "
                "Install scipy for improved sampling: pip install scipy"
            )
            return [
                self.sample_random(tier, phase, seed=seed + i if seed else None)
                for i in range(n_samples)
            ]

        params = self.get_all_params()
        if tier:
            params = [p for p in params if p.tier == tier]
        if phase:
            params = [p for p in params if phase in p.phases]

        # Filter to continuous params for Sobol
        continuous_params = [p for p in params if p.param_type == ParameterType.CONTINUOUS]

        if not continuous_params:
            return [
                self.sample_random(tier, phase, seed=seed + i if seed else None)
                for i in range(n_samples)
            ]

        # Generate Sobol sequence
        sampler = qmc.Sobol(d=len(continuous_params), scramble=True, seed=seed)
        sobol_samples = sampler.random(n_samples)

        configs = []
        for i, sample in enumerate(sobol_samples):
            # Start with defaults
            values = {}
            for p in self._params.values():
                if p.default is not None:
                    values[p.name] = p.default

            # Map Sobol [0,1] to actual parameter ranges
            for j, param in enumerate(continuous_params):
                unit_value = sample[j]
                if param.scale == ScaleType.LOG:
                    log_low = np.log(param.low)
                    log_high = np.log(param.high)
                    values[param.name] = float(np.exp(log_low + unit_value * (log_high - log_low)))
                else:
                    values[param.name] = float(param.low + unit_value * (param.high - param.low))

            # Sample discrete/categorical params randomly
            rng = np.random.default_rng(seed + i if seed else None)
            for param in params:
                if param.param_type in [ParameterType.DISCRETE, ParameterType.CATEGORICAL]:
                    values[param.name] = param.sample(rng)

            # Handle conditions
            for param in params:
                if param.condition:
                    cond_param, cond_value = param.condition
                    if values.get(cond_param) != cond_value:
                        values.pop(param.name, None)

            configs.append(HyperparameterConfig(
                values=values,
                tier=tier or TuningTier.CRITICAL,
                phase=phase,
            ))

        return configs

    def get_default_config(self) -> HyperparameterConfig:
        """Get configuration with all defaults."""
        values = {}
        for param in self._params.values():
            if param.default is not None:
                values[param.name] = param.default
        return HyperparameterConfig(
            values=values,
            tier=TuningTier.CRITICAL,
        )

    def validate_config(self, config: HyperparameterConfig) -> Tuple[bool, List[str]]:
        """
        Validate a configuration.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        for name, value in config.values.items():
            param = self._params.get(name)
            if param is None:
                errors.append(f"Unknown parameter: {name}")
                continue

            if not param.is_valid(value):
                errors.append(f"Invalid value for {name}: {value}")

            # Check condition
            if param.condition:
                cond_param, cond_value = param.condition
                if config.values.get(cond_param) != cond_value:
                    errors.append(
                        f"Parameter {name} requires {cond_param}={cond_value}, "
                        f"but got {config.values.get(cond_param)}"
                    )

        return len(errors) == 0, errors

    def save(self, path: Path):
        """Save space definition to JSON."""
        data = {
            "params": {name: param.to_dict() for name, param in self._params.items()},
            "tiers": {
                "critical": [p.name for p in self.get_params_by_tier(TuningTier.CRITICAL)],
                "important": [p.name for p in self.get_params_by_tier(TuningTier.IMPORTANT)],
                "finetune": [p.name for p in self.get_params_by_tier(TuningTier.FINETUNE)],
            },
            "phases": {
                phase.value: [p.name for p in self.get_params_by_phase(phase)]
                for phase in TuningPhase
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> Dict[str, Any]:
        """Generate summary of the hyperparameter space."""
        return {
            "total_params": len(self._params),
            "by_tier": {
                tier.name: len(self.get_params_by_tier(tier))
                for tier in TuningTier
            },
            "by_phase": {
                phase.name: len(self.get_params_by_phase(phase))
                for phase in TuningPhase
            },
            "by_type": {
                ptype.name: len([p for p in self._params.values() if p.param_type == ptype])
                for ptype in ParameterType
            },
            "conditional_params": [
                p.name for p in self._params.values() if p.condition
            ],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_phase_params(phase: TuningPhase, tier: Optional[TuningTier] = None) -> List[ParameterSpec]:
    """Get parameters for a specific phase and optionally tier."""
    space = HyperparameterSpace()
    if tier:
        return space.get_params_by_tier_and_phase(tier, phase)
    return space.get_params_by_phase(phase)


def get_critical_vae_params() -> List[ParameterSpec]:
    """Shortcut for critical VAE parameters."""
    return get_phase_params(TuningPhase.VAE, TuningTier.CRITICAL)


def get_critical_scorer_params() -> List[ParameterSpec]:
    """Shortcut for critical scorer parameters."""
    return get_phase_params(TuningPhase.SCORER, TuningTier.CRITICAL)


def get_critical_gp_params() -> List[ParameterSpec]:
    """Shortcut for critical GP parameters."""
    return get_phase_params(TuningPhase.GP, TuningTier.CRITICAL)


def get_critical_inference_params() -> List[ParameterSpec]:
    """Shortcut for critical inference parameters."""
    return get_phase_params(TuningPhase.INFERENCE, TuningTier.CRITICAL)
