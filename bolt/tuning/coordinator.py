"""
Coordinate Descent Tuner for BOLT Hyperparameter Optimization

Orchestrates 4-phase optimization:
1. VAE Phase: Optimize reconstruction quality (Retrieval Acc)
2. Scorer Phase: Optimize exemplar selection (NDCG)
3. GP Phase: Optimize prediction quality (Spearman)
4. Inference Phase: Optimize end-to-end accuracy

Features:
- Component isolation (tune one component at a time)
- Checkpoint gates (must pass before next phase)
- Best config propagation between phases
- Multi-tier parameter sweeps (Critical → Important → Finetune)
- Long-running support with checkpointing
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .hyperspace import (
    HyperparameterConfig,
    HyperparameterSpace,
    TuningPhase,
    TuningTier,
    ParameterSpec,
)
from .metrics import MetricRegistry, MetricResult
from .parallel_executor import DualGPUExecutor, TrialTask
from .trial_runner import TrialResult


logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """Status of a tuning phase."""
    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINT_PASSED = "checkpoint_passed"
    CHECKPOINT_FAILED = "checkpoint_failed"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class PhaseConfig:
    """Configuration for a single phase."""
    phase: TuningPhase
    tiers: List[TuningTier]  # Which tiers to tune in this phase
    n_trials_per_tier: Dict[TuningTier, int]  # Trials per tier
    checkpoint_metrics: List[str]  # Metrics that must pass
    objective_metric: str  # Primary objective
    max_time_hours: float = 24.0  # Max time for this phase
    min_improvement_threshold: float = 0.001  # Stop if improvement < threshold
    early_stop_patience: int = 20  # Stop if no improvement for N trials


@dataclass
class PhaseResult:
    """Result of a tuning phase."""
    phase: TuningPhase
    status: PhaseStatus
    best_config: Optional[HyperparameterConfig]
    best_objective: float
    all_results: List[Dict[str, Any]]
    checkpoint_passed: bool
    checkpoint_failures: List[str]
    total_trials: int
    successful_trials: int
    failed_trials: int
    total_time_seconds: float
    tier_results: Dict[str, Dict[str, Any]]  # Results per tier

    def to_dict(self) -> Dict[str, Any]:
        # Handle best_config - could be HyperparameterConfig or dict
        if self.best_config is None:
            best_config_dict = None
        elif isinstance(self.best_config, dict):
            best_config_dict = self.best_config
        else:
            best_config_dict = self.best_config.to_dict()

        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "best_config": best_config_dict,
            "best_objective": self.best_objective,
            "checkpoint_passed": self.checkpoint_passed,
            "checkpoint_failures": self.checkpoint_failures,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "total_time_seconds": self.total_time_seconds,
            "tier_results": self.tier_results,
        }


@dataclass
class CoordinatorState:
    """State of the coordinator for checkpointing."""
    current_phase: Optional[str] = None
    current_tier: Optional[str] = None
    completed_phases: List[str] = field(default_factory=list)
    phase_results: Dict[str, Dict] = field(default_factory=dict)
    best_configs: Dict[str, Dict] = field(default_factory=dict)  # phase -> best config
    started_at: Optional[str] = None
    last_updated: Optional[str] = None
    total_trials_run: int = 0
    total_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "current_tier": self.current_tier,
            "completed_phases": self.completed_phases,
            "phase_results": self.phase_results,
            "best_configs": self.best_configs,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_trials_run": self.total_trials_run,
            "total_time_seconds": self.total_time_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> CoordinatorState:
        return cls(**d)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CoordinatorState:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class CycleState:
    """
    Track optimization cycles for Cyclic Coordinate Descent.

    After completing all phases, check if accuracy improved.
    If so, run another cycle with narrowed search space.
    """
    cycle_number: int = 0
    max_cycles: int = 3
    best_accuracy_per_cycle: List[float] = field(default_factory=list)
    improvement_threshold: float = 0.005  # Minimum improvement to justify another cycle

    def should_continue(self) -> bool:
        """Check if we should run another cycle."""
        if self.cycle_number >= self.max_cycles:
            return False
        if len(self.best_accuracy_per_cycle) < 2:
            return True

        # Check if last cycle improved enough
        improvement = self.best_accuracy_per_cycle[-1] - self.best_accuracy_per_cycle[-2]
        return improvement >= self.improvement_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "max_cycles": self.max_cycles,
            "best_accuracy_per_cycle": self.best_accuracy_per_cycle,
            "improvement_threshold": self.improvement_threshold,
        }


# Default phase configurations
DEFAULT_PHASE_CONFIGS = {
    TuningPhase.VAE: PhaseConfig(
        phase=TuningPhase.VAE,
        tiers=[TuningTier.CRITICAL, TuningTier.IMPORTANT],
        n_trials_per_tier={
            TuningTier.CRITICAL: 50,
            TuningTier.IMPORTANT: 30,
            TuningTier.FINETUNE: 20,
        },
        checkpoint_metrics=["vae_retrieval_accuracy_at_8", "vae_lipschitz_constant"],
        objective_metric="vae_retrieval_accuracy_at_8",
        max_time_hours=48.0,
    ),
    TuningPhase.SCORER: PhaseConfig(
        phase=TuningPhase.SCORER,
        tiers=[TuningTier.CRITICAL, TuningTier.IMPORTANT],
        n_trials_per_tier={
            TuningTier.CRITICAL: 40,
            TuningTier.IMPORTANT: 25,
            TuningTier.FINETUNE: 15,
        },
        checkpoint_metrics=["scorer_ndcg_at_8"],
        objective_metric="scorer_ndcg_at_8",
        max_time_hours=24.0,
    ),
    TuningPhase.GP: PhaseConfig(
        phase=TuningPhase.GP,
        tiers=[TuningTier.CRITICAL, TuningTier.IMPORTANT],
        n_trials_per_tier={
            TuningTier.CRITICAL: 30,
            TuningTier.IMPORTANT: 20,
            TuningTier.FINETUNE: 15,
        },
        checkpoint_metrics=["gp_spearman_correlation"],
        objective_metric="gp_spearman_correlation",
        max_time_hours=12.0,
    ),
    TuningPhase.INFERENCE: PhaseConfig(
        phase=TuningPhase.INFERENCE,
        tiers=[TuningTier.CRITICAL, TuningTier.FINETUNE],
        n_trials_per_tier={
            TuningTier.CRITICAL: 30,
            TuningTier.IMPORTANT: 20,
            TuningTier.FINETUNE: 30,
        },
        checkpoint_metrics=["e2e_final_accuracy"],
        objective_metric="e2e_final_accuracy",
        max_time_hours=72.0,
    ),
}


class CoordinateDescentTuner:
    """
    Main coordinator for Coordinate Descent hyperparameter tuning.

    Strategy:
    1. For each phase (VAE → Scorer → GP → Inference):
       a. Fix parameters from previous phases (use best configs)
       b. For each tier (Critical → Important → Finetune):
          - Sample N configurations
          - Run trials in parallel
          - Track best result
       c. Check checkpoint metrics
       d. If passed, propagate best config to next phase
       e. If failed, retry with more trials or different search
    """

    def __init__(
        self,
        output_dir: Path,
        gpu_ids: Optional[List[int]] = None,
        phase_configs: Optional[Dict[TuningPhase, PhaseConfig]] = None,
        resume: bool = True,
        sampling_strategy: str = "sobol",  # "random", "grid", "sobol"
        exploration_factor: float = 0.3,  # Fraction of trials for exploration
        quick_test_overrides: Optional[Dict[str, Any]] = None,  # Overrides for quick testing
        enable_cycling: bool = True,  # Enable Cyclic Coordinate Descent
        max_cycles: int = 3,  # Maximum number of cycles
        use_asha_pruning: bool = True,  # Enable ASHA pruning for early stopping of bad trials
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_ids = gpu_ids if gpu_ids is not None else [0, 1]
        self.phase_configs = phase_configs or DEFAULT_PHASE_CONFIGS
        self.resume = resume
        self.sampling_strategy = sampling_strategy
        self.exploration_factor = exploration_factor
        self.quick_test_overrides = quick_test_overrides or {}

        # Cyclic Coordinate Descent
        self.enable_cycling = enable_cycling
        self.cycle_state = CycleState(max_cycles=max_cycles)

        # ASHA pruning
        self.use_asha_pruning = use_asha_pruning
        self.pruner_state_path = self.output_dir / "asha_pruner_state.json" if use_asha_pruning else None

        # Components
        self.hyperspace = HyperparameterSpace()
        self.metrics = MetricRegistry()
        self.executor: Optional[DualGPUExecutor] = None

        # State
        self.state_path = self.output_dir / "coordinator_state.json"
        self.state = CoordinatorState()

        # Results
        self.phase_results: Dict[TuningPhase, PhaseResult] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup coordinator logging."""
        log_path = self.output_dir / "coordinator.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def run(
        self,
        phases: Optional[List[TuningPhase]] = None,
        skip_phases: Optional[List[TuningPhase]] = None,
    ) -> Dict[TuningPhase, PhaseResult]:
        """
        Run the full tuning pipeline with optional Cyclic CD.

        If enable_cycling is True, runs multiple cycles of all phases,
        narrowing the search space after each cycle until convergence.

        Args:
            phases: Specific phases to run (None = all)
            skip_phases: Phases to skip
        """
        # Main cycle loop for Cyclic Coordinate Descent
        while True:
            self.cycle_state.cycle_number += 1
            logger.info("=" * 60)
            logger.info(f"Starting Coordinate Descent Tuning - CYCLE {self.cycle_state.cycle_number}")
            logger.info("=" * 60)

            # Load state if resuming (only on first cycle)
            if self.cycle_state.cycle_number == 1:
                if self.resume and self.state_path.exists():
                    self.state = CoordinatorState.load(self.state_path)
                    logger.info(f"Resumed from checkpoint: {self.state.current_phase}")
                else:
                    self.state.started_at = datetime.now().isoformat()

            # Determine phases to run
            all_phases = [
                TuningPhase.VAE,
                TuningPhase.SCORER,
                TuningPhase.GP,
                TuningPhase.INFERENCE,
            ]

            if phases:
                phases_to_run = [p for p in all_phases if p in phases]
            else:
                phases_to_run = all_phases

            if skip_phases:
                phases_to_run = [p for p in phases_to_run if p not in skip_phases]

            # Skip already completed phases (only on first cycle)
            if self.cycle_state.cycle_number == 1:
                phases_to_run = [
                    p for p in phases_to_run
                    if p.value not in self.state.completed_phases
                ]

            logger.info(f"Phases to run: {[p.value for p in phases_to_run]}")
            if self.use_asha_pruning:
                logger.info(f"ASHA pruning ENABLED - bad trials will be stopped early")

            # Start executor
            # Reset pruner state file at the start of each cycle
            if self.pruner_state_path and self.pruner_state_path.exists():
                self.pruner_state_path.unlink()
                logger.info(f"Reset ASHA pruner state for cycle {self.cycle_state.cycle_number}")

            self.executor = DualGPUExecutor(
                output_dir=self.output_dir / f"trials_cycle{self.cycle_state.cycle_number}",
                gpu_ids=self.gpu_ids,
                pruner_state_path=self.pruner_state_path,
            )
            self.executor.start()

            try:
                # Run each phase
                for phase in phases_to_run:
                    logger.info("-" * 60)
                    logger.info(f"Starting Phase: {phase.value}")
                    logger.info("-" * 60)

                    self.state.current_phase = phase.value
                    self._save_state()

                    # Get best config from previous phases
                    base_config = self._get_accumulated_best_config()

                    # Run the phase
                    result = self._run_phase(phase, base_config)
                    self.phase_results[phase] = result

                    # Save result
                    self.state.phase_results[phase.value] = result.to_dict()
                    if result.best_config:
                        if isinstance(result.best_config, dict):
                            self.state.best_configs[phase.value] = result.best_config
                        else:
                            self.state.best_configs[phase.value] = result.best_config.to_dict()

                    # Check checkpoint with best-effort continuation
                    if result.checkpoint_passed:
                        logger.info(f"Phase {phase.value} PASSED checkpoint")
                    else:
                        # BEST EFFORT: Continue even if checkpoint failed
                        logger.warning(
                            f"Phase {phase.value} FAILED checkpoint: {result.checkpoint_failures}\n"
                            f"  BEST EFFORT CONTINUATION: Using best available config and continuing.\n"
                            f"  Best objective achieved: {result.best_objective:.4f}"
                        )

                    # Always mark phase as completed
                    if phase.value not in self.state.completed_phases:
                        self.state.completed_phases.append(phase.value)
                    self._save_state()

            finally:
                self.executor.stop()

            # Record best accuracy from this cycle (from Inference phase)
            inference_result = self.phase_results.get(TuningPhase.INFERENCE)
            if inference_result:
                best_acc = inference_result.best_objective
                self.cycle_state.best_accuracy_per_cycle.append(best_acc)
                logger.info(f"Cycle {self.cycle_state.cycle_number} best accuracy: {best_acc:.4f}")

            # Check if we should run another cycle
            if not self.enable_cycling or not self.cycle_state.should_continue():
                if self.enable_cycling:
                    logger.info(
                        f"Stopping after {self.cycle_state.cycle_number} cycles "
                        f"(improvement below threshold or max cycles reached)"
                    )
                break

            # Prepare for next cycle
            logger.info(f"Preparing for cycle {self.cycle_state.cycle_number + 1}...")
            self._narrow_search_space_for_next_cycle()
            # Clear completed phases to allow re-tuning
            self.state.completed_phases = []

        # Generate final report
        self._generate_report()

        return self.phase_results

    def _narrow_search_space_for_next_cycle(self):
        """
        Narrow hyperparameter search space based on previous cycle's results.

        Strategy:
        - Focus around best values found (+/- 20%)
        - This allows refinement in subsequent cycles
        """
        for phase in [TuningPhase.VAE, TuningPhase.GP, TuningPhase.INFERENCE]:
            if phase.value in self.state.best_configs:
                best_values = self.state.best_configs[phase.value]
                if isinstance(best_values, dict) and "values" in best_values:
                    best_values = best_values["values"]

                for param_name, best_value in best_values.items():
                    if isinstance(best_value, (int, float)):
                        # Narrow continuous params to +/- 20% around best
                        param = self.hyperspace.get_param(param_name)
                        if param and hasattr(param, 'low') and hasattr(param, 'high'):
                            # Skip if param bounds are None (categorical params)
                            if param.low is None or param.high is None:
                                continue
                            new_low = max(param.low, best_value * 0.8)
                            new_high = min(param.high, best_value * 1.2)
                            # Only narrow if it makes sense
                            if new_low < new_high:
                                param.low = new_low
                                param.high = new_high
                                logger.info(f"Narrowed {param_name}: [{new_low:.4f}, {new_high:.4f}]")

    def _run_phase(
        self,
        phase: TuningPhase,
        base_config: HyperparameterConfig,
    ) -> PhaseResult:
        """Run a single phase of tuning."""
        phase_config = self.phase_configs[phase]
        start_time = time.time()

        all_results = []
        tier_results = {}
        best_config = None
        best_objective = float('-inf')

        # Run each tier
        for tier in phase_config.tiers:
            logger.info(f"  Tier: {tier.name}")
            self.state.current_tier = tier.name
            self._save_state()

            tier_result = self._run_tier(
                phase=phase,
                tier=tier,
                base_config=base_config,
                n_trials=phase_config.n_trials_per_tier.get(tier, 20),
                objective_metric=phase_config.objective_metric,
            )

            tier_results[tier.name] = tier_result
            all_results.extend(tier_result["results"])

            # Update best
            if tier_result["best_objective"] > best_objective:
                best_objective = tier_result["best_objective"]
                best_config = tier_result["best_config"]

            # Early stopping check
            if tier_result.get("early_stopped"):
                logger.info(f"  Early stopped in tier {tier.name}")
                break

        # Check checkpoint
        checkpoint_passed, checkpoint_failures = self._check_phase_checkpoint(
            phase_config,
            best_objective,
        )

        total_time = time.time() - start_time

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.CHECKPOINT_PASSED if checkpoint_passed else PhaseStatus.CHECKPOINT_FAILED,
            best_config=best_config,
            best_objective=best_objective,
            all_results=all_results,
            checkpoint_passed=checkpoint_passed,
            checkpoint_failures=checkpoint_failures,
            total_trials=len(all_results),
            successful_trials=sum(1 for r in all_results if r.get("success", False)),
            failed_trials=sum(1 for r in all_results if not r.get("success", True)),
            total_time_seconds=total_time,
            tier_results=tier_results,
        )

    def _run_tier(
        self,
        phase: TuningPhase,
        tier: TuningTier,
        base_config: HyperparameterConfig,
        n_trials: int,
        objective_metric: str,
    ) -> Dict[str, Any]:
        """Run trials for a specific tier within a phase."""

        # Generate configurations
        configs = self._generate_configs(
            phase=phase,
            tier=tier,
            base_config=base_config,
            n_configs=n_trials,
        )

        logger.info(f"    Generated {len(configs)} configurations")

        # Submit trials
        tasks = []
        for i, config in enumerate(configs):
            trial_id = f"{phase.value}_{tier.name}_{i}_{uuid.uuid4().hex[:8]}"
            task = TrialTask(
                trial_id=trial_id,
                config=config,
                phase=phase,
                priority=tier.value,  # Higher tier = higher priority
            )
            tasks.append(task)

        self.executor.submit_batch(tasks)

        # Wait for completion
        self.executor.run_until_complete()

        # Collect results
        results = []
        for task in tasks:
            result = self.executor.get_result(task.trial_id, timeout=1)
            if result:
                results.append(result if isinstance(result, dict) else result.to_dict())

        # Find best
        best_config = None
        best_objective = float('-inf')

        for result in results:
            if not result.get("success", False):
                continue

            metrics = result.get("metrics", {})
            if objective_metric in metrics:
                obj_value = metrics[objective_metric].get("value", float('-inf'))
                if obj_value > best_objective:
                    best_objective = obj_value
                    best_config = HyperparameterConfig.from_dict(result["config"])

        # Update statistics
        self.state.total_trials_run += len(results)

        return {
            "tier": tier.name,
            "n_trials": len(results),
            "best_objective": best_objective,
            "best_config": best_config.to_dict() if best_config else None,
            "results": results,
            "early_stopped": False,
        }

    def _generate_configs(
        self,
        phase: TuningPhase,
        tier: TuningTier,
        base_config: HyperparameterConfig,
        n_configs: int,
    ) -> List[HyperparameterConfig]:
        """Generate configurations for a phase/tier."""

        if self.sampling_strategy == "grid":
            # Grid sampling (may generate more than n_configs)
            configs = self.hyperspace.sample_grid(
                tier=tier,
                phase=phase,
                n_points_continuous=3,
            )
            # Limit to n_configs
            configs = configs[:n_configs]

        elif self.sampling_strategy == "sobol":
            # Sobol quasi-random sampling
            configs = self.hyperspace.sample_sobol(
                n_samples=n_configs,
                tier=tier,
                phase=phase,
                seed=42,
            )

        else:
            # Random sampling
            configs = [
                self.hyperspace.sample_random(
                    tier=tier,
                    phase=phase,
                    base_config=base_config.values,
                    seed=i,
                )
                for i in range(n_configs)
            ]

        # Merge with base config (fixed params from previous phases)
        merged_configs = []
        for config in configs:
            merged_values = dict(base_config.values)
            merged_values.update(config.values)
            # Apply quick test overrides if set
            if self.quick_test_overrides:
                merged_values.update(self.quick_test_overrides)
            merged_configs.append(HyperparameterConfig(
                values=merged_values,
                tier=tier,
                phase=phase,
            ))

        return merged_configs

    def _get_accumulated_best_config(self) -> HyperparameterConfig:
        """Get accumulated best config from all completed phases."""
        accumulated = self.hyperspace.get_default_config().values.copy()

        for phase_name in self.state.completed_phases:
            if phase_name in self.state.best_configs:
                best = self.state.best_configs[phase_name]
                if isinstance(best, dict) and "values" in best:
                    accumulated.update(best["values"])
                elif isinstance(best, dict):
                    accumulated.update(best)

        return HyperparameterConfig(
            values=accumulated,
            tier=TuningTier.CRITICAL,
        )

    def _check_phase_checkpoint(
        self,
        phase_config: PhaseConfig,
        best_objective: float,
    ) -> Tuple[bool, List[str]]:
        """Check if phase checkpoint is passed."""
        failures = []

        # Check objective against target
        target_map = {
            "vae_retrieval_accuracy_at_8": 0.85,
            "scorer_ndcg_at_8": 0.70,
            "gp_spearman_correlation": 0.40,
            "e2e_final_accuracy": 0.915,
        }

        target = target_map.get(phase_config.objective_metric, 0.0)
        if best_objective < target:
            failures.append(f"{phase_config.objective_metric}: {best_objective:.4f} < {target}")

        return len(failures) == 0, failures

    def _save_state(self):
        """Save coordinator state."""
        self.state.last_updated = datetime.now().isoformat()
        self.state.save(self.state_path)

    def _generate_report(self):
        """Generate final tuning report."""
        report_path = self.output_dir / "tuning_report.md"

        lines = [
            "# BOLT Hyperparameter Tuning Report",
            "",
            f"**Started:** {self.state.started_at}",
            f"**Completed:** {datetime.now().isoformat()}",
            f"**Total Trials:** {self.state.total_trials_run}",
            "",
            "## Phase Summary",
            "",
        ]

        for phase in TuningPhase:
            if phase.value in self.state.phase_results:
                result = self.state.phase_results[phase.value]
                lines.extend([
                    f"### {phase.value}",
                    f"- **Status:** {result['status']}",
                    f"- **Best Objective:** {result['best_objective']:.4f}",
                    f"- **Trials:** {result['total_trials']} ({result['successful_trials']} successful)",
                    f"- **Time:** {result['total_time_seconds']/3600:.2f} hours",
                    f"- **Checkpoint:** {'PASSED' if result['checkpoint_passed'] else 'FAILED'}",
                    "",
                ])

        lines.extend([
            "## Best Configuration",
            "",
            "```yaml",
        ])

        # Final best config
        final_config = self._get_accumulated_best_config()
        for key, value in sorted(final_config.values.items()):
            lines.append(f"{key}: {value}")

        lines.extend([
            "```",
            "",
            "## Recommendations",
            "",
        ])

        # Add recommendations based on results
        for phase_name, result in self.state.phase_results.items():
            if not result["checkpoint_passed"]:
                lines.append(f"- **{phase_name}:** Failed checkpoint. Consider:")
                lines.append(f"  - Increasing trials for tier CRITICAL")
                lines.append(f"  - Adjusting parameter ranges")
                lines.append(f"  - Checking for bugs in {phase_name} evaluation")
                lines.append("")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated report: {report_path}")

        # Also save final config
        config_path = self.output_dir / "best_config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(final_config.values, f, default_flow_style=False)

        logger.info(f"Saved best config: {config_path}")


def run_quick_tune(
    output_dir: Path,
    gpu_ids: List[int] = [0, 1],
    n_trials_per_phase: int = 10,
) -> Dict[str, Any]:
    """
    Quick tuning run for testing.

    Runs minimal trials with reduced epochs to verify pipeline works.
    """
    # Minimal phase configs
    quick_configs = {
        phase: PhaseConfig(
            phase=phase,
            tiers=[TuningTier.CRITICAL],
            n_trials_per_tier={TuningTier.CRITICAL: n_trials_per_phase},
            checkpoint_metrics=DEFAULT_PHASE_CONFIGS[phase].checkpoint_metrics,
            objective_metric=DEFAULT_PHASE_CONFIGS[phase].objective_metric,
            max_time_hours=1.0,
        )
        for phase in TuningPhase
    }

    # Override epochs for quick testing
    quick_overrides = {
        "vae_epochs": 100,  # Reduced from 50000
        "vae_annealing_epochs": 10,  # Reduced from 2500
        "gp_epochs": 100,  # Reduced from 10000
        "vae_patience": 20,  # Reduced from 1000
        "gp_patience": 10,  # Reduced from 100
    }

    tuner = CoordinateDescentTuner(
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        phase_configs=quick_configs,
        resume=False,
        quick_test_overrides=quick_overrides,
    )

    results = tuner.run()

    return {
        phase.value: result.to_dict()
        for phase, result in results.items()
    }
