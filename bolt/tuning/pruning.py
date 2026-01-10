"""
ASHA-style Pruning for BOLT Hyperparameter Tuning

Asynchronous Successive Halving Algorithm (ASHA) for early stopping
of underperforming trials. Saves 30-50% GPU time by killing bad trials early.

Features:
- Rung-based evaluation (e.g., at epochs 100, 500, 1000, 2500, 5000)
- Prunes bottom 50% at each rung
- Thread-safe for parallel execution
- Grace period before pruning starts

Usage:
    pruner = ASHAPruner(rungs=[100, 500, 1000])

    # In training loop:
    for epoch in range(epochs):
        train_step()
        if pruner.should_stop(trial_id, epoch, val_loss):
            break

    pruner.mark_completed(trial_id)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class TrialReport:
    """Intermediate report from a trial."""
    trial_id: str
    step: int  # epoch or iteration
    metric_name: str
    metric_value: float
    timestamp: float = field(default_factory=time.time)


class ASHAPruner:
    """
    ASHA (Asynchronous Successive Halving) pruner.

    Prunes trials in the bottom fraction at each rung.

    Args:
        reduction_factor: Fraction to keep at each rung (default 2.0 = keep top 50%)
        grace_period: Minimum steps before pruning starts
        rungs: Steps at which to evaluate and potentially prune
        direction: "maximize" or "minimize"
        min_trials_for_pruning: Minimum trials at a rung before pruning decision
    """

    def __init__(
        self,
        reduction_factor: float = 2.0,  # eta: prune bottom 50%
        grace_period: int = 100,  # Minimum steps before pruning
        rungs: Optional[List[int]] = None,  # Steps at which to prune
        direction: str = "maximize",  # "maximize" or "minimize"
        min_trials_for_pruning: int = 3,  # Need at least 3 trials to compare
    ):
        self.reduction_factor = reduction_factor
        self.grace_period = grace_period
        self.direction = direction
        self.min_trials_for_pruning = min_trials_for_pruning

        # Default rungs for VAE training (100 epochs quick, 50k full)
        self.rungs = rungs or [100, 500, 1000, 2500, 5000, 10000, 25000]

        # Thread-safe state
        self._lock = threading.Lock()
        self._reports: Dict[str, List[TrialReport]] = {}  # trial_id -> reports
        self._pruned: set = set()  # trial_ids that have been pruned
        self._completed: set = set()  # trial_ids that completed

        # Statistics
        self._prune_decisions: List[Dict] = []  # Log of pruning decisions

    def report(
        self,
        trial_id: str,
        step: int,
        metric_value: float,
        metric_name: str = "val_loss",
    ) -> bool:
        """
        Report intermediate metric from a trial.

        Args:
            trial_id: Unique identifier for the trial
            step: Current training step (epoch)
            metric_value: Current metric value
            metric_name: Name of the metric

        Returns:
            True if trial should continue, False if pruned
        """
        report = TrialReport(
            trial_id=trial_id,
            step=step,
            metric_name=metric_name,
            metric_value=metric_value,
        )

        with self._lock:
            if trial_id in self._pruned:
                return False

            if trial_id not in self._reports:
                self._reports[trial_id] = []
            self._reports[trial_id].append(report)

            # Check if at a rung
            if step in self.rungs and step >= self.grace_period:
                should_prune = self._should_prune_at_rung(trial_id, step, metric_value)

                if should_prune:
                    self._pruned.add(trial_id)
                    logger.info(
                        f"PRUNED trial {trial_id} at step {step} "
                        f"(value={metric_value:.4f}, reason: bottom {int(100/self.reduction_factor)}%)"
                    )
                    return False

            return True

    def should_stop(
        self,
        trial_id: str,
        step: int,
        metric_value: float,
        metric_name: str = "val_loss",
    ) -> bool:
        """
        Check if trial should stop (convenience method).

        Returns True if trial should stop, False if should continue.
        """
        should_continue = self.report(trial_id, step, metric_value, metric_name)
        return not should_continue

    def _should_prune_at_rung(self, trial_id: str, step: int, value: float) -> bool:
        """Check if trial should be pruned at current rung."""
        # Get all trials that have reported at this rung
        rung_values = []
        for tid, reports in self._reports.items():
            if tid in self._pruned or tid in self._completed:
                continue
            # Find report at this rung (or closest step <= rung)
            for r in reversed(reports):  # Start from most recent
                if r.step == step:
                    rung_values.append((tid, r.metric_value))
                    break

        if len(rung_values) < self.min_trials_for_pruning:
            # Not enough trials to make pruning decision
            return False

        # Sort by metric value
        if self.direction == "maximize":
            rung_values.sort(key=lambda x: -x[1])  # Higher is better
        else:
            rung_values.sort(key=lambda x: x[1])  # Lower is better

        # Determine cutoff
        keep_fraction = 1.0 / self.reduction_factor
        cutoff_idx = max(1, int(len(rung_values) * keep_fraction))

        # Check if trial is in bottom half
        bottom_half_ids = {x[0] for x in rung_values[cutoff_idx:]}

        # Log decision
        self._prune_decisions.append({
            "step": step,
            "trial_id": trial_id,
            "value": value,
            "rung_values": rung_values,
            "cutoff_idx": cutoff_idx,
            "pruned": trial_id in bottom_half_ids,
        })

        return trial_id in bottom_half_ids

    def mark_completed(self, trial_id: str):
        """Mark a trial as completed (not pruned)."""
        with self._lock:
            self._completed.add(trial_id)
            if trial_id in self._pruned:
                self._pruned.remove(trial_id)

    def is_pruned(self, trial_id: str) -> bool:
        """Check if trial was pruned."""
        with self._lock:
            return trial_id in self._pruned

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        with self._lock:
            return {
                "total_reported": len(self._reports),
                "pruned": len(self._pruned),
                "completed": len(self._completed),
                "active": len(self._reports) - len(self._pruned) - len(self._completed),
                "prune_rate": len(self._pruned) / max(1, len(self._reports)),
                "decisions": len(self._prune_decisions),
            }

    def reset(self):
        """Reset pruner state (for new phase)."""
        with self._lock:
            self._reports.clear()
            self._pruned.clear()
            self._completed.clear()
            self._prune_decisions.clear()


class PruningCallback:
    """
    Callback for integration with training loops.

    Can be passed to trainers that support callbacks.
    """

    def __init__(
        self,
        pruner: ASHAPruner,
        trial_id: str,
        metric_name: str = "val_loss",
        check_interval: int = 100,
    ):
        self.pruner = pruner
        self.trial_id = trial_id
        self.metric_name = metric_name
        self.check_interval = check_interval
        self._should_stop = False

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Call during training to report and check pruning.

        Args:
            epoch: Current epoch
            metrics: Dict of metric name -> value

        Returns:
            True if training should stop (pruned)
        """
        if self._should_stop:
            return True

        if epoch % self.check_interval != 0:
            return False

        metric_value = metrics.get(self.metric_name, 0.0)
        should_continue = self.pruner.report(
            self.trial_id, epoch, metric_value, self.metric_name
        )

        self._should_stop = not should_continue
        return self._should_stop

    @property
    def should_stop(self) -> bool:
        return self._should_stop


class MedianPruner:
    """
    Simple median-based pruner.

    Prunes trials that are below median of completed trials at same step.
    Simpler than ASHA, good for quick experiments.
    """

    def __init__(
        self,
        n_warmup_steps: int = 100,
        n_min_trials: int = 3,
        direction: str = "maximize",
    ):
        self.n_warmup_steps = n_warmup_steps
        self.n_min_trials = n_min_trials
        self.direction = direction

        self._lock = threading.Lock()
        self._step_values: Dict[int, List[float]] = {}  # step -> list of values
        self._pruned: set = set()

    def report(self, trial_id: str, step: int, value: float) -> bool:
        """Report and check if should prune."""
        with self._lock:
            if trial_id in self._pruned:
                return False

            # Store value at this step
            if step not in self._step_values:
                self._step_values[step] = []
            self._step_values[step].append(value)

            # Don't prune during warmup
            if step < self.n_warmup_steps:
                return True

            # Need enough trials
            if len(self._step_values.get(step, [])) < self.n_min_trials:
                return True

            # Check against median
            values = sorted(self._step_values[step])
            median = values[len(values) // 2]

            if self.direction == "maximize" and value < median:
                self._pruned.add(trial_id)
                logger.info(f"PRUNED trial {trial_id} at step {step} (below median)")
                return False
            elif self.direction == "minimize" and value > median:
                self._pruned.add(trial_id)
                logger.info(f"PRUNED trial {trial_id} at step {step} (above median)")
                return False

            return True
