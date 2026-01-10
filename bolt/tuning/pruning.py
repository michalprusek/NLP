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

import fcntl
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
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


class SharedASHAPruner:
    """
    ASHA Pruner with file-based shared state for multi-process execution.

    Uses file locking to ensure safe concurrent access from multiple processes.
    Each process reads/writes to a shared JSON file to coordinate pruning decisions.

    Usage:
        # In coordinator (creates the state file):
        pruner = SharedASHAPruner(state_path="/path/to/pruner_state.json")

        # In each trial process:
        pruner = SharedASHAPruner(state_path="/path/to/pruner_state.json")
        should_stop = pruner.should_stop(trial_id, epoch, metric_value)
    """

    def __init__(
        self,
        state_path: Path,
        reduction_factor: float = 2.0,
        grace_period: int = 100,
        rungs: Optional[List[int]] = None,
        direction: str = "maximize",
        min_trials_for_pruning: int = 3,
    ):
        # Validate inputs
        if reduction_factor <= 0:
            raise ValueError(f"reduction_factor must be positive, got {reduction_factor}")
        if grace_period < 0:
            raise ValueError(f"grace_period must be non-negative, got {grace_period}")
        if min_trials_for_pruning < 1:
            raise ValueError(f"min_trials_for_pruning must be >= 1, got {min_trials_for_pruning}")
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")

        self.state_path = Path(state_path)
        self.reduction_factor = reduction_factor
        self.grace_period = grace_period
        self.direction = direction
        self.min_trials_for_pruning = min_trials_for_pruning
        self.rungs = rungs or [100, 500, 1000, 2500, 5000, 10000, 25000]

        # Initialize state file if it doesn't exist
        if not self.state_path.exists():
            self._write_state({
                "reports": {},  # trial_id -> list of {step, value, timestamp}
                "pruned": [],   # list of pruned trial_ids
                "completed": [],  # list of completed trial_ids
                "decisions": [],  # pruning decision log
            })

    def _read_state(self) -> Dict[str, Any]:
        """Read state from file with locking."""
        empty_state = {
            "reports": {},
            "pruned": [],
            "completed": [],
            "decisions": [],
        }
        try:
            with open(self.state_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except FileNotFoundError:
            logger.debug(f"ASHA state file not found at {self.state_path}, using fresh state")
            return empty_state
        except json.JSONDecodeError as e:
            logger.error(
                f"ASHA state file at {self.state_path} is corrupted "
                f"(line {e.lineno}, col {e.colno}): {e.msg}. "
                f"Using fresh state - existing pruning decisions may be lost."
            )
            return empty_state

    def _write_state(self, state: Dict[str, Any]):
        """Write state to file with locking."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _atomic_update(self, update_fn: Callable[[Dict], Dict]) -> Dict[str, Any]:
        """Atomically read, update, and write state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Open for read+write, create if doesn't exist
        fd = os.open(str(self.state_path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                # Read current state
                with os.fdopen(os.dup(fd), "r") as f:
                    f.seek(0)
                    content = f.read()
                    if content:
                        state = json.loads(content)
                    else:
                        state = {
                            "reports": {},
                            "pruned": [],
                            "completed": [],
                            "decisions": [],
                        }

                # Apply update
                new_state = update_fn(state)

                # Write back
                with os.fdopen(os.dup(fd), "w") as f:
                    f.seek(0)
                    f.truncate()
                    json.dump(new_state, f, indent=2)

                return new_state
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def report(
        self,
        trial_id: str,
        step: int,
        metric_value: float,
        metric_name: str = "val_loss",
    ) -> bool:
        """
        Report intermediate metric and check if trial should continue.

        Returns:
            True if trial should continue, False if pruned
        """
        def update(state):
            # Check if already pruned
            if trial_id in state["pruned"]:
                return state

            # Add report
            if trial_id not in state["reports"]:
                state["reports"][trial_id] = []

            state["reports"][trial_id].append({
                "step": step,
                "value": metric_value,
                "metric_name": metric_name,
                "timestamp": time.time(),
            })

            # Check if at a rung
            if step in self.rungs and step >= self.grace_period:
                should_prune = self._should_prune_at_rung(state, trial_id, step, metric_value)
                if should_prune:
                    state["pruned"].append(trial_id)
                    state["decisions"].append({
                        "step": step,
                        "trial_id": trial_id,
                        "value": metric_value,
                        "action": "pruned",
                        "timestamp": time.time(),
                    })
                    logger.info(
                        f"PRUNED trial {trial_id} at step {step} "
                        f"(value={metric_value:.4f}, bottom {int(100/self.reduction_factor)}%)"
                    )

            return state

        new_state = self._atomic_update(update)
        return trial_id not in new_state["pruned"]

    def _should_prune_at_rung(
        self,
        state: Dict[str, Any],
        trial_id: str,
        step: int,
        value: float,
    ) -> bool:
        """Check if trial should be pruned at current rung."""
        # Get all trials that have reported at this rung
        rung_values = []
        for tid, reports in state["reports"].items():
            if tid in state["pruned"] or tid in state["completed"]:
                continue
            # Find report at this rung
            for r in reversed(reports):
                if r["step"] == step:
                    rung_values.append((tid, r["value"]))
                    break

        if len(rung_values) < self.min_trials_for_pruning:
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

        return trial_id in bottom_half_ids

    def should_stop(
        self,
        trial_id: str,
        step: int,
        metric_value: float,
        metric_name: str = "val_loss",
    ) -> bool:
        """Check if trial should stop (convenience method)."""
        should_continue = self.report(trial_id, step, metric_value, metric_name)
        return not should_continue

    def mark_completed(self, trial_id: str):
        """Mark a trial as completed."""
        def update(state):
            if trial_id not in state["completed"]:
                state["completed"].append(trial_id)
            if trial_id in state["pruned"]:
                state["pruned"].remove(trial_id)
            return state

        self._atomic_update(update)

    def is_pruned(self, trial_id: str) -> bool:
        """Check if trial was pruned."""
        state = self._read_state()
        return trial_id in state["pruned"]

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        state = self._read_state()
        total = len(state["reports"])
        pruned = len(state["pruned"])
        completed = len(state["completed"])
        return {
            "total_reported": total,
            "pruned": pruned,
            "completed": completed,
            "active": total - pruned - completed,
            "prune_rate": pruned / max(1, total),
            "decisions": len(state["decisions"]),
            "state_path": str(self.state_path),
        }

    def reset(self):
        """Reset pruner state."""
        self._write_state({
            "reports": {},
            "pruned": [],
            "completed": [],
            "decisions": [],
        })
