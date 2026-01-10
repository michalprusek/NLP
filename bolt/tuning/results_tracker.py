"""
Results Tracker for BOLT Hyperparameter Tuning

Comprehensive experiment tracking with:
- Trial history with full metadata
- Metric trends and analysis
- Best configurations per phase
- Comparison utilities
- Export to various formats (JSON, CSV, Parquet)
- Visualization helpers
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .hyperspace import HyperparameterConfig, TuningPhase, TuningTier
from .metrics import MetricResult, MetricCategory


@dataclass
class TrialRecord:
    """Complete record of a single trial."""
    trial_id: str
    phase: str
    tier: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    success: bool
    objective_value: float
    objective_name: str
    checkpoint_passed: bool
    timestamp: str
    duration_seconds: float
    gpu_id: int
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "phase": self.phase,
            "tier": self.tier,
            "config": self.config,
            "metrics": self.metrics,
            "success": self.success,
            "objective_value": self.objective_value,
            "objective_name": self.objective_name,
            "checkpoint_passed": self.checkpoint_passed,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "gpu_id": self.gpu_id,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrialRecord:
        return cls(**d)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten config and metrics into single dict for CSV/DataFrame."""
        flat = {
            "trial_id": self.trial_id,
            "phase": self.phase,
            "tier": self.tier,
            "success": self.success,
            "objective_value": self.objective_value,
            "objective_name": self.objective_name,
            "checkpoint_passed": self.checkpoint_passed,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "gpu_id": self.gpu_id,
        }

        # Flatten config
        for k, v in self.config.items():
            flat[f"config_{k}"] = v

        # Flatten metrics
        for k, v in self.metrics.items():
            flat[f"metric_{k}"] = v

        return flat


class ResultsTracker:
    """
    Central tracker for all tuning results.

    Features:
    - SQLite backend for efficient querying
    - JSON export for portability
    - CSV export for analysis
    - Metric aggregation and trends
    - Best config tracking
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database
        self.db_path = self.output_dir / "results.db"
        self._init_db()

        # In-memory cache for fast access
        self._cache: Dict[str, TrialRecord] = {}
        self._best_per_phase: Dict[str, TrialRecord] = {}

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                tier TEXT NOT NULL,
                config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                success INTEGER NOT NULL,
                objective_value REAL NOT NULL,
                objective_name TEXT NOT NULL,
                checkpoint_passed INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                gpu_id INTEGER NOT NULL,
                error_message TEXT
            )
        """)

        # Metrics time series table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
            )
        """)

        # Best configs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS best_configs (
                phase TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL,
                objective_value REAL NOT NULL,
                config TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_tier ON trials(tier)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_objective ON trials(objective_value)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_history_trial ON metric_history(trial_id)")

        conn.commit()
        conn.close()

    def add_trial(self, record: TrialRecord):
        """Add a trial record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert trial
        cursor.execute("""
            INSERT OR REPLACE INTO trials
            (trial_id, phase, tier, config, metrics, success, objective_value,
             objective_name, checkpoint_passed, timestamp, duration_seconds, gpu_id, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.trial_id,
            record.phase,
            record.tier,
            json.dumps(record.config),
            json.dumps(record.metrics),
            int(record.success),
            record.objective_value,
            record.objective_name,
            int(record.checkpoint_passed),
            record.timestamp,
            record.duration_seconds,
            record.gpu_id,
            record.error_message,
        ))

        # Insert metrics into history
        for metric_name, metric_value in record.metrics.items():
            cursor.execute("""
                INSERT INTO metric_history (trial_id, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            """, (record.trial_id, metric_name, metric_value, record.timestamp))

        # Update best config if applicable
        if record.success:
            cursor.execute("""
                SELECT objective_value FROM best_configs WHERE phase = ?
            """, (record.phase,))
            row = cursor.fetchone()

            if row is None or record.objective_value > row[0]:
                cursor.execute("""
                    INSERT OR REPLACE INTO best_configs
                    (phase, trial_id, objective_value, config, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    record.phase,
                    record.trial_id,
                    record.objective_value,
                    json.dumps(record.config),
                    datetime.now().isoformat(),
                ))

        conn.commit()
        conn.close()

        # Update cache
        self._cache[record.trial_id] = record

    def get_trial(self, trial_id: str) -> Optional[TrialRecord]:
        """Get a specific trial."""
        if trial_id in self._cache:
            return self._cache[trial_id]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_record(row)
        return None

    def get_trials_by_phase(self, phase: str) -> List[TrialRecord]:
        """Get all trials for a phase."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM trials WHERE phase = ? ORDER BY timestamp",
            (phase,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_record(row) for row in rows]

    def get_trials_by_tier(self, phase: str, tier: str) -> List[TrialRecord]:
        """Get trials for a specific phase and tier."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM trials WHERE phase = ? AND tier = ? ORDER BY timestamp",
            (phase, tier)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_record(row) for row in rows]

    def get_best_trial(self, phase: str) -> Optional[TrialRecord]:
        """Get best trial for a phase."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT trial_id FROM best_configs WHERE phase = ?
        """, (phase,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self.get_trial(row[0])
        return None

    def get_best_config(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get best configuration for a phase."""
        trial = self.get_best_trial(phase)
        if trial:
            return trial.config
        return None

    def get_metric_history(
        self,
        metric_name: str,
        phase: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """Get history of a metric over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if phase:
            cursor.execute("""
                SELECT mh.trial_id, mh.metric_value, mh.timestamp
                FROM metric_history mh
                JOIN trials t ON mh.trial_id = t.trial_id
                WHERE mh.metric_name = ? AND t.phase = ?
                ORDER BY mh.timestamp
            """, (metric_name, phase))
        else:
            cursor.execute("""
                SELECT trial_id, metric_value, timestamp
                FROM metric_history
                WHERE metric_name = ?
                ORDER BY timestamp
            """, (metric_name,))

        rows = cursor.fetchall()
        conn.close()

        return [(row[0], row[1], row[2]) for row in rows]

    def get_statistics(self, phase: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregate statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_clause = "WHERE phase = ?" if phase else ""
        params = (phase,) if phase else ()

        # Total counts
        cursor.execute(f"SELECT COUNT(*) FROM trials {where_clause}", params)
        total = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM trials {where_clause} AND success = 1" if phase else "SELECT COUNT(*) FROM trials WHERE success = 1", params if phase else ())
        successful = cursor.fetchone()[0]

        # Time statistics
        cursor.execute(f"SELECT SUM(duration_seconds), AVG(duration_seconds) FROM trials {where_clause}", params)
        time_row = cursor.fetchone()
        total_time = time_row[0] or 0
        avg_time = time_row[1] or 0

        # Objective statistics
        cursor.execute(f"SELECT MAX(objective_value), MIN(objective_value), AVG(objective_value) FROM trials {where_clause} AND success = 1" if phase else "SELECT MAX(objective_value), MIN(objective_value), AVG(objective_value) FROM trials WHERE success = 1", params if phase else ())
        obj_row = cursor.fetchone()

        conn.close()

        return {
            "total_trials": total,
            "successful_trials": successful,
            "failed_trials": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_time_hours": total_time / 3600,
            "avg_time_seconds": avg_time,
            "best_objective": obj_row[0] if obj_row[0] else None,
            "worst_objective": obj_row[1] if obj_row[1] else None,
            "avg_objective": obj_row[2] if obj_row[2] else None,
        }

    def get_parameter_importance(
        self,
        phase: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Estimate parameter importance using correlation with objective.

        Simple approach: correlation between parameter value and objective.
        """
        trials = self.get_trials_by_phase(phase)
        if len(trials) < 10:
            return []

        # Collect parameter values and objectives
        param_values: Dict[str, List[float]] = {}
        objectives = []

        for trial in trials:
            if not trial.success:
                continue

            objectives.append(trial.objective_value)

            for param, value in trial.config.items():
                if isinstance(value, (int, float)):
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(float(value))

        if not objectives:
            return []

        # Compute correlations
        importances = []
        objectives_arr = np.array(objectives)

        for param, values in param_values.items():
            if len(values) != len(objectives):
                continue
            values_arr = np.array(values)

            # Spearman correlation
            from scipy import stats
            corr, _ = stats.spearmanr(values_arr, objectives_arr)

            if not np.isnan(corr):
                importances.append((param, abs(corr)))

        # Sort by importance
        importances.sort(key=lambda x: -x[1])

        return importances[:top_k]

    def export_to_json(self, path: Optional[Path] = None) -> Path:
        """Export all results to JSON."""
        path = path or (self.output_dir / "results_export.json")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trials ORDER BY timestamp")
        rows = cursor.fetchall()
        conn.close()

        records = [self._row_to_record(row).to_dict() for row in rows]

        with open(path, "w") as f:
            json.dump({
                "exported_at": datetime.now().isoformat(),
                "total_trials": len(records),
                "trials": records,
            }, f, indent=2)

        return path

    def export_to_csv(self, path: Optional[Path] = None) -> Path:
        """Export all results to CSV (flattened)."""
        path = path or (self.output_dir / "results_export.csv")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trials ORDER BY timestamp")
        rows = cursor.fetchall()
        conn.close()

        records = [self._row_to_record(row).to_flat_dict() for row in rows]

        if not records:
            return path

        # Get all possible columns
        all_columns = set()
        for record in records:
            all_columns.update(record.keys())
        columns = sorted(all_columns)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

        return path

    def _row_to_record(self, row: tuple) -> TrialRecord:
        """Convert database row to TrialRecord."""
        return TrialRecord(
            trial_id=row[0],
            phase=row[1],
            tier=row[2],
            config=json.loads(row[3]),
            metrics=json.loads(row[4]),
            success=bool(row[5]),
            objective_value=row[6],
            objective_name=row[7],
            checkpoint_passed=bool(row[8]),
            timestamp=row[9],
            duration_seconds=row[10],
            gpu_id=row[11],
            error_message=row[12],
        )

    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        lines = [
            "=" * 60,
            "BOLT Hyperparameter Tuning Results Summary",
            "=" * 60,
            "",
        ]

        # Overall stats
        overall = self.get_statistics()
        lines.extend([
            "Overall Statistics:",
            f"  Total Trials: {overall['total_trials']}",
            f"  Success Rate: {overall['success_rate']*100:.1f}%",
            f"  Total Time: {overall['total_time_hours']:.1f} hours",
            f"  Best Objective: {overall['best_objective']:.4f}" if overall['best_objective'] else "",
            "",
        ])

        # Per-phase stats
        for phase in ["vae", "scorer", "gp", "inference"]:
            stats = self.get_statistics(phase)
            if stats['total_trials'] == 0:
                continue

            best = self.get_best_trial(phase)

            lines.extend([
                f"{phase.upper()} Phase:",
                f"  Trials: {stats['total_trials']} ({stats['success_rate']*100:.1f}% success)",
                f"  Best: {stats['best_objective']:.4f}" if stats['best_objective'] else "  Best: N/A",
                f"  Time: {stats['total_time_hours']:.2f} hours",
            ])

            if best:
                # Show top 3 most important params for this phase
                importance = self.get_parameter_importance(phase, top_k=3)
                if importance:
                    lines.append("  Top Parameters:")
                    for param, imp in importance:
                        lines.append(f"    - {param}: correlation={imp:.3f}")

            lines.append("")

        return "\n".join(lines)
