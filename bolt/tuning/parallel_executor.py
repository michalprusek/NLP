"""
Parallel Executor for Dual-GPU Hyperparameter Tuning

Features:
- Dual-GPU management (GPU 0 and GPU 1)
- Process-based isolation for memory safety
- Automatic retry on OOM
- Load balancing
- Async result collection
- Long-running experiment support (weeks/months)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, Future, as_completed

# IMPORTANT: CUDA requires 'spawn' start method, not 'fork'
# This must be set early, before any CUDA operations
mp_context = mp.get_context('spawn')
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .hyperspace import HyperparameterConfig, TuningPhase, TuningTier
from .trial_runner import TrialRunner, TrialResult, run_trial


logger = logging.getLogger(__name__)


class GPUStatus(Enum):
    """Status of a GPU."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class GPUInfo:
    """Information about a GPU."""
    gpu_id: int
    status: GPUStatus = GPUStatus.AVAILABLE
    current_trial: Optional[str] = None
    trials_completed: int = 0
    trials_failed: int = 0
    total_time_busy: float = 0.0
    last_error: Optional[str] = None
    last_active: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "status": self.status.value,
            "current_trial": self.current_trial,
            "trials_completed": self.trials_completed,
            "trials_failed": self.trials_failed,
            "total_time_busy": self.total_time_busy,
            "last_error": self.last_error,
            "last_active": self.last_active.isoformat() if self.last_active else None,
        }


@dataclass
class TrialTask:
    """A task to be executed."""
    trial_id: str
    config: HyperparameterConfig
    phase: TuningPhase
    priority: int = 0  # Higher = more important
    retries: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutorStats:
    """Statistics for the executor."""
    total_trials_submitted: int = 0
    total_trials_completed: int = 0
    total_trials_failed: int = 0
    total_trials_retried: int = 0
    total_wall_time_seconds: float = 0.0
    total_gpu_time_seconds: float = 0.0
    start_time: Optional[datetime] = None
    last_checkpoint: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trials_submitted": self.total_trials_submitted,
            "total_trials_completed": self.total_trials_completed,
            "total_trials_failed": self.total_trials_failed,
            "total_trials_retried": self.total_trials_retried,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "total_gpu_time_seconds": self.total_gpu_time_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
        }


def _init_worker(gpu_id: int):
    """Initialize worker process with GPU assignment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ignore keyboard interrupt in workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_trial_in_process(
    trial_id: str,
    config_dict: Dict[str, Any],
    phase: str,
    output_dir: str,
    gpu_id: int,
) -> Dict[str, Any]:
    """
    Run a trial in a separate process.

    This function is designed to be called via multiprocessing.
    """
    try:
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Import here to ensure fresh CUDA context
        import torch
        torch.cuda.set_device(0)

        # Run the trial
        result = run_trial(
            trial_id=trial_id,
            config=config_dict,
            phase=phase,
            output_dir=output_dir,
            gpu_id=gpu_id,
        )

        return result

    except Exception as e:
        return {
            "trial_id": trial_id,
            "success": False,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
        }


class DualGPUExecutor:
    """
    Executor for running trials on dual GPUs.

    Features:
    - Automatic GPU assignment
    - Process isolation (prevents CUDA memory leaks)
    - Retry on failure
    - Priority queue
    - Checkpointing every N trials
    - Graceful shutdown
    - Long-running support (heartbeat, auto-restart)
    """

    def __init__(
        self,
        output_dir: Path,
        gpu_ids: List[int] = [0, 1],
        max_retries: int = 3,
        checkpoint_interval: int = 10,  # Checkpoint every N completed trials
        heartbeat_interval: int = 60,  # Heartbeat every N seconds
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_ids = gpu_ids
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        self.heartbeat_interval = heartbeat_interval

        # GPU tracking
        self.gpus: Dict[int, GPUInfo] = {
            gpu_id: GPUInfo(gpu_id=gpu_id)
            for gpu_id in gpu_ids
        }

        # Task queue (priority queue)
        self.pending_tasks: List[TrialTask] = []
        self.running_tasks: Dict[str, TrialTask] = {}  # trial_id -> task
        self.completed_results: Dict[str, TrialResult] = {}
        self.failed_tasks: List[TrialTask] = []

        # Statistics
        self.stats = ExecutorStats()

        # Threading
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Process pool
        self._executor: Optional[ProcessPoolExecutor] = None
        self._futures: Dict[Future, Tuple[str, int]] = {}  # future -> (trial_id, gpu_id)

        # State persistence
        self.state_path = self.output_dir / "executor_state.json"

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup executor logging."""
        log_path = self.output_dir / "executor.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def start(self):
        """Start the executor."""
        logger.info("Starting DualGPUExecutor")

        # Load previous state if exists
        if self.state_path.exists():
            self._load_state()
            logger.info(f"Resumed from checkpoint with {len(self.pending_tasks)} pending tasks")

        self.stats.start_time = self.stats.start_time or datetime.now()

        # Verify GPUs are available
        self._verify_gpus()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # Start process pool (one process per GPU)
        # Use 'spawn' context for CUDA compatibility
        self._executor = ProcessPoolExecutor(
            max_workers=len(self.gpu_ids),
            mp_context=mp_context,
        )

        logger.info(f"Executor started with GPUs: {self.gpu_ids}")

    def stop(self, wait: bool = True):
        """Stop the executor gracefully."""
        logger.info("Stopping executor...")
        self._shutdown_event.set()

        # Save state
        self._save_state()

        # Shutdown pool
        if self._executor:
            self._executor.shutdown(wait=wait)

        logger.info("Executor stopped")

    def submit(self, task: TrialTask) -> str:
        """Submit a task for execution."""
        with self._lock:
            self.pending_tasks.append(task)
            # Sort by priority (higher first)
            self.pending_tasks.sort(key=lambda t: -t.priority)
            self.stats.total_trials_submitted += 1

        logger.info(f"Submitted task {task.trial_id} (priority={task.priority})")
        return task.trial_id

    def submit_batch(self, tasks: List[TrialTask]) -> List[str]:
        """Submit multiple tasks."""
        trial_ids = []
        for task in tasks:
            trial_ids.append(self.submit(task))
        return trial_ids

    def get_result(self, trial_id: str, timeout: Optional[float] = None) -> Optional[TrialResult]:
        """Get result for a specific trial (blocking)."""
        start = time.time()
        while True:
            with self._lock:
                if trial_id in self.completed_results:
                    return self.completed_results[trial_id]

            if timeout and (time.time() - start) > timeout:
                return None

            time.sleep(1)

    def run_until_complete(self, poll_interval: float = 1.0):
        """Run until all tasks are complete."""
        while True:
            # Check shutdown
            if self._shutdown_event.is_set():
                break

            # Schedule pending tasks
            self._schedule_pending()

            # Collect completed futures
            self._collect_results()

            # Check if done
            with self._lock:
                if not self.pending_tasks and not self.running_tasks:
                    logger.info("All tasks complete")
                    break

            time.sleep(poll_interval)

    def run_forever(self, poll_interval: float = 1.0):
        """
        Run continuously, processing tasks as they are submitted.

        Designed for long-running experiments (days/weeks/months).
        """
        logger.info("Running in continuous mode (Ctrl+C to stop)")

        try:
            while not self._shutdown_event.is_set():
                # Schedule pending tasks
                self._schedule_pending()

                # Collect completed futures
                self._collect_results()

                # Checkpoint periodically
                if self._should_checkpoint():
                    self._save_state()
                    self.stats.last_checkpoint = datetime.now()

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            self.stop()

    def _verify_gpus(self):
        """Verify GPUs are available."""
        for gpu_id in self.gpu_ids:
            try:
                # Test GPU access
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                import torch
                if torch.cuda.is_available():
                    torch.cuda.get_device_name(0)
                    self.gpus[gpu_id].status = GPUStatus.AVAILABLE
                    logger.info(f"GPU {gpu_id} available: {torch.cuda.get_device_name(0)}")
                else:
                    self.gpus[gpu_id].status = GPUStatus.DISABLED
                    logger.warning(f"GPU {gpu_id} not available")
            except Exception as e:
                self.gpus[gpu_id].status = GPUStatus.ERROR
                self.gpus[gpu_id].last_error = str(e)
                logger.error(f"GPU {gpu_id} error: {e}")

    def _get_available_gpu(self) -> Optional[int]:
        """Get an available GPU."""
        with self._lock:
            for gpu_id, info in self.gpus.items():
                if info.status == GPUStatus.AVAILABLE:
                    return gpu_id
        return None

    def _schedule_pending(self):
        """Schedule pending tasks to available GPUs."""
        while True:
            gpu_id = self._get_available_gpu()
            if gpu_id is None:
                break

            with self._lock:
                if not self.pending_tasks:
                    break

                task = self.pending_tasks.pop(0)

            self._start_task(task, gpu_id)

    def _start_task(self, task: TrialTask, gpu_id: int):
        """Start a task on a specific GPU."""
        with self._lock:
            self.gpus[gpu_id].status = GPUStatus.BUSY
            self.gpus[gpu_id].current_trial = task.trial_id
            self.gpus[gpu_id].last_active = datetime.now()
            self.running_tasks[task.trial_id] = task

        logger.info(f"Starting task {task.trial_id} on GPU {gpu_id}")

        # Submit to process pool
        future = self._executor.submit(
            _run_trial_in_process,
            task.trial_id,
            task.config.values,
            task.phase.value,
            str(self.output_dir),
            gpu_id,
        )

        self._futures[future] = (task.trial_id, gpu_id)

    def _collect_results(self):
        """Collect results from completed futures."""
        completed_futures = []

        for future, (trial_id, gpu_id) in list(self._futures.items()):
            if future.done():
                completed_futures.append((future, trial_id, gpu_id))

        for future, trial_id, gpu_id in completed_futures:
            try:
                result_dict = future.result(timeout=1)
                self._handle_result(result_dict, trial_id, gpu_id)
            except Exception as e:
                self._handle_failure(trial_id, gpu_id, str(e))

            del self._futures[future]

    def _handle_result(self, result_dict: Dict[str, Any], trial_id: str, gpu_id: int):
        """Handle a completed trial result."""
        with self._lock:
            task = self.running_tasks.pop(trial_id, None)

            # Update GPU status
            start_time = self.gpus[gpu_id].last_active
            if start_time:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.gpus[gpu_id].total_time_busy += elapsed

            self.gpus[gpu_id].status = GPUStatus.AVAILABLE
            self.gpus[gpu_id].current_trial = None

        if result_dict.get("success", False):
            # Success
            with self._lock:
                self.gpus[gpu_id].trials_completed += 1
                self.stats.total_trials_completed += 1
                # Store result (convert dict to TrialResult if needed)
                self.completed_results[trial_id] = result_dict

            logger.info(f"Task {trial_id} completed successfully")

        else:
            # Failure - possibly retry
            error_msg = result_dict.get("error_message", "Unknown error")
            logger.error(f"Task {trial_id} failed: {error_msg}")

            if task and task.retries < task.max_retries:
                # Retry
                task.retries += 1
                with self._lock:
                    self.pending_tasks.append(task)
                    self.stats.total_trials_retried += 1
                logger.info(f"Retrying task {trial_id} (attempt {task.retries})")
            else:
                # Give up
                with self._lock:
                    self.gpus[gpu_id].trials_failed += 1
                    self.stats.total_trials_failed += 1
                    if task:
                        self.failed_tasks.append(task)
                logger.error(f"Task {trial_id} failed permanently after {task.retries if task else 0} retries")

    def _handle_failure(self, trial_id: str, gpu_id: int, error: str):
        """Handle a task failure."""
        logger.error(f"Task {trial_id} failed with error: {error}")

        with self._lock:
            task = self.running_tasks.pop(trial_id, None)
            self.gpus[gpu_id].status = GPUStatus.AVAILABLE
            self.gpus[gpu_id].current_trial = None
            self.gpus[gpu_id].last_error = error

        if task and task.retries < task.max_retries:
            task.retries += 1
            with self._lock:
                self.pending_tasks.append(task)
                self.stats.total_trials_retried += 1
        else:
            with self._lock:
                self.gpus[gpu_id].trials_failed += 1
                self.stats.total_trials_failed += 1
                if task:
                    self.failed_tasks.append(task)

    def _heartbeat_loop(self):
        """Heartbeat loop for monitoring."""
        while not self._shutdown_event.is_set():
            time.sleep(self.heartbeat_interval)

            # Log status
            with self._lock:
                pending = len(self.pending_tasks)
                running = len(self.running_tasks)
                completed = self.stats.total_trials_completed
                failed = self.stats.total_trials_failed

            uptime = (datetime.now() - self.stats.start_time).total_seconds() / 3600 if self.stats.start_time else 0

            logger.info(
                f"Heartbeat: pending={pending}, running={running}, "
                f"completed={completed}, failed={failed}, uptime={uptime:.1f}h"
            )

            # Write heartbeat file
            heartbeat_path = self.output_dir / "heartbeat.json"
            with open(heartbeat_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "pending": pending,
                    "running": running,
                    "completed": completed,
                    "failed": failed,
                    "uptime_hours": uptime,
                    "gpus": {str(k): v.to_dict() for k, v in self.gpus.items()},
                }, f, indent=2)

    def _should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint."""
        if self.stats.last_checkpoint is None:
            return True

        # Checkpoint every N trials
        if self.stats.total_trials_completed % self.checkpoint_interval == 0:
            return True

        # Checkpoint every hour
        if (datetime.now() - self.stats.last_checkpoint).total_seconds() > 3600:
            return True

        return False

    def _save_state(self):
        """Save executor state for resume."""
        state = {
            "stats": self.stats.to_dict(),
            "pending_tasks": [
                {
                    "trial_id": t.trial_id,
                    "config": t.config.to_dict(),
                    "phase": t.phase.value,
                    "priority": t.priority,
                    "retries": t.retries,
                }
                for t in self.pending_tasks
            ],
            "completed_results": {
                k: v if isinstance(v, dict) else v.to_dict()
                for k, v in self.completed_results.items()
            },
            "failed_tasks": [t.trial_id for t in self.failed_tasks],
            "gpus": {str(k): v.to_dict() for k, v in self.gpus.items()},
        }

        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved state to {self.state_path}")

    def _load_state(self):
        """Load executor state from checkpoint."""
        with open(self.state_path, "r") as f:
            state = json.load(f)

        # Restore stats
        self.stats.total_trials_submitted = state["stats"].get("total_trials_submitted", 0)
        self.stats.total_trials_completed = state["stats"].get("total_trials_completed", 0)
        self.stats.total_trials_failed = state["stats"].get("total_trials_failed", 0)
        self.stats.total_trials_retried = state["stats"].get("total_trials_retried", 0)

        if state["stats"].get("start_time"):
            self.stats.start_time = datetime.fromisoformat(state["stats"]["start_time"])

        # Restore pending tasks
        self.pending_tasks = []
        for t in state.get("pending_tasks", []):
            task = TrialTask(
                trial_id=t["trial_id"],
                config=HyperparameterConfig.from_dict(t["config"]),
                phase=TuningPhase(t["phase"]),
                priority=t.get("priority", 0),
                retries=t.get("retries", 0),
            )
            self.pending_tasks.append(task)

        # Restore completed results
        self.completed_results = state.get("completed_results", {})

        logger.info(f"Loaded state: {len(self.pending_tasks)} pending, {len(self.completed_results)} completed")

    def get_stats(self) -> Dict[str, Any]:
        """Get current executor statistics."""
        with self._lock:
            return {
                "stats": self.stats.to_dict(),
                "pending": len(self.pending_tasks),
                "running": len(self.running_tasks),
                "completed": len(self.completed_results),
                "failed": len(self.failed_tasks),
                "gpus": {str(k): v.to_dict() for k, v in self.gpus.items()},
            }

    def get_completed_results(self) -> Dict[str, Any]:
        """Get all completed results."""
        with self._lock:
            return dict(self.completed_results)

    def get_best_result(self, phase: TuningPhase) -> Optional[Dict[str, Any]]:
        """Get the best result for a phase."""
        best = None
        best_value = float('-inf')

        # Objective varies by phase
        objective_map = {
            TuningPhase.VAE: "vae_retrieval_accuracy_at_8",
            TuningPhase.SCORER: "scorer_ndcg_at_8",
            TuningPhase.GP: "gp_spearman_correlation",
            TuningPhase.INFERENCE: "e2e_final_accuracy",
        }
        objective_name = objective_map.get(phase)

        with self._lock:
            for trial_id, result in self.completed_results.items():
                if isinstance(result, dict):
                    if result.get("phase") != phase.value:
                        continue
                    metrics = result.get("metrics", {})
                    if objective_name in metrics:
                        value = metrics[objective_name].get("value", float('-inf'))
                        if value > best_value:
                            best_value = value
                            best = result

        return best


class SingleGPUExecutor(DualGPUExecutor):
    """Executor for single GPU (convenience class)."""

    def __init__(self, output_dir: Path, gpu_id: int = 0, **kwargs):
        super().__init__(output_dir, gpu_ids=[gpu_id], **kwargs)
