"""
Checkpointing Utilities for LID-O++.

Handles saving and loading of:
1. FlowDiT model weights
2. Latent Projector weights
3. Training state (optimizer, scheduler, epoch)
4. Metrics history
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""
    timestamp: str
    epoch: int
    step: int
    loss: float
    metrics: Dict[str, float]
    config: Dict[str, Any]


class CheckpointManager:
    """
    Manages checkpoints for LID-O++ training.

    Features:
    - Save/load model weights
    - Track best checkpoints
    - Automatic cleanup of old checkpoints
    - Resume training from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str = "lido_pp/checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        metric_name: str = "loss",
        metric_mode: str = "min",  # "min" or "max"
    ):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            save_best_only: Only save when metric improves
            metric_name: Metric to track for best checkpoint
            metric_mode: "min" (lower is better) or "max" (higher is better)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        # Track best metric
        self.best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self.best_checkpoint_path: Optional[Path] = None

        # List of saved checkpoints (oldest first)
        self.checkpoints: list = []

    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than best."""
        if self.metric_mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        name_prefix: str = "checkpoint",
    ) -> Optional[Path]:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            config: Training configuration
            extra_state: Additional state to save
            name_prefix: Prefix for checkpoint filename

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        metrics = metrics or {}
        config = config or {}

        # Check if we should save
        current_metric = metrics.get(self.metric_name, 0)
        is_best = self._is_better(current_metric)

        if self.save_best_only and not is_best:
            return None

        # Update best metric
        if is_best:
            self.best_metric = current_metric

        # Create checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name_prefix}_epoch{epoch:04d}_step{step:06d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare state dict
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "config": config,
            "timestamp": timestamp,
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state is not None:
            state["extra_state"] = extra_state

        # Save
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Track checkpoint
        self.checkpoints.append(checkpoint_path)

        # Update best checkpoint
        if is_best:
            self.best_checkpoint_path = checkpoint_path
            # Save pointer to best
            best_link = self.checkpoint_dir / f"{name_prefix}_best.pt"
            if best_link.exists():
                best_link.unlink()
            torch.save(state, best_link)
            logger.info(f"New best checkpoint: {self.metric_name}={current_metric:.6f}")

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint_path

    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists() and old_checkpoint != self.best_checkpoint_path:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            device: Device to load to

        Returns:
            Dict with loaded state (epoch, step, metrics, etc.)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model.load_state_dict(state["model_state_dict"])

        # Load optimizer if provided
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # Load scheduler if provided
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return {
            "epoch": state.get("epoch", 0),
            "step": state.get("step", 0),
            "metrics": state.get("metrics", {}),
            "config": state.get("config", {}),
            "extra_state": state.get("extra_state", {}),
        }

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        name_prefix: str = "checkpoint",
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / f"{name_prefix}_best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        return self.load(best_path, model, optimizer, scheduler, device)

    def get_latest_checkpoint(self, name_prefix: str = "checkpoint") -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob(f"{name_prefix}_epoch*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)


class MetricsLogger:
    """
    Logger for training metrics.

    Saves metrics to JSON for later analysis and visualization.
    """

    def __init__(self, log_dir: str = "lido_pp/results"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: Dict[str, list] = {}
        self.step = 0

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics at current step."""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append({"step": self.step, "value": value})

    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics to {filepath}")

    def load(self, filename: str = "metrics.json"):
        """Load metrics from JSON file."""
        filepath = self.log_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                self.metrics_history = json.load(f)
            logger.info(f"Loaded metrics from {filepath}")

    def get_best(self, metric_name: str, mode: str = "min") -> Dict[str, Any]:
        """Get best value of a metric."""
        if metric_name not in self.metrics_history:
            return {}

        values = self.metrics_history[metric_name]
        if mode == "min":
            best = min(values, key=lambda x: x["value"])
        else:
            best = max(values, key=lambda x: x["value"])

        return best

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            vals = [v["value"] for v in values]
            summary[key] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "last": vals[-1],
                "count": len(vals),
            }
        return summary


def save_training_results(
    results: Dict[str, Any],
    output_dir: str = "lido_pp/results",
    filename: str = "training_results.json",
):
    """Save training results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved training results to {filepath}")


def load_training_results(
    output_dir: str = "lido_pp/results",
    filename: str = "training_results.json",
) -> Dict[str, Any]:
    """Load training results from JSON."""
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Results not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import tempfile

    print("Testing Checkpointing...")

    # Create temporary directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CheckpointManager
        print("\n1. Testing CheckpointManager...")
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            max_checkpoints=3,
            save_best_only=False,
            metric_name="loss",
            metric_mode="min",
        )

        # Create dummy model
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoints
        for epoch in range(5):
            loss = 1.0 / (epoch + 1)  # Decreasing loss
            path = manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * 100,
                metrics={"loss": loss, "accuracy": epoch * 0.1},
            )
            print(f"   Epoch {epoch}: loss={loss:.4f}, saved={path is not None}")

        print(f"   Best checkpoint: {manager.best_checkpoint_path}")
        print(f"   Remaining checkpoints: {len(list(Path(tmpdir).glob('*.pt')))}")

        # Test loading
        print("\n2. Testing checkpoint loading...")
        new_model = torch.nn.Linear(10, 10)
        state = manager.load_best(new_model, device="cpu")
        print(f"   Loaded epoch: {state['epoch']}")
        print(f"   Loaded metrics: {state['metrics']}")

        # Test MetricsLogger
        print("\n3. Testing MetricsLogger...")
        logger_test = MetricsLogger(log_dir=tmpdir)

        for step in range(10):
            logger_test.log({
                "loss": 1.0 / (step + 1),
                "accuracy": step * 0.1,
            }, step=step)

        logger_test.save()
        summary = logger_test.summary()
        print(f"   Loss summary: min={summary['loss']['min']:.4f}, max={summary['loss']['max']:.4f}")

        best = logger_test.get_best("loss", mode="min")
        print(f"   Best loss: {best}")

    print("\n[OK] Checkpointing tests passed!")
