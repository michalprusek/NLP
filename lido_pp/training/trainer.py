"""
Main Trainer for LID-O++.

Orchestrates the complete training pipeline:
1. Projector pre-training (optional)
2. Flow Matching training with OAT-FM
3. Reflow for trajectory straightening
4. Active Learning loop with FCU gating

This trainer integrates all LID-O++ components into a unified training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from lido_pp.config import LIDOPPConfig
from lido_pp.flow.flow_dit import FlowDiT
from lido_pp.flow.losses import oat_flow_matching_loss, measure_trajectory_straightness
from lido_pp.flow.reflow import ReflowTrainer, ReflowConfig, verify_one_step_inference
from lido_pp.active_learning.curvature import compute_flow_curvature, batch_fcu_analysis
from lido_pp.active_learning.value_head import ValueHead, ValueHeadTrainer
from lido_pp.active_learning.gating import EvaluationGate, EvaluationType
from lido_pp.training.checkpointing import CheckpointManager, MetricsLogger
from lido_pp.training.data_prep import InstructionDataset, FlowMatchingBatch

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current training state."""
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    phase: str = "flow"  # "projector", "flow", "reflow", "active_learning"


class LIDOPPTrainer:
    """
    Main trainer for LID-O++ pipeline.

    Training phases:
    1. Projector Training: Train latent injection projector for reconstruction
    2. Flow Training: Train FlowDiT with CFM + OAT-FM loss
    3. Reflow: Straighten trajectories for 1-step inference
    4. Active Learning: Fine-tune with FCU-gated evaluations

    The trainer handles all phases and provides utilities for:
    - Checkpoint management
    - Metrics logging
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        config: LIDOPPConfig,
        encoder=None,  # GritLMUnifiedEncoder (optional, loaded if None)
        flow_model: Optional[FlowDiT] = None,
        value_head: Optional[ValueHead] = None,
    ):
        """
        Args:
            config: LID-O++ configuration
            encoder: Pre-loaded GritLM encoder (loaded if None)
            flow_model: Pre-initialized FlowDiT (created if None)
            value_head: Pre-initialized ValueHead (created if None)
        """
        self.config = config
        self.device = config.device

        # Initialize models
        self.encoder = encoder
        self.flow_model = flow_model or self._create_flow_model()
        self.value_head = value_head or self._create_value_head()

        # Optimizers (created during training)
        self.flow_optimizer: Optional[optim.Optimizer] = None
        self.flow_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # Training state
        self.state = TrainingState()

        # Checkpoint and metrics
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=5,
            metric_name="loss",
            metric_mode="min",
        )
        self.metrics_logger = MetricsLogger(log_dir=config.results_dir)

        # Value Head trainer
        self.vh_trainer: Optional[ValueHeadTrainer] = None

    def _create_flow_model(self) -> FlowDiT:
        """Create FlowDiT model from config."""
        model = FlowDiT(
            latent_dim=self.config.embedding_dim,  # Flow in embedding space
            hidden_dim=self.config.flow_hidden_dim,
            num_layers=self.config.flow_num_layers,
            num_heads=self.config.flow_num_heads,
            context_dim=self.config.embedding_dim,
            num_context_tokens=self.config.flow_context_tokens,
            dropout=self.config.flow_dropout,
        ).to(self.device)

        logger.info(f"Created FlowDiT: {sum(p.numel() for p in model.parameters()):,} params")
        return model

    def _create_value_head(self) -> ValueHead:
        """Create ValueHead from config."""
        model = ValueHead(
            latent_dim=self.config.embedding_dim,
            hidden_dim=self.config.value_head_hidden,
        ).to(self.device)

        logger.info(f"Created ValueHead: {sum(p.numel() for p in model.parameters()):,} params")
        return model

    def _create_optimizers(self):
        """Create optimizers and schedulers."""
        self.flow_optimizer = optim.AdamW(
            self.flow_model.parameters(),
            lr=self.config.flow_lr,
            weight_decay=1e-5,
        )

        self.flow_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.flow_optimizer,
            T_0=self.config.flow_warmup_epochs,
            T_mult=2,
            eta_min=self.config.flow_lr * 0.01,
        )

    def train_flow_epoch(
        self,
        dataloader,
        use_oat: bool = True,
    ) -> Dict[str, float]:
        """
        Train FlowDiT for one epoch.

        Args:
            dataloader: DataLoader yielding FlowMatchingBatch
            use_oat: Use OAT-FM regularization

        Returns:
            Dict with epoch metrics
        """
        self.flow_model.train()
        epoch_losses = []
        epoch_metrics = []

        for batch in dataloader:
            # Get batch data
            x_0 = batch.x_0.to(self.device)
            x_1 = batch.x_1.to(self.device)
            context = batch.context.to(self.device) if batch.context is not None else None

            # Compute loss
            loss, metrics = oat_flow_matching_loss(
                self.flow_model,
                x_0,
                x_1,
                context,
                sigma_min=self.config.sigma_min,
                oat_weight=self.config.oat_weight if use_oat else 0.0,
                oat_steps=self.config.oat_steps,
            )

            # Backward
            self.flow_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
            self.flow_optimizer.step()

            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)

            self.state.step += 1

        # Aggregate metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_metrics = {
            key: sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            for key in epoch_metrics[0].keys()
        }

        return {"loss": avg_loss, **avg_metrics}

    def train_flow(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: Optional[int] = None,
        use_oat: bool = True,
        eval_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Train FlowDiT model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            epochs: Number of epochs (uses config if None)
            use_oat: Use OAT-FM regularization
            eval_interval: Epochs between evaluations

        Returns:
            Dict with training results
        """
        epochs = epochs or self.config.flow_epochs
        self._create_optimizers()

        logger.info(f"Starting Flow training for {epochs} epochs")
        self.state.phase = "flow"

        best_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            self.state.epoch = epoch

            # Train epoch
            train_metrics = self.train_flow_epoch(train_dataloader, use_oat=use_oat)

            # Update scheduler
            self.flow_scheduler.step()
            current_lr = self.flow_scheduler.get_last_lr()[0]

            # Log metrics
            train_metrics["lr"] = current_lr
            self.metrics_logger.log(train_metrics, step=epoch)

            # Evaluate
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                self._log_flow_progress(epoch, epochs, train_metrics, start_time)

                # Validation
                if val_dataloader is not None:
                    val_metrics = self._evaluate_flow(val_dataloader)
                    self.metrics_logger.log({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)

                # Checkpoint
                self.checkpoint_manager.save(
                    model=self.flow_model,
                    optimizer=self.flow_optimizer,
                    scheduler=self.flow_scheduler,
                    epoch=epoch,
                    step=self.state.step,
                    metrics=train_metrics,
                    config={"phase": "flow"},
                    name_prefix="flow",
                )

            # Early stopping
            if train_metrics["loss"] < best_loss:
                best_loss = train_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.flow_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final metrics
        self.metrics_logger.save("flow_metrics.json")

        return {
            "final_loss": train_metrics["loss"],
            "best_loss": best_loss,
            "epochs_trained": epoch + 1,
            "time_elapsed": time.time() - start_time,
        }

    def _evaluate_flow(self, dataloader) -> Dict[str, float]:
        """Evaluate flow model on validation data."""
        self.flow_model.eval()
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                x_0 = batch.x_0.to(self.device)
                x_1 = batch.x_1.to(self.device)
                context = batch.context.to(self.device) if batch.context is not None else None

                loss, _ = oat_flow_matching_loss(
                    self.flow_model, x_0, x_1, context,
                    oat_weight=0.0,  # No OAT for eval
                )
                losses.append(loss.item())

        return {"loss": sum(losses) / len(losses)}

    def _log_flow_progress(self, epoch: int, total_epochs: int, metrics: Dict, start_time: float):
        """Log training progress."""
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (total_epochs - epoch - 1)

        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"CFM: {metrics.get('cfm_loss', 0):.6f} | "
            f"OAT: {metrics.get('oat_loss', 0):.6f} | "
            f"LR: {metrics.get('lr', 0):.2e} | "
            f"ETA: {eta/60:.1f}m"
        )

    def run_reflow(
        self,
        context_source: Optional[torch.Tensor] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Reflow to straighten trajectories.

        Args:
            context_source: Source of context vectors
            epochs: Number of reflow epochs

        Returns:
            Dict with reflow results
        """
        epochs = epochs or self.config.reflow_epochs
        self.state.phase = "reflow"

        logger.info(f"Starting Reflow for {epochs} epochs")

        reflow_config = ReflowConfig(
            num_rounds=1,
            epochs_per_round=epochs,
            lr=self.config.flow_lr * 0.5,
            batch_size=self.config.flow_batch_size,
            integration_steps=self.config.reflow_ode_steps,
            use_oat=True,
            oat_weight=self.config.oat_weight * 2,
            sigma_min=self.config.sigma_min,
        )

        trainer = ReflowTrainer(self.flow_model, reflow_config, self.device)

        result = trainer.train(
            x_0_source=None,
            context_source=context_source,
            latent_dim=self.config.embedding_dim,
        )

        # Verify 1-step inference
        x_0_test = torch.randn(10, self.config.embedding_dim, device=self.device)
        ctx_test = context_source[:10].to(self.device) if context_source is not None else None
        quality = verify_one_step_inference(self.flow_model, x_0_test, ctx_test)

        logger.info(f"Reflow complete. 1-step L2 error: {quality['l2_error']:.6f}")

        # Save reflowed model
        self.checkpoint_manager.save(
            model=self.flow_model,
            epoch=self.state.epoch,
            step=self.state.step,
            metrics={"l2_error": quality["l2_error"]},
            name_prefix="flow_reflowed",
        )

        return {
            "straightness": result.final_straightness,
            "one_step_quality": quality,
        }

    def train_value_head(
        self,
        instructions: List[str],
        error_rates: List[float],
        epochs: int = 100,
    ) -> Dict[str, float]:
        """
        Train ValueHead on evaluation data.

        Args:
            instructions: List of instructions (will be encoded)
            error_rates: Corresponding error rates
            epochs: Training epochs

        Returns:
            Dict with training metrics
        """
        self.state.phase = "value_head"
        logger.info(f"Training ValueHead on {len(instructions)} samples")

        # Initialize trainer
        self.vh_trainer = ValueHeadTrainer(
            self.value_head,
            lr=1e-3,
            buffer_size=10000,
        )

        # Encode instructions and add to buffer
        if self.encoder is not None:
            embeddings = self.encoder.encode_batch(instructions, show_progress=True)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

            for i, (emb, er) in enumerate(zip(embeddings_tensor, error_rates)):
                self.vh_trainer.add_observation(emb, er)
        else:
            logger.warning("No encoder available, using random latents")
            for er in error_rates:
                z = torch.randn(self.config.embedding_dim)
                self.vh_trainer.add_observation(z, er)

        # Train
        for epoch in range(epochs):
            metrics = self.vh_trainer.train_epoch(
                steps_per_epoch=50,
                batch_size=64,
                device=self.device,
            )

            if (epoch + 1) % 20 == 0:
                logger.info(f"VH Epoch {epoch+1}: loss={metrics['avg_loss']:.6f}")

        return self.vh_trainer.get_buffer_statistics()

    def analyze_fcu(
        self,
        dataloader,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Analyze FCU distribution on data.

        Args:
            dataloader: Data to analyze
            num_batches: Number of batches to analyze

        Returns:
            Dict with FCU statistics
        """
        self.flow_model.eval()
        all_stats = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            z = batch.x_0.to(self.device)
            context = batch.context.to(self.device) if batch.context is not None else None

            stats = batch_fcu_analysis(self.flow_model, z, context, num_steps=20)
            all_stats.append(stats)

        # Aggregate
        aggregated = {
            key: sum(s[key] for s in all_stats) / len(all_stats)
            for key in all_stats[0].keys()
        }

        logger.info(f"FCU Analysis: mean={aggregated['curvature_mean']:.4f}, "
                   f"p90={aggregated['curvature_p90']:.4f}")

        return aggregated


class SimplifiedTrainer:
    """
    Simplified trainer for quick experiments.

    Provides a minimal training loop without all the bells and whistles
    of the full LIDOPPTrainer.
    """

    def __init__(
        self,
        flow_model: FlowDiT,
        device: str = "cuda",
        lr: float = 1e-4,
    ):
        self.model = flow_model.to(device)
        self.device = device

        self.optimizer = optim.AdamW(flow_model.parameters(), lr=lr)

    def train_step(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        oat_weight: float = 0.1,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        x_0 = x_0.to(self.device)
        x_1 = x_1.to(self.device)
        if context is not None:
            context = context.to(self.device)

        loss, metrics = oat_flow_matching_loss(
            self.model, x_0, x_1, context,
            oat_weight=oat_weight,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return metrics

    def train_epochs(
        self,
        x_1_data: torch.Tensor,
        context_data: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 64,
        log_interval: int = 10,
    ) -> List[float]:
        """Train for multiple epochs on data."""
        losses = []
        n_samples = x_1_data.shape[0]

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            x_1_shuffled = x_1_data[perm]
            ctx_shuffled = context_data[perm] if context_data is not None else None

            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                x_1 = x_1_shuffled[i:i+batch_size]
                x_0 = torch.randn_like(x_1)
                ctx = ctx_shuffled[i:i+batch_size] if ctx_shuffled is not None else None

                metrics = self.train_step(x_0, x_1, ctx)
                epoch_losses.append(metrics["total_loss"])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)

            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        return losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing LIDOPPTrainer...")

    # Test SimplifiedTrainer
    print("\n1. Testing SimplifiedTrainer...")
    flow = FlowDiT(latent_dim=768, hidden_dim=256, num_layers=4)
    trainer = SimplifiedTrainer(flow, device="cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic data
    x_1 = torch.randn(200, 768)
    context = x_1.unsqueeze(1).expand(-1, 4, -1)

    losses = trainer.train_epochs(x_1, context, epochs=20, batch_size=32, log_interval=5)
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Loss reduction: {losses[0]:.6f} -> {losses[-1]:.6f}")

    print("\n[OK] Trainer tests passed!")
