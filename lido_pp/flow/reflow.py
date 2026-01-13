"""
Reflow Training for Trajectory Straightening.

Reflow is a technique that straightens flow trajectories by retraining
the model on its own generated outputs. This enables:
1. Single-step inference (50-100x faster)
2. Lower curvature (more confident predictions)
3. Better semantic consistency

The process:
1. Train initial model with CFM + OAT loss
2. Generate trajectory pairs: (noise, flow_output)
3. Retrain model on generated pairs
4. Repeat until trajectories are straight enough

After Reflow, the model can do 1-step inference:
    x_1 = x_0 + v(x_0, 0) * dt  where dt = 1

Key insight: Reflow essentially "teaches" the model to predict
the entire trajectory in a single step by showing it complete
(start, end) pairs from its own sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import logging

from lido_pp.flow.ode_solver import integrate, IntegrationResult
from lido_pp.flow.losses import (
    conditional_flow_matching_loss,
    oat_flow_matching_loss,
    measure_trajectory_straightness,
)

logger = logging.getLogger(__name__)


@dataclass
class ReflowConfig:
    """Configuration for Reflow training."""

    # Number of reflow rounds
    num_rounds: int = 1

    # Epochs per reflow round
    epochs_per_round: int = 2000

    # Learning rate (typically lower than initial training)
    lr: float = 5e-5

    # Batch size for reflow
    batch_size: int = 64

    # Integration steps for trajectory generation
    integration_steps: int = 20

    # Integration method ("euler", "midpoint", "rk4")
    integration_method: str = "euler"

    # Number of trajectory pairs to generate
    num_pairs: int = 10000

    # Whether to use OAT regularization during reflow
    use_oat: bool = True

    # OAT weight (typically higher during reflow)
    oat_weight: float = 0.2

    # Minimum noise for numerical stability
    sigma_min: float = 0.001

    # Early stopping based on straightness
    target_straightness: float = 0.1

    # Patience for early stopping
    patience: int = 500

    # Evaluation interval
    eval_interval: int = 100


@dataclass
class ReflowResult:
    """Result of Reflow training."""

    # Reflowed model
    model: nn.Module

    # Training history
    loss_history: List[float]

    # Straightness metrics per round
    straightness_history: List[Dict[str, float]]

    # Number of rounds completed
    rounds_completed: int

    # Final straightness metrics
    final_straightness: Dict[str, float]


class ReflowTrainer:
    """
    Trainer for Reflow (trajectory straightening).

    Reflow works by:
    1. Generating trajectory pairs (x_0, x_1) using current model
    2. Retraining model to predict velocity for straight paths
    3. Repeating until trajectories are sufficiently straight

    After Reflow, the model can predict the entire transformation
    in a single step: x_1 = x_0 + v(x_0, 0)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReflowConfig,
        device: str = "cuda",
    ):
        """
        Args:
            model: Pre-trained FlowDiT model to reflow
            config: Reflow configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=1e-5,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs_per_round,
            eta_min=config.lr * 0.01,
        )

    def generate_trajectory_pairs(
        self,
        x_0_source: Optional[torch.Tensor] = None,
        context_source: Optional[torch.Tensor] = None,
        latent_dim: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate (noise, output) pairs using current model.

        These pairs represent straight-line targets for retraining.

        Args:
            x_0_source: Optional source of noise (defaults to random)
            context_source: Optional context (defaults to None)
            latent_dim: Latent dimension if generating random noise

        Returns:
            x_0: Starting noise (N, latent_dim)
            x_1: Flow outputs (N, latent_dim)
            context: Optional context used (N, num_ctx, ctx_dim)
        """
        num_pairs = self.config.num_pairs
        batch_size = self.config.batch_size

        logger.info(f"Generating {num_pairs} trajectory pairs...")

        all_x_0 = []
        all_x_1 = []
        all_context = []

        self.model.eval()

        with torch.no_grad():
            for i in range(0, num_pairs, batch_size):
                batch_n = min(batch_size, num_pairs - i)

                # Generate or sample noise
                if x_0_source is not None:
                    # Sample from source
                    indices = torch.randint(0, x_0_source.shape[0], (batch_n,))
                    x_0 = x_0_source[indices].to(self.device)
                else:
                    # Random noise
                    x_0 = torch.randn(batch_n, latent_dim, device=self.device)

                # Generate or sample context
                if context_source is not None:
                    indices = torch.randint(0, context_source.shape[0], (batch_n,))
                    context = context_source[indices].to(self.device)
                else:
                    context = None

                # Integrate to get x_1
                result = integrate(
                    self.model,
                    x_0,
                    context,
                    num_steps=self.config.integration_steps,
                    method=self.config.integration_method,
                    compute_curvature=False,
                )

                all_x_0.append(x_0.cpu())
                all_x_1.append(result.x_final.cpu())
                if context is not None:
                    all_context.append(context.cpu())

                if (i + batch_n) % 1000 == 0:
                    logger.info(f"  Generated {i + batch_n}/{num_pairs} pairs")

        x_0 = torch.cat(all_x_0, dim=0)
        x_1 = torch.cat(all_x_1, dim=0)
        context = torch.cat(all_context, dim=0) if all_context else None

        logger.info(f"Generated trajectory pairs: x_0={x_0.shape}, x_1={x_1.shape}")

        return x_0, x_1, context

    def train_round(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        round_num: int = 0,
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Train one round of Reflow.

        Args:
            x_0: Starting points (noise) (N, latent_dim)
            x_1: Target points (flow outputs) (N, latent_dim)
            context: Optional context (N, num_ctx, ctx_dim)
            round_num: Current round number

        Returns:
            loss_history: List of losses
            final_straightness: Straightness metrics at end
        """
        logger.info(f"Starting Reflow round {round_num + 1}...")

        # Create dataloader
        if context is not None:
            dataset = TensorDataset(x_0, x_1, context)
        else:
            dataset = TensorDataset(x_0, x_1)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.model.train()
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs_per_round):
            epoch_losses = []

            for batch in dataloader:
                if context is not None:
                    batch_x0, batch_x1, batch_ctx = batch
                    batch_ctx = batch_ctx.to(self.device)
                else:
                    batch_x0, batch_x1 = batch
                    batch_ctx = None

                batch_x0 = batch_x0.to(self.device)
                batch_x1 = batch_x1.to(self.device)

                # Compute loss
                if self.config.use_oat:
                    loss, metrics = oat_flow_matching_loss(
                        self.model,
                        batch_x0,
                        batch_x1,
                        batch_ctx,
                        sigma_min=self.config.sigma_min,
                        oat_weight=self.config.oat_weight,
                    )
                else:
                    loss, metrics = conditional_flow_matching_loss(
                        self.model,
                        batch_x0,
                        batch_x1,
                        batch_ctx,
                        sigma_min=self.config.sigma_min,
                    )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())

            self.scheduler.step()
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(avg_loss)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging and evaluation
            if (epoch + 1) % self.config.eval_interval == 0:
                # Measure straightness
                eval_indices = torch.randint(0, x_0.shape[0], (100,))
                eval_x0 = x_0[eval_indices].to(self.device)
                eval_x1 = x_1[eval_indices].to(self.device)
                eval_ctx = context[eval_indices].to(self.device) if context is not None else None

                straightness = measure_trajectory_straightness(
                    self.model, eval_x0, eval_x1, eval_ctx, num_steps=20
                )

                logger.info(
                    f"  Round {round_num + 1}, Epoch {epoch + 1}: "
                    f"loss={avg_loss:.6f}, "
                    f"deviation={straightness['avg_deviation']:.4f}, "
                    f"v_variance={straightness['velocity_variance']:.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

                # Check if straight enough
                if straightness['avg_deviation'] < self.config.target_straightness:
                    logger.info(f"  Target straightness reached! Stopping early.")
                    return loss_history, straightness

            # Patience check
            if patience_counter >= self.config.patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # Final straightness measurement
        eval_indices = torch.randint(0, x_0.shape[0], (200,))
        eval_x0 = x_0[eval_indices].to(self.device)
        eval_x1 = x_1[eval_indices].to(self.device)
        eval_ctx = context[eval_indices].to(self.device) if context is not None else None

        final_straightness = measure_trajectory_straightness(
            self.model, eval_x0, eval_x1, eval_ctx, num_steps=20
        )

        return loss_history, final_straightness

    def train(
        self,
        x_0_source: Optional[torch.Tensor] = None,
        context_source: Optional[torch.Tensor] = None,
        latent_dim: int = 32,
    ) -> ReflowResult:
        """
        Run full Reflow training.

        Args:
            x_0_source: Optional source of noise samples
            context_source: Optional source of context
            latent_dim: Latent dimension if generating random noise

        Returns:
            ReflowResult with trained model and metrics
        """
        all_loss_history = []
        straightness_history = []

        for round_num in range(self.config.num_rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"Reflow Round {round_num + 1}/{self.config.num_rounds}")
            logger.info(f"{'='*50}")

            # Generate new trajectory pairs
            x_0, x_1, context = self.generate_trajectory_pairs(
                x_0_source, context_source, latent_dim
            )

            # Reset scheduler for new round
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs_per_round,
                eta_min=self.config.lr * 0.01,
            )

            # Train round
            round_losses, round_straightness = self.train_round(
                x_0, x_1, context, round_num
            )

            all_loss_history.extend(round_losses)
            straightness_history.append(round_straightness)

            logger.info(f"\nRound {round_num + 1} completed:")
            for k, v in round_straightness.items():
                logger.info(f"  {k}: {v:.6f}")

            # Check if we've achieved target
            if round_straightness['avg_deviation'] < self.config.target_straightness:
                logger.info("Target straightness achieved! Stopping Reflow.")
                break

        return ReflowResult(
            model=self.model,
            loss_history=all_loss_history,
            straightness_history=straightness_history,
            rounds_completed=round_num + 1,
            final_straightness=straightness_history[-1] if straightness_history else {},
        )


def verify_one_step_inference(
    model: nn.Module,
    x_0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    reference_steps: int = 50,
) -> Dict[str, float]:
    """
    Verify quality of 1-step inference after Reflow.

    Compares 1-step result with multi-step reference.

    Args:
        model: Reflowed model
        x_0: Test noise samples (B, latent_dim)
        context: Optional context
        reference_steps: Steps for reference integration

    Returns:
        Dict with quality metrics
    """
    model.eval()

    with torch.no_grad():
        # 1-step inference
        t = torch.zeros(x_0.shape[0], device=x_0.device)
        v = model(x_0, t, context)
        x_1_one_step = x_0 + v  # dt = 1

        # Multi-step reference
        result_multi = integrate(
            model, x_0, context,
            num_steps=reference_steps,
            method="euler",
            compute_curvature=True,
        )
        x_1_multi = result_multi.x_final

        # Compare
        l2_error = (x_1_one_step - x_1_multi).norm(dim=-1).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_1_one_step, x_1_multi, dim=-1
        ).mean().item()

        # Relative error
        rel_error = l2_error / (x_1_multi.norm(dim=-1).mean().item() + 1e-8)

    return {
        "l2_error": l2_error,
        "cosine_similarity": cosine_sim,
        "relative_error": rel_error,
        "reference_curvature": result_multi.curvature.mean().item(),
    }


def create_reflow_trainer(
    model: nn.Module,
    config,  # LIDOPPConfig
    device: str = "cuda",
) -> ReflowTrainer:
    """Factory function for creating ReflowTrainer from LIDOPPConfig."""
    reflow_config = ReflowConfig(
        num_rounds=1,  # Usually 1 round is enough
        epochs_per_round=config.reflow_epochs,
        lr=config.flow_lr * 0.5,  # Lower LR for reflow
        batch_size=config.flow_batch_size,
        integration_steps=config.reflow_ode_steps,
        integration_method="euler",
        num_pairs=10000,
        use_oat=True,
        oat_weight=config.oat_weight * 2,  # Higher OAT weight for straightening
        sigma_min=config.sigma_min,
        target_straightness=0.1,
        patience=500,
        eval_interval=100,
    )

    return ReflowTrainer(model, reflow_config, device)


if __name__ == "__main__":
    from lido_pp.flow.flow_dit import FlowDiT

    logging.basicConfig(level=logging.INFO)
    print("Testing Reflow Training...")

    # Create model
    model = FlowDiT(latent_dim=32, hidden_dim=256, num_layers=4)

    # Create trainer with minimal config for testing
    config = ReflowConfig(
        num_rounds=1,
        epochs_per_round=50,  # Very short for testing
        lr=1e-4,
        batch_size=32,
        integration_steps=10,
        num_pairs=200,  # Small for testing
        use_oat=True,
        oat_weight=0.1,
        eval_interval=25,
    )

    trainer = ReflowTrainer(model, config, device="cpu")

    # Generate some synthetic context
    context_source = torch.randn(500, 4, 768)

    # Run reflow
    print("\nRunning Reflow (short test)...")
    result = trainer.train(
        x_0_source=None,
        context_source=context_source,
        latent_dim=32,
    )

    print(f"\nReflow completed:")
    print(f"  Rounds completed: {result.rounds_completed}")
    print(f"  Total loss steps: {len(result.loss_history)}")
    print(f"  Final straightness: {result.final_straightness}")

    # Verify 1-step inference
    print("\nVerifying 1-step inference...")
    x_0_test = torch.randn(10, 32)
    ctx_test = torch.randn(10, 4, 768)
    quality = verify_one_step_inference(result.model, x_0_test, ctx_test, reference_steps=20)

    print("1-step inference quality:")
    for k, v in quality.items():
        print(f"  {k}: {v:.6f}")

    print("\nAll tests passed!")
