"""
OT-CFM Training for ManifoldKeeper.

Trains the ManifoldKeeper velocity field using Optimal Transport
Conditional Flow Matching (OT-CFM).

Key features:
- OT pairing: Matches noise x_0 with data x_1 via optimal transport
- U-shaped timestep sampling: More weight at t=0 and t=1 boundaries
- Simulation-free: No ODE solver during training
- DDP support: Multi-GPU training with DistributedDataParallel

Reference:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- OT-CFM: Improving Training of Rectified Flows (Liu et al., 2024)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Optional scipy for exact OT
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from flowpo_hd.manifold_keeper import ManifoldKeeperMLP
from flowpo_hd.training.data_loader import (
    InstructionDataset,
    create_dataloader,
    load_or_encode_dataset,
)

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMAL TRANSPORT PAIRING
# =============================================================================

def compute_ot_plan_exact(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """Compute exact OT assignment using Hungarian algorithm.

    Returns permutation indices such that x_0[perm] is optimally paired with x_1.
    """
    if not SCIPY_AVAILABLE:
        logger.warning(
            "scipy not available - using approximate Sinkhorn OT instead of exact Hungarian. "
            "Install scipy for exact OT: pip install scipy"
        )
        return compute_ot_plan_approx(x_0, x_1)

    cost = torch.cdist(x_0, x_1, p=2).pow(2)
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    # row_ind[i] is the x_0 index that should be paired with x_1[col_ind[i]]
    # Since col_ind is always [0, 1, ..., n-1], row_ind is the permutation
    perm = torch.tensor(row_ind, device=x_0.device, dtype=torch.long)

    return perm


def compute_ot_plan_approx(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    num_iters: int = 20,
    reg: float = 0.05,
) -> torch.Tensor:
    """Approximate OT using Sinkhorn algorithm (GPU-friendly).

    Args:
        x_0: Source points (B, D)
        x_1: Target points (B, D)
        num_iters: Number of Sinkhorn iterations
        reg: Regularization parameter (must be positive)

    Returns:
        Permutation indices such that x_0[perm] is approximately optimally paired with x_1
    """
    if reg <= 0:
        raise ValueError(f"Sinkhorn regularization must be positive, got {reg}")

    B = x_0.shape[0]
    device = x_0.device

    C = torch.cdist(x_0, x_1, p=2).pow(2)
    K = torch.exp(-C / reg)
    u = torch.ones(B, device=device) / B

    for _ in range(num_iters):
        v = 1.0 / (K.T @ u + 1e-8)
        u = 1.0 / (K @ v + 1e-8)

    P = torch.diag(u) @ K @ torch.diag(v)
    indices = P.argmax(dim=0)

    return indices


def apply_ot_pairing(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    use_exact: bool = True,
    exact_threshold: int = 256,
) -> torch.Tensor:
    """Reorder x_0 to optimally match x_1 via Optimal Transport."""
    batch_size = x_0.shape[0]

    if use_exact and SCIPY_AVAILABLE and batch_size <= exact_threshold:
        perm = compute_ot_plan_exact(x_0, x_1)
    else:
        perm = compute_ot_plan_approx(x_0, x_1)

    return x_0[perm]


# =============================================================================
# TIMESTEP SAMPLING
# =============================================================================

def sample_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Uniform timestep sampling t ~ U[0, 1]."""
    return torch.rand(batch_size, device=device)


def sample_timesteps_u_shaped(
    batch_size: int,
    device: torch.device,
    a: float = 4.0,
) -> torch.Tensor:
    """
    U-shaped timestep distribution.

    Training loss is large at t≈0 and t≈1 but small in the middle.
    U-shaped distribution improves generation quality.
    """
    u = torch.rand(batch_size, device=device)
    centered = 2.0 * u - 1.0
    sign = torch.sign(centered)
    abs_centered = torch.abs(centered)
    t = 0.5 + 0.5 * sign * (1.0 - torch.exp(-a * abs_centered))
    return t.clamp(0.001, 0.999)


# =============================================================================
# TRAINING LOSS
# =============================================================================

def compute_flow_matching_loss(
    model: ManifoldKeeperMLP,
    x_1: torch.Tensor,
    timestep_sampling: Literal["uniform", "u_shaped"] = "u_shaped",
    use_ot: bool = True,
    u_shaped_a: float = 4.0,
    use_aux_losses: bool = False,
    aux_semantic_weight: float = 0.1,
    aux_norm_weight: float = 0.1,
    aux_proj_steps: int = 10,
    aux_t_threshold: float = 0.0,  # Only apply aux loss when t > threshold
) -> Dict[str, torch.Tensor]:
    """
    Compute OT-CFM loss for ManifoldKeeper training.

    Loss: E_t,x_0,x_1 [||v_θ(x_t, t) - (x_1 - x_0)||²]
          + λ_sem * (1 - cos_sim(x_1, proj(x_1)))  # Semantic preservation
          + λ_norm * MSE(||x_1||, ||proj(x_1)||)   # Norm preservation

    where x_t = t·x_1 + (1-t)·x_0 is the linear interpolation.

    Args:
        model: ManifoldKeeperMLP to train
        x_1: (B, D) data embeddings (target)
        timestep_sampling: "uniform" or "u_shaped"
        use_ot: Use optimal transport pairing
        u_shaped_a: Concentration for U-shaped sampling
        use_aux_losses: Enable auxiliary losses for semantic/norm preservation
        aux_semantic_weight: Weight for semantic preservation loss
        aux_norm_weight: Weight for norm preservation loss (MSE)
        aux_proj_steps: Number of ODE steps for projection (fewer = faster)
        aux_t_threshold: Only apply aux loss for samples where t > threshold (0.0 = all)

    Returns:
        Dict with loss and metrics
    """
    batch_size = x_1.shape[0]
    device = x_1.device

    # Sample source noise
    x_0 = torch.randn_like(x_1)

    # OT pairing: reorder x_0 to optimally match x_1
    if use_ot and batch_size > 1:
        x_0 = apply_ot_pairing(x_0, x_1)

    # Sample timesteps
    if timestep_sampling == "u_shaped":
        t = sample_timesteps_u_shaped(batch_size, device, a=u_shaped_a)
    else:
        t = sample_timesteps_uniform(batch_size, device)

    # Linear interpolation: x_t = t·x_1 + (1-t)·x_0
    t_view = t.view(-1, 1)
    x_t = t_view * x_1 + (1 - t_view) * x_0

    # Target velocity (straight line)
    u_t = x_1 - x_0

    # Predict velocity
    v_t = model(t, x_t)

    # Flow Matching loss (MSE)
    flow_loss = F.mse_loss(v_t, u_t)

    # Initialize auxiliary losses
    semantic_loss = torch.tensor(0.0, device=device)
    norm_loss = torch.tensor(0.0, device=device)
    aux_applied_ratio = torch.tensor(0.0, device=device)

    # Auxiliary losses for semantic and norm preservation
    if use_aux_losses:
        # Conditional: only apply aux loss for high-t samples (near manifold)
        if aux_t_threshold > 0:
            high_t_mask = t > aux_t_threshold
            n_high_t = high_t_mask.sum().item()

            if n_high_t > 0:
                x_1_high = x_1[high_t_mask]
                t_high = t[high_t_mask]

                # Project from current t to t=1.0
                x_proj = model.integrate(
                    x_1_high,
                    t_start=t_high.mean().item(),  # Average t for batch
                    t_end=1.0,
                    num_steps=aux_proj_steps,
                )

                # Semantic preservation
                cos_sim_proj = F.cosine_similarity(x_1_high, x_proj, dim=-1)
                semantic_loss = (1 - cos_sim_proj).mean()

                # Norm preservation
                orig_norms = x_1_high.norm(dim=-1)
                proj_norms = x_proj.norm(dim=-1)
                norm_loss = F.mse_loss(proj_norms, orig_norms)

                aux_applied_ratio = torch.tensor(n_high_t / batch_size, device=device)
        else:
            # Apply to all samples (original behavior)
            x_proj = model.integrate(
                x_1,
                t_start=0.5,
                t_end=1.0,
                num_steps=aux_proj_steps,
            )

            # Semantic preservation
            cos_sim_proj = F.cosine_similarity(x_1, x_proj, dim=-1)
            semantic_loss = (1 - cos_sim_proj).mean()

            # Norm preservation
            orig_norms = x_1.norm(dim=-1)
            proj_norms = x_proj.norm(dim=-1)
            norm_loss = F.mse_loss(proj_norms, orig_norms)

            aux_applied_ratio = torch.tensor(1.0, device=device)

    # Total loss
    total_loss = flow_loss
    if use_aux_losses and aux_applied_ratio > 0:
        total_loss = total_loss + aux_semantic_weight * semantic_loss + aux_norm_weight * norm_loss

    # Metrics
    with torch.no_grad():
        velocity_norm = v_t.norm(dim=-1).mean()
        target_norm = u_t.norm(dim=-1).mean()
        cos_sim = F.cosine_similarity(v_t, u_t, dim=-1).mean()

    result = {
        "loss": total_loss,
        "flow_loss": flow_loss,
        "velocity_norm": velocity_norm,
        "target_norm": target_norm,
        "cos_sim": cos_sim,
    }

    if use_aux_losses:
        result["semantic_loss"] = semantic_loss
        result["norm_loss"] = norm_loss
        result["aux_ratio"] = aux_applied_ratio

    return result


# =============================================================================
# TRAINING LOOP
# =============================================================================

class ManifoldKeeperTrainer:
    """Trainer for ManifoldKeeper with DDP support."""

    def __init__(
        self,
        model: ManifoldKeeperMLP,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        use_ot: bool = True,
        timestep_sampling: str = "u_shaped",
        u_shaped_a: float = 4.0,
        device: torch.device = None,
        checkpoint_dir: str = "flowpo_hd/checkpoints",
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        # Auxiliary losses for semantic/norm preservation
        use_aux_losses: bool = False,
        aux_semantic_weight: float = 0.1,
        aux_norm_weight: float = 0.1,
        aux_proj_steps: int = 10,
        aux_t_threshold: float = 0.0,
    ):
        """
        Initialize trainer.

        Args:
            model: ManifoldKeeperMLP to train
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
            lr: Learning rate
            grad_clip: Gradient clipping norm
            warmup_steps: LR warmup steps
            use_ot: Use optimal transport pairing
            timestep_sampling: Timestep sampling strategy
            u_shaped_a: U-shaped concentration parameter
            device: Torch device
            checkpoint_dir: Directory for checkpoints
            use_ddp: Whether to use DDP
            rank: Process rank for DDP
            world_size: Total processes for DDP
            use_aux_losses: Enable auxiliary losses for semantic/norm preservation
            aux_semantic_weight: Weight for semantic preservation loss
            aux_norm_weight: Weight for norm preservation loss (MSE)
            aux_proj_steps: Number of ODE steps for projection in aux losses
            aux_t_threshold: Only apply aux loss when t > threshold (0=all, 0.8=near manifold)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.use_ot = use_ot
        self.timestep_sampling = timestep_sampling
        self.u_shaped_a = u_shaped_a
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        # Auxiliary losses
        self.use_aux_losses = use_aux_losses
        self.aux_semantic_weight = aux_semantic_weight
        self.aux_norm_weight = aux_norm_weight
        self.aux_proj_steps = aux_proj_steps
        self.aux_t_threshold = aux_t_threshold

        # DDP wrapping
        if use_ddp:
            self.model = DDP(self.model, device_ids=[rank])

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Scheduler with warmup
        self.scheduler = None
        self.global_step = 0

        # Best metrics for checkpointing
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_lr_scale(self) -> float:
        """Get learning rate scale with warmup."""
        if self.global_step < self.warmup_steps:
            return self.global_step / self.warmup_steps
        return 1.0

    def _step_optimizer(self, loss: torch.Tensor) -> float:
        """Perform optimizer step with gradient clipping."""
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )
        else:
            grad_norm = 0.0

        # LR warmup
        lr_scale = self._get_lr_scale()
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr * lr_scale

        self.optimizer.step()
        self.global_step += 1

        return grad_norm

    def _unwrap_model(self) -> ManifoldKeeperMLP:
        """Get underlying model from DDP wrapper."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_cos_sim = 0.0
        n_batches = 0

        for batch in self.val_loader:
            x_1 = batch.to(self.device)
            result = compute_flow_matching_loss(
                self._unwrap_model(),
                x_1,
                timestep_sampling=self.timestep_sampling,
                use_ot=self.use_ot,
            )
            total_loss += result["loss"].item()
            total_cos_sim += result["cos_sim"].item()
            n_batches += 1

        self.model.train()

        return {
            "val_loss": total_loss / n_batches,
            "val_cos_sim": total_cos_sim / n_batches,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        model_state = self._unwrap_model().state_dict()

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            logger.info(f"Saved best checkpoint at epoch {epoch}")

        # Periodic save
        if epoch % 1000 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch}.pt")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns starting epoch.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint is missing required keys
            RuntimeError: If model state dict is incompatible
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except (RuntimeError, pickle.UnpicklingError) as e:
            # Checkpoint may contain non-tensor data (optimizer state, config)
            if "weights_only" in str(e).lower() or "unpickl" in str(e).lower():
                logger.warning(f"weights_only=True failed (legacy format?): {e}. Retrying.")
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            else:
                raise
        except Exception as e:
            # Let other errors (FileNotFoundError, PermissionError) propagate
            raise

        required_keys = ["model_state_dict", "epoch"]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")

        try:
            self._unwrap_model().load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Model state dict incompatible - checkpoint may be from different architecture: {e}"
            )

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        logger.info(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}")

        return checkpoint["epoch"]

    def train(
        self,
        epochs: int,
        log_interval: int = 100,
        val_interval: int = 500,
        save_interval: int = 1000,
        patience: int = 2000,
    ) -> Dict[str, list]:
        """
        Train ManifoldKeeper.

        Args:
            epochs: Number of training epochs
            log_interval: Steps between logging
            val_interval: Steps between validation
            save_interval: Steps between checkpointing
            patience: Early stopping patience (steps without improvement)

        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "cos_sim": [],
        }

        steps_without_improvement = 0
        start_time = datetime.now()

        self.model.train()

        for epoch in range(epochs):
            if self.use_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            for batch in self.train_loader:
                x_1 = batch.to(self.device)

                # Forward pass
                result = compute_flow_matching_loss(
                    self._unwrap_model(),
                    x_1,
                    timestep_sampling=self.timestep_sampling,
                    use_ot=self.use_ot,
                    u_shaped_a=self.u_shaped_a,
                    use_aux_losses=self.use_aux_losses,
                    aux_semantic_weight=self.aux_semantic_weight,
                    aux_norm_weight=self.aux_norm_weight,
                    aux_proj_steps=self.aux_proj_steps,
                    aux_t_threshold=self.aux_t_threshold,
                )

                # Backward pass
                grad_norm = self._step_optimizer(result["loss"])

                # Logging
                if self.global_step % log_interval == 0 and self.rank == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_msg = (
                        f"Step {self.global_step} | "
                        f"loss={result['loss'].item():.4f} | "
                        f"cos_sim={result['cos_sim'].item():.4f} | "
                        f"grad_norm={grad_norm:.4f} | "
                        f"lr={lr:.6f}"
                    )
                    if self.use_aux_losses:
                        log_msg += (
                            f" | sem={result['semantic_loss'].item():.4f}"
                            f" | norm={result['norm_loss'].item():.4f}"
                        )
                    logger.info(log_msg)
                    history["train_loss"].append(result["loss"].item())
                    history["cos_sim"].append(result["cos_sim"].item())

                # Validation
                if self.global_step % val_interval == 0:
                    val_metrics = self.validate()
                    if val_metrics and self.rank == 0:
                        logger.info(
                            f"Validation | "
                            f"loss={val_metrics['val_loss']:.4f} | "
                            f"cos_sim={val_metrics['val_cos_sim']:.4f}"
                        )
                        history["val_loss"].append(val_metrics["val_loss"])

                        # Check improvement
                        if val_metrics["val_loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["val_loss"]
                            steps_without_improvement = 0
                            self.save_checkpoint(epoch, is_best=True)
                        else:
                            steps_without_improvement += val_interval

                # Checkpointing
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(epoch)

                # Early stopping
                if patience > 0 and steps_without_improvement >= patience:
                    if self.rank == 0:
                        logger.info(f"Early stopping at step {self.global_step}")
                    return history

        # Final save
        self.save_checkpoint(epochs - 1)

        elapsed = datetime.now() - start_time
        if self.rank == 0:
            logger.info(f"Training completed in {elapsed}")

        return history


def train_manifold_keeper(
    config,
    resume_from: Optional[str] = None,
) -> ManifoldKeeperMLP:
    """
    Main training function for ManifoldKeeper.

    Args:
        config: FlowPOHDConfig
        resume_from: Optional checkpoint path to resume from

    Returns:
        Trained ManifoldKeeperMLP
    """
    device = torch.device(config.device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_or_encode_dataset(
        instructions_path=config.ape_instructions_path,
        embeddings_path=config.sonar_embeddings_path,
        device=config.device,
        normalize=config.sonar_normalize,
    )
    logger.info(f"Dataset stats: {dataset.get_stats()}")

    # Create data loaders
    train_loader = create_dataloader(
        dataset,
        batch_size=config.mk_batch_size,
        shuffle=True,
    )

    # Create model
    logger.info("Creating ManifoldKeeper...")
    model = ManifoldKeeperMLP(
        dim=config.sonar_dim,
        hidden_dim=config.mk_hidden_dim,
        time_dim=config.mk_time_dim,
        num_blocks=config.mk_num_blocks,
        dropout=config.mk_dropout,
    )
    logger.info(f"Parameters: {model.num_params:,}")

    # Create trainer
    trainer = ManifoldKeeperTrainer(
        model=model,
        train_loader=train_loader,
        lr=config.mk_lr,
        grad_clip=config.mk_grad_clip,
        warmup_steps=config.mk_warmup_steps,
        use_ot=config.mk_use_ot,
        timestep_sampling=config.mk_timestep_sampling,
        u_shaped_a=config.mk_u_shaped_a,
        device=device,
        checkpoint_dir=config.checkpoints_dir,
    )

    # Resume if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)

    # Train
    logger.info("Starting training...")
    trainer.train(
        epochs=config.mk_epochs,
        patience=config.mk_patience,
    )

    # Load best checkpoint
    best_path = Path(config.checkpoints_dir) / "best.pt"
    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        except (RuntimeError, pickle.UnpicklingError) as e:
            if "weights_only" in str(e).lower() or "unpickl" in str(e).lower():
                logger.warning(f"weights_only=True failed for best checkpoint: {e}")
                checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            else:
                raise
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best checkpoint with val_loss={checkpoint['best_val_loss']:.4f}")

    return model


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    print("Testing ManifoldKeeper training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create synthetic dataset
    print("\n--- Creating Synthetic Dataset ---")
    embeddings = torch.randn(500, 1024, device=device) * 0.2
    dataset = InstructionDataset(embeddings.cpu())
    train_loader = create_dataloader(dataset, batch_size=64)

    # Create model
    model = ManifoldKeeperMLP(
        dim=1024,
        hidden_dim=1024,  # Smaller for testing
        num_blocks=2,
    ).to(device)
    print(f"Model params: {model.num_params:,}")

    # Test loss computation
    print("\n--- Testing Loss Computation ---")
    batch = next(iter(train_loader)).to(device)
    result = compute_flow_matching_loss(model, batch)
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"Cos sim: {result['cos_sim'].item():.4f}")

    # Test a few training steps
    print("\n--- Testing Training Loop ---")
    trainer = ManifoldKeeperTrainer(
        model=model,
        train_loader=train_loader,
        lr=1e-4,
        warmup_steps=10,
        device=device,
        checkpoint_dir="/tmp/flowpo_hd_test",
    )

    # Train for a few steps
    history = trainer.train(
        epochs=1,
        log_interval=10,
        val_interval=50,
        patience=0,  # No early stopping
    )

    print(f"\nFinal loss: {history['train_loss'][-1]:.4f}")

    print("\n[OK] ManifoldKeeper training tests passed!")
