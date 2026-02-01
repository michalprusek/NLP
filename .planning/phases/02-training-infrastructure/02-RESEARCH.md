# Phase 2: Training Infrastructure - Research

**Researched:** 2026-02-01
**Domain:** PyTorch training infrastructure with Wandb experiment tracking
**Confidence:** HIGH

## Summary

This research covers the technical implementation for building a training infrastructure for flow matching experiments on SONAR embeddings. The phase involves creating a training loop with EMA (Exponential Moving Average), gradient clipping, early stopping, checkpoint management, Wandb integration for experiment tracking, and training resumption support. All experiments run on GPU 1 (A5000) with `CUDA_VISIBLE_DEVICES=1`.

The existing codebase in `ecoflow/train_flow.py` provides an excellent foundation with EMA, gradient clipping, cosine annealing with warmup, and basic checkpointing. The Phase 1 data pipeline (`study/data/dataset.py`) provides `FlowDataset` and `create_dataloader()` utilities ready for use. The key additions needed are: (1) Wandb integration with proper grouping by ablation dimension, (2) early stopping based on validation loss, (3) best-only checkpoint strategy, and (4) explicit resume support.

**Primary recommendation:** Use the existing `EMAModel` class pattern from `ecoflow/train_flow.py`, integrate Wandb with `group` parameter for ablation organization, implement early stopping as a simple class tracking patience, and save only the best checkpoint based on validation loss.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.8+ | Training framework | Core framework, already in use |
| wandb | 0.19+ | Experiment tracking | NeurIPS-standard, rich visualization |
| torch.optim.AdamW | built-in | Optimizer | Standard for transformer-style models |
| torch.nn.utils.clip_grad_norm_ | built-in | Gradient clipping | Prevents exploding gradients |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.optim.lr_scheduler.LambdaLR | built-in | Custom LR schedule | Cosine annealing with warmup |
| torch.Generator | built-in | Reproducible shuffling | DataLoader seeding |
| pathlib.Path | stdlib | Path handling | Checkpoint directory management |
| copy.deepcopy | stdlib | EMA state copy | Shadow parameter storage |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom EMA class | ema-pytorch 0.7.9 | ema-pytorch adds features but adds dependency; custom is simpler |
| Custom EarlyStopping | pytorch-ignite EarlyStopping | Ignite adds large dependency; custom class is 20 lines |
| LambdaLR custom | CosineAnnealingLR | LambdaLR allows warmup; CosineAnnealingLR does not |
| wandb | tensorboard | wandb has better grouping, filtering, and collaboration |

**Installation:**
```bash
# wandb is likely already installed, if not:
uv add wandb
```

## Architecture Patterns

### Recommended Project Structure
```
study/
├── flow_matching/
│   ├── train.py            # Main training script
│   ├── config.py           # TrainingConfig dataclass
│   ├── trainer.py          # Trainer class with wandb, EMA, early stopping
│   └── utils.py            # EarlyStopping, checkpoint utilities
├── checkpoints/
│   └── {run_name}/
│       └── best.pt         # Best checkpoint only
├── data/                   # Phase 1 data loading (existing)
└── datasets/               # Phase 1 split files (existing)
```

### Pattern 1: Wandb Initialization with Grouping
**What:** Initialize Wandb run with group for ablation organization
**When to use:** At training start
**Example:**
```python
# Source: https://docs.wandb.ai/models/runs/grouping
import wandb

def init_wandb(config: dict, group: str) -> wandb.Run:
    """Initialize Wandb run with proper grouping."""
    # Auto-generate run name from config
    run_name = f"{config['arch']}-{config['flow']}-{config['dataset']}-{config['aug']}"

    run = wandb.init(
        project="flow-matching-study",  # Recommended project name
        group=group,                     # e.g., 'flow-methods', 'architectures'
        name=run_name,
        config=config,
        resume="allow",                  # Allow resuming if run_id provided
    )

    # Define metrics for automatic summary
    run.define_metric("train/loss", summary="min")
    run.define_metric("val/loss", summary="min")
    run.define_metric("epoch", step_metric="epoch")

    return run
```

### Pattern 2: EMA Model Wrapper
**What:** Exponential moving average of model parameters
**When to use:** Always for flow matching (standard practice)
**Example:**
```python
# Source: ecoflow/train_flow.py (existing implementation)
import copy
from typing import Dict
import torch
import torch.nn as nn

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with EMA."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module) -> None:
        """Apply shadow parameters to model for inference."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict for checkpointing."""
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load state dict from checkpoint."""
        self.shadow = copy.deepcopy(state_dict)
```

### Pattern 3: Early Stopping
**What:** Stop training when validation loss stops improving
**When to use:** Every validation epoch
**Example:**
```python
# Source: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop. Returns True if improved."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # Did not improve
```

### Pattern 4: Cosine Annealing with Linear Warmup
**What:** Learning rate schedule with warmup then cosine decay
**When to use:** Standard for transformer training
**Example:**
```python
# Source: ecoflow/train_flow.py (existing implementation)
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create schedule with linear warmup and cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
```

### Pattern 5: Best-Only Checkpoint Saving
**What:** Save checkpoint only when validation improves
**When to use:** When early stopping indicates improvement
**Example:**
```python
# Source: PyTorch documentation + ecoflow/train_flow.py
import os
import torch

def save_best_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_loss: float,
    config: dict,
    stats: dict,
) -> None:
    """Save checkpoint with verification."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "config": config,
        "normalization_stats": stats,
    }

    torch.save(checkpoint, path)

    # Verify saved correctly
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"Checkpoint file missing or empty: {path}")
```

### Pattern 6: Training Resume
**What:** Resume training from checkpoint with explicit flag
**When to use:** When `--resume <path>` is provided
**Example:**
```python
# Source: https://docs.wandb.ai/guides/runs/resuming
def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_loss)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    ema.load_state_dict(checkpoint["ema_shadow"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]

    return start_epoch, best_loss
```

### Pattern 7: Wandb Logging
**What:** Log metrics and handle failures
**When to use:** Every log_interval steps and every epoch
**Example:**
```python
# Source: https://docs.wandb.ai/guides/track/log
import wandb

def log_training_step(
    step: int,
    loss: float,
    grad_norm: float,
    lr: float,
    epoch: int,
) -> None:
    """Log training metrics."""
    wandb.log({
        "train/loss": loss,
        "train/grad_norm": grad_norm,
        "train/lr": lr,
        "epoch": epoch,
    }, step=step)

def log_validation(
    epoch: int,
    val_loss: float,
    best_loss: float,
    checkpoint_path: str | None = None,
) -> None:
    """Log validation metrics."""
    metrics = {
        "val/loss": val_loss,
        "val/best_loss": best_loss,
        "epoch": epoch,
    }
    if checkpoint_path:
        metrics["checkpoint_path"] = checkpoint_path

    wandb.log(metrics)

def handle_failed_run(error: Exception) -> None:
    """Archive failed run to 'failed' group."""
    if wandb.run is not None:
        wandb.run.tags = list(wandb.run.tags) + ["failed"]
        wandb.run.notes = f"Failed with: {error}"
        wandb.finish(exit_code=1)
```

### Anti-Patterns to Avoid
- **Logging every step:** High overhead; log every 10 steps instead
- **Saving checkpoints every epoch:** Wastes disk; save best only
- **Uploading checkpoints to Wandb:** Use local storage, log path only
- **Auto-detecting resume:** Explicit `--resume` flag is safer and clearer
- **Warmup on resume:** Skip warmup since already warmed; use scheduler state_dict
- **Computing EMA before first step:** Initialize EMA after model is on correct device
- **Forgetting to denormalize for inference:** EMA model operates in normalized space

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Learning rate scheduling | Manual LR decay | `LambdaLR` with custom function | Handles edge cases, step tracking |
| Gradient clipping | Manual norm computation | `clip_grad_norm_` | Handles inf/nan, returns actual norm |
| Reproducible DataLoader | Manual seeding | `Generator` + `worker_init_fn` | Handles multi-worker seeding correctly |
| Experiment grouping | Custom tagging | Wandb `group` parameter | Built-in UI support, filtering |
| Run naming | Manual string concat | Config-based auto-naming | Consistent, searchable |

**Key insight:** PyTorch and Wandb provide battle-tested utilities for common training patterns. Custom implementations often miss edge cases (nan gradients, multi-worker seeding, etc.).

## Common Pitfalls

### Pitfall 1: Wandb Step Synchronization
**What goes wrong:** Metrics logged with different step counts appear misaligned in UI
**Why it happens:** Calling `wandb.log()` without explicit `step` parameter uses auto-incrementing internal step
**How to avoid:** Always pass `step=global_step` to `wandb.log()`, or use `define_metric` with `step_metric`
**Warning signs:** Train and validation curves don't align; epoch jumps in charts

### Pitfall 2: EMA Initialization Order
**What goes wrong:** EMA shadow contains CPU tensors when model is on GPU
**Why it happens:** Creating EMA before moving model to GPU
**How to avoid:** Create EMA after `model.to(device)`
**Warning signs:** Device mismatch errors; slow training due to CPU-GPU transfers

### Pitfall 3: Resume Without Scheduler State
**What goes wrong:** Learning rate restarts from initial value on resume
**Why it happens:** Only loading model and optimizer state, forgetting scheduler
**How to avoid:** Save and load `scheduler.state_dict()` in checkpoint
**Warning signs:** LR jumps up on resume; training destabilizes

### Pitfall 4: Early Stopping Without Best Model
**What goes wrong:** Final model is from last epoch, not best validation loss
**Why it happens:** Early stopping triggers but no checkpoint was saved at best epoch
**How to avoid:** Save checkpoint whenever validation improves, before checking patience
**Warning signs:** Test metrics worse than validation metrics at best epoch

### Pitfall 5: Validation During Warmup
**What goes wrong:** Early stopping triggers during warmup when loss is high
**Why it happens:** Running validation from epoch 1 while LR is still warming up
**How to avoid:** Start early stopping patience after warmup completes (or use high initial patience)
**Warning signs:** Training stops after few epochs with high loss

### Pitfall 6: Multi-GPU Wandb Conflicts
**What goes wrong:** Multiple processes try to log to same Wandb run
**Why it happens:** Not checking rank before Wandb calls
**How to avoid:** Only log from rank 0 process; use `if rank == 0: wandb.log(...)`
**Warning signs:** N/A - this study uses single GPU (A5000) per context

### Pitfall 7: Checkpoint Path Collisions
**What goes wrong:** Different runs overwrite each other's checkpoints
**Why it happens:** Using generic checkpoint path without run name
**How to avoid:** Include run_name in checkpoint path: `study/checkpoints/{run_name}/best.pt`
**Warning signs:** Checkpoint modified timestamp doesn't match training time

## Code Examples

Verified patterns from official sources:

### Complete Training Loop Structure
```python
# Source: ecoflow/train_flow.py + wandb docs + PyTorch docs
import os
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW

def train(config: dict, resume_path: str | None = None):
    """Main training function with Wandb, EMA, and early stopping."""

    # Set device (CUDA_VISIBLE_DEVICES=1 is set externally)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Wandb
    run_name = f"{config['arch']}-{config['flow']}-{config['dataset']}-{config['aug']}"
    run = wandb.init(
        project="flow-matching-study",
        group=config.get("group", "default"),
        name=run_name,
        config=config,
        resume="allow" if resume_path else "never",
    )

    # Define metric summaries
    run.define_metric("val/loss", summary="min")

    # Create model, optimizer, scheduler
    model = create_model(config).to(device)
    ema = EMAModel(model, decay=0.9999)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    total_steps = config["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", 1000),
        num_training_steps=total_steps,
    )

    # Resume if requested
    start_epoch = 1
    best_loss = float("inf")
    if resume_path:
        start_epoch, best_loss = load_checkpoint(
            resume_path, model, ema, optimizer, scheduler, device
        )
        # Note: warmup is skipped since scheduler state is restored

    # Early stopping
    early_stopping = EarlyStopping(patience=20)
    early_stopping.best_loss = best_loss  # Restore best loss if resuming

    # Checkpoint path
    checkpoint_dir = f"study/checkpoints/{run_name}"
    checkpoint_path = f"{checkpoint_dir}/best.pt"

    # Training loop
    global_step = (start_epoch - 1) * len(train_loader)

    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            model.train()
            epoch_loss = 0.0

            for batch_idx, x1 in enumerate(train_loader):
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)

                # Flow matching loss
                t = torch.rand(x1.shape[0], device=device)
                x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
                v_target = x1 - x0
                v_pred = model(x_t, t)
                loss = F.mse_loss(v_pred, v_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                ema.update(model)

                epoch_loss += loss.item()
                global_step += 1

                # Log every 10 steps
                if global_step % 10 == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }, step=global_step)

            # Validation
            val_loss = validate(model, val_loader, device)

            # Check if improved
            improved = early_stopping(val_loss)

            if improved:
                best_loss = val_loss
                save_best_checkpoint(
                    checkpoint_path, epoch, model, ema, optimizer,
                    scheduler, best_loss, config, normalization_stats
                )

            # Log validation
            wandb.log({
                "val/loss": val_loss,
                "val/best_loss": best_loss,
                "epoch": epoch,
            }, step=global_step)

            # Check early stopping
            if early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Log final summary
        wandb.summary["final_epoch"] = epoch
        wandb.summary["best_val_loss"] = best_loss
        wandb.summary["checkpoint_path"] = checkpoint_path

    except Exception as e:
        # Mark run as failed
        wandb.run.tags = list(wandb.run.tags or []) + ["failed"]
        wandb.finish(exit_code=1)
        raise

    wandb.finish()
```

### Validation Function
```python
# Source: PyTorch documentation
@torch.no_grad()
def validate(model, val_loader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x1 in val_loader:
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)

        t = torch.rand(x1.shape[0], device=device)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        v_target = x1 - x0
        v_pred = model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / n_batches
```

### GPU Configuration
```python
# Source: PyTorch documentation
import os
import torch

def setup_device():
    """Configure GPU 1 (A5000) for training."""
    # CUDA_VISIBLE_DEVICES=1 should be set in environment
    # This makes GPU 1 appear as cuda:0 within the process

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda:0")

    # Verify correct GPU
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TensorBoard logging | Wandb with grouping | 2023-2024 | Better collaboration, filtering |
| Linear LR decay | Cosine annealing + warmup | 2019 (standard now) | Better convergence |
| Save every epoch | Save best only | Standard practice | Reduces storage |
| Fixed patience | Patience after warmup | Standard practice | Avoids false early stops |
| timm ModelEmaV2 | Custom EMA or ema-pytorch | 2024 | More control, simpler |

**Deprecated/outdated:**
- MLflow for academic research: Wandb is now standard for NeurIPS-tier papers
- Manual LR schedules: Use `LambdaLR` or built-in schedulers
- Saving all checkpoints: Wastes storage; save best only unless debugging

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal validation batch size**
   - What we know: Larger batch sizes use more memory but faster
   - What's unclear: Exact memory footprint for A5000 (24GB) with 1024D embeddings
   - Recommendation: Start with training batch size, increase until OOM or diminishing returns

2. **Failed run archival mechanism**
   - What we know: Can add "failed" tag via `wandb.run.tags`
   - What's unclear: Whether Wandb supports moving runs to different groups after creation
   - Recommendation: Add "failed" tag and set notes; manual group movement in UI if needed

3. **Exact step for warmup completion detection on resume**
   - What we know: Scheduler state_dict contains step count
   - What's unclear: Best way to check if warmup already completed
   - Recommendation: Check `scheduler.last_epoch >= warmup_steps` or skip warmup check entirely (scheduler handles it)

## Sources

### Primary (HIGH confidence)
- [Wandb Grouping Documentation](https://docs.wandb.ai/models/runs/grouping) - Group parameter, UI organization
- [Wandb Run Resumption](https://docs.wandb.ai/guides/runs/resuming) - Resume modes, run ID management
- [Wandb Summary Metrics](https://docs.wandb.ai/models/track/log/log-summary) - define_metric, automatic summaries
- [Wandb Custom Axes](https://docs.wandb.ai/models/track/log/customize-logging-axes) - define_metric for step_metric
- [Wandb init Parameters](https://docs.wandb.ai/ref/python/init) - project, name, group, config
- PyTorch Documentation - clip_grad_norm_, LambdaLR, DataLoader
- Existing codebase: `ecoflow/train_flow.py` - EMAModel, cosine schedule, checkpoint pattern
- Existing codebase: `study/data/dataset.py` - FlowDataset, create_dataloader

### Secondary (MEDIUM confidence)
- [ema-pytorch GitHub](https://github.com/lucidrains/ema-pytorch) - EMA wrapper patterns
- [pytorch_ema GitHub](https://github.com/fadel/pytorch_ema) - state_dict/load_state_dict patterns
- [PyTorch Lightning Early Stopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html) - Patience patterns
- [early-stopping-pytorch GitHub](https://github.com/Bjarten/early-stopping-pytorch) - Simple early stopping class
- [Cosine Annealing PyTorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) - Scheduler API

### Tertiary (LOW confidence)
- WebSearch results on EMA best practices - confirmed with official sources
- WebSearch results on early stopping patterns - confirmed with GitHub implementations

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch and Wandb are well-documented, patterns verified
- Architecture: HIGH - Based on existing ecoflow implementation + official docs
- Pitfalls: HIGH - Verified against official documentation and community resources

**Research date:** 2026-02-01
**Valid until:** 2026-04-01 (60 days - stable domain, well-established patterns)
