# Phase 2: Training Infrastructure - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Training system for all flow matching experiments with EMA, gradient clipping, early stopping, checkpointing, Wandb integration, and resumption support. All experiments run on GPU 1 (A5000) with CUDA_VISIBLE_DEVICES=1.

</domain>

<decisions>
## Implementation Decisions

### Experiment Organization
- Wandb runs grouped by ablation dimension (e.g., 'flow-methods', 'architectures', 'dataset-size')
- Descriptive run naming: `{arch}-{flow}-{dataset}-{aug}` format
- Names auto-generated from config — no manual naming required
- Full config logged as Wandb summary fields for filtering
- Failed/crashed runs auto-archived to separate 'failed' group

### Training Behavior
- EMA decay: 0.9999 (standard for flow matching)
- Gradient clipping: max_norm=1.0
- Early stopping patience: 20 epochs (prevent overfitting on small datasets)
- Validation frequency: every epoch
- Learning rate warmup: 1000 steps linear warmup

### Checkpoint Strategy
- Save best checkpoint only (by validation loss)
- Local storage: `study/checkpoints/{run_name}/best.pt`
- Wandb logs checkpoint path, does not upload artifacts
- Checkpoint includes: model state, EMA state, optimizer state, epoch, best loss

### Recovery & Resumption
- Explicit `--resume <path>` flag required to continue training
- Resume restores: model, EMA, optimizer, scheduler, epoch counter
- No auto-detection of interrupted runs
- Warmup skipped on resume (already warmed)

### Claude's Discretion
- Exact Wandb project name (suggest: `flow-matching-study`)
- Scheduler type (cosine annealing recommended)
- Logging frequency (every 10 steps reasonable)
- Validation batch size optimization

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for PyTorch training infrastructure.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-training-infrastructure*
*Context gathered: 2026-02-01*
