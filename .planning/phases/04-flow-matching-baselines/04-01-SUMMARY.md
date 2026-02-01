---
phase: 04-flow-matching-baselines
plan: 01
subsystem: flow-coupling
tags: [flow-matching, optimal-transport, coupling, torchcfm]

dependency-graph:
  requires: [03-baseline-architectures]
  provides: [coupling-abstraction, otcfm-support]
  affects: [04-02, 04-03]

tech-stack:
  added: []
  patterns: [factory-pattern, strategy-pattern]

key-files:
  created:
    - study/flow_matching/coupling/__init__.py
    - study/flow_matching/coupling/icfm.py
    - study/flow_matching/coupling/otcfm.py
  modified:
    - study/flow_matching/config.py
    - study/flow_matching/trainer.py

decisions:
  - id: D04-01-01
    choice: "Use OTPlanSampler from torchcfm with method='exact'"
    rationale: "Exact OT gives better coupling than approximate methods"
  - id: D04-01-02
    choice: "Default reg=0.5 and normalize_cost=True for high-D stability"
    rationale: "1024D SONAR embeddings need numerical stability"

metrics:
  duration: 5min
  completed: 2026-02-01
---

# Phase 04 Plan 01: Coupling Abstraction Summary

**One-liner:** Factory pattern coupling abstraction enabling I-CFM/OT-CFM switching via config.flow parameter with OTPlanSampler integration.

## What Was Done

### Task 1: Coupling Abstraction Module
Created `study/flow_matching/coupling/` with three files:

1. **icfm.py** - ICFMCoupling class wrapping existing I-CFM logic
   - Independent random pairing of x0, x1
   - Linear interpolation: x_t = (1-t)*x0 + t*x1
   - Constant velocity: u_t = x1 - x0

2. **otcfm.py** - OTCFMCoupling using torchcfm
   - Uses `OTPlanSampler(method="exact", reg=0.5, normalize_cost=True)`
   - Mini-batch Sinkhorn optimal transport coupling
   - Reorders x0 to minimize transport cost

3. **__init__.py** - Factory function
   - `create_coupling(method, **kwargs)` for method selection
   - Exports: `create_coupling`, `ICFMCoupling`, `OTCFMCoupling`

### Task 2: FlowTrainer Refactoring
Updated trainer and config to use coupling abstraction:

1. **config.py** changes:
   - Added `flow: str = "icfm"` default for backward compatibility
   - Added OT-CFM parameters: `otcfm_sigma`, `otcfm_reg`, `otcfm_normalize_cost`
   - Updated `to_dict()` to include OT-CFM parameters

2. **trainer.py** changes:
   - Import `create_coupling` from coupling module
   - Create coupling in `_setup()` based on `config.flow`
   - Replaced inline I-CFM logic in `train_epoch()` and `validate()` with `coupling.sample()`

## Key Implementation Details

### Coupling Interface
```python
class ICFMCoupling:
    def sample(self, x0, x1) -> tuple[t, x_t, u_t]

class OTCFMCoupling:
    def sample(self, x0, x1) -> tuple[t, x_t, u_t]
```

Both return:
- `t`: Uniformly sampled timesteps [B]
- `x_t`: Interpolated samples [B, D]
- `u_t`: Target velocity [B, D]

### OT-CFM Configuration
| Parameter | Default | Purpose |
|-----------|---------|---------|
| otcfm_sigma | 0.0 | Noise level (deterministic) |
| otcfm_reg | 0.5 | Sinkhorn regularization |
| otcfm_normalize_cost | True | Prevents overflow in high-D |

## Verification Results

### Coupling Module Tests
```
icfm: OK
otcfm: OK
```

### Training Smoke Tests (2 epochs, 1k dataset)

| Method | Train Loss | Val Loss | Notes |
|--------|------------|----------|-------|
| I-CFM | 2.004 | 2.011 | Baseline |
| OT-CFM | 1.859 | 1.884 | ~6% lower loss |

OT-CFM shows lower loss because optimal transport coupling produces straighter paths, reducing the complexity of the velocity field the model needs to learn.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| e23719e | feat | Add coupling abstraction module |
| 2a82c67 | refactor | Use coupling in FlowTrainer |

## Next Phase Readiness

**Ready for 04-02 (Validation Strategies):**
- Coupling abstraction allows easy A/B testing of I-CFM vs OT-CFM
- Config parameters logged to Wandb for experiment tracking
- Both methods verified working with MLP architecture

**Dependencies satisfied:**
- FlowTrainer supports both coupling methods
- Config includes all OT-CFM parameters
- Factory pattern enables easy extension for future coupling methods
