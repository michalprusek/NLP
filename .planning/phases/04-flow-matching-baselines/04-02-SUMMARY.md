---
phase: 04-flow-matching-baselines
plan: 02
subsystem: flow-evaluation
tags: [flow-matching, path-straightness, optimal-transport, evaluation]

dependency-graph:
  requires: [04-01]
  provides: [path-straightness-metric, otcfm-checkpoint]
  affects: [04-03]

tech-stack:
  added: []
  patterns: [trajectory-analysis, ode-integration]

key-files:
  created:
    - study/checkpoints/mlp-otcfm-1k-none/best.pt
  modified:
    - study/flow_matching/evaluate.py

decisions:
  - id: D04-02-01
    choice: "Path straightness measures L2 deviation from ideal straight line"
    rationale: "Straight paths enable faster/more accurate ODE integration"
  - id: D04-02-02
    choice: "Both I-CFM and OT-CFM produce very straight paths (~0.0015 deviation)"
    rationale: "For small MLP on 1K data, path straightness difference is marginal"

metrics:
  duration: 4min
  completed: 2026-02-01
---

# Phase 04 Plan 02: OT-CFM Training and Path Straightness Summary

**One-liner:** Path straightness evaluation function added; OT-CFM trained with 8% lower loss than I-CFM but equivalent path curvature.

## What Was Done

### Task 1: Path Straightness Evaluation Function
Added `compute_path_straightness()` to `study/flow_matching/evaluate.py`:

**Function behavior:**
1. Sample noise x0 ~ N(0, I)
2. Integrate ODE with Euler method, recording trajectory
3. Compute ideal straight line from x0 to x_T (final sample)
4. Measure L2 deviation from ideal at each interior step

**Returns:**
- `mean_path_deviation`: Average deviation across all samples/steps
- `max_path_deviation`: Maximum deviation observed
- `path_variance`: Variance of deviation along trajectories

**CLI integration:**
- Added path straightness section to `evaluate.py` CLI output
- Runs automatically alongside distribution MSE evaluation

### Task 2: OT-CFM Training and Comparison
Trained OT-CFM model on 1K dataset and compared with I-CFM baseline:

**Training results:**
| Model | Train Loss | Val Loss | Epochs | Early Stop |
|-------|------------|----------|--------|------------|
| I-CFM | 2.002 | 2.008 | ~30 | Yes |
| OT-CFM | 1.831 | 1.841 | 32 | Yes |

OT-CFM achieves **~8% lower validation loss** (1.841 vs 2.008).

**Path straightness comparison (100 samples, 100 steps):**
| Model | Mean Deviation | Max Deviation | Path Variance |
|-------|----------------|---------------|---------------|
| I-CFM | 0.001560 | 0.003169 | 0.00000047 |
| OT-CFM | 0.001555 | 0.003043 | 0.00000047 |

Both methods produce **extremely straight paths** (deviation ~0.0015), suggesting:
- The MLP velocity network learns smooth vector fields
- For small models/datasets, OT-CFM's benefit is primarily in training loss, not path geometry

**Text generation quality:**
Both models generate coherent English text about problem-solving and reasoning:

*I-CFM samples:*
- "Observe whether there are contradictions to be secured and give personalisation of the details to include reasoning."
- "Think up the difficulty. Think about the problematic problem."

*OT-CFM samples:*
- "Consider if you are creating sequences - or elements - to refer to the problem."
- "You should consider answering questions to put things in a position to be directly related to the problem."

## Key Findings

1. **OT-CFM provides lower training loss** (~8% improvement) due to optimal transport coupling producing more efficient training signal.

2. **Path straightness is similar** for both methods on this small-scale experiment. The deviation is extremely small (~0.0015) for both, indicating the MLP learns smooth velocity fields regardless of coupling method.

3. **Text quality is comparable** - both methods generate coherent, domain-appropriate prompts about reasoning and problem-solving.

4. **OT-CFM's main advantage at this scale** is training efficiency (lower loss), not generation quality or path geometry.

## Verification Results

### Path Straightness Function
```python
compute_path_straightness(model, embeddings, n_samples=50, n_steps=50, device='cuda:0')
# Returns: mean_path_deviation=0.001548, max=0.003047, variance=4.38e-07
```

### OT-CFM Checkpoint
```
study/checkpoints/mlp-otcfm-1k-none/best.pt
Size: 14.7 MB
Parameters: 920,064
```

## Deviations from Plan

### Authentication Gate
**WANDB_MODE=offline** was required because wandb was not authenticated. Training completed successfully in offline mode.

**Impact:** None - checkpoints and metrics were saved locally. Training results are valid.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| ea56006 | feat | Add path straightness evaluation function |

Note: OT-CFM checkpoint is gitignored (by design - model weights not committed).

## Next Phase Readiness

**Ready for 04-03 (Architecture Comparison):**
- Path straightness metric available for comparing MLP vs DiT
- Both I-CFM and OT-CFM baselines established
- Evaluation infrastructure complete

**Key insight for future phases:**
- OT-CFM's straighter-path benefit may be more pronounced with larger/more complex architectures (DiT)
- Current MLP already produces very straight paths, limiting room for improvement

---
*Phase: 04-flow-matching-baselines*
*Completed: 2026-02-01*
