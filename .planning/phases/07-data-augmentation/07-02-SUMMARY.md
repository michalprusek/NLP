---
phase: 07-data-augmentation
plan: 02
subsystem: training
tags: [augmentation, dropout, ablation, flow-matching]

# Dependency graph
requires:
  - phase: 07-01
    provides: mixup and noise augmentation with AugmentationConfig
provides:
  - dimension_dropout function using F.dropout
  - parse_aug_string helper for config parsing
  - Complete augmentation module with all three strategies
  - Ablation results comparing augmentation methods
affects: [07-03, 09-experimental-matrix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "F.dropout for dimension masking with auto-scaling"
    - "Augmentation order: mixup -> noise -> dropout"

key-files:
  created: []
  modified:
    - study/data/augmentation.py
    - study/flow_matching/train.py

key-decisions:
  - "F.dropout IS dimension masking - satisfies DATA-07 dropout/masking requirement"
  - "Augmentation order: mixup -> noise -> dropout"
  - "Augmentation did NOT improve val loss in 30-epoch ablation (marginal ~0.2% difference)"

patterns-established:
  - "parse_aug_string() for config shorthand: 'mixup', 'noise', 'dropout', 'all'"

# Metrics
duration: 6min
completed: 2026-02-01
---

# Phase 7 Plan 2: Dimension Dropout and Ablation Summary

**Complete dimension dropout augmentation with ablation showing marginal impact on val loss for short training runs**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-01T13:23:26Z
- **Completed:** 2026-02-01T13:29:05Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Added dimension_dropout() using F.dropout with automatic scaling
- Updated augment_batch() to apply all three augmentations in order: mixup -> noise -> dropout
- Added parse_aug_string() helper for config shorthand parsing
- Added --dropout-rate CLI argument
- Ran ablation comparing baseline vs mixup vs noise vs mixup+noise
- Documented finding: augmentation has marginal impact (~0.2%) in short training

## Task Commits

Each task was committed atomically:

1. **Task 1: Add dimension dropout to augmentation module** - `7608bb3` (feat)
2. **Task 2: Update CLI with dropout argument** - `7572033` (feat)
3. **Task 3: Run ablation comparing augmentation methods** - (no file changes, checkpoints saved)

## Files Created/Modified

- `study/data/augmentation.py` - Added dimension_dropout(), updated augment_batch(), added parse_aug_string()
- `study/flow_matching/train.py` - Added --dropout-rate CLI argument

## Ablation Results

| Aug Method    | Best Val Loss | Delta from Baseline |
|---------------|---------------|---------------------|
| none          | 1.992204      | baseline            |
| mixup         | 1.996063      | +0.0039 (+0.19%)    |
| noise         | 1.995363      | +0.0032 (+0.16%)    |
| mixup+noise   | 1.996343      | +0.0041 (+0.21%)    |

**Finding:** In this 30-epoch ablation with 800 training samples, augmentation did NOT improve generalization. The baseline (no augmentation) had the lowest val loss.

**Analysis:** The differences are marginal (~0.2%). All methods converged to similar val loss (~1.99). This suggests:
1. Augmentation may help more with longer training (overfitting prevention)
2. The 1K dataset may already have sufficient diversity
3. Augmentation benefits may emerge with larger models

**Recommendation:** For short training runs, augmentation is optional. For production training (100+ epochs), consider mixup+noise with default parameters.

## Phase 7 Verification Checklist

1. [x] Mixup generates valid training pairs (similarity ~0.58)
2. [x] Gaussian noise augments training data (stats preserved)
3. [x] Dimension dropout/masking augments training data (F.dropout with ~10% zeros)
4. [x] Combined augmentations work together (mixup -> noise -> dropout)
5. [x] CLI supports --dropout-rate argument
6. [x] Ablation runs complete with results documented

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| F.dropout for dimension masking | F.dropout IS stochastic masking - zeros random dimensions. Satisfies DATA-07 requirement. |
| Augmentation order: mixup -> noise -> dropout | Research-based ordering, consistent with 07-01 |
| Default dropout_rate=0.1 in parse_aug_string | Matches mixup/noise defaults for consistency |

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed successfully.

## Next Phase Readiness

- All three augmentation strategies implemented and verified
- Ablation results provide baseline for future comparison
- Ready for Phase 7 Plan 3 (augmentation ablation with longer training)
- Augmentation infrastructure ready for Phase 9 experimental matrix

---
*Phase: 07-data-augmentation*
*Completed: 2026-02-01*
