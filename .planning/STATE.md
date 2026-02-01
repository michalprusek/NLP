# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Find the best flow matching architecture for GP-guided prompt generation in SONAR space
**Current focus:** Phase 2 - Training Infrastructure (In Progress)

## Current Position

Phase: 2 of 11 (Training Infrastructure)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-02-01 -- Completed 02-01-PLAN.md (Core training infrastructure)

Progress: [███░░░░░░░] 15%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 49 min
- Total execution time: 2.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline | 2 | 133min | 67min |
| 02-training-infrastructure | 1 | 4min | 4min |

**Recent Trend:**
- Last 5 plans: 01-01 (124min), 01-02 (9min), 02-01 (4min)
- Trend: Improving (02-01 very fast, straightforward implementation)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Study location: study/flow_matching/ (separate from production ecoflow)
- Train on VS dataset for higher quality prompts
- Test 1K/5K/10K sample sizes for dataset scaling ablation
- GPU 1 (A5000) only for all experiments
- Unnormalized embeddings required for SONAR decoder compatibility
- Split pools first then subset for proper nested property across all splits (01-01)
- Semantic deduplication at 0.95 cosine similarity threshold (01-01)
- Pre-normalize embeddings at dataset load time for training efficiency (01-02)
- Store original embeddings alongside normalized for decoder access (01-02)
- Use epsilon 1e-8 clamp on std for numerical stability (01-02)
- Locked EMA decay at 0.9999, grad_clip at 1.0, patience at 20 (02-01)
- Flow matching uses ICFM formulation: x_t = (1-t)*x0 + t*x1 (02-01)
- GPU verification at startup warns if not on A5000/L40S or CUDA_VISIBLE_DEVICES!=1 (02-01)

### Pending Todos

None.

### Blockers/Concerns

- GPU 0 (A100 80GB) unavailable during 01-01 execution, used GPU 1 (A5000 24GB) which is slower
- 10K generation took longer than estimated (~114 min vs ~67 min) due to deduplication overhead
- 2 samples (0.4%) failed round-trip verification - acceptable edge cases (mixed language, grammar reconstruction)

## Session Continuity

Last session: 2026-02-01 08:13 UTC
Stopped at: Completed 02-01-PLAN.md
Resume file: None

## Phase 2 Progress

**Plan 01 complete.** Core training infrastructure established.

Delivered:
- TrainingConfig dataclass with locked EMA/patience/grad_clip defaults
- EarlyStopping, EMAModel, cosine schedule utilities
- FlowTrainer class with train/validate methods
- train.py CLI with GPU verification

Next: Plan 02 - Wandb integration and checkpoint saving
