# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Find the best flow matching architecture for GP-guided prompt generation in SONAR space
**Current focus:** Phase 3 - Baseline Architectures (In Progress)

## Current Position

Phase: 3 of 11 (Baseline Architectures)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-02-01 -- Completed 03-01-PLAN.md (SimpleMLP velocity network)

Progress: [█████░░░░░] 25%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: 31 min
- Total execution time: 2.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline | 2 | 133min | 67min |
| 02-training-infrastructure | 2 | 9min | 5min |
| 03-baseline-architectures | 1 | 2min | 2min |

**Recent Trend:**
- Last 5 plans: 01-02 (9min), 02-01 (4min), 02-02 (5min), 03-01 (2min)
- Trend: Fast execution continues (simple model creation)

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
- Checkpoint save/load order: model -> EMA -> optimizer -> scheduler (02-02)
- Stats consistency check on resume warns but doesn't fail (02-02)
- Wandb.log every 10 steps with step parameter for proper alignment (02-02)
- SimpleMLP uses hidden_dim=256, num_layers=5 for ~920K params (03-01)
- Output layer initialized near zero (std=0.01) for stable training (03-01)
- Model factory pattern: create_model(arch_name) for CLI selection (03-01)

### Pending Todos

None.

### Blockers/Concerns

- GPU 0 (A100 80GB) unavailable during 01-01 execution, used GPU 1 (A5000 24GB) which is slower
- 10K generation took longer than estimated (~114 min vs ~67 min) due to deduplication overhead
- 2 samples (0.4%) failed round-trip verification - acceptable edge cases (mixed language, grammar reconstruction)

## Session Continuity

Last session: 2026-02-01 09:11 UTC
Stopped at: Completed 03-01-PLAN.md
Resume file: None

## Phase 3 Progress

**Phase 3 in progress.** First baseline architecture (SimpleMLP) complete.

Delivered (03-01):
- SimpleMLP velocity network (~920K params)
- Sinusoidal timestep embedding function
- Model factory create_model() for CLI architecture selection
- Smoke test verified training stability

Next: 03-02-PLAN.md (DiT velocity network)
