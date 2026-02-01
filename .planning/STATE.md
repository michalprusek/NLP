# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Find the best flow matching architecture for GP-guided prompt generation in SONAR space
**Current focus:** Phase 5 - Advanced Flow Methods (in progress)

## Current Position

Phase: 5 of 11 (Advanced Flow Methods)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-02-01 -- Completed 05-01-PLAN.md (Reflow/Rectified Flow)

Progress: [████████░░] 46%

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 18 min
- Total execution time: 3.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline | 2 | 133min | 67min |
| 02-training-infrastructure | 2 | 9min | 5min |
| 03-baseline-architectures | 3 | 10min | 3min |
| 04-flow-matching-baselines | 3 | 12min | 4min |
| 05-advanced-flow-methods | 1 | 5min | 5min |

**Recent Trend:**
- Last 5 plans: 04-01 (5min), 04-02 (4min), 04-03 (3min), 05-01 (5min)
- Trend: Fast execution continues

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
- DiTVelocityNetwork uses hidden_dim=384, num_layers=3, num_heads=6 for ~9.3M params (03-02)
- Import timestep_embedding from mlp.py to avoid code duplication (03-02)
- Velocity prediction loss ~2.0 is expected for normalized ICFM training (03-02)
- ICFM distribution MSE ~1.0 is expected (comparing generated to random targets) (03-03)
- Key quality metric is coherent text generation, not reconstruction MSE (03-03)
- OT-CFM uses OTPlanSampler with method='exact', reg=0.5, normalize_cost=True (04-01)
- Factory pattern for coupling selection: create_coupling(method) (04-01)
- OT-CFM produces ~8% lower val loss than I-CFM (1.841 vs 2.008) (04-02)
- Path straightness similar for both methods (~0.0016 deviation) on small MLP (04-02/04-03)
- CFG-Zero* uses 4% zero-init fraction matching ecoflow/guided_flow.py (04-03)
- Gradient clipping at max_grad_norm=10.0 for guidance stability (04-03)
- Reflow uses 10K pairs (10x dataset) from OT-CFM teacher (05-01)
- Reflow pairs cached at study/datasets/reflow_pairs_1k.pt (05-01)

### Pending Todos

None.

### Blockers/Concerns

- GPU 0 (A100 80GB) unavailable during 01-01 execution, used GPU 1 (A5000 24GB) which is slower
- 10K generation took longer than estimated (~114 min vs ~67 min) due to deduplication overhead
- 2 samples (0.4%) failed round-trip verification - acceptable edge cases (mixed language, grammar reconstruction)
- WANDB authentication missing - using WANDB_MODE=offline for training runs
- GPU 1 memory pressure from other processes - use GPU 0 for SONAR decoder when needed

## Session Continuity

Last session: 2026-02-01 11:34 UTC
Stopped at: Completed 05-01-PLAN.md (Reflow)
Resume file: None

## Phase 5 Progress

**Phase 5 in progress.** Advanced flow methods.

Delivered (05-01):
- ReflowPairGenerator for synthetic pair generation via teacher ODE
- ReflowCoupling for training on pre-generated pairs
- Training script study/flow_matching/reflow/train_reflow.py
- 2-rectified flow checkpoint: study/checkpoints/mlp-reflow-1k-none/best.pt
- Path straightness 3.1x better than I-CFM (0.00052 vs 0.0016)

**Reflow results (100 epochs, 10K pairs from OT-CFM teacher):**
| Metric | Reflow | I-CFM | OT-CFM |
|--------|--------|-------|--------|
| Train Loss | 0.000038 | 2.008 | 1.841 |
| Mean Path Dev | 0.00052 | 0.0016 | 0.0016 |
| Text Quality | Coherent | Coherent | Coherent |

**Key finding:** Reflow produces significantly straighter paths (3x) by training on deterministically coupled pairs from the teacher.

Remaining plans:
- 05-02: Stochastic Interpolant with GVP schedule
- 05-03: Minibatch OT coupling

## Phase 4 Summary (COMPLETE)

**Phase 4 complete.** Flow matching baselines established.

Delivered (04-01):
- Coupling abstraction module (study/flow_matching/coupling/)
- ICFMCoupling and OTCFMCoupling classes
- Factory function create_coupling() for method selection
- FlowTrainer refactored to use coupling abstraction
- OT-CFM config parameters (sigma, reg, normalize_cost)

Delivered (04-02):
- Path straightness evaluation function (compute_path_straightness)
- OT-CFM checkpoint: study/checkpoints/mlp-otcfm-1k-none/best.pt
- Quantitative comparison: OT-CFM ~8% lower loss, similar path geometry

Delivered (04-03):
- CFG-Zero* guidance module (study/flow_matching/guidance.py)
- get_guidance_lambda() with 4% zero-init schedule
- guided_euler_ode_integrate() with gradient clipping
- sample_with_guidance() convenience function
- Phase 4 verification complete

**Full training results (100 epochs, 1k):**
| Method | Val Loss | Mean Path Dev | Text Quality |
|--------|----------|---------------|--------------|
| I-CFM | 2.008 | 0.0016 | Coherent |
| OT-CFM | 1.841 | 0.0016 | Coherent |

**Phase 4 success criteria - ALL VERIFIED:**
1. [x] I-CFM trains with independent coupling
2. [x] OT-CFM trains with Sinkhorn coupling
3. [x] Path straightness measured and compared
4. [x] CFG-Zero* zeros first 4% of steps
5. [x] Both methods generate valid SONAR embeddings

**Key finding:** OT-CFM's benefit at this scale is training efficiency (lower loss), not straighter paths. Both methods produce very straight paths (~0.0016 deviation).

Ready for: Phase 5 (Advanced Flow Methods) and Phase 8 (GP-Guided Sampling)
