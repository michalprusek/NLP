# Roadmap: Flow Matching Architecture Study

## Overview

This roadmap delivers a comprehensive flow matching architecture study for GP-guided prompt generation in SONAR embedding space. The journey progresses from establishing a solid data pipeline and training infrastructure, through implementing and comparing flow matching methods and velocity network architectures, to GP-guided sampling integration, rigorous ablation studies, and final paper deliverables. Each phase builds upon verified foundations to ensure NeurIPS-quality results.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Pipeline** - Dataset generation, splits, normalization, SONAR decoder verification
- [x] **Phase 2: Training Infrastructure** - Training loop, checkpointing, experiment tracking, GPU config
- [x] **Phase 3: Baseline Architectures** - Simple MLP and DiT velocity networks
- [x] **Phase 4: Flow Matching Baselines** - I-CFM and OT-CFM implementations with CFG-Zero*
- [x] **Phase 5: Advanced Flow Methods** - Rectified Flow and Stochastic Interpolants
- [x] **Phase 6: Advanced Architectures** - U-Net MLP, Mamba, and scaled variants
- [x] **Phase 7: Data Augmentation** - Mixup, noise injection, dimension dropout
- [ ] **Phase 8: GP-Guided Sampling** - UCB gradient injection, adaptive guidance, manifold projection
- [ ] **Phase 9: Evaluation Suite** - Metrics implementation and statistical infrastructure
- [ ] **Phase 10: Ablation Studies** - Systematic comparisons across all dimensions
- [ ] **Phase 11: Paper Deliverables** - Results compilation, visualizations, best model selection

## Phase Details

### Phase 1: Data Pipeline
**Goal**: Establish verified data foundation with proper splits and SONAR decoder compatibility
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. 10K verbosed sampling dataset exists with SONAR embeddings
  2. Train/val/test splits exist for 1K, 5K, and 10K dataset sizes
  3. Normalization statistics (mean/std) are computed and saved
  4. SONAR decoder produces coherent text from denormalized embeddings
  5. Data loading pipeline returns properly normalized tensors
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md - Generate 10K VS dataset and create nested splits
- [x] 01-02-PLAN.md - Normalization pipeline and SONAR decoder verification

### Phase 2: Training Infrastructure
**Goal**: Training system ready for all experiments with proper tracking and reproducibility
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):
  1. Training loop runs with EMA, gradient clipping, and early stopping
  2. Checkpoints save best model based on validation loss
  3. Wandb logs training metrics, hyperparameters, and artifacts
  4. All experiments run on GPU 1 (A5000) with CUDA_VISIBLE_DEVICES=1
  5. Training can resume from checkpoint
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md - Core training loop with EMA, early stopping, and FlowTrainer class
- [x] 02-02-PLAN.md - Checkpoint management and Wandb integration

### Phase 3: Baseline Architectures
**Goal**: Two baseline velocity networks trained and evaluated for comparison
**Depends on**: Phase 2
**Requirements**: ARCH-01, ARCH-02
**Success Criteria** (what must be TRUE):
  1. Simple MLP velocity network trains without NaN loss
  2. DiT velocity network (ported from ecoflow) trains without NaN loss
  3. Both architectures produce reasonable reconstruction MSE (<0.1)
  4. Generated embeddings decode to coherent text
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md - Simple MLP implementation with model factory
- [x] 03-02-PLAN.md - DiT port from ecoflow and verification
- [x] 03-03-PLAN.md - Reconstruction MSE and text generation verification (gap closure)

### Phase 4: Flow Matching Baselines
**Goal**: I-CFM and OT-CFM flow matching methods working with baseline architectures
**Depends on**: Phase 3
**Requirements**: FLOW-01, FLOW-02, FLOW-05
**Success Criteria** (what must be TRUE):
  1. I-CFM trains with independent noise-data coupling
  2. OT-CFM trains with mini-batch Sinkhorn coupling
  3. OT-CFM produces straighter paths than I-CFM (lower path variance)
  4. CFG-Zero* schedule zeros guidance for first 4% of steps
  5. Both methods generate valid SONAR embeddings
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md - Coupling abstraction and OT-CFM implementation
- [x] 04-02-PLAN.md - OT-CFM training and path straightness evaluation
- [x] 04-03-PLAN.md - CFG-Zero* guidance integration and phase verification

### Phase 5: Advanced Flow Methods
**Goal**: Rectified Flow and Stochastic Interpolants implemented and compared
**Depends on**: Phase 4
**Requirements**: FLOW-03, FLOW-04
**Success Criteria** (what must be TRUE):
  1. Rectified Flow reflow procedure runs on trained I-CFM model
  2. Reflow produces straighter paths (fewer ODE steps needed)
  3. Stochastic Interpolants with learnable interpolation trains
  4. All flow methods produce comparable sample quality
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md - Rectified Flow with reflow pair generation and training
- [x] 05-02-PLAN.md - Stochastic Interpolants with GVP schedule and phase comparison

### Phase 6: Advanced Architectures
**Goal**: U-Net MLP, Mamba, and scaled variants implemented for ablation
**Depends on**: Phase 4
**Requirements**: ARCH-03, ARCH-04, ARCH-05
**Success Criteria** (what must be TRUE):
  1. U-Net MLP with FiLM conditioning trains without NaN loss
  2. Mamba/SSM velocity network trains (experimental)
  3. Tiny/Small/Base variants exist for dataset size scaling
  4. All architectures produce valid SONAR embeddings
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md - U-Net MLP with FiLM time conditioning
- [x] 06-02-PLAN.md - Mamba velocity network (experimental, graceful fallback)
- [x] 06-03-PLAN.md - Architecture scaling variants (Tiny/Small/Base)

### Phase 7: Data Augmentation
**Goal**: Three augmentation strategies implemented and tested
**Depends on**: Phase 3
**Requirements**: DATA-05, DATA-06, DATA-07
**Success Criteria** (what must be TRUE):
  1. Linear interpolation (mixup) generates valid training pairs
  2. Gaussian noise injection augments training data
  3. Dimension dropout/masking augments training data
  4. Augmented training improves generalization (lower val loss)
**Plans**: 2 plans

Plans:
- [x] 07-01-PLAN.md - Mixup and noise injection with trainer integration
- [x] 07-02-PLAN.md - Dimension dropout and ablation validation

### Phase 8: GP-Guided Sampling
**Goal**: Full GP-UCB guided sampling pipeline with adaptive guidance
**Depends on**: Phase 4
**Requirements**: GUIDE-01, GUIDE-02, GUIDE-03, GUIDE-04
**Success Criteria** (what must be TRUE):
  1. GP-UCB gradients inject into ODE sampling steps
  2. Adaptive guidance strength varies with GP uncertainty
  3. Gradient clipping prevents GP gradient explosion
  4. Manifold projection via encode-decode produces valid embeddings
  5. Guided samples achieve higher GP acquisition values than unguided
**Plans**: TBD

Plans:
- [ ] 08-01: GP-UCB gradient injection and clipping
- [ ] 08-02: Adaptive guidance and manifold projection

### Phase 9: Evaluation Suite
**Goal**: Complete evaluation infrastructure for NeurIPS-quality experiments
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06
**Success Criteria** (what must be TRUE):
  1. Reconstruction MSE computed between generated and target embeddings
  2. SONAR round-trip fidelity measured (encode-decode-encode)
  3. Text quality metrics (BLEU, semantic similarity) computed for decoded text
  4. Guidance responsiveness metric quantifies GP guidance effectiveness
  5. Diversity metrics (pairwise distances, coverage) computed
  6. Confidence intervals computed via bootstrap with 10K+ samples
**Plans**: TBD

Plans:
- [ ] 09-01: Embedding metrics (MSE, round-trip, diversity)
- [ ] 09-02: Text quality and guidance responsiveness
- [ ] 09-03: Statistical significance infrastructure

### Phase 10: Ablation Studies
**Goal**: Systematic ablations across all experimental dimensions
**Depends on**: Phase 5, Phase 6, Phase 7, Phase 8, Phase 9
**Requirements**: ABLAT-01, ABLAT-02, ABLAT-03, ABLAT-04, ABLAT-05, ABLAT-06, TRAIN-05
**Success Criteria** (what must be TRUE):
  1. Flow method comparison (I-CFM vs OT-CFM vs RF vs SI) completed with stats
  2. Architecture comparison (MLP vs DiT vs U-Net vs Mamba) completed with stats
  3. Dataset size scaling (1K vs 5K vs 10K) analysis completed
  4. Augmentation effectiveness analysis completed
  5. Guidance strength sweep completed
  6. ODE solver comparison (Euler vs Heun) completed
  7. All results include confidence intervals
**Plans**: TBD

Plans:
- [ ] 10-01: Flow method comparison ablation
- [ ] 10-02: Architecture comparison ablation
- [ ] 10-03: Dataset and augmentation ablations
- [ ] 10-04: Guidance and solver ablations

### Phase 11: Paper Deliverables
**Goal**: All paper artifacts ready for NeurIPS submission
**Depends on**: Phase 10
**Requirements**: PAPER-01, PAPER-02, PAPER-03, PAPER-04
**Success Criteria** (what must be TRUE):
  1. Comprehensive results table with 50+ configurations exists
  2. t-SNE/UMAP visualizations of generated embeddings created
  3. Training curves comparison figure exists
  4. Best model identified and ecoflow replacement prepared
  5. All figures and tables are publication-ready
**Plans**: TBD

Plans:
- [ ] 11-01: Results compilation and tables
- [ ] 11-02: Visualizations and best model selection

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11

Note: Phases 5, 6, 7, 8 can partially parallelize after Phase 4 completes.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 2/2 | Complete | 2026-02-01 |
| 2. Training Infrastructure | 2/2 | Complete | 2026-02-01 |
| 3. Baseline Architectures | 3/3 | Complete | 2026-02-01 |
| 4. Flow Matching Baselines | 3/3 | Complete | 2026-02-01 |
| 5. Advanced Flow Methods | 2/2 | Complete | 2026-02-01 |
| 6. Advanced Architectures | 3/3 | Complete | 2026-02-01 |
| 7. Data Augmentation | 2/2 | Complete | 2026-02-01 |
| 8. GP-Guided Sampling | 0/2 | Not started | - |
| 9. Evaluation Suite | 0/3 | Not started | - |
| 10. Ablation Studies | 0/4 | Not started | - |
| 11. Paper Deliverables | 0/2 | Not started | - |

---
*Roadmap created: 2026-01-31*
*Last updated: 2026-02-01 (Phase 7 complete)*
