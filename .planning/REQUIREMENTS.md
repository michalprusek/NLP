# Requirements: Flow Matching Architecture Study

**Defined:** 2026-01-31
**Core Value:** Find the best flow matching architecture for GP-guided prompt generation in SONAR space

## v1 Requirements

Requirements for NeurIPS paper submission. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: Generate new 10K verbosed sampling dataset using existing VS pipeline
- [x] **DATA-02**: Create train/val/test splits for 1K, 5K, 10K dataset sizes
- [x] **DATA-03**: Implement normalization with stored mean/std statistics
- [x] **DATA-04**: Verify SONAR decoder works with denormalized embeddings
- [ ] **DATA-05**: Implement data augmentation: linear interpolation (mixup)
- [ ] **DATA-06**: Implement data augmentation: Gaussian noise injection
- [ ] **DATA-07**: Implement data augmentation: dimension dropout/masking

### Flow Matching Methods

- [ ] **FLOW-01**: Implement I-CFM (Independent Coupling) baseline
- [ ] **FLOW-02**: Implement OT-CFM with mini-batch Sinkhorn coupling (use torchcfm/POT)
- [ ] **FLOW-03**: Implement Rectified Flow with reflow procedure
- [ ] **FLOW-04**: Implement Stochastic Interpolants with learnable interpolation
- [ ] **FLOW-05**: Integrate CFG-Zero* guidance schedule (zero first 4% steps)

### Velocity Network Architectures

- [x] **ARCH-01**: Implement Simple MLP baseline (~1M params)
- [x] **ARCH-02**: Port DiT baseline from rielbo (~9.4M params)
- [ ] **ARCH-03**: Implement U-Net MLP with FiLM conditioning (~2.5M params)
- [ ] **ARCH-04**: Implement Mamba/SSM velocity network (experimental)
- [ ] **ARCH-05**: Implement scaled versions for different dataset sizes (Tiny/Small/Base)

### Training Infrastructure

- [x] **TRAIN-01**: Training loop with EMA, gradient clipping, early stopping
- [x] **TRAIN-02**: Checkpoint management with best model selection
- [x] **TRAIN-03**: Wandb experiment tracking integration
- [x] **TRAIN-04**: GPU 1 (A5000) configuration with CUDA_VISIBLE_DEVICES=1
- [ ] **TRAIN-05**: Hyperparameter sweep infrastructure

### GP-Guided Sampling

- [ ] **GUIDE-01**: Integrate GP-UCB gradient injection into ODE sampling
- [ ] **GUIDE-02**: Implement adaptive guidance strength based on GP uncertainty
- [ ] **GUIDE-03**: Gradient clipping for GP gradients (prevent explosion)
- [ ] **GUIDE-04**: Manifold projection via encode-decode cycle

### Evaluation Suite

- [ ] **EVAL-01**: Reconstruction MSE between generated and target embeddings
- [ ] **EVAL-02**: SONAR round-trip fidelity (encode->decode->encode)
- [ ] **EVAL-03**: Text generation quality (BLEU, semantic similarity)
- [ ] **EVAL-04**: Guidance responsiveness metric (GP guidance effectiveness)
- [ ] **EVAL-05**: Sample diversity metrics (pairwise distances, coverage)
- [ ] **EVAL-06**: Statistical significance with confidence intervals

### Ablation Study

- [ ] **ABLAT-01**: Flow method comparison (I-CFM vs OT-CFM vs RF vs SI)
- [ ] **ABLAT-02**: Architecture comparison (MLP vs DiT vs U-Net vs Mamba)
- [ ] **ABLAT-03**: Dataset size scaling (1K vs 5K vs 10K)
- [ ] **ABLAT-04**: Augmentation effectiveness (none vs mixup vs noise vs dropout)
- [ ] **ABLAT-05**: Guidance strength sweep
- [ ] **ABLAT-06**: ODE solver comparison (Euler vs Heun)

### Paper Deliverables

- [ ] **PAPER-01**: Comprehensive results table with all configurations (~50+)
- [ ] **PAPER-02**: Visualization: t-SNE/UMAP of generated embeddings
- [ ] **PAPER-03**: Visualization: training curves comparison
- [ ] **PAPER-04**: Best model selection and rielbo replacement

## v2 Requirements

Deferred to future work / paper extensions.

### Advanced Methods

- **ADV-01**: Model-Aligned Coupling (MAC) for better OT geometry
- **ADV-02**: Large Sinkhorn couplings (n=2M+ with GPU sharding)
- **ADV-03**: Multi-fidelity GP with cheap/expensive evaluations

### Extended Evaluation

- **EVAL-EXT-01**: Human evaluation protocol for prompt quality
- **EVAL-EXT-02**: Downstream task performance (GSM8K accuracy)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-GPU distributed training | A5000 sufficient for 1024D embeddings, adds complexity |
| Real-time inference optimization | Focus on quality first for paper |
| Other embedding spaces (GTR, etc.) | SONAR only for this study |
| Consistency models / 1-step distillation | Incompatible with iterative GP guidance |
| Discrete flow matching | Wrong domain (continuous embeddings) |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 7 | Pending |
| DATA-06 | Phase 7 | Pending |
| DATA-07 | Phase 7 | Pending |
| FLOW-01 | Phase 4 | Pending |
| FLOW-02 | Phase 4 | Pending |
| FLOW-03 | Phase 5 | Pending |
| FLOW-04 | Phase 5 | Pending |
| FLOW-05 | Phase 4 | Pending |
| ARCH-01 | Phase 3 | Complete |
| ARCH-02 | Phase 3 | Complete |
| ARCH-03 | Phase 6 | Pending |
| ARCH-04 | Phase 6 | Pending |
| ARCH-05 | Phase 6 | Pending |
| TRAIN-01 | Phase 2 | Complete |
| TRAIN-02 | Phase 2 | Complete |
| TRAIN-03 | Phase 2 | Complete |
| TRAIN-04 | Phase 2 | Complete |
| TRAIN-05 | Phase 10 | Pending |
| GUIDE-01 | Phase 8 | Pending |
| GUIDE-02 | Phase 8 | Pending |
| GUIDE-03 | Phase 8 | Pending |
| GUIDE-04 | Phase 8 | Pending |
| EVAL-01 | Phase 9 | Pending |
| EVAL-02 | Phase 9 | Pending |
| EVAL-03 | Phase 9 | Pending |
| EVAL-04 | Phase 9 | Pending |
| EVAL-05 | Phase 9 | Pending |
| EVAL-06 | Phase 9 | Pending |
| ABLAT-01 | Phase 10 | Pending |
| ABLAT-02 | Phase 10 | Pending |
| ABLAT-03 | Phase 10 | Pending |
| ABLAT-04 | Phase 10 | Pending |
| ABLAT-05 | Phase 10 | Pending |
| ABLAT-06 | Phase 10 | Pending |
| PAPER-01 | Phase 11 | Pending |
| PAPER-02 | Phase 11 | Pending |
| PAPER-03 | Phase 11 | Pending |
| PAPER-04 | Phase 11 | Pending |

**Coverage:**
- v1 requirements: 40 total
- Mapped to phases: 40
- Unmapped: 0

---
*Requirements defined: 2026-01-31*
*Last updated: 2026-01-31 after roadmap creation*
