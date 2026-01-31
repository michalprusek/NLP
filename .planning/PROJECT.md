# Flow Matching Architecture Study

## What This Is

Comprehensive flow matching architecture study for guided prompt generation in SONAR embedding space. This study systematically evaluates flow matching methods (Rectified Flow, OT-CFM, Stochastic Interpolants), velocity network backbones (DiT, U-Net MLP, Simple MLP, Mamba), data augmentation strategies, and dataset scaling to find the optimal configuration for GP-guided generation. Results will replace the existing ecoflow implementation and form the core contribution of a NeurIPS paper.

## Core Value

Find the best flow matching architecture that maximizes both embedding reconstruction quality AND downstream text generation quality while remaining compatible with GP-guided sampling.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Research state-of-the-art flow matching methods from 2023-2025 literature
- [ ] Implement Rectified Flow with linear interpolation paths
- [ ] Implement Optimal Transport CFM with mini-batch OT coupling
- [ ] Implement Stochastic Interpolants with learnable interpolation
- [ ] Implement DiT velocity network as baseline (port from ecoflow)
- [ ] Implement U-Net style MLP velocity network with skip connections
- [ ] Implement Simple MLP velocity network as ablation baseline
- [ ] Implement Mamba/SSM velocity network for sequence modeling
- [ ] Create data augmentation pipeline: linear interpolation (mixup)
- [ ] Create data augmentation pipeline: Gaussian noise injection
- [ ] Create data augmentation pipeline: dimension dropout/masking
- [ ] Dataset subsetting: 1000, 5000, 10000 VS samples
- [ ] Evaluation metrics: reconstruction MSE, SONAR round-trip fidelity
- [ ] Evaluation metrics: text generation quality (BLEU, semantic similarity)
- [ ] Evaluation metrics: guidance responsiveness (GP-guided sampling quality)
- [ ] Training infrastructure: GPU 1 (A5000), checkpointing, logging
- [ ] Experiment tracking: comprehensive ablation with 50+ configurations
- [ ] Final analysis: select best configuration, replace ecoflow
- [ ] Paper-ready visualizations and tables

### Out of Scope

- Multi-GPU distributed training — A5000 sufficient for 1024D embeddings
- Real-time inference optimization — focus on quality first
- Conditional generation beyond GP guidance — pure flow matching study
- Other embedding spaces — SONAR only

## Context

**Dataset:**
- `datasets/gsm8k_instructions_vs.pt`: 4070 verbosed sampling instructions
- 1024-dimensional SONAR embeddings (Meta SONAR encoder)
- CRITICAL: Unnormalized embeddings required for SONAR decoder compatibility

**Existing Implementation:**
- `ecoflow/velocity_network.py`: DiT-style VelocityNetwork with AdaLN-Zero
- `ecoflow/flow_model.py`: FlowMatchingModel with Euler/Heun ODE integration
- `ecoflow/guided_flow.py`: GuidedFlowSampler with GP-UCB guidance
- This study aims to REPLACE ecoflow with better architecture

**Hardware:**
- Training: GPU 1 (NVIDIA A5000, 24GB VRAM)
- CUDA_VISIBLE_DEVICES=1 for all experiments

**Target Venue:**
- NeurIPS paper — requires rigorous ablation, statistical significance, reproducibility

## Constraints

- **Hardware**: Single A5000 (24GB) — batch sizes must fit in memory
- **Embedding format**: SONAR 1024D, unnormalized (no L2 normalization)
- **Compatibility**: Must work with existing GP surrogate and guided sampling
- **Reproducibility**: Fixed seeds, logged hyperparameters, checkpoint every configuration

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Study in study/flow_matching/ | Keep separate from production ecoflow | — Pending |
| Train on VS dataset | Higher quality prompts than random sampling | — Pending |
| Test 1K/5K/10K samples | Dataset scaling ablation for paper | — Pending |
| GPU 1 (A5000) only | Available hardware, sufficient for study | — Pending |
| Unnormalized embeddings | Required for SONAR decoder compatibility | — Pending |

---
*Last updated: 2026-01-31 after initialization*
