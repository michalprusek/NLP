# Project Research Summary

**Project:** Flow Matching Architecture Study for SONAR Embeddings
**Domain:** Generative Modeling - Flow Matching for Prompt Optimization
**Researched:** 2026-01-31
**Confidence:** HIGH

## Executive Summary

This research covers flow matching architectures for generating optimal prompts via GP-guided sampling in 1024D SONAR embedding space with limited training data (1K-10K samples). The core challenge is balancing model capacity against overfitting risk while integrating Bayesian optimization guidance that respects the SONAR manifold constraints.

The recommended approach uses custom flow matching implementations (not heavy frameworks) with architecture size scaled to dataset regime: MLP or U-Net style networks for <5K samples, small DiT (6 layers, 512 hidden dim) for 5K-10K samples. Critical to success is implementing OT-CFM coupling for straighter paths, proper normalization/denormalization pipeline for SONAR compatibility, and CFG-Zero* guidance schedule to prevent early trajectory corruption. The existing RieLBO implementation already contains well-designed components (DiT architecture, GP surrogate with MSR initialization, CFG-Zero* sampling) but may need regularization tuning and architecture downsizing for small datasets.

Key risks center on overfitting (the 9.4M parameter DiT-Base will memorize 1K samples), mode collapse from insufficient diversity monitoring, and SONAR decoder incompatibility from normalization errors. Prevention requires aggressive regularization (weight decay 0.01-0.05, EMA decay tuned to dataset size), validation-based early stopping, and empirical verification of decoded output quality throughout training. For NeurIPS submission, proper evaluation metrics (MMD in SONAR space, not raw FID), fair baseline comparisons, and ablation studies of guidance strength, coupling methods, and architecture variants are essential.

## Key Findings

### Recommended Stack

The optimal stack for this domain prioritizes control and simplicity over framework abstraction. For 1024D SONAR embeddings with 1K-10K samples, velocity network architecture matters far more than which flow matching library you use.

**Core technologies:**
- **PyTorch 2.8+**: Core framework with improved compilation and CUDA support
- **Custom Flow Matching**: Direct implementation for control over architecture and GP guidance integration
- **BoTorch 0.16.1 + GPyTorch 1.14+**: State-of-the-art GP surrogate with MSR initialization for high-dimensional spaces
- **ema-pytorch 0.7.9**: Exponential moving average for stable generative model training
- **Weights & Biases**: Experiment tracking for NeurIPS-quality research

**Why custom over frameworks:** The existing RieLBO implementation already has well-structured components (velocity_network.py, flow_model.py, guided_flow.py). For small-scale research with specific requirements (GP gradient injection, SONAR denormalization), direct PyTorch is faster and more controllable than facebookresearch/flow_matching or torchcfm. The latter can serve as reference implementations for OT coupling but don't need to be dependencies.

**Critical versions:** BoTorch 0.16.1 includes MSR initialization (ICLR 2025) which is essential for 1024D GP optimization. PyTorch 2.8+ recommended for control flow operators and improved compilation, though 2.1+ will work.

### Expected Features

**Must implement for credible NeurIPS paper (Table Stakes):**
- **Conditional Flow Matching (CFM)** with independent coupling — Foundation baseline
- **Optimal Transport CFM (OT-CFM)** with mini-batch Sinkhorn — Standard improvement over I-CFM
- **DiT velocity network** with AdaLN-Zero conditioning — Already implemented, proven architecture
- **CFG-Zero* guidance schedule** — Zero guidance for first 4% of steps, prevents trajectory corruption
- **Proper evaluation suite** — MMD in SONAR space, downstream task performance, sample diversity

**Core novel contributions (Differentiators):**
- **GP-UCB Guided Flow Sampling** — External reward guidance via Bayesian optimization acquisition function (primary novelty)
- **Embedding-space optimization** — Flow matching in pretrained SONAR space enables prompt optimization application
- **Adaptive guidance strength** — Schedule guidance lambda based on GP uncertainty (natural CFG-Zero* extension)

**Optional strengthening features:**
- **Rectified Flow** — One reflow iteration for straighter paths and fewer ODE steps
- **Large Sinkhorn couplings** — If compute permits, n=2M+ batch OT with GPU sharding
- **Model-Aligned Coupling** — 2025 cutting-edge, addresses OT geometric limitations
- **Multi-fidelity GP** — Cheap LLM evaluations for surrogate, expensive for validation

**Incompatible with GP guidance (Anti-features):**
- Consistency models / 1-step distillation — No iterative ODE to guide
- Discrete flow matching — Works on categorical data, not continuous embeddings
- Fisher flow / statistical manifold methods — Euclidean flow is simpler and works

### Architecture Approach

For small datasets (1K-10K), architecture complexity must scale with data availability to prevent overfitting. The key insight is that SONAR embeddings are 1D vectors (1024D), not sequences or images, so architectures designed for spatial structure are overkill. Simple networks with strong regularization outperform deep transformers in the low-data regime.

**Major components and sizing:**

1. **Velocity Network** (architecture varies by dataset size)
   - 1K samples: MLP-Small (512-256-256-512 bottleneck, ~1M params) or DiT-Tiny (256 hidden, 4 layers, ~1.6M params)
   - 5K samples: U-Net MLP (768→128 bottleneck, FiLM conditioning, ~2.5M params) or DiT-Small (384 hidden, 6 layers, ~5.3M params)
   - 10K samples: DiT-Base (512 hidden, 6 layers, ~9.4M params) — current RieLBO default
   - Rule of thumb: Parameters < 10x (num_samples * input_dim) for safe training

2. **Time Conditioning Method** (critical for training stability)
   - Sinusoidal embeddings (256D) for continuous time representation
   - **AdaLN-Zero** for transformers (DiT) — essential for stable training from scratch
   - **FiLM** (feature-wise linear modulation) for MLPs/U-Net — proven effective in low-data regimes
   - Never use learnable affine in LayerNorm when using AdaLN (sets elementwise_affine=False)

3. **Normalization Strategy** (SONAR compatibility constraint)
   - Train in normalized space (z-score: (x - mean) / std) for stability
   - Store mean/std statistics and denormalize before SONAR decoder
   - CRITICAL: SONAR embeddings should NOT be unit-normalized; decoder expects original distribution
   - Verify decoder output quality early (epoch 1) to catch normalization errors

4. **Regularization for Small Datasets**
   - Weight decay: 0.01-0.05 (higher for smaller datasets)
   - Dropout: 0.1-0.2 in MLP blocks (not attention layers)
   - EMA decay: 0.99 for 1K samples, 0.999 for 5K, 0.9999 for 10K+
   - Gradient clipping: max_norm=1.0 to prevent NaN loss
   - Early stopping on validation set

5. **Flow Matching Method** (coupling strategy)
   - Baseline: I-CFM with random noise-data pairing
   - Primary: OT-CFM with mini-batch Sinkhorn (n=256-1024, reg=0.05)
   - Optional: Rectified Flow (one reflow iteration after initial training)
   - OT coupling creates straighter paths, reduces training variance, enables fewer ODE steps

6. **ODE Integration for Sampling**
   - Heun solver (2nd order) with 50 steps as default for production
   - Euler (1st order) acceptable for prototyping, needs 80-100 steps
   - Adaptive solvers (dopri5) unnecessary — flow matching creates straight paths
   - CFG-Zero*: Skip first 4% of steps (already implemented in guided_flow.py)

7. **GP Surrogate Configuration** (already well-designed in gp_surrogate.py)
   - Matern-5/2 kernel with ARD for 1024D
   - MSR initialization (ICLR 2025) for high-dimensional lengthscale priors
   - LogNormal prior scaled by sqrt(D)
   - UCB acquisition with alpha=1.0-2.0 for exploration-exploitation

### Critical Pitfalls

**Top 5 risks requiring active monitoring:**

1. **Training data normalization vs decoder compatibility** — Train in normalized space but MUST denormalize before SONAR decoder. Failure produces gibberish decoded text despite reasonable embedding distances. Prevention: Store mean/std, implement denormalize(), verify decoded output at epoch 1. Already correctly implemented in flow_model.py but must not break this pattern.

2. **Overfitting in 1K-10K regime** — Model memorizes training data exactly, generates only interpolations. Warning signs: near-zero training loss, generated samples very close to training samples (low min L2 distance), poor generalization. Prevention: Aggressive regularization (weight_decay=0.01-0.05), EMA, validation-based early stopping, reduce model capacity for <5K samples, measure novelty via min-distance-to-training.

3. **Mode collapse from insufficient diversity** — Flow model covers only subset of distribution, ignoring rare modes. Warning signs: Low sample diversity (high pairwise cosine similarity), Thompson sampling always picks similar points. Prevention: Monitor diversity metrics (pairwise distances, coverage), use OT coupling to reduce path variance, visualize t-SNE/UMAP at checkpoints.

4. **CFG-Zero* early steps corruption** — Applying guidance from t=0 amplifies errors, corrupts trajectories. Warning signs: Guided samples worse than unguided, artifacts at high guidance strength. Prevention: Zero-init schedule for first 4% of steps (already implemented), ramp guidance strength linearly, tune guidance strength carefully (start with 0.1-0.5).

5. **Inappropriate evaluation metrics** — Using FID in Inception feature space for SONAR embeddings produces meaningless numbers. Reviewers will question metric validity. Prevention: Use MMD in SONAR space directly, downstream task performance (decoded text quality), sample diversity metrics, human evaluation. Report multiple metrics with confidence intervals from 10K+ samples.

## Implications for Roadmap

Based on research, suggested phase structure balances baseline establishment, core contribution validation, and NeurIPS-quality evaluation rigor.

### Phase 1: Architecture Baseline & Data Pipeline
**Rationale:** Must establish data infrastructure and verify SONAR compatibility before any modeling work. The normalization/denormalization pipeline is critical — errors here invalidate all downstream experiments.

**Delivers:**
- Train/validation/test split with proper stratification
- Normalization statistics (mean/std) computed and saved
- SONAR decoder integration with verified decode quality
- Simple MLP baseline trained and evaluated
- Current DiT-Base architecture evaluation on target dataset

**Addresses:** Data pipeline, normalization strategy (PITFALLS.md critical risk #1)

**Avoids:** Training in wrong space, decoder incompatibility, unfair comparisons from inconsistent data splits

**Research needs:** Minimal — standard ML practices, existing implementation provides pattern

### Phase 2: Flow Matching Methods
**Rationale:** Establish flow matching baselines before adding GP guidance. OT-CFM is table stakes for credible paper; reviewers expect this comparison.

**Delivers:**
- OT-CFM implementation with mini-batch Sinkhorn coupling
- I-CFM vs OT-CFM comparison (training stability, sample quality, ODE steps required)
- Architecture scaling study (MLP vs U-Net vs DiT at different dataset sizes)
- Regularization tuning (weight decay, dropout, EMA decay for target dataset size)

**Uses:** torchcfm as reference for OT coupling (STACK.md), POT library for Sinkhorn algorithm

**Implements:** Velocity network variants, time conditioning methods (ARCHITECTURE.md)

**Avoids:** Path crossing variance (PITFALLS.md #3), overfitting from oversized models (PITFALLS.md #2)

**Research needs:** Reference torchcfm implementation for OT coupling, POT documentation for Sinkhorn

### Phase 3: GP-Guided Sampling
**Rationale:** Core novel contribution. Requires pretrained flow model from Phase 2. CFG-Zero* schedule already implemented, focus on integration testing and guidance strength tuning.

**Delivers:**
- GP-UCB gradient injection into ODE sampling (verify existing implementation)
- CFG-Zero* schedule tuning (optimize zero-init fraction)
- Adaptive guidance strength based on GP uncertainty
- Gradient clipping tuning for GP gradients (max_grad_norm parameter)
- Manifold projection via encode-decode cycle

**Addresses:** GP-guided flow sampling (FEATURES.md core contribution), CFG-Zero* guidance

**Avoids:** GP gradient explosion (PITFALLS.md #4.2), guidance-manifold conflict (PITFALLS.md #4.3), early step corruption (PITFALLS.md #4.1)

**Research needs:** Minimal — implementation mostly exists, needs empirical tuning

### Phase 4: Advanced Methods & Ablations
**Rationale:** Strengthens NeurIPS submission with ablations and advanced baselines. Rectified Flow is standard in 2025-2026 literature.

**Delivers:**
- Rectified Flow implementation (one reflow iteration)
- Ablation studies: guidance strength sweep, UCB alpha sweep, ODE step count, OT vs I-CFM coupling
- Architecture ablations: time conditioning (AdaLN vs FiLM), normalization (LayerNorm vs RMSNorm)
- ODE solver comparison (Euler vs Heun vs RK4)

**Uses:** Existing flow model as teacher for reflow data generation

**Implements:** Rectified Flow (FEATURES.md table stakes)

**Avoids:** Insufficient ablations, unfair baseline comparisons (PITFALLS.md #5.3)

**Research needs:** Rectified Flow paper for reflow procedure details

### Phase 5: Evaluation & Baselines
**Rationale:** NeurIPS papers require rigorous evaluation with fair baseline comparisons and multiple metrics. Must generate sufficient samples for metric stability.

**Delivers:**
- MMD computation in SONAR embedding space
- Downstream task evaluation (decoded text quality via BLEU/ROUGE/BERTScore)
- Sample diversity metrics (pairwise distances, coverage, density)
- Baseline comparisons: VAE, random sampling, unguided flow, I-CFM without guidance
- Confidence intervals via bootstrap (10K+ samples)
- Human evaluation protocol for decoded prompt quality

**Addresses:** Proper evaluation metrics (FEATURES.md must-have)

**Avoids:** FID misuse (PITFALLS.md #5.1), insufficient sample sizes (PITFALLS.md #5.2), unfair baselines (PITFALLS.md #5.3)

**Research needs:** Minimal — standard evaluation practices

### Phase Ordering Rationale

- **Phase 1 before 2:** Data pipeline errors invalidate all downstream work. Must verify SONAR compatibility early.
- **Phase 2 before 3:** GP guidance requires pretrained flow model. Must establish flow baselines first.
- **Phase 4 parallel with 3:** Ablations can run alongside guidance experiments once flow models exist.
- **Phase 5 throughout:** Evaluation should happen incrementally, but comprehensive suite comes last.

**Dependency chain:** Data pipeline → Flow baselines → GP guidance → Ablations + Evaluation

**Overfitting mitigation:** Architecture scaling in Phase 2 identifies right model size before expensive Phase 3 experiments.

**Critical path:** Phases 1-2-3 are sequential and required. Phase 4 strengthens but isn't blocking. Phase 5 runs throughout but comprehensive evaluation is final step.

### Research Flags

**Phases likely needing deeper research during planning:**

- **Phase 2 (OT-CFM):** Sinkhorn algorithm tuning (regularization parameter, convergence criteria), large-batch OT if scaling beyond n=1024. Reference torchcfm and POT library documentation.
- **Phase 4 (Rectified Flow):** Reflow data generation procedure, optimal number of reflow iterations. Limited documentation beyond original paper.

**Phases with standard patterns (skip research-phase):**

- **Phase 1 (Data Pipeline):** Standard ML data handling, existing RieLBO implementation provides clear pattern.
- **Phase 3 (GP Guidance):** Implementation already exists and follows well-founded posterior guidance approach from ICML 2025 paper.
- **Phase 5 (Evaluation):** Standard metrics for generative models, well-documented in literature.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | BoTorch, PyTorch versions verified via official docs and PyPI. Custom flow matching validated via existing implementation. |
| Features | HIGH | CFM/OT-CFM table stakes verified across multiple NeurIPS 2024/ICLR 2025 papers. GP-UCB guidance novelty confirmed (no prior work found). |
| Architecture | MEDIUM-HIGH | DiT architecture well-documented, but sizing for small datasets based on general deep learning principles not flow-matching-specific papers. U-Net MLP and FiLM recommendations from related domains. |
| Pitfalls | HIGH | Critical pitfalls verified in existing implementation code (normalization, CFG-Zero*, gradient clipping). Mode collapse and overfitting risks from multiple authoritative sources. |

**Overall confidence:** HIGH

### Gaps to Address

**Empirical validation required during Phase 2:**
- Optimal architecture size for specific dataset (1K vs 5K vs 10K samples) — research provides heuristics but needs empirical sweep
- EMA decay tuning for target dataset size — research suggests 0.99-0.9999 range but optimal value needs validation
- OT coupling batch size (n=256 vs 512 vs 1024) — performance vs compute tradeoff needs measurement

**Empirical validation required during Phase 3:**
- Optimal guidance strength (lambda) for GP-UCB gradients — likely 0.1-1.0 range but needs sweep
- UCB exploration parameter (alpha) — research suggests 1.0-2.0 but depends on GP fit quality
- CFG-Zero* zero-init fraction — research suggests 4% but may need tuning for SONAR embeddings

**Evaluation metric development needed in Phase 5:**
- Adaptation of FID to SONAR space (compute on embeddings directly, not Inception features)
- Coverage metric for 1024D space (standard coverage metrics may not scale)
- Human evaluation protocol design for prompt quality

**Baseline selection for Phase 5:**
- Identify canonical VAE implementation for embedding generation baseline
- Find prior work on SONAR embedding generation for comparison
- Establish performance targets from related prompt optimization work

## Sources

### Primary (HIGH confidence)
- Existing RieLBO implementation: velocity_network.py, flow_model.py, guided_flow.py, gp_surrogate.py
- [PyTorch 2.8 Release Blog](https://pytorch.org/blog/pytorch-2-8/)
- [BoTorch Documentation](https://botorch.org/docs/overview/) — v0.16.1 with MSR initialization
- [torchcfm GitHub](https://github.com/atong01/conditional-flow-matching) — v1.0.7 reference implementation
- [CFG-Zero* Paper](https://arxiv.org/abs/2503.18886) — Improved classifier-free guidance
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., foundation paper
- [OT-CFM Paper](https://arxiv.org/abs/2302.00482) — Tong et al., mini-batch optimal transport

### Secondary (MEDIUM confidence)
- [ICLR 2025: Visual Dive into Conditional Flow Matching](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/)
- [ICML 2025: On the Guidance of Flow Matching](https://raw.githubusercontent.com/mlresearch/v267/main/assets/feng25s/feng25s.pdf)
- [DiT: Scalable Diffusion Transformers](https://www.wpeebles.com/DiT.html)
- [Cambridge MLG: Flow Matching Introduction](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- [Standard GP in High Dimensions](https://arxiv.org/html/2402.02746v3) — Feb 2024
- [Detecting Mode Collapse in Flow-Based Sampling](https://link.aps.org/doi/10.1103/PhysRevD.108.114501)
- [EMA in Diffusion Models](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/)

### Tertiary (LOW confidence, needs validation)
- [Large Sinkhorn Couplings](https://arxiv.org/abs/2506.05526) — July 2025, not peer-reviewed
- [Model-Aligned Coupling](https://www.alphaxiv.org/overview/2505.23346v1) — Preprint
- [Rectified Flow](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html) — Implementation details
- [NF-BO: Normalizing Flows for BO](https://arxiv.org/abs/2504.14889) — ICLR 2025, related approach

---
*Research completed: 2026-01-31*
*Ready for roadmap: yes*
