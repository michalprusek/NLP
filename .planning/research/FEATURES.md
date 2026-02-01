# Features Research: Flow Matching Methods for GP-Guided Generation

**Domain:** Flow matching for 1024D SONAR embeddings with GP-UCB guidance
**Researched:** 2026-01-31
**Target:** NeurIPS paper

## Executive Summary

Flow matching has emerged as the dominant paradigm for continuous generative modeling, superseding diffusion models in many domains. For GP-guided generation in SONAR embedding space, the key distinction is between methods that support external gradient-based guidance and those that don't. This document maps the feature landscape for a credible NeurIPS submission.

---

## Flow Matching Methods

### Table Stakes (Must Implement for Credible Paper)

These are baseline methods that reviewers expect to see compared. Missing any of these signals incomplete work.

| Method | Description | Complexity | Guidance Compatible | Notes |
|--------|-------------|------------|---------------------|-------|
| **Conditional Flow Matching (CFM)** | Standard CFM with independent coupling (I-CFM) | Low | YES | Foundation method. Linear interpolation between noise and data. Must implement as baseline. |
| **Optimal Transport CFM (OT-CFM)** | Mini-batch OT coupling using Sinkhorn/Hungarian | Medium | YES | Creates straighter paths than I-CFM. Standard upgrade. Expect ~10-20% improvement in sample quality. |
| **Rectified Flow (RF)** | Reflow procedure to straighten paths | Medium | YES | Liu et al. 2023. One reflow iteration sufficient in practice. Enables few-step generation. |
| **CFG-Zero* Schedule** | Zero guidance for first 4% of steps | Low | N/A (guidance schedule) | Prevents early trajectory corruption. Already in codebase. Required for any guidance work. |
| **DiT Velocity Network** | Transformer with AdaLN-Zero conditioning | Medium | YES | Standard architecture for flow matching. Already implemented. Scale varies (6-24 layers typical). |

**Verdict:** Your existing codebase already implements CFM, DiT architecture, and CFG-Zero*. You need to add OT-CFM coupling and Rectified Flow for completeness.

---

### Differentiators (Novel Contributions for NeurIPS)

These are features that distinguish your work. Pick 2-3 as main contributions.

| Feature | Value Proposition | Complexity | Guidance Compatible | Publication Potential |
|---------|-------------------|------------|---------------------|----------------------|
| **GP-UCB Guided Flow Sampling** | External reward guidance via UCB gradients | Medium | CORE CONTRIBUTION | HIGH - Novel combination of BO acquisition functions with flow matching |
| **Embedding-Space Flow Matching** | Flow matching in pretrained SONAR space vs raw data | Medium | YES | MEDIUM - Not novel alone but enables downstream applications |
| **Adaptive Guidance Strength** | Schedule guidance lambda based on GP uncertainty | Low | YES | MEDIUM - Natural extension of CFG-Zero* |
| **Multi-Fidelity Surrogate** | Use cheap LLM scores for GP, expensive for validation | High | YES | HIGH - Novel BO approach |
| **Projection to Manifold** | Encode-decode cycle to stay on SONAR manifold | Low | YES | LOW alone, but necessary for validity |
| **Local Perturbation Sampling** | Generate from best embedding with noise + reflow | Low | YES | MEDIUM - Practical contribution |
| **Large Sinkhorn Couplings** | n=2M+ batch OT with low epsilon | High | YES | HIGH - Recent 2025 result shows major gains |
| **Model-Aligned Coupling (MAC)** | Select couplings based on model learnability not just geometry | High | YES | HIGH - 2025 cutting-edge, addresses OT limitations |

**Recommended Novel Contributions:**
1. **GP-UCB Guided Flow Sampling** - Your core differentiator
2. **Embedding-Space Optimization** - Enables the application domain
3. **Adaptive Guidance Schedule** - Natural extension that improves quality

---

### Anti-Features (Methods NOT Suitable for GP Guidance)

These methods are either incompatible with external guidance or would require significant modification.

| Method | Why Unsuitable | Alternative |
|--------|----------------|-------------|
| **Consistency Models / Flow Map Matching** | Distills to 1-step generation, no iterative ODE to guide | Use standard multi-step sampling |
| **Discrete Flow Matching** | Works on categorical data, not continuous embeddings | N/A for SONAR space |
| **Fisher Flow Matching** | Operates on statistical manifolds with Fisher-Rao metric | Standard Euclidean flow |
| **Action Matching** | Learns stochastic dynamics, guidance semantics unclear | CFM-based approaches |
| **SlimFlow / Distillation Methods** | Reduce model size, not relevant for research paper | Full-size models |
| **Variance Exploding (VE) Paths** | Less common, harder to guide due to exploding variance | OT or VP paths |
| **Blockwise Flow Matching** | Temporal segmentation orthogonal to guidance | Single unified model |

---

## Method Comparison Matrix

### Path/Interpolant Comparison

| Interpolant | Formula | Path Shape | Guidance Stability | Training Cost | Recommendation |
|-------------|---------|------------|-------------------|---------------|----------------|
| **Linear (OT)** | x_t = (1-t)x_0 + tx_1 | Straight | HIGH | Base | **Primary choice** |
| **Variance Preserving (VP)** | x_t = alpha_t * x_1 + sigma_t * x_0 | Curved | MEDIUM | Base | Secondary option |
| **Rectified (post-reflow)** | Learned straight paths | Straighter | HIGH | +50% | For few-step |
| **Stochastic Interpolants** | Learnable interpolation | Variable | MEDIUM | +20% | Research option |

### Coupling Strategy Comparison

| Coupling | Description | Path Quality | Compute Cost | When to Use |
|----------|-------------|--------------|--------------|-------------|
| **Independent (I-CFM)** | Random pairing | Baseline | O(1) | Always as baseline |
| **Mini-batch OT (n=256)** | Standard Sinkhorn | Better | O(n^2) | Default improvement |
| **Large Sinkhorn (n=2M)** | Multi-GPU OT | Best | O(n^2) sharded | If compute available |
| **Hungarian (exact)** | Exact OT | Good | O(n^3) | Small batches only |
| **Model-Aligned (MAC)** | Learnable coupling | Best fit | O(n) + training | Novel direction |

### Architecture Comparison

| Architecture | Parameters | Suitable for 1024D | Notes |
|--------------|------------|-------------------|-------|
| **MLP** | ~1M | YES but limited | Fast training, limited capacity |
| **DiT (6 layers)** | ~10M | YES | Good balance, current impl |
| **DiT (12 layers)** | ~50M | YES | Better with 10K+ samples |
| **U-ViT** | Variable | Possible | Alternative to cross-attention |

---

## Guidance Compatibility Assessment

### Methods That Work Well with GP Guidance

| Method | Why Compatible | Integration Approach |
|--------|---------------|---------------------|
| **CFM/OT-CFM** | Clean velocity field, ODE formulation | Add UCB gradient to velocity |
| **Rectified Flow** | Even straighter paths than OT-CFM | Same gradient addition |
| **Guided Flows (CFG)** | Designed for guidance | Replace classifier with GP-UCB |
| **Energy-Based Guidance** | General energy function framework | UCB is an energy function |

### Guidance Integration Techniques

From recent research (ICML 2025 Spotlight - "On the Guidance of Flow Matching"):

1. **Training-Free Asymptotically Exact Guidance** - Most relevant for GP-UCB
2. **Posterior Guidance** - Gradients of external loss at each step
3. **Fast Direct** - GP surrogate fitted to queries, universal guidance direction

**Your current implementation uses posterior guidance with UCB gradients - this is well-founded.**

---

## Feature Dependencies

```
CFM (base)
  |
  +-- OT-CFM (requires Sinkhorn/POT library)
  |     |
  |     +-- Large Sinkhorn (requires multi-GPU sharding)
  |     |
  |     +-- Model-Aligned Coupling (requires coupling training)
  |
  +-- Rectified Flow (requires teacher model + reflow data)
  |
  +-- GP-UCB Guidance (requires trained flow + GP surrogate)
        |
        +-- CFG-Zero* Schedule (already implemented)
        |
        +-- Adaptive Guidance (requires GP uncertainty estimates)
        |
        +-- Manifold Projection (requires encode-decode cycle)
```

---

## Recommended Implementation Priority

### Phase 1: Baselines (Required for Paper)

1. **I-CFM baseline** - Random coupling, linear interpolation
2. **OT-CFM** - Mini-batch OT with Sinkhorn (n=256-1024)
3. **Proper evaluation metrics** - FID-like for embeddings, downstream task perf

### Phase 2: Core Contribution

4. **GP-UCB Guided Sampling** - Already partially implemented
5. **CFG-Zero* tuning** - Optimize zero-init fraction
6. **Ablation studies** - Guidance strength, alpha, ODE steps

### Phase 3: Strengthening Results

7. **Rectified Flow** - One reflow iteration for comparison
8. **Adaptive guidance schedule** - Based on GP uncertainty
9. **Larger coupling batches** - If compute permits

### Phase 4: Optional Extensions

10. **VP path comparison** - For completeness
11. **Model-Aligned Coupling** - If time permits (high novelty)
12. **Multi-fidelity GP** - If evaluation is expensive

---

## What Reviewers Will Expect

### Ablations Required

- Guidance strength (lambda) sweep
- UCB alpha parameter sweep
- Number of ODE steps
- OT coupling vs independent coupling
- Zero-init fraction for CFG-Zero*
- Architecture depth/width (if claiming architecture contribution)

### Baselines Required

- Random sampling (no flow model)
- Flow sampling without guidance
- CFM with I-CFM coupling
- OT-CFM
- Possibly: VAE-based generation for comparison

### Metrics Expected

- Sample quality in embedding space (MMD, FID-like)
- Downstream task performance
- Sample diversity
- Optimization efficiency (samples to reach target)
- Wall-clock time comparison

---

## Confidence Assessment

| Finding | Confidence | Source |
|---------|------------|--------|
| CFM/OT-CFM are table stakes | HIGH | Multiple NeurIPS 2024 papers, ICLR 2025 blogpost |
| Guidance via velocity modification works | HIGH | ICML 2025 spotlight paper on FM guidance |
| CFG-Zero* improves guided generation | HIGH | arXiv 2503.18886, applied to Flux/SD3 |
| Large Sinkhorn couplings help | MEDIUM | arXiv 2506.05526, not yet peer-reviewed at major venue |
| Model-Aligned Coupling is novel | MEDIUM | arXiv 2505.23346, preprint status |
| GP-UCB + Flow Matching is novel combination | HIGH | No direct prior work found |
| SONAR embedding space is suitable | MEDIUM | SONAR-LLM (2025) shows embedding space viable |

---

## Sources

### Core Flow Matching Papers
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al.
- [Improving and Generalizing Flow-Based Models with Minibatch OT](https://arxiv.org/abs/2302.00482) - Tong et al.
- [Stochastic Interpolants](https://arxiv.org/abs/2303.08797) - Albergo et al.

### Guidance Methods
- [CFG-Zero*](https://arxiv.org/abs/2503.18886) - Improved CFG for Flow Matching
- [Guided Flows for Generative Modeling](https://arxiv.org/abs/2311.13443) - CFG for Flow Matching
- [On the Guidance of Flow Matching](https://raw.githubusercontent.com/mlresearch/v267/main/assets/feng25s/feng25s.pdf) - ICML 2025 Spotlight

### Recent Advances
- [On Fitting Flow Models with Large Sinkhorn Couplings](https://arxiv.org/abs/2506.05526)
- [Model-Aligned Coupling](https://www.alphaxiv.org/overview/2505.23346v1)
- [Rectified Flow](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)
- [Optimal Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/file/bc8f76d9caadd48f77025b1c889d2e2d-Paper-Conference.pdf) - NeurIPS 2024

### Embedding Space Generation
- [SONAR-LLM](https://arxiv.org/abs/2508.05305) - Generation in SONAR space
- [Flow Matching in Latent Space](https://vinairesearch.github.io/LFM/)

### Bayesian Optimization + Generative Models
- [NF-BO: Latent Bayesian Optimization via Normalizing Flows](https://arxiv.org/abs/2504.14889) - ICLR 2025
- [A Visual Dive into Conditional Flow Matching](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/) - ICLR 2025 Blogpost

### Libraries
- [TorchCFM](https://github.com/atong01/conditional-flow-matching) - Reference implementation
