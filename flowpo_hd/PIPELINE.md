# FlowPO-HD: Manifold-Guided High-Dimensional Prompt Optimization

> **📋 See also:** [`FINDINGS.md`](FINDINGS.md) for experimental results and lessons learned.

## Overview

FlowPO-HD optimizes instruction prompts directly in **1024D SONAR embedding space** without compression. The key innovation is using a Flow Matching model as a "Manifold Keeper" that **regularizes** optimization to stay near the valid instruction manifold.

> **Note**: Initial design used manifold velocity as a "force direction". Experiments showed this doesn't work (see FINDINGS.md). The recommended approach is to use velocity magnitude as a **penalty** instead.

### Why This Approach?

| Aspect | lido_pp (compressed) | FlowPO-HD (direct) |
|--------|---------------------|-------------------|
| Latent dim | 128D (8:1 compression) | 1024D (no compression) |
| Compression loss | ~10% cosine loss | 0% |
| Adversarial risk | Low (smooth latent) | High (mitigated by ManifoldKeeper) |
| GP difficulty | Easy (128D) | Hard (mitigated by TuRBO-1024) |

**FlowPO-HD Advantages:**
- Full SONAR fidelity - no information loss from compression
- ManifoldKeeper prevents "adversarial examples" that decode poorly
- TuRBO trust regions handle curse of dimensionality

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         FlowPO-HD Pipeline                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐   ┌─────────────────┐   ┌───────────────────────┐ │
│  │  SONAR      │   │ ManifoldKeeper  │   │   TuRBO-1024          │ │
│  │  Encoder    │──▶│ (15M params)    │──▶│   Trust Regions       │ │
│  │  1024D      │   │ Flow Matching   │   │   ARD scaling         │ │
│  └─────────────┘   └─────────────────┘   └───────────────────────┘ │
│        │                   │                        │               │
│        │                   │                        │               │
│        ▼                   ▼                        ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Flow-Guided Acquisition                         │   │
│  │   x_{k+1} = x_k + η·∇α_GP(x_k) + λ·v_θ(x_k, t=0.9)          │   │
│  │                                                              │   │
│  │   ∇α_GP: GP acquisition gradient (UCB)                      │   │
│  │   v_θ:   Manifold velocity (points towards valid text)      │   │
│  │   λ:     Adaptive weight (0.5 → 2.0)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────┐   ┌─────────────────┐   ┌───────────────────────┐ │
│  │  SONAR      │◀──│ Candidate       │──▶│   LLM Evaluation      │ │
│  │  Decoder    │   │ Embedding       │   │   GSM8K Error Rate    │ │
│  └─────────────┘   └─────────────────┘   └───────────────────────┘ │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. ManifoldKeeper (`manifold_keeper.py`)

MLP velocity field that learns the flow from noise to valid instruction embeddings.

**Architecture:**
```
Input: x(1024D) + t
       │
       ▼
┌──────────────────────────┐
│  TimestepEmbedding       │
│  t → sinusoidal → MLP    │
│  Output: 2048D           │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  ManifoldResBlock ×3     │
│  x → AdaLN(t) → MLP → +x │
│  1024 → 2048 → 1024      │
│  + Residual connection   │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Output Projection       │
│  1024D velocity          │
│  (zero-initialized)      │
└──────────────────────────┘
```

**Parameters:** ~15M
- Input projection: 1024 × 1024 = 1M
- Time embedding: 256 × 2048 × 2 = 1M
- ResBlocks: 3 × (1024×2048 + 2048×1024) = 12.5M
- Output: 1024 × 1024 = 1M

**Key design:**
- **AdaLN conditioning**: Allows timestep-dependent behavior
- **Zero-init output**: Starts as identity flow for stable training
- **No bottleneck**: Full 1024D throughout (unlike autoencoder)

### 2. TuRBO-1024 (`turbo_1024.py`)

Trust region manager adapted for 1024D space.

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| L_init | 0.4 | Smaller initial for high-D |
| L_max | 1.6 | Standard TuRBO |
| L_min | 0.0078 | 2^-7 |
| τ_succ | 3 | Expand after 3 successes |
| τ_fail | 128 | ceil(1024/8) for high-D |

**ARD scaling formula:**
```
L_i = λ_i × L / geom_mean(λ)
```
- Dimensions with large lengthscale (smooth) get wider bounds
- Dimensions with small lengthscale (sensitive) get tighter bounds
- Volume preserved: ∏ L_i = L^d

### 3. Flow-Guided Acquisition (`flow_guided_acquisition.py`)

Combines GP gradient with manifold regularization.

> **⚠️ IMPORTANT FINDING**: Using velocity as a "force direction" doesn't work well.
> See `FINDINGS.md` for detailed experimental results.

**Recommended approach:**
```
x_{k+1} = x_k + η·∇α_GP(x_k) - λ·penalty(x_k)
```

Where `penalty(x) = ||v_θ(x, t=0.9)||²` penalizes high velocity magnitude.

**What Works:**
- ✅ Seeding from perturbations of training data
- ✅ Velocity magnitude as soft penalty
- ✅ Proximity to training data constraint

**What Doesn't Work:**
- ❌ Using v(x, t) as direction to push towards manifold
- ❌ ODE projection of optimized embeddings
- ❌ Velocity as manifold distance metric

**Why the original approach fails:**
Flow matching learns `x_t = (1-t)·noise + t·data` transport.
The velocity v(x, t) is only meaningful for interpolated states,
NOT for arbitrary off-manifold points or real data.

### 4. GP Configuration (SAAS + qLogEI - Benchmark Winner)

> **NEW in v2.0:** Based on GP benchmark study on 1024D SONAR space, SAAS + medium_600 achieved **Spearman 0.87** correlation between predicted and actual rankings.

**Recommended: SAAS GP with Warm-Start**
- Uses pre-evaluated HbBoPs data (~26 points with fidelity ≥ 600)
- SAAS (Sparse Axis-Aligned Subspaces) via MCMC identifies relevant dimensions
- qLogEI acquisition (numerically stable, better than UCB)
- No cold-start problem - GP starts with real data

```
┌─────────────────────────────────────────────────────────────┐
│  SAAS GP Pipeline                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HbBoPs Results (267 evals)                                 │
│         │                                                   │
│         ▼ filter(fidelity >= 600)                          │
│  Medium-Fidelity Data (~26 points)                         │
│         │                                                   │
│         ▼ SONAR encode + Beta posterior smoothing           │
│  Warm-Start Tensors (X, y, variances)                       │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SAAS GP (Fully Bayesian)                            │   │
│  │  - NUTS MCMC (warmup=128, samples=64)               │   │
│  │  - HalfCauchy prior on lengthscales (sparsity)      │   │
│  │  - Identifies ~5-10 relevant dims out of 1024       │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  qLogEI Acquisition                                  │   │
│  │  - Log Expected Improvement (numerically stable)    │   │
│  │  - Marginalizes over MCMC samples                   │   │
│  │  - Optional velocity penalty filter                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**SAAS Configuration (from benchmark):**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| warmup_steps | 128 | NUTS burn-in |
| num_samples | 64 | Posterior samples |
| thinning | 2 | Reduce autocorrelation |
| min_fidelity | 600 | medium_600 strategy |

**Benchmark Results:**
| Strategy | Spearman ↑ | RMSE | Coverage90 |
|----------|------------|------|------------|
| SAAS + medium_600 | **0.87** | 0.020 | 96% |
| SAAS + high_1000 | 0.60 | 0.033 | 94% |
| Isotropic | -0.26 | 0.036 | 75% |

**Fallback: UCB (for debugging)**
- β_start = 4.0 (high exploration)
- β_end = 2.0 (more exploitation)
- Linear decay over iterations

---

## Training Pipeline

### Phase 1: Data Preparation

```bash
# Encode APE instructions with SONAR (unnormalized)
# Output: flowpo_hd/data/sonar_unnorm.pt
```

**Data source:** `lipo/data/ape_instructions.json` (2000 instructions)

**SONAR settings:**
- `normalize=False` (decoder requires natural magnitude ~0.18)
- `source_lang="eng_Latn"`

### Phase 2: ManifoldKeeper Training

```bash
uv run python -m flowpo_hd.training.train_manifold_keeper \
    --epochs 50000 \
    --batch-size 256 \
    --lr 1e-4
```

**Training:**
- OT-CFM loss: ||v_pred - (x_1 - x_0)||²
- OT pairing via Sinkhorn (GPU-friendly)
- U-shaped timestep sampling (more weight at t≈0, t≈1)
- Early stopping with patience=2000

**Target:** >90% valid instruction generation rate

### Phase 3: Optimization

```bash
uv run python -m flowpo_hd.scripts.run_flowpo_hd \
    --iterations 50 \
    --manifold-keeper-path flowpo_hd/checkpoints/best.pt
```

---

## Key Parameters

### FlowPOHDConfig

```python
@dataclass
class FlowPOHDConfig:
    # SONAR
    sonar_dim: int = 1024           # Fixed by SONAR
    sonar_normalize: bool = False   # Keep unnormalized for decoder

    # ManifoldKeeper
    mk_hidden_dim: int = 2048       # Hidden dimension
    mk_num_blocks: int = 3          # Residual blocks
    mk_time_dim: int = 256          # Timestep embedding
    mk_dropout: float = 0.1

    # TuRBO
    turbo_L_init: float = 0.4       # Smaller for 1024D
    turbo_tau_fail: int = 128       # ceil(1024/8)

    # Flow-Guided Acquisition
    fga_manifold_time: float = 0.9  # Near-clean time
    fga_lambda_start: float = 0.5   # Initial manifold weight
    fga_lambda_end: float = 2.0     # Final manifold weight
    fga_num_steps: int = 50         # Gradient steps
    fga_num_restarts: int = 32      # Random restarts

    # GP
    gp_ucb_beta_start: float = 4.0  # High exploration
    gp_ucb_beta_end: float = 2.0    # More exploitation
    gp_switch_threshold: int = 30   # Switch to SAAS
```

---

## Verification

### ManifoldKeeper Quality Test

```bash
uv run python -m flowpo_hd.scripts.evaluate_manifold
```

**Metrics:**
1. **Sample validity rate**: noise → ODE → decode → valid English?
   - Target: >90%
2. **Reconstruction cosine**: text → SONAR → project → decode → re-encode
   - Target: >0.85
3. **Velocity quality**: does v(x, t=0.9) improve text validity?

### End-to-End Test (no LLM)

```bash
uv run python -m flowpo_hd.scripts.run_flowpo_hd \
    --iterations 10 \
    --skip-llm-eval
```

**Checks:**
- GP fits correctly
- TuRBO adapts (expand/shrink/restart)
- Candidates decode to valid text

### Full Optimization

```bash
tmux new-session -d -s flowpo_hd \
    "CUDA_VISIBLE_DEVICES=0,1 uv run python -m flowpo_hd.scripts.run_flowpo_hd \
        --iterations 50 2>&1 | tee flowpo_hd/results/run_$(date +%Y%m%d_%H%M%S).log"
```

---

## File Structure

```
flowpo_hd/
├── __init__.py
├── config.py                      # FlowPOHDConfig dataclass
├── manifold_keeper.py             # MLP velocity field
├── turbo_1024.py                  # TuRBO for 1024D
├── flow_guided_acquisition.py     # GP + manifold optimization (UCB & SAAS)
├── saas_gp.py                     # NEW: SAAS GP with qLogEI (benchmark winner)
├── warm_start.py                  # NEW: HbBoPs data loading for warm-start
├── utils.py                       # Utilities
├── training/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset and DataLoader
│   └── train_manifold_keeper.py   # OT-CFM training
├── scripts/
│   ├── run_flowpo_hd.py           # Main optimization
│   └── evaluate_manifold.py       # Quality metrics
├── data/
│   ├── sonar_unnorm.pt            # SONAR embeddings (gitignored)
│   └── warm_start_embeddings.pt   # Cached warm-start embeddings (gitignored)
├── checkpoints/                   # Model checkpoints (gitignored)
├── results/                       # Run results (gitignored)
├── PIPELINE.md                    # This file
└── FINDINGS.md                    # Experimental results
```

---

## References

1. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
2. **OT-CFM**: Liu et al., "Improving the Training of Rectified Flows" (2024)
3. **TuRBO**: Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization" (NeurIPS 2019)
4. **SONAR**: Duquenne et al., "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations" (2023)
5. **SAAS**: Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces" (NeurIPS 2021)
