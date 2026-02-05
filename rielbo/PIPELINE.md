# RieLBO: Spherical Subspace Bayesian Optimization

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Algorithm Overview](#algorithm-overview)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Experimental Setup](#experimental-setup)
7. [Results](#results)
8. [Alternative Methods (Notes)](#alternative-methods-notes)
9. [References](#references)

---

## Executive Summary

**Spherical Subspace BO** is a sample-efficient Bayesian Optimization algorithm for high-dimensional molecular latent spaces. The key innovation is projecting the unit hypersphere S^(D-1) to a low-dimensional subspace S^(d-1) via orthonormal projection, enabling tractable GP modeling with limited data.

**Key Results:**
- **+24.5% improvement** over cold start on pdop task
- **+13.7% improvement** over cold start on adip task
- Outperforms TuRBO baseline (full 256D GP) by +1.6% on pdop
- ~0.25 seconds per iteration (100× faster than SAASBO)

---

## Problem Statement

### The Challenge of High-Dimensional BO

Bayesian Optimization with Gaussian Processes suffers from the **curse of dimensionality**:

```
Given: N observations in D-dimensional space
Ratio: N/D = points per dimension

If N/D << 1: GP overfits (memorizes training data, poor generalization)
If N/D > 1:  GP generalizes (meaningful uncertainty estimates)
```

**Our setting:**
- SELFIES VAE latent space: D = 256
- Cold start budget: N = 100
- Ratio: 100/256 = **0.39 points/dim** → severe overfitting

### Symptoms of GP Overfitting

| Metric | Healthy | Overfitting |
|--------|---------|-------------|
| Train correlation | 0.7-0.9 | ≈1.0 |
| Lengthscale min | >0.1 | <0.01 |
| Posterior std ratio | >0.05 | <0.01 |
| LOO correlation | >0.5 | <0.3 |

When GP overfits, it provides no useful guidance for exploration.

---

## Algorithm Overview

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Spherical Subspace BO Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. Encode                2. Normalize              3. Project to Subspace     │
│   ┌────────────────┐      ┌────────────────┐        ┌──────────────────┐       │
│   │ SMILES → VAE   │  ─►  │ x ─► u = x/‖x‖ │   ─►   │ v = normalize(uA)│       │
│   │ x ∈ ℝ^256      │      │ u ∈ S^255      │        │ v ∈ S^15         │       │
│   └────────────────┘      └────────────────┘        └──────────────────┘       │
│                                                            │                    │
│                                                            ▼                    │
│                                                   ┌──────────────────┐         │
│                                                   │ GP on S^15       │         │
│                                                   │ ArcCosine kernel │         │
│                                                   └──────────────────┘         │
│                                                            │                    │
│   6. Evaluate             5. Decode               4. Acquire & Lift            │
│   ┌────────────────┐      ┌────────────────┐      ┌──────────────────┐        │
│   │ oracle(SMILES) │  ◄─  │ VAE ─► SMILES  │  ◄─  │ v* ─► u* = Av*   │        │
│   │ score ∈ [0,1]  │      │ x* = u* × r̄    │      │ Thompson/EI/UCB  │        │
│   └────────────────┘      └────────────────┘      └──────────────────┘        │
│                                    │                                           │
│                           7. Update GP with (v*, score)                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Algorithm Pseudocode

```python
Algorithm: Spherical Subspace BO

Input:
    D = 256          # Original latent dimension
    d = 16           # Subspace dimension
    N_cold = 100     # Cold start samples
    N_iter = 500     # BO iterations

1. Initialize:
    A ← QR(random(D, d))        # Orthonormal projection matrix
    X_cold ← encode(SMILES_cold)
    U_cold ← X_cold / ‖X_cold‖   # Directions on S^(D-1)
    r̄ ← mean(‖X_cold‖)           # Mean norm for reconstruction

2. For i = 1 to N_iter:
    # Project to subspace
    V ← normalize(U @ A)         # V ∈ S^(d-1)

    # Fit GP on subspace sphere
    GP ← fit_gp(V, Y, kernel=ArcCosine)

    # Optimize acquisition
    v* ← thompson_sampling(GP, candidates)

    # Lift to original space
    u* ← normalize(v* @ A^T)     # u* ∈ S^(D-1)
    x* ← u* × r̄                  # Reconstruct embedding

    # Decode and evaluate
    smiles* ← decode(x*)
    score* ← oracle(smiles*)

    # Update
    U ← append(U, u*)
    Y ← append(Y, score*)

Output: best_smiles, best_score
```

---

## Mathematical Foundation

### 1. Spherical Representation

**Motivation:** VAE latent codes have approximately constant norm (embedding vectors cluster near a hypersphere). We decompose embeddings into direction and magnitude:

```
x ∈ ℝ^D (VAE embedding)
u = x / ‖x‖ ∈ S^(D-1) (unit direction)
r = ‖x‖ ∈ ℝ^+ (magnitude)
```

**Key insight:** Directions carry semantic information; magnitudes are approximately constant.

### 2. Orthonormal Subspace Projection

**Construction:** Random orthonormal matrix via QR decomposition:

```
A_raw ∈ ℝ^{D×d} ∼ N(0, 1)
A, _ = QR(A_raw)

Properties:
    A^T A = I_d     (orthonormal columns)
    ‖Ax‖ ≤ ‖x‖     (non-expansive)
```

**Projection:** S^(D-1) → S^(d-1)

```
Project:  u ↦ v = normalize(u A) = (u A) / ‖u A‖
Lift:     v ↦ u = normalize(v A^T) = (v A^T) / ‖v A^T‖
```

**Why this works:**
- Random projection preserves pairwise angles (Johnson-Lindenstrauss)
- Spherical structure is maintained (both input and output are unit spheres)
- QR ensures no information is doubly-counted

**Points per dimension improvement:**
```
Original: 100 / 256 = 0.39 pts/dim → overfitting
Subspace: 100 / 16  = 6.25 pts/dim → generalization ✓
```

### 3. ArcCosine Kernel

**Definition:** Kernel based on geodesic distance on the sphere:

```
k(x, y) = 1 - arccos(x̂ · ŷ) / π

where x̂ = x / ‖x‖, ŷ = y / ‖y‖
```

**Properties:**
| Property | Value |
|----------|-------|
| Range | [0, 1] |
| k(x, x) | 1 (identical points) |
| k(x, -x) | 0 (antipodal points) |
| Lengthscale | **None** (no hyperparameter to overfit) |

**Why no lengthscale:** The geodesic distance on a unit sphere is already normalized. Adding a lengthscale would distort the natural geometry.

**Implementation:**

```python
class ArcCosineKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False  # Critical: no learnable lengthscale

    def forward(self, x1, x2, diag=False, **params):
        # Normalize inputs
        x1_norm = x1 / (x1.norm(dim=-1, keepdim=True) + 1e-8)
        x2_norm = x2 / (x2.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity
        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.T

        # Clamp for numerical stability
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)

        # Geodesic kernel
        return 1.0 - torch.arccos(cos_sim) / torch.pi
```

### 4. Magnitude Reconstruction

**Problem:** After lifting v* → u*, we need to reconstruct the full embedding x* = u* × r.

**Solution:** Use mean norm from training data:

```
r̄ = mean(‖X_train‖)
x* = u* × r̄
```

**Why this works:**
- VAE embeddings cluster near a hypersphere of radius r̄
- More sophisticated approaches (NormPredictor MLP) showed no improvement
- Simplicity reduces overfitting risk

### 5. Acquisition Functions

**Thompson Sampling (recommended):**

```python
# Draw one posterior sample and maximize
f_sample ~ GP_posterior(V)
v* = argmax_v f_sample(v)

# Implementation
thompson = MaxPosteriorSampling(model=gp, replacement=False)
v_opt = thompson(candidates.unsqueeze(0), num_samples=1)
```

**Advantages:**
- O(N) complexity (one sample evaluation)
- Natural exploration-exploitation balance
- No hyperparameters to tune

**Expected Improvement:**

```
EI(v) = E[(f(v) - f_best)^+]
      = (μ(v) - f_best) Φ(Z) + σ(v) φ(Z)

where Z = (μ(v) - f_best) / σ(v)
```

**Upper Confidence Bound:**

```
UCB(v) = μ(v) + β × σ(v)

Default: β = 2.0 (≈95% confidence bound)
```

### 6. Trust Region (TuRBO-style)

**Candidate generation:** Sobol sequence centered on best point:

```python
# Find best point in subspace
v_best = V[Y.argmax()]

# Trust region bounds
tr_lb = v_best - length / 2
tr_ub = v_best + length / 2

# Generate Sobol candidates in trust region
sobol = SobolEngine(d, scramble=True)
pert = sobol.draw(n_candidates)  # Uniform [0, 1]^d
v_cand = tr_lb + (tr_ub - tr_lb) * pert

# Project to sphere
v_cand = normalize(v_cand)
```

**Default:** `trust_region = 0.8`

---

## Implementation Details

### Core Class: SphericalSubspaceBO

**File:** `rielbo/subspace_bo.py`

```python
class SphericalSubspaceBO:
    def __init__(
        self,
        codec,                    # SELFIESVAECodec
        oracle,                   # GuacaMolOracle
        input_dim: int = 256,     # D (SELFIES VAE latent)
        subspace_dim: int = 16,   # d (GP operates on S^(d-1))
        device: str = "cuda",
        n_candidates: int = 2000, # Sobol candidates
        ucb_beta: float = 2.0,    # UCB exploration parameter
        acqf: str = "ts",         # "ts", "ei", "ucb"
        trust_region: float = 0.8,
        seed: int = 42,
        kernel: str = "arccosine",# "arccosine", "matern"
    ):
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `cold_start(smiles_list, scores)` | Initialize with N pre-scored molecules |
| `step()` | Single BO iteration, returns `{score, best_score, smiles, is_duplicate, ...}` |
| `optimize(n_iterations)` | Full optimization loop with progress bar |
| `project_to_subspace(u)` | S^(D-1) → S^(d-1) |
| `lift_to_original(v)` | S^(d-1) → S^(D-1) |

**Internal State:**

```python
self.A           # Orthonormal projection [D, d]
self.train_X     # Original embeddings [N, D]
self.train_U     # Directions [N, D] on S^(D-1)
self.train_V     # Subspace [N, d] on S^(d-1)
self.train_Y     # Scores [N]
self.mean_norm   # Mean embedding norm for reconstruction
self.gp          # SingleTaskGP with ArcCosine kernel
self.history     # Tracked metrics per iteration
```

### SELFIES VAE Codec

**File:** `shared/guacamol/codec.py`

**Architecture:**
- Transformer encoder/decoder
- 2 bottleneck tokens × 128 d_model = **256D latent space**
- SELFIES molecular representation (guarantees valid molecules)

```python
codec = SELFIESVAECodec.from_pretrained(device="cuda")

# Encode SMILES to embeddings
embeddings = codec.encode(["CCO", "c1ccccc1"])  # [2, 256]

# Decode embeddings to SMILES
smiles_list = codec.decode(embeddings)
```

### GP Diagnostics

**File:** `rielbo/gp_diagnostics.py`

**Tracked Metrics:**

| Metric | Healthy Range | Overfitting Indicator |
|--------|---------------|----------------------|
| `train_correlation` | 0.7-0.9 | >0.99 |
| `lengthscale_min` | >0.1 | <0.01 |
| `train_std_ratio` | >0.05 | <0.01 |
| `candidate_in_hull_frac` | >0.5 | <0.5 (extrapolating) |

**Usage:**

```python
from rielbo.gp_diagnostics import GPDiagnostics

diag = GPDiagnostics()
metrics = diag.analyze(gp, train_X, train_Y, candidate_X)
diag.log_summary(metrics)

# Output:
# [Iter 100] GP Diag: train_corr=0.85, ℓ=[0.2,1.5], noise=1e-2, std_ratio=0.15 ✓
# [Iter 200] GP Diag: train_corr=0.99, ℓ=[0.01,0.5], noise=1e-4, std_ratio=0.01 ⚠️ OVERFIT
```

---

## Experimental Setup

### Standard Benchmark Configuration

```bash
# Cold start
N_COLD_START=100

# BO iterations
N_ITERATIONS=500

# Seeds for statistical significance
SEEDS=(42 43 44 45 46)

# Tasks (GuacaMol molecular optimization)
TASKS=(adip med2 pdop)
```

### CLI Usage

```bash
# Run Spherical Subspace BO
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --task-id adip \
    --subspace-dim 16 \
    --n-cold-start 100 \
    --iterations 500 \
    --acqf ts \
    --kernel arccosine \
    --trust-region 0.8 \
    --seed 42
```

### Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--task-id` | `pdop` | GuacaMol task: `pdop`, `adip`, `med2` |
| `--n-cold-start` | `100` | Number of pre-evaluated molecules |
| `--iterations` | `500` | BO iterations |
| `--subspace-dim` | `16` | Subspace dimension d |
| `--kernel` | `arccosine` | Kernel: `arccosine`, `matern` |
| `--acqf` | `ts` | Acquisition: `ts`, `ei`, `ucb` |
| `--ucb-beta` | `2.0` | UCB exploration (only for `--acqf ucb`) |
| `--trust-region` | `0.8` | Trust region length |
| `--n-candidates` | `2000` | Sobol candidates |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Compute device |

### Benchmark Script

```bash
#!/bin/bash
# Run full benchmark across seeds

for task in adip med2; do
  for seed in 42 43 44 45 46; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
      --task-id $task \
      --subspace-dim 16 \
      --acqf ts \
      --kernel arccosine \
      --trust-region 0.8 \
      --n-cold-start 100 \
      --iterations 500 \
      --seed $seed
  done
done
```

---

## Results

### Main Results (2026-02-04)

| Task | Cold Start | Subspace BO | TuRBO (256D) | Improvement |
|------|------------|-------------|--------------|-------------|
| **pdop** | 0.4558 | **0.5676 ± 0.033** | 0.5587 ± 0.011 | **+24.5%** |
| **adip** | 0.4910 | **0.5582** | - | **+13.7%** |
| **med2** | 0.1856 | 0.1856 | - | +0.0%* |

*Med2 shows no improvement because the score range is extremely narrow [0.02, 0.19] and cold start already found near-optimal molecule.

### Subspace BO vs Alternatives

| Method | adip Score | vs Cold Start | Speed |
|--------|-----------|---------------|-------|
| Cold Start | 0.4910 | baseline | - |
| **Subspace BO (S^15)** | **0.5582** | **+13.7%** | ~0.25 s/iter |
| PLS BO | 0.5576 | +13.6% | ~0.3 s/iter |
| Graph Laplacian BO | 0.5394 | +9.9% | ~1.0 s/iter |
| TuRBO (256D) | 0.5066* | +3.2% | ~0.3 s/iter |
| SAASBO | - | - | ~25 s/iter ❌ |

*TuRBO struggles due to overfitting in 256D with 100 samples.

### Key Findings

1. **Random projection works surprisingly well**
   - PLS (supervised) achieves 0.5576, nearly identical to random Subspace BO (0.5582)
   - VAE latent space already compressed information; PLS doesn't find special directions

2. **ArcCosine kernel prevents overfitting**
   - No lengthscale parameter to overfit
   - Respects spherical geometry naturally

3. **16D is the sweet spot**
   - 100 pts / 16 dims = 6.25 pts/dim → GP generalizes
   - Lower dims lose information; higher dims overfit

4. **Mean norm reconstruction is sufficient**
   - Learned NormPredictor MLP showed no improvement
   - Simplicity reduces overfitting risk

---

## Alternative Methods (Notes)

### 1. TuRBO Baseline

**File:** `rielbo/turbo_baseline.py`

Standard Trust Region BO in full 256D latent space.

**Key differences from Subspace BO:**
- GP operates in full ℝ^256 (not sphere)
- Uses Matern-5/2 kernel with ARD lengthscales
- Trust region adapts based on success/failure

**Trust Region State Machine:**

```python
class TurboState:
    # Expand on success streak
    if success_counter >= success_tolerance:
        length = min(2.0 * length, length_max)

    # Shrink on failure streak
    if failure_counter >= failure_tolerance:
        length = length / 2.0

    # Restart if collapsed
    if length < length_min:
        restart_triggered = True
```

**Usage:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.turbo_baseline \
    --task-id adip --n-cold-start 100 --iterations 500 --seed 42
```

### 2. PLS BO (Partial Least Squares)

**File:** `rielbo/pls_bo.py`

Supervised dimensionality reduction via PLS correlation with objective.

**Key idea:** Find directions in ℝ^D that maximally correlate with scores.

```
PLS: max corr(X·w, y)
vs
PCA: max var(X·w)
```

**Findings:**
- Performs nearly identically to random projection (0.5576 vs 0.5582)
- Conclusion: VAE already removed irrelevant dimensions

**Usage:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_pls \
    --n-components 15 --task-id adip --iterations 500
```

### 3. PLS Spherical BO

**File:** `rielbo/pls_spherical_bo.py`

Combines PLS projection with spherical geometry for fair comparison.

**Pipeline:**
```
x → u = normalize(x) → PLS transform → v = normalize(PLS(u)) → GP with ArcCosine
```

### 4. Graph Laplacian BO

**File:** `rielbo/graph_laplacian_gp.py`

Manifold-aware BO using spectral embedding of data graph.

**Algorithm:**
1. Build k-NN graph from labeled + unlabeled anchor points
2. Compute normalized Graph Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
3. Extract eigenvectors (smallest eigenvalues, excluding trivial 0)
4. Use RBF kernel in spectral embedding space

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | 15 | k for k-NN graph |
| `n_components` | 32 | Spectral embedding dimension |
| `n_anchors` | 2000 | Unlabeled anchor points |
| `graph_sigma` | "auto" | Gaussian edge weight bandwidth |
| `rebuild_interval` | 50 | Iterations between graph rebuilds |

**Advantage:** Captures manifold structure, avoids "forbidden zones" (invalid molecule regions)

**Result:** 0.5394 on adip (+9.9% vs cold start, but -3.2% vs Subspace BO)

**Usage:**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_graph \
    --n-neighbors 15 --n-components 32 --task-id adip --iterations 500
```

### 5. Flow-Based Methods (Experimental)

**Files:**
- `rielbo/velocity_network.py` - DiT-style velocity network
- `rielbo/flow_model.py` - Flow matching wrapper
- `rielbo/guided_flow.py` - GP-guided sampling

These methods use trained flow models to sample on the manifold.

**Not recommended:** Current implementation shows poor round-trip fidelity (~0.50 cosine instead of >0.95).

### 6. Matern Kernel Alternative

For non-spherical settings, use Matern-5/2:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --kernel matern --subspace-dim 16 --task-id adip
```

**Tradeoff:** Learnable lengthscales can overfit with limited data.

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| GP overfit | train_corr=1.0, no improvement | Reduce `--subspace-dim` |
| Too many duplicates | >30% duplicates | Increase `--n-candidates`, add perturbation |
| Slow iterations | >1s per iteration | Check CUDA, reduce `--n-candidates` |
| No improvement | Flat best_score | Task may be saturated (see med2) |
| Numerical errors | LinAlgError | Increase GP noise prior |

---

## References

### Core Methods

1. **TuRBO**: Eriksson et al. (2019) "Scalable Global Optimization via Local Bayesian Optimization" NeurIPS
2. **ArcCosine Kernel**: Cho & Saul (2009) "Kernel Methods for Deep Learning" NeurIPS
3. **Random Projection**: Johnson-Lindenstrauss Lemma, Achlioptas (2003)

### Related Work

4. **SELFIES VAE**: Krenn et al. (2020) "Self-referencing embedded strings (SELFIES)"
5. **LOLBO**: Gruver et al. (2023) "Effective Surrogate Models for Molecular Property Prediction"
6. **Graph Laplacian**: Belkin & Niyogi (2003) "Laplacian Eigenmaps for Dimensionality Reduction"
7. **PLS**: Wold et al. (2001) "PLS-regression: a basic tool of chemometrics"

### High-Dimensional BO

8. **BAxUS**: Papenmeier et al. (2022) "Increasing the Scope as You Learn"
9. **SAASBO**: Eriksson & Jankowiak (2021) "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces"
10. **NFBO**: Grosnit et al. (2024) "High-Dimensional Bayesian Optimisation with Normalizing Flows"

---

## File Structure

```
rielbo/
├── # Core BO (Recommended)
├── subspace_bo.py           # SphericalSubspaceBO - main algorithm
├── run_guacamol_subspace.py # CLI entry point
├── gp_diagnostics.py        # GP health monitoring
│
├── # Baselines
├── turbo_baseline.py        # TuRBO in full 256D
├── pls_bo.py                # PLS-projected Euclidean BO
├── pls_spherical_bo.py      # PLS with sphere geometry
├── graph_laplacian_gp.py    # Manifold-aware BO
│
├── # Run Scripts
├── run_guacamol_pls.py      # PLS BO entry point
├── run_guacamol_pls_spherical.py
├── run_guacamol_graph.py    # Graph Laplacian BO entry point
│
├── # Flow Model (Experimental)
├── velocity_network.py      # DiT-style velocity network
├── flow_model.py            # FlowMatchingModel
├── guided_flow.py           # GP-UCB guided sampling
├── norm_predictor.py        # Direction → magnitude MLP
│
├── # Benchmarking
├── benchmark/
│   ├── base.py              # BaseMethod interface
│   ├── runner.py            # Multi-seed benchmark orchestration
│   ├── plotting.py          # Publication-quality plots
│   └── methods/             # Method wrappers
│
└── results/                 # Output JSON files
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{rielbo2026,
  title={Spherical Subspace Bayesian Optimization for High-Dimensional Molecular Design},
  author={...},
  journal={...},
  year={2026}
}
```
