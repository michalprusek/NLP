# RieLBO Pipeline: Spherical Subspace BO (V2 Explore)

## Overview

RieLBO performs sample-efficient Bayesian Optimization in the 256D latent space of a SELFIES VAE by projecting onto a 16D subspace sphere. The key idea: a GP on S^15 with ~100 training points is tractable, while a GP on S^255 is not.

**Best configuration**: V2 Explore preset — ArcCosine kernel + geodesic trust region + UR-TR (uncertainty-responsive) + LASS (look-ahead subspace selection) + acquisition schedule (TS/UCB).

**Results (adip, 10 seeds, 500 iter)**: 0.5555 ± 0.013 (p=0.021 vs geodesic baseline)

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

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Spherical Subspace BO Pipeline (Explore)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. Encode                2. Normalize              3. Project to Subspace     │
│   ┌────────────────┐      ┌────────────────┐        ┌──────────────────┐       │
│   │ SMILES → VAE   │  ─►  │ x ─► u = x/‖x‖ │   ─►   │ v = normalize(uA)│       │
│   │ x ∈ ℝ^256      │      │ u ∈ S^255      │        │ v ∈ S^15         │       │
│   └────────────────┘      └────────────────┘        └──────────────────┘       │
│                                                    (LASS: best of 50 QR)       │
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
│   │ score ∈ [0,1]  │      │ x* = u* × r̄    │      │ TS/UCB (schedule)│        │
│   └────────────────┘      └────────────────┘      └──────────────────┘        │
│          │                                                                     │
│          ▼                                                                     │
│   7. Update GP with (v*, score)                                                │
│   8. Update UR-TR: expand/shrink/rotate based on GP posterior std              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Steps

### 1. Cold Start: Scoring Initial Molecules

```
Input: 100 SMILES strings from GuacaMol dataset
Output: (embeddings [100, 256], scores [100])
```

Load molecules from the GuacaMol CSV, encode each through the SELFIES VAE encoder to get 256D latent vectors, and score with the GuacaMol oracle (e.g., Amlodipine MPO).

**Key files**: `shared/guacamol/data.py` (data loading), `shared/guacamol/codec.py` (VAE encoding), `shared/guacamol/oracle.py` (scoring).

**Codec architecture**: SELFIES VAE has a Transformer encoder/decoder with:
- 2 bottleneck tokens x 128 d_model = 256D latent
- Encoder: SMILES -> SELFIES -> tokens -> mu, sigma
- We use mu (deterministic) as the embedding

### 2. Spherical Projection: S^255 -> S^15

```
Input: embeddings [N, 256]
Output: directions [N, 16] on S^15
```

**Step 2a — Normalize to unit sphere**:
```python
norms = embeddings.norm(dim=-1)        # save for reconstruction
mean_norm = norms.mean()
U = F.normalize(embeddings, dim=-1)    # [N, 256] on S^255
```

**Step 2b — LASS: Look-Ahead Subspace Selection** (explore preset):

Instead of a single random QR projection, LASS evaluates 50 candidate projections and picks the most informative one:

```python
for k in range(50):
    A_k, _ = torch.linalg.qr(torch.randn(256, 16))
    V_k = normalize(U @ A_k)            # project cold start data
    gp_k = fit_gp(V_k, scores)          # fit ArcCosine GP
    log_ml_k = gp_k.log_marginal_likelihood()
best_A = argmax(log_ml)                 # pick best projection
```

**Criterion**: GP log marginal likelihood — how well the ArcCosine GP explains score variation in this subspace. Higher log ML = the kernel captures meaningful structure.

**CRITICAL**: Max posterior std is WRONG as a LASS criterion. A projection where the GP is maximally uncertain may be one where the score landscape is flat and the GP can't model anything useful.

**Step 2c — Random QR projection** (if LASS disabled):
```python
A_raw = torch.randn(256, 16)
A, _ = torch.linalg.qr(A_raw)          # [256, 16] orthonormal columns
V = F.normalize(U @ A, dim=-1)          # [N, 16] on S^15
```

The orthonormal matrix A preserves angular relationships. Points close on S^255 stay close on S^15. Random QR matches supervised PLS projection (0.558 vs 0.558) — no training needed.

### 3. GP Fitting: ArcCosine Kernel on S^15

```
Input: V [N, 16], Y [N] (scores)
Output: trained GP model
```

**Kernel**: ArcCosine order 0 (geodesic distance kernel):
```
k(x, y) = 1 - arccos(x . y) / pi
```
This is the natural kernel for spherical data — equivalent to a stationary kernel in geodesic distance. No lengthscale parameters to optimize, which prevents overfitting with limited data.

**GP**: BoTorch `SingleTaskGP` with `ScaleKernel(ArcCosineKernel())`. The scale kernel adds an output variance parameter that IS fitted via MLL.

**Fitting**: `fit_gpytorch_mll` optimizes hyperparameters (output scale, noise) via L-BFGS on the exact marginal log-likelihood.

**Diagnostics** (every iteration when verbose): Train correlation, posterior std, noise level. See `rielbo/gp_diagnostics.py`.

**Key insight — GP is NOT the hero**: Train correlation is 0.14–0.54 and *decreases* over time. Noise ~0.98. The GP deliberately underfits — it provides just enough signal to bias sampling toward promising regions. The geodesic trust region does most of the work.

### 4. Candidate Generation: Geodesic Trust Region

```
Input: v_best [1, 16] (best training point on S^15), trust radius
Output: v_candidates [2000, 16] on S^15
```

**Geodesic sampling** (80% local + 20% global):

Local candidates via exponential map on S^15:
```
1. Generate random tangent vectors at v_best
2. Project out component along v_best (orthogonalize)
3. Sample angles uniformly in [0, radius]
4. exp_map: x(theta) = cos(theta) * v_best + sin(theta) * tangent
```

Global candidates: uniform random on S^15 (exploration).

**UR-TR: Uncertainty-Responsive Trust Region** (explore preset):

The trust region radius adapts based on GP posterior std, NOT improvement counting:

| GP Posterior Std | Action | Rationale |
|-----------------|--------|-----------|
| std < 5% of noise_std | TR × 1.5 (expand) | GP is certain everywhere → needs broader exploration |
| std > 15% of noise_std | TR × 0.8 (shrink) | GP has meaningful uncertainty → exploit locally |

This is **counter-intuitive** and opposite of TuRBO: EXPAND when GP is certain (collapsing), SHRINK when GP is uncertain (informative). Driven by GP health, not score improvement.

**Relative thresholds** (`ur_relative=True`, default): Thresholds are expressed as fractions of GP noise_std, making them **scale-invariant** across different datasets, kernels, and score ranges. With the current noise_var≈0.98 (noise_std≈0.99), relative 0.05 gives effective threshold ≈0.05 — nearly identical to the previous absolute value. Use `--no-ur-relative` for absolute thresholds.

**UR-TR parameters** (from V2Config):
- `ur_relative`: True (thresholds relative to noise_std)
- `ur_std_low`: 0.05 (expand when std < 5% of noise_std)
- `ur_std_high`: 0.15 (shrink when std > 15% of noise_std)
- `ur_expand_factor`: 1.5, `ur_shrink_factor`: 0.8
- `ur_tr_min`: 0.1 rad, `ur_tr_max`: 1.2 rad

**Legacy adaptive TR** (geodesic preset, TuRBO-style):
- On 3 consecutive improvements: tr_length grows by 1.5× (capped at 0.8)
- On 10 consecutive failures: tr_length shrinks by 0.5×
- When tr_length < 0.02: restart with fresh random basis
- `max_restarts`: 5

### 5. Acquisition: Thompson Sampling

```
Input: GP model, v_candidates [2000, 16]
Output: v_opt [1, 16]
```

**Thompson Sampling** (default): Draw one sample from the GP posterior over all 2000 candidates, pick the argmax. BoTorch's `MaxPosteriorSampling` handles this.

**UCB** (alternative, `--acqf ucb`): UCB(v) = μ(v) + β × σ(v). Used with `--ucb-beta` parameter.

### 6. Lifting & Reconstruction: S^15 -> S^255 -> R^256

```
Input: v_opt [1, 16]
Output: x_opt [1, 256]
```

**Step 6a — Lift to S^255**:
```python
u_opt = F.normalize(v_opt @ A.T, dim=-1)  # [1, 256] on S^255
```

**Step 6b — Inverse whitening** (if whitening enabled):
```python
u_opt = F.normalize(u_opt @ H.T, dim=-1)  # undo Householder
```

**Step 6c — Norm reconstruction**:
```python
x_opt = u_opt * mean_norm                  # [1, 256] in R^256
```

We use the mean norm from cold start. Simpler and more robust than probabilistic norm reconstruction (tested, no improvement).

### 7. Decode & Score

```
Input: x_opt [1, 256]
Output: score (float), SMILES (str)
```

```python
z = x_opt.reshape(1, 2, 128)              # bottleneck shape
tokens = model.sample(z=z)                 # autoregressive decode
selfies_str = dataset.decode(tokens)       # tokens -> SELFIES
smiles = sf.decoder(selfies_str)           # SELFIES -> SMILES
score = oracle.score(smiles)               # GuacaMol MPO score
```

If decode fails or produces a duplicate SMILES, count as failure (no training data added, TR adjusts accordingly).

### 8. Update & Iterate

- Add (x_opt, u_opt, score, smiles) to training data
- Refit GP every 10 iterations (not every iteration for speed)
- Update UR-TR based on GP posterior std (expand/shrink)
- Update legacy adaptive TR if enabled (grow/shrink/restart)

---

## Three Key V2 Features (Explore Preset)

### LASS — Look-Ahead Subspace Selection

**Problem**: Random QR projection works well on average, but some projections capture more score variation than others.

**Solution**: At cold start, evaluate K=50 random QR projections by fitting a quick GP on each and measuring log marginal likelihood. Pick the projection where the GP best explains the data.

**Implementation**: `_select_best_projection()` in `subspace_bo_v2.py`

**Criterion**: GP log marginal likelihood (NOT posterior std — that picks flat landscapes where GP is uncertain because there's nothing to model).

### UR-TR — Uncertainty-Responsive Trust Region

**Problem**: When GP posterior std → 0, Thompson Sampling degenerates into random local search. Seeds where GP maintains meaningful uncertainty show a trend toward better scores (r=0.488, p=0.152, not statistically significant with n=10).

**Solution**: Monitor GP posterior std and adapt geodesic TR radius:
- **Low std (collapsing)**: EXPAND radius — force broader exploration to escape dead regions
- **High std (informative)**: SHRINK radius — exploit GP's useful predictions locally
- **Sustained collapse**: Rotate to a fresh QR subspace

**Key insight**: This is opposite of TuRBO-style adaptive TR, which expands on success and shrinks on failure. UR-TR is driven by GP health, not score improvement.

**Implementation**: `_update_ur_tr()` and `_ur_rotate_subspace()` in `subspace_bo_v2.py`

### Acquisition Schedule

**Problem**: Thompson Sampling relies on GP posterior uncertainty for exploration. When GP std collapses, TS becomes greedy.

**Solution**: Switch to UCB with high β=4.0 when GP std drops below threshold. This forces explicit exploration (high β inflates the upper confidence bound) even when the GP is uninformative.

**Contribution**: +0.010 to mean score (explore 0.5555 vs lass_ur 0.5453).

**Implementation**: `_get_effective_acqf()` in `subspace_bo_v2.py`

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
A_raw ∈ ℝ^{D×d} ~ N(0, 1)
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

**Why random projection works:**
- Random projection preserves pairwise geodesic distances (Johnson-Lindenstrauss on S^{D-1})
- Spherical structure is maintained (both input and output are unit spheres)
- QR ensures orthonormal columns (isometric on the subspace)
- Random QR ≈ supervised PLS (0.558 vs 0.558) — no training data needed
- Jacobian-guided projections (pullback metric) are *worse* than random — decoder sensitivity ≠ objective sensitivity

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
| Smoothness | Matérn-1/2 analog (rough) |

**Why rough is better**: Chemical property landscapes have sharp transitions. ArcCosine o0 (0.5555) beats smoother o2 (0.5458) and Matérn-5/2 ARD (0.5403). With ~100-200 training points on S^15, the 17 extra kernel hyperparameters (16 lengthscales + outputscale) of Matérn ARD overfit.

**Implementation:**

```python
class ArcCosineKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False  # Critical: no learnable lengthscale

    def forward(self, x1, x2, diag=False, **params):
        x1_norm = x1 / (x1.norm(dim=-1, keepdim=True) + 1e-8)
        x2_norm = x2 / (x2.norm(dim=-1, keepdim=True) + 1e-8)

        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.T

        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        return 1.0 - torch.arccos(cos_sim) / torch.pi
```

### 4. Why Subspace Projection Works

The SELFIES VAE 256D latent space has intrinsic dimensionality ~16 (TwoNN=16.8, DANCo=11.3, FisherS=18.9). This means:
- The data lives on a ~16D manifold embedded in R^256
- A GP on S^255 with 100 points has terrible data density
- A random 16D projection captures most of the structure
- A GP on S^15 with 100 points has good data density

Per-task intrinsic dimensionality (TwoNN, score-conditioned, top-100 from 250K ZINC): adip=6, med2=7, pdop=10, rano=11, osmb=19, siga=13, zale=16, valt=22.

### 5. Why Geodesic Trust Region

Euclidean box constraints in R^d make no geometric sense on the sphere:
- A box around a point on S^15 wastes samples outside the sphere
- After projection back to S^15, density is non-uniform

Geodesic trust region samples within a spherical cap:
- All candidates are exactly on S^15
- Uniform density in geodesic distance
- Natural notion of "local" on a curved space

**Geometry**: 0.4 rad cap = 1/12.8M of S^15 surface. Extremely local.

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `rielbo/subspace_bo_v2.py` | Main optimizer (`SphericalSubspaceBOv2`, `V2Config`) |
| `rielbo/subspace_bo.py` | V1 baseline (`SphericalSubspaceBO`) |
| `rielbo/kernels.py` | ArcCosine kernel (order 0, order 2), `create_kernel` factory |
| `rielbo/spherical_transforms.py` | SphericalWhitening, GeodesicTrustRegion |
| `rielbo/gp_diagnostics.py` | GP health monitoring |
| `rielbo/turbo_baseline.py` | TuRBO in full 256D |
| `rielbo/vanilla_bo.py` | Vanilla BO with Hvarfner priors (256D) |
| `rielbo/run_guacamol_subspace_v2.py` | CLI entry point (recommended) |
| `rielbo/run_guacamol_subspace.py` | CLI entry point (v1) |
| `rielbo/ensemble_bo.py` | Multi-scale ensemble BO (K=6, d=[4,8,12,16,20,24]) |
| `rielbo/benchmark/runner.py` | Benchmark orchestrator (wraps V1, not V2) |
| `rielbo/benchmark/methods/` | Per-method adapters (turbo, lolbo, baxus, cmaes, invbo, etc.) |
| `shared/guacamol/codec.py` | SELFIES VAE (encode/decode) |
| `shared/guacamol/oracle.py` | GuacaMol scoring |
| `shared/guacamol/data.py` | Data loaders |

---

## Configuration Reference

### V2Config Presets

| Preset | Features | Score (adip) | Use case |
|--------|----------|------|----------|
| `baseline` | None (matches v1) | ~0.544 | Ablation baseline |
| `geodesic` | Geodesic TR + Adaptive TR | 0.542±0.017 | Reproducible baseline |
| `ur_tr` | Geodesic TR + UR-TR | — | UR-TR only |
| `lass` | Geodesic TR + LASS | — | LASS only |
| `lass_ur` | Geodesic TR + UR-TR + LASS | 0.545±0.016 | Without acqf_schedule |
| **`explore`** | **Geodesic TR + UR-TR + LASS** | **0.556±0.013** | **Recommended (BEST)** |
| `portfolio` | explore + multi-subspace (K=5) | — | Experimental |
| `order2` | ArcCosine order 2 | 0.546±0.013 | Smoother GP |
| `whitening` | Spherical whitening | — | Data centering |
| `full` | order2 + whitening + geodesic + adaptive_dim + prob_norm | — | Experimental |

### CLI Usage

```bash
# Recommended: explore preset (UR-TR + LASS, relative thresholds)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
    --preset explore --task-id adip --iterations 500 --seed 42

# Full benchmark (10 seeds)
for seed in 42 43 44 45 46 47 48 49 50 51; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset explore --task-id adip --iterations 500 --seed $seed
done

# Geodesic baseline (for ablation)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
    --preset geodesic --task-id adip --iterations 500 --seed 42
```

### Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--task-id` | `adip` | GuacaMol task: `adip`, `med2`, etc. |
| `--preset` | `geodesic` | V2Config preset (see table above) |
| `--n-cold-start` | `100` | Number of pre-evaluated molecules |
| `--iterations` | `500` | BO iterations |
| `--subspace-dim` | `16` | Subspace dimension d |
| `--kernel` | `arccosine` | Kernel: `arccosine`, `matern`, `hvarfner` |
| `--acqf` | `ts` | Acquisition: `ts`, `ei`, `ucb` |
| `--ucb-beta` | `2.0` | UCB exploration (only for `--acqf ucb`) |
| `--trust-region` | `0.8` | Trust region length |
| `--n-candidates` | `2000` | Candidates per step |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Compute device |
| `--ur-std-low` | `0.05` | UR-TR: GP std threshold for expanding (relative to noise_std) |
| `--ur-std-high` | `0.15` | UR-TR: GP std threshold for shrinking (relative to noise_std) |
| `--no-ur-relative` | `False` | Use absolute thresholds instead of noise-relative |
| `--kernel-type` | (from preset) | Override: `arccosine`, `geodesic_matern`, `matern` |
| `--kernel-ard` | `False` | Enable per-dimension ARD lengthscales |
| `--kernel-order` | (from preset) | ArcCosine order: 0 or 2 |
| `--adaptive-tr` | (from preset) | Enable TuRBO-style adaptive TR |

---

## Results

### Benchmark Results (adip, 10 seeds, 500 iter)

| Method | Score | Notes |
|--------|-------|-------|
| **V2 Explore** | **0.5555 ± 0.013** | Best — UR-TR + LASS. p=0.021 vs baseline |
| V2 LASS+UR | 0.5453 ± 0.016 | UR-TR + LASS (historical, now = explore) |
| V1 Subspace | 0.5440 ± 0.006 | Benchmark runner (V1) |
| V2 Geodesic | 0.5424 ± 0.017 | Reproducible baseline (0207 batch) |
| Ensemble (K=6) | 0.5422 ± 0.009 | Multi-scale d=[4,8,12,16,20,24] |
| CMA-ES | 0.5371 ± 0.018 | Non-BO evolutionary baseline |
| InvBO | 0.5255 ± 0.017 | Inverse BO (NeurIPS 2024) |
| LOL-BO | 0.5228 ± 0.019 | Latent space BO baseline |
| TuRBO (R^256) | 0.5060 ± 0.007 | Full-dimensional BO |
| Cold Start | 0.4910 | No optimization |

### Explore Preset Per-Seed Results

| Seed | Score |
|------|-------|
| 42 | 0.5571 |
| 43 | 0.5585 |
| 44 | 0.5528 |
| 45 | 0.5368 |
| 46 | 0.5477 |
| **47** | **0.5867** |
| 48 | 0.5560 |
| 49 | 0.5576 |
| 50 | 0.5419 |
| 51 | 0.5600 |

### Ablation: Feature Contributions

| Config | Score | Δ vs geodesic | Component tested |
|--------|-------|---------------|-----------------|
| explore | 0.5555 ± 0.013 | **+0.013** | All three features |
| lass_ur | 0.5453 ± 0.016 | +0.003 | LASS + UR-TR (historical batch) |
| geodesic | 0.5424 ± 0.017 | baseline | Geodesic TR + adaptive TR |
| geodesic w/o adaptive_tr | ~0.533 | −0.009 | Geodesic TR only |

**Key**: UR-TR expands search when GP collapses. LASS picks a better starting projection. Together they provide +0.013 over geodesic baseline.

### Negative Results

**Pullback Metric (2026-02-06):** Tested using the VAE decoder's Riemannian geometry (pullback metric G = J^T J) to guide subspace selection.

| Config | Score (5 seeds) | Finding |
|--------|----------------|---------|
| Pure metric basis | 0.4910 (= cold start) | **Complete failure** |
| Score-weighted metric | 0.5422 ± 0.014 | Worse than random |
| Hybrid 75% metric | 0.5336 ± 0.006 | Low variance, low mean |
| Random QR (with restart fix) | 0.5531 ± 0.018 | Best in pullback framework |

**Conclusion**: Decoder sensitivity (which latent directions change the output most) does NOT correlate with objective sensitivity (which directions improve the score). Random QR projection works best because it provides unbiased exploration.

**Other approaches tested and rejected:**
- **Vanilla BO (Hvarfner, 256D)**: ~33s/iter, marginal improvement. Impractical.
- **SAASBO**: ~25s/iter due to MCMC. Impractical.
- **PCA subspace**: No improvement over random projection.
- **Active Subspace**: Linear projections miss nonlinear VAE structure.
- **PLS BO**: Supervised projection nearly identical to random (0.558 vs 0.558).
- **Graph Laplacian GP**: Experimental, no clear benefit. Removed.
- **ArcCosine order 2**: Smoother kernel (Matérn-5/2-like, 0.5458) worse than order 0 (0.5555).
- **Geodesic Matérn 5/2 ARD**: 0.5403 ± 0.014 — learnable kernels overfit on S^15 with ~100 points.
- **Ensemble (K=6)**: 0.5422 ± 0.009 — halves variance but same mean. Max-std selection doesn't unlock genuinely new exploration.

### Key Findings

1. **Random projection works surprisingly well** — PLS (supervised) achieves 0.558, nearly identical to random QR (0.558). The VAE already compressed information. JL lemma on S^{D-1} preserves geodesic distances when d ≥ ID(data).

2. **ArcCosine kernel prevents overfitting** — No lengthscale parameter to overfit. 0 hyperparams beats 16 ARD lengthscales. Rough kernel (Matérn-1/2 analog) matches sharp chemical property transitions.

3. **16D is the sweet spot** — 100 pts / 16 dims = 6.25 pts/dim → GP generalizes. Validated by intrinsic dimensionality estimates (TwoNN=16.8). Dim sweep d=8..20 shows flat profile — ArcCosine handles excess dims gracefully.

4. **Adaptive restart on GP collapse is the most impactful fix** — drove scores from ~0.533 to 0.555.

5. **GP posterior std predicts seed success** (r=0.746) — Seeds where GP maintains meaningful uncertainty (std > 0.3) perform best. s47 (0.5867, best ever) had sustained GP uncertainty. s45 (0.5368, worst) had early GP collapse.

6. **UR-TR direction is counter-intuitive** — EXPAND when GP std drops (collapsing), SHRINK when std is high. Opposite of TuRBO. Driven by GP health, not score improvement.

7. **LASS criterion matters critically** — Max posterior std picks flat landscapes (WRONG). Max GP log marginal likelihood picks informative projections (CORRECT).

8. **Med2 is effectively unsolvable** — only 0.6% of 20K molecules beat cold start. Score range [0.02, 0.19] is too narrow.

9. **UR-TR relative thresholds are scale-invariant** — Thresholds relative to GP noise_std (5% and 15%) work across different datasets and kernels. With noise_var≈0.98, these match the previously tuned absolute values.

10. **GP is a weak surrogate** — Train correlation 0.14–0.54, noise ~0.98. Geodesic TR does the heavy lifting. GP provides just enough signal to bias sampling.

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| GP overfit | train_corr=1.0, no improvement | Reduce `--subspace-dim` |
| GP collapse | gp_std→0, TS degenerates | Use `--preset explore` (UR-TR handles this) |
| Too many duplicates | >30% duplicates | Increase `--n-candidates`, add perturbation |
| Slow iterations | >1s per iteration | Check CUDA, reduce `--n-candidates` |
| No improvement | Flat best_score | Task may be saturated (see med2) |
| Numerical errors | LinAlgError | Increase GP noise prior |
| Z-score normalization | Singular kernel | Use min-max to [0,1]^D (Hvarfner only) |
| LASS picks bad projection | Flat landscape | Verify criterion is log_ml, not posterior std |

---

## References

1. **TuRBO**: Eriksson et al. (2019) "Scalable Global Optimization via Local Bayesian Optimization" NeurIPS
2. **ArcCosine Kernel**: Cho & Saul (2009) "Kernel Methods for Deep Learning" NeurIPS
3. **Random Projection**: Johnson-Lindenstrauss Lemma, Achlioptas (2003)
4. **SELFIES VAE**: Krenn et al. (2020) "Self-referencing embedded strings (SELFIES)"
5. **LOLBO**: Gruver et al. (2023) "Effective Surrogate Models for Molecular Property Prediction"
6. **BAxUS**: Papenmeier et al. (2022) "Increasing the Scope as You Learn"
7. **SAASBO**: Eriksson & Jankowiak (2021) "High-Dimensional BO with Sparse Axis-Aligned Subspaces"
8. **InvBO**: Deshwal et al. (2024) "Bayesian Optimization in the Latent Space of Generative Models" NeurIPS
9. **SA-cREMBO**: Liu et al. (2025) "Subspace-Adaptive Random Embeddings for BO" (related to LASS)
10. **COWBOYS**: Daulton et al. (2025) "High-Dimensional BO via Coordinate-Wise Bayesian Optimization" ICML
11. **Understanding HDBO**: Hvarfner et al. (2025) "Understanding High-Dimensional Bayesian Optimization" ICML
