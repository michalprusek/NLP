# RieLBO Pipeline: Spherical Subspace BO (V2 Geodesic)

## Overview

RieLBO performs sample-efficient Bayesian Optimization in the 256D latent space of a SELFIES VAE by projecting onto a 16D subspace sphere. The key idea: a GP on S^15 with ~100 training points is tractable, while a GP on S^255 is not.

**Best configuration**: V2 Geodesic preset — ArcCosine kernel + geodesic trust region + adaptive TR + random restart on collapse.

**Results (adip, 10 seeds, 500 iter)**: 0.5581 +/- 0.022

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

**Step 2b — Spherical whitening** (Householder rotation):
The VAE embeddings cluster in one cone of S^255. We rotate the mean direction to the north pole [1, 0, ..., 0] via Householder reflection:
```
mu = normalize(mean(U))
v = (mu - e1) / ||mu - e1||
H = I - 2*v*v^T                        # orthogonal, self-inverse
U_white = U @ H^T
```
This centers the data, maximizing information captured by projection.

**Step 2c — Random QR projection**:
```python
A_raw = torch.randn(256, 16)
A, _ = torch.linalg.qr(A_raw)          # [256, 16] orthonormal columns
V = F.normalize(U_white @ A, dim=-1)    # [N, 16] on S^15
```

The orthonormal matrix A preserves angular relationships. Points close on S^255 stay close on S^15.

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

**Diagnostics** (every 10 iterations): Train correlation, posterior std, noise level. See `rielbo/gp_diagnostics.py`.

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

**Adaptive trust region** (TuRBO-style):
- Start: tr_length = 0.4, giving radius = 0.2 rad (= 0.4 × 0.5 max_angle)
- On 3 consecutive improvements: tr_length grows by 1.5× (capped at 0.8, radius capped at 0.4 rad)
- On 10 consecutive failures: tr_length shrinks by 0.5×
- When tr_length < 0.02 (radius < 0.01 rad): restart with fresh random basis

### 5. Acquisition: Thompson Sampling

```
Input: GP model, v_candidates [2000, 16]
Output: v_opt [1, 16]
```

Draw one sample from the GP posterior over all 2000 candidates, pick the argmax. Thompson Sampling provides natural exploration-exploitation balance and is computationally cheap (single posterior sample vs. optimization loop for EI/UCB).

BoTorch's `MaxPosteriorSampling` handles this efficiently.

### 6. Lifting & Reconstruction: S^15 -> S^255 -> R^256

```
Input: v_opt [1, 16]
Output: x_opt [1, 256]
```

**Step 6a — Lift to S^255**:
```python
u_opt = F.normalize(v_opt @ A.T, dim=-1)  # [1, 256] on S^255
```

**Step 6b — Inverse whitening**:
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

If decode fails or produces a duplicate SMILES, count as failure (no training data added, TR shrinks).

### 8. Update & Iterate

- Add (x_opt, u_opt, score, smiles) to training data
- Refit GP every 10 iterations (not every iteration for speed)
- Update adaptive trust region counters
- If TR collapses: restart with fresh random QR basis (see below)

### 9. Subspace Restart (on TR Collapse)

When the adaptive trust region shrinks below `tr_min=0.02`, the current subspace is exhausted. The optimizer:

1. Increments restart counter
2. Generates a fresh random QR projection with a different seed
3. Resets trust region to `tr_init=0.4`
4. Re-projects all training data through the new basis
5. Refits the GP

Up to `max_restarts=5` allowed. After that, the TR just resets without changing the basis.

**Why random restart matters**: This is the single most impactful improvement from our experiments. Without restart, the optimizer gets stuck in a dead subspace. With restart, it can explore fundamentally different 16D slices of the 256D space.

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `rielbo/subspace_bo_v2.py` | Main optimizer (`SphericalSubspaceBOv2`) |
| `rielbo/kernels.py` | ArcCosine kernel (order 0, order 2), ProductSphereKernel, `create_kernel` factory |
| `rielbo/spherical_transforms.py` | SphericalWhitening, GeodesicTrustRegion |
| `rielbo/gp_diagnostics.py` | GP health monitoring |
| `rielbo/run_guacamol_subspace_v2.py` | CLI entry point |
| `shared/guacamol/codec.py` | SELFIES VAE (encode/decode) |
| `shared/guacamol/oracle.py` | GuacaMol scoring |
| `shared/guacamol/data.py` | Data loaders |

---

## Configuration Reference

### V2Config Presets

| Preset | Features | Use case |
|--------|----------|----------|
| `baseline` | None (matches v1) | Ablation baseline |
| `geodesic` | Geodesic TR + Adaptive TR | **Recommended** |
| `order2` | ArcCosine order 2 | Smoother GP |
| `whitening` | Spherical whitening | Data centering |
| `full` | All features | Experimental |

### CLI Usage

```bash
# Recommended: geodesic preset
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
    --preset geodesic --task-id adip --iterations 500 --seed 42

# Full benchmark
for seed in 42 43 44 45 46; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
        --preset geodesic --task-id adip --iterations 500 --seed $seed
done
```

---

## Mathematical Foundation

### Why Subspace Projection Works

The SELFIES VAE 256D latent space has intrinsic dimensionality ~16 (TwoNN=16.8, DANCo=11.3, FisherS=18.9). This means:
- The data lives on a ~16D manifold embedded in R^256
- A GP on S^255 with 100 points has terrible data density
- A random 16D projection captures most of the structure
- A GP on S^15 with 100 points has good data density

### Why ArcCosine Kernel

The ArcCosine kernel k(x,y) = 1 - arccos(x.y)/pi has:
- **No lengthscale parameters**: Prevents overfitting with limited data
- **Natural spherical geometry**: Monotone in geodesic distance
- **Scale invariance**: Only depends on angle between points
- **Positive definiteness**: Proven for S^d for all d

The order-0 kernel is rough (like Matern-1/2), but this is fine because:
- Chemical property landscapes have sharp transitions
- With 100-200 training points, we don't need smooth interpolation

### Why Geodesic Trust Region

Euclidean box constraints in R^d make no geometric sense on the sphere:
- A box around a point on S^15 wastes samples outside the sphere
- After projection back to S^15, density is non-uniform

Geodesic trust region samples within a spherical cap:
- All candidates are exactly on S^15
- Uniform density in geodesic distance
- Natural notion of "local" on a curved space

---

## Ablation Results (adip, 10 seeds, 500 iter)

| Variant | Score | Notes |
|---------|-------|-------|
| V2 Geodesic | 0.5581 +/- 0.022 | **Best** |
| V2 Baseline | 0.5361 +/- 0.025 | No geometric features |
| V2 Order-2 | 0.5329 +/- 0.028 | Smoother kernel hurts |
| V2 Whitening | 0.5372 +/- 0.021 | Marginal improvement |
| LOL-BO | 0.5228 +/- 0.019 | Baseline comparator |
| TuRBO (R^256) | 0.5060 +/- 0.007 | Full-dimensional BO |

### Negative Results (Pullback Metric Experiment, 2026-02-06)

We tested using the VAE decoder's Riemannian geometry (pullback metric G = J^T J) to guide subspace selection. Results:

| Config | Score (5 seeds) | Finding |
|--------|----------------|---------|
| Pure metric basis | 0.4910 (= cold start) | **Complete failure** |
| Score-weighted metric | 0.5422 +/- 0.014 | Worse than random |
| Hybrid 75% metric | 0.5336 +/- 0.006 | Low variance, low mean |
| Random QR (with restart fix) | 0.5531 +/- 0.018 | Best pullback framework |

**Conclusion**: Decoder sensitivity (which latent directions change the output most) does NOT correlate with objective sensitivity (which directions improve the score). The pullback metric is theoretically elegant but practically useless for optimization. Random QR projection works best because it provides unbiased exploration of all latent directions.
