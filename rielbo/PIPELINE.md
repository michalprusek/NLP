# RieLBO Pipeline

Riemannian Latent Bayesian Optimization for molecular and prompt optimization.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Spherical Subspace BO Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. Encode                2. Project to Subspace      3. GP on S^15            │
│   ┌────────────────┐      ┌────────────────────┐      ┌──────────────────┐     │
│   │ SMILES → VAE   │  ─►  │ u = x/||x|| ∈ S^255│  ─►  │ v = A'u ∈ S^15   │     │
│   │ x ∈ R^256      │      │ direction sphere   │      │ ArcCosine kernel │     │
│   └────────────────┘      └────────────────────┘      └──────────────────┘     │
│                                                              │                  │
│   6. Evaluate              5. Decode               4. Acquire & Lift           │
│   ┌────────────────┐      ┌────────────────┐      ┌──────────────────┐        │
│   │ oracle(SMILES) │  ◄─  │ VAE → SMILES   │  ◄─  │ v* → u* = Av*    │        │
│   │ score ∈ [0,1]  │      │ x* = u* × norm │      │ Thompson/EI/UCB  │        │
│   └────────────────┘      └────────────────┘      └──────────────────┘        │
│                                    │                                           │
│                           7. Update GP with (v*, score)                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Recommended Pipeline (Subspace BO)

| Component | File | Description |
|-----------|------|-------------|
| **SphericalSubspaceBO** | `subspace_bo.py` | Main BO class: S^255 → S^15 projection |
| **run_guacamol_subspace** | `run_guacamol_subspace.py` | CLI entry point for GuacaMol |
| **GPDiagnostics** | `gp_diagnostics.py` | Debug GP overfitting, extrapolation |
| **SELFIESVAECodec** | `shared/guacamol/codec.py` | SELFIES VAE encoder/decoder (256D) |

### Baseline

| Component | File | Description |
|-----------|------|-------------|
| **TuRBOBaseline** | `turbo_baseline.py` | TuRBO in full 256D (comparison baseline) |

### Flow-Based (Legacy/Experimental)

| Component | File | Description |
|-----------|------|-------------|
| VelocityNetwork | `velocity_network.py` | DiT-style network with AdaLN |
| FlowMatchingModel | `flow_model.py` | ODE-based sampling |
| GuidedFlowSampler | `guided_flow.py` | GP-UCB guided ODE |
| NormPredictor | `norm_predictor.py` | Direction → magnitude MLP |
| run_guacamol_direct | `run_guacamol_direct.py` | Direct sphere BO (no flow) |

## File Structure

```
rielbo/
├── # Core BO (Recommended)
├── subspace_bo.py           # SphericalSubspaceBO - main algorithm
├── run_guacamol_subspace.py # CLI for Subspace BO
├── turbo_baseline.py        # TuRBO baseline for comparison
├── gp_diagnostics.py        # GP debugging tools
│
├── # Flow Model (Optional)
├── velocity_network.py      # DiT-style velocity network
├── flow_model.py            # FlowMatchingModel
├── train_flow.py            # Train on SONAR embeddings
├── train_flow_guacamol.py   # Train on SELFIES VAE
├── guided_flow.py           # GP-UCB guided sampling
├── norm_predictor.py        # Magnitude prediction
│
├── # Legacy/SONAR
├── run.py                   # SONAR/GSM8K optimization
├── run_bo_full.py           # Full GSM8K evaluation
├── gp_surrogate.py          # MSR/BAxUS GP surrogates
├── optimization_loop.py     # BO orchestrator
├── decoder.py               # SONAR decoder
├── latent_bo.py             # z-space BO (experimental)
├── run_guacamol_direct.py   # Direct sphere BO
│
├── # Utilities
├── data.py                  # Data loading
├── batch_selection.py       # Local penalization batching
├── validate.py              # Generation quality metrics
└── __init__.py
```

## Quick Start

### Subspace BO (Recommended)

```bash
# Run Subspace BO on GuacaMol
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --task-id adip --subspace-dim 16 --n-cold-start 100 --iterations 500

# With different kernel/acquisition
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --task-id adip --kernel matern --acqf ucb --ucb-beta 2.0 --iterations 500
```

### TuRBO Baseline

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.turbo_baseline \
    --task-id adip --n-cold-start 100 --iterations 500 --seed 42
```

## Algorithm Details

### Spherical Subspace BO

**Problem**: GP in 256D with ~100 points overfits (0.39 pts/dim).

**Solution**: Orthonormal projection S^(D-1) → S^(d-1):

```
Original:  u ∈ S^255 ⊂ R^256  (unit sphere, D=256)
Subspace:  v ∈ S^15  ⊂ R^16   (unit sphere, d=16)

Projection: A ∈ R^{256×16}, orthonormal (QR decomposition)
Project:    u → v = normalize(u @ A)
Lift:       v → u = normalize(v @ A.T)
Magnitude:  x = u × mean_norm
```

**Why 16D works**: 100 points / 16 dims = 6.25 pts/dim → GP generalizes.

### Kernels

| Kernel | Formula | Lengthscale | Best For |
|--------|---------|-------------|----------|
| **ArcCosine** | `1 - arccos(x·y)/π` | None | Spherical data |
| Matern-5/2 | Standard | Learned | General |

### Acquisition Functions

| Function | Description | Parameter |
|----------|-------------|-----------|
| **ts** | Thompson Sampling | - |
| ei | Expected Improvement | - |
| ucb | Upper Confidence Bound | `--ucb-beta` |

## GP Diagnostics

```python
from rielbo.gp_diagnostics import GPDiagnostics

diag = GPDiagnostics()
metrics = diag.analyze(gp, train_X, train_Y, candidates)
diag.log_summary(metrics)
```

**Metrics tracked**:
- `train_correlation`: >0.99 = overfit
- `lengthscale_min/max`: <0.01 = overfit
- `train_std_ratio`: <0.01 = collapsed uncertainty
- `candidate_in_hull_frac`: <0.5 = extrapolating

**Example output**:
```
[Iter 100] GP Diag: train_corr=0.85, ℓ=[0.2,1.5], noise=1e-2, std_ratio=0.15 ✓
[Iter 200] GP Diag: train_corr=0.99, ℓ=[0.01,0.5], noise=1e-4, std_ratio=0.01 ⚠️ OVERFIT
```

## Benchmark Results (2026-02-04)

### Subspace BO vs TuRBO

| Task | Cold Start | Subspace BO | TuRBO | Winner |
|------|------------|-------------|-------|--------|
| pdop | 0.4558 | **0.5676 ± 0.033** | 0.5587 ± 0.011 | Subspace +1.6% |
| adip | 0.4910 | **0.5447 ± 0.008** | 0.5066 | Subspace +7.5% |
| med2 | 0.1856 | 0.1856 | 0.1856 | Tie (task is hard) |

**Standard config**: 100 cold start, 500 iterations, seeds 42-46.

## Key Insights

1. **Subspace projection is critical** for high-D with limited data
2. **ArcCosine kernel** respects spherical geometry (no lengthscale to overfit)
3. **Mean norm reconstruction** is simpler and more robust than NormPredictor
4. **Trust region** (TuRBO-style) helps focus search
5. **Med2 is fundamentally hard** - score range [0.02, 0.19], cold start finds near-optimal

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| GP overfit | train_corr=1.0, no improvement | Reduce subspace_dim, add noise |
| Extrapolation | candidates far from train | Reduce trust_region |
| Duplicates | many repeated SMILES | Increase n_candidates |
| No improvement | flat best_score | Task may be hard (check med2) |
