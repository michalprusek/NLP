# RieLBO Pipeline

Flow matching + Bayesian optimization for prompt optimization in SONAR embedding space.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RieLBO BO Loop                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. GP-UCB Optimization      2. Guided Flow           3. Decode & Eval    │
│   ┌─────────────────────┐    ┌─────────────────────┐   ┌────────────────┐  │
│   │ z* = argmax UCB(z)  │ ─► │ flow(z*) → u        │ ─►│ x = r̂·u       │  │
│   │ in latent space     │    │ GP-guided ODE       │   │ decode → eval  │  │
│   └─────────────────────┘    └─────────────────────┘   └────────────────┘  │
│            │                          │                        │           │
│            └──────────────────────────┴────────────────────────┘           │
│                                    │                                       │
│                           4. Update GP                                     │
│                           ┌─────────────────┐                              │
│                           │ gp.update(z, y) │                              │
│                           └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Algorithm

Each iteration:
1. **GP-UCB Optimization**: Find `z* = argmax [μ(z) + α·σ(z)]` in latent z-space via gradient ascent
2. **Guided Flow Sampling**: GP-guided ODE transforms `z* → x` (or `z* → u` for spherical)
3. **Magnitude Recovery** (spherical only): `x = NormPredictor(u) · u`
4. **Decode & Evaluate**: Decoder → text/molecule → oracle evaluation
5. **Update GP**: Add (z*, score) observation, refit GP

## File Structure

```
rielbo/
├── __init__.py              # Package exports
├── run.py                   # CLI entry point (SONAR/GSM8K)
├── run_guacamol.py          # GuacaMol molecular optimization (flow-based)
├── run_guacamol_direct.py   # GuacaMol with Direct Sphere BO (no flow)
├── run_guacamol_subspace.py # GuacaMol with Spherical Subspace BO
│
├── # Core BO Pipeline
├── optimization_loop.py     # BOOptimizationLoop - main orchestrator
├── gp_surrogate.py          # GP surrogates (MSR, BAxUS, Heteroscedastic)
├── subspace_bo.py           # Spherical Subspace BO (S^255 → S^15)
├── guided_flow.py           # GuidedFlowSampler with GP-guided ODE
│
├── # Flow Model
├── velocity_network.py      # DiT-style velocity network with AdaLN
├── flow_model.py            # FlowMatchingModel (ODE sampling)
├── train_flow.py            # Training script with OT-CFM (SONAR)
├── train_flow_guacamol.py   # Training script (GuacaMol/SELFIES)
│
├── # Magnitude Prediction (for spherical flows)
├── norm_predictor.py        # NormPredictor: direction → magnitude MLP
│
├── # Utilities
├── decoder.py               # SonarDecoder (embedding → text)
├── data.py                  # Dataset and dataloader
├── batch_selection.py       # Batch candidate selection (optional)
├── validate.py              # Model validation utilities
└── test_flow_quality.py     # Quality metrics for flow models
```

## Key Classes

### `SonarGPSurrogate` / `BAxUSGPSurrogate`
GP surrogate with MSR initialization for 1024D optimization.

```python
gp = SonarGPSurrogate(D=1024, device='cuda')
gp.fit(X, Y)                           # Fit GP
mean, std = gp.predict(X_new)          # Predict
z_opt, ucb = gp.optimize_ucb(alpha=1.96)  # Find optimal point
```

### `GuidedFlowSampler`
Generates samples with optional UCB guidance.

```python
sampler = GuidedFlowSampler(flow_model, gp)

# Single optimal candidate (main method)
z, info = sampler.sample_optimal(device='cuda')

# Guided sampling from noise (exploration)
z = sampler.sample(n_samples=10)
```

### `BOOptimizationLoop`
Main optimization orchestrator.

```python
loop = BOOptimizationLoop(flow_model, gp, sampler, decoder, evaluator, llm)
loop.warm_start('embeddings.pt')  # Initialize from pre-evaluated data

for _ in range(100):
    result = loop.step()
    print(f"Iter {result['iteration']}: {result['score']:.3f}")
```

## Usage

### Training Flow Model

```bash
# SONAR embeddings (prompt optimization)
uv run python -m rielbo.train_flow \
  --data datasets/sonar_embeddings.pt \
  --epochs 100 --batch-size 1024

# GuacaMol/SELFIES (molecular optimization)
uv run python -m rielbo.train_flow_guacamol \
  --spherical --zinc --n-samples 250000 --epochs 500
```

### Running Optimization

```bash
# SONAR/GSM8K prompt optimization
uv run python -m rielbo.run \
  --flow-checkpoint path/to/flow.pt \
  --warm-start datasets/evaluated_instructions.pt \
  --iterations 100

# GuacaMol molecular optimization
uv run python -m rielbo.run_guacamol \
  --flow-checkpoint path/to/spherical_flow.pt \
  --norm-predictor path/to/norm_predictor.pt \
  --task-id adip --iterations 500
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ucb-alpha` | UCB exploration weight | 1.96 |
| `--n-restarts` | GP optimization restarts | 5 |
| `--n-opt-steps` | Gradient steps per restart | 100 |
| `--num-steps` | ODE integration steps | 50 |
| `--guidance-strength` | λ for guided sampling | 1.0 |

## Stereographic Projection (Recommended)

Deterministic bijection R^D ↔ S^D that encodes magnitude in geometry.
**Replaces NormPredictor with exact magnitude recovery.**

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Stereographic Projection Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Training:                                                                 │
│   x [N, D] → lift → u [N, D+1] → train spherical flow on S^D               │
│                                                                             │
│   Inference:                                                                │
│   z* → flow → u [D+1] → project → x [D] with exact ||x||                   │
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐    │
│   │ z* = argmax UCB │ ─► │ flow(z*) → u    │ ─► │ x = project(u)      │    │
│   │ z on S^D        │    │ u on S^D        │    │ exact magnitude     │    │
│   └─────────────────┘    └─────────────────┘    └─────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Foundation

**Lift (R^D → S^D)**: For x ∈ R^D, with R = mean(||x||):
```
x̂ = x/R
u_i = 2x̂_i / (1 + ||x̂||²)     for i=1..D
u_{D+1} = (||x̂||² - 1) / (||x̂||² + 1)
```

**Project (S^D → R^D)**: For u ∈ S^D:
```
x̂ = u_{1:D} / (1 - u_{D+1})
x = x̂ * R
```

### Advantages over NormPredictor

| Aspect | Stereographic | NormPredictor |
|--------|---------------|---------------|
| Parameters | 0 (deterministic) | ~17K learned |
| Magnitude recovery | Exact (bijection) | Approximate |
| Requires training | No | Yes (100+ epochs) |
| Geometry | Unified S^D | Product S^{D-1} × R^+ |

### Usage

```bash
# Train with stereographic projection
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.train_flow_guacamol \
    --stereographic --zinc --n-samples 50000 --epochs 500 \
    --output-dir rielbo/checkpoints/guacamol_stereo

# Run BO (auto-detects stereographic from checkpoint)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol \
    --flow-checkpoint rielbo/checkpoints/guacamol_stereo/best.pt \
    --task-id pdop --iterations 500
```

Note: `--norm-predictor` is NOT needed for stereographic checkpoints.

---

## Legacy: Magnitude Prediction (NormPredictor)

For non-stereographic spherical flows, `NormPredictor` learns to recover magnitude.

### Pipeline with Magnitude Prediction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Spherical Flow + Magnitude Prediction Pipeline                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. BO in z-space        2. Guided Flow           3. Magnitude + Decode   │
│   ┌─────────────────┐    ┌─────────────────┐      ┌─────────────────────┐  │
│   │ z* = argmax UCB │ ─► │ flow(z*) → u    │  ─►  │ r̂ = NormPred(u)    │  │
│   │ z ~ N(0,I)      │    │ GP-guided ODE   │      │ x = r̂·u → decode   │  │
│   └─────────────────┘    └─────────────────┘      └─────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

Embedding decomposition: `x = r · u` where:
- `u = x/||x|| ∈ S^{d-1}` — direction on unit sphere (semantics)
- `r = ||x|| ∈ R^+` — magnitude (scale)

Spherical flow outputs direction `u`. `NormPredictor` learns `f: S^{d-1} → R^+` to recover magnitude.

### NormPredictor Architecture

```python
NormPredictor(
    Linear(D, 128) → ReLU →
    Linear(128, 64) → ReLU →
    Linear(64, 1) → Softplus  # Ensures positive output
)
```

### Training

```bash
# Train NormPredictor on SELFIES VAE embeddings
uv run python -m rielbo.norm_predictor \
    --n-samples 10000 \
    --epochs 100 \
    --output rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt

# With ZINC dataset (larger, recommended)
uv run python -m rielbo.norm_predictor \
    --zinc --zinc-path datasets/zinc/zinc_all.txt \
    --n-samples 250000 \
    --epochs 200
```

### Usage in BO Pipeline

```python
from rielbo.norm_predictor import NormPredictor

# After guided flow sampling
z_opt = gp.optimize_ucb()              # Optimal point in latent z-space
u = guided_flow.sample(z_opt)          # Flow: z → u (direction on sphere)
r = norm_predictor(u)                  # Predict magnitude
x = u * r                              # Reconstruct full embedding
text = decoder(x)                      # Decode to text/molecule
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MAE | Mean absolute error on norm | < 0.5 |
| MAPE | Mean absolute percentage error | < 5% |

---

## Spherical Subspace BO (Recommended for High-D)

**Problem**: GP in 256D with ~100 training points fails:
- Top scorers have cosine similarity ~0.11 (essentially random in high-D)
- GP overfits training data (correlation=1.0) but predicts mean for new points
- 256D is severely underdetermined with 100 observations

**Solution**: Orthonormal random subspace projection S^(D-1) → S^(d-1)

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Spherical Subspace BO Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Project data to subspace:                                                 │
│   u ∈ S^255 → v = normalize(u @ A) ∈ S^15  (orthonormal A ∈ R^{256×16})    │
│                                                                             │
│   1. Train GP on S^15      2. Optimize UCB        3. Lift & Decode         │
│   ┌──────────────────┐    ┌──────────────────┐   ┌──────────────────────┐  │
│   │ GP fits well in  │ ─► │ v* = argmax UCB  │ ─►│ u* = normalize(v*@Aᵀ)│  │
│   │ 16D (6.25 pts/d) │    │ on S^15          │   │ x = NormPred(u*)·u*  │  │
│   └──────────────────┘    └──────────────────┘   └──────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Foundation

```
Original sphere: u ∈ S^(D-1) ⊂ R^D (D=256, unit sphere)
Subspace sphere: v ∈ S^(d-1) ⊂ R^d (d=16, unit sphere)
Projection matrix: A ∈ R^(D×d), orthonormal columns (via QR decomposition)

Project: u → v = normalize(u @ A)    [S^255 → S^15]
Lift:    v → u = normalize(v @ A.T)  [S^15 → S^255]

GP: operates on S^15 with ArcCosine kernel (geodesic-aware)
Acquisition: optimize UCB on S^15 (bounded, spherical)
```

### Why This Works

| Aspect | S^255 (256D) | S^15 (16D) |
|--------|--------------|------------|
| Points per dimension | 0.39 | 6.25 |
| GP train correlation | 1.0 (overfit) | ~0.7-0.9 (healthy) |
| Prediction at new points | Mean only | Varies meaningfully |
| ArcCosine kernel | Valid | **Valid** (both spheres!) |

**Key insight**: Orthonormal projection approximately preserves angles (Johnson-Lindenstrauss-like property), so ArcCosine kernel geodesic distance is preserved.

### Usage

```bash
# Standard subspace BO (S^255 → S^15)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --norm-predictor rielbo/checkpoints/guacamol_stereo/norm_predictor.pt \
    --subspace-dim 16 --task-id pdop --n-cold-start 100 --iterations 500

# Try different subspace dimensions
for dim in 8 16 32; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
        --subspace-dim $dim --task-id pdop --iterations 200 &
done

# Adaptive subspace (BAxUS-style: starts at d=4, grows when stuck)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --adaptive --initial-dim 4 --max-dim 64 \
    --no-improve-threshold 50 \
    --task-id pdop --iterations 500
```

### Key Classes

```python
from rielbo.subspace_bo import SphericalSubspaceBO, AdaptiveSphericalSubspaceBO

# Fixed subspace dimension
bo = SphericalSubspaceBO(
    norm_predictor_path="path/to/norm_predictor.pt",
    codec=codec,
    oracle=oracle,
    input_dim=256,      # S^255
    subspace_dim=16,    # → S^15
    ucb_beta=2.0,
)

# Adaptive (grows from 4D when stuck)
bo = AdaptiveSphericalSubspaceBO(
    ...,
    initial_dim=4,
    max_dim=64,
    no_improve_threshold=50,  # Expand after 50 iters without improvement
)
```

---

## Design Principles

1. **BO in latent space**: GP-UCB optimization in z-space where geometry is Gaussian
2. **Guided flow for manifold projection**: GP-guided ODE ensures samples stay on data manifold
3. **Decoupled geometry**: For spherical flows, direction (semantics) via flow, magnitude via predictor
4. **Subspace projection for high-D**: When data is sparse, project to lower-D sphere where GP has enough coverage
5. **Simple over complex**: No hybrid strategies, no batch selection complexity
6. **Minimal dependencies**: Core pipeline uses only GP, flow model, decoder, evaluator
