# EcoFlow Pipeline Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Bayesian Optimization Loop](#bayesian-optimization-loop)
9. [Key Hyperparameters](#key-hyperparameters)
10. [Quality Metrics](#quality-metrics)
11. [Troubleshooting](#troubleshooting)

---

## Overview

EcoFlow is a **flow matching-based Bayesian optimization framework** for discovering optimal prompts in continuous embedding space. It combines:

1. **Flow Matching**: Learns the distribution of SONAR text embeddings (1024D) using Optimal Transport Conditional Flow Matching (OT-CFM)
2. **UCB-Guided Sampling**: Steers the generative ODE toward high-scoring regions using GP surrogate gradients
3. **SONAR Decoder**: Converts optimized embeddings back to natural language prompts
4. **LLM Evaluation**: Scores decoded prompts on GSM8K math reasoning tasks

### Key Innovation

The **CFG-Zero* schedule** applies zero guidance for the first 4% of ODE steps, preventing early trajectory corruption from inaccurate velocity estimates at t≈0.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EcoFlow Pipeline                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                      TRAINING PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Text Corpus ──► SONAR Encoder ──► Embeddings [N, 1024]                    │
│                                         │                                   │
│                                         ▼                                   │
│                              ┌──────────────────────┐                       │
│                              │ Normalization        │                       │
│                              │ (mean=0, std=1)      │                       │
│                              └──────────────────────┘                       │
│                                         │                                   │
│                                         ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Flow Matching Training                            │   │
│  │                                                                      │   │
│  │   z₀ ~ N(0,I) ─────────────────────► z₁ = data                      │   │
│  │        │                                  │                          │   │
│  │        │    OT-CFM Interpolation          │                          │   │
│  │        │    zₜ = (1-t)z₀ + tz₁            │                          │   │
│  │        │    uₜ = z₁ - z₀                  │                          │   │
│  │        ▼                                  │                          │   │
│  │   ┌─────────────────────┐                 │                          │   │
│  │   │  VelocityNetwork    │◄────────────────┘                          │   │
│  │   │  (DiT + AdaLN-Zero) │                                            │   │
│  │   │  vθ(zₜ, t)          │                                            │   │
│  │   └─────────────────────┘                                            │   │
│  │            │                                                         │   │
│  │            ▼                                                         │   │
│  │   L = MSE(vθ(zₜ, t), uₜ)  ◄─── Minimize via AdamW/LAMB              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


                      INFERENCE PHASE (BO Loop)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    GuidedFlowSampler                                   │ │
│  │                                                                        │ │
│  │   z₀ ~ N(0,I)                                                         │ │
│  │       │                                                                │ │
│  │       ▼                                                                │ │
│  │   ┌──────────────────────────────────────────────────────────────┐    │ │
│  │   │  Guided ODE (Heun Method, 50 steps)                          │    │ │
│  │   │                                                               │    │ │
│  │   │  for i in range(num_steps):                                  │    │ │
│  │   │      λₜ = CFG-Zero*(step=i, λ_max=1.0, zero_frac=0.04)      │    │ │
│  │   │                                                               │    │ │
│  │   │      ┌─────────────────────────────────────────────────┐     │    │ │
│  │   │      │ if i < 2:  # First 4% = zero guidance           │     │    │ │
│  │   │      │     λₜ = 0                                       │     │    │ │
│  │   │      │ else:                                            │     │    │ │
│  │   │      │     λₜ = 1.0                                     │     │    │ │
│  │   │      └─────────────────────────────────────────────────┘     │    │ │
│  │   │                                                               │    │ │
│  │   │      v = vθ(z, t) + λₜ · ∇UCB(z)                            │    │ │
│  │   │                        ▲                                      │    │ │
│  │   │                        │                                      │    │ │
│  │   │                   ┌────┴─────────────┐                       │    │ │
│  │   │                   │  GP Surrogate    │                       │    │ │
│  │   │                   │  μ(z) + α·σ(z)   │                       │    │ │
│  │   │                   └──────────────────┘                       │    │ │
│  │   │                                                               │    │ │
│  │   │      z = z + v·dt  (Heun predictor-corrector)               │    │ │
│  │   │                                                               │    │ │
│  │   └──────────────────────────────────────────────────────────────┘    │ │
│  │       │                                                                │ │
│  │       ▼                                                                │ │
│  │   z₁ (SONAR embedding, 1024D)                                         │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌────────────────────┐      ┌─────────────────┐     ┌──────────────────┐  │
│  │ L2-r Filter        │ ──►  │ SONAR Decoder   │ ──► │ Prompt Text      │  │
│  │ (threshold=0.5)    │      │ (beam=5, ngram) │     │ "Solve step..."  │  │
│  └────────────────────┘      └─────────────────┘     └──────────────────┘  │
│                                                              │              │
│                                                              ▼              │
│                                      ┌───────────────────────────────────┐ │
│                                      │ LLM Evaluation (Qwen-7B)         │ │
│                                      │ Q_end format: Q: {q}\n{instr}\nA: │ │
│                                      │ Accuracy on GSM8K (150 questions) │ │
│                                      └───────────────────────────────────┘ │
│                                                              │              │
│                                                              ▼              │
│                                      ┌───────────────────────────────────┐ │
│                                      │ GP Update                         │ │
│                                      │ X_train ← cat(X_train, z₁)        │ │
│                                      │ Y_train ← cat(Y_train, accuracy)  │ │
│                                      │ refit_gp(X_train, Y_train)        │ │
│                                      └───────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundations

### Flow Matching (Lipman et al., 2022)

Flow matching learns a time-dependent velocity field `vθ(z, t)` that transports samples from noise `z₀ ~ N(0,I)` at `t=0` to data `z₁ ~ p_data` at `t=1`:

```
dz/dt = vθ(z, t)
```

**Optimal Transport CFM (OT-CFM)**: Uses optimal transport couplings between noise and data for straight-line interpolation paths:

```
zₜ = (1-t)·z₀ + t·z₁
uₜ = z₁ - z₀  (target velocity)

L = E_{t,z₀,z₁}[||vθ(zₜ, t) - uₜ||²]
```

### UCB Acquisition (Srinivas et al., 2010)

Upper Confidence Bound for maximization problems:

```
UCB(z) = μ(z) + α·σ(z)

where:
- μ(z) = GP posterior mean
- σ(z) = GP posterior standard deviation
- α = exploration weight (default 1.96 for 95% CI)
```

### Guided ODE

The guided ODE combines the flow velocity with the UCB gradient:

```
dz/dt = vθ(z, t) + λ(t)·∇UCB(z)

where λ(t) follows CFG-Zero* schedule:
- λ(t) = 0      for t < 0.04 (first 4%)
- λ(t) = λ_max  for t ≥ 0.04
```

### MSR Initialization (Hvarfner et al., 2024)

For high-dimensional GP (D=1024), lengthscales are initialized using the MSR method:

```
l_init = √D / 10 = √1024 / 10 ≈ 3.2

LogNormal prior: loc = √2 + 0.5·log(D), scale = √3
```

### Heteroscedastic Noise Model

For binomial accuracy observations (k correct out of n questions):

```
Var(p̂) = p(1-p) / n

where:
- p = true accuracy
- n = number of evaluation questions (150)
```

---

## Core Components

### 1. VelocityNetwork (`velocity_network.py`)

**Purpose**: Predicts velocity `v(z, t)` for flow ODE integration.

**Architecture**:
- Input projection: `[B, 1024] → [B, 512]`
- Time embedding: Sinusoidal → MLP → [B, 512]
- 6× AdaLN-Zero transformer blocks with 8 heads
- Output projection: `[B, 512] → [B, 1024]`

**AdaLN-Zero Block**:
```python
# Modulation parameters from time embedding
shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp =
    adaLN_modulation(time_embed).chunk(6)

# Attention with modulation
x_mod = LayerNorm(x) * (1 + scale_attn) + shift_attn
attn_out = SelfAttention(x_mod)
x = x + gate_attn * attn_out

# MLP with modulation
x_mod = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
mlp_out = MLP(x_mod)
x = x + gate_mlp * mlp_out
```

**Key Design Choices**:
- Zero-initialization of output projection and modulation layers for stable training
- Layer norms without learnable parameters (AdaLN provides modulation)
- GELU activation in MLP

### 2. FlowMatchingModel (`flow_model.py`)

**Purpose**: Wrapper for ODE-based sampling with denormalization support.

**Key Methods**:
```python
def sample(n_samples, device, method="heun", num_steps=50, denormalize=True):
    """Generate samples via ODE integration from noise to data."""
    z = randn(n_samples, 1024)  # Start at t=0

    for i in range(num_steps):
        if method == "heun":
            # Predictor-corrector (2nd order)
            v1 = velocity_net(z, t)
            z_pred = z + v1 * dt
            v2 = velocity_net(z_pred, t + dt)
            z = z + 0.5 * (v1 + v2) * dt
        else:
            # Euler (1st order)
            z = z + velocity_net(z, t) * dt

    if denormalize and norm_stats:
        z = z * std + mean
    return z
```

### 3. GuidedFlowSampler (`guided_flow.py`)

**Purpose**: UCB-guided ODE sampling with CFG-Zero* schedule.

**Key Algorithm**:
```python
def sample(n_samples, num_steps=50, method="heun"):
    z = randn(n_samples, 1024)

    for i in range(num_steps):
        # CFG-Zero* schedule
        lambda_t = 0 if i < int(0.04 * num_steps) else guidance_strength

        # Base velocity
        v = flow_model.ode_func(t, z)

        # Add UCB gradient if in guidance phase
        if lambda_t > 0 and gp.model is not None:
            z_sonar = denormalize(z)
            grad_ucb = gp.ucb_gradient(z_sonar, alpha=alpha)
            grad_ucb = grad_ucb / (norm_std + 1e-8)  # Scale to flow space
            grad_ucb = clip_grad_norm(grad_ucb, max_norm=10.0)
            v = v + lambda_t * grad_ucb

        # Integration step (Heun or Euler)
        z = integrate_step(z, v, dt, method)

    return denormalize(z)
```

### 4. GP Surrogates (`gp_surrogate.py`)

**Three GP variants**:

| Surrogate | Use Case | Key Feature |
|-----------|----------|-------------|
| `SonarGPSurrogate` | Standard 1024D GP | MSR initialization |
| `BAxUSGPSurrogate` | Random subspace projection | Projects 1024D → 128D |
| `HeteroscedasticSonarGP` | Accuracy optimization | Binomial noise model |

**MSR Initialization**:
```python
lengthscale_init = sqrt(D) / 10  # ~3.2 for D=1024
prior = LogNormalPrior(
    loc=sqrt(2) + 0.5 * log(D),  # ~4.0
    scale=sqrt(3)
)
```

### 5. SonarDecoder (`decoder.py`)

**Purpose**: Converts 1024D SONAR embeddings back to text.

**Key Features**:
- Uses `text_sonar_basic_decoder` model
- N-gram repeat blocking (default: 3-grams) to prevent degenerate outputs
- Beam search with beam_size=5

```python
decoder = SonarDecoder(device="cuda:0", ngram_block_size=3)
texts = decoder.decode(embeddings, max_seq_len=256, beam_size=5)
```

### 6. Batch Selection (`batch_selection.py`)

**Purpose**: Select diverse batch candidates for parallel BO.

**Local Penalization Algorithm** (Gonzalez et al., 2016):
```python
def select_batch_candidates(gp, candidates, batch_size, method="local_penalization"):
    # Estimate Lipschitz constant L from GP gradients
    L = estimate_lipschitz_constant(gp.model, bounds)

    selected = []
    ucb = gp.predict(candidates)[0] + alpha * gp.predict(candidates)[1]

    for i in range(batch_size):
        # Select best among remaining
        best_idx = ucb.argmax()
        selected.append(best_idx)

        # Penalize UCB near selected point
        x_sel = candidates[best_idx]
        radius = abs(mu[best_idx] - min_Y) / L
        dists = norm(candidates - x_sel, dim=-1)

        # Hammer function penalty
        penalty = 1 - Phi(-(dists - radius) / s)
        ucb = ucb * penalty

    return candidates[selected]
```

### 7. Flow Density (`flow_density.py`)

**Purpose**: Filter off-manifold samples using flow log-density.

**Hutchinson Trace Estimator**:
```python
def hutchinson_trace_estimate(velocity_net, z, t, n_hutchinson=1):
    """Estimate tr(∂v/∂z) using random probes."""
    traces = []
    for _ in range(n_hutchinson):
        epsilon = randn_like(z)  # Gaussian probe
        z.requires_grad_(True)
        v = velocity_net(z, t)

        # tr(A) = E[ε^T A ε]
        v_dot_eps = (v * epsilon).sum()
        grad = autograd.grad(v_dot_eps, z)[0]
        trace = (epsilon * grad).sum(dim=-1)
        traces.append(trace)

    return stack(traces).mean(dim=0)
```

**Density Computation**:
```python
def compute_flow_log_density(flow_model, z_final, num_steps=50):
    """Compute log p(z) by backward ODE integration."""
    z = z_final.clone()
    accumulated_trace = 0

    for i in range(num_steps):
        t = 1.0 - i * dt  # Backward from t=1 to t=0
        trace = hutchinson_trace_estimate(velocity_net, z, t)
        accumulated_trace += trace * dt
        z = z - velocity_net(z, t) * dt

    # Prior at t=0 is standard Gaussian
    log_p_z0 = -0.5 * z.pow(2).sum(dim=-1) - 0.5 * D * log(2*pi)

    # Change of variables
    log_p = log_p_z0 - accumulated_trace
    return log_p
```

### 8. BOOptimizationLoop (`optimization_loop.py`)

**Purpose**: Orchestrate the full BO pipeline with checkpointing.

**Main Loop**:
```python
loop = BOOptimizationLoop(
    flow_model=flow_model,
    gp=gp_surrogate,
    sampler=guided_sampler,
    decoder=sonar_decoder,
    evaluator=gsm8k_evaluator,
    llm_client=llm_client,
    n_initial=10,
    batch_size=8,
    eval_subset_size=150,
)

# Initialize with random samples
loop.initialize()

# Optimization loop
for i in range(100):
    result = loop.batch_step(
        batch_size=8,
        n_candidates=64,
        use_local_penalization=True,
    )
    print(f"Iter {i}: best={result['best_so_far']:.3f}")

    if i % 10 == 0:
        loop.save_checkpoint(f"checkpoint_{i}.pt")
```

---

## Data Flow

### Training Data Preparation

```
1. Text Corpus (1.5M sentences)
       │
       ▼
2. SONAR Encoder ("text_sonar_basic_encoder")
       │
       ▼
3. Raw Embeddings: [1.5M, 1024]
   - Mean: ~0.0
   - Std: ~0.01
   - L2 norm: ~0.32
       │
       ▼
4. Normalization (per-dimension)
   - x_norm = (x - mean) / std
   - Save norm_stats for denormalization
       │
       ▼
5. Normalized Embeddings: [1.5M, 1024]
   - Mean: 0.0
   - Std: 1.0
       │
       ▼
6. DataLoader (batch_size=1024, pin_memory=True)
```

### Inference Data Flow

```
1. Sample noise z₀ ~ N(0, I)           [B, 1024]
       │
       ▼
2. Guided ODE (50 steps, Heun)
   - CFG-Zero* schedule
   - UCB gradient guidance
       │
       ▼
3. Normalized samples z₁               [B, 1024]
       │
       ▼
4. Denormalization
   - z_sonar = z₁ * std + mean
       │
       ▼
5. L2-r Filtering (optional)
   - Compute round-trip fidelity
   - Reject samples with L2-r > 0.5
       │
       ▼
6. Batch Selection (Local Penalization)
   - Select diverse subset by UCB
       │
       ▼
7. SONAR Decoder
   - Beam search, ngram blocking
       │
       ▼
8. Text Prompts                         [B]
       │
       ▼
9. LLM Evaluation (Qwen-7B)
   - Q_end format on GSM8K
       │
       ▼
10. Accuracy scores                     [B]
        │
        ▼
11. GP Update
    - Concatenate (z_sonar, accuracy)
    - Refit GP
```

---

## Training Pipeline

### train_flow.py Arguments

```bash
uv run python -m src.ecoflow.train_flow \
  --data-path datasets/sonar_embeddings.pt \
  --output-dir results/flow_checkpoints \
  --batch-size 1024 \
  --epochs 100 \
  --lr 1e-4 \
  --hidden-dim 512 \
  --num-layers 6 \
  --num-heads 8 \
  --optimizer adamw \
  --use-ot \
  --warmup-steps 1000 \
  --ema-decay 0.9999
```

### Training Loop Details

1. **Data Loading**: Normalized embeddings with `pin_memory=True`, `num_workers=8`
2. **OT-CFM Sampling**: Sample `(t, zₜ, uₜ)` using optimal transport couplings
3. **Forward Pass**: `vθ = VelocityNetwork(zₜ, t)`
4. **Loss**: `L = MSE(vθ, uₜ)`
5. **Optimization**: AdamW with gradient clipping (max_norm=1.0)
6. **EMA**: Exponential moving average of weights (decay=0.9999)
7. **Checkpointing**: Save model, EMA, optimizer, scheduler, norm_stats

### Expected Training Metrics

| Epoch | Loss | Notes |
|-------|------|-------|
| 1 | ~0.5 | Initial random |
| 10 | ~0.1 | Learning basic structure |
| 30 | ~0.05 | Convergence |
| 50+ | ~0.03 | Diminishing returns |

---

## Inference Pipeline

### validate.py Usage

```bash
uv run python -m src.ecoflow.validate \
  --checkpoint results/flow_checkpoints/best_flow.pt \
  --n-samples 100 \
  --num-steps 50 \
  --method heun \
  --decode \
  --use-ema
```

### Expected Validation Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Sample mean | ~0.0 | Centered distribution |
| Sample std | ~0.01 | SONAR scale |
| L2 norm mean | ~0.32 | SONAR manifold |
| Cosine sim mean | <0.5 | Diverse samples |
| Coherent ratio | >80% | Valid decoded text |

---

## Bayesian Optimization Loop

### run_bo_optimization.py Arguments

```bash
uv run python scripts/run_bo_optimization.py \
  --flow-checkpoint results/flow_checkpoints/best_flow.pt \
  --iterations 100 \
  --batch-size 8 \
  --n-candidates 64 \
  --eval-subset-size 150 \
  --guidance-strength 1.0 \
  --alpha 1.0 \
  --l2r-filter \
  --l2r-threshold 0.5 \
  --checkpoint-dir results/bo_run \
  --checkpoint-freq 10
```

### BO Loop Phases

1. **Initialization** (warm-start or random)
   - Generate initial samples from flow (no guidance)
   - Decode and evaluate on GSM8K
   - Fit GP on initial observations

2. **Batch Iteration**
   - Generate `n_candidates` guided samples
   - L2-r filter: reject off-manifold samples
   - Local Penalization: select diverse batch
   - Decode batch to text
   - Evaluate on LLM
   - Update GP with new observations

3. **Checkpointing**
   - Save training data (X, Y)
   - Save best prompt and score
   - Save metrics history

---

## Key Hyperparameters

### Flow Model

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `hidden_dim` | 512 | 256-1024 | Larger = more capacity |
| `num_layers` | 6 | 4-12 | Deeper = better quality |
| `num_heads` | 8 | 4-16 | More heads = more attention patterns |
| `batch_size` | 1024 | 256-2048 | Larger = better OT coupling |
| `lr` | 1e-4 | 1e-5 to 1e-3 | AdamW learning rate |
| `ema_decay` | 0.9999 | 0.999-0.9999 | EMA smoothing |

### Guided Sampling

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `num_steps` | 50 | 25-100 | More = better quality, slower |
| `method` | heun | heun/euler | Heun is 2nd order |
| `guidance_strength` | 1.0 | 0.5-2.0 | λ in guided ODE |
| `alpha` | 1.0 | 0.5-2.0 | UCB exploration weight |
| `zero_init_fraction` | 0.04 | 0.02-0.1 | CFG-Zero* fraction |

### BO Loop

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `n_initial` | 10 | 5-50 | Initial random samples |
| `batch_size` | 8 | 1-16 | Parallel evaluations |
| `n_candidates` | 64 | 32-128 | Candidates before selection |
| `eval_subset_size` | 150 | 50-500 | GSM8K questions per eval |
| `l2r_threshold` | 0.5 | 0.3-0.7 | On-manifold filter |

---

## Quality Metrics

### Training Quality

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Final loss | <0.05 | 0.05-0.1 | >0.1 |
| Gradient norm | <1.0 | 1.0-5.0 | >5.0 (exploding) |
| L2 norm mean | 0.30-0.35 | 0.25-0.40 | <0.2 or >0.5 |

### Generation Quality

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Cosine sim | <0.3 | 0.3-0.5 | >0.5 (mode collapse) |
| L2-r mean | <0.4 | 0.4-0.5 | >0.6 (off-manifold) |
| Coherent ratio | >90% | 70-90% | <70% |

### BO Quality

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Best accuracy | >85% | 80-85% | <80% |
| Batch diversity | >0.4 | 0.2-0.4 | <0.2 (clustering) |
| GP variance | <0.05 | 0.05-0.1 | >0.1 (poor model) |

---

## Troubleshooting

### Issue: Off-manifold samples (high L2-r)

**Symptoms**: L2-r mean > 0.5, decoded text is nonsense

**Causes**:
1. Flow not converged → Train longer
2. Guidance too strong → Reduce `guidance_strength` to 0.5
3. CFG-Zero* fraction too small → Increase to 0.1

**Solutions**:
```bash
# Reduce guidance
--guidance-strength 0.5

# Increase zero-init phase
--zero-init-fraction 0.1

# Use density filtering
--use-density-filter --density-percentile 25
```

### Issue: Low diversity (high cosine similarity)

**Symptoms**: All samples similar, batch diversity < 0.2

**Causes**:
1. Guidance too strong → Reduce `guidance_strength`
2. GP overfitting → More exploration (`alpha=2.0`)
3. Local Penalization disabled → Enable it

**Solutions**:
```bash
# More exploration
--alpha 2.0

# Force diversity
--use-local-penalization
```

### Issue: Poor decoding (repetitive text)

**Symptoms**: "word word word word..."

**Causes**:
1. N-gram blocking disabled → Enable it
2. Beam size too small → Increase
3. Off-manifold embeddings → Filter by L2-r

**Solutions**:
```python
decoder = SonarDecoder(ngram_block_size=3)
texts = decoder.decode(emb, beam_size=10)
```

### Issue: GP fitting fails

**Symptoms**: NaN in lengthscales, optimization errors

**Causes**:
1. Too few observations → Add more initial samples
2. Duplicate observations → Add small noise
3. Poor initialization → Use MSR method

**Solutions**:
```python
gp = SonarGPSurrogate(D=1024)
# MSR initialization is automatic

# Or use BAxUS for more stable fitting
gp = BAxUSGPSurrogate(D=1024, target_dim=128)
```

### Issue: Slow BO progress

**Symptoms**: Best score plateaus early

**Causes**:
1. Too few candidates → Increase `n_candidates`
2. Not enough exploration → Increase `alpha`
3. Warm-start with poor data → Use better initial data

**Solutions**:
```bash
# More candidates
--n-candidates 128

# More exploration
--alpha 2.0

# Warm-start with good prompts
--warm-start datasets/evaluated_instructions.pt
```

---

## File Structure

```
src/ecoflow/
├── __init__.py           # Public API exports
├── velocity_network.py   # DiT-style velocity network with AdaLN-Zero
├── flow_model.py         # FlowMatchingModel wrapper for ODE sampling
├── guided_flow.py        # GuidedFlowSampler with UCB + CFG-Zero*
├── gp_surrogate.py       # GP surrogates: MSR, BAxUS, Heteroscedastic
├── decoder.py            # SONAR embedding → text decoder
├── data.py               # SonarEmbeddingDataset and dataloader
├── train_flow.py         # Training script with EMA, cosine schedule
├── validate.py           # Generation quality validation
├── optimization_loop.py  # BOOptimizationLoop orchestrator
├── batch_selection.py    # Local Penalization for diverse batches
├── flow_density.py       # Hutchinson trace estimator for density
└── pipeline.md           # This documentation
```

---

## References

1. **Flow Matching**: Lipman et al. (2022) "Flow Matching for Generative Modeling"
2. **OT-CFM**: Tong et al. (2023) "Improving and Generalizing Flow-Based Generative Models"
3. **DiT**: Peebles & Xie (2023) "Scalable Diffusion Models with Transformers"
4. **MSR Method**: Hvarfner et al. (2024) "Vanilla Bayesian Optimization Performs Great in High Dimensions"
5. **BAxUS**: Papenmeier et al. (2022) "Increasing the Scope as You Learn: Adaptive BO in Nested Subspaces"
6. **Local Penalization**: Gonzalez et al. (2016) "Batch Bayesian Optimization via Local Penalization"
7. **CFG**: Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"
8. **SONAR**: FAIR (2023) "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations"
