# LID-O++ Pipeline Documentation

**Latent Instruction Diffusion Optimization++** for NeurIPS 2026

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LID-O++ Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   GritLM    │    │   FlowDiT   │    │   Value     │    │  Evaluation │  │
│  │   Encoder   │───▶│  (Rectified │───▶│    Head     │───▶│   Gating    │  │
│  │   (768D)    │    │    Flow)    │    │   (32→1)    │    │    (FCU)    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                                     │          │
│         │                  │                                     │          │
│         ▼                  ▼                                     ▼          │
│  ┌─────────────┐    ┌─────────────┐                      ┌─────────────┐   │
│  │   Latent    │    │     ODE     │                      │     LLM     │   │
│  │  Injection  │◀───│  Solver +   │                      │  Evaluator  │   │
│  │   Decoder   │    │  Curvature  │                      │  (GSM8K)    │   │
│  └─────────────┘    └─────────────┘                      └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. GritLM Unified Backbone

**Model**: `GritLM/GritLM-7B` (7B parameters, Mistral-based)

**Purpose**: Unified encoder/decoder using the same model with different attention masks.

### Encoder Mode
- **Input**: Text instruction
- **Output**: 768D normalized embedding
- **Pooling**: Latent Attention (4 queries × 4096D → 768D)
- **Normalization**: L2 normalized

```python
# Architecture
GritLMUnifiedEncoder(
    model_name="GritLM/GritLM-7B",
    output_dim=768,           # Final embedding dimension
    dtype=torch.float16,      # Memory efficient
    quantize=False,           # Full precision for quality
)
```

### Decoder Mode (Latent Injection)
- **Input**: 768D latent vector
- **Projection**: 768D → 4 × 4096D prefix tokens
- **Generation**: Autoregressive with prefix conditioning

```python
# Latent Projector Architecture
LatentProjector(
    latent_dim=768,           # From encoder
    hidden_dim=4096,          # GritLM hidden size
    num_prefix_tokens=4,      # Conditioning tokens
    intermediate_dim=3072,    # MLP intermediate
    dropout=0.1,
)

# MLP Structure:
# 768 → 3072 (GELU) → 3072 (GELU) → 16384 (4×4096)
# + LayerNorm + Learnable scale (init=10.0)
```

**Training**:
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Mixed precision: GradScaler + autocast
- Gradient clipping: max_norm=1.0
- Loss: Cross-entropy on next-token prediction

## 2. FlowDiT (Flow Diffusion Transformer)

**Purpose**: Learn velocity field for latent space navigation.

### Architecture
```python
FlowDiT(
    latent_dim=32,            # VAE latent dimension
    context_dim=768,          # GritLM embedding dimension
    hidden_dim=512,           # Transformer hidden size
    num_layers=6,             # Transformer blocks
    num_heads=8,              # Attention heads
    dropout=0.1,
)
```

### Components
1. **Input Projection**: 32D → 512D
2. **Timestep Embedding**: Sinusoidal + MLP (512D)
3. **Context Cross-Attention**: Query from x_t, Key/Value from context
4. **AdaLayerNorm**: Timestep-conditioned normalization
5. **Output Projection**: 512D → 32D velocity

### Flow Matching Training
```python
# Conditional Flow Matching (CFM) Loss
x_t = (1 - t) * x_0 + t * x_1  # OT interpolation
v_target = x_1 - x_0           # Constant velocity (straight line)
v_pred = model(x_t, t, context)
loss = MSE(v_pred, v_target)
```

### OAT-FM Regularization (Optional)
```python
# Optimal Affine Transport regularization
oat_loss = ||d²x_t/dt²||²  # Minimize acceleration
total_loss = cfm_loss + 0.1 * oat_loss
```

## 3. ODE Solvers with Curvature Tracking

### Available Solvers
| Solver | Steps | Error | Use Case |
|--------|-------|-------|----------|
| Euler | 20 | O(dt) | Fast inference |
| Midpoint | 10 | O(dt²) | Balanced |
| RK4 | 5 | O(dt⁴) | High quality |
| One-step | 1 | Learned | After Reflow |

### Curvature Computation
```python
# Flow Curvature Uncertainty (FCU)
curvature = Σ ||v(x_{t+dt}) - v(x_t)||²  # Sum of velocity changes

# High curvature = uncertain trajectory = needs LLM evaluation
# Low curvature = confident trajectory = use Value Head
```

### Guided Flow Matching (Classifier Guidance for Flow)

**Purpose**: Guide FlowDiT generation toward high-reward (low error) regions using GP/ValueHead gradients.

Standard flow generates random instructions from the learned distribution. Guided flow modifies the velocity field to follow the reward gradient:

```
v_guided(x_t, t) = v_base(x_t, t) + s(t) · ∇_x R(x_t)
```

Where:
- `v_base`: Original FlowDiT velocity
- `s(t)`: Time-dependent guidance scale
- `R(x)`: Reward function (GP UCB, EI, or ValueHead)

```
Standard Flow:                      Guided Flow:

x₀ ──v──▶ ──v──▶ ──v──▶ x₁         x₀ ──v+∇R──▶ ──v+∇R──▶ x₁*
(noise)              (random)       (noise)              (optimal!)
```

#### Time-Dependent Guidance (Critical!)

At t=0, x_t is pure Gaussian noise where GP gradients are meaningless. Guidance must ramp up as structure forms:

| Schedule | Formula | Use Case |
|----------|---------|----------|
| `linear` | s(t) = s_base · t | **Recommended** - smooth ramp |
| `cosine` | s(t) = s_base · (1-cos(πt))/2 | Very smooth S-curve |
| `quadratic` | s(t) = s_base · t² | Conservative start |
| `sqrt` | s(t) = s_base · √t | Aggressive early guidance |
| `step` | s(t) = s_base if t > t₀ else 0 | Hard threshold |
| `warmup` | Linear to t₀, then constant | Hybrid |

```python
# Recommended configuration
result = guided_euler_integrate(
    flowdit,
    x_0=noise,
    reward_fn=gp_reward,
    guidance_scale=1.0,
    guidance_schedule="linear",  # s(t) = 1.0 * t
    num_steps=20,
)
```

#### Reward Functions with Regularization

To prevent guided generation from leaving the VAE's learned distribution:

```
Total Reward(z) = UCB(z) - λ · ||z - μ_train||²
```

| Regularization | Formula | When to Use |
|----------------|---------|-------------|
| `none` | No penalty | Debugging only |
| `l2` | λ·\|\|z\|\|² | VAE prior is N(0,I) |
| `l2_centered` | λ·\|\|z - μ_train\|\|² | **Recommended** for small data |
| `mahalanobis` | λ·(z-μ)ᵀΣ⁻¹(z-μ) | N > 50 samples |

```python
from lido_pp.flow.ode_solver import guided_euler_integrate, GPRewardWrapper

# Create regularized reward function
gp_reward = GPRewardWrapper(
    gp_model,
    mode="ucb",              # UCB acquisition
    beta=2.0,                # Exploration coefficient
    regularization="l2_centered",  # Stay near training data
    reg_lambda=0.1,          # Regularization strength
)

# Guided generation
result = guided_euler_integrate(
    flowdit,
    x_0=torch.randn(batch_size, 32, device="cuda"),
    reward_fn=gp_reward,
    guidance_scale=1.0,
    guidance_schedule="linear",
    num_steps=20,
)

optimized_latent = result.x_final  # (B, 32) - guided toward high UCB
```

#### Hyperparameter Guidelines

| Parameter | Range | Effect |
|-----------|-------|--------|
| `guidance_scale` | 0.1 - 2.0 | Higher = stronger pull toward reward |
| `reg_lambda` | 0.05 - 0.5 | Higher = more conservative (stays closer to training) |
| `beta` (UCB) | 1.0 - 4.0 | Higher = more exploration |

**Trade-offs**:
- High `guidance_scale` + low `reg_lambda` → May leave VAE distribution (gibberish text)
- Low `guidance_scale` + high `reg_lambda` → Conservative, may miss optima
- **Recommended**: `guidance_scale=1.0`, `reg_lambda=0.1`, `guidance_schedule="linear"`

## 4. Value Head / GP Surrogate

**Purpose**: Predict instruction quality without expensive LLM evaluation.

### Option A: Neural Value Head (N > 100 samples)

For larger datasets, a neural network provides fast inference:

```python
ValueHead(
    latent_dim=32,
    hidden_dim=128,
    dropout=0.1,
)

# Structure:
# 32 → 128 (GELU, LN) → 128 (GELU, LN) → 1 (Sigmoid)
# Output: error_rate ∈ [0, 1]
# Parameters: ~21K
```

**Training**:
- Replay buffer: 10,000 samples
- Loss: MSE on (predicted_error, actual_error)
- Online updates during BO

**With Uncertainty (MC Dropout)**:
```python
ValueHeadWithUncertainty(
    num_mc_samples=10,  # Forward passes for uncertainty
)
# Returns: (mean_prediction, std_prediction)
```

### Option B: Gaussian Process (N < 100 samples) ⭐ Recommended for Small Data

For small datasets (10-100 labeled prompts), GP is superior:

| Aspect | Neural Network | GP |
|--------|---------------|-----|
| Min samples | ~100-500 | **5-50** |
| Overfitting | High risk | **None** |
| Uncertainty | MC Dropout (hack) | **Analytical** |
| Interpretability | Black box | **ARD lengthscales** |

```python
from lipo.gp import GPWithEI

# GP with Matern 5/2 ARD kernel on 32D VAE latent
gp = GPWithEI(device="cuda")
gp.vae_with_adapter = vae_encoder  # For embedding → latent conversion

# Fit on labeled prompts
gp.set_training_data(
    embeddings,      # (N, 768) GritLM embeddings
    error_rates,     # (N,) in [0, 1]
    fidelities,      # (N,) sample counts for Beta posterior
)
gp.train(epochs=1000)

# Predict with uncertainty
mean, std = gp.predict(new_embedding)  # Returns positive error rate
```

**GP Architecture**:
```
embeddings (768D) → frozen VAE encoder → z (32D) → normalize [0,1] → GP

GP Kernel: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=32))
- ARD: 32 per-dimension lengthscales (learns relevance)
- Small lengthscale = dimension is important
- Large lengthscale = dimension is ignored
```

**Heteroscedastic Noise (Beta Posterior)**:
```python
# For error_rate p measured on n samples:
posterior_variance = p * (1-p) / (n + α + β + 1)

# Low fidelity (n=50) → high uncertainty
# High fidelity (n=1319) → low uncertainty
```

### When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Initial exploration (10-50 prompts) | **GP** - no overfitting |
| After Hyperband (100-500 prompts) | **GP** or Value Head |
| Large-scale BO (1000+ evaluations) | **Value Head** - faster inference |
| Guided Flow Matching | **GP** - differentiable + uncertainty |

## 5. Evaluation Gating

**Purpose**: Decide between expensive LLM evaluation and cheap Value Head.

### Decision Logic
```
1. Check cache (hash-based, tolerance=1e-4)
2. Compute FCU from ODE integration
3. If FCU > percentile_threshold (90th): → LLM evaluation
4. Else: → Value Head prediction
```

### Adaptive Threshold
```python
EvaluationGate(
    percentile_threshold=90.0,    # FCU percentile for LLM
    adaptive_threshold=True,      # Adjust based on accuracy
    min_samples_for_threshold=50, # Minimum history
)
```

### Budget Management
```python
AdaptiveGate(
    total_budget=100,           # Total LLM evaluations allowed
    target_llm_ratio=0.2,       # Target 20% LLM usage
)
```

## 6. Cost-Aware Acquisition

**Purpose**: Balance exploration (high uncertainty) with exploitation (high value).

### Acquisition Function
```python
# qLogEI with FCU weighting
acquisition = EI(z) * exp(-fcu_weight * curvature)

# High EI + Low curvature = good candidate
# High EI + High curvature = uncertain, maybe evaluate with LLM
```

## 7. Reflow Training

**Purpose**: Straighten ODE trajectories for 1-step inference.

### Process
```
1. Generate trajectories with current model
2. Store (x_0, x_1) pairs from trajectory endpoints
3. Retrain model on straightened pairs
4. Repeat until 1-step error < threshold
```

### Metrics
```python
# Straightness verification
straightness = {
    "avg_deviation": mean ||x_t - linear_interp||,
    "max_deviation": max ||x_t - linear_interp||,
    "velocity_variance": var(v_t),
    "path_length_ratio": actual_length / straight_length,
}
```

## Training Pipeline

### Phase 1: Projector Training (Latent Injection)
```bash
uv run python -m lido_pp.run train-projector \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda:0
```

### Phase 2: FlowDiT Training
```bash
uv run python -m lido_pp.run train \
    --iterations 100 \
    --batch-size 32 \
    --flow-lr 1e-4 \
    --device cuda:0
```

### Phase 3: Reflow (Optional)
```bash
uv run python -m lido_pp.run reflow \
    --rounds 3 \
    --trajectories-per-round 1000 \
    --target-straightness 0.01
```

### Phase 4: GSM8K Evaluation
```bash
uv run python -m lido_pp.run evaluate \
    --dataset gsm8k \
    --num-samples 500 \
    --use-value-head
```

## ⚠️ CRITICAL: Training Policy — Freeze vs. Trainable

**This is one of the most important principles for architecture stability.**

During task-specific training (e.g., GSM8K), **ONLY train FlowDiT and Value Head**. The Latent Projector must be **FROZEN** after pre-training.

### Component Training Status

| Component | Status | Role |
|-----------|--------|------|
| GritLM (7B) | 🧊 **FROZEN** | Intelligence. Knows everything about the world and math. |
| Latent Projector | 🧊 **FROZEN** | Translator. Maintains stable language between vectors and text. |
| FlowDiT | 🔥 **TRAINABLE** | Strategist. Searches the latent map for optimal instructions. |
| Value Head | 🔥 **TRAINABLE** | Evaluator. Assesses if found paths lead to success. |

### Why NOT Train Projector on Task?

**1. Moving Target Problem**
- FlowDiT is an archer, Projector is the target
- FlowDiT learns to hit specific vectors meaning "solve math"
- If Projector changes weights simultaneously, the target moves while aiming
- Result: FlowDiT gets confused — vector that meant "add" yesterday means "multiply" today
- **Model will not converge**

**2. Projector is Just a Dictionary**
- Projector's job is NOT to be smart — only to translate
- Word "Integral" has the same meaning in math and general English
- Once Projector learns (during pre-training) that vector `[0.5, -0.2, ...]` → "Integral", don't touch it
- Retraining on small dataset (GSM8K) causes **Catastrophic Forgetting** — may forget other words needed for creative solutions

**3. Loss of GritLM Anchoring**
- GritLM Encoder and Decoder are fixed reference points
- Projector serves as a bridge between them
- Bending the bridge for a specific task breaks correspondence with what GritLM Encoder originally intended
- Latent space becomes misaligned

### Correct Training Workflow

```python
# Pre-training (once, on Alpaca/UltraChat data)
projector = LatentProjector(...)
train_projector(projector, alpaca_data)
torch.save(projector.state_dict(), "projector_v1.pth")

# Task training (GSM8K) — Projector FROZEN
projector.load_state_dict(torch.load("projector_v1.pth"))
for param in projector.parameters():
    param.requires_grad = False  # 🧊 FREEZE

flowdit = FlowDiT(...)  # 🔥 Trainable
value_head = ValueHead(...)  # 🔥 Trainable

# Only FlowDiT and ValueHead get gradients during task training
```

## Dimensions Summary

| Component | Input | Output | Parameters | Status |
|-----------|-------|--------|------------|--------|
| GritLM Encoder | text | 768D | 7B | 🧊 frozen |
| Latent Attention | 4096D×L | 768D | 106M | 🧊 frozen |
| Latent Projector | 768D | 4×4096D | 62M | 🧊 frozen |
| FlowDiT | 32D + 768D | 32D | 35M | 🔥 trainable |
| Value Head | 32D | 1D | 21K | 🔥 trainable |

## Memory Requirements

| Configuration | GPU Memory |
|--------------|------------|
| GritLM fp16 | ~14GB |
| + FlowDiT training | ~16GB |
| + Batch size 32 | ~20GB |
| Full pipeline | ~24GB |

**Recommended**: 2× L40S (48GB each) for parallel experiments.

## Key Hyperparameters

```python
# config.py defaults
GRITLM_MODEL = "GritLM/GritLM-7B"
EMBEDDING_DIM = 768
VAE_LATENT_DIM = 32
FLOW_HIDDEN_DIM = 512
FLOW_NUM_LAYERS = 6
OAT_WEIGHT = 0.1
FCU_PERCENTILE = 90.0
VALUE_HEAD_HIDDEN = 128
PROJECTOR_NUM_TOKENS = 4
```

## File Structure

```
lido_pp/
├── __init__.py
├── config.py              # Hyperparameters
├── run.py                 # CLI entry point
├── backbone/
│   ├── gritlm_encoder.py  # GritLM unified encoder
│   ├── latent_attention.py # Attention pooling
│   └── latent_injection.py # Decoder + Projector
├── flow/
│   ├── flow_dit.py        # FlowDiT model
│   ├── losses.py          # CFM + OAT losses
│   ├── ode_solver.py      # Euler/Midpoint/RK4
│   └── reflow.py          # Trajectory straightening
├── active_learning/
│   ├── curvature.py       # FCU computation
│   ├── value_head.py      # Cheap predictor
│   ├── acquisition.py     # Cost-aware acquisition
│   └── gating.py          # Evaluation gating
└── training/
    ├── data_prep.py       # Dataset handling
    ├── trainer.py         # Training orchestration
    └── checkpointing.py   # Model saving
```

## Complete Guided Flow Pipeline (Small Data Regime)

End-to-end example for prompt optimization with only 20 labeled examples:

```python
import torch
from lipo.gp import GPWithEI
from lido_pp.flow.ode_solver import guided_euler_integrate, GPRewardWrapper
from lido_pp.flow.flow_dit import FlowDiT
from lido_pp.vae import InstructionVAE

# ═══════════════════════════════════════════════════════════════════
# 1. SETUP: Load pre-trained components
# ═══════════════════════════════════════════════════════════════════

device = "cuda"

# Load VAE (frozen, pre-trained on Alpaca)
vae = InstructionVAE(input_dim=768, latent_dim=32).to(device)
vae.load_state_dict(torch.load("checkpoints/vae_best.pt")["model_state_dict"])
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# Load FlowDiT (frozen or fine-tuned)
flowdit = FlowDiT(latent_dim=32, context_dim=768).to(device)
flowdit.load_state_dict(torch.load("checkpoints/flowdit_best.pt")["model_state_dict"])
flowdit.eval()

# ═══════════════════════════════════════════════════════════════════
# 2. FIT GP: Train on 20 labeled prompts
# ═══════════════════════════════════════════════════════════════════

# Your labeled data (from LLM evaluation on GSM8K)
embeddings = torch.load("data/labeled_embeddings.pt")  # (20, 768)
error_rates = torch.tensor([0.35, 0.22, 0.18, ...])    # (20,)
fidelities = torch.ones(20) * 100  # Each evaluated on 100 samples

# Create and train GP
gp = GPWithEI(device=device)
gp.vae_with_adapter = vae  # VAE for embedding → latent
gp.set_training_data(embeddings, error_rates, fidelities)
gp.train(epochs=1000, verbose=True)

print(f"Best observed error: {gp.best_error_rate:.2%}")

# ═══════════════════════════════════════════════════════════════════
# 3. CREATE REWARD: UCB with regularization
# ═══════════════════════════════════════════════════════════════════

reward_fn = GPRewardWrapper(
    gp,
    mode="ucb",                    # Explore + exploit
    beta=2.0,                      # Exploration coefficient
    regularization="l2_centered",  # Stay near training data
    reg_lambda=0.1,                # Regularization strength
)

# ═══════════════════════════════════════════════════════════════════
# 4. GUIDED GENERATION: Generate optimized prompts
# ═══════════════════════════════════════════════════════════════════

batch_size = 16
noise = torch.randn(batch_size, 32, device=device)

# Optional: context from existing good prompt
context = None  # Or: gritlm.encode(["Solve step by step"])

result = guided_euler_integrate(
    flowdit,
    x_0=noise,
    reward_fn=reward_fn,
    context=context,
    num_steps=20,
    guidance_scale=1.0,
    guidance_schedule="linear",    # Ramp from 0 to 1
    guidance_start_t=0.2,          # For warmup schedule
    return_trajectory=False,
)

optimized_latents = result.x_final  # (16, 32)

# ═══════════════════════════════════════════════════════════════════
# 5. DECODE: Convert latents back to text
# ═══════════════════════════════════════════════════════════════════

# Decode through VAE → embedding → Projector → text
with torch.no_grad():
    # VAE decode (32D → 768D approximation)
    reconstructed_embeddings = vae.decode(optimized_latents)

    # Use Projector for text generation
    instructions = projector.generate(reconstructed_embeddings)

for i, instr in enumerate(instructions):
    print(f"Candidate {i}: {instr}")

# ═══════════════════════════════════════════════════════════════════
# 6. EVALUATE & ITERATE: Test best candidates, update GP
# ═══════════════════════════════════════════════════════════════════

# Evaluate top candidates on GSM8K
for candidate_embedding, error_rate in evaluated_candidates:
    gp.add_observation(candidate_embedding, error_rate, fidelity=100)

# Retrain GP with new data
gp.train(epochs=500)

# Repeat steps 3-6 until convergence
```

## References

1. **GritLM**: Muennighoff et al., "Generative Representational Instruction Tuning"
2. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling"
3. **OAT-FM**: Pooladian et al., "Optimal Affine Transport Flow Matching"
4. **Reflow**: Liu et al., "Rectified Flow: A Marginal Preserving Approach"
5. **Classifier Guidance**: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis"
6. **Gaussian Processes**: Rasmussen & Williams, "Gaussian Processes for Machine Learning"
