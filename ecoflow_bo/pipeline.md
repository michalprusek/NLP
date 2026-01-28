# EcoFlow-BO: Complete Pipeline Documentation

## 1. System Overview

EcoFlow-BO transforms 768-dimensional GTR text embeddings into a 16-dimensional tractable latent space optimized for Bayesian Optimization.

**Purpose**: Enable efficient prompt optimization by operating in a compressed latent space where GP modeling is computationally tractable and the decoder ensures generated points remain on the learned embedding manifold.

**Core Innovation**: Matryoshka representation learning ensures information is hierarchically distributed across latent dimensions, enabling coarse-to-fine optimization that starts in 4D (cheap) and progressively unlocks to 16D (full expressivity).

### Component Summary

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| MatryoshkaEncoder | 768D embedding | 16D latent (mu, log_sigma) | Compress to tractable latent space |
| VelocityNetwork (DiT) | x_t, t, z | velocity v | Learn flow from noise to embedding |
| RectifiedFlowDecoder | 16D latent | 768D embedding | Deterministic 1-step decoding |
| PerceiverDecoder | 16D latent | 768D embedding | Alternative: query-based decoding |
| CoarseToFineGP | z, y observations | posterior(z) | Surrogate model for BO |
| DensityAwareAcquisition | GP, z_best | candidate z | Manifold-respecting exploration |
| CycleConsistencyChecker | z | valid/invalid | Hallucination detection |

---

## 2. Architecture Diagrams

### 2.1 High-Level System

```
                           TRAINING (Phase 1)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   x (768D)   ──→  [Encoder]  ──→  z (16D)                  │
    │       │              │              │                       │
    │       │              ▼              │                       │
    │       │         KL + InfoNCE       │                       │
    │       │                            │                       │
    │       │                            ▼                       │
    │       └────────────→  [DiT Decoder]  ──→  x_recon          │
    │                            │                                │
    │                            ▼                                │
    │                     Matryoshka CFM Loss                     │
    └─────────────────────────────────────────────────────────────┘

                        BAYESIAN OPTIMIZATION (Phase 2)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   z_candidates ──→ [Acquisition] ──→ z_best candidates     │
    │        │                                      │             │
    │        ▼                                      ▼             │
    │   [Cycle Check] ◄──────────────────── [Decoder]            │
    │        │                                      │             │
    │   valid?                                      ▼             │
    │    yes ──→ x_decoded ──→ [Objective] ──→ score ──→ [GP]    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### 2.2 MatryoshkaEncoder Architecture

```
    x (768D)
       │
       ▼
┌─────────────────────┐
│  ResidualDownBlock  │  768 → 768 (+ same-dim ResBlock)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  ResidualDownBlock  │  768 → 512 (+ same-dim ResBlock)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  ResidualDownBlock  │  512 → 256 (+ same-dim ResBlock)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  ResidualDownBlock  │  256 → 128 (+ same-dim ResBlock)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  ResidualDownBlock  │  128 → 64 (+ same-dim ResBlock)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  ResidualDownBlock  │  64 → 32 (+ same-dim ResBlock)
└─────────────────────┘
       │
       ├──→ [fc_mu]      → mu (16D)
       │
       └──→ [fc_log_sigma] → log_sigma (16D)

       ▼ (reparameterization)
       z = mu + sigma * eps,  eps ~ N(0,I)
```

**Matryoshka Structure**:
- Dims 0-3 (4D): 40% of CFM loss weight - coarse semantics
- Dims 4-7 (8D): 35% of CFM loss weight - medium detail
- Dims 8-15 (16D): 25% of CFM loss weight - fine nuances

### 2.3 VelocityNetwork (DiT) Architecture

```
    x_t (768D)           t (scalar)           z (16D)
        │                    │                   │
        ▼                    ▼                   ▼
┌───────────────┐    ┌─────────────┐    ┌─────────────────┐
│InputTokenizer │    │  Sinusoidal │    │ LatentExpander  │
│ 768→12×64→512 │    │  + MLP      │    │  16→16×512      │
└───────────────┘    └─────────────┘    └─────────────────┘
        │                    │                   │
        │               t_emb (512D)             │
        │                    │                   │
        ▼                    │                   ▼
   12 tokens              (AdaLN)           16 tokens
        │                    │                   │
        ├────────────────────┼───────────────────┤
        │                    │                   │
        ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────┐
│                 DiTBlock ×16                        │
│  ┌──────────────────────────────────────────────┐  │
│  │ Self-Attn(x, x, x) with AdaLN(t_emb)         │  │
│  │           ↓                                   │  │
│  │ Cross-Attn(x, z_tokens, z_tokens) + AdaLN   │  │
│  │           ↓                                   │  │
│  │ MLP + AdaLN                                   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│ OutputProjector │  12×512 → 768D
└─────────────────┘
        │
        ▼
     v (768D)
```

**Key Design Choices**:
- 12 input tokens: 768/12 = 64D per chunk, enables meaningful self-attention
- 16 latent tokens: Each z dimension is a separate token for selective attention
- AdaLN: Time modulates all normalizations (scale, shift)
- Matryoshka masking: z tokens are zeroed when z[i]=0

### 2.4 PerceiverDecoder Architecture (Alternative)

```
    z (16D)
       │
       ▼
┌────────────────────┐
│  LatentExpander    │  16 scalars → 16×1024 tokens
│  (16 separate MLPs)│  Matryoshka: zero tokens where z=0
└────────────────────┘
       │
       ▼
   16 tokens
       │
       ▼
┌────────────────────┐
│  ProcessorBlock    │  Self-attention on 16 tokens
│       ×12          │  (16×16 attention - very cheap!)
└────────────────────┘
       │
       ▼
   16 rich tokens
       │
       ▼
┌────────────────────────────────────┐
│    CrossAttentionReadout           │
│    768 learned queries attend      │
│    to 16 latent tokens             │
│    → 768 scalar outputs            │
└────────────────────────────────────┘
       │
       ▼
    x (768D)
```

**Why Perceiver for Embeddings**:
- No patching bias: Each GTR dimension has its own learned query
- Compute-efficient: Deep processing on only 16 tokens
- Semantic precision: Queries learn "what does dimension i mean?"

### 2.5 Coarse-to-Fine GP Stages

```
Stage 0: 4D (≤10 points)         Stage 1: 8D (10-25 points)       Stage 2: 16D (>25 points)
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  z = [z0,z1,z2,z3,      │     │  z = [z0,...,z7,        │     │  z = [z0,...,z15]       │
│       0, 0, 0, 0,       │     │       0, 0, ..., 0]     │     │                         │
│       0, 0, ..., 0]     │     │                         │     │                         │
│                         │     │                         │     │                         │
│  GP model on 4D space   │ ──→ │  GP model on 8D space   │ ──→ │  GP model on 16D space  │
│  Search bounds ±1.5σ    │     │  Search bounds ±2.0σ    │     │  Search bounds ±3.0σ    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
         │                               │                               │
         ▼                               ▼                               ▼
    Coarse semantic             Medium detail                Full expressivity
    exploration                 refinement                   fine-tuning
```

---

## 3. Component Deep-Dives

### 3.1 MatryoshkaEncoder (`encoder.py`)

**Purpose**: Compress 768D GTR embeddings to 16D latent space with hierarchical information structure.

**Key Features**:
- **Probabilistic output**: Returns (mu, log_sigma) for VAE training with reparameterization trick
- **Residual architecture**: Every layer has skip connections for gradient flow
- **Dropout for SimCSE**: Two forward passes with different dropout masks create positive pairs for contrastive learning

**Critical Methods**:
```python
# Training: sample with reparameterization
z, mu, log_sigma = encoder(x)

# Inference: deterministic (just mu)
z = encoder.encode_deterministic(x)

# Matryoshka: get embeddings at each level
[z_4d, z_8d, z_16d], mu, log_sigma = encoder.get_matryoshka_embeddings(x)
```

### 3.2 VelocityNetwork (`velocity_network.py`)

**Purpose**: Learn the velocity field v_θ(x_t, t, z) for conditional flow matching.

**Architecture Highlights**:
- **InputTokenizer**: Splits 768D into 12×64D chunks, projects to hidden_dim, adds position embeddings
- **LatentExpander**: Each z scalar → hidden_dim token with Matryoshka masking
- **DiTBlock**: Self-attention (x↔x) + Cross-attention (x→z) + MLP, all AdaLN-modulated by time
- **OutputProjector**: 12 tokens → 768D

**Parameter Count** (~55M with default config):
- InputTokenizer: 64×512 + 12×512 = ~39K
- LatentExpander: 16×512 = ~8K
- Time MLP: 512×512×2 = ~525K
- DiTBlocks: 16 × (3×self-attn + cross-attn + MLP) ≈ 54M
- OutputProjector: 512×64 = ~33K

### 3.3 RectifiedFlowDecoder (`cfm_decoder.py`)

**Purpose**: Transform noise to embeddings via ODE integration, with reflow for 1-step decoding.

**Key Concepts**:
- **CFM Training**: Learn straight-line interpolation x_t = t·x_1 + (1-t)·x_0
- **Target velocity**: u_t = x_1 - x_0 (constant along trajectory)
- **Reflow**: Re-train on generated (x_0, x_1) pairs to straighten curved trajectories
- **After reflow**: 1-step Euler decoding (dt=1.0, x_1 = x_0 + v)

**Decoding Modes**:
```python
# Standard (may need more steps before reflow)
x = decoder.decode(z, n_steps=20)

# Deterministic (same seed = same output, critical for GP)
x = decoder.decode_deterministic(z, seed=42)

# With iterative refinement
x = decoder.decode_with_refinement(z, encoder, n_refinement_steps=3)
```

### 3.4 PerceiverDecoder (`perceiver_decoder.py`)

**Purpose**: Alternative decoder using query-based attention, better suited for embedding reconstruction.

**Why Not DiT for Embeddings?**:
- DiT's patching creates artificial groupings of embedding dimensions
- Embedding dimensions have no spatial relationship (unlike image pixels)
- Perceiver treats each dimension independently with dedicated queries

**Parameter Count** (~70M with default config):
- LatentExpander: 16 × (2 × 1024²) ≈ 34M
- ProcessorBlocks: 12 × standard transformer block ≈ 35M
- Readout: 768 queries + cross-attn + MLP ≈ 1M

### 3.5 Losses (`losses.py`)

**KL Divergence**:
```
KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + 2*log_sigma - mu² - sigma²)
```
Regularizes latent space to standard normal, enabling:
- Sampling new z from prior for exploration
- Tractable density estimation: log p(z) ≈ -0.5‖z‖²

**InfoNCE (SimCSE-style)**:
```
L = -log(exp(sim(z1_i, z2_i)/τ) / sum_j(exp(sim(z1_i, z2_j)/τ)))
```
- Positive pairs: Two dropout-augmented views of same input
- Negatives: All other samples in batch
- Temperature τ=0.05 (low = hard negatives)

**MatryoshkaCFM**:
```
L = 0.4×CFM(z[:4]) + 0.35×CFM(z[:8]) + 0.25×CFM(z[:16])
```
Hierarchical loss ensuring information distribution across dimensions.

### 3.6 CoarseToFineGP (`latent_gp.py`)

**Purpose**: Gaussian Process surrogate for Bayesian Optimization with progressive dimension unlocking.

**Stage Advancement**:
```python
# Default schedule
points_per_stage = [10, 15, 30]  # Cumulative thresholds
active_dims_schedule = [[0,1,2,3], [0..7], [0..15]]

# Advances when n_points >= sum(points_per_stage[:stage+1])
```

**GP Configuration**:
- Kernel: Matern 5/2 with ARD (Automatic Relevance Determination)
- Priors: Gamma on lengthscales and output scale
- Outcome transform: Standardize(m=1) for numerical stability

### 3.7 DensityAwareAcquisition (`density_acquisition.py`)

**Purpose**: Generate candidates that balance exploration (UCB) with staying on the learned manifold (density).

**Acquisition Function**:
```
α(z) = UCB(z) + λ·log p_prior(z)
     = (μ(z) + β·σ(z)) + λ·(-0.5‖z‖²)
```

**Candidate Generation Strategy**:
- 40%: Random samples from N(0,I) - exploration
- 40%: Local perturbations around z_best - exploitation
- 20%: Latin Hypercube in search bounds - coverage

### 3.8 CycleConsistencyChecker (`cycle_consistency.py`)

**Purpose**: Detect decoder hallucinations before expensive objective evaluation.

**Workflow**:
```
z → [Decoder] → x_decoded → [Encoder] → z' → ‖z - z'‖ < threshold?
```

**Modes**:
- **Fixed threshold** (`adaptive=False`): Use config.error_threshold
- **Adaptive threshold** (`adaptive=True`): Calibrate from observed errors at 95th percentile

---

## 4. Data Flow

### 4.1 Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FORWARD PASS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  x (GTR embedding)                                                  │
│       │                                                             │
│       ├──→ Encoder(x) ──→ z1, mu, log_sigma  (view 1)              │
│       │                                                             │
│       └──→ Encoder(x) ──→ z2, _, _           (view 2, different dropout) │
│                                                                     │
│  Losses computed:                                                   │
│    1. Matryoshka CFM: decoder.compute_cfm_loss(x, z1_masked)       │
│       - For dim ∈ [4, 8, 16]: zero z1[dim:], compute CFM           │
│       - Weight: 0.4, 0.35, 0.25                                    │
│                                                                     │
│    2. KL Divergence: KL(mu, log_sigma)                             │
│       - Regularize to N(0,I)                                        │
│                                                                     │
│    3. Contrastive: InfoNCE(z1[:dim_rand], z2[:dim_rand])           │
│       - Random Matryoshka dim, dropout-augmented views              │
│                                                                     │
│  Total: CFM + 0.001×KL + 0.05×Contrastive                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 BO Optimization Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SINGLE BO ITERATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. CANDIDATE GENERATION                                            │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │  acquisition.generate_candidates(gp, z_best)             │    │
│     │  - 400 from N(0,I)                                       │    │
│     │  - 400 local perturbations around z_best                 │    │
│     │  - 200 Latin Hypercube samples                           │    │
│     │  All: zero out inactive dims, clamp to search bounds     │    │
│     └─────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  2. ACQUISITION SCORING                                             │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │  α(z) = UCB(z) + 0.5×log_prior(z)                        │    │
│     │  Sort candidates by α(z) descending                      │    │
│     └─────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  3. CYCLE CONSISTENCY CHECK (up to 5 retries)                       │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │  For each candidate in order:                            │    │
│     │    x_decoded = decoder(z)                                │    │
│     │    z_reenc = encoder(x_decoded)                          │    │
│     │    error = ‖z - z_reenc‖                                 │    │
│     │    if error < threshold: accept                          │    │
│     └─────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  4. OBJECTIVE EVALUATION                                            │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │  score = objective(x_decoded)                            │    │
│     │  (expensive - only call on valid candidates!)            │    │
│     └─────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  5. GP UPDATE                                                       │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │  gp.update(z_selected, score)                            │    │
│     │  - Add observation                                       │    │
│     │  - Check stage advancement                               │    │
│     │  - Refit GP                                              │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Phases

### Phase 1: Joint Encoder-Decoder Training

**Goal**: Learn bidirectional mapping between 768D and 16D spaces.

**Script**: `train_manifold.py`

**Command**:
```bash
torchrun --nproc_per_node=2 ecoflow_bo/train_manifold.py \
    --batch-size 2048 \
    --epochs 200 \
    --lr 1e-4 \
    --latent-dim 16 \
    --kl-weight 0.001 \
    --contrastive-weight 0.05
```

**Loss Components**:
- CFM (weight=1.0): Main reconstruction via flow matching
- KL (weight=0.001): Regularize latent to N(0,I)
- Contrastive (weight=0.05): Preserve semantic structure

**Optimizer**: LAMB with separate LRs (velocity net 3× encoder LR)

**Scheduler**: 10-epoch linear warmup → cosine annealing

### Phase 1b: Reflow Training (Optional)

**Goal**: Straighten flow trajectories for 1-step decoding.

**Process**:
1. Generate (x_0, z, x_1) pairs using multi-step ODE
2. Retrain velocity net on straight-line targets
3. Mark model as reflowed → config.euler_steps = 1

### Phase 1-alt: Perceiver Decoder Training

**Goal**: Train alternative query-based decoder.

**Script**: `train_perceiver.py`

**Key Differences**:
- Direct MSE reconstruction loss (no flow matching)
- Matryoshka weighting on reconstruction at each level
- Faster training, deterministic output

### Phase 2: Bayesian Optimization

**Goal**: Find optimal embeddings for a given objective.

**Usage**:
```python
optimizer = EcoFlowBO.from_checkpoint("results/best.pt")

def objective(embedding):
    text = decode_to_text(embedding)
    return evaluate_quality(text)

best_z, best_embedding, best_score = optimizer.optimize(
    initial_embeddings, initial_scores,
    objective=objective,
    n_iterations=50,
)
```

---

## 6. Mathematical Foundations

### 6.1 VAE ELBO and Reparameterization

**Evidence Lower Bound**:
```
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
        = Reconstruction - KL Regularization
```

**Reparameterization Trick**:
```
z = μ + σ ⊙ ε,  where ε ~ N(0,I)
```
Enables backprop through sampling by moving stochasticity to ε.

### 6.2 Conditional Flow Matching

**Goal**: Learn vector field v_θ that transforms source p_0 = N(0,I) to target p_1 = p_data.

**OT Interpolation**:
```
x_t = t·x_1 + (1-t)·x_0,  t ∈ [0,1]
```

**Target Velocity** (for OT path):
```
u_t = dx_t/dt = x_1 - x_0  (constant!)
```

**CFM Loss**:
```
L_CFM = E_{x_0~N(0,I), x_1~data, t~U[0,1]} [‖v_θ(x_t, t, z) - (x_1 - x_0)‖²]
```

### 6.3 Rectified Flow

**Problem**: Standard CFM may learn curved trajectories.

**Solution**: "Reflow" - retrain on generated straight-line pairs:
1. Sample z, x_0 ~ N(0,I)
2. Integrate ODE: x_1 = x_0 + ∫₀¹ v_θ(x_t, t, z) dt
3. Retrain with target u = x_1 - x_0

**Result**: Nearly straight trajectories → 1-step Euler sufficient.

### 6.4 Gaussian Process Posterior

**Prior**: f(z) ~ GP(0, k(z, z'))

**Kernel**: Matern 5/2 with ARD:
```
k(z, z') = σ² · (1 + √5r + 5r²/3) · exp(-√5r)
where r² = Σᵢ (zᵢ - z'ᵢ)² / ℓᵢ²
```

**Posterior** (conditioning on observations (Z, y)):
```
μ(z*) = k(z*, Z) · [K + σ²I]⁻¹ · y
σ²(z*) = k(z*, z*) - k(z*, Z) · [K + σ²I]⁻¹ · k(Z, z*)
```

### 6.5 UCB Acquisition Function

**Upper Confidence Bound**:
```
α_UCB(z) = μ(z) + β·σ(z)
```

**Density-Aware Extension**:
```
α(z) = α_UCB(z) + λ·log p(z)
     = μ(z) + β·σ(z) + λ·(-0.5‖z‖²)
```

The density term penalizes points far from the prior, where the decoder may hallucinate.

### 6.6 InfoNCE Contrastive Loss

**Objective**:
```
L = -E[log(exp(sim(z_i, z_i⁺)/τ) / Σⱼexp(sim(z_i, z_j)/τ))]
```

Where:
- z_i, z_i⁺: Positive pair (dropout-augmented views)
- z_j: Negatives (other samples in batch)
- τ: Temperature (0.05)
- sim: Cosine similarity

---

## 7. Configuration Reference

### EncoderConfig
```python
@dataclass
class EncoderConfig:
    input_dim: int = 768        # GTR embedding dimension
    latent_dim: int = 16        # Matryoshka max dimension
    hidden_dims: List[int] = [768, 512, 256, 128, 64, 32]
    dropout: float = 0.1        # For SimCSE augmentation
    matryoshka_dims: List[int] = [4, 8, 16]
    matryoshka_weights: List[float] = [0.4, 0.35, 0.25]
```

### DiTVelocityNetConfig
```python
@dataclass
class DiTVelocityNetConfig:
    data_dim: int = 768         # GTR dimension
    condition_dim: int = 16     # Must match encoder latent_dim
    hidden_dim: int = 512       # Token dimension
    n_layers: int = 16          # DiT blocks
    n_heads: int = 8            # Attention heads
    mlp_ratio: int = 4          # MLP hidden = hidden_dim × mlp_ratio
    dropout: float = 0.1
    n_input_tokens: int = 12    # 768 / 12 = 64D per token
    input_token_dim: int = 64
```

### GPConfig
```python
@dataclass
class GPConfig:
    active_dims_schedule: List[List[int]] = [[0,1,2,3], [0..7], [0..15]]
    points_per_stage: List[int] = [10, 15, 30]
    noise_prior_mean: float = 0.1
    lengthscale_prior_mean: float = 1.0
    use_ard: bool = True
```

### AcquisitionConfig
```python
@dataclass
class AcquisitionConfig:
    beta: float = 2.0           # UCB exploration parameter
    density_weight: float = 0.5 # Weight for prior density term
    n_candidates: int = 1000    # Candidates per iteration
    n_restarts: int = 10        # For gradient-based optimization
```

### CycleConfig
```python
@dataclass
class CycleConfig:
    error_threshold: float = 1.0  # Max ‖z - z'‖ for valid sample
    max_retries: int = 5          # Candidates to try before giving up
```

---

## 8. Example Usage

### 8.1 Training from Scratch

```bash
# Phase 1: Joint training (multi-GPU)
tmux new-session -d -s ecoflow_train \
  "torchrun --nproc_per_node=2 ecoflow_bo/train_manifold.py \
   --batch-size 2048 --epochs 200 --lr 1e-4 \
   2>&1 | tee results/train_$(date +%Y%m%d_%H%M%S).log; exec bash"

# Monitor
tmux attach -t ecoflow_train
```

### 8.2 Running BO Optimization

```python
from ecoflow_bo import EcoFlowBO
import torch

# Load trained model
optimizer = EcoFlowBO.from_checkpoint(
    "results/ecoflow_checkpoints/best.pt",
    device="cuda"
)

# Define objective (example: prompt quality)
def objective(embedding: torch.Tensor) -> float:
    # Your evaluation logic here
    # e.g., decode to text and score
    return score

# Initial points
initial_embeddings = torch.randn(10, 768, device="cuda")
initial_scores = torch.tensor([objective(e) for e in initial_embeddings])

# Run optimization
best_z, best_embedding, best_score = optimizer.optimize(
    initial_embeddings=initial_embeddings,
    initial_scores=initial_scores,
    objective=objective,
    n_iterations=50,
    verbose=True,
)

print(f"Best score: {best_score:.4f}")
```

### 8.3 Custom Objective Functions

```python
from sentence_transformers import SentenceTransformer
from vec2text import Inverter

# Setup decoders
inverter = Inverter.from_pretrained("gtr-base")
encoder_model = SentenceTransformer("gtr-t5-base")

def text_quality_objective(embedding: torch.Tensor) -> float:
    """Score based on decoded text quality."""
    # Decode embedding to text
    text = inverter.invert(embedding.cpu().numpy().reshape(1, -1))[0]

    # Your quality metric (examples):
    # - Perplexity under language model
    # - Task-specific score (e.g., QA accuracy)
    # - Human preference model score

    return compute_quality(text)
```

---

## 9. Troubleshooting Guide

### Training Issues

**Problem: NaN losses**
- Check: Learning rate too high (try 1e-5)
- Check: Gradient clipping (max_norm=1.0)
- Check: log_sigma clamping in encoder

**Problem: Poor reconstruction (low cosine similarity)**
- Increase: hidden_dim, n_layers
- Decrease: latent_dim (try 8)
- Try: Perceiver decoder instead of DiT

**Problem: Mode collapse (all z similar)**
- Increase: KL weight (try 0.01)
- Increase: Contrastive weight (try 0.1)
- Check: Encoder dropout enabled

### BO Issues

**Problem: All candidates fail cycle check**
- Increase: error_threshold (try 2.0)
- Enable: adaptive=True for auto-calibration
- Check: Encoder-decoder alignment (validate on training data)

**Problem: GP fitting fails (Cholesky error)**
- Check: Duplicate observations (add jitter)
- Try: Reduce points_per_stage
- Try: Add noise to y values

**Problem: Slow GP (>1s per update)**
- Reduce: n_candidates
- Use: Sparse GP approximation
- Limit: Max observations (e.g., keep only top 100)

### Memory Optimization

**48GB VRAM per GPU (L40S)**:
```python
# Maximize batch size
batch_size = 2048  # Per GPU

# Enable memory optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Use bfloat16 (no loss scaling needed)
with autocast(device_type="cuda", dtype=torch.bfloat16):
    ...
```

**For smaller GPUs**:
- Reduce batch_size (512-1024)
- Reduce hidden_dim (256-384)
- Reduce n_layers (8-12)
- Use gradient checkpointing

---

## 10. Appendices

### 10.1 File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~220 | All configuration dataclasses + utilities |
| `encoder.py` | ~200 | MatryoshkaEncoder with residual blocks |
| `velocity_network.py` | ~410 | DiT-based velocity predictor |
| `cfm_decoder.py` | ~380 | CFM training + rectified flow decoding |
| `perceiver_decoder.py` | ~360 | Alternative query-based decoder |
| `losses.py` | ~180 | KL, InfoNCE, MatryoshkaCFM losses |
| `latent_gp.py` | ~325 | Coarse-to-fine GP for BO |
| `density_acquisition.py` | ~310 | UCB + density acquisition |
| `cycle_consistency.py` | ~310 | Hallucination detection |
| `optimizer.py` | ~440 | Main EcoFlowBO class |
| `train_manifold.py` | ~350 | Joint training script |
| `train_perceiver.py` | ~350 | Perceiver training script |
| `data.py` | ~100 | DataLoader utilities |

### 10.2 Parameter Counts

**MatryoshkaEncoder** (~15M):
- Residual blocks: ~14M
- Output heads: ~1M

**VelocityNetwork (DiT)** (~55M):
- Input/Output tokenizers: ~100K
- DiTBlocks (16×): ~54M
- Time MLP: ~500K

**PerceiverDecoder** (~70M):
- LatentExpander: ~34M
- ProcessorBlocks (12×): ~35M
- Readout: ~1M

### 10.3 Hyperparameter Guidance

**Training**:
| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| batch_size | 2048 | 512-4096 | Scale LR with batch |
| lr | 1e-4 | 1e-5 to 5e-4 | LAMB scales internally |
| epochs | 200 | 100-500 | Until val_cosine plateaus |
| kl_weight | 0.001 | 0.0001-0.01 | Lower = less regularization |
| contrastive_weight | 0.05 | 0.01-0.1 | Higher = stronger clustering |

**BO**:
| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| beta | 2.0 | 1.0-4.0 | Higher = more exploration |
| density_weight | 0.5 | 0.1-1.0 | Higher = stay closer to prior |
| n_candidates | 1000 | 500-5000 | More = better coverage, slower |
| error_threshold | 1.0 | 0.5-2.0 | Lower = stricter validation |
