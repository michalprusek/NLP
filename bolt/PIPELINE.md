# BOLT Pipeline Documentation

**BOLT** = Bayesian Optimization with Latent Transformations

Joint optimization over instruction + exemplar selection using VAE latent space, Gaussian Process, Hyperband multi-fidelity scheduling, and Bayesian Optimization.

---

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Data Flow Diagram](#data-flow-diagram)
3. [Component Deep Dives](#component-deep-dives)
   - [GTR Encoder](#1-gtr-encoder)
   - [Structure-Aware VAE](#2-structure-aware-vae)
   - [Set Transformer](#3-set-transformer-exemplar-encoding)
   - [CrossAttention Scorer](#4-crossattention-scorer)
   - [Gaussian Process](#5-gaussian-process-gp)
   - [Hyperband](#6-hyperband-multi-fidelity-optimization)
   - [BO Inference](#7-bo-inference-loop)
4. [Training Pipeline](#training-pipeline)
5. [Loss Functions](#loss-functions)
6. [Dimensions Reference](#dimensions-reference)
7. [Parameters Reference](#parameters-reference)
8. [Usage Examples](#usage-examples)

---

## High-Level Overview

BOLT optimizes **prompts for few-shot learning** by jointly searching over:
1. **Instruction text** - what the model should do (e.g., "Think step by step...")
2. **Exemplar selection** - which 8 Q/A pairs to include as demonstrations

The key insight is that the optimal exemplars depend on the instruction, and vice versa. BOLT learns this joint relationship in a **24-dimensional latent space** (16D instruction + 8D exemplar) and uses **Bayesian Optimization** to find the best combination.

### Pipeline Stages

```
Stage 1: Data Preparation
  └── APE generates 2000 candidate instructions
  └── Load 6154 Q/A pairs from train.json as exemplar pool
  └── Load 1319 validation samples for evaluation

Stage 2: Hyperband Exploration
  └── Multi-fidelity search over (instruction, exemplar_set) space
  └── BO proposals guide exploration using GP surrogate
  └── Builds design_data for VAE/GP training

Stage 3: Model Training
  └── VAE learns to encode instructions + exemplars to 24D latent
  └── GP learns error_rate = f(z_joint) mapping (with DKL)
  └── CrossAttentionScorer learns which exemplars work with which instructions

Stage 4: BO Inference
  └── Optimize z_joint to minimize predicted error via UCB/LogEI
  └── Decode z_inst → Vec2Text → new instruction text
  └── Decode (z_inst, z_ex) → Scorer → top-8 exemplars
  └── Evaluate and update GP iteratively
```

---

## Data Flow Diagram

```
                            ┌─────────────────────────────────────┐
                            │         INPUT DATA                  │
                            ├─────────────────────────────────────┤
                            │  Instructions: 2000 (APE-generated) │
                            │  Q/A Pool: 6154 pairs (train.json)  │
                            │  Validation: 1319 samples           │
                            └──────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                              ▼
        ┌───────────────────────┐                    ┌───────────────────────┐
        │   GTR-T5-Base (768D)  │                    │   GTR-T5-Base (768D)  │
        │   Instruction → emb   │                    │   Q/A pairs → emb     │
        └───────────┬───────────┘                    └───────────┬───────────┘
                    │                                            │
                    ▼                                            ▼
        ┌───────────────────────┐                    ┌───────────────────────┐
        │  InstructionEncoder   │                    │  ExemplarSetEncoder   │
        │  (VAE: 768→256→128→32)│                    │  (Set Transformer)    │
        │  → μ_inst, σ_inst     │                    │  ISAB→ISAB→PMA        │
        └───────────┬───────────┘                    │  → μ_ex, σ_ex         │
                    │                                └───────────┬───────────┘
                    │                                            │
                    └──────────────┬─────────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │      Joint Refinement         │
                    │  z = [z_inst ‖ z_ex] (24D)    │
                    │  z = z + MLP(z)               │
                    └───────────┬───────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ InstructionDec  │ │  GP Model   │ │ CrossAttnScorer │
    │ 16→128→256→768  │ │ Matérn 5/2  │ │ (z,pool)→scores │
    │ → inst_emb_recon│ │ → μ(error)  │ │ → top-8 indices │
    └────────┬────────┘ └──────┬──────┘ └────────┬────────┘
             │                 │                  │
             ▼                 ▼                  ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │    Vec2Text     │ │  UCB/LogEI  │ │   Pool Lookup   │
    │ emb → text      │ │ acquisition │ │ indices → Q/A   │
    └────────┬────────┘ └──────┬──────┘ └────────┬────────┘
             │                 │                  │
             └─────────────────┴──────────────────┘
                               │
                               ▼
                    ┌───────────────────────────────┐
                    │       FINAL OUTPUT            │
                    │  instruction: "Think step..." │
                    │  exemplars: [Q/A₁,...,Q/A₈]   │
                    │  error_rate: 0.15             │
                    └───────────────────────────────┘
```

---

## Component Deep Dives

### 1. GTR Encoder

**File:** `encoder.py:21-72`

The GTR (Generalized T5 Retriever) encoder converts text to dense 768D embeddings that are compatible with Vec2Text for inversion.

```python
class GTREncoder:
    """GTR-T5-Base encoder for instruction and exemplar embeddings."""

    model = SentenceTransformer("sentence-transformers/gtr-t5-base")

    def encode(texts: List[str]) -> Tensor:  # (N, 768) L2-normalized
```

**Why GTR?**
- Produces high-quality semantic embeddings
- L2-normalized outputs work well with cosine similarity
- Compatible with Vec2Text for embedding → text inversion
- Trained on massive text corpora for general-purpose understanding

**Usage:**
```python
gtr = GTREncoder(device="cuda")
instruction_embs = gtr.encode(["Think step by step", "Show your work"])
# Shape: (2, 768)
```

---

### 2. Structure-Aware VAE

**File:** `encoder.py:268-547`

The VAE compresses high-dimensional embeddings (768D) to a compact latent space (24D) that captures the essential structure for optimization.

```
                    ┌─────────────────────────────────────┐
                    │        StructureAwareVAE            │
                    ├─────────────────────────────────────┤
                    │                                     │
  instruction_emb   │  ┌─────────────────────┐            │
  (768D) ──────────►│  │ InstructionEncoder  │            │
                    │  │ 768→256→128→32      │            │
                    │  │ → (μ_inst, logvar)  │            │
                    │  └─────────┬───────────┘            │
                    │            │                        │
                    │            │ reparameterize         │
                    │            ▼                        │
                    │       z_inst (16D)                  │
                    │            │                        │
  exemplar_embs     │  ┌─────────┴───────────┐            │
  (K×768D) ────────►│  │ ExemplarSetEncoder  │            │
                    │  │ (Set Transformer)   │            │
                    │  │ → (μ_ex, logvar)    │            │
                    │  └─────────┬───────────┘            │
                    │            │                        │
                    │            │ reparameterize         │
                    │            ▼                        │
                    │       z_ex (8D)                     │
                    │            │                        │
                    │  ┌─────────┴───────────┐            │
                    │  │  Joint Refinement   │            │
                    │  │  z = [z_inst‖z_ex]  │            │
                    │  │  z = z + MLP(z)     │            │
                    │  └─────────┬───────────┘            │
                    │            │                        │
                    │       z_joint (24D)                 │
                    │            │                        │
                    │     ┌──────┴──────┐                 │
                    │     ▼             ▼                 │
                    │  Decode       Scorer                │
                    │  z_inst→emb   (z,pool)→scores       │
                    └─────────────────────────────────────┘
```

**Key Methods:**

```python
# Encode to latent (for GP training)
z_joint = vae.encode_joint(inst_emb, ex_embs, ex_mask)  # (batch, 24)

# Full forward pass with loss
loss, loss_dict = vae.forward(
    instruction_emb,      # (batch, 768)
    exemplar_embs,        # (batch, K, 768)
    exemplar_mask,        # (batch, K)
    pool_embeddings,      # (N_pool, 768)
    target_exemplar_mask, # (batch, N_pool)
)

# Decode latent to outputs
inst_emb_recon, indices, scores = vae.decode(z_inst, z_ex, pool_embs, k=8)
```

---

### 3. Set Transformer (Exemplar Encoding)

**File:** `set_transformer.py`

The Set Transformer encodes variable-length exemplar sets in a **permutation-invariant** way - the order of exemplars doesn't matter, only which ones are included.

```
Input: exemplar_embeddings (batch, K, 768)
       mask (batch, K) - True = valid exemplar

       ┌────────────────────────────────────────────┐
       │              Set Transformer               │
       ├────────────────────────────────────────────┤
       │                                            │
       │  ┌──────────────────────────────────────┐  │
       │  │          ISAB₁ (768→128)             │  │
       │  │  I = learned inducing points (4)     │  │
       │  │  H = MAB(I, X) - inducing attend X   │  │
       │  │  out = MAB(X, H) - X attend inducing │  │
       │  └──────────────────────────────────────┘  │
       │                    ↓                       │
       │  ┌──────────────────────────────────────┐  │
       │  │          ISAB₂ (128→64)              │  │
       │  │  Same structure, smaller dim         │  │
       │  └──────────────────────────────────────┘  │
       │                    ↓                       │
       │  ┌──────────────────────────────────────┐  │
       │  │           PMA (64→64)                │  │
       │  │  S = learned seed (1)                │  │
       │  │  out = MAB(S, X) - pool to 1 vector  │  │
       │  └──────────────────────────────────────┘  │
       │                    ↓                       │
       │  ┌──────────────────────────────────────┐  │
       │  │       Linear heads (64→8)            │  │
       │  │  fc_mu, fc_logvar → VAE params       │  │
       │  └──────────────────────────────────────┘  │
       │                                            │
       └────────────────────────────────────────────┘

Output: μ_ex (batch, 8), logvar_ex (batch, 8)
```

**Key Insight:** ISAB (Induced Set Attention Block) reduces attention complexity from O(K²) to O(KM) where M=4 inducing points. This allows efficient processing of large exemplar sets.

**Why Set Transformer?**
- Permutation invariance: order doesn't matter
- Handles variable-length inputs (0-8 exemplars)
- Captures interactions between exemplars
- Produces fixed-size output for VAE latent

**Instruction-Conditioned Encoding:**

The PMA (Pooling Multihead Attention) now accepts instruction embedding for conditioning:

```python
# PMA seeds are modulated by instruction context
# This makes z_ex aware of what instruction it's paired with

# Standard PMA: seeds are learned parameters
S = self.seeds  # (num_seeds, embed_dim)

# Conditioned PMA: seeds are modulated by instruction
if instruction_emb is not None:
    inst_context = instruction_proj(instruction_emb)  # (batch, embed_dim)
    S = S + inst_context  # Additive modulation

# Then attention: output = MAB(S, X)
```

**Why instruction conditioning?**
- z_ex now encodes "exemplars relevant to THIS instruction" vs just "exemplars in general"
- Creates tighter coupling between instruction and exemplar latents
- Improves GP's ability to predict accuracy for instruction-exemplar pairs

---

### 4. CrossAttention Scorer

**File:** `encoder.py:157-266`

The CrossAttentionScorer learns which exemplars work well with which instructions using attention mechanisms. Given the joint latent (z_inst, z_ex), it scores all exemplars in the pool using cross-attention and selects the top-8.

**Advantages over concat+MLP:**
- Natural alignment mechanism between instruction and exemplars
- Attention weights are interpretable (which exemplars match the query)
- Multi-head attention captures different aspects of matching

```
                    ┌─────────────────────────────────────┐
                    │       CrossAttentionScorer          │
                    ├─────────────────────────────────────┤
                    │                                     │
  z_inst (16D) ─────┤  ┌─────────────────┐                │
                    │  │ query_proj      │                │
  z_ex (8D) ────────┤  │ concat→64D      │                │
                    │  │ Linear+LayerNorm│                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │           ▼                         │
                    │       query (64D)                   │
                    │           │                         │
  pool_emb          │  ┌────────┴────────┐                │
  (N×768D) ─────────┤  │ key_proj: 768→64│                │
                    │  │ value_proj:768→64                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │  ┌────────┴────────┐                │
                    │  │ MultiheadAttention               │
                    │  │ 4 heads, dropout=0.1             │
                    │  │ query attends to keys            │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │  ┌────────┴────────┐                │
                    │  │ Score MLP       │                │
                    │  │ 64→32→1         │                │
                    │  │ → base_scores   │                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │  scores = base_scores + attn_weights│
                    │           │                         │
                    │  ┌────────┴────────┐                │
                    │  │ Top-K Selection │                │
                    │  │ k=8             │                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │       indices (8,)                  │
                    └─────────────────────────────────────┘
```

**Training Signal:**
The scorer is trained with **ListMLE** ranking loss, which directly optimizes exemplar ranking:

```python
# ListMLE: negative log-likelihood of correct ranking order
# Targets come from Hyperband's best-performing exemplar sets

# For each instruction, get scores for all exemplars
scores = scorer(z_inst, z_ex, pool_embs)  # (batch, N_pool)

# Compute ListMLE loss (normalized by K)
# Loss encourages: scores[good_exemplars] > scores[bad_exemplars]
loss = listmle_loss(scores, target_mask) / K  # K=8
```

---

### 5. Gaussian Process (GP)

**File:** `gp.py`

The GP learns the mapping from latent space to error rate: `error = f(z_joint)`. It provides **uncertainty estimates** that guide exploration via acquisition functions.

```
                    ┌─────────────────────────────────────┐
                    │           GPWithEI                  │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │  Normalization:                     │
                    │    X_norm = (X - X_min) / range     │
                    │    y_norm = (y - y_mean) / y_std    │
                    │                                     │
                    │  Deep Kernel Learning (Optional):   │
                    │    z (24D) → FeatureExtractor → φ   │
                    │    φ_inst (16D), φ_ex (16D)         │
                    │                                     │
                    │  Kernel (Product Structure):        │
                    │    k(z,z') = k_inst(φ) × k_ex(φ)    │
                    │                                     │
                    │    k_inst: Matérn 5/2, ARD (16 dim) │
                    │    k_ex:   Matérn 5/2, ARD (8 dim)  │
                    │                                     │
                    │  Noise Model:                       │
                    │    FixedNoiseGaussianLikelihood     │
                    │    noise = Beta variance (fidelity) │
                    │                                     │
                    │  Output:                            │
                    │    μ(z): predicted error rate       │
                    │    σ(z): uncertainty estimate       │
                    │                                     │
                    └─────────────────────────────────────┘
```

**Deep Kernel Learning (DKL) - HbBoPs-Inspired Architecture:**

DKL transforms the VAE latent space through a neural network (JointFeatureExtractor) before
applying the kernel. Based on HbBoPs paper (Section 3.2) and DKL best practices (2-10D output):

```python
# JointFeatureExtractor architecture (HbBoPs-inspired)
# Maps 24D VAE latent → 10D joint representation
feature_extractor = JointFeatureExtractor(
    input_dim=24,    # VAE latent (16D inst + 8D ex)
    hidden_dim=32,   # Single hidden layer
    output_dim=10,   # Low-dim for GP (DKL best practice: 2-10D)
)

# Architecture: Lin(24, 32) → ReLU → Lin(32, 10)

# Example forward pass:
z_joint = [z_inst, z_ex]          # (batch, 24)
features = feature_extractor(z_joint)  # (batch, 10)

# GP operates on 10D joint features with single Matern kernel
k(z, z') = k_matern(φ(z), φ(z'))
```

**Why 10D output?**
- GPyTorch DKL examples use 2D output
- HbBoPs paper uses 10D joint representation
- Literature consensus: DKL feature extractors should compress to 2-10D
- Single kernel on joint features is simpler than product kernel

**Single vs Product Kernel:**

```python
# HbBoPs-style (default, use_product_kernel=False):
# Single Matern kernel on 10D joint features
covar_module = ScaleKernel(
    MaternKernel(nu=2.5, ard_num_dims=10)
)
# k(z, z') = k_matern(φ(z), φ(z'))

# Legacy product kernel (use_product_kernel=True):
# Splits 10D output into two halves for separate kernels
k_inst = MaternKernel(nu=2.5, ard_num_dims=5, active_dims=list(range(0, 5)))
k_ex = MaternKernel(nu=2.5, ard_num_dims=5, active_dims=list(range(5, 10)))
# k(z, z') = k_inst(φ(z)[:5]) × k_ex(φ(z)[5:])
```

**Beta Smoothing (Heteroscedastic Noise):**

Lower fidelity evaluations have higher variance. We model this with Beta distribution:

```python
# Smoothed error (posterior mean)
smoothed_error = (num_errors + α) / (fidelity + α + β)

# Noise variance (posterior variance)
noise_var = p(1-p) / (fidelity + α + β + 1)

# High fidelity → low variance → GP trusts observation more
# Low fidelity → high variance → GP treats as noisy
```

---

### 6. Hyperband (Multi-Fidelity Optimization)

**File:** `hyperband.py`

Hyperband efficiently explores the (instruction, exemplar_set) space using **successive halving** with multiple brackets. Each bracket trades off exploration breadth vs. evaluation depth.

```
                    ┌─────────────────────────────────────┐
                    │          HYPERBAND                  │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │  Parameters:                        │
                    │    nvalid = 1319 (max fidelity)     │
                    │    bmin = 10 (min fidelity)         │
                    │    η = 2 (halving rate)             │
                    │    smax = 7 (max bracket)           │
                    │                                     │
                    │  Bracket s=7 (aggressive):          │
                    │    n=100 candidates                 │
                    │    Start at fidelity 10             │
                    │    7 halving rounds                 │
                    │                                     │
                    │  Bracket s=0 (conservative):        │
                    │    n=12 candidates                  │
                    │    Start at fidelity 1319           │
                    │    0 halving rounds                 │
                    │                                     │
                    └─────────────────────────────────────┘
```

**Successive Halving Example (Bracket s=4):**

```
Round 0: n=32 candidates, fidelity=82
  ├── Evaluate all 32 at fidelity 82
  ├── Keep top 16 (halve)
  └── Cost: 32 × 82 = 2,624 samples

Round 1: n=16 candidates, fidelity=164
  ├── Extend evaluation to fidelity 164
  ├── Keep top 8 (halve)
  └── Cost: 16 × 82 = 1,312 new samples (reuse previous 82)

Round 2: n=8 candidates, fidelity=329
  ├── Extend to fidelity 329
  ├── Keep top 4 (halve)
  └── Cost: 8 × 165 new samples

Round 3: n=4 candidates, fidelity=659
  ├── Extend to fidelity 659
  ├── Keep top 2 (halve)
  └── Cost: 4 × 330 new samples

Round 4: n=2 candidates, fidelity=1319
  ├── Extend to full fidelity
  ├── Winner!
  └── Cost: 2 × 660 new samples
```

**BO Proposal Generation:**

```python
def propose_prompt():
    if random() < 0.1:  # 10% random interleaving
        return random_proposal()

    # BO-guided (90%)
    # 1. Sample 50 candidate instructions
    # 2. For each: optimize z_ex via gradient descent
    # 3. Return candidate with highest UCB

    for inst_id in sample(instructions, 50):
        z_inst = vae.encode_instruction(inst_id)
        z_ex_opt, ucb = optimize_exemplar_latent(z_inst)
        if ucb > best_ucb:
            best = (inst_id, z_ex_opt)

    return decode_to_prompt(best)
```

**Fidelity Extension Caching:**

```python
# Cache: (inst_id, frozenset(exemplar_ids), fidelity) → error_rate

# If we have fidelity=100 cached and need fidelity=200:
# 1. Evaluate only samples 100-200 (not 0-200)
# 2. Combine: total_error = (err_100 × 100 + err_new × 100) / 200
# 3. Cache result at fidelity=200

# Saves ~50% LLM calls in successive halving!
```

---

### 7. BO Inference Loop

**File:** `inference.py`

After Hyperband, BO continues optimization by generating novel prompts via latent space optimization and Vec2Text inversion.

```
                    ┌─────────────────────────────────────┐
                    │           BO INFERENCE              │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │  FOR iteration = 1 to N:            │
                    │                                     │
                    │    1. Optimize latent via UCB/LogEI │
                    │       z_opt = argmax UCB(z)         │
                    │       └─ L-BFGS-B, 64 restarts      │
                    │                                     │
                    │    2. Decode instruction            │
                    │       z_inst = z_opt[0:16]          │
                    │       emb = vae.decode(z_inst)      │
                    │       text = Vec2Text(emb)          │
                    │                                     │
                    │    3. Select exemplars              │
                    │       z_ex = z_opt[16:24]           │
                    │       indices = scorer.top_k(       │
                    │         z_inst, z_ex, pool, k=8)    │
                    │                                     │
                    │    4. Rejection sampling            │
                    │       IF cosine(emb, emb_recon)     │
                    │          < 0.90: retry with noise   │
                    │                                     │
                    │    5. Evaluate and update GP        │
                    │       error = evaluate(text, exs)   │
                    │       gp.add_observation(z, error)  │
                    │                                     │
                    │    6. Track best                    │
                    │       IF error < best_error:        │
                    │         best = (text, exemplars)    │
                    │                                     │
                    └─────────────────────────────────────┘
```

**UCB Acquisition Function:**

```python
# UCB for minimization (error rate)
UCB(z) = -μ(z) + β × σ(z)

# Adaptive beta: high early (exploration) → low late (exploitation)
β = 8.0 - progress × (8.0 - 2.0)

# At iteration 0: β=8.0 (explore widely)
# At iteration 50: β=2.0 (exploit best regions)
```

**Distance from Prior Penalty:**

```python
# Vec2Text produces gibberish when latent optimization strays far from N(0,I)
# We add a Gaussian prior penalty to keep z near the origin

# Acquisition with penalty
Score(z) = UCB(z) - λ_dist × 0.5 × ||z||²

# The penalty term is log P(z) where P(z) = N(0, I):
# log P(z) = -0.5 × ||z||² (ignoring constant)

# Config:
distance_penalty_enabled: bool = True
distance_weight: float = 2.0  # λ_dist
```

**MMR Exemplar Selection (Maximal Marginal Relevance):**

```python
# Pure top-k can select 8 redundant exemplars (same problem type)
# MMR balances relevance with diversity

# Score for each candidate:
MMR_score(ex) = λ × Relevance(ex) - (1-λ) × max_sim(ex, selected_set)

# Greedy selection:
# 1. First exemplar: pure relevance (highest score)
# 2. Subsequent: balance relevance vs diversity to already-selected

# Config:
use_mmr: bool = True
mmr_lambda: float = 0.7  # 1.0=pure relevance, 0.5=balanced, 0.0=pure diversity
```

**Vec2Text Inversion:**

```python
# Vec2Text inverts GTR embeddings back to text
# Uses iterative correction (50 steps) with beam search

inverter = Vec2TextInverter(
    model_type="32_tokens",  # Supports up to 32 tokens
    beam_width=8,
    max_length=128,
    num_steps=50,
)

# Input: GTR embedding (768D)
# Output: Text that would produce similar embedding
text = inverter.invert(embedding)
```

---

## Training Pipeline

### Stage 1: APE Instruction Generation

```python
# Generate diverse instructions using LLM
generator = APEGenerator(model="Qwen/Qwen2.5-7B-Instruct", backend="vllm")

instructions = generator.generate(
    validation_data=validation_samples[:10],  # Few examples for prompt
    num_instructions=2000,
    augment=True,  # Paraphrase augmentation
)
# Output: 2000 diverse instruction candidates
```

### Stage 2: Hyperband Evaluation

```python
hyperband = BOLTHyperband(
    instructions=instructions,      # 2000
    qa_pool=qa_pool,               # 6154 Q/A pairs
    validation_data=validation,     # 1319 samples
    vae=vae,
    ...
)

best_prompt, best_error = hyperband.run_hyperband()
design_data = hyperband.get_design_data_for_vae()
# design_data: ~200-500 evaluated (instruction, exemplar_set, error, fidelity)
```

### Stage 3: VAE Training

```python
# Compute selection targets from design_data
targets = compute_selection_targets(design_data, n_pool=6154)

trainer = VAETrainer(
    vae=vae,
    gtr_encoder=gtr,
    qa_pool=qa_pool,
    instructions=instructions,
    selection_targets=targets,
    config=config,
)

# Train with KL annealing
history = trainer.train(samples=design_data)
```

### Stage 4: GP Training

```python
# Encode all design points to latent space
latents = [vae.encode_joint(inst_emb, ex_embs, mask) for ...]
errors = [entry['error_rate'] for entry in design_data]
fidelities = [entry['fidelity'] for entry in design_data]

gp = GPWithEI(instruction_dim=16, exemplar_dim=8, use_deep_kernel=True)
gp.fit(X=latents, y=errors, fidelities=fidelities)
```

---

## Loss Functions

### VAE Total Loss

```
L_total = L_recon + β × L_KL + λ_sel × L_selection
```

### 1. Reconstruction Loss (Instruction)

```python
# Weighted combination of cosine and MSE
cosine_loss = 1 - cosine_similarity(emb_orig, emb_recon)
mse_loss = MSE(emb_orig, emb_recon)

L_recon = 0.8 × cosine_loss + 0.2 × mse_loss
```

**Why both?**
- Cosine: preserves direction (semantic meaning)
- MSE: preserves magnitude (for Vec2Text compatibility)

### 2. KL Divergence

```python
# Regularize both latent components toward N(0,1)
L_KL = KL(q(z_inst|x) || N(0,1)) + KL(q(z_ex|x) || N(0,1))

# With KL annealing during warmup
β_current = β × min(1, epoch / annealing_epochs)
```

### 3. Selection Loss (CrossAttention Scorer)

```python
# ListMLE: Learns to rank good exemplars above bad ones
# Normalized by K to make loss scale-invariant

# Compute weighted log-likelihood of correct ordering
weights = target_mask  # (batch, N_pool) - 1 for selected, 0 otherwise
log_likelihood = -log(softmax(scores))
weighted_ll = log_likelihood * weights

# Normalize by number of selected exemplars (K=8)
L_selection = -(weighted_ll.sum(dim=1) / K).mean()

# Encourages: scores[good_exemplars] > scores[bad_exemplars]
```

---

## Dimensions Reference

| Component | Dimension | Description |
|-----------|-----------|-------------|
| GTR embedding | 768 | SentenceTransformer output |
| Instruction latent (z_inst) | 16 | VAE encoder output |
| Exemplar latent (z_ex) | 8 | Set Transformer output (smaller - less complex) |
| Joint latent (z_joint) | 24 | Concatenation for GP (16 + 8) |
| ISAB hidden | 128 → 64 | Set Transformer internal |
| CrossAttention hidden | 64 | Query/Key/Value projection dimension |
| Score MLP | 64 → 32 → 1 | Value representation → score |
| Number of exemplars | 8 | Fixed K=8 always |
| Exemplar pool | 6154 | Q/A pairs from train.json |
| Instructions | 2000 | APE-generated candidates |
| Validation set | 1319 | GSM8K validation |
| Inducing points | 4 | ISAB efficiency |
| DKL output dim | 10 | JointFeatureExtractor output (HbBoPs-style) |

---

## Parameters Reference

```python
@dataclass
class BOLTConfig:
    # === Latent Dimensions ===
    embedding_dim: int = 768
    instruction_latent_dim: int = 16
    exemplar_latent_dim: int = 8   # Smaller - exemplars less complex
    # total_latent_dim = 24

    # === Set Transformer ===
    set_transformer_hidden: int = 128
    set_transformer_heads: int = 4
    num_inducing_points: int = 4

    # === Exemplar Selection ===
    num_exemplars: int = 8        # Fixed K=8
    scorer_hidden_dim: int = 128

    # === MMR (Maximal Marginal Relevance) Selection ===
    use_mmr: bool = True
    mmr_lambda: float = 0.7       # 1.0=relevance, 0.5=balanced, 0.0=diversity

    # === VAE Training ===
    vae_beta: float = 0.02        # KL weight (increased for better regularization)
    vae_mse_weight: float = 0.2   # 20% MSE + 80% cosine
    selection_weight: float = 0.2 # Reduced to not dominate training
    vae_epochs: int = 50000
    vae_annealing_epochs: int = 2500  # 5% warmup
    vae_lr: float = 0.0006
    vae_batch_size: int = 64
    vae_patience: int = 1000

    # === Hyperband ===
    bmin: int = 10               # Min fidelity
    eta: float = 2.0             # Halving rate
    random_interleaving_prob: float = 0.1

    # === GP ===
    gp_epochs: int = 10000
    gp_lr: float = 0.0025
    gp_patience: int = 100

    # === Deep Kernel Learning (HbBoPs-inspired) ===
    use_deep_kernel: bool = True   # Use DKL feature extractor
    dkl_output_dim: int = 10       # JointFeatureExtractor output (HbBoPs: 10D)
    dkl_hidden_dim: int = 32       # Hidden layer size
    use_product_kernel: bool = False  # False=single kernel, True=legacy product

    # === Inference ===
    iterations: int = 50
    num_restarts: int = 64       # L-BFGS restarts
    raw_samples: int = 4096
    acquisition_type: str = "ucb"  # or "logei"
    ucb_beta: float = 8.0        # Initial exploration
    ucb_beta_final: float = 2.0  # After decay
    cosine_sim_threshold: float = 0.90
    max_rejection_attempts: int = 10

    # === Distance Penalty (keeps latent near N(0,I) for good Vec2Text) ===
    distance_penalty_enabled: bool = True
    distance_weight: float = 2.0

    # === Contrastive Loss (DISABLED) ===
    # Dropout augmentation is too weak for meaningful contrastive learning
    # Paraphrase-based positives would be needed but require LLM calls
    use_contrastive: bool = False
    contrastive_weight: float = 0.0

    # === Vec2Text ===
    vec2text_beam: int = 8
    vec2text_model: str = "32_tokens"
    vec2text_max_length: int = 128
```

---

## Usage Examples

### Full Pipeline

```bash
# Full run: APE → Hyperband → VAE → GP → BO
uv run python -m bolt.run \
    --iterations 50 \
    --qa-pool-size 6154 \
    --ape-num-instructions 2000
```

### Skip APE (Use Cached)

```bash
uv run python -m bolt.run \
    --no-use-ape \
    --instructions bolt/data/ape_instructions.json
```

### Hyperband Only

```bash
uv run python -m bolt.run --hyperband-only
```

### Resume from Hyperband Results

```bash
uv run python -m bolt.run \
    --load-hyperband bolt/results/20260108_150000/hyperband_results.json
```

### Custom Parameters

```bash
uv run python -m bolt.run \
    --vae-epochs 30000 \
    --vae-beta 0.01 \
    --ucb-beta 10.0 \
    --iterations 100
```

---

## Hyperparameter Tuning System

**Directory:** `bolt/tuning/`

The tuning system provides automated hyperparameter optimization using Coordinate Descent with Bayesian Optimization guidance.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TuningCoordinator                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ VAE Phase   │→ │ Scorer Phase│→ │ GP Phase    │→ Inference  │
│  │ (Tier 1-3)  │  │ (Tier 1-3)  │  │ (Tier 1-3)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         ↑                ↑                ↑                     │
│         └────────────────┴────────────────┘                     │
│              Cyclic Coordinate Descent                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ParallelExecutor                              │
│  ┌────────────┐  ┌────────────┐                                │
│  │   GPU 0    │  │   GPU 1    │   Process isolation per trial  │
│  │  Trial A   │  │  Trial B   │   Automatic retry on failure   │
│  └────────────┘  └────────────┘   Checkpointing every N trials │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ArtifactCache                                │
│  SHA-256 hash-based caching: VAE, Scorer, GP checkpoints       │
│  Skip redundant training when config unchanged                  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| File | Description |
|------|-------------|
| `coordinator.py` | Orchestrates phases and cyclic optimization |
| `parallel_executor.py` | Multi-GPU trial execution with process isolation |
| `trial_runner.py` | Single trial execution: train VAE/GP, compute metrics |
| `hyperspace.py` | Parameter definitions with tier/phase organization |
| `metrics.py` | 25+ metrics across VAE, Scorer, GP, Optimization |
| `artifact_cache.py` | Hash-based caching to skip redundant training |
| `pruning.py` | ASHA early stopping with SharedASHAPruner for multi-process |
| `results_tracker.py` | JSON/CSV export and statistical analysis |
| `run_tuning.py` | CLI entry point |

### Tuning Phases

1. **VAE Phase**: Tune VAE architecture and training (latent dims, beta, etc.)
2. **Scorer Phase**: Tune CrossAttention Scorer (attention heads, hidden dim)
3. **GP Phase**: Tune Gaussian Process (DKL, kernel params)
4. **Inference Phase**: Tune BO acquisition (UCB beta, MMR lambda)

### Parameter Tiers

- **Tier 1 (CRITICAL)**: Highest impact, tune first (e.g., latent dims, beta)
- **Tier 2 (IMPORTANT)**: Medium impact, tune after Tier 1 stable
- **Tier 3 (FINETUNE)**: Low impact, final polish

### Usage

```bash
# Run tuning with pre-evaluated Hyperband results
uv run python -m bolt.tuning.run_tuning \
    --output-dir bolt/tuning_results \
    --gpu-ids 0 1 \
    --resume

# Quick test (reduced iterations)
uv run python -m bolt.tuning.run_tuning \
    --output-dir bolt/tuning_test \
    --quick-test
```

### ASHA Pruning (Asynchronous Successive Halving)

**File:** `pruning.py`

ASHA provides early stopping of unpromising trials during VAE training, saving 30-50% GPU time by killing bottom-performing trials at checkpoints.

```
                    ┌────────────────────────────────────────────────────┐
                    │                   ASHA Pruning                     │
                    ├────────────────────────────────────────────────────┤
                    │                                                    │
                    │  Rungs: [100, 500, 1000, 2500, 5000, 10000, 25000] │
                    │  Reduction Factor: 2 (kill bottom 50%)             │
                    │  Direction: maximize (cosine_mean)                 │
                    │  Min trials for pruning: 3                         │
                    │                                                    │
                    │  At each rung epoch:                               │
                    │    1. Report metric to shared state                │
                    │    2. Compare to other trials at rung              │
                    │    3. If in bottom 50% → PRUNE                     │
                    │                                                    │
                    └────────────────────────────────────────────────────┘
```

**Inter-Process Communication:**

Since trials run in separate processes (ProcessPoolExecutor), ASHA uses file-based shared state with atomic locking:

```python
class SharedASHAPruner:
    """File-based shared state for multi-process ASHA."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        # State structure: {"reports": {}, "pruned": [], "completed": [], "decisions": []}

    def _atomic_update(self, update_fn):
        """Atomically read, update, write state with fcntl.LOCK_EX."""
        fd = os.open(str(self.state_path), os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)  # Exclusive lock
        try:
            state = json.load(...)
            state = update_fn(state)
            json.dump(state, ...)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def should_stop(self, trial_id, step, metric_value, metric_name) -> bool:
        """Report metric and check if trial should be pruned."""
```

**Integration with VAE Training:**

The pruner integrates via `epoch_callback` parameter in `VAETrainer.train()`:

```python
# In trial_runner.py
def _run_vae_phase(self):
    # Create pruning callback
    def pruning_callback(epoch: int, metrics: Dict[str, float]) -> bool:
        metric_value = metrics.get("cosine_mean", 0.0)
        should_stop = self.pruner.should_stop(
            self.trial_id, epoch, metric_value, metric_name="cosine_mean"
        )
        if should_stop:
            logger.info(f"Trial {self.trial_id} PRUNED at epoch {epoch}")
        return should_stop

    # Train with callback
    trainer.train(samples=training_samples, epoch_callback=pruning_callback)

# In training.py (VAETrainer.train)
for epoch in range(epochs):
    train_step()

    # ASHA pruning callback
    if epoch_callback is not None:
        should_stop = epoch_callback(epoch + 1, epoch_losses)
        if should_stop:
            print(f"PRUNED at epoch {epoch + 1} by ASHA")
            break
```

**Coordinator Integration:**

```python
# In coordinator.py
class CoordinateDescentTuner:
    def __init__(self, ..., use_asha_pruning: bool = True):
        self.pruner_state_path = output_dir / "asha_pruner_state.json"

    def run(self):
        # Reset pruner state at start of each cycle
        if self.pruner_state_path.exists():
            self.pruner_state_path.unlink()

        # Pass pruner path to executor
        self.executor = DualGPUExecutor(
            ...,
            pruner_state_path=self.pruner_state_path,
        )
```

**State File Structure:**

```json
{
  "reports": {
    "trial_id_1": [
      {"step": 100, "value": 0.75, "metric_name": "cosine_mean", "timestamp": ...},
      {"step": 500, "value": 0.82, "metric_name": "cosine_mean", "timestamp": ...}
    ],
    "trial_id_2": [...]
  },
  "pruned": ["trial_id_3", "trial_id_5"],
  "completed": ["trial_id_1", "trial_id_2"],
  "decisions": [
    {"trial_id": "trial_id_3", "step": 500, "decision": "pruned", "percentile": 0.25}
  ]
}
```

**Expected Savings:**

| Scenario | Without ASHA | With ASHA | Savings |
|----------|--------------|-----------|---------|
| 50 VAE trials | 50 × 50k epochs | ~25 × 50k + 25 × avg 1.5k | ~40% |
| Bad beta config | 50k epochs wasted | Killed at epoch 500 | 99% |

### Key Metrics

| Category | Metric | Target |
|----------|--------|--------|
| VAE | Retrieval Accuracy @ 8 | ≥ 0.85 |
| VAE | Reconstruction Cosine | ≥ 0.90 |
| Scorer | NDCG @ 8 | ≥ 0.70 |
| GP | Spearman Correlation | ≥ 0.30 |
| End-to-End | Best Error Rate | ≤ 0.15 |

---

## Expected Results

Based on instruction-only baseline:
- Instruction-only accuracy: ~82-85%
- With 8 exemplars: expected +2-5% improvement
- **Target: ~87-90% accuracy**

Typical run statistics:
- Hyperband evaluations: ~200-500 unique (instruction, exemplar_set) pairs
- Total LLM calls: ~5,000-10,000 (with fidelity extension)
- Runtime: 2-4 hours on single L40S GPU
