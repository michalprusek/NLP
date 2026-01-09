# LIPO-E Pipeline Documentation

**LIPO-E** = Latent Instruction Prompt Optimization with Exemplars

Joint optimization over instruction + exemplar selection using VAE latent space, Gaussian Process, Hyperband multi-fidelity scheduling, and Bayesian Optimization.

---

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Data Flow Diagram](#data-flow-diagram)
3. [Component Deep Dives](#component-deep-dives)
   - [GTR Encoder](#1-gtr-encoder)
   - [Structure-Aware VAE](#2-structure-aware-vae)
   - [Set Transformer](#3-set-transformer-exemplar-encoding)
   - [Exemplar Scorer](#4-exemplar-scorer)
   - [Gaussian Process](#5-gaussian-process-gp)
   - [Hyperband](#6-hyperband-multi-fidelity-optimization)
   - [InvBO Inference](#7-invbo-inference-loop)
4. [Training Pipeline](#training-pipeline)
5. [Loss Functions](#loss-functions)
6. [Dimensions Reference](#dimensions-reference)
7. [Parameters Reference](#parameters-reference)
8. [Usage Examples](#usage-examples)

---

## High-Level Overview

LIPO-E optimizes **prompts for few-shot learning** by jointly searching over:
1. **Instruction text** - what the model should do (e.g., "Think step by step...")
2. **Exemplar selection** - which 8 Q/A pairs to include as demonstrations

The key insight is that the optimal exemplars depend on the instruction, and vice versa. LIPO-E learns this joint relationship in a **32-dimensional latent space** and uses **Bayesian Optimization** to find the best combination.

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
  └── VAE learns to encode instructions + exemplars to 32D latent
  └── GP learns error_rate = f(z_joint) mapping
  └── ExemplarScorer learns which exemplars work with which instructions

Stage 4: InvBO Inference
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
                    │  z = [z_inst ‖ z_ex] (32D)    │
                    │  z = z + MLP(z)               │
                    └───────────┬───────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ InstructionDec  │ │  GP Model   │ │ ExemplarScorer  │
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

The VAE compresses high-dimensional embeddings (768D) to a compact latent space (32D) that captures the essential structure for optimization.

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
                    │       z_ex (16D)                    │
                    │            │                        │
                    │  ┌─────────┴───────────┐            │
                    │  │  Joint Refinement   │            │
                    │  │  z = [z_inst‖z_ex]  │            │
                    │  │  z = z + MLP(z)     │            │
                    │  └─────────┬───────────┘            │
                    │            │                        │
                    │       z_joint (32D)                 │
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
z_joint = vae.encode_joint(inst_emb, ex_embs, ex_mask)  # (batch, 32)

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
       │  │       Linear heads (64→16)           │  │
       │  │  fc_mu, fc_logvar → VAE params       │  │
       │  └──────────────────────────────────────┘  │
       │                                            │
       └────────────────────────────────────────────┘

Output: μ_ex (batch, 16), logvar_ex (batch, 16)
```

**Key Insight:** ISAB (Induced Set Attention Block) reduces attention complexity from O(K²) to O(KM) where M=4 inducing points. This allows efficient processing of large exemplar sets.

**Why Set Transformer?**
- Permutation invariance: order doesn't matter
- Handles variable-length inputs (0-8 exemplars)
- Captures interactions between exemplars
- Produces fixed-size output for VAE latent

---

### 4. Exemplar Scorer

**File:** `encoder.py:157-266`

The ExemplarScorer learns which exemplars work well with which instructions. Given the joint latent (z_inst, z_ex), it scores all exemplars in the pool and selects the top-8.

```
                    ┌─────────────────────────────────────┐
                    │          ExemplarScorer             │
                    ├─────────────────────────────────────┤
                    │                                     │
  z_inst (16D) ─────┤  ┌─────────────────┐                │
                    │  │ latent_proj     │                │
  z_ex (16D) ───────┤  │ concat→64D      │                │
                    │  │ Linear+GELU+LN  │                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │           ▼                         │
                    │       z_proj (64D)                  │
                    │           │                         │
  pool_emb          │  ┌────────┴────────┐                │
  (N×768D) ─────────┤  │ pool_proj       │                │
                    │  │ 768→64          │                │
                    │  │ Linear+GELU+LN  │                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │           ▼                         │
                    │      p_proj (N×64D)                 │
                    │           │                         │
                    │  ┌────────┴────────┐                │
                    │  │ Pairwise concat │                │
                    │  │ [z_proj, p_proj]│                │
                    │  │ → (batch,N,128) │                │
                    │  └────────┬────────┘                │
                    │           │                         │
                    │  ┌────────┴────────┐                │
                    │  │ Scorer MLP      │                │
                    │  │ 128→128→64→1    │                │
                    │  │ → scores (N,)   │                │
                    │  └────────┬────────┘                │
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
The scorer is trained with **Binary Cross-Entropy** loss, where targets come from Hyperband's best-performing exemplar sets:

```python
# For each instruction, find best exemplar set → binary mask
targets = compute_selection_targets(design_data, n_pool)
# targets[inst_id] = [0,0,1,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,...]
#                     ^ these 8 positions are 1.0 (best set)

# BCE loss trains scorer to predict high scores for good exemplars
loss = F.binary_cross_entropy_with_logits(scores, target_mask)
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
                    │  Kernel (Product Structure):        │
                    │    k(z,z') = k_inst × k_ex          │
                    │                                     │
                    │    k_inst: Matérn 5/2, ARD (16 dim) │
                    │    k_ex:   Matérn 5/2, ARD (16 dim) │
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

**Product Kernel Structure:**
The kernel separates instruction and exemplar subspaces, allowing them to have different smoothness properties:

```python
# Instruction kernel: dims 0-15
k_inst = MaternKernel(nu=2.5, ard_num_dims=16, active_dims=[0:16])

# Exemplar kernel: dims 16-31
k_ex = MaternKernel(nu=2.5, ard_num_dims=16, active_dims=[16:32])

# Combined (multiplicative)
k(z, z') = k_inst(z[0:16], z'[0:16]) × k_ex(z[16:32], z'[16:32])
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

### 7. InvBO Inference Loop

**File:** `inference.py`

After Hyperband, InvBO continues optimization by generating novel prompts via latent space optimization and Vec2Text inversion.

```
                    ┌─────────────────────────────────────┐
                    │         INVBO INFERENCE             │
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
                    │       z_ex = z_opt[16:32]           │
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
hyperband = LIPOEHyperband(
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

gp = GPWithEI(instruction_dim=16, exemplar_dim=16)
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

### 3. Selection Loss (Exemplar Scorer)

```python
# Binary cross-entropy on pool-wide scores
L_selection = BCE(scores, target_mask)

# target_mask[i] = 1 if exemplar i is in best set for this instruction
# Encourages scorer to give high scores to good exemplars
```

---

## Dimensions Reference

| Component | Dimension | Description |
|-----------|-----------|-------------|
| GTR embedding | 768 | SentenceTransformer output |
| Instruction latent (z_inst) | 16 | VAE encoder output |
| Exemplar latent (z_ex) | 16 | Set Transformer output |
| Joint latent (z_joint) | 32 | Concatenation for GP |
| ISAB hidden | 128 → 64 | Set Transformer internal |
| Scorer projection | 64 | Latent → score space |
| Scorer hidden | 128 | MLP middle layer |
| Number of exemplars | 8 | Fixed K=8 always |
| Exemplar pool | 6154 | Q/A pairs from train.json |
| Instructions | 2000 | APE-generated candidates |
| Validation set | 1319 | GSM8K validation |
| Inducing points | 4 | ISAB efficiency |

---

## Parameters Reference

```python
@dataclass
class LIPOEConfig:
    # === Latent Dimensions ===
    embedding_dim: int = 768
    instruction_latent_dim: int = 16
    exemplar_latent_dim: int = 16
    # total_latent_dim = 32

    # === Set Transformer ===
    set_transformer_hidden: int = 128
    set_transformer_heads: int = 4
    num_inducing_points: int = 4

    # === Exemplar Selection ===
    num_exemplars: int = 8        # Fixed K=8
    scorer_hidden_dim: int = 128

    # === VAE Training ===
    vae_beta: float = 0.005       # KL weight
    vae_mse_weight: float = 0.2   # 20% MSE + 80% cosine
    selection_weight: float = 1.0
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

    # === Inference ===
    iterations: int = 50
    num_restarts: int = 64       # L-BFGS restarts
    raw_samples: int = 4096
    acquisition_type: str = "ucb"  # or "logei"
    ucb_beta: float = 8.0        # Initial exploration
    ucb_beta_final: float = 2.0  # After decay
    cosine_sim_threshold: float = 0.90
    max_rejection_attempts: int = 10

    # === Vec2Text ===
    vec2text_beam: int = 8
    vec2text_model: str = "32_tokens"
    vec2text_max_length: int = 128
```

---

## Usage Examples

### Full Pipeline

```bash
# Full run: APE → Hyperband → VAE → GP → InvBO
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

## Expected Results

Based on LIPO baseline (instruction-only):
- Instruction-only accuracy: ~82-85%
- With 8 exemplars: expected +2-5% improvement
- **Target: ~87-90% accuracy**

Typical run statistics:
- Hyperband evaluations: ~200-500 unique (instruction, exemplar_set) pairs
- Total LLM calls: ~5,000-10,000 (with fidelity extension)
- Runtime: 2-4 hours on single L40S GPU
