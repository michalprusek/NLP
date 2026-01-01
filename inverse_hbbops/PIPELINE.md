# Inverse HbBoPs Pipeline Documentation

## Overview

**Inverse HbBoPs** is a VAE-based prompt optimization system that combines:
- **Automatic Prompt Engineering (APE)** for instruction generation
- **Variational Autoencoder (VAE)** for smooth latent space representation
- **Hyperband** for multi-fidelity Bayesian optimization
- **InvBO Inference** with Vec2Text inversion for novel instruction generation

The key innovation is **instruction-only optimization** (no exemplars), enabling efficient search in a 10-dimensional latent space while maintaining Vec2Text compatibility for text generation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              run.py (Entry Point)                           │
│    Orchestrates: APE Generation → VAE Training → Hyperband → InvBO         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  GRID LOADING   │      │ STANDARD MODE   │      │  INFERENCE      │
    │  (Pre-evaluated)│      │ (Full Pipeline) │      │  (InvBO Loop)   │
    └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
             │                        │                        │
             │   ┌────────────────────┴────────────────────┐   │
             │   │                                         │   │
             ▼   ▼                                         ▼   ▼
    ┌─────────────────┐                           ┌─────────────────┐
    │   APE GENERATOR │                           │   Vec2Text      │
    │   (training.py) │                           │   INVERTER      │
    │   5 Style Types │                           │   (inference.py)│
    └────────┬────────┘                           └────────┬────────┘
             │                                             │
             ▼                                             ▼
    ┌─────────────────┐      ┌─────────────────┐  ┌─────────────────┐
    │ GTR ENCODER     │      │  INSTRUCTION    │  │  INVERSION      │
    │ (encoder.py)    │◄────►│  VAE            │  │  LOOP           │
    │ 768D Embeddings │      │  (encoder.py)   │  │  (inference.py) │
    └────────┬────────┘      └────────┬────────┘  └────────┬────────┘
             │                        │                    │
             │         ┌──────────────┴──────────────┐     │
             │         │                             │     │
             ▼         ▼                             ▼     ▼
    ┌─────────────────────────┐          ┌─────────────────────────┐
    │      HYPERBAND          │          │    BoTorch ACQUISITION  │
    │    (hyperband.py)       │          │    (botorch_acq.py)     │
    │  Successive Halving +   │          │    CompositeLogEI +     │
    │  GP-guided Selection    │          │    L-BFGS-B Optimizer   │
    └────────────┬────────────┘          └────────────┬────────────┘
                 │                                    │
                 └──────────────┬─────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    GAUSSIAN PROCESS     │
                    │       (gp.py)           │
                    │  Deep Kernel GP +       │
                    │  Numerically Stable     │
                    │  LogEI Acquisition      │
                    └─────────────────────────┘
```

---

## Module Reference

### File Structure

```
inverse_hbbops/
├── __init__.py           # Package initialization
├── run.py                # CLI entry point (580 lines)
├── training.py           # APE + VAE + Hyperband training (986 lines)
├── inference.py          # InvBO inference loop (753 lines)
├── gp.py                 # Gaussian Process + LogEI (692 lines)
├── hyperband.py          # Multi-fidelity BO (442 lines)
├── encoder.py            # GTR + VAE encoders (353 lines)
├── botorch_acq.py        # BoTorch acquisition (246 lines)
├── evaluate.py           # GSM8K evaluation (243 lines)
├── instruction.py        # Prompt dataclass (43 lines)
├── data/                 # Cached APE instructions
│   └── ape_instructions.json
└── results/              # Experiment outputs (gitignored)
```

**Total: ~4,388 lines of Python code**

---

## 1. encoder.py - Embedding and Latent Space

### GTRInstructionEncoder (Lines 17-92)

SentenceTransformer wrapper for Vec2Text-compatible embeddings.

```
Input: Text string
  ↓
GTR-T5-Base (sentence-transformers/gtr-t5-base)
  ↓
L2 Normalization (required for Vec2Text)
  ↓
Output: 768D embedding tensor
```

**Key Methods:**

| Method | Lines | Signature | Description |
|--------|-------|-----------|-------------|
| `encode` | 55-61 | `(text: str) → np.ndarray` | Single text to L2-normalized embedding |
| `encode_tensor` | 63-70 | `(text: str) → torch.Tensor` | Single text to tensor on device |
| `encode_batch` | 72-80 | `(texts: List[str]) → np.ndarray` | Batch encode with progress bar |
| `encode_batch_tensor` | 82-92 | `(texts: List[str]) → torch.Tensor` | Batch encode to tensor |

**Design Decisions:**
- **L2 normalization** (lines 27-35, 60, 78): Required for Vec2Text compatibility
- **Auto device detection** (lines 46-53): CUDA → MPS → CPU fallback
- **Lazy loading**: Model loaded on first use

---

### InstructionVAE (Lines 95-323)

Variational Autoencoder for smooth latent space representation.

```
Encoder Architecture:
  768D → Linear(64) + ReLU + LayerNorm
       → Linear(32) + ReLU + LayerNorm
       → Linear(2 × latent_dim)  # Output: μ and log_var

Decoder Architecture:
  latent_dim → Linear(32) + ReLU + LayerNorm
             → Linear(64) + ReLU + LayerNorm
             → Linear(256) + ReLU + LayerNorm
             → Linear(768)
             → L2-normalize  # Match GTR embedding space
```

**Loss Function (Lines 234-277):**

```
L_total = L_recon + β × L_KL

where:
  L_recon = 1 - cosine_similarity(x, x_recon)  # Cosine distance
  L_KL = -0.5 × Σ(1 + log_var - μ² - exp(log_var))  # Standard VAE KL
```

**Key Methods:**

| Method | Lines | Description |
|--------|-------|-------------|
| `encode` | 161-181 | Map 768D embedding to (μ, log_var) |
| `reparameterize` | 183-195 | Sample z from N(μ, σ²) with gradient |
| `decode` | 197-216 | Map latent z back to L2-normalized 768D |
| `forward` | 218-232 | Full VAE forward: encode → sample → decode |
| `loss` | 234-277 | Compute reconstruction + KL loss |
| `encode_mu` | 279-289 | Deterministic encoding (for GP inference) |

**Design Decisions:**
- **LayerNorm over BatchNorm** (lines 140-157): Stable with batch_size=1
- **Cosine loss** (line 260-261): Aligned with L2-normalized GTR embeddings
- **L2-normalized decoder output** (line 213): Matches embedding space geometry

---

### VAEWithAdapter (Lines 325-354)

Feature extractor for GP with frozen VAE.

```
Architecture:
  768D → [Frozen VAE Encoder] → μ
       → [Trainable Adapter MLP] → 768D
       → [VAE encode_mu] → 10D latent

Adapter MLP:
  768 → Linear(768×2) → ReLU → LayerNorm → Linear(768)
```

**Purpose:** Allows GP to learn on top of frozen VAE features while preserving pre-trained representations.

---

## 2. instruction.py - Prompt Dataclass

### InstructionOnlyPrompt (Lines 10-43)

Minimal dataclass for instruction-only prompts (no exemplars).

```python
@dataclass
class InstructionOnlyPrompt:
    instruction: str      # The instruction text
    instruction_id: int   # Unique identifier
```

**Q_end Format (Lines 24-35):**

```
Q: {question}
{instruction}
A:
```

This format places the instruction AFTER the question, following the OPRO paper convention.

---

## 3. evaluate.py - GSM8K Evaluation

### Answer Extraction (Lines 12-64)

Robust answer extraction supporting multiple formats.

| Function | Lines | Description |
|----------|-------|-------------|
| `extract_answer` | 12-39 | Extract **last number** from model output |
| `extract_gold_answer` | 42-64 | Extract gold answer (#### format or last number) |

**Extraction Pattern (Line 27):**
```python
r'[-+]?\d+(?:[.,]\d+)?'  # Matches integers, decimals, commas
```

Returns **last match only** for robustness to chain-of-thought outputs.

---

### GSM8KEvaluator (Lines 67-243)

Stateful evaluator with LLM call tracking.

```python
GSM8KEvaluator(
    model="Qwen/Qwen2.5-7B-Instruct",  # Evaluation model
    backend="vllm",                     # LLM backend
    max_tokens=512,
    temperature=0.0                     # Greedy decoding
)
```

**Key Methods:**

| Method | Lines | Description |
|--------|-------|-------------|
| `format_prompt` | 106-116 | Create Q_end format prompt |
| `evaluate_single` | 118-151 | Evaluate one Q/A pair, track calls |
| `evaluate_batch` | 153-201 | Batch evaluation with error rate |
| `__call__` | 203-217 | Callable interface for Hyperband |

**Call Tracking:**
```python
self.total_calls += 1          # Single evaluation (line 143)
self.total_calls += len(prompts)  # Batch evaluation (line 186)
```

---

## 4. gp.py - Gaussian Process with LogEI

### Numerically Stable LogEI (Lines 26-125)

Three-branch implementation for numerical stability across all z values.

**The Problem:**
Standard EI computation fails for very negative z (tiny improvements):
```
EI = (y_best - μ) × Φ(z) + σ × φ(z)
   ≈ 10^-300  # Underflows to 0
```

**The Solution: LogEI with 3 branches**

```
log_h(z) where h(z) = φ(z) + z × Φ(z)

Branch 1 (z > -1): Direct computation
  log_h = log(φ(z) + z × Φ(z))

Branch 2 (-1/√ε < z ≤ -1): erfcx-based
  log_h = -z²/2 - log(2π)/2 + log(erfcx(-z/√2) × |z|)

Branch 3 (z ≤ -1/√ε): Asymptotic approximation
  log_h ≈ -z²/2 - log(2π)/2 - 2 × log(|z|)
```

**Functions:**

| Function | Lines | Description |
|----------|-------|-------------|
| `log_h(z)` | 50-85 | Scalar LogEI (numpy) |
| `log_h_tensor(z)` | 88-124 | Differentiable LogEI (torch autograd) |

---

### InstructionDeepKernelGP (Lines 127-195)

Deep kernel GP with ARD Matern 5/2.

```
Architecture:
  768D embedding (normalized)
    ↓
  VAEWithAdapter (feature extractor)
    ↓
  10D latent features
    ↓
  ARD Matern 5/2 Kernel (per-dimension lengthscales)
    ↓
  GP posterior: N(μ(x), K(x, x'))
```

**Kernel Configuration (Lines 166-173):**
```python
kernel = ScaleKernel(
    MaternKernel(nu=2.5, ard_num_dims=10),
    outputscale_prior=GammaPrior(2.0, 0.15)
)
kernel.base_kernel.lengthscale_prior = GammaPrior(3.0, 6.0)
```

---

### GPWithEI (Lines 198-692)

Full GP wrapper with training, prediction, and acquisition.

**Training Flow (Lines 255-366):**

```
1. Data Normalization
   ├─ Inputs: Unit cube normalization
   │    X_norm = (X - X_min) / (X_max - X_min)
   │
   └─ Outputs: Standardization (BoTorch)
        y_norm = (y - y_mean) / y_std

2. Model Setup
   ├─ GaussianLikelihood(noise_constraint=Interval(0.001, 0.1))
   └─ InstructionDeepKernelGP(X_norm, y_norm, likelihood, feature_extractor)

3. Training Loop
   ├─ Optimizer: AdamW(lr=0.01)
   ├─ Loss: -ExactMarginalLogLikelihood
   ├─ Cholesky jitter: 1e-4
   └─ Early stopping: patience=10 epochs
```

**Key Methods:**

| Method | Lines | Description |
|--------|-------|-------------|
| `train` | 255-366 | Full GP training with normalization |
| `predict` | 368-406 | Denormalized (mean, std) prediction |
| `expected_improvement` | 408-437 | Standard EI formula |
| `log_expected_improvement` | 439-471 | Stable LogEI computation |
| `log_expected_improvement_tensor` | 473-523 | Differentiable LogEI for BoTorch |
| `add_observation_and_retrain` | 552-688 | Incremental retraining with warm-start |

**Noise Constraint (Line 303):**
```python
noise_constraint = Interval(0.001, 0.1)
```
- Too low (< 0.001): GP over-confident, poor exploration
- Too high (> 0.1): GP underfits, ignores data
- [0.001, 0.1]: Balanced between fit and regularization

---

## 5. hyperband.py - Multi-Fidelity Optimization

### HyperbandConfig (Lines 28-38)

```python
@dataclass
class HyperbandConfig:
    bmin: int = 10                      # Minimum fidelity (samples)
    eta: float = 2.0                    # Downsampling rate
    random_interleaving_prob: float = 0.1  # 10% random exploration
    top_fidelity_pct: float = 0.75      # Train GP on top 75% fidelities
    gp_epochs: int = 3000
    gp_lr: float = 0.01
    gp_patience: int = 10
```

---

### InverseHbBoPs (Lines 45-442)

Multi-fidelity successive halving with GP-guided selection.

**Schedule Computation (Lines 107-112):**
```python
r = nvalid / bmin          # Ratio: 1319 / 10 = 131.9
smax = floor(log(r) / log(eta))  # Max stages: floor(log₂(131.9)) = 7
B = (smax + 1) * nvalid    # Total budget: 8 × 1319 = 10552
```

**Successive Halving Algorithm:**

```
For each bracket s = smax, smax-1, ..., 0:
    n = ceil(B / (s+1) / r)     # Initial number of prompts
    b = r × η^(-s)              # Initial fidelity

    For each stage i = 0, 1, ..., s:
        nᵢ = floor(n × η^(-i))  # Prompts remaining
        bᵢ = b × η^i            # Current fidelity

        Evaluate nᵢ prompts at fidelity bᵢ
        Train GP on top-75% fidelity data
        Keep top nᵢ₊₁ = floor(nᵢ / η) prompts
```

**Evaluation Caching (Lines 148-177):**

Smart caching with fidelity extension:
```python
# If we have cached result at fidelity f' < f:
prev_error = cache[(inst_id, f')]
remaining = validation_data[f':f]
new_error = evaluate(inst, remaining)
total_error = (prev_error × f' + new_error × (f - f')) / f

# Saves ~50% LLM calls on successive fidelities
```

**GP-Guided Selection (Lines 287-299):**
```python
# 10% random exploration
if random() < 0.1:
    return random_unevaluated_prompt

# 90% BO selection: maximize EI
return argmax(expected_improvement(prompt) for prompt in unevaluated)
```

---

## 6. training.py - Training Pipeline

### TrainingConfig (Lines 28-67)

Central configuration dataclass.

```python
@dataclass
class TrainingConfig:
    # APE
    ape_model: str = "haiku"
    ape_num_instructions: int = 1000
    ape_cache_path: str = "inverse_hbbops/data/ape_instructions.json"

    # VAE
    vae_beta: float = 0.02
    vae_epochs: int = 10000
    vae_annealing_epochs: int = 500
    vae_latent_dim: int = 10

    # Hyperband
    bmin: int = 10
    eta: float = 2.0

    # GP
    gp_epochs: int = 10000
    gp_lr: float = 0.01
    gp_patience: int = 100
```

---

### APEGenerator (Lines 70-260)

Style-constrained instruction generation.

**Style Categories (Lines 76-114):**

| Style | Description | Temperature | Example |
|-------|-------------|-------------|---------|
| `minimalist` | Ultra-short (1-5 words) | 1.0 | "Solve step by step" |
| `direct_command` | Commands (6-15 words) | 0.9 | "Calculate the answer carefully" |
| `chain_of_thought` | Step-by-step triggers | 0.8 | "Let's think step by step..." |
| `pedagogical` | Teacher persona | 0.9 | "As a math tutor, guide..." |
| `analytical` | Logical structure | 0.8 | "Analyze the problem..." |

**Generation Pipeline (Lines 158-201):**
```
1. For each style (5 styles):
   - Target: num_instructions / 5 per style
   - Batch size: 10 instructions per LLM call
   - Deduplicate within style

2. Combine all styles
3. Deduplicate globally
4. Return first num_instructions
```

---

### InverseHbBoPsTrainer (Lines 263-986)

Full training pipeline orchestrator.

**Pipeline Flow:**

```
load_validation_data()          # Load GSM8K validation (1319 samples)
         │
         ▼
generate_instructions()         # APE generation + GTR encoding
         │
         ▼
train_vae()                     # VAE training with KL annealing
         │
         ▼
run_hyperband()                 # Multi-fidelity optimization
         │
         ▼
get_gp_for_inference()          # Extract trained GP + VAE
```

**KL Annealing (Lines 394-396):**
```python
# Gradually increase β from 0 to vae_beta over annealing epochs
if epoch < vae_annealing_epochs:
    current_beta = vae_beta × (epoch / vae_annealing_epochs)
else:
    current_beta = vae_beta
```

**Key Methods:**

| Method | Lines | Description |
|--------|-------|-------------|
| `load_validation_data` | 308-319 | Load GSM8K from JSON |
| `generate_instructions` | 321-360 | APE + embedding computation |
| `train_vae` | 362-462 | VAE training with annealing |
| `run_hyperband` | 464-507 | Execute Hyperband |
| `load_from_grid` | 617-719 | Load pre-evaluated grid |
| `train_gp_from_grid` | 904-970 | Train GP from grid data |

---

## 7. botorch_acq.py - BoTorch Acquisition

### CompositeLogEI (Lines 26-97)

LogEI acquisition through VAE decoder transformation.

```
Gradient Flow:
  Latent z (10D)
    ↓ [VAE.decode] differentiable
  Embedding (768D)
    ↓ [GP.posterior]
  qLogExpectedImprovement
    ↓
  Acquisition Value (scalar)
```

**Shape Handling (Lines 75-82):**
```python
# Handle various input shapes:
# (d,) → (1, 1, d)
# (q, d) → (1, q, d)
# (batch, q, d) → preserved
```

---

### LatentSpaceAcquisition (Lines 100-246)

Multi-start L-BFGS-B optimizer using BoTorch.

**Optimization (Lines 134-202):**
```python
candidate, acq_value = optimize_acqf(
    acq_function=CompositeLogEI(...),
    bounds=latent_bounds,        # (2, 10) bounds tensor
    q=1,                         # Single point acquisition
    num_restarts=20,             # Multi-start L-BFGS-B
    raw_samples=512,             # Initialization seeding
    options={"maxiter": 200, "batch_limit": 5},
)
```

**Latent Bounds (Lines 205-246):**
```python
# Compute bounds from training data with exploration margin
bounds = [
    min(latent_values) - 0.2 × range,
    max(latent_values) + 0.2 × range
]
```

---

## 8. inference.py - InvBO Inference

### Vec2TextInverter (Lines 60-234)

Embedding-to-text conversion using Vec2Text models.

**Models:**

| Model Type | Lines | Token Limit | Description |
|------------|-------|-------------|-------------|
| `32_tokens` | 111-146 | 32 | ielabgroup/vec2text_gtr-base-st + corrector |
| `512_tokens` | 148-183 | 512 | vec2text/gtr-512-noise-0.00001 (default) |

**Generation Config (Lines 219-224):**
```python
{
    "num_beams": 8,
    "max_length": 128,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2
}
```

---

### InverseHbBoPsInference (Lines 237-753)

Complete InvBO inference loop.

**Single Iteration Flow (Lines 557-696):**

```
1. OPTIMIZE LATENT (BoTorch)
   ├─ LatentSpaceAcquisition.optimize()
   ├─ CompositeLogEI acquisition
   └─ Returns: z_opt (10D), log_ei

2. DECODE TO EMBEDDING
   └─ embedding = VAE.decode(z_opt)  # 768D

3. INVERT TO TEXT
   └─ instruction = Vec2Text(embedding)  # String

4. INVERSION LOOP (Optional, up to 3 iterations)
   ├─ Problem: VAE.decode(z) ≠ GTR(Vec2Text(VAE.decode(z)))
   ├─ Solution: Find z_inv minimizing gap
   │
   │  For each iteration:
   │    target_emb = GTR(instruction)
   │    z_inv = argmin ||VAE.decode(z) - target_emb||_cosine
   │    gap = cosine_distance(VAE.decode(z_inv), target_emb)
   │
   │    If gap ≤ 0.1: Accept z_inv
   │    Else: Re-decode and re-invert
   │
   └─ Returns: final instruction with gap ≤ threshold

5. RE-ENCODE AND PREDICT
   ├─ reencoded = GTR(instruction)
   ├─ cosine_sim = cos_sim(embedding, reencoded)
   └─ pred_error, pred_std = GP.predict(reencoded)

6. EVALUATE (Optional)
   └─ actual_error = evaluator(instruction, validation_data)

7. UPDATE GP
   ├─ Add observation: (reencoded, error)
   └─ Incremental retrain (warm-start, 500 epochs)
```

**Inversion Step (Lines 386-464):**
```python
# Find z_inv such that GTR(Vec2Text(decode(z_inv))) matches target
z = z_init.clone().requires_grad_(True)
optimizer = Adam([z], lr=0.1)

for step in range(n_steps):
    decoded = vae.decode(z)
    loss = 1 - cosine_similarity(decoded, target_emb)
    loss.backward()
    optimizer.step()

    if loss < convergence_threshold:
        break
```

---

## 9. run.py - CLI Entry Point

### Main Pipeline (Lines 119-474)

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 10 | InvBO iterations |
| `--ape-instructions` | 1000 | APE generation count |
| `--model` | Qwen/Qwen2.5-7B-Instruct | Evaluation model |
| `--vae-beta` | 0.02 | KL regularization weight |
| `--vae-epochs` | 10000 | Max VAE training epochs |
| `--vae-annealing` | 500 | KL annealing epochs |
| `--bmin` | 10 | Hyperband minimum fidelity |
| `--eta` | 2.0 | Hyperband downsampling rate |
| `--vec2text-model` | 512_tokens | Vec2Text model type |
| `--load-grid` | False | Load pre-evaluated grid |
| `--top-k` | None | Load top-k from grid |
| `--skip-ape` | False | Skip APE generation |
| `--skip-eval` | False | Skip LLM evaluation |

**Execution Branches:**

```
if args.load_grid:
    if args.validate_hyperband:
        # Run Hyperband on all grid instructions
        load_from_grid(top_k=None)
        run_hyperband(evaluator)
        validate_against_ground_truth()
    else:
        # Load top-k, train GP directly (NO Hyperband)
        load_from_grid(top_k=args.top_k)
        train_gp_from_grid()
else:
    # Full pipeline
    generate_instructions()  # or load from cache
    train_vae()
    run_hyperband(evaluator)

# InvBO inference (always runs)
inference.run(iterations=args.iterations)
```

---

## Complete Execution Trace

### Example: Full Pipeline with 10 Iterations

```
run.py:main()
│
├── TrainingConfig(ape_instructions=1000, vae_beta=0.02, ...)
│
├── InverseHbBoPsTrainer(config)
│   ├── GTRInstructionEncoder.load()    # Lazy load GTR-T5-Base
│   └── Ready for training
│
├── GSM8KEvaluator(model="Qwen/Qwen2.5-7B-Instruct")
│
├── trainer.load_validation_data()      # Load 1319 GSM8K samples
│
├── trainer.generate_instructions(num=1000)
│   ├── APEGenerator.generate_or_load()
│   │   ├── Check cache: inverse_hbbops/data/ape_instructions.json
│   │   │
│   │   └── If not cached:
│   │       ├── For each style (5 styles):
│   │       │   └── Generate ~200 instructions
│   │       │       (APE LLM calls: ~100-200 total)
│   │       └── Deduplicate and save to cache
│   │
│   └── Pre-compute GTR embeddings for all 1000 instructions
│       └── GTR batch encode (~30 calls)
│
├── trainer.train_vae()
│   ├── Create InstructionVAE(768D → 10D)
│   │
│   ├── For epoch 1 to 10000 (or early stop):
│   │   ├── KL annealing:
│   │   │   β = 0.02 × min(1.0, epoch / 500)
│   │   │
│   │   ├── For each batch (64 samples):
│   │   │   ├── Forward: x → encode → (μ, log_var)
│   │   │   ├── Reparameterize: z = μ + σ × ε
│   │   │   ├── Decode: z → x_recon
│   │   │   ├── Loss: recon + β × kl
│   │   │   └── Backward + optimizer step
│   │   │
│   │   └── CosineAnnealing LR scheduler
│   │
│   └── Output: Trained VAE + VAEWithAdapter
│
├── trainer.run_hyperband(evaluator)
│   └── InverseHbBoPs(prompts=1000, validation_data=1319)
│       │
│       ├── Pre-compute embeddings for all 1000 instructions
│       │
│       ├── Calculate schedule:
│       │   r = 1319 / 10 = 131.9
│       │   smax = floor(log₂(131.9)) = 7
│       │   B = 8 × 1319 = 10552
│       │
│       ├── For bracket s = 7, 6, 5, 4, 3, 2, 1, 0:
│       │   ├── n = ceil(B / (s+1) / r)
│       │   ├── b = r × 2^(-s)
│       │   │
│       │   └── For stage i = 0, 1, ..., s:
│       │       ├── Evaluate nᵢ prompts at fidelity bᵢ
│       │       │   └── LLM calls: nᵢ × bᵢ (with caching)
│       │       │
│       │       ├── Train GP on top-75% fidelity data
│       │       │
│       │       └── Select top nᵢ₊₁ by GP prediction
│       │           (10% random for exploration)
│       │
│       ├── Total Hyperband LLM calls: ~500-1000
│       │
│       └── Return: (best_prompt, best_error)
│
├── gp = trainer.get_gp_for_inference()
│   └── GPWithEI from Hyperband's final state
│
├── InverseHbBoPsInference(gp, vae, gtr, evaluator)
│   ├── Load Vec2TextInverter(model_type="512_tokens")
│   └── Initialize tracking: best_error, iteration_history
│
├── inference.run(iterations=10)
│   │
│   └── For iteration = 1 to 10:
│       │
│       ├── optimize_latent_botorch(num_restarts=20, raw_samples=512)
│       │   ├── Create CompositeLogEI acquisition
│       │   │
│       │   └── botorch.optimize_acqf()
│       │       ├── Sample 512 random latents
│       │       ├── Evaluate: z → VAE.decode → GP → LogEI
│       │       ├── Top 20 → L-BFGS-B refinement (200 iters each)
│       │       └── Return: z_opt, log_ei
│       │
│       ├── Decode: embedding = VAE.decode(z_opt)  # 768D
│       │
│       ├── Invert: instruction = Vec2Text(embedding)
│       │
│       ├── Inversion loop (up to 3 iterations):
│       │   ├── target = GTR(instruction)
│       │   ├── z_inv = optimize(||VAE.decode(z) - target||)
│       │   ├── gap = cosine_distance
│       │   │
│       │   └── If gap > 0.1: Re-decode, re-invert
│       │
│       ├── Re-encode: reencoded = GTR(instruction)
│       │
│       ├── Predict: (mean, std) = GP.predict(reencoded)
│       │
│       ├── Evaluate: actual_error = evaluator(instruction)
│       │   └── LLM calls: 1319 (full validation set)
│       │
│       ├── Update best if actual_error < best_error
│       │
│       └── GP.add_observation_and_retrain(reencoded, actual_error)
│           └── Incremental retrain: 500 epochs, warm-start
│
├── Total LLM calls:
│   ├── APE generation: ~100-200
│   ├── Hyperband: ~500-1000
│   └── InvBO (10 iters): 10 × 1319 = 13190
│
└── _save_results()
    └── results/result_{timestamp}.json
```

---

## Hyperparameter Reference

### VAE

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `vae_beta` | 0.02 | KL regularization weight | Higher = smoother latent, less reconstruction |
| `vae_epochs` | 10000 | Max training epochs | Usually early-stops around 500-1000 |
| `vae_annealing_epochs` | 500 | KL warm-up duration | Prevents posterior collapse |
| `vae_latent_dim` | 10 | Latent space dimension | Trade-off: expressiveness vs tractability |
| `vae_lr` | 0.001 | Learning rate | Standard for Adam |
| `vae_batch_size` | 64 | Training batch size | Stable with LayerNorm |

### Gaussian Process

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `noise_constraint` | [0.001, 0.1] | Observation noise bounds | Balance confidence vs uncertainty |
| `gp_lr` | 0.01 | Learning rate | Standard for deep kernel GP |
| `gp_patience` | 100 | Early stopping patience | Prevents overfitting |
| `gp_epochs` | 10000 | Max training epochs | Usually early-stops |
| `kernel` | Matern 5/2 | Kernel function | Smooth with flexibility |
| `ard_num_dims` | 10 | ARD dimensions | Learns importance per latent dim |

### Hyperband

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `bmin` | 10 | Minimum fidelity (samples) | Lower = faster but noisier |
| `eta` | 2.0 | Downsampling rate | 2.0 = halving each stage |
| `random_interleaving_prob` | 0.1 | Random exploration rate | Mitigates GP bias |
| `top_fidelity_pct` | 0.75 | GP training fidelity threshold | Use top 75% fidelities |

### BoTorch Acquisition

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `num_restarts` | 20 | L-BFGS-B restarts | More = better optimization |
| `raw_samples` | 512 | Initialization samples | More = better coverage |
| `maxiter` | 200 | L-BFGS-B iterations | Sufficient for 10D |
| `batch_limit` | 5 | Parallel L-BFGS-B | Memory trade-off |

### InvBO Inference

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `iterations` | 10 | InvBO iterations | More = more exploration |
| `gap_threshold` | 0.1 | Inversion tolerance | Cosine distance acceptance |
| `max_inversion_iters` | 3 | Max inversion loop | Trade-off: quality vs speed |
| `vec2text_model` | 512_tokens | Vec2Text variant | 512 = more exploratory |

---

## Key Design Decisions

### 1. Instruction-Only (No Exemplars)

Unlike standard HbBoPs which optimizes (Instruction × Exemplar) pairs:
- Simplifies search space from (I × E) to just I
- Enables pure latent space optimization
- Better suited for instruction transfer

### 2. GTR + VAE + Vec2Text Pipeline

```
GTR (768D) → VAE (10D) → Vec2Text (text)
```

- GTR provides Vec2Text-compatible embeddings
- VAE creates smooth 10D latent for optimization
- Vec2Text enables novel text generation

### 3. LogEI for Numerical Stability

Standard EI underflows for tiny improvements. LogEI uses 3 computational branches to handle the entire z-value range without numerical issues.

### 4. Fidelity Extension in Hyperband

Reuses lower-fidelity evaluations:
```
error[f] = (error[f'] × f' + error[f'-f] × (f - f')) / f
```
Saves ~50% LLM calls on successive fidelities.

### 5. Inversion Loop for VAE↔Vec2Text Alignment

VAE decoder output ≠ Vec2Text reconstruction. The inversion loop finds `z_inv` that minimizes this gap:
```
z_inv = argmin ||VAE.decode(z) - GTR(Vec2Text(VAE.decode(z)))||_cosine
```

### 6. Incremental GP Retraining

Each InvBO iteration adds one observation and retrains with:
- Preserved input normalization (X_min, X_max)
- Recomputed output normalization
- Warm-start from previous kernel hyperparameters
- Reduced epochs (500 vs 10000) for efficiency

---

## Data Formats

### Validation Data (JSON)

```json
[
    {"question": "Janet's ducks lay 16 eggs per day...", "answer": "#### 64"},
    ...
]
```

### APE Cache (JSON)

```json
{
    "instructions": [
        "Solve step by step",
        "Let's think through this carefully...",
        ...
    ]
}
```

### Grid Data (JSONL)

```json
{"instruction_text": "...", "exemplar_text": "...", "error_rate": 0.25, ...}
{"instruction_text": "...", "exemplar_text": "...", "error_rate": 0.30, ...}
```

### Results (JSON)

```json
{
    "best_instruction": "...",
    "best_error_rate": 0.15,
    "total_llm_calls": 15000,
    "iteration_history": [...],
    "hyperparameters": {...}
}
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| VAE posterior collapse | β too high early | Increase annealing epochs |
| GP over-confident | Noise constraint too low | Increase lower bound |
| Poor BoTorch convergence | Latent bounds too tight | Increase margin |
| Vec2Text truncation | Token limit hit | Use 512_tokens model |
| Inversion loop stuck | Gap threshold too low | Increase to 0.15 |

### Debugging Commands

```bash
# Quick debug run (10 instructions, 2 iterations)
uv run python -m inverse_hbbops.run --debug --iterations 2

# Skip APE (use cached)
uv run python -m inverse_hbbops.run --skip-ape

# Load from pre-evaluated grid
uv run python -m inverse_hbbops.run --load-grid --top-k 25

# Skip LLM evaluation (GP predictions only)
uv run python -m inverse_hbbops.run --skip-eval
```

---

## References

- **HbBoPs**: Hyperband-based Bayesian Optimization for Prompt Selection
- **Vec2Text**: Text embeddings reveal (almost) as much as text (Morris et al., 2023)
- **BoTorch**: Bayesian optimization in PyTorch
- **GPyTorch**: Gaussian processes in PyTorch
