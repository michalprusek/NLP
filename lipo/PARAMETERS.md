# LIPO Parameters Reference

Complete reference for all configurable hyperparameters in the LIPO (Latent Instruction Prompt Optimization) pipeline.

**Single Source of Truth**: All parameters are defined in `config.py` with CLI overrides available in `run.py`.

---

## Table of Contents

1. [APE Generation](#1-ape-generation)
2. [VAE Training](#2-vae-training)
3. [Latent Dimensions](#3-latent-dimensions)
4. [Round-Trip Validation](#4-round-trip-validation)
5. [Hyperband Multi-Fidelity](#5-hyperband-multi-fidelity)
6. [GP Training](#6-gp-training)
7. [GP Retrain During Inference](#7-gp-retrain-during-inference)
8. [Inference / BoTorch Optimization](#8-inference--botorch-optimization)
9. [Vec2Text Settings](#9-vec2text-settings)
10. [TuRBO Trust Region](#10-turbo-trust-region)
11. [PAS (Potential-Aware Anchor Selection)](#11-pas-potential-aware-anchor-selection)
12. [Latent Space](#12-latent-space)
13. [Device & Paths](#13-device--paths)
14. [CLI-Only Arguments](#14-cli-only-arguments)
16. [Architecture Details](#architecture-details)
17. [Critical Thresholds Summary](#critical-thresholds-summary)

---

## 1. APE Generation

Automatic Prompt Enumeration (APE) generates diverse instruction candidates for VAE training.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `ape_num_instructions` | 2000 | int | `--ape-instructions` | Number of instructions to generate |
| `ape_model` | "Qwen/Qwen2.5-7B-Instruct" | str | `--ape-model` | LLM model for APE generation |
| `ape_backend` | "vllm" | str | - | Backend for APE (vllm/openai/deepinfra) |
| `ape_cache_path` | "lipo/data/ape_instructions.json" | str | `--ape-cache` | Cache path for generated instructions |
| `ape_batch_size` | 10 | int | - | Batch size for LLM generation |
| `ape_max_tokens` | 100 | int | - | Max tokens per generated instruction |
| `ape_max_length` | 500 | int | - | Max character length for valid instructions |

**Purpose**: APE creates a diverse pool of instruction candidates using 8 different style templates (minimalist, direct command, methodical, chain-of-thought, socratic, formal academic, conversational, structured). This diversity ensures the VAE learns a smooth latent space covering different instruction styles.

**Impact**: More instructions (higher `ape_num_instructions`) lead to better VAE generalization but longer training time. The style templates ensure latent space coverage of different prompting approaches.

---

## 2. VAE Training

Variational Autoencoder training parameters for learning the instruction latent space.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `vae_beta` | 0.01 | float | `--vae-beta` | KL regularization weight |
| `vae_gamma` | 0.0 | float | `--vae-gamma` | Cycle consistency weight (disabled) |
| `vae_epochs` | 50000 | int | `--vae-epochs` | Maximum training epochs |
| `vae_annealing_epochs` | 2500 | int | `--vae-annealing` | KL annealing warmup period (5% of epochs) |
| `vae_patience` | 1000 | int | - | Early stopping patience (after annealing) |
| `vae_lr` | 0.0006 | float | - | AdamW learning rate |
| `vae_batch_size` | 64 | int | - | Training batch size |
| `vae_grad_clip` | 1.0 | float | - | Gradient clipping threshold |
| `vae_eta_min` | 1e-4 | float | - | Minimum LR for cosine scheduler |

**Purpose**: The VAE compresses 768D GTR embeddings to a 32D latent space while maintaining reconstruction quality. The beta parameter controls the trade-off between reconstruction fidelity and latent space regularity.

**Impact**:
- **Higher `vae_beta`** (e.g., 0.1): Tighter, more regular latent space but potentially worse reconstruction. Better for interpolation and optimization.
- **Lower `vae_beta`** (e.g., 0.001): Better reconstruction but more irregular latent space. May cause optimization difficulties.
- **KL Annealing**: Linear warmup from 0 to `vae_beta` over `vae_annealing_epochs` prevents posterior collapse. Early stopping resets after annealing completes.

**Loss Function**:
```
total_loss = cosine_recon_loss + beta * kl_loss + gamma * cycle_loss
```

---

## 3. Latent Dimensions

Core dimensionality settings for the embedding and latent spaces.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `embedding_dim` | 768 | int | GTR-T5-Base embedding dimension (fixed) |
| `latent_dim` | 32 | int | VAE latent space dimension (24x compression) |

**Purpose**: Defines the compression ratio of the VAE. The 768D GTR embeddings are compressed to 32D latent vectors (24x compression).

**Impact**:
- **Higher `latent_dim`** (e.g., 64): More expressive but harder to optimize with GP, risk of overfitting
- **Lower `latent_dim`** (e.g., 16): Easier optimization but worse reconstruction quality (48x compression loses nuances)

**Architecture Flow**:
```
Encoder: 768D → 256 → 128 → 2×32 (mu + log_var)
Decoder: 32D → 128 → 256 → 768D (L2 normalized)
```

---

## 4. Round-Trip Validation

Quality checks for VAE → Vec2Text → GTR round-trip reconstruction.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `roundtrip_validation_threshold` | 0.90 | float | Minimum acceptable cosine similarity |
| `roundtrip_validation_samples` | 20 | int | Number of samples for validation |

**Purpose**: Validates that the full pipeline (VAE decode → Vec2Text invert → GTR encode) maintains semantic fidelity. Run after VAE training to ensure the system can generate meaningful instructions.

**Impact**: Lower threshold allows more diverse but potentially lower-quality outputs. Higher threshold ensures quality but may reject valid variations.

---

## 5. Hyperband Multi-Fidelity

Multi-fidelity Bayesian optimization using Hyperband scheduling.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `bmin` | 10 | int | `--bmin` | Minimum fidelity (validation samples) |
| `eta` | 2.0 | float | `--eta` | Downsampling/acceleration rate |
| `random_interleaving_prob` | 0.1 | float | - | Probability of random selection in BO |
| `min_fidelity_pct` | 0.75 | float | - | Min fidelity percentage for GP training |

**Purpose**: Hyperband efficiently allocates evaluation budget across instruction candidates. Low-fidelity evaluations (few samples) quickly eliminate bad candidates; promising ones get full evaluation.

**Impact**:
- **Higher `eta`** (e.g., 3.0): More aggressive elimination, faster but may miss good candidates
- **Lower `eta`** (e.g., 1.5): More conservative, thorough but slower
- **`random_interleaving_prob`**: Controls exploration vs exploitation in BO proposals
- **`min_fidelity_pct`**: GP only trains on high-fidelity observations (top 25% by default)

**Schedule Computation**:
```
smax = floor(log_eta(nvalid/bmin))
B = (smax + 1) * nvalid  # Total budget
n = ceil((B/nvalid) * (eta^s) / (s+1))  # Prompts per bracket
b = nvalid * (eta^(-s))  # Initial fidelity per bracket
```

---

## 6. GP Training

Gaussian Process model training parameters for initial fitting.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `gp_epochs` | 10000 | int | `--gp-epochs` | Maximum training epochs |
| `gp_lr` | 0.0025 | float | `--gp-lr` | AdamW learning rate (scaled for 32D) |
| `gp_patience` | 100 | int | `--gp-patience` | Early stopping patience |

**Purpose**: Trains the GP surrogate model on evaluated instructions. The GP models the relationship between 32D VAE latents and instruction accuracy.

**Impact**:
- **Higher `gp_epochs`**: Better convergence but longer training
- **Lower `gp_lr`**: More stable but slower convergence
- Learning rate is automatically scaled by 0.25x for 32D latent (line 446 in gp.py)

**GP Configuration**:
- Kernel: Matern 5/2 with ARD (32 lengthscales)
- Mean: ZeroMean
- Priors: Data-driven Gamma prior for lengthscales (mean=median pairwise distance), GammaPrior(2.0, 2.0) for outputscale
- Noise: FixedNoiseGaussianLikelihood with Beta posterior variance from fidelity

---

## 7. GP Retrain During Inference

Parameters for retraining GP after each new observation during optimization.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `gp_retrain_epochs` | 1000 | int | Epochs for GP retraining |
| `gp_retrain_lr` | 0.001 | float | Learning rate (lower for fine-tuning) |
| `gp_retrain_patience` | 50 | int | Early stopping patience |

**Purpose**: After evaluating a new instruction, the GP is retrained to incorporate the new observation. This keeps the surrogate model accurate as optimization progresses.

**Impact**: Lower epochs/patience for faster iteration cycles. The smaller learning rate prevents catastrophic forgetting of previous observations.

---

## 8. Inference / BoTorch Optimization

Parameters for the BoTorch-based acquisition function optimization.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `acquisition_type` | "ucb" | str | `--acquisition-type` | Acquisition function: "ucb" or "logei" |
| `ucb_beta` | 8.0 | float | `--ucb-beta` | UCB exploration parameter (higher = more exploration) |
| `num_restarts` | 64 | int | `--num-restarts` | L-BFGS-B multi-start restarts |
| `raw_samples` | 4096 | int | `--raw-samples` | Initial random samples for seeding (higher for better coverage in high-D) |
| `cosine_sim_threshold` | 0.90 | float | - | Min cosine similarity for acceptance |
| `max_rejection_attempts` | 10 | int | - | Max rejection attempts before forced accept |

**Purpose**: Controls the inner optimization loop that finds the best latent point according to the GP acquisition function.

**Acquisition Functions**:
- **UCB (Upper Confidence Bound)**: `μ(x) + β·σ(x)` - prefers regions with high uncertainty
  - β=8.0 (default): Very aggressive exploration, forces GP to try unexplored regions
  - Distance penalty is automatically DISABLED for UCB
- **LogEI**: Numerically stable Expected Improvement - more balanced but may get stuck in local optima
  - Distance penalty is applied when using LogEI

**Impact**:
- **Higher `ucb_beta`**: More exploration, less exploitation
- **Higher `num_restarts`**: More likely to find global optimum but slower
- **Higher `raw_samples`**: Better initialization coverage (important in high-D without TuRBO)
- **Higher `cosine_sim_threshold`**: Rejects more candidates, ensuring quality but may slow convergence

---

## 9. Vec2Text Settings

Text inversion model configuration for converting embeddings back to text.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `vec2text_model` | "32_tokens" | str | `--vec2text-model` | Model variant (32_tokens preferred) |
| `vec2text_beam` | 8 | int | `--vec2text-beam` | Beam search width |
| `vec2text_max_length` | 128 | int | `--vec2text-max-length` | Max output tokens |

**Purpose**: Vec2Text inverts GTR embeddings back to text.

**Impact**:
- **Higher `vec2text_beam`**: Better quality but slower generation
- **"32_tokens" model**: Faster, more diverse outputs, **recommended**
- **"512_tokens" model**: Longer sequences but has unicode garbage issues (»â€ artifacts)

**Model Variants**:
- `"32_tokens"`: ielabgroup/vec2text with corrector (fast, more diverse, **recommended**)
- `"512_tokens"`: vec2text official (longer sequences but produces unicode artifacts)

---

## 10. TuRBO Trust Region

Trust Region Bayesian Optimization for focused local search (Eriksson et al., NeurIPS 2019).

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `turbo_enabled` | False | bool | TuRBO disabled by default (use distance penalty instead) |
| `turbo_L_init` | 0.8 | float | Initial trust region side length (paper default) |
| `turbo_L_max` | 1.6 | float | Maximum trust region side length (paper default) |
| `turbo_L_min` | 0.0078 | float | Minimum side length (2^-7, triggers restart) |
| `turbo_tau_succ` | 3 | int | Consecutive successes to expand (double L) |
| `turbo_tau_fail` | 32 | int | Consecutive failures to shrink (paper: ⌈d/q⌉ for d=32, q=1) |

**Purpose**: TuRBO maintains a local trust region around the best observed point. It expands when optimization succeeds and shrinks when it fails, balancing exploration and exploitation.

**Impact**:
- **Larger `turbo_L_init`**: More exploration initially
- **Smaller `turbo_L_min`**: Allows finer local search before restart
- **Higher `turbo_tau_fail`**: More patient before shrinking (slower adaptation)
- **Lower `turbo_tau_succ`**: Faster expansion when succeeding

**Trust Region Dynamics**:
```
If success_count >= tau_succ: L = min(2*L, L_max), reset counters
If fail_count >= tau_fail: L = L/2, reset counters
If L < L_min: Restart from best observed point
```

---

## 11. PAS (Potential-Aware Anchor Selection)

Thompson Sampling-based anchor selection for trust region initialization.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `pas_enabled` | True | bool | Enable PAS for anchor selection |
| `pas_n_candidates` | 100 | int | Candidates per anchor for Thompson Sampling |

**Purpose**: Instead of always centering the trust region on the best observed point, PAS samples from the GP posterior to potentially find better anchor points that haven't been fully explored.

**Impact**:
- **Enabled**: More exploration, may find better regions
- **Higher `pas_n_candidates`**: Better sampling coverage but slower

**Algorithm**:
1. Sample GP posterior at `pas_n_candidates` random points
2. Select point with best sampled value as new anchor
3. Center trust region around selected anchor

---

## 12. Latent Space

Parameters for latent space bounds.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `latent_margin` | 0.2 | float | Margin for expanding latent bounds (20% each side) |

**Purpose**: Expands optimization bounds beyond observed data range to prevent edge effects.

---

## 13. Device & Paths

Device configuration and file paths.

| Parameter | Default | Type | CLI Flag | Description |
|-----------|---------|------|----------|-------------|
| `device` | "cuda" | str | `--device` | Device: "cuda", "cpu", "mps", "auto" |
| `validation_path` | "hbbops_improved_2/data/validation.json" | str | `--validation-path` | Validation dataset path |
| `seed` | 42 | int | - | Random seed for reproducibility |

**Purpose**: Basic configuration for hardware and data locations.

---

## 14. CLI-Only Arguments

Arguments available only via command line, not in config.py.

| Argument | Default | Type | Description |
|----------|---------|------|-------------|
| `--iterations` | 10 | int | Number of InvBO inference iterations |
| `--skip-hbbops` | False | bool | Load pre-evaluated results, skip Hyperband |
| `--hyperband-evals-path` | "lipo/data/hbbops_results_*.json" | str | Path to pre-evaluated results |
| `--load-grid` | None | str | Path to pre-evaluated grid JSONL |
| `--top-k` | 25 | int | Number of top instructions from grid |
| `--skip-ape` | False | bool | Skip APE generation, use cached |
| `--force-regenerate-ape` | False | bool | Force APE regeneration |
| `--no-ape-augment` | False | bool | Disable APE augmentation |
| `--hyperband-only` | False | bool | Run only HbBoPs, skip inference |
| `--skip-eval` | False | bool | Skip LLM evaluation, use GP predictions |
| `--validate-hyperband` | False | bool | Validate HbBoPs ranking vs grid |
| `--eval-model` | "Qwen/Qwen2.5-7B-Instruct" | str | Model for prompt evaluation |
| `--eval-backend` | "vllm" | str | Backend for evaluation |
| `--output-dir` | "lipo/results" | str | Output directory |
| `--debug` | False | bool | Debug mode (reduced epochs) |
| `--diverse-instructions` | None | str | Path to diverse instructions for VAE |
| `--instructions-path` | None | str | Path to instructions text file |

---

## Architecture Details

### VAE Architecture
```
Input: 768D GTR embedding (L2 normalized)

Encoder:
  Linear(768 → 256) → GELU → LayerNorm → Dropout(0.1)
  Linear(256 → 128) → GELU → LayerNorm
  Linear(128 → 64) → Split into mu (32D) + log_var (32D)

Reparameterization: z = mu + exp(0.5 * log_var) * epsilon

Decoder:
  Linear(32 → 128) → GELU → LayerNorm
  Linear(128 → 256) → GELU → LayerNorm → Dropout(0.1)
  Linear(256 → 768) → L2 Normalize
```

### GP Architecture
```
Model: ExactGP with FixedNoiseGaussianLikelihood

Mean: ZeroMean()

Kernel: ScaleKernel(
    MaternKernel(nu=2.5, ard_num_dims=32)
) with Gamma priors

Noise: Heteroscedastic from Beta posterior variance
  variance = (y * (1-y)) / (fidelity + 3)
  Clamped to [1e-8, 0.1]

Output: Error rates are negated for BoTorch maximization
  y_train = -error_rates (so max(-error) = min(error))
```

### Acquisition Function
```
qLogExpectedImprovement:
  acq_value = log_expected_improvement(model, best_f, x)

  TuRBO constrains optimization to trust region bounds
  (no separate distance penalty - TuRBO handles exploration/exploitation)
```

---

## Critical Thresholds Summary

Quick reference for the most important thresholds:

| Threshold | Value | Impact |
|-----------|-------|--------|
| `cosine_sim_threshold` | 0.90 | Rejects Vec2Text outputs with low fidelity |
| `vae_beta` | 0.01 | KL regularization (10x baseline for tight latent) |
| `latent_dim` | 32 | VAE latent dimension (24x compression from 768D) |
| `vec2text_model` | 32_tokens | Recommended model (512_tokens has unicode issues) |
| `turbo_L_min` | 0.0078 | Triggers trust region restart |
| `turbo_L_init` | 0.8 | Initial trust region size (paper default) |
| `roundtrip_validation_threshold` | 0.90 | Min quality for full pipeline |
| GP noise constraint | [0.001, 0.1] | Balances confidence vs overfitting |
| Lengthscale prior | Gamma(3.0, 6.0) | Inductive bias toward moderate scales |

---

## Parameter Tuning Guidelines

### For Better Exploration
- Increase `turbo_L_init` and `turbo_L_max`
- Increase `random_interleaving_prob`
- Enable PAS with higher `pas_n_candidates`

### For Better Exploitation
- Decrease `turbo_L_max`
- Lower `turbo_tau_succ` (expand faster when succeeding)
- Increase `turbo_tau_fail` (slower shrinking)

### For Faster Inference
- Decrease `num_restarts` and `raw_samples`
- Use "32_tokens" Vec2Text model (faster generation)

### For Better Quality
- Increase `cosine_sim_threshold`
- Increase `vec2text_beam`
