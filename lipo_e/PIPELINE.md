# LIPO-E Pipeline Documentation

**LIPO-E** = Latent Instruction Prompt Optimization with Exemplars

Joint optimization over instruction × exemplar selection using Hyperband + Bayesian Optimization.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LIPO-E Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA PREPARATION                                                 │
│     ├── APE Generator → 2000 instructions                           │
│     ├── train.json → 6154 Q/A pairs (exemplar pool)                 │
│     └── validation.json → 1319 samples (Hyperband evaluation)       │
│                                                                      │
│  2. ENCODING (GTR + StructureAwareVAE)                              │
│     ├── Instruction: text → GTR(768D) → VAE → z_inst (16D)          │
│     └── Exemplars: {Q/A}* → GTR(768D) → SetTransformer → z_ex (16D) │
│         └── z_joint = [z_inst || z_ex] = 32D                        │
│                                                                      │
│  3. HYPERBAND OPTIMIZATION                                          │
│     FOR each bracket s = smax...0:                                   │
│       ├── Sample n candidates with BO proposals                      │
│       ├── Evaluate at fidelity b (with caching + extension)         │
│       ├── Train GP on ALL fidelities (Beta smoothing + hetero noise)│
│       └── Successive halving: keep top, extend fidelity             │
│                                                                      │
│  4. INVBO INFERENCE (after Hyperband)                               │
│     FOR each iteration:                                              │
│       ├── Optimize z_joint via LogEI acquisition                    │
│       ├── Decode z_inst → Vec2Text → instruction text               │
│       ├── Decode z_ex → slot attention → exemplar selection         │
│       ├── Evaluate → Update GP                                       │
│       └── Repeat                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. StructureAwareVAE (`encoder.py`)

```
Input: (instruction_emb [768D], exemplar_embs [K × 768D])
       ↓
InstructionEncoder: 768 → 256 → 128 → (μ_inst, logvar_inst) [16D each]
       ↓
ExemplarSetEncoder (Set Transformer):
  ISAB → ISAB → PMA → (μ_ex, logvar_ex) [16D each]
       ↓
Joint Refinement: [z_inst || z_ex] → MLP → z_joint [32D]
```

### 2. JointPromptGP (`gp.py`)

```
z_joint [32D] → ProductKernel:
  k(z, z') = k_inst(z[0:16], z'[0:16]) × k_ex(z[16:32], z'[16:32])

Kernel: Matérn 5/2 with ARD (separate lengthscales per dimension)
Noise: FixedNoiseGaussianLikelihood with Beta variance
```

### 3. Hyperband (`hyperband.py`)

**Schedule:**
- `smax = floor(log_η(nvalid/bmin))` = 7 for nvalid=1319, bmin=10, η=2
- `B = (smax+1) × nvalid` = 10552 (total budget)

**Two-Stage Proposal:**
1. Sample 50 candidate instructions
2. For each: optimize z_ex via gradient (20 steps × 5 restarts)
3. Decode z_ex → discrete exemplar selection
4. Return (instruction, exemplars) with highest EI

**Beta Smoothing:**
```python
# Empirical Bayes prior (Method of Moments)
alpha, beta = fit_beta_prior(raw_error_rates)

# Posterior mean (smoothed error)
smoothed_error = (num_errors + alpha) / (fidelity + alpha + beta)

# Heteroscedastic noise (Beta variance)
noise_var = smoothed_error * (1 - smoothed_error) / (fidelity + alpha + beta + 1)
```

**Cache with Fidelity Extension:**
```python
cache_key = (instruction_id, frozenset(exemplar_ids), fidelity)

# Extend from lower fidelity instead of re-evaluating
if has_lower_fidelity:
    remaining = validation_data[prev_fidelity:new_fidelity]
    new_error = evaluate(remaining)  # Only new samples!
    total_error = (prev_error * prev_f + new_error * len(remaining)) / new_f
```

## Dimensions

| Component | Dimension | Notes |
|-----------|-----------|-------|
| GTR embedding | 768D | SentenceTransformer GTR-T5-Base |
| Instruction latent | 16D | z_inst |
| Exemplar latent | 16D | z_ex |
| Joint latent | 32D | z_joint = [z_inst \|\| z_ex] |
| Max exemplars | 8 | num_slots |
| Exemplar pool | 6154 | All train.json Q/A pairs |
| Instructions | 2000 | APE-generated |
| Validation | 1319 | GSM8K validation subset |

## Parameters

```python
# Hyperband
bmin = 10              # Minimum fidelity
eta = 2.0              # Halving rate
random_interleaving_prob = 0.1  # 10% random proposals

# VAE
vae_beta = 0.005       # KL weight
vae_epochs = 50000     # Training epochs

# GP
gp_epochs = 1000       # Training epochs
gp_lr = 0.01           # Learning rate
gp_patience = 50       # Early stopping

# Exemplar selection
num_slots = 8          # Max exemplars
min_exemplars = 0      # Allow zero-shot
```

## Usage

```bash
# Full run with Hyperband + InvBO
uv run python -m lipo_e.run \
    --qa-pool-size 6154 \
    --ape-num-instructions 2000 \
    --iterations 50

# Skip APE (use cached)
uv run python -m lipo_e.run \
    --no-use-ape \
    --instructions lipo_e/data/ape_instructions.json
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclass |
| `encoder.py` | GTREncoder + StructureAwareVAE |
| `set_transformer.py` | ISAB + PMA for exemplar encoding |
| `gp.py` | JointPromptGP + GPWithEI |
| `hyperband.py` | LIPOEHyperband with BO proposals |
| `training.py` | Data loading, APE generation |
| `inference.py` | InvBO inference loop |
| `run.py` | CLI entry point |

## Expected Results

Based on LIPO baseline (instruction-only):
- Best instruction accuracy: ~82-85%
- With exemplars: expected +2-5% improvement
- Target: ~87-90% accuracy

Hyperband typically evaluates:
- ~200-500 unique (instruction, exemplar_set) pairs
- ~5000-10000 total LLM calls (with fidelity extension)
- Runtime: 2-4 hours on single L40S
