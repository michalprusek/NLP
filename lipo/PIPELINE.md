# LIPO - Latent Instruction Prompt Optimization

## Přehled

**LIPO** je instruction-only optimalizační framework kombinující:
1. **APE** - generování diverzních instrukcí (2000 instrukcí, 8 stylových kategorií)
2. **VAE** - mapování do 32D latentního prostoru s KL annealing (50k epochs)
3. **Hyperband** - multi-fidelity evaluace s successive halving
4. **InvBO** - optimalizace v latentním prostoru + Vec2Text inverze (32_tokens model)

---

## Architektura

```
FÁZE 1: GENEROVÁNÍ INSTRUKCÍ
┌──────────────────────────────────────────────────────────┐
│ APE Generace (2000 instrukcí)                            │
│ - 8 stylových kategorií (minimalist, direct_command,     │
│   chain_of_thought, pedagogical, analytical,             │
│   persona_roleplay, programmatic, adversarial)           │
│ - 3-úrovňová augmentace:                                 │
│   L1: Base generation                                    │
│   L2: LLM paraphrasing (30% šance)                       │
│   L3: Noise injection (50% šance)                        │
│ - GTR Encoding (768D, L2-normalized)                     │
└──────────────────────────────────────────────────────────┘
                    ↓
FÁZE 2: VAE TRÉNINK (IMPROVED v2.0)
┌──────────────────────────────────────────────────────────┐
│ InstructionVAE (vylepšená architektura)                  │
│ - Encoder: 768D → 512 → 256 → 128 → 2×32 (mu + log_var) │
│ - Decoder: 32D → 128 → 256 → 512 → 768D + L2 norm       │
│ - Aktivace: GELU + LayerNorm + Dropout(0.1)              │
│ - Loss: (1-mse_w)·cosine + mse_w·MSE + β·KL             │
│ - MSE weight: 0.2 (20% MSE + 80% cosine)                │
│ - KL annealing: β=0 → 0.005 přes 2500 epoch             │
│ - Curriculum learning: 30% → 100% přes 5000 epoch       │
│ - Early stopping: patience=1000 (po annealing)           │
│ - Max epochs: 50000                                      │
└──────────────────────────────────────────────────────────┘
                    ↓
FÁZE 3: HYPERBAND EVALUACE
┌──────────────────────────────────────────────────────────┐
│ Multi-fidelity evaluace                                  │
│ - bmin=10, η=2.0                                         │
│ - Successive halving: 10→20→40→80→160→...→1319          │
│ - GP trénován na VŠECH evaluacích                        │
│ - Heteroskedastický noise: Beta posterior variance       │
│ - Empirical Bayes prior: α, β fitted from data           │
└──────────────────────────────────────────────────────────┘
                    ↓
FÁZE 4: InvBO INFERENCE (IMPROVED v2.0)
┌──────────────────────────────────────────────────────────┐
│ UCB Acquisition with Adaptive β + Noise Injection        │
│ - UCB = μ + β·σ                                          │
│ - Adaptive β: 8.0 → 2.0 lineárně přes iterace           │
│   (více explorace na začátku, více exploitace na konci) │
│ - Noise injection: scale=0.05 pro diverzitu             │
│ - 64 restarts, 4096 raw samples                          │
│ - GP přímo na 32D VAE latent (ARD kernel)               │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ Text Generation (Vec2Text)                               │
│ - z_opt (32D) + noise → VAE decoder → embedding (768D)   │
│ - embedding → Vec2Text (32_tokens) → instruction text    │
│ - Garbage filtering: reject unicode artifacts            │
│ - 32_tokens model (doporučeno, bez unicode issues)       │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ Candidate Rejection                                      │
│ - Cosine similarity threshold: 0.90                      │
│ - Max rejection attempts: 10                             │
│ - Low-quality acceptance s WARNING                       │
└──────────────────────────────────────────────────────────┘
```

---

## Komponenty - Detailní Specifikace

### GTRInstructionEncoder (`encoder.py:17-83`)

```python
model_name = "sentence-transformers/gtr-t5-base"
embedding_dim = 768
normalize = True  # L2 normalizace (kritické pro Vec2Text)
```

**Metody:**
- `encode(text) -> np.ndarray[768]` - single text
- `encode_tensor(text) -> torch.Tensor[768]` - single text na device
- `encode_batch(texts, batch_size=32) -> np.ndarray[N, 768]`

### InstructionVAE (`encoder.py:86-359`) - IMPROVED v2.0

**Vylepšená Architektura (větší kapacita):**
```
Encoder: 768D → 512 (1.5×) → 256 (2×) → 128 (2×) → 64 (2×, mu+log_var)
         GELU + LayerNorm + Dropout(0.1) mezi vrstvami

Decoder: 32D → 128 (4×) → 256 (2×) → 512 (2×) → 768D (1.5×)
         GELU + LayerNorm + Dropout(0.1)
         + L2 normalizace na výstupu (kritické pro GTR)
```

**Parametry:**
| Parametr | Hodnota | Popis |
|----------|---------|-------|
| `input_dim` | 768 | GTR embedding dimension |
| `latent_dim` | 32 | VAE latent dimension (24× komprese) |
| `beta` | 0.005 | KL regularization weight (sníženo z 0.01) |
| `gamma` | 0.0 | Cycle consistency (vypnuto) |
| `mse_weight` | 0.2 | MSE složka v recon loss (20% MSE + 80% cosine) |

**Loss Function (IMPROVED):**
```python
# Reconstruction loss (kombinace cosine + MSE)
cosine_sim = F.cosine_similarity(x, x_recon, dim=-1)
cosine_loss = (1 - cosine_sim).mean()
mse_loss = F.mse_loss(x, x_recon)
recon_loss = (1 - mse_weight) * cosine_loss + mse_weight * mse_loss

# KL divergence to N(0,1)
kl_loss = -0.5 * (1 + log_var - mu² - exp(log_var)).sum(dim=-1).mean()

# Total loss
total_loss = recon_loss + beta * kl_loss
```

**KL Annealing:**
```python
# Beta warmup přes prvních 2500 epoch (vae_annealing_epochs)
if epoch < annealing_epochs:
    current_beta = target_beta * (epoch / annealing_epochs)
else:
    current_beta = target_beta

# Early stopping se resetuje po annealing warmup
```

**Curriculum Learning (NEW):**
```python
# Postupně zvyšuje komplexitu trénovacích dat
# Instrukce jsou seřazeny podle délky (kratší = jednodušší)

if epoch < curriculum_epochs:
    # Lineární progress: 30% → 100%
    progress = epoch / curriculum_epochs
    current_pct = start_pct + (1.0 - start_pct) * progress
    n_embeddings = int(len(embeddings) * current_pct)
    curr_embeddings = embeddings[:n_embeddings]  # Kratší instrukce první
else:
    curr_embeddings = embeddings  # 100% dat

# Parametry:
# vae_curriculum = True
# vae_curriculum_start_pct = 0.3  # 30%
# vae_curriculum_epochs = 5000
```

### VAEWithAdapter (`encoder.py:362-469`)

**Wrapper pro frozen VAE (bez adapteru):**
```python
# Pipeline pro GP:
x (768D) → VAE encoder → z (32D) → GP

# Pipeline pro dekódování:
z (32D) → VAE decoder → embedding (768D) → Vec2Text → text
```

### InstructionDeepKernelGP (`gp.py:127-240`)

**GP s ARD Matern 5/2 kernelem:**

```python
# Kernel architecture
kernel = ScaleKernel(
    MaternKernel(
        nu=2.5,  # Matern 5/2
        ard_num_dims=32,  # Per-dimension lengthscales
        lengthscale_prior=GammaPrior(α, β)  # Data-driven
    ),
    outputscale_prior=GammaPrior(2.0, 2.0)
)
```

**Data-driven Prior Estimation:**
```python
# Lengthscale prior: centered at median pairwise distance
# Outputscale prior: Gamma(2,2) s mean=1 pro standardizované y
```

**Heteroskedastický Noise (Beta Posterior Variance):**
```python
# Pro error_rate p měřenou na n vzorcích:
# Variance = p(1-p) / (n + α + β + 1)
# kde α, β jsou Empirical Bayes prior parameters
```

### GPWithEI (`gp.py:243-966`)

**Wrapper pro GP s Expected Improvement:**

**DŮLEŽITÉ - Negace error_rate:**
```python
# BoTorch maximizuje, my minimalizujeme error_rate
# GP predikuje -error_rate:
#   Low error (0.05)  → y = -0.05 (higher = better)
#   High error (0.30) → y = -0.30 (lower = worse)
# EI hledá maximum (-error) = minimum (error)
```

**Training Pipeline:**
```python
# 1. Transform embeddings to VAE latents
z_vae = vae.encode_mu(embeddings)  # 768D → 32D

# 2. Unit cube normalization
X_norm = (z_vae - X_min) / (X_max - X_min)

# 3. Standardize outputs (BoTorch Standardize transform)
y_norm = (y - y_mean) / y_std

# 4. Compute heteroscedastic noise from Beta posterior
noise = p(1-p) / (fidelity + α + β + 1)
noise_standardized = noise / y_std²

# 5. Train GP with FixedNoiseGaussianLikelihood
```

---

## Konfigurace (`config.py`)

### APE Generace
| Parametr | Default | Popis |
|----------|---------|-------|
| `ape_num_instructions` | 2000 | Počet instrukcí k generování |
| `ape_model` | Qwen/Qwen2.5-7B-Instruct | Model pro generování |
| `ape_backend` | vllm | Backend (vllm, openai, deepinfra) |
| `ape_cache_path` | lipo/data/ape_instructions.json | Cache file |
| `ape_batch_size` | 10 | Batch size pro generování |
| `ape_max_tokens` | 100 | Max tokens per instruction |
| `ape_max_length` | 500 | Max character length |

### VAE Trénink (IMPROVED v2.0)
| Parametr | Default | Popis |
|----------|---------|-------|
| `embedding_dim` | 768 | GTR embedding dimension |
| `latent_dim` | 32 | VAE latent dimension (24× komprese) |
| `vae_beta` | **0.005** | KL regularization weight (sníženo z 0.01) |
| `vae_gamma` | 0.0 | Cycle consistency (vypnuto) |
| `vae_mse_weight` | **0.2** | MSE složka v recon loss (NEW) |
| `vae_epochs` | 50000 | Max training epochs |
| `vae_annealing_epochs` | 2500 | KL annealing warmup period |
| `vae_patience` | 1000 | Early stopping patience |
| `vae_lr` | 0.0006 | Learning rate |
| `vae_batch_size` | 64 | Training batch size |
| `vae_grad_clip` | 1.0 | Gradient clipping threshold |
| `vae_eta_min` | 1e-4 | Cosine annealing minimum LR |
| `vae_curriculum` | **True** | Curriculum learning (NEW) |
| `vae_curriculum_start_pct` | **0.3** | Start s 30% dat |
| `vae_curriculum_epochs` | **5000** | Epochs pro ramp-up na 100% |

### Round-Trip Validation
| Parametr | Default | Popis |
|----------|---------|-------|
| `roundtrip_validation_threshold` | 0.90 | Min cosine sim pro VAE kvalitu |
| `roundtrip_validation_samples` | 20 | Počet vzorků k testování |

### Hyperband
| Parametr | Default | Popis |
|----------|---------|-------|
| `bmin` | 10 | Min fidelity (samples) |
| `eta` | 2.0 | Successive halving rate |
| `random_interleaving_prob` | 0.1 | Random candidate probability |
| `min_fidelity_pct` | 0.75 | GP na top 25% fidelity dat |

### GP Trénink
| Parametr | Default | Popis |
|----------|---------|-------|
| `gp_epochs` | 10000 | Max training epochs |
| `gp_lr` | 0.0025 | Learning rate (scaled for 32D) |
| `gp_patience` | 100 | Early stopping patience |

### Inference (Acquisition) - IMPROVED v2.0
| Parametr | Default | Popis |
|----------|---------|-------|
| `acquisition_type` | "ucb" | "ucb" nebo "logei" |
| `ucb_beta` | 8.0 | UCB exploration (počáteční hodnota) |
| `ucb_beta_adaptive` | **True** | Adaptive UCB β scheduling (NEW) |
| `ucb_beta_final` | **2.0** | Koncová hodnota β (NEW) |
| `latent_noise_scale` | **0.05** | Noise injection pro diverzitu (NEW) |
| `num_restarts` | 64 | L-BFGS-B restarts |
| `raw_samples` | 4096 | Inicializační vzorky |
| `cosine_sim_threshold` | 0.90 | Min cosine pro akceptaci kandidáta |
| `max_rejection_attempts` | 10 | Max pokusů před forced acceptance |

### Vec2Text
| Parametr | Default | Popis |
|----------|---------|-------|
| `vec2text_model` | 32_tokens | Model typ ("32_tokens" pro diverzitu, "512_tokens" má unicode issues) |
| `vec2text_beam` | 8 | Beam width pro generation |
| `vec2text_max_length` | 128 | Max output tokens |

### TuRBO (Trust Region)
| Parametr | Default | Popis |
|----------|---------|-------|
| `turbo_enabled` | False | **VYPNUTO** (použij UCB s distance penalty) |
| `turbo_L_init` | 0.8 | Počáteční velikost trust region |
| `turbo_L_max` | 1.6 | Max velikost |
| `turbo_L_min` | 0.0078 | Min velikost (2^-7, triggers restart) |
| `turbo_tau_succ` | 3 | Úspěchy pro expand |
| `turbo_tau_fail` | 32 | Neúspěchy pro shrink |

### PAS (Potential-Aware Anchor Selection)
| Parametr | Default | Popis |
|----------|---------|-------|
| `pas_enabled` | True | Thompson Sampling based anchor selection |
| `pas_n_candidates` | 100 | Candidates per anchor |

### Distance Penalty
| Parametr | Default | Popis |
|----------|---------|-------|
| `distance_penalty_enabled` | True | Povoleno v configu |
| `distance_weight` | 2.0 | Penalty strength |
| `distance_threshold` | 0.3 | Min distance before penalty |

**DŮLEŽITÉ:** Distance penalty se aplikuje pouze pro **LogEI**, ne pro **UCB** (UCB už exploruje přes σ člen).

### Latent Space
| Parametr | Default | Popis |
|----------|---------|-------|
| `latent_margin` | 0.2 | Margin pro latent bounds expansion (20% each side) |

### GP Retrain (during inference)
| Parametr | Default | Popis |
|----------|---------|-------|
| `gp_retrain_epochs` | 1000 | Epochs pro retrain po každé iteraci |
| `gp_retrain_lr` | 0.001 | Learning rate |
| `gp_retrain_patience` | 50 | Early stopping |

### Device/Paths
| Parametr | Default | Popis |
|----------|---------|-------|
| `device` | cuda | Compute device |
| `validation_path` | hbbops_improved_2/data/validation.json | Validation data |
| `seed` | 42 | Random seed |

---

## Acquisition Functions (`botorch_acq.py`)

### UCB (Upper Confidence Bound) - IMPROVED v2.0
```python
UCB(x) = μ(x) + β·σ(x)

# Adaptive β scheduling (NEW):
# β lineárně klesá z 8.0 na 2.0 přes všechny iterace
# progress = iteration / total_iterations
# β = 8.0 * (1 - progress) + 2.0 * progress

# Noise injection (NEW):
# Po optimalizaci se přidá malý noise k z_opt pro diverzitu
# z_noisy = z_opt + N(0, 0.05²)
# z_clipped = clamp(z_noisy, bounds)
```

### LogEI (Log Expected Improvement)
```python
# Numericky stabilní implementace přes BoTorch qLogExpectedImprovement
# EI(x) = (best - μ(x))·Φ(z) + σ(x)·φ(z)
# LogEI = log(EI) - stabilní i pro velmi malé hodnoty
# kde z = (best - μ(x)) / σ(x)
```

### DistancePenalizedLogEI
```python
# Penalizuje body daleko od trénovacích dat
penalty = weight * max(0, min_dist - threshold)
penalized_logEI = logEI - penalty  # V log-space
```

---

## Empirical Bayes Prior (`training.py:28-83`)

**Fit Beta prior z dat (Method of Moments):**
```python
# Pro error rates s mean μ a variance σ²:
common = μ(1-μ)/σ² - 1
α = μ · common
β = (1-μ) · common

# Posterior mean pro nové pozorování:
posterior_mean = (errors + α) / (n + α + β)

# Posterior variance (pro GP noise):
variance = p(1-p) / (n + α + β + 1)
```

**Výhody oproti Beta(1,1):**
- Prior centered at data mean (ne 50%)
- Přesnější uncertainty pro low-fidelity samples
- Lepší smoothing pro extrémní hodnoty

---

## Quality KPIs

### VAE Quality (`quality_kpi.py`)
| KPI | Threshold | Popis |
|-----|-----------|-------|
| `cosine_mean` | > 0.95 | Mean reconstruction similarity |
| `percentile_10` | > 0.90 | 10th percentile (worst 10%) |
| `below_90_pct` | < 5% | Percentage below 0.90 threshold |
| `quality_tier` | Excellent/Good/Fair/Poor | Overall quality rating |

### GP Spearman
| KPI | Popis |
|-----|-------|
| `correlation` | Spearman ρ between predicted and actual |
| `p_value` | Statistical significance |

### System Gap
| KPI | Popis |
|-----|-------|
| `mean_gap` | Mean L2 distance z_opt vs z_real |
| `max_gap` | Worst case gap |

---

## CLI Usage

```bash
# Standard run s UCB
uv run python -m lipo.run --iterations 10

# S LogEI místo UCB
uv run python -m lipo.run --acquisition-type logei

# Změna UCB beta (více explorace)
uv run python -m lipo.run --ucb-beta 12.0

# Skip HbBoPs (načti pre-evaluované)
uv run python -m lipo.run --skip-hbbops

# Grid loading
uv run python -m lipo.run --load-grid datasets/hbbops/full_grid_combined.jsonl --top-k 25

# Debug mode (rychlý test)
uv run python -m lipo.run --iterations 1 --debug

# Hyperband only (bez inference)
uv run python -m lipo.run --hyperband-only

# Custom VAE parameters
uv run python -m lipo.run --vae-beta 0.02 --vae-epochs 30000 --vae-annealing 2000
```

---

## Shrnutí parametrů (IMPROVED v2.0)

| Komponenta | Parametr | Default | Poznámka |
|------------|----------|---------|----------|
| **VAE** | latent_dim | 32 | 24× komprese |
| | architecture | 768→512→256→128→32 | **Větší kapacita** |
| | beta | **0.005** | KL weight (sníženo) |
| | mse_weight | **0.2** | MSE v recon loss (NEW) |
| | curriculum | **True** | Curriculum learning (NEW) |
| | curriculum_start_pct | **0.3** | 30% dat na začátku |
| | curriculum_epochs | **5000** | Ramp-up period |
| | annealing_epochs | 2500 | Warmup |
| | epochs | 50000 | Max |
| | patience | 1000 | Early stop |
| | lr | 0.0006 | |
| **GP** | epochs | 10000 | |
| | lr | 0.0025 | Scaled for 32D |
| | patience | 100 | |
| | kernel | Matern 5/2 | ARD (32 lengthscales) |
| **Inference** | acquisition_type | ucb | ucb/logei |
| | ucb_beta | 8.0 | Počáteční exploration |
| | ucb_beta_adaptive | **True** | Adaptive scheduling (NEW) |
| | ucb_beta_final | **2.0** | Koncová hodnota |
| | latent_noise_scale | **0.05** | Noise pro diverzitu (NEW) |
| | num_restarts | 64 | L-BFGS-B |
| | raw_samples | 4096 | Initialization |
| **Vec2Text** | model | 32_tokens | Doporučeno (bez unicode issues) |
| **Candidate** | cosine_threshold | 0.90 | Candidate acceptance |
| | garbage_filter | enabled | Reject unicode artifacts |
| **TuRBO** | enabled | False | Use UCB instead |
| **Distance** | enabled | True | Auto-disabled for UCB |

---

## Changelog

### v2.0 (2025-01-05)
- **VAE Architecture**: Zvětšena kapacita encoderu/decoderu (přidána 512D vrstva)
- **VAE Loss**: Přidána MSE složka (20%) k cosine loss pro lepší rekonstrukci
- **VAE Beta**: Sníženo z 0.01 na 0.005 pro lepší rekonstrukci
- **Curriculum Learning**: Trénink začíná s kratšími (jednoduššími) instrukcemi
- **Adaptive UCB β**: Lineární decay z 8.0 na 2.0 (explorace → exploitace)
- **Noise Injection**: Malý noise (0.05) k z_opt pro diverzitu generovaných kandidátů
- **Vec2Text**: Default změněn na 32_tokens (512_tokens má unicode issues)
