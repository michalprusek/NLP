# InvBO Decoder Inversion Pipeline

Implementace dekóderu z GP latentního prostoru (10D) do Vec2Text embedding prostoru (768D) s cyklickou ztrátou. Řeší "misalignment problem" z článku InvBO (NeurIPS 2024).

## Klíčové Features

- **VAE Mode**: Variational Autoencoder pro hladký latentní prostor (doporučeno)
- **Trust Region**: TuRBO-style adaptivní trust region pro iterativní optimalizaci
- **LogEI**: Numericky stabilní Log Expected Improvement (NeurIPS 2023)
- **Inversion Loop**: InvBO-style inverze pro uzavření misalignment gap
- **UMAP Visualization**: 2D vizualizace EI landscape s diagnostikami

## Quick Start

```bash
# Doporučené: VAE mode s inversion loop a trust region
uv run python -m generation.invbo_decoder.run \
    --use-vae --use-inversion --iterations 50 \
    --trust-region --skip-eval --visualize

# S optimálními hyperparametry z tuningu
uv run python -m generation.invbo_decoder.run \
    --use-vae --use-inversion --iterations 50 \
    --vae-beta 0.01 --vae-epochs 1000 --vae-annealing 800 \
    --trust-region --skip-eval

# Jednoduchý běh (1 iterace)
uv run python -m generation.invbo_decoder.run --use-vae --use-inversion
```

## Architektura

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STANDARD MODE (--use-vae disabled):                                        │
│  ═══════════════════════════════════                                         │
│                                                                              │
│  Phase 1: GP + Encoder Training                                             │
│  ─────────────────────────────────                                           │
│  Instructions (100) → GTR (768D) → Unit-Cube Norm → FeatureExtractor → 10D  │
│                                                         │                    │
│                                                         ▼                    │
│                                              DeepKernelGP (Matérn 5/2)       │
│                                                         │                    │
│                                                         ▼                    │
│                                              Predict Error Rate              │
│                                                                              │
│  Phase 2: Decoder Training (frozen encoder)                                 │
│  ───────────────────────────────────────────                                 │
│  Diverse Instructions (1000+) → GTR → FeatureExtractor → 10D                │
│                                   │                        │                 │
│                                   ▼                        ▼                 │
│                          Target 768D        LatentDecoder → 768D             │
│                                   │                        │                 │
│                                   └────────────────────────┘                 │
│                                            │                                 │
│                                   L = λ_cycle × ||z - E(D(z))||²             │
│                                     + λ_cosine × (1 - cos_sim)               │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VAE MODE (--use-vae enabled, RECOMMENDED):                                 │
│  ══════════════════════════════════════════                                  │
│                                                                              │
│  Phase 1: VAE Training (KL Annealing)                                       │
│  ─────────────────────────────────────                                       │
│  Diverse Instructions (1100) → GTR → 768D → VAE Encoder → μ, σ → Sample z   │
│                                                                │             │
│                                                                ▼             │
│                                               VAE Decoder → 768D → L2 Norm   │
│                                                                              │
│  Loss = Recon(cosine) + β × KL(q(z|x) || N(0,1))                            │
│  KL Annealing: β = 0 → target over vae_annealing_epochs                     │
│                                                                              │
│  Phase 2: GP Training (with VAE encoder)                                    │
│  ─────────────────────────────────────────                                   │
│  Grid Instructions (100) → GTR → VAE.encode_mu() → 10D → GP                 │
│                                                                              │
│  Decoder = VAE.decode() (no separate training needed)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. EI Optimization in 10D Latent Space                                     │
│  ───────────────────────────────────────                                     │
│                                                                              │
│     ┌──────────────────────────────────────────────────────┐                 │
│     │            Trust Region Sampling                      │                │
│     │  ──────────────────────────────────                   │                │
│     │  if --trust-region:                                   │                │
│     │    Sample z ∈ L∞-ball(anchor, radius)                │                │
│     │  else:                                                │                │
│     │    Sample z ∈ Convex Hull of training latents        │                │
│     └──────────────────────────────────────────────────────┘                 │
│                              │                                               │
│                              ▼                                               │
│     For each candidate z:                                                   │
│        embedding = decoder(z)                                               │
│        LogEI(z) = log_h((y_best - μ - ξ) / σ) + log(σ)                     │
│                              │                                               │
│                              ▼                                               │
│        z* = argmax LogEI(z)                                                 │
│                                                                              │
│  2. Inversion Loop (--use-inversion enabled)                                │
│  ────────────────────────────────────────────                                │
│                                                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  for iter in 1..max_inversion_iters:                            │     │
│     │    1. embedding* = decoder(z*)                                  │     │
│     │    2. text* = Vec2Text(embedding*)                              │     │
│     │    3. z_inv = argmin ||decoder(z) - GTR(text*)||²              │     │
│     │       (solved via Adam, ~100 steps)                             │     │
│     │    4. gap = cosine_distance(decoder(z*), decoder(z_inv))        │     │
│     │    5. if gap ≤ threshold (0.1):                                 │     │
│     │         ACCEPT and break                                        │     │
│     │       else:                                                     │     │
│     │         z* = z_inv  # Use inverted latent                       │     │
│     │         continue                                                │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  3. Vec2Text Inversion                                                      │
│  ─────────────────────                                                       │
│                                                                              │
│     embedding* (768D) → Vec2Text Corrector → Novel Instruction Text         │
│                         (50 steps, beam=4)                                  │
│                                                                              │
│  4. GP Update & Trust Region Adaptation                                     │
│  ───────────────────────────────────────                                     │
│                                                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  New observation: (GTR(text*), actual_error)                    │     │
│     │  GP.add_observation_and_retrain()                               │     │
│     │                                                                 │     │
│     │  Trust Region Update (TuRBO-style):                             │     │
│     │    if improved:                                                 │     │
│     │      success_count++                                            │     │
│     │      if success_count >= 2: radius *= 1.5 (expand)              │     │
│     │    else:                                                        │     │
│     │      failure_count++                                            │     │
│     │      if failure_count >= 3: radius *= 0.5 (contract)            │     │
│     │    if radius < min_radius:                                      │     │
│     │      RESTART from best_z with initial_radius                    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  5. Visualization (--visualize enabled)                                     │
│  ──────────────────────────────────────                                      │
│                                                                              │
│     UMAP(10D → 2D) projection showing:                                      │
│     - EI surface (contour plot)                                             │
│     - z_opt (red star)                                                      │
│     - z_realized (white X)                                                  │
│     - Inversion gap (dashed line)                                           │
│     - Trust region boundary (yellow dashed)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Komponenty

### 1. GTRInstructionEncoder (`encoder.py`)

```python
GTRInstructionEncoder(
    model_name="sentence-transformers/gtr-t5-base",
    normalize=True,  # L2 normalization (required for Vec2Text)
    device="auto",
)
```

- **Model**: `sentence-transformers/gtr-t5-base`
- **Output**: 768D L2-normalized embedding
- **Kompatibilita**: Vec2Text

### 2. InstructionFeatureExtractor (`encoder.py`)

Deep kernel encoder pro GP:

```
768D GTR embedding
    │
Linear(768, 64) → ReLU → BatchNorm
    │
Linear(64, 32) → ReLU → BatchNorm
    │
Linear(32, 10)
    │
10D latent
```

- Trénuje se společně s GP (Phase 1)
- Zamrzne se pro decoder training (Phase 2)

### 3. InstructionVAE (`encoder.py`)

Variational Autoencoder pro hladký latentní prostor:

```
Encoder:                           Decoder:
768D → 64 → 32 → 2×latent_dim     latent_dim → 32 → 64 → 256 → 768D
         (μ, log_var)                                     ↓
              ↓                                    L2 Normalize
         Sample z                                         ↓
                                                   768D output
```

**Klíčové vlastnosti:**
- **KL Annealing**: β = 0 → target přes `vae_annealing_epochs` (prevence posterior collapse)
- **encode_mu()**: Deterministický encoder pro GP
- **decode()**: Decoder pro Vec2Text inversion
- **Loss**: `L = (1 - cosine_sim) + β × KL(q(z|x) || N(0,1))`

**Doporučené hyperparametry:**
```
--vae-beta 0.01         # Nižší KL weight pro lepší rekonstrukci
--vae-epochs 1000       # Delší trénink
--vae-annealing 800     # Pomalý ramp-up KL
--vae-patience 100      # Vyšší patience
```

### 4. LatentDecoder (`decoder.py`)

Mirror architektura k encoderu:

```
10D latent
    │
Linear(10, 32) → ReLU → BatchNorm
    │
Linear(32, 64) → ReLU → BatchNorm
    │
Linear(64, 256) → ReLU → BatchNorm
    │
Linear(256, 768)
    │
L2 Normalize
    │
768D Vec2Text-compatible embedding
```

**Poznámka**: V VAE mode se používá `VAE.decode()` místo samostatného decoderu.

### 5. GPWithEI (`gp.py`)

Gaussian Process s Expected Improvement:

```python
GPWithEI(
    device="cuda",
    latent_dim=10,
)
```

**Kernel:**
- Matérn 5/2 s ARD (10 lengthscales)
- Prior na lengthscale: `Gamma(3.0, 6.0)`
- Prior na outputscale: `Gamma(2.0, 0.15)`

**Klíčové metody:**
- `set_training_data(embeddings, error_rates)`
- `train(epochs, lr, patience)`
- `predict(embedding) → (mean, std)`
- `expected_improvement(embedding, xi) → EI`
- `log_expected_improvement(embedding, xi) → LogEI` (numericky stabilní)
- `add_observation_and_retrain(embedding, error_rate)` (online update)

### 6. LogEI Implementation (`gp.py`)

Numericky stabilní Log Expected Improvement z článku "Unexpected Improvements to Expected Improvement" (NeurIPS 2023):

```python
def log_h(z: float) -> float:
    """log(h(z)) where h(z) = φ(z) + z·Φ(z)

    Uses three branches for numerical stability:
    1. z > -1: Direct computation
    2. -1/√ε < z ≤ -1: erfcx-based
    3. z ≤ -1/√ε: Asymptotic approximation
    """

LogEI(x) = log_h(z) + log(σ)
where z = (y_best - μ - ξ) / σ
```

**Výhody:**
- Numericky stabilní i pro velmi malé EI hodnoty
- Umožňuje gradient-based optimalizaci
- Nedochází k underflow (log místo tiny floats)

### 7. TrustRegionManager (`trust_region.py`)

TuRBO-style trust region pro 10D latentní prostor:

```python
TRConfig(
    initial_radius=0.5,    # Menší pro 10D (vs 1.0 pro 32D)
    min_radius=0.05,       # Minimum před restartem
    max_radius=2.0,        # Maximum (prevent over-expansion)
    expand_factor=1.5,     # Násobit při úspěchu
    contract_factor=0.5,   # Násobit při neúspěchu
    success_threshold=2,   # Po-sobě-jdoucí úspěchy pro expanzi
    failure_threshold=3,   # Po-sobě-jdoucí neúspěchy pro kontrakci
    n_restarts_max=5,      # Maximum restartů
)
```

**Sampling:**
- L∞-ball kolem anchor: `|z_i - anchor_i| ≤ radius`
- Uniform sampling v hyperkrychli

**Update logika:**
```
if improved:
    success_count++
    if success_count >= threshold: EXPAND
else:
    failure_count++
    if failure_count >= threshold: CONTRACT
    if radius < min_radius: RESTART from best_z
```

### 8. Vec2TextInverter (`inference.py`)

Lazy-loaded Vec2Text pro embedding-to-text inverzi:

```python
Vec2TextInverter(
    num_steps=50,      # Correction iterations
    beam_width=4,      # Beam search width
    max_length=128,    # Max output tokens
    device="auto",
)
```

**Models:**
- `ielabgroup/vec2text_gtr-base-st_inversion`
- `ielabgroup/vec2text_gtr-base-st_corrector`

### 9. InvBOInference (`inference.py`)

Kompletní inference pipeline:

```python
InvBOInference(
    gp=trained_gp,
    decoder=trained_decoder,
    gtr=gtr_encoder,
    vec2text_steps=50,
    vec2text_beam=4,
    trust_region_config=TRConfig(),
)
```

**Klíčové metody:**
- `get_best_training_latent() → (latent, idx, error)`
- `optimize_latent_adaptive(n_candidates, use_trust_region) → (z_opt, log_ei)`
- `run_single_iteration(use_trust_region, use_inversion) → (result, gap, log_ei)`
- `inversion_step(text) → InversionStepResult` (InvBO-style inversion)
- `validate_inversion_gap(n_samples) → stats`

### 10. Visualization (`visualize.py`)

UMAP-based EI landscape visualization:

```python
visualize_ei_landscape(
    inference=inference,
    center_latent=z_opt,
    realized_text=generated_text,
    best_y=best_error,
    trust_region=tr_manager,
    span=1.0,
    n_grid_samples=300,
    save_path="ei_landscape.png",
)
```

**Zobrazuje:**
- EI surface (contour plot)
- z_opt (red star) - kde optimalizace našla vysoké EI
- z_realized (white X) - kam Vec2Text skutečně přistálo
- Inversion gap (red dashed line)
- Trust region boundary (yellow dashed)

**Metriky:**
- `cosine_gap`: Cosine distance mezi z_opt a z_realized embeddingy
- `inversion_gap_2d`: Vzdálenost v 2D UMAP prostoru
- `log_ei_at_opt`, `log_ei_at_realized`: LogEI hodnoty

## Datové Soubory

| Soubor | Popis | Použití |
|--------|-------|---------|
| `datasets/inversion/instructions_100.txt` | 100 instrukcí v 10 kategoriích | GP training, baseline |
| `datasets/inversion/grid_100_qend.jsonl` | Error rates pro 100 instrukcí | GP targets |
| `datasets/inversion/diverse_instructions_1000.json` | 1000+ diverse instrukcí (APE) | VAE/Decoder training |

## Loss Funkce

### VAE Mode (doporučeno)

```python
# VAE Training
L_vae = (1 - cosine_sim(x, x_recon)) + β × KL(q(z|x) || N(0,1))

# β annealing: 0 → vae_beta over vae_annealing_epochs
β_current = vae_beta × min(1, epoch / vae_annealing_epochs)
```

### Standard Mode

```python
# Phase 1: GP Training
L_gp = -MLL(GP(encoder(normalize(embedding))), error_rate)

# Phase 2: Decoder Training
L_decoder = λ_cycle × ||z - encoder(normalize(decoder(z)))||²
          + λ_cosine × (1 - cosine_sim(decoder(z), target_embedding))
```

**Hyperparametry (Standard Mode):**
- `λ_cycle = 1.0`
- `λ_cosine = 5.0`
- `cycle_tolerance = 0.0` (striktní)

## Normalizace

### Unit-Cube Normalization (pro GP vstupy)
```python
X_norm = (X - X_min) / (X_max - X_min)
# X_min, X_max z training dat (100 instrukcí)
```

### Z-Score Standardization (pro GP targets)
```python
y_norm = (y - y_mean) / y_std
```

### L2 Normalization (pro Vec2Text kompatibilitu)
```python
embedding = embedding / ||embedding||₂
# Kritické pro GTR embeddings
```

## CLI Parametry

### Základní

| Parametr | Default | Popis |
|----------|---------|-------|
| `--iterations` | 1 | Počet optimalizačních iterací |
| `--skip-eval` | False | Použít GP predikci místo LLM evaluace |
| `--use-vae` | False | Použít VAE mode (doporučeno) |
| `--use-inversion` | False | Použít InvBO inversion loop |
| `--trust-region` | False | Použít adaptivní trust region |
| `--visualize` | False | Generovat EI landscape vizualizace |

### VAE Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--vae-beta` | 0.1 | KL regularization weight |
| `--vae-epochs` | 1000 | VAE training epochs |
| `--vae-annealing` | 500 | KL annealing epochs (0 → beta) |
| `--vae-patience` | 100 | Early stopping patience |

### Trust Region Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--tr-initial` | 0.5 | Initial trust region radius |
| `--tr-min` | 0.05 | Minimum radius (triggers restart) |
| `--tr-max` | 2.0 | Maximum radius |

### Inversion Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--max-inversion-iters` | 3 | Maximum inversion loop iterations |
| `--gap-threshold` | 0.1 | Cosine distance threshold pro accept |

### Vec2Text Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--vec2text-steps` | 50 | Correction steps |
| `--vec2text-beam` | 4 | Beam width |

### GP Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--gp-epochs` | 3000 | GP training epochs |
| `--retrain-interval` | 1 | Retrain GP every N iterations |
| `--retrain-epochs` | 500 | Epochs pro GP retraining |

## Výstupní Formáty

### Results JSON

```json
{
  "timestamp": "20251229_120000",
  "method": "InvBO Decoder",
  "args": { ... },
  "grid_best": {
    "instruction_id": 42,
    "error_rate": 0.1077
  },
  "optimized": {
    "instruction": "Solve step by step...",
    "error_rate": 0.0950
  },
  "iteration_history": [
    {
      "iteration": 1,
      "instruction": "...",
      "cosine_similarity": 0.85,
      "predicted_error": 0.10,
      "actual_error": 0.11,
      "gap": 0.05,
      "log_ei": -2.5,
      "improved": true,
      "trust_region_radius": 0.5,
      "gp_samples": 101
    }
  ],
  "improvement": 0.0127,
  "vae_quality_metrics": {
    "cosine_mean": 0.98,
    "latent_var_mean": 0.45,
    "active_dims": 10,
    "kld_mean": 5.2
  }
}
```

### Log File

Automaticky ukládán do `generation/invbo_decoder/results/run_TIMESTAMP.log`

## Známé Problémy a Řešení

### 1. Misalignment Problem (InvBO paper)
**Problém**: Decoder může produkovat embeddingy, které po Vec2Text inverzi dají odlišný text.

**Řešení**:
- Inversion loop: `z_inv = argmin ||decoder(z) - GTR(text)||²`
- Cosine gap threshold pro accept/reject

### 2. Out-of-Distribution Decoding
**Problém**: Decoder může produkovat embeddingy mimo distribuci reálných textů.

**Řešení**:
- Trust region omezuje sampling na oblasti blízko známých latentů
- VAE mode s KL regularizací zajišťuje smooth latent space

### 3. Posterior Collapse (VAE)
**Problém**: VAE může ignorovat latenty (z → konstanta).

**Řešení**:
- KL annealing: β = 0 → target přes `vae_annealing_epochs`
- Nižší `vae_beta` (0.01-0.1)
- Delší trénink s vyšší patience

### 4. Vec2Text Limitations
**Problém**: Selhává pro delší texty (>30 tokenů).

**Řešení**:
- Instruction-only přístup (ne celé prompty)
- `max_length=128` v Vec2Text

### 5. Numerical Instability in EI
**Problém**: Standard EI underflows pro velmi malé hodnoty.

**Řešení**:
- LogEI implementace z NeurIPS 2023 paper
- Tři branches pro různé rozsahy z-score

## Porovnání s COWBOYS

| Feature | InvBO Decoder | COWBOYS |
|---------|---------------|---------|
| Latent Dim | 10D | 32D |
| Encoder | Deep Kernel / VAE | VAE only |
| Optimizer | LogEI sampling | pCN MCMC |
| Trust Region | TuRBO-style (L∞) | Custom (L2) |
| Inversion | InvBO-style loop | Direct Vec2Text |
| GP Retraining | Online (každá iterace) | Batch |

## Reference

- **InvBO**: Deshwal et al., 2024 - "Inversion-Based BO with Structured Inputs" (NeurIPS 2024)
- **TuRBO**: Eriksson et al., 2019 - "Scalable Global Optimization via Local BO"
- **LogEI**: Ament et al., 2023 - "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)
- **Vec2Text**: Morris et al., 2023 - "Text Embeddings Reveal (Almost) As Much As Text"
