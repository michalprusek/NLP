# InvBO Decoder Inversion Pipeline

Implementace dekóderu z GP latentního prostoru (10D) do Vec2Text embedding prostoru (768D) s cyklickou ztrátou. Řeší "misalignment problem" z článku InvBO (NeurIPS 2024).

## Klíčové Features

- **VAE Mode (default)**: Variational Autoencoder pro hladký latentní prostor (beta=0.1)
- **BoTorch qLogEI**: Gradient-based optimalizace s numericky stabilním Log Expected Improvement
- **Trust Region (default ON)**: Omezuje exploration do známých oblastí latentního prostoru
- **Inversion Loop**: InvBO-style inverze pro uzavření misalignment gap
- **Standardize Transform**: BoTorch outcome transform pro správnou denormalizaci

## Quick Start

```bash
# Doporučené: Standardní běh (VAE beta=0.1, Trust Region ON, BoTorch qLogEI)
uv run python -m generation.invbo_decoder.run --iterations 10

# S explicitními hyperparametry (defaults)
uv run python -m generation.invbo_decoder.run \
    --iterations 50 --vae-beta 0.1 --vae-annealing 500

# Bez trust region (pokud experimentujete)
uv run python -m generation.invbo_decoder.run --no-trust-region --iterations 10

# Jednoduchý běh (1 iterace) s vizualizací
uv run python -m generation.invbo_decoder.run --visualize
```

## DŮLEŽITÉ

- **Trust Region je zapnutý defaultně** - použijte `--no-trust-region` pro vypnutí
- **Vždy evaluovat na plném validation setu (1319 samples)** - nikdy nesnižovat `--eval-samples`
- **VAE beta=0.1** zajišťuje hladký latentní prostor pro stabilní optimalizaci
- **GP noise constraint 0.1** zabraňuje overconfidence (MAE validation ~0.02)

## Architektura

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VAE MODE (default, doporučeno):                                            │
│  ═══════════════════════════════                                             │
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
│  Early stopping: Tracks reconstruction loss (not total) to avoid premature  │
│                  stop during KL annealing phase                              │
│                                                                              │
│  Phase 2: GP Training (with VAE encoder)                                    │
│  ─────────────────────────────────────────                                   │
│  Grid Instructions (100) → GTR → VAE.encode_mu() → 10D → GP                 │
│                                                                              │
│  Decoder = VAE.decode() (no separate training needed)                       │
│                                                                              │
│  GP Features:                                                               │
│  - InstructionDeepKernelGP inherits from GPyTorchModel (BoTorch compatible)│
│  - Noise constraint GreaterThan(0.05) prevents overconfidence               │
│  - Matern 5/2 kernel with ARD (10 lengthscales)                             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STANDARD MODE (--no-vae, deprecated):                                      │
│  ══════════════════════════════════════                                      │
│                                                                              │
│  Phase 1: GP + Encoder Training                                             │
│  ─────────────────────────────────                                           │
│  Instructions (100) → GTR (768D) → Unit-Cube Norm → FeatureExtractor → 10D  │
│                                                         │                    │
│                                                         ▼                    │
│                                              DeepKernelGP (Matérn 5/2)       │
│                                                                              │
│  Phase 2: Decoder Training (frozen encoder)                                 │
│  ───────────────────────────────────────────                                 │
│  Diverse Instructions (1000+) → GTR → FeatureExtractor → 10D                │
│                                   │                        │                 │
│                                   ▼                        ▼                 │
│                          Target 768D        LatentDecoder → 768D             │
│                                                                              │
│                   L = λ_cycle × ||z - E(D(z))||²  + λ_cosine × (1 - cos)    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. BoTorch qLogEI Optimization in 10D Latent Space                        │
│  ───────────────────────────────────────────────────                         │
│                                                                              │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │  CompositeLogEI(z):                                              │    │
│     │    1. embedding = decoder(z)               # 10D → 768D          │    │
│     │    2. posterior = GP.posterior(embedding)  # GP prediction       │    │
│     │    3. LogEI = qLogExpectedImprovement(posterior, best_f)         │    │
│     │                                                                   │    │
│     │  optimize_acqf():                                                │    │
│     │    - raw_samples (512) → seed random points                      │    │
│     │    - num_restarts (20) → L-BFGS-B from best seeds                │    │
│     │    - Returns z* = argmax LogEI(z)                                │    │
│     └──────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│     z* = optimal latent in bounds [z_min - margin, z_max + margin]          │
│                                                                              │
│  2. Inversion Loop (default enabled, --no-inversion to disable)            │
│  ───────────────────────────────────────────────────────────────             │
│                                                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  for iter in 1..max_inversion_iters (10):                       │     │
│     │    1. embedding* = decoder(z*)                                  │     │
│     │    2. text* = Vec2Text(embedding*)                              │     │
│     │    3. z_inv = argmin ||decoder(z) - GTR(text*)||²              │     │
│     │       (solved via Adam, ~100 steps)                             │     │
│     │    4. gap = cosine_distance(embedding(z*), embedding(z_inv))    │     │
│     │       (measured in embedding space, NOT latent L2!)             │     │
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
│                         (50 steps, beam=8)                                  │
│                                                                              │
│  4. GP Update (every retrain_interval iterations)                           │
│  ─────────────────────────────────────────────────                           │
│                                                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  New observation: (GTR(text*), actual_error)                    │     │
│     │  GP.add_observation_and_retrain():                              │     │
│     │    - Preserve original normalization (X_min, X_max, y_mean)     │     │
│     │    - Warm-start feature extractor from previous weights         │     │
│     │    - Lower learning rate (0.001) for stability                  │     │
│     │    - Early stopping with patience                               │     │
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

Deep kernel encoder pro GP (používá se v non-VAE mode):

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

### 3. InstructionVAE (`encoder.py`)

Variational Autoencoder pro hladký latentní prostor (default mode):

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
- **Early stopping**: Sleduje **reconstruction loss** (ne total) - zabraňuje předčasnému zastavení během KL annealing

**Doporučené hyperparametry (defaults):**
```
--vae-beta 0.1          # Vyšší KL weight pro hladší latentní prostor
--vae-epochs 10000      # Delší trénink
--vae-annealing 500     # Pomalý ramp-up KL
--vae-patience 500      # Vysoká patience pro 10000 epochs
```

### 4. InstructionDeepKernelGP (`gp.py`)

Gaussian Process s deep kernel pro instruction optimization:

```python
class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
    """
    Inherits from GPyTorchModel for BoTorch compatibility.
    Enables use with qLogExpectedImprovement.
    """
```

**Kernel:**
- Matérn 5/2 s ARD (10 lengthscales)
- Prior na lengthscale: `Gamma(3.0, 6.0)`
- Prior na outputscale: `Gamma(2.0, 0.15)`
- **Noise constraint**: `GreaterThan(0.1)` - vyšší hodnota zabraňuje overconfidence
- **Standardize outcome transform**: BoTorch-kompatibilní denormalizace

### 5. GPWithEI (`gp.py`)

Wrapper pro GP s Expected Improvement:

```python
GPWithEI(
    device="cuda",
    latent_dim=10,
)
```

**Klíčové metody:**
- `set_training_data(embeddings, error_rates)`
- `train(epochs, lr, patience)` - s noise constraint 0.05
- `predict(embedding) → (mean, std)`
- `expected_improvement(embedding, xi) → EI`
- `log_expected_improvement(embedding, xi) → LogEI`
- `log_expected_improvement_tensor(embedding, xi) → LogEI tensor` (pro gradienty)
- `add_observation_and_retrain(embedding, error_rate)` - preserves normalization
- `get_training_size() → int`

### 6. BoTorch Acquisition (`botorch_acq.py`)

**CompositeLogEI** - Acquisition function pro optimalizaci přes decoder:

```python
class CompositeLogEI(AcquisitionFunction):
    """
    qLogEI that works through decoder transformation.

    Pipeline: z (10D) → decoder → embedding (768D) → GP → qLogEI

    Enables gradient-based optimization in latent space
    while evaluating improvement in embedding space.
    """
```

**LatentSpaceAcquisition** - Optimizer:

```python
class LatentSpaceAcquisition:
    """
    Uses BoTorch's optimize_acqf with multi-start L-BFGS-B.

    Advantages over random sampling:
    1. Gradient-based refinement finds local optima
    2. Multi-start avoids poor local optima
    3. Efficient for smooth acquisition landscapes
    """

    def optimize(
        self,
        best_f: float,
        num_restarts: int = 20,
        raw_samples: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (optimal_latent, log_ei_value)"""
```

### 7. LogEI Implementation (`gp.py`)

Numericky stabilní Log Expected Improvement z článku "Unexpected Improvements to Expected Improvement" (NeurIPS 2023):

```python
def log_h(z: float) -> float:
    """log(h(z)) where h(z) = φ(z) + z·Φ(z)

    Uses three branches for numerical stability:
    1. z > -1: Direct computation
    2. -1/√ε < z ≤ -1: erfcx-based
    3. z ≤ -1/√ε: Asymptotic approximation
    """

def log_h_tensor(z: torch.Tensor) -> torch.Tensor:
    """Tensor version with autograd support for gradient optimization."""

LogEI(x) = log_h(z) + log(σ)
where z = (y_best - μ - ξ) / σ
```

**Výhody:**
- Numericky stabilní i pro velmi malé EI hodnoty
- Umožňuje gradient-based optimalizaci (BoTorch)
- Nedochází k underflow (log místo tiny floats)

### 8. LatentDecoder (`decoder.py`)

Mirror architektura k encoderu (používá se pouze v non-VAE mode):

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

### 9. Vec2TextInverter (`inference.py`)

Lazy-loaded Vec2Text pro embedding-to-text inverzi:

```python
Vec2TextInverter(
    num_steps=50,      # Correction iterations
    beam_width=8,      # Beam search width (8 recommended)
    max_length=128,    # Max output tokens
    device="auto",
)
```

**Models:**
- `ielabgroup/vec2text_gtr-base-st_inversion`
- `ielabgroup/vec2text_gtr-base-st_corrector`

### 10. InvBOInference (`inference.py`)

Kompletní inference pipeline:

```python
InvBOInference(
    gp=trained_gp,
    decoder=trained_decoder,
    gtr=gtr_encoder,
    vec2text_steps=50,
    vec2text_beam=8,
    trust_region_config=None,  # NOT recommended
)
```

**Klíčové metody:**
- `get_best_training_latent() → (latent, idx, error)`
- `optimize_latent_botorch(num_restarts, raw_samples) → (z_opt, log_ei)` **[HLAVNÍ METODA]**
- `run_single_iteration(num_restarts, raw_samples, use_inversion, ...) → (result, gap, log_ei)`
- `inversion_step(text) → InversionStepResult` (InvBO-style inversion)
- `validate_inversion_gap(n_samples) → stats`

### 11. Visualization (`visualize.py`)

UMAP-based EI landscape visualization:

```python
visualize_ei_landscape(
    inference=inference,
    center_latent=z_opt,
    realized_text=generated_text,
    best_y=best_error,
    trust_region=None,  # NOT recommended
    span=1.0,
    n_grid_samples=300,
    save_path="ei_landscape.png",
)
```

## Datové Soubory

| Soubor | Popis | Použití |
|--------|-------|---------|
| `datasets/inversion/instructions_100.txt` | 100 instrukcí v 10 kategoriích | GP training, baseline |
| `datasets/inversion/grid_100_qend.jsonl` | Error rates pro 100 instrukcí | GP targets |
| `datasets/inversion/diverse_instructions_1000.json` | 1000+ diverse instrukcí (APE) | VAE training |

## Loss Funkce

### VAE Mode (default)

```python
# VAE Training
L_vae = (1 - cosine_sim(x, x_recon)) + β × KL(q(z|x) || N(0,1))

# β annealing: 0 → vae_beta over vae_annealing_epochs
β_current = vae_beta × min(1, epoch / vae_annealing_epochs)

# Early stopping sleduje POUZE reconstruction loss (ne total)
# To zabraňuje předčasnému zastavení během KL annealing
```

### Standard Mode (deprecated)

```python
# Phase 1: GP Training
L_gp = -MLL(GP(encoder(normalize(embedding))), error_rate)

# Phase 2: Decoder Training
L_decoder = λ_cycle × ||z - encoder(normalize(decoder(z)))||²
          + λ_cosine × (1 - cosine_sim(decoder(z), target_embedding))
```

## Normalizace

### Unit-Cube Normalization (pro GP vstupy)
```python
X_norm = (X - X_min) / (X_max - X_min)
# X_min, X_max z training dat (100 instrukcí)
# Preserved during incremental retraining
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
| `--no-vae` | False | Vypnout VAE mode (VAE je default) |
| `--visualize` | False | Generovat EI landscape vizualizace |

### BoTorch Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--n-restarts` | 20 | Počet L-BFGS-B restarts pro BoTorch qLogEI |
| `--raw-samples` | 512 | Raw samples pro inicializaci optimalizace |

### VAE Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--vae-beta` | 0.1 | KL regularization weight (vyšší = hladší prostor) |
| `--vae-epochs` | 10000 | VAE training epochs |
| `--vae-annealing` | 500 | KL annealing epochs (0 → beta) |
| `--vae-patience` | 500 | Early stopping patience |

### Inversion Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--use-inversion` | True | Použít InvBO inversion loop (default) |
| `--no-inversion` | False | Vypnout inversion loop |
| `--max-inversion-iters` | 10 | Maximum inversion loop iterations |
| `--gap-threshold` | 0.1 | Cosine distance threshold pro accept |

### Vec2Text Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--vec2text-steps` | 50 | Correction steps |
| `--vec2text-beam` | 8 | Beam width (8 recommended) |

### GP Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--gp-epochs` | 10000 | GP training epochs |
| `--gp-patience` | 100 | GP early stopping patience |
| `--retrain-interval` | 1 | Retrain GP every N iterations |
| `--retrain-epochs` | 10000 | Epochs pro GP retraining |

### LLM Evaluation

| Parametr | Default | Popis |
|----------|---------|-------|
| `--model` | Qwen/Qwen2.5-7B-Instruct | Model pro evaluaci |
| `--eval-samples` | 1319 | Samples (1319 = plný GSM8K validation) |

### Trust Region (ENABLED BY DEFAULT)

| Parametr | Default | Popis |
|----------|---------|-------|
| `--no-trust-region` | False | Vypnout trust region (je zapnutý defaultně) |
| `--tr-initial` | 0.5 | Initial trust region radius |
| `--tr-min` | 0.05 | Minimum radius (triggers restart) |
| `--tr-max` | 2.0 | Maximum radius |

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
      "trust_region_radius": null,
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
- Gap měřen jako **cosine distance v embedding space** (stabilnější než L2 v latentu)
- Threshold 0.1 pro accept/reject

### 2. GP Overconfidence
**Problém**: GP může být příliš jistý svými predikcemi, což vede k LogEI = -5000.

**Řešení**:
- Noise constraint zvýšen na `GreaterThan(0.05)`
- Zajišťuje dostatečnou uncertainty pro exploration

### 3. Posterior Collapse (VAE)
**Problém**: VAE může ignorovat latenty (z → konstanta).

**Řešení**:
- KL annealing: β = 0 → target přes `vae_annealing_epochs`
- Nižší `vae_beta` (0.01)
- Early stopping sleduje **reconstruction loss** (ne total)
- Delší trénink s vysokou patience (10000 epochs, patience 500)

### 4. Vec2Text Limitations
**Problém**: Selhává pro delší texty (>30 tokenů).

**Řešení**:
- Instruction-only přístup (ne celé prompty)
- `max_length=128` v Vec2Text

### 5. Numerical Instability in EI
**Problém**: Standard EI underflows pro velmi malé hodnoty.

**Řešení**:
- BoTorch `qLogExpectedImprovement` - numericky stabilní
- `log_h_tensor()` s autograd pro gradient optimization
- Tři branches pro různé rozsahy z-score

### 6. Hladký latentní prostor
**Problém**: Latentní prostor může mít "díry" kde optimalizace selhává.

**Řešení**:
- **VAE beta=0.1** pro hladší latentní prostor
- **Trust region** omezuje exploration do známých oblastí
- **GP noise=0.1** zabraňuje overconfidence

## Porovnání s COWBOYS

| Feature | InvBO Decoder | COWBOYS |
|---------|---------------|---------|
| Latent Dim | 10D | 32D |
| Encoder | VAE (default) / Deep Kernel | VAE only |
| VAE Beta | 0.1 (hladký prostor) | 0.01 |
| Optimizer | BoTorch qLogEI (gradient) | pCN MCMC (sampling) |
| Trust Region | **ENABLED** | Custom (L2) |
| Inversion | InvBO-style loop (10 iters) | Direct Vec2Text |
| GP Retraining | Incremental (preserved norm) | Batch |
| Noise Constraint | 0.1 (prevents overconfidence) | Default |
| Gap Metric | Cosine in embedding space | L2 in latent |
| Standardize | BoTorch outcome transform | Manual |

## Reference

- **InvBO**: Deshwal et al., 2024 - "Inversion-Based BO with Structured Inputs" (NeurIPS 2024)
- **LogEI**: Ament et al., 2023 - "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)
- **Vec2Text**: Morris et al., 2023 - "Text Embeddings Reveal (Almost) As Much As Text"
- **BoTorch**: Balandat et al., 2020 - "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization"
