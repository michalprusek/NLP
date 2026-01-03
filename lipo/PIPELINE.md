# LIPO - Latent Instruction Prompt Optimization

## Přehled

**LIPO (Latent Instruction Prompt Optimization)** je instruction-only optimalizační framework kombinující:
1. **APE (Automatic Prompt Engineering)** - generování diverzních instrukcí
2. **VAE (Variational Autoencoder)** - mapování do hladkého latentního prostoru
3. **Hyperband** - efektivní multi-fidelity evaluace
4. **InvBO (Inversion Bayesian Optimization)** - optimalizace v latentním prostoru + inverze zpět na text

Na rozdíl od standardního HbBoPs pracuje pouze s instrukcemi (bez exemplárů), používá GTR embeddingy (kompatibilní s Vec2Text) a optimalizuje ve VAE latentním prostoru.

---

## Architektura pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LIPO FULL PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

FÁZE 1: GENEROVÁNÍ INSTRUKCÍ
┌──────────────────────────────────────────┐
│ APE Generace (2000 instrukcí)            │
│ - 5 stylových kategorií                  │
│ - Batch generace (10 per batch)          │
│ - Caching do JSON                        │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ GTR Encoding (768D embeddingy)           │
│ - SentenceTransformer GTR-T5-Base        │
│ - L2-normalizované (pro Vec2Text)        │
└──────────────────────────────────────────┘

FÁZE 2: VAE TRÉNINK
┌──────────────────────────────────────────┐
│ InstructionVAE Training                  │
│ - Input: 768D GTR embeddingy             │
│ - Latent: 16D (hladký přes KL)           │
│ - KL annealing: β roste od 0 do β_max    │
│ - Cycle consistency: γ·cycle_loss        │
│ - Output: L2-normalizovaná rekonstrukce  │
│ - Loss: cosine_recon + β·KL + γ·cycle    │
└──────────────────────────────────────────┘

FÁZE 3: HYPERBAND EVALUACE
┌──────────────────────────────────────────┐
│ LIPO Hyperband                           │
│ - Successive halving s BO návrhy         │
│ - Multi-fidelity: bmin až full fidelity  │
│ - GP trénovaný na top-75% fidelity       │
│ - Evaluator: LLM na validation subsetu   │
│ - Caching: (instruction_id, fidelity)    │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ GP Training (z Hyperband dat)            │
│ - VAEWithAdapter: VAE(frozen) + MLP      │
│ - 16D VAE latent → 10D (přes adapter)    │
│ - Kumaraswamy input warping (dynamic D)  │
│ - Matern 5/2 kernel s ARD (10D)          │
│ - FixedNoiseGaussianLikelihood           │
│ - Heteroscedastic noise: p(1-p)/n        │
│ - Training: adapter + kernel learnable   │
└──────────────────────────────────────────┘

FÁZE 4: InvBO INFERENCE
┌──────────────────────────────────────────┐
│ BoTorch qLogEI Optimalizace              │
│ - Optimalizace 16D VAE latent            │
│ - TuRBO trust region (optional)          │
│ - PAS anchor selection (optional)        │
│ - Multi-start L-BFGS-B (64 restartů)     │
│ - Raw samples inicializace (1024)        │
│ - LogEI: numericky stabilní EI           │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ Candidate Rejection Loop                 │
│ - Cosine sim threshold: 0.90             │
│ - Max rejection attempts: 5              │
│ - Different seed per attempt             │
│ - Fallback: accept low-quality candidate │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ VAE Decoder + Vec2Text Inverze           │
│ - z_opt (16D) → decoder → embedding (768D)
│ - embedding → Vec2Text → text            │
│ - 512_tokens model (max 128 tokenů)      │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ InvBO Inversion Loop                     │
│ - Najdi z_inv: GTR(text) ≈ decoder(z_inv)│
│ - Adam optimalizace (100 kroků)          │
│ - Cosine loss minimalizace               │
│ - Re-inverze pokud gap > threshold       │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ GP Update & Retraining                   │
│ - Re-encode text s GTR                   │
│ - Predikce error rate s GP               │
│ - LLM evaluace (volitelná)               │
│ - Laplace smoothing: (err*n+1)/(n+2)     │
│ - Přidání do training dat                │
│ - Full retrain (1000 epoch)              │
└──────────────────────────────────────────┘
```

---

## Struktura souborů

```
lipo/
├── config.py           # Unified configuration (SSOT) + get_device() utility
├── encoder.py          # GTRInstructionEncoder, InstructionVAE, VAEWithAdapter
├── gp.py               # InstructionDeepKernelGP, GPWithEI, LogEI funkce
├── hyperband.py        # LIPOHyperband (Hyperband + BO)
├── training.py         # APEGenerator, LIPOHyperbandTrainer
├── inference.py        # Vec2TextInverter, LIPOHyperbandInference, _log_iteration_summary()
├── botorch_acq.py      # CompositeLogEI, LatentSpaceAcquisition
├── evaluate.py         # GSM8KEvaluator
├── instruction.py      # InstructionOnlyPrompt dataclass
├── hbbops_results.py   # Extract/save HbBoPs evaluation results
├── run.py              # CLI entry point
└── data/
    └── ape_instructions.json  # Cache generovaných instrukcí
```

---

## Konfigurace (config.py)

### Utility funkce

**`get_device(device: str = "auto") -> str`** - Centralizovaná detekce zařízení:
```python
from lipo.config import get_device

device = get_device("auto")  # "cuda" pokud dostupné, jinak "cpu"
device = get_device("cuda")  # Force CUDA
device = get_device("cpu")   # Force CPU
```

Tato funkce je sdílená napříč všemi moduly (encoder.py, hyperband.py, inference.py),
eliminující duplicitní implementace.

### Config dataclass

Veškeré parametry jsou centralizované v `Config` dataclass:

### APE Generace
| Parametr | Default | Popis |
|----------|---------|-------|
| `ape_num_instructions` | 2000 | Počet instrukcí k vygenerování |
| `ape_model` | `Qwen/Qwen2.5-7B-Instruct` | Model pro generování |
| `ape_backend` | `vllm` | Backend (vllm/openai/deepinfra) |
| `ape_cache_path` | `lipo/data/ape_instructions.json` | Cesta k cache |
| `ape_batch_size` | 10 | Velikost batche |
| `ape_max_tokens` | 100 | Max tokenů na instrukci |
| `ape_max_length` | 500 | Max délka znakově |

### VAE Trénink
| Parametr | Default | Popis |
|----------|---------|-------|
| `vae_beta` | 0.015 | KL regularizace (5x silnější pro latent_dim=16) |
| `vae_gamma` | 1.0 | Cycle consistency weight: z ≈ encode(decode(z)) |
| `vae_epochs` | 15000 | Max počet epoch |
| `vae_annealing_epochs` | 500 | Počet epoch pro KL annealing |
| `vae_patience` | 500 | Early stopping patience |
| `vae_lr` | 0.0006 | Learning rate |
| `vae_batch_size` | 64 | Batch size |
| `vae_grad_clip` | 1.0 | Gradient clipping |
| `vae_eta_min` | 1e-4 | Min LR pro cosine scheduler |

### Latentní dimenze
| Parametr | Default | Popis |
|----------|---------|-------|
| `embedding_dim` | 768 | GTR embedding dimenze |
| `latent_dim` | 16 | VAE latentní dimenze (kompaktnější pro hustší reprezentaci) |
| `gp_latent_dim` | 10 | Adapter output pro GP |

### Hyperband
| Parametr | Default | Popis |
|----------|---------|-------|
| `bmin` | 10 | Minimální fidelity (vzorků) |
| `eta` | 2.0 | Downsampling rate |
| `random_interleaving_prob` | 0.1 | Pravděpodobnost random návrhu (10%) |
| `min_fidelity_pct` | 0.75 | GP trénován na top 25% fidelity |

### GP Trénink
| Parametr | Default | Popis |
|----------|---------|-------|
| `gp_epochs` | 10000 | Max počet epoch |
| `gp_lr` | 0.01 | Learning rate |
| `gp_patience` | 100 | Early stopping patience |

Pro Hyperband (rychlý retrain):
```python
for_hyperband_gp() → {epochs: 3000, lr: 0.01, patience: 50}
```

### Inference
| Parametr | Default | Popis |
|----------|---------|-------|
| `num_restarts` | 64 | L-BFGS-B restarty |
| `raw_samples` | 1024 | Inicializační vzorky |
| `use_inversion` | True | Použít InvBO inversion loop |
| `max_inversion_iters` | 3 | Max iterací inverze |
| `gap_threshold` | 0.1 | Threshold pro re-inverzi (cosine distance) |
| `cosine_sim_threshold` | 0.90 | Min cosine similarity pro akceptaci kandidáta |
| `max_rejection_attempts` | 5 | Max pokusů před fallback akceptací |

### TuRBO (Trust Region)
| Parametr | Default | Popis |
|----------|---------|-------|
| `turbo_enabled` | True | Povolit trust region optimalizaci |
| `turbo_L_init` | 0.8 | Počáteční velikost regionu (% unit cube) |
| `turbo_L_max` | 1.6 | Maximální velikost regionu |
| `turbo_L_min` | 0.0078 | Minimální velikost (0.5^7, trigger restart) |
| `turbo_tau_succ` | 3 | Počet úspěchů pro zdvojnásobení L |
| `turbo_tau_fail` | 40 | Počet neúspěchů pro zmenšení L |

### PAS (Potential-Aware Anchor Selection)
| Parametr | Default | Popis |
|----------|---------|-------|
| `pas_enabled` | True | Povolit potential-aware anchor selection |
| `pas_n_candidates` | 100 | Počet kandidátů pro Thompson Sampling |

### Vec2Text
| Parametr | Default | Popis |
|----------|---------|-------|
| `vec2text_beam` | 8 | Beam width |
| `vec2text_model` | `512_tokens` | Model typ |
| `vec2text_max_length` | 128 | Max output tokenů |

### Inverze optimalizace
| Parametr | Default | Popis |
|----------|---------|-------|
| `inversion_n_steps` | 100 | Adam kroky |
| `inversion_lr` | 0.1 | Adam learning rate |
| `inversion_convergence_threshold` | 0.01 | Early stop threshold |
| `latent_margin` | 0.2 | Rozšíření bounds (20%) |

### GP Retrain (inference)
| Parametr | Default | Popis |
|----------|---------|-------|
| `gp_retrain_epochs` | 1000 | Počet epoch |
| `gp_retrain_lr` | 0.001 | Learning rate |
| `gp_retrain_patience` | 50 | Patience |

---

## Komponenty v detailu

### 1. GTRInstructionEncoder (encoder.py)

Wrapper kolem SentenceTransformer GTR-T5-Base:

```python
class GTRInstructionEncoder:
    model_name: str = "sentence-transformers/gtr-t5-base"
    embedding_dim: int = 768
    normalize: bool = True  # L2 normalizace (pro Vec2Text)
```

**Metody:**
- `encode(text) → np.ndarray[768]` - NumPy array
- `encode_tensor(text) → torch.Tensor[768]` - PyTorch tensor
- `encode_batch_tensor(texts) → torch.Tensor[N, 768]` - Batch encoding

### 2. InstructionVAE (encoder.py)

Variational autoencoder pro hladký latentní prostor:

```
Encoder: 768D → 384 → GELU → LayerNorm → 192 → GELU → LayerNorm → 2×16 (mu + log_var)
Decoder: 16 → 192 → GELU → LayerNorm → 384 → GELU → LayerNorm → 768D (L2 normalized)
```

**GELU aktivace:** Hladší gradienty než ReLU, používá se v transformerech.
**Xavier inicializace:** Všechny Linear vrstvy používají `xavier_uniform_` pro stabilní trénink.

**Loss funkce:**
```python
recon_loss = mean(1 - cosine_similarity(x, x_recon))
kl_loss = -0.5 * mean(sum(1 + log_var - mu² - exp(log_var)))
cycle_loss = mean(1 - cosine_similarity(z, encode_mu(decode(z))))  # Cycle consistency
total_loss = recon_loss + beta(epoch) * kl_loss + gamma * cycle_loss
```

**KL Annealing:**
```python
if epoch < annealing_epochs:
    beta = vae_beta * (epoch / annealing_epochs)  # Lineární růst od 0
else:
    beta = vae_beta  # Konstanta (0.015 pro latent_dim=16)
```

**Cycle consistency:** Zajišťuje, že z ≈ encode(decode(z)), čímž předchází "dírám" v latentním prostoru.

**Účel:** Zajišťuje hladký latentní prostor (žádné "díry"), distribuci kolem N(0,1), a lepší generalizaci.

### 3. VAEWithAdapter (encoder.py)

Kombinuje zmrazenou VAE s trénovatelným adaptérem:

```
x (768D) → VAE.encode_mu (frozen) → z_vae (16D) → adapter → z_gp (10D)
```

**Architektura adaptéru:**
```python
adapter = nn.Sequential(
    nn.Linear(16, 20),   # 16 → 20 (intermediate)
    nn.ReLU(),
    nn.LayerNorm(20),
    nn.Linear(20, 10),   # 20 → 10
)
```

**Metody:**
- `encode_vae(x)` - 768D → 16D (pouze VAE encoder)
- `forward(x)` - 768D → 10D (celá pipeline)
- `adapt(z)` - 16D → 10D (pouze adapter)
- `decode(z)` - 16D → 768D (pouze VAE decoder)

### 4. InstructionDeepKernelGP (gp.py)

GP model s deep kernel pro instrukční optimalizaci:

```python
class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
    # Kumaraswamy input warping (dynamická dimenze z dat)
    input_transform = Warp(
        d=input_dim,  # Dynamicky z train_x.shape[-1]
        indices=list(range(input_dim)),
    )

    mean_module = ZeroMean()
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,  # Matern 5/2
            ard_num_dims=10,  # ARD na adapter output
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
```

**Forward pass:**
```python
def forward(self, x):  # x: (batch, 16) - 16D VAE latent
    latent = self.adapter(x)  # (batch, 10)
    return MultivariateNormal(
        self.mean_module(latent),
        self.covar_module(latent),
    )
```

**DŮLEŽITÉ:** Warp dimenze je nyní dynamická (`d=self._input_dim` z `train_x.shape[-1]`),
nikoliv hardcoded, což zabraňuje CUDA "index out of bounds" chybám při změně `latent_dim`.

### 5. GPWithEI (gp.py) - DETAILNÍ POPIS

`GPWithEI` je wrapper pro Gaussian Process s Expected Improvement acquisition function.
Kombinuje normalizaci dat, heteroscedastic noise modeling, a správnou konverzi
error rates pro BoTorch maximizaci.

#### 5.1 Architektura

```
                    GPWithEI Architecture
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  768D embedding                                                │
│       ↓                                                        │
│  VAEWithAdapter.encode_vae() [frozen]                          │
│       ↓                                                        │
│  16D VAE latent (X_train)                                      │
│       ↓                                                        │
│  Unit cube normalization: (X - X_min) / (X_max - X_min)        │
│       ↓                                                        │
│  ┌────────────────────────────────────────┐                    │
│  │ InstructionDeepKernelGP                │                    │
│  │   ├─ input_transform: Warp(d=16)       │                    │
│  │   ├─ adapter: 16D → 10D [trainable]    │                    │
│  │   ├─ mean_module: ZeroMean             │                    │
│  │   └─ covar_module: ScaleKernel(        │                    │
│  │         MaternKernel(nu=2.5, ARD=10D)) │                    │
│  └────────────────────────────────────────┘                    │
│       ↓                                                        │
│  MultivariateNormal(mean, covariance)                          │
│       ↓                                                        │
│  FixedNoiseGaussianLikelihood                                  │
│    ├─ noise = Bernoulli variance p(1-p)/n                      │
│    └─ learn_additional_noise = False (CUDA stability)          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### 5.2 Negace Error Rate pro BoTorch

**KRITICKÉ ROZHODNUTÍ:** BoTorch's `qLogExpectedImprovement` MAXIMALIZUJE by default.
Pro minimalizaci error rate musíme negovat hodnoty:

```python
# set_training_data()
self._error_rates_original = error_rates  # Positive, pro noise výpočet
self.y_train = -error_rates               # Negative, pro GP training

# y_best je maximum negovaných error rates = minimum originálních
self.y_best = self.y_train.max()  # max(-error) = -min(error)
```

**Příklad:**
```
Instrukce A: error_rate = 0.10 → y_train = -0.10
Instrukce B: error_rate = 0.25 → y_train = -0.25
Instrukce C: error_rate = 0.05 → y_train = -0.05  ← best!

y_best = max(-0.10, -0.25, -0.05) = -0.05

EI hledá body kde GP predikuje hodnoty > -0.05 (= error < 0.05)
```

**Při predikci negujeme zpět:**
```python
def predict(self, embedding):
    # GP predikuje -error_rate (standardizované)
    neg_error = mean_norm * y_std + y_mean

    # Neguj zpět na pozitivní error_rate
    mean = -neg_error  # Positive error rate
    return mean, std
```

#### 5.3 Heteroscedastic Noise (Bernoulli Variance)

GP dostává informaci o spolehlivosti každé observace prostřednictvím
heteroscedastic noise. Variance je počítána z Bernoulli statistiky:

```python
def _compute_observation_noise(self, y, fidelity):
    """
    Pro error_rate měřenou na n vzorcích:
    Var = p(1-p)/n

    kde:
    - p = error_rate (proportion of errors)
    - n = fidelity (number of samples)
    """
    variance = (y * (1 - y)) / fidelity

    # Clamp to prevent numerical issues
    variance = torch.clamp(variance, min=1e-6, max=0.1)

    return variance
```

**Příklad vlivu fidelity:**
```
Instrukce s error_rate = 0.20:

fidelity=10:   Var = 0.20 * 0.80 / 10   = 0.016   (HIGH uncertainty)
fidelity=100:  Var = 0.20 * 0.80 / 100  = 0.0016  (medium uncertainty)
fidelity=1319: Var = 0.20 * 0.80 / 1319 = 0.00012 (LOW uncertainty)
```

GP pak více důvěřuje high-fidelity observacím.

#### 5.4 FixedNoiseGaussianLikelihood

```python
# Noise je transformován do standardizovaného prostoru
raw_noise = self._compute_observation_noise(error_rates_original, fidelity_train)
y_std_sq = max(y_std.item() ** 2, 1e-6)  # Robustní extrakce
noise_standardized = raw_noise / y_std_sq
noise_standardized = torch.clamp(noise_standardized, min=1e-4, max=1.0)
noise_standardized = noise_standardized.view(-1).contiguous()

# Likelihood s fixním heteroscedastic noise
self.likelihood = FixedNoiseGaussianLikelihood(
    noise=noise_standardized,
    learn_additional_noise=False,  # CUDA stability - avoid second_noise_covar errors
)
```

**Proč `learn_additional_noise=False`?**
- `learn_additional_noise=True` způsoboval CUDA chyby v `second_noise_covar`
- Bernoulli variance p(1-p)/n je dostatečná pro modelování observačního šumu
- Vyšší stabilita při trénování GP

#### 5.5 Normalizace Detaily

**Input normalizace (Unit cube):**
```python
# Každá dimenze normalizována do [0, 1]
X_min = X_train.min(dim=0)[0]  # (16,)
X_max = X_train.max(dim=0)[0]  # (16,)
denom = X_max - X_min
denom[denom == 0] = 1.0  # Avoid division by zero for constant dims

X_norm = (X_train - X_min) / denom  # (N, 16) in [0, 1]
```

**Output normalizace (Z-score via BoTorch Standardize):**
```python
self.outcome_transform = Standardize(m=1)
y_transformed, _ = self.outcome_transform(y_train.unsqueeze(-1))
y_norm = y_transformed.squeeze(-1)

# outcome_transform ukládá mean a std pro denormalizaci
self.y_mean = self.outcome_transform.means  # scalar
self.y_std = self.outcome_transform.stdvs   # scalar
```

#### 5.6 Training Loop

```python
def train(self, epochs=10000, lr=0.01, patience=100):
    self.gp_model.train()
    self.likelihood.train()

    optimizer = AdamW(self.gp_model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

    best_loss = float("inf")
    patience_counter = 0

    with gpytorch.settings.cholesky_jitter(1e-4):
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass through GP
            output = self.gp_model(X_norm)  # MultivariateNormal

            # Negative marginal log-likelihood
            loss = -mll(output, y_norm)

            loss.backward()
            optimizer.step()

            # Early stopping on loss plateau
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break
```

**Cholesky jitter:** `1e-4` pro numerickou stabilitu při inverzi covariance matice.

#### 5.7 Validace GP

**POZNÁMKA:** GP validace na trénovacích datech byla odstraněna, protože dávala
optimisticky zkreslený výsledek. GP se nyní validuje pouze na inference výsledcích:
porovnáním `predicted_error` vs `actual_error` v každé iteraci.

Metriky pro hodnocení kvality GP predikcí jsou součástí iteration summary:
- Predicted error vs actual error
- Optimization gap metrics (z_opt vs z_real)

#### 5.8 Prediction Pipeline

```python
def predict(self, embedding):
    """
    embedding: 768D GTR embedding
    returns: (mean_error, std_error) as positive values
    """
    # 1. Encode to VAE latent if needed
    if embedding.shape[-1] == 768:
        z_vae = self.vae_with_adapter.encode_vae(embedding)  # 768D → 16D
    else:
        z_vae = embedding

    # 2. Normalize to unit cube
    X_norm = (z_vae - self.X_min) / (self.X_max - self.X_min)

    # 3. GP prediction (in standardized -error space)
    # DŮLEŽITÉ: Používáme gp_model přímo, NE přes likelihood!
    # FixedNoiseGaussianLikelihood ukládá noise pro trénovací body,
    # proto jej nepoužíváme pro predikci na nových bodech
    self.gp_model.eval()
    self.likelihood.eval()
    with torch.no_grad():
        posterior = self.gp_model(X_norm)  # Přímo z GP modelu
        mean_norm = posterior.mean
        std_norm = posterior.stddev

    # 4. Denormalize from standardized space
    neg_error = mean_norm * self.y_std + self.y_mean
    std = std_norm * self.y_std

    # 5. Negate back to positive error rate
    mean = -neg_error  # CRITICAL: GP predicts -error, return +error

    return mean, std
```

**DŮLEŽITÉ:** Predikce nyní používá `self.gp_model(X_norm)` přímo místo
`self.likelihood(self.gp_model(X_norm))`. Toto zabraňuje `GPInputWarning`
o nesouladu velikosti šumu při predikci na nových bodech.

#### 5.9 add_observation s Laplace Smoothing

Při přidávání nových observací aplikujeme Laplace smoothing pro konzistenci
s trénovacími daty:

```python
def add_observation(self, embedding, error_rate, fidelity=1319):
    """
    Přidá novou observaci do training dat.

    Args:
        embedding: 768D GTR embedding
        error_rate: Positive error rate [0, 1]
        fidelity: Number of samples used for evaluation
    """
    # Validate inputs
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")

    # Encode to VAE latent (768D → 16D)
    new_z = self.vae_with_adapter.encode_vae(embedding)

    # Laplace smoothing: (errors + 1) / (n + 2)
    # Penalizes "lucky guesses" on low-fidelity samples
    num_errors = error_rate * fidelity
    smoothed_error = (num_errors + 1) / (fidelity + 2)

    # Store original (smoothed) for noise computation
    self._error_rates_original = torch.cat([
        self._error_rates_original,
        torch.tensor([smoothed_error])
    ])

    # Negate for GP (BoTorch maximization)
    self.y_train = torch.cat([
        self.y_train,
        torch.tensor([-smoothed_error])
    ])

    self.fidelity_train = torch.cat([
        self.fidelity_train,
        torch.tensor([fidelity])
    ])

    self.X_train = torch.cat([self.X_train, new_z])

    # Update best if improved
    if -smoothed_error > self.y_best:
        self.y_best = -smoothed_error
```

**Laplace smoothing příklad:**
```
Raw error 0/10 (0.0)   → smoothed: (0+1)/(10+2)   = 0.083
Raw error 0/100 (0.0)  → smoothed: (0+1)/(100+2)  = 0.0098
Raw error 0/1319 (0.0) → smoothed: (0+1)/(1319+2) = 0.00076

Low-fidelity "perfect" results jsou penalizovány více!
```

#### 5.10 LogEI Implementace

Numericky stabilní log Expected Improvement z "Unexpected Improvements to
Expected Improvement" (NeurIPS 2023):

```python
def log_h(z):
    """
    log(h(z)) kde h(z) = φ(z) + z·Φ(z)

    Tři větve pro numerickou stabilitu:
    """
    if z > -1:
        # Direct computation
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        h_val = phi_z + z * Phi_z
        return math.log(h_val)

    elif z > -1 / math.sqrt(EPS):
        # erfcx-based computation
        z_scaled = -z / math.sqrt(2)
        erfcx_val = erfcx(z_scaled)
        inner = math.log(erfcx_val * abs(z)) + C2
        return -z*z/2 - C1 + log1mexp(inner)

    else:
        # Asymptotic approximation
        return -z*z/2 - C1 - 2*math.log(abs(z))

def log_expected_improvement(self, embedding, xi=0.01):
    mean, std = self.predict(embedding)  # Positive error rates
    best = self.best_error_rate          # Positive best error

    # z-score: improvement / uncertainty
    z = (best - mean - xi) / std

    # LogEI = log_h(z) + log(σ)
    return log_h(z) + math.log(std)
```

#### 5.11 best_error_rate Property

```python
@property
def best_error_rate(self):
    """
    Get best observed error rate (positive).

    Internally y_best stores -min_error for BoTorch compatibility.
    This property returns the positive error rate.
    """
    if self.y_best is None:
        return None
    return -self.y_best  # Negate back: -(-min_error) = min_error
```

#### 5.12 Souhrn GP Flow

```
TRAINING:
  embeddings (768D) → VAE encoder → latents (16D)
                                        ↓
  error_rates → negate → y_train = -error_rates
                               ↓
  fidelities → Bernoulli variance → noise = p(1-p)/n
                                        ↓
  Unit cube normalize X, Z-score normalize y
                    ↓
  Train adapter + GP kernel + Warp jointly (MLL loss)

PREDICTION:
  embedding (768D) → VAE encoder → latent (16D)
                                      ↓
  Normalize to unit cube
                    ↓
  GP forward (BEZ likelihood!) → MultivariateNormal(mean, var)
                    ↓
  Denormalize: neg_error = mean * y_std + y_mean
                    ↓
  Negate back: error = -neg_error
                    ↓
  Return (error, std) as positive values

ADD OBSERVATION:
  new_embedding (768D) + error_rate + fidelity
                    ↓
  Laplace smoothing: (errors + 1) / (fidelity + 2)
                    ↓
  Append to X_train, y_train, fidelity_train
                    ↓
  Update y_best if improved
                    ↓
  Call train() to retrain GP from scratch
```

**POZNÁMKA:** GP validace probíhá výhradně na inference výsledcích (predicted vs actual error).
Predikce používá `self.gp_model(X_norm)` přímo, NE `self.likelihood(self.gp_model(X_norm))`.

### 6. LIPOHyperband (hyperband.py)

Hyperband algoritmus s BO návrhy:

**Hyperband schedule:**
```python
r = nvalid / bmin  # Poměr full fidelity k min fidelity
smax = floor(log(r) / log(η))  # Max bracket index
B = (smax + 1) * nvalid  # Celkový budget
```

Pro každý bracket s od smax dolů k 0:
- Počáteční n = ceil((B/nvalid) * η^s / (s+1)) instrukcí
- Pro každé stage i v 0..s:
  - n_i = floor(n * η^-i) instrukcí zachovat
  - b_i = bmin * η^i vzorků na evaluaci
  - Successive halving: zachovat top n_i

**GP Training (během Hyperband):**
- Pouze na high-fidelity datech (≥ 75% full fidelity)
- Rychlé parametry: 3000 epoch, patience 50

**BO Proposals:**
- 10% random interleaving (explorace)
- 90% maximize Expected Improvement (exploitace)

**Evaluation caching:**
```python
cache_key = (instruction_id, fidelity)
# Rozšíření z nižší fidelity:
total_error = (prev_error * prev_f + new_error * (f - prev_f)) / f
```

### 7. Vec2TextInverter (inference.py)

Embedding-to-text inverze:

**Dva modely:**
- `32_tokens`: ielabgroup corrector + inversion model (starší, limitovaný)
- `512_tokens`: Vec2Text InversionModel (doporučený, delší sekvence)

**Generace (512_tokens):**
```python
gen_kwargs = {
    "num_beams": beam_width,  # default 8
    "max_length": 128,  # zvýšeno z 32
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
}
```

### 8. LIPOHyperbandInference (inference.py)

Kompletní inference pipeline:

**Candidate Rejection Loop:**
```python
for attempt in range(max_rejection_attempts):  # default 5
    z_opt, log_ei = optimize_latent_botorch(seed=iteration * 5 + attempt)
    embedding = vae_with_adapter.decode(z_opt)
    instruction = vec2text.invert(embedding)

    reencoded = GTR.encode(instruction)
    cosine_sim = cosine_similarity(embedding, reencoded)

    if cosine_sim >= cosine_sim_threshold:  # default 0.90
        break  # Dobrý kandidát
    elif attempt == max_rejection_attempts - 1:
        print("WARNING: Accepting low-quality candidate")
        # Použij i přes nízkou kvalitu
```

**Optimalizační tok:**

1. **optimize_latent_botorch()**
   ```python
   # BoTorch qLogEI s multi-start L-BFGS-B
   bounds = get_latent_bounds(X_train, margin=0.2)  # 20% rozšíření
   acq_fn = CompositeLogEI(model=gp_model, best_f=y_best)
   z_opt, log_ei = optimize_acqf(
       acq_function=acq_fn,
       bounds=bounds,
       q=1,
       num_restarts=64,
       raw_samples=1024,
       options={"maxiter": 200},
   )
   ```

2. **Decode & Invert**
   ```python
   z_unnorm = z_opt * (X_max - X_min) + X_min  # Denormalizace
   embedding = vae_with_adapter.decode(z_unnorm)  # 16D → 768D
   text = vec2text.invert(embedding)  # 768D → text
   ```

3. **Inversion Loop**
   ```python
   for inv_iter in range(max_inversion_iters):
       # Najdi z_inv: ||decoder(z_inv) - GTR(text)|| minimální
       z = z_init.clone().requires_grad_(True)
       optimizer = Adam([z], lr=0.1)

       for step in range(100):
           z_vae = z * x_range + X_min  # Denormalizace do 16D
           decoded = vae.decode(z_vae)
           loss = 1 - cosine_similarity(decoded, GTR(text))
           loss.backward()
           optimizer.step()

       gap = 1 - cosine_similarity(original_emb, inverted_emb)
       if gap <= gap_threshold:
           break

       # Re-decode a re-invert
       embedding = vae.decode(z_inv)
       text = vec2text.invert(embedding)
   ```

4. **Evaluation & GP Update**
   ```python
   reencoded = GTR.encode(text)
   pred_error, pred_std = GP.predict(reencoded)
   actual_error = LLM.evaluate(text, validation_data)  # Volitelné

   # Laplace smoothing pro nové observace
   smoothed_error = (error * fidelity + 1) / (fidelity + 2)

   GP.add_observation(reencoded, error_to_use, fidelity)
   GP.train(epochs=1000, patience=50)  # Full retrain
   ```

**Consecutive Retrain Failure Handling:**
```python
if not retrain_success:
    consecutive_failures += 1
    if consecutive_failures >= 3:
        raise RuntimeError("GP retraining failed 3x consecutively")
```

**Centralizovaný Iteration Summary (`_log_iteration_summary`):**

Na konci každé iterace se vypisuje konsolidovaný přehled všech metrik:

```
═══════════════════════════════════════════════════════════════
ITERATION 1 SUMMARY
═══════════════════════════════════════════════════════════════
Instruction: Let's think step by step...
─────────────────────────────────────────────────────────────────
PERFORMANCE METRICS:
  Predicted Error:    0.1523 ± 0.0234
  Actual Error:       0.1489
  Best Error So Far:  0.1489
  Improved:           YES
─────────────────────────────────────────────────────────────────
OPTIMIZATION GAP METRICS:
  VAE Latent Cosine:  0.9234
  VAE Latent L2:      0.4521
  GP Space Cosine:    0.9456
  Pred @ z_real:      0.1534
─────────────────────────────────────────────────────────────────
GENERATION QUALITY:
  Cosine Similarity:  0.9512
  LogEI:              -2.3456
  Gap (inversion):    0.0489
  Inversion Iters:    2
  Rejection Attempts: 0
  Low Quality Accept: False
─────────────────────────────────────────────────────────────────
GP Status:
  Training Samples:   226
═══════════════════════════════════════════════════════════════
```

Tento formát centralizuje všechny metriky na jednom místě pro snadné sledování optimalizace.

### 9. LatentSpaceAcquisition (botorch_acq.py)

BoTorch optimalizace v latentním prostoru:

```python
class LatentSpaceAcquisition:
    def optimize(self, best_f, num_restarts=64, raw_samples=1024, seed=None):
        acq_fn = CompositeLogEI(model=gp_model, best_f=best_f)

        candidate, acq_value = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,  # (2, 16) pro 16D VAE latent
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"maxiter": 200, "batch_limit": 5},
        )

        return candidate, acq_value  # (1, 16), scalar
```

**Bounds výpočet:**
```python
def get_latent_bounds(X_train, X_min, X_max, margin=0.2):
    latents_norm = (X_train - X_min) / (X_max - X_min)
    z_min = latents_norm.min(dim=0)[0]  # (16,)
    z_max = latents_norm.max(dim=0)[0]  # (16,)
    z_range = z_max - z_min
    return torch.stack([
        z_min - margin * z_range,  # 20% pod minimum
        z_max + margin * z_range,  # 20% nad maximum
    ])
```

---

## APE Generator (training.py)

Generátor diverzních instrukcí s 5 stylovými kategoriemi:

```python
STYLE_CATEGORIES = {
    "minimalist": {
        "description": "Ultra-short, terse (1-5 words only)",
        "examples": ["Solve:", "Answer:", "Calculate.", "Find:"],
        "temperature": 1.0,
    },
    "direct_command": {
        "description": "Short imperative commands (6-15 words)",
        "examples": ["Find the answer to this problem.", ...],
        "temperature": 0.9,
    },
    "chain_of_thought": {
        "description": "Step-by-step reasoning triggers",
        "examples": ["Let's think step by step.", ...],
        "temperature": 0.8,
    },
    "pedagogical": {
        "description": "Patient teacher/tutor persona",
        "examples": ["You are a patient tutor...", ...],
        "temperature": 0.9,
    },
    "analytical": {
        "description": "Focus on logical structure and analysis",
        "examples": ["Analyze the logical structure...", ...],
        "temperature": 0.8,
    },
}
```

---

## Spouštěcí režimy

### 1. Standardní režim (APE → VAE → Hyperband → InvBO)
```bash
uv run python -m lipo.run \
    --ape-instructions 2000 \
    --iterations 10
```

### 2. Skip HbBoPs (načti pre-evaluované)
```bash
uv run python -m lipo.run \
    --skip-hbbops \
    --hyperband-evals-path lipo/data/ape_instructions.json
```

### 3. Grid loading (top-K + InvBO)
```bash
uv run python -m lipo.run \
    --load-grid datasets/hbbops/full_grid_combined.jsonl \
    --top-k 25 \
    --iterations 10
```

### 4. Debug režim
```bash
uv run python -m lipo.run \
    --debug \
    --iterations 1 \
    --ape-instructions 10
```

### 5. Pouze Hyperband (bez inference)
```bash
uv run python -m lipo.run --hyperband-only
```

---

## Klíčové designové rozhodnutí

### 1. Proč 16D VAE latent (místo 64D)?
- Kompaktnější reprezentace → hustější latentní prostor
- Silnější KL regularizace (β=0.015) zabraňuje posterior collapse
- Cycle consistency (γ=1.0) zajišťuje z ≈ encode(decode(z))
- Cosine loss (rotačně invariantní) místo MSE
- Menší dimenze = rychlejší optimalizace

### 2. Proč Adapter 16D → 10D?
- Efektivní GP modelování
- Rychlejší BoTorch optimalizace
- Trénovatelný - učí se task-relevant features společně s GP
- VAE zůstává zmrazený - zabraňuje mode collapse

### 3. Proč Multi-fidelity Hyperband?
- Min fidelity 10 vzorků = levná evaluace
- Successive halving = geometrický růst fidelity
- Pouze high-fidelity (top 25%) pro GP = nezkreslené pořadí

### 4. Proč LogEI místo EI?
- Standardní EI underflowuje pro malá zlepšení
- Tříbodová implementace pro numerickou stabilitu
- Umožňuje velmi jemnou optimalizaci

### 5. Proč InvBO Inversion Loop?
- Problém: decoder output ≠ Vec2Text rekonstrukce
- Řešení: Najdi z_inv takové, že decoder(z_inv) ≈ GTR(Vec2Text(embedding))
- Konvergence: Re-invert pokud gap > threshold (cosine distance)

### 6. Proč re-encoding po inverzi?
- GP trénován na GTR embeddingech, ne decoder outputs
- Predikce: použij re-encoded embedding pro konzistenci s GP
- Evaluace: LLM evaluuje text, ne embedding

### 7. Proč Heteroscedastic Noise?
- Bernoulli variance: Var = p(1-p)/n
- Nízká fidelity (malé n) → vysoká variance → GP méně důvěřuje
- Vysoká fidelity (velké n) → nízká variance → GP více důvěřuje
- `FixedNoiseGaussianLikelihood` s `learn_additional_noise=False` (CUDA stabilita)

### 8. Proč Candidate Rejection?
- Problém: Vec2Text může generovat text vzdálený od decoder outputu
- Řešení: Ověř cosine similarity ≥ 0.90 mezi decoder a re-encoded
- Max 5 pokusů s různými seedy před fallback akceptací
- Varování při nízké kvalitě: indikuje problémy s Vec2Text

### 9. Proč Laplace Smoothing?
- Penalizuje "lucky guesses" na low-fidelity vzorcích
- Formula: (errors + 1) / (n + 2)
- 0/10 → 1/12 = 0.083 (bylo 0.0 - nerealisticky nízké)
- 0/1319 → 1/1321 = 0.00076 (zůstává blízko 0)

### 10. Proč negovat error_rate pro GP?
- BoTorch's qLogEI MAXIMALIZUJE by default
- Ukládáme y = -error_rate
- GP predikuje -error_rate, EI hledá maximum
- Maximum (-error) = minimum (error) = nejlepší prompt

---

## Loss funkce a trénovací detaily

### VAE Loss
```python
recon_loss = mean(1 - cosine_similarity(x, x_recon))
kl_loss = -0.5 * mean(sum(1 + log_var - mu² - exp(log_var)))
cycle_loss = mean(1 - cosine_similarity(z, encode_mu(decode(z))))
total_loss = recon_loss + beta(epoch) * kl_loss + gamma * cycle_loss

# KL Annealing
beta(epoch) = vae_beta * (epoch / annealing_epochs)  # epoch < annealing
            = vae_beta                                 # jinak (0.015 pro 16D)
```

**Optimizer:** AdamW, lr=0.0006
**Scheduler:** CosineAnnealingLR, T_max=2×epochs, eta_min=1e-4
**Early stopping:** Na recon_loss, patience=500

### GP Training Loss
```python
loss = -MarginalLogLikelihood(GP output, normalized targets)
# ExactGP s Matern 5/2 kernel, trénovaný AdamW
# FixedNoiseGaussianLikelihood s heteroscedastic noise
```

**Optimizer:** AdamW, lr=0.01
**Early stopping:** patience=100
**Cholesky jitter:** 1e-4

### Inversion Loss
```python
loss = 1 - cosine_similarity(decoder(z), GTR(text))
# Minimalizováno Adam, 100 kroků na iteraci
```

**Optimizer:** Adam, lr=0.1
**Convergence threshold:** 0.01

---

## Výstupní formát

Výsledky se ukládají do `lipo/results/result_{timestamp}.json`:

```json
{
  "timestamp": "20260102_123456",
  "vae_quality_metrics": {
    "cosine_mean": 0.98,
    "cosine_std": 0.01,
    "cosine_min": 0.92,
    "cosine_max": 0.999,
    "mse_mean": 5e-05,
    "mse_std": 3e-05,
    "l2_relative_error": 0.18,
    "latent_norm_mean": 4.0,
    "latent_norm_std": 0.5,
    "latent_var_mean": 1.01,
    "latent_var_min": 0.87,
    "latent_var_max": 1.21,
    "active_dims": 16,
    "total_dims": 16,
    "kld_mean": 8.4,
    "kld_std": 2.8,
    "posterior_collapsed": false
  },
  "config": {
    "mode": "standard",
    "vae_beta": 0.015,
    "vae_gamma": 1.0,
    "vae_epochs": 15000,
    "vae_annealing": 500,
    "vae_latent_dim": 16,
    "gp_latent_dim": 10,
    "gp_epochs": 10000,
    "gp_lr": 0.01,
    "gp_patience": 100,
    "turbo_enabled": true,
    "pas_enabled": true,
    ...
  },
  "vae_training": {
    "epochs_trained": 6000,
    "final_recon_loss": 0.02,
    "final_kl_loss": 8.0,
    "final_cosine_similarity": 0.98,
    "early_stopped": true
  },
  "gp_training": {
    "epochs_trained": 1000,
    "final_loss": 0.63,
    "early_stopped": true,
    "num_samples": 267,
    "best_observed_error": 0.083
  },
  "hyperband": {
    "best_instruction": "...",
    "best_error": 0.15,
    "llm_calls": 5000
  },
  "inference": {
    "best_instruction": "...",
    "best_error": 0.12,
    "improvement": 0.03,
    "iterations": 10,
    "llm_calls": 1319,
    "history": [
      {
        "iteration": 1,
        "instruction": "...",
        "predicted_error": 0.14,
        "actual_error": 0.13,
        "improved": true,
        "best_so_far": 0.13,
        "cosine_similarity": 0.95,
        "log_ei": -2.5,
        "gap": 0.05,
        "inversion_iters": 2,
        "rejection_attempts": 0,
        "low_quality_accepted": false,
        "gp_samples": 226,
        "z_opt_z_real_cosine": 0.92,
        "z_opt_z_real_euclidean": 0.45,
        "z_opt_z_real_gp_cosine": 0.95,
        "predicted_error_at_z_real": 0.15
      },
      ...
    ]
  },
  "total_llm_calls": 6319
}
```

---

## Shrnutí defaultních parametrů

| Komponenta | Parametr | Default | Účel |
|------------|----------|---------|------|
| APE | num_instructions | 2000 | Diverzita instrukcí |
| VAE | latent_dim | 16 | Kompaktní latentní prostor |
| VAE | beta | 0.015 | KL regularizace (5x silnější pro 16D) |
| VAE | gamma | 1.0 | Cycle consistency weight |
| VAE | annealing_epochs | 500 | β ramp období |
| VAE | epochs | 15000 | Max trénovací iterace |
| VAE | lr | 0.0006 | Learning rate |
| VAE | patience | 500 | Early stopping |
| Hyperband | bmin | 10 | Min fidelity |
| Hyperband | eta | 2.0 | Successive halving rate |
| Hyperband | min_fidelity_pct | 0.75 | Top 25% pro GP |
| GP | gp_latent_dim | 10 | Adapter output dimenze |
| GP | epochs | 10000 | Max trénovací iterace |
| GP | lr | 0.01 | Learning rate |
| GP | patience | 100 | Early stopping |
| TuRBO | L_init | 0.8 | Počáteční trust region |
| TuRBO | tau_succ | 3 | Úspěchy pro expand |
| TuRBO | tau_fail | 40 | Neúspěchy pro shrink |
| Inference | num_restarts | 64 | L-BFGS-B multi-start |
| Inference | raw_samples | 1024 | Inicializace |
| Inference | max_inversion_iters | 3 | InvBO refinement |
| Inference | gap_threshold | 0.1 | Re-inversion threshold |
| Inference | cosine_sim_threshold | 0.90 | Candidate acceptance threshold |
| Inference | max_rejection_attempts | 5 | Max attempts before fallback |
| GP Retrain | epochs | 1000 | Inference GP retrain |
| GP Retrain | patience | 50 | Retrain early stopping |
| Vec2Text | beam_width | 8 | Beam search width |
| Vec2Text | max_length | 128 | Max output tokenů |
| Vec2Text | model_type | 512_tokens | Podpora delších sekvencí |
