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
│ APE Generace (1000 instrukcí)            │
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
│ - Latent: 64D (hladký přes KL)           │
│ - KL annealing: β roste od 0 do β_max    │
│ - Output: L2-normalizovaná rekonstrukce  │
│ - Loss: cosine_recon + β·KL              │
└──────────────────────────────────────────┘

FÁZE 3: HYPERBAND EVALUACE
┌──────────────────────────────────────────┐
│ LIPO Hyperband                  │
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
│ - 64D VAE latent → 10D (přes adapter)    │
│ - Matern 5/2 kernel s ARD (10D)          │
│ - Training: adapter + kernel learnable   │
└──────────────────────────────────────────┘

FÁZE 4: InvBO INFERENCE
┌──────────────────────────────────────────┐
│ BoTorch qLogEI Optimalizace              │
│ - Optimalizace 64D VAE latent            │
│ - Multi-start L-BFGS-B (64 restartů)     │
│ - Raw samples inicializace (1024)        │
│ - LogEI: numericky stabilní EI           │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ VAE Decoder + Vec2Text Inverze           │
│ - z_opt (64D) → decoder → embedding (768D)
│ - embedding → Vec2Text → text            │
│ - 512_tokens model (až 512 tokenů)       │
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
│ - Přidání do training dat                │
│ - Inkrementální retrain (500 epoch)      │
└──────────────────────────────────────────┘
```

---

## Struktura souborů

```
lipo/
├── config.py           # Unified configuration (SSOT)
├── encoder.py          # GTRInstructionEncoder, InstructionVAE, VAEWithAdapter
├── gp.py               # InstructionDeepKernelGP, GPWithEI, LogEI funkce
├── hyperband.py        # LIPO (Hyperband + BO)
├── training.py         # APEGenerator, LIPOTrainer
├── inference.py        # Vec2TextInverter, LIPOInference
├── botorch_acq.py      # CompositeLogEI, LatentSpaceAcquisition
├── evaluate.py         # GSM8KEvaluator
├── instruction.py      # InstructionOnlyPrompt dataclass
├── run.py              # CLI entry point
└── data/
    └── ape_instructions.json  # Cache generovaných instrukcí
```

---

## Konfigurace (config.py)

Veškeré parametry jsou centralizované v `Config` dataclass:

### APE Generace
| Parametr | Default | Popis |
|----------|---------|-------|
| `ape_num_instructions` | 1000 | Počet instrukcí k vygenerování |
| `ape_model` | `Qwen/Qwen2.5-7B-Instruct` | Model pro generování |
| `ape_backend` | `vllm` | Backend (vllm/openai/deepinfra) |
| `ape_cache_path` | `lipo/data/ape_instructions.json` | Cesta k cache |
| `ape_batch_size` | 10 | Velikost batche |
| `ape_max_tokens` | 100 | Max tokenů na instrukci |
| `ape_max_length` | 500 | Max délka znakově |

### VAE Trénink
| Parametr | Default | Popis |
|----------|---------|-------|
| `vae_beta` | 0.003 | KL regularizace (škálováno pro 64D) |
| `vae_epochs` | 10000 | Max počet epoch |
| `vae_annealing_epochs` | 500 | Počet epoch pro KL annealing |
| `vae_patience` | 500 | Early stopping patience |
| `vae_lr` | 0.0003 | Learning rate |
| `vae_batch_size` | 64 | Batch size |
| `vae_hidden_dim` | 256 | Skryté dimenze |
| `vae_grad_clip` | 1.0 | Gradient clipping |
| `vae_eta_min` | 1e-4 | Min LR pro cosine scheduler |

### Latentní dimenze
| Parametr | Default | Popis |
|----------|---------|-------|
| `embedding_dim` | 768 | GTR embedding dimenze |
| `latent_dim` | 64 | VAE latentní dimenze |
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

### Vec2Text
| Parametr | Default | Popis |
|----------|---------|-------|
| `vec2text_beam` | 8 | Beam width |
| `vec2text_model` | `512_tokens` | Model typ |

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
| `gp_retrain_epochs` | 500 | Počet epoch |
| `gp_retrain_lr` | 0.001 | Learning rate |
| `gp_retrain_patience` | 10 | Patience |

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
Encoder: 768D → 64 → LayerNorm → 32 → LayerNorm → 2×64 (mu + log_var)
Decoder: 64 → 32 → LayerNorm → 64 → LayerNorm → 256 → LayerNorm → 768D (L2 normalized)
```

**Loss funkce:**
```python
recon_loss = mean(1 - cosine_similarity(x, x_recon))
kl_loss = -0.5 * mean(sum(1 + log_var - mu² - exp(log_var)))
total_loss = recon_loss + beta(epoch) * kl_loss
```

**KL Annealing:**
```python
if epoch < annealing_epochs:
    beta = vae_beta * (epoch / annealing_epochs)  # Lineární růst od 0
else:
    beta = vae_beta  # Konstanta
```

**Účel:** Zajišťuje hladký latentní prostor (žádné "díry"), distribuci kolem N(0,1), a lepší generalizaci.

### 3. VAEWithAdapter (encoder.py)

Kombinuje zmrazenou VAE s trénovatelným adaptérem:

```
x (768D) → VAE.encode_mu (frozen) → z_vae (64D) → adapter → z_gp (10D)
```

**Architektura adaptéru:**
```python
adapter = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.LayerNorm(32),
    nn.Linear(32, 10),
)
```

**Metody:**
- `encode_vae(x)` - 768D → 64D (pouze VAE encoder)
- `forward(x)` - 768D → 10D (celá pipeline)
- `adapt(z)` - 64D → 10D (pouze adapter)
- `decode(z)` - 64D → 768D (pouze VAE decoder)

### 4. InstructionDeepKernelGP (gp.py)

GP model s deep kernel pro instrukční optimalizaci:

```python
class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
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
def forward(self, x):  # x: (batch, 64)
    latent = self.adapter(x)  # (batch, 10)
    return MultivariateNormal(
        self.mean_module(latent),
        self.covar_module(latent),
    )
```

### 5. GPWithEI (gp.py)

Wrapper pro GP s Expected Improvement:

**Normalizace:**
```python
# Input (64D VAE latenty)
X_norm = (X - X_min) / (X_max - X_min)  # Unit cube [0,1]^64

# Output (error rates)
y_norm = (y - y_mean) / y_std  # Z-score standardizace
```

**Training:**
```python
likelihood = GaussianLikelihood(noise_constraint=Interval(0.001, 0.1))
optimizer = AdamW(gp_model.parameters(), lr=0.01)
loss = -ExactMarginalLogLikelihood(likelihood, gp_model)
```

**LogEI implementace** (numericky stabilní):
```python
# Pro z = (y_best - μ - ξ) / σ
# LogEI = log_h(z) + log(σ)
# kde h(z) = φ(z) + z·Φ(z)
```

Tři větve pro numerickou stabilitu:
1. z > -1: Přímý výpočet
2. -1/√ε < z ≤ -1: erfcx-based
3. z ≤ -1/√ε: Asymptotická aproximace

### 6. LIPO (hyperband.py)

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
    "max_length": 128,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
}
```

### 8. LIPOInference (inference.py)

Kompletní inference pipeline:

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
   embedding = vae_with_adapter.decode(z_unnorm)  # 64D → 768D
   text = vec2text.invert(embedding)  # 768D → text
   ```

3. **Inversion Loop**
   ```python
   for inv_iter in range(max_inversion_iters):
       # Najdi z_inv: ||decoder(z_inv) - GTR(text)|| minimální
       z = z_init.clone().requires_grad_(True)
       optimizer = Adam([z], lr=0.1)

       for step in range(100):
           z_vae = z * x_range + X_min
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

   GP.add_observation_and_retrain(
       reencoded, error_to_use,
       epochs=500, patience=10
   )
   ```

### 9. LatentSpaceAcquisition (botorch_acq.py)

BoTorch optimalizace v latentním prostoru:

```python
class LatentSpaceAcquisition:
    def optimize(self, best_f, num_restarts=64, raw_samples=1024):
        acq_fn = CompositeLogEI(model=gp_model, best_f=best_f)

        candidate, acq_value = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,  # (2, 64) pro 64D VAE latent
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"maxiter": 200, "batch_limit": 5},
        )

        return candidate, acq_value  # (1, 64), scalar
```

**Bounds výpočet:**
```python
def get_latent_bounds(X_train, X_min, X_max, margin=0.2):
    latents_norm = (X_train - X_min) / (X_max - X_min)
    z_min = latents_norm.min(dim=0)[0]
    z_max = latents_norm.max(dim=0)[0]
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
    --ape-instructions 1000 \
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

### 1. Proč 64D VAE latent?
- Poskytuje hladký optimalizační povrch vs 768D embeddingy
- KL annealing zabraňuje posterior collapse
- Cosine loss (rotačně invariantní) místo MSE

### 2. Proč Adapter 64D → 10D?
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

---

## Loss funkce a trénovací detaily

### VAE Loss
```python
recon_loss = mean(1 - cosine_similarity(x, x_recon))
kl_loss = -0.5 * mean(sum(1 + log_var - mu² - exp(log_var)))
total_loss = recon_loss + beta(epoch) * kl_loss

# KL Annealing
beta(epoch) = vae_beta * (epoch / annealing_epochs)  # epoch < annealing
            = vae_beta                                 # jinak
```

**Optimizer:** AdamW, lr=0.0003
**Scheduler:** CosineAnnealingLR, T_max=2×epochs, eta_min=1e-4
**Early stopping:** Na recon_loss, patience=500

### GP Training Loss
```python
loss = -MarginalLogLikelihood(GP output, normalized targets)
# ExactGP s Matern 5/2 kernel, trénovaný AdamW
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
    "cosine_mean": 0.95,
    "cosine_std": 0.02,
    "kld_mean": 12.5,
    "active_dims": 60,
    "total_dims": 64
  },
  "config": {
    "mode": "standard",
    "vae_beta": 0.003,
    "vae_epochs": 10000,
    "gp_epochs": 10000,
    ...
  },
  "vae_training": {
    "epochs_trained": 5000,
    "final_recon_loss": 0.05,
    "final_kl_loss": 10.2,
    "early_stopped": true
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
    "history": [...]
  },
  "total_llm_calls": 6319
}
```

---

## Shrnutí defaultních parametrů

| Komponenta | Parametr | Default | Účel |
|------------|----------|---------|------|
| APE | num_instructions | 1000 | Diverzita instrukcí |
| VAE | beta | 0.003 | KL regularizace (škálováno pro 64D) |
| VAE | annealing_epochs | 500 | β ramp období |
| VAE | epochs | 10000 | Max trénovací iterace |
| VAE | lr | 0.0003 | Learning rate |
| VAE | patience | 500 | Early stopping |
| Hyperband | bmin | 10 | Min fidelity |
| Hyperband | eta | 2.0 | Successive halving rate |
| Hyperband | min_fidelity_pct | 0.75 | Top 25% pro GP |
| GP | epochs | 10000 | Max trénovací iterace |
| GP | lr | 0.01 | Learning rate |
| GP | patience | 100 | Early stopping |
| Inference | num_restarts | 64 | L-BFGS-B multi-start |
| Inference | raw_samples | 1024 | Inicializace |
| Inference | max_inversion_iters | 3 | InvBO refinement |
| Inference | gap_threshold | 0.1 | Re-inversion threshold |
| Vec2Text | beam_width | 8 | Beam search width |
| Vec2Text | model_type | 512_tokens | Podpora delších sekvencí |
