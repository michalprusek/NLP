# InvBO Decoder Inversion Pipeline

Implementace dekóderu z GP latentního prostoru (10D) do Vec2Text embedding prostoru (768D) s cyklickou ztrátou. Řeší "misalignment problem" z článku InvBO (NeurIPS 2024).

## Architektura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: GP + Encoder Training (100 instrukcí s error rates)               │
│  ═══════════════════════════════════════════════════════════                 │
│                                                                              │
│  Instruction Text ──► GTR Encoder ──► 768D Embedding                        │
│        │                                    │                                │
│        │                                    ▼                                │
│        │                        ┌──────────────────────┐                     │
│        │                        │  Unit-Cube Normalize │                     │
│        │                        │  (X - X_min) / range │                     │
│        │                        └──────────────────────┘                     │
│        │                                    │                                │
│        │                                    ▼                                │
│        │                        ┌──────────────────────┐                     │
│        │                        │ InstructionFeature   │                     │
│        │                        │ Extractor (Deep      │                     │
│        │                        │ Kernel Encoder)      │                     │
│        │                        │                      │                     │
│        │                        │ 768 → 64 → 32 → 10   │                     │
│        │                        │ (ReLU + BatchNorm)   │                     │
│        │                        └──────────────────────┘                     │
│        │                                    │                                │
│        │                                    ▼                                │
│        │                              10D Latent                             │
│        │                                    │                                │
│        │                                    ▼                                │
│        │                        ┌──────────────────────┐                     │
│        │                        │   DeepKernelGP       │                     │
│        │                        │   Matérn 5/2 + ARD   │                     │
│        │                        └──────────────────────┘                     │
│        │                                    │                                │
│        ▼                                    ▼                                │
│  Error Rate ◄────────────────────── GP Prediction                           │
│  (from grid)                        (μ, σ)                                   │
│                                                                              │
│  Loss: MLL (Marginal Log Likelihood)                                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 2: Decoder Training (1000 diverse instrukcí)                         │
│  ══════════════════════════════════════════════════                          │
│                                                                              │
│  Diverse Instructions (1000) ──► GTR ──► 768D ──► Frozen Encoder ──► 10D    │
│                                                                              │
│  Training Loop:                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  10D Latent (z) ──► LatentDecoder ──► 768D Decoded Embedding                │
│        │                                    │                                │
│        │                ┌───────────────────┤                                │
│        │                │                   │                                │
│        │                ▼                   ▼                                │
│        │         ┌────────────┐     ┌────────────────┐                       │
│        │         │  Normalize │     │ Cosine Loss    │                       │
│        │         │  + Frozen  │     │ vs Original    │                       │
│        │         │  Encoder   │     │ Embedding      │                       │
│        │         └────────────┘     └────────────────┘                       │
│        │                │                   │                                │
│        │                ▼                   │                                │
│        │         10D z_recon                │                                │
│        │                │                   │                                │
│        ▼                ▼                   ▼                                │
│   ┌─────────────────────────────────────────────────────┐                    │
│   │                   TOTAL LOSS                        │                    │
│   │  L = λ_cycle * MSE(z, z_recon) + λ_cosine * (1-cos) │                    │
│   └─────────────────────────────────────────────────────┘                    │
│                                                                              │
│  LatentDecoder Architecture:                                                 │
│  ┌──────────────────────────────────────────┐                                │
│  │ 10 → 32 → 64 → 256 → 768                 │                                │
│  │ (ReLU + BatchNorm + L2 Normalize)        │                                │
│  └──────────────────────────────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. EI Optimization in 10D Latent Space                                     │
│  ───────────────────────────────────────                                     │
│                                                                              │
│     Sample z from training distribution                                      │
│           │                                                                  │
│           ▼                                                                  │
│     ┌─────────────┐                                                          │
│     │  Decoder    │ ──► 768D Embedding                                       │
│     └─────────────┘          │                                               │
│                              ▼                                               │
│                    ┌─────────────────┐                                       │
│                    │ GP Prediction   │ ──► μ(z), σ(z)                        │
│                    └─────────────────┘          │                            │
│                                                 ▼                            │
│                              ┌──────────────────────────────┐                │
│                              │ Expected Improvement (EI)    │                │
│                              │                              │                │
│                              │ EI(z) = (y_best - μ) * Φ(z)  │                │
│                              │       + σ * φ(z)             │                │
│                              │                              │                │
│                              │ z = (y_best - μ - ξ) / σ     │                │
│                              └──────────────────────────────┘                │
│                                                 │                            │
│                                                 ▼                            │
│                                          z* = argmax EI(z)                   │
│                                                                              │
│  2. Decoding to Text via Vec2Text                                           │
│  ─────────────────────────────────                                           │
│                                                                              │
│     z* (optimal 10D) ──► Decoder ──► 768D Embedding*                         │
│                                            │                                 │
│                                            ▼                                 │
│                               ┌────────────────────────┐                     │
│                               │      Vec2Text          │                     │
│                               │  (Iterative Corrector) │                     │
│                               │                        │                     │
│                               │  InversionModel +      │                     │
│                               │  CorrectorEncoder      │                     │
│                               │  (50 steps, beam=4)    │                     │
│                               └────────────────────────┘                     │
│                                            │                                 │
│                                            ▼                                 │
│                                   Novel Instruction Text                     │
│                                                                              │
│  3. Validation                                                               │
│  ─────────────                                                               │
│                                                                              │
│     Novel Text ──► GTR ──► 768D Re-encoded                                   │
│                                    │                                         │
│                                    ▼                                         │
│                        Cosine Similarity vs 768D Embedding*                  │
│                        (měří kvalitu Vec2Text inverze)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Datové Soubory

| Soubor | Popis | Použití |
|--------|-------|---------|
| `instructions_100.txt` | 100 instrukcí v 10 kategoriích | GP training (Phase 1) |
| `grid_100_qend.jsonl` | Error rates pro 100 instrukcí | GP targets |
| `diverse_instructions_1000.json` | 881 diverse instrukcí | Decoder training (Phase 2) |

## Komponenty

### 1. GTRInstructionEncoder
- Model: `sentence-transformers/gtr-t5-base`
- Output: 768D L2-normalized embedding
- Kompatibilní s Vec2Text

### 2. InstructionFeatureExtractor (Deep Kernel Encoder)
```
768D → Linear(768, 64) → ReLU → BatchNorm
    → Linear(64, 32)  → ReLU → BatchNorm
    → Linear(32, 10)  → 10D Latent
```
- Trénuje se společně s GP (Phase 1)
- Zamrzne se pro decoder training (Phase 2)

### 3. InstructionDeepKernelGP
- Kernel: Matérn 5/2 s ARD (10 lengthscales)
- Prior na lengthscale: Gamma(3.0, 6.0)
- Prior na outputscale: Gamma(2.0, 0.15)
- Predikuje error rate pro instrukce

### 4. LatentDecoder
```
10D → Linear(10, 32)  → ReLU → BatchNorm
   → Linear(32, 64)  → ReLU → BatchNorm
   → Linear(64, 256) → ReLU → BatchNorm
   → Linear(256, 768) → L2 Normalize → 768D
```
- Zrcadlová architektura k encoderu
- L2 normalizace je kritická pro Vec2Text kompatibilitu

### 5. Vec2TextInverter
- Models: `ielabgroup/vec2text_gtr-base-st_inversion` + `_corrector`
- Iterativní oprava (50 kroků)
- Beam search (width=4)

## Loss Funkce

### Phase 1: GP Training
```
L = -MLL(GP(encoder(normalize(embedding))), error_rate)
```

### Phase 2: Decoder Training
```
L = λ_cycle * ||z - encoder(normalize(decoder(z)))||²
  + λ_cosine * (1 - cosine_sim(decoder(z), target_embedding))
```

**Hyperparametry:**
- `λ_cycle = 1.0` - váha cyklické ztráty
- `λ_cosine = 5.0` - váha kosínové ztráty
- `cycle_tolerance = 0.0` - striktní cyklická ztráta

## Normalizace

### Unit-Cube Normalization (pro GP)
```python
X_norm = (X - X_min) / (X_max - X_min)
```
- Aplikuje se na 768D embeddingy před encodérem
- X_min, X_max se počítají z training dat (100 instrukcí)

### Z-Score Standardization (pro GP targets)
```python
y_norm = (y - y_mean) / y_std
```

### L2 Normalization (pro Vec2Text)
```python
embedding = embedding / ||embedding||₂
```

## Inference: Expected Improvement

```python
def expected_improvement(z, y_best, xi=0.01):
    embedding = decoder(z)
    mu, sigma = gp.predict(normalize(embedding))

    if sigma <= 0:
        return max(y_best - mu, 0)

    z_score = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * Φ(z_score) + sigma * φ(z_score)
    return max(ei, 0)
```

## Známé Problémy

### 1. Misalignment Problem (InvBO paper)
Decoder může produkovat embeddingy, které po Vec2Text inverzi dají text odlišný od zamýšleného. Řešení:
- Cyklická ztráta ||z - encoder(decoder(z))||²
- Ale Vec2Text má vlastní reconstruction gap

### 2. Out-of-Distribution Decoding
Decoder může produkovat embeddingy mimo distribuci reálných textů:
- Vec2Text pak generuje nesmyslný text
- Projevuje se nízkou cosine similarity (< 0.2)

### 3. Vec2Text Limitations
- Funguje dobře pro krátké texty (~30 tokenů)
- Selhává pro delší texty (exempláry ~200+ tokenů)
- Proto používáme instruction-only přístup

## Spuštění

```bash
# Plný trénink + inference
uv run python -m generation.invbo_decoder.run

# S vlastními parametry
uv run python -m generation.invbo_decoder.run \
    --gp-epochs 500 \
    --decoder-epochs 300 \
    --method random \
    --n-candidates 500

# Validace inversion gap
uv run python -m generation.invbo_decoder.run --validate-gap
```

## Výsledky

### Metriky Decoderu
- **Cosine similarity**: ~0.79 (průměr na grid_100)
- **Cycle distance**: ~0.03 (průměr)

### Problém s Generací
Generované instrukce jsou často nesmyslné:
```
"where Hindu isolated Grama 102"
"jade patch known as Pustasy INPUT has the following fragrance..."
```

**Důvod**: Decoder produkuje embeddingy mimo distribuci, Vec2Text nedokáže invertovat.

### Srovnání s Grid
Best from grid: `error_rate = 0.1077`
Best generated: `predicted_error = 0.135` (ale text je nesmyslný)

## Možná Vylepšení

1. **VAE místo deterministického decoderu** - regularizace latentního prostoru
2. **Adversarial training** - diskriminátor pro real vs fake embeddingy
3. **Constrained optimization** - optimalizovat pouze v konvexním obalu training latentů
4. **InvBO inversion** - z_inv = argmin d(x, decoder(z)) jako v původním článku
