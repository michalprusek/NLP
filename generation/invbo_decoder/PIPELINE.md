# InvBO Decoder Inversion Pipeline

VAE-based Bayesian optimization for instruction generation. Uses BoTorch qLogEI for gradient-based optimization in 10D latent space with Vec2Text inversion.

## Quick Start

```bash
# Standard run
uv run python -m generation.invbo_decoder.run --iterations 10

# With custom hyperparameters
uv run python -m generation.invbo_decoder.run \
    --iterations 50 --vae-beta 0.02 --vae-annealing 500

# Skip-eval mode (GP prediction instead of LLM evaluation)
uv run python -m generation.invbo_decoder.run --iterations 10 --skip-eval
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: VAE Training (KL Annealing)                                   │
│  ─────────────────────────────────────                                   │
│                                                                          │
│  Diverse Instructions (1000) + Grid Instructions (100)                  │
│                              │                                           │
│                              ▼                                           │
│                     GTR Encoder (gtr-t5-base)                           │
│                              │                                           │
│                              ▼                                           │
│                    768D L2-normalized embedding                         │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  InstructionVAE (encoder.py)                                   │     │
│  │  ───────────────────────────                                   │     │
│  │                                                                │     │
│  │  Encoder: 768D → 64 → 32 → (μ, log_var)                       │     │
│  │  Reparameterization: z = μ + σ × ε                            │     │
│  │  Decoder: 10D → 32 → 64 → 256 → 768D (L2 normalized)          │     │
│  │                                                                │     │
│  │  Loss = Recon(cosine) + β × KL(q(z|x) || N(0,1))              │     │
│  │  KL Annealing: β = 0 → target over 500 epochs                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Phase 2: GP Training (frozen VAE + trainable adapter)                  │
│  ──────────────────────────────────────────────────────                  │
│                                                                          │
│  Grid Instructions (top-k=25) → GTR → 768D                              │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  VAEWithAdapter (training.py)                                  │     │
│  │  ────────────────────────────                                  │     │
│  │                                                                │     │
│  │  1. VAE.encode_mu(x) → 10D latent  [FROZEN]                   │     │
│  │  2. Adapter MLP: 10D → 20 → 10D    [TRAINABLE]                │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  InstructionDeepKernelGP (gp.py)                               │     │
│  │  ───────────────────────────────                               │     │
│  │                                                                │     │
│  │  - Matérn 5/2 kernel with ARD (10 lengthscales)               │     │
│  │  - GaussianLikelihood(noise_constraint=GreaterThan(1e-4))     │     │
│  │  - Standardize outcome transform for BoTorch                  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Decoder = VAE.decode() (wrapped in VAEDecoderWrapper)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Main Loop: for iteration in 1..N:                                      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: BoTorch qLogEI Optimization                            │   │
│  │  ───────────────────────────────────                             │   │
│  │                                                                  │   │
│  │  CompositeLogEI: z (10D) → decoder → embedding (768D) → GP     │   │
│  │  Multi-start L-BFGS-B (64 restarts, 512 raw samples)           │   │
│  │  → z* = argmax LogEI(z)                                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Decode and Invert                                      │   │
│  │  ─────────────────────────                                       │   │
│  │                                                                  │   │
│  │  embedding* = decoder(z*)                                       │   │
│  │  text* = Vec2Text(embedding*)                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: InvBO Inversion Loop (closes misalignment gap)        │   │
│  │  ────────────────────────────────────────────────────            │   │
│  │                                                                  │   │
│  │  for inv_iter in 1..max_inversion_iters:                        │   │
│  │    z_inv = argmin ||decoder(z) - GTR(text*)||²                  │   │
│  │    gap = 1 - cosine_sim(decoder(z*), decoder(z_inv))           │   │
│  │    if gap <= 0.1: ACCEPT                                        │   │
│  │    else: z* = z_inv, re-decode, re-invert                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: Evaluation + GP Update                                 │   │
│  │  ───────────────────────────────                                 │   │
│  │                                                                  │   │
│  │  --skip-eval: actual_error = GP.predict(embedding*)             │   │
│  │  default: actual_error = LLM evaluation on GSM8K                │   │
│  │                                                                  │   │
│  │  GP update: add observation, incremental retrain                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### InstructionVAE (`encoder.py`)

Variational Autoencoder for smooth latent space.

```python
# Encoder: 768D → 64 → 32 → (μ, log_var)
# Decoder: 10D → 32 → 64 → 256 → 768D (L2 normalized)
# Loss: cosine_recon + β × KL(q(z|x) || N(0,1))
```

Key methods:
- `encode_mu(x) → z`: Deterministic encoding (for GP)
- `decode(z) → x`: L2-normalized embedding (for Vec2Text)

### GPWithEI (`gp.py`)

Deep kernel GP with Expected Improvement.

```python
GPWithEI(device="cuda", latent_dim=10)
```

Key features:
- Unit-cube normalization: `X_norm = (X - X_min) / (X_max - X_min)`
- Noise constraint: `GreaterThan(1e-4)` prevents overconfidence
- Incremental retraining preserves input normalization

### Vec2TextInverter (`inference.py`)

Embedding-to-text inversion.

```python
Vec2TextInverter(
    num_steps=50,
    beam_width=8,
    model_type="32_tokens",  # or "512_tokens"
)
```

---

## CLI Parameters

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 1 | Optimization iterations |
| `--skip-eval` | False | Use GP prediction instead of LLM |
| `--vae-beta` | 0.02 | KL regularization weight |
| `--vae-epochs` | 10000 | VAE training epochs |
| `--vae-annealing` | 500 | KL annealing epochs |
| `--n-restarts` | 64 | BoTorch L-BFGS-B restarts |
| `--raw-samples` | 512 | BoTorch initialization samples |

### Inversion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-inversion` | True | Enable InvBO inversion loop |
| `--max-inversion-iters` | 10 | Max iterations |
| `--gap-threshold` | 0.1 | Cosine distance threshold |

---

## Output Format

Results saved to `generation/invbo_decoder/results/result_TIMESTAMP.json`:

```json
{
  "timestamp": "20251229_120000",
  "seed": 42,
  "grid_best": { "instruction_id": 42, "error_rate": 0.1077 },
  "optimized": { "instruction": "...", "error_rate": 0.0950 },
  "iteration_history": [...],
  "improvement": 0.0127,
  "vae_quality_metrics": { "cosine_mean": 0.98, ... }
}
```

---

## Known Issues and Solutions

### 1. Misalignment Problem
Decoder may produce embeddings that don't reconstruct properly via Vec2Text.
**Solution:** InvBO inversion loop closes the gap by optimizing `z_inv = argmin ||decoder(z) - GTR(text)||²`

### 2. GP Overconfidence
GP can be overconfident, leading to LogEI = -5000.
**Solution:** Noise constraint `GreaterThan(1e-4)` per BoTorch recommendation.

### 3. Posterior Collapse (VAE)
VAE may ignore latents.
**Solution:** KL annealing (β = 0 → target), low beta (0.02), early stopping on reconstruction loss.

---

## References

- **InvBO**: Deshwal et al., 2024 - "Inversion-Based BO with Structured Inputs" (NeurIPS 2024)
- **LogEI**: Ament et al., 2023 - "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)
- **Vec2Text**: Morris et al., 2023 - "Text Embeddings Reveal (Almost) As Much As Text"
- **BoTorch**: Balandat et al., 2020 - "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization"
