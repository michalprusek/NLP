# InvBO Decoder Inversion Pipeline

Kompletní implementace dekóderu z GP latentního prostoru (10D) do Vec2Text embedding prostoru (768D) s cyklickou ztrátou. Řeší "misalignment problem" z článku InvBO (NeurIPS 2024).

## Klíčové Features

- **VAE Mode (default)**: Variational Autoencoder pro hladký latentní prostor (beta=0.02)
- **BoTorch qLogEI**: Gradient-based optimalizace s numericky stabilním Log Expected Improvement
- **Trust Region (disabled by default)**: TuRBO-style omezení exploration do známých oblastí
- **Inversion Loop**: InvBO-style iterativní inverze pro uzavření misalignment gap
- **Standardize Transform**: BoTorch outcome transform pro správnou denormalizaci
- **Incremental GP Retraining**: Warm-start s preserved normalization

---

## Quick Start

```bash
# Doporučené: Standardní běh (VAE beta=0.02, Trust Region OFF, BoTorch qLogEI)
uv run python -m generation.invbo_decoder.run --iterations 10

# S explicitními hyperparametry (defaults)
uv run python -m generation.invbo_decoder.run \
    --iterations 50 --vae-beta 0.02 --vae-annealing 500

# S trust region (pokud experimentujete)
uv run python -m generation.invbo_decoder.run --trust-region --iterations 10

# Jednoduchý běh (1 iterace) s vizualizací
uv run python -m generation.invbo_decoder.run --visualize

# Skip-eval mode (GP prediction místo LLM evaluace)
uv run python -m generation.invbo_decoder.run --iterations 10 --skip-eval
```

---

## DŮLEŽITÉ

- **Trust Region je vypnutý defaultně** - použijte `--trust-region` pro zapnutí
- **Vždy evaluovat na plném validation setu (1319 samples)** - nikdy nesnižovat `--eval-samples`
- **VAE beta=0.02** zajišťuje hladký latentní prostor pro stabilní optimalizaci
- **GP noise constraint 1e-4** zabraňuje overconfidence (MAE validation ~0.02)
- **Seed 42** pro reprodukovatelnost (nastavuje se pro Python, NumPy, PyTorch, cuDNN)

---

## Architektura - High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TRAINING PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  VAE MODE (default, doporučeno):                                                        │
│  ═══════════════════════════════                                                         │
│                                                                                          │
│  Phase 1: VAE Training (KL Annealing)                                                   │
│  ─────────────────────────────────────                                                   │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                 │    │
│  │  Diverse Instructions (1000) + Grid Instructions (100) = 1100 samples          │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │                     GTR Encoder (gtr-t5-base)                                   │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │                    768D L2-normalized embedding                                 │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │  ┌────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  VAE Encoder                                                           │    │    │
│  │  │  ───────────                                                           │    │    │
│  │  │  768D → Linear(768, 64) → ReLU → LayerNorm(64)                        │    │    │
│  │  │      → Linear(64, 32) → ReLU → LayerNorm(32)                          │    │    │
│  │  │      → Linear(32, 2 × latent_dim)                                     │    │    │
│  │  │                    │                                                   │    │    │
│  │  │                    ▼                                                   │    │    │
│  │  │           Split → (μ, log_var)                                        │    │    │
│  │  │                    │                                                   │    │    │
│  │  │                    ▼                                                   │    │    │
│  │  │    Reparameterization: z = μ + σ × ε,  ε ~ N(0,1)                     │    │    │
│  │  │                    │                                                   │    │    │
│  │  │                    ▼                                                   │    │    │
│  │  │                10D latent z                                            │    │    │
│  │  └────────────────────────────────────────────────────────────────────────┘    │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │  ┌────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  VAE Decoder                                                           │    │    │
│  │  │  ───────────                                                           │    │    │
│  │  │  10D → Linear(10, 32) → ReLU → LayerNorm(32)                          │    │    │
│  │  │      → Linear(32, 64) → ReLU → LayerNorm(64)                          │    │    │
│  │  │      → Linear(64, 256) → ReLU → LayerNorm(256)                        │    │    │
│  │  │      → Linear(256, 768)                                                │    │    │
│  │  │                    │                                                   │    │    │
│  │  │                    ▼                                                   │    │    │
│  │  │              L2 Normalize                                              │    │    │
│  │  │                    │                                                   │    │    │
│  │  │                    ▼                                                   │    │    │
│  │  │     768D reconstructed embedding (Vec2Text compatible)                 │    │    │
│  │  └────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                                 │    │
│  │  Loss = Recon(cosine) + β × KL(q(z|x) || N(0,1))                               │    │
│  │                                                                                 │    │
│  │  KL Annealing: β = 0 → target over vae_annealing_epochs (500)                  │    │
│  │                β_current = vae_beta × min(1, epoch / vae_annealing_epochs)     │    │
│  │                                                                                 │    │
│  │  Early stopping: Tracks RECONSTRUCTION loss (not total) to avoid               │    │
│  │                  premature stop during KL annealing phase                       │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  Phase 2: GP Training (with frozen VAE + trainable adapter)                             │
│  ──────────────────────────────────────────────────────────                              │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                 │    │
│  │  Grid Instructions (top-k=25, lowest error rate)                               │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │                     GTR Encoder → 768D                                          │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │                  Unit-Cube Normalization                                        │    │
│  │                  X_norm = (X - X_min) / (X_max - X_min)                        │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │  ┌────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  VAEWithAdapter (training.py:27)                                       │    │    │
│  │  │  ───────────────────────────────                                       │    │    │
│  │  │                                                                        │    │    │
│  │  │  1. VAE.encode_mu(x) → 10D latent z  [FROZEN weights]                 │    │    │
│  │  │                    │                                                   │    │    │
│  │  │                    ▼                                                   │    │    │
│  │  │  2. Adapter MLP:                                                       │    │    │
│  │  │     10D → Linear(10, 20) → ReLU → LayerNorm(20)                       │    │    │
│  │  │         → Linear(20, 10) → 10D adapted latent  [TRAINABLE]            │    │    │
│  │  │                                                                        │    │    │
│  │  │  Účel: GP může učit lepší reprezentaci při zachování                  │    │    │
│  │  │        smooth latent space z VAE                                       │    │    │
│  │  └────────────────────────────────────────────────────────────────────────┘    │    │
│  │                              │                                                  │    │
│  │                              ▼                                                  │    │
│  │  ┌────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │  InstructionDeepKernelGP (gp.py:127)                                   │    │    │
│  │  │  ───────────────────────────────────                                   │    │    │
│  │  │                                                                        │    │    │
│  │  │  - Inherits from ExactGP + GPyTorchModel (BoTorch compatible)         │    │    │
│  │  │  - ZeroMean                                                            │    │    │
│  │  │  - ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=10))                 │    │    │
│  │  │     - Lengthscale prior: Gamma(3.0, 6.0)                              │    │    │
│  │  │     - Outputscale prior: Gamma(2.0, 0.15)                             │    │    │
│  │  │  - GaussianLikelihood(noise_constraint=GreaterThan(1e-4))             │    │    │
│  │  │                                                                        │    │    │
│  │  │  - Standardize(m=1) outcome transform pro BoTorch compatibility       │    │    │
│  │  │    (automatická denormalizace v posterior() a acquisition)             │    │    │
│  │  └────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                                 │    │
│  │  Training:                                                                      │    │
│  │  - Loss: -MarginalLogLikelihood (ExactMLL)                                     │    │
│  │  - Optimizer: AdamW (lr=0.01)                                                  │    │
│  │  - Early stopping: patience=50                                                 │    │
│  │  - Cholesky jitter: 1e-4                                                       │    │
│  │                                                                                 │    │
│  │  Decoder = VAE.decode() (wrapped in VAEDecoderWrapper, no separate training)  │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  STANDARD MODE (--no-vae, deprecated):                                                  │
│  ══════════════════════════════════════                                                  │
│                                                                                          │
│  Phase 1: GP + Encoder Training (jointly)                                               │
│  ────────────────────────────────────────                                                │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                 │    │
│  │  Grid Instructions (100) → GTR (768D) → Unit-Cube Norm                         │    │
│  │                                           │                                     │    │
│  │                                           ▼                                     │    │
│  │  InstructionFeatureExtractor (encoder.py:94):                                  │    │
│  │  ────────────────────────────────────────────                                   │    │
│  │  768D → Linear(768, 128) → ReLU → LayerNorm(128)                               │    │
│  │      → Linear(128, 32) → ReLU → LayerNorm(32)     [32D bottleneck]             │    │
│  │      → Linear(32, 10) → 10D latent                                              │    │
│  │                                           │                                     │    │
│  │                                           ▼                                     │    │
│  │                               InstructionDeepKernelGP                           │    │
│  │                                                                                 │    │
│  │  Note: LayerNorm (not BatchNorm) for stability with batch_size=1               │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  Phase 2: Decoder Training (frozen encoder)                                             │
│  ──────────────────────────────────────────                                              │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                 │    │
│  │  Diverse Instructions (1000+) → GTR → FeatureExtractor [FROZEN] → 10D          │    │
│  │                              │                                    │             │    │
│  │                              ▼                                    ▼             │    │
│  │                       Target 768D           LatentDecoder (decoder.py:18)       │    │
│  │                                             ─────────────────────────────        │    │
│  │                                             10D → Linear(10, 32) → ReLU → BN   │    │
│  │                                                 → Linear(32, 64) → ReLU → BN   │    │
│  │                                                 → Linear(64, 256) → ReLU → BN  │    │
│  │                                                 → Linear(256, 768)              │    │
│  │                                                 → L2 Normalize → 768D           │    │
│  │                                                                                 │    │
│  │  Note: BatchNorm with eval-mode switch for single-sample inference             │    │
│  │                                                                                 │    │
│  │  Loss = λ_cycle × ||z - E(D(z))||² + λ_cosine × (1 - cosine_sim)               │    │
│  │       = 1.0 × cyclic_mse + 5.0 × (1 - cosine)                                  │    │
│  │                                                                                 │    │
│  │  Training: AdamW, CosineAnnealingLR, early stopping (patience=30)              │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INFERENCE PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  Main Loop: for iteration in 1..N:                                                      │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 1: BoTorch qLogEI Optimization in 10D Latent Space                       │    │
│  │  ──────────────────────────────────────────────────────                          │    │
│  │                                                                                 │    │
│  │  a) Compute latent bounds from training data (botorch_acq.py:207):             │    │
│  │                                                                                 │    │
│  │     z_all = encoder(X_train)                                                   │    │
│  │     z_min = z_all.min(dim=0) - margin * range                                  │    │
│  │     z_max = z_all.max(dim=0) + margin * range                                  │    │
│  │     bounds = stack([z_min, z_max])  # shape (2, 10)                            │    │
│  │     margin = 0.2 (20% expansion for exploration)                               │    │
│  │                                                                                 │    │
│  │  b) Create CompositeLogEI acquisition function (botorch_acq.py:28):            │    │
│  │                                                                                 │    │
│  │     class CompositeLogEI(AcquisitionFunction):                                 │    │
│  │         """Computes LogEI(decoder(z)) for latent z"""                          │    │
│  │                                                                                 │    │
│  │         def forward(self, X):  # X: (batch, q, 10)                             │    │
│  │             embeddings = decoder(X)          # (batch, q, 768)                 │    │
│  │             return qLogEI(embeddings, best_f) # (batch,)                       │    │
│  │                                                                                 │    │
│  │  c) Multi-start L-BFGS-B optimization:                                         │    │
│  │                                                                                 │    │
│  │     LatentSpaceAcquisition.optimize():                                         │    │
│  │       1. Sample raw_samples (512) random points in bounds                      │    │
│  │       2. Evaluate CompositeLogEI on all                                        │    │
│  │       3. Select top num_restarts (64) as starting points                       │    │
│  │       4. Run L-BFGS-B from each (maxiter=200)                                  │    │
│  │       5. Return best z* = argmax LogEI(z)                                      │    │
│  │                                                                                 │    │
│  │     Options: {"maxiter": 200, "batch_limit": 5}                                │    │
│  │     RuntimeWarnings suppressed but logged at debug level                       │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                           │
│                              ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 2: Decode and Invert                                                      │    │
│  │  ─────────────────────────                                                       │    │
│  │                                                                                 │    │
│  │  embedding* = decoder(z*)   # 10D → 768D, L2 normalized                        │    │
│  │  text* = Vec2Text(embedding*)                                                  │    │
│  │                                                                                 │    │
│  │  Vec2TextInverter (inference.py:64):                                           │    │
│  │  ───────────────────────────────────                                            │    │
│  │                                                                                 │    │
│  │  model_type="32_tokens" (default):                                             │    │
│  │    - ielabgroup/vec2text_gtr-base-st_inversion (InversionModel)               │    │
│  │    - ielabgroup/vec2text_gtr-base-st_corrector (CorrectorEncoderModel)        │    │
│  │    - vec2text.invert_embeddings(embeddings, corrector, num_steps=50)          │    │
│  │    - Limit: ~32 tokens                                                         │    │
│  │                                                                                 │    │
│  │  model_type="512_tokens":                                                      │    │
│  │    - vec2text/gtr-512-noise-0.00001 (InversionModel only)                     │    │
│  │    - Direct generation without corrector                                       │    │
│  │    - Limit: ~512 tokens                                                        │    │
│  │                                                                                 │    │
│  │  Parameters: beam_width=8, max_length=128                                      │    │
│  │  Lazy loading: Models loaded on first invert() call                            │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                           │
│                              ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 3: InvBO Inversion Loop (default enabled, --no-inversion to disable)    │    │
│  │  ────────────────────────────────────────────────────────────────────           │    │
│  │                                                                                 │    │
│  │  Purpose: Close the "misalignment gap" between decoder output and              │    │
│  │           what Vec2Text can actually reconstruct                                │    │
│  │                                                                                 │    │
│  │  for inv_iter in 1..max_inversion_iters (10):                                  │    │
│  │                                                                                 │    │
│  │    ┌─────────────────────────────────────────────────────────────────────┐     │    │
│  │    │  inversion_step(text*) → InversionStepResult                       │     │    │
│  │    │  ──────────────────────────────────────────                         │     │    │
│  │    │                                                                     │     │    │
│  │    │  target_emb = GTR(text*)                                           │     │    │
│  │    │                                                                     │     │    │
│  │    │  Warm start: z_init = encoder(normalize(target_emb))               │     │    │
│  │    │                                                                     │     │    │
│  │    │  z = z_init.clone().requires_grad_(True)                           │     │    │
│  │    │  optimizer = Adam([z], lr=0.1)                                     │     │    │
│  │    │                                                                     │     │    │
│  │    │  for step in 1..100:                                               │     │    │
│  │    │      decoded = decoder(z)                                          │     │    │
│  │    │      loss = 1 - cosine_similarity(decoded, target_emb)             │     │    │
│  │    │      loss.backward()                                               │     │    │
│  │    │      optimizer.step()                                              │     │    │
│  │    │      if loss < convergence_threshold (0.01): break                 │     │    │
│  │    │                                                                     │     │    │
│  │    │  z_inv = z.detach()                                                │     │    │
│  │    │                                                                     │     │    │
│  │    │  Gap calculation (in embedding space, NOT latent L2!):             │     │    │
│  │    │  gap = 1 - cosine_sim(decoder(z_original), decoder(z_inv))         │     │    │
│  │    │                                                                     │     │    │
│  │    └─────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                                 │    │
│  │    if gap <= gap_threshold (0.1):                                              │    │
│  │        ACCEPT and break                                                         │    │
│  │    else:                                                                        │    │
│  │        z* = z_inv                     # Use inverted latent                    │    │
│  │        embedding* = decoder(z*)        # Re-decode                              │    │
│  │        text* = Vec2Text(embedding*)    # Re-invert                              │    │
│  │        continue                                                                 │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                           │
│                              ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 4: Evaluation                                                             │    │
│  │  ──────────────────                                                              │    │
│  │                                                                                 │    │
│  │  --skip-eval mode:                                                              │    │
│  │    actual_error = GP.predict(embedding*).mean                                  │    │
│  │                                                                                 │    │
│  │  LLM evaluation mode (default):                                                │    │
│  │    1. Create LLM client (Qwen/Qwen2.5-7B-Instruct via vLLM)                   │    │
│  │    2. Load validation data (1319 GSM8K samples)                                │    │
│  │    3. Evaluate instruction with evaluate_instruction()                         │    │
│  │    4. actual_error = errors / total                                            │    │
│  │                                                                                 │    │
│  │  Client reused across iterations for efficiency                                │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                           │
│                              ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 5: Trust Region Update (optional, --trust-region to enable)             │    │
│  │  ────────────────────────────────────────────────────────────────               │    │
│  │                                                                                 │    │
│  │  TuRBO-style L-infinity trust region (trust_region.py):                        │    │
│  │                                                                                 │    │
│  │  TRConfig defaults:                                                            │    │
│  │    initial_radius = 0.5                                                        │    │
│  │    min_radius = 0.05                                                           │    │
│  │    max_radius = 2.0                                                            │    │
│  │    expand_factor = 1.5                                                         │    │
│  │    contract_factor = 0.5                                                       │    │
│  │    success_threshold = 2  (consecutive successes to expand)                    │    │
│  │    failure_threshold = 3  (consecutive failures to contract)                   │    │
│  │    n_restarts_max = 5                                                          │    │
│  │                                                                                 │    │
│  │  Update logic:                                                                 │    │
│  │    if actual_error < best_error:  # SUCCESS                                    │    │
│  │        success_count++                                                         │    │
│  │        if success_count >= 2: radius *= 1.5                                    │    │
│  │    else:  # FAILURE                                                            │    │
│  │        failure_count++                                                         │    │
│  │        if failure_count >= 3: radius *= 0.5                                    │    │
│  │        if radius < 0.05: RESTART from best known                               │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                           │
│                              ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 6: GP Update (every retrain_interval iterations, default=1)             │    │
│  │  ──────────────────────────────────────────────────────────────                  │    │
│  │                                                                                 │    │
│  │  add_observation_and_retrain() in gp.py:554:                                   │    │
│  │                                                                                 │    │
│  │  1. Re-encode generated text for aligned observation:                          │    │
│  │     new_embedding = GTR(text*)                                                 │    │
│  │                                                                                 │    │
│  │  2. Add to training data:                                                      │    │
│  │     X_train = cat([X_train, new_embedding])                                    │    │
│  │     y_train = cat([y_train, actual_error])                                     │    │
│  │                                                                                 │    │
│  │  3. Update best if improved:                                                   │    │
│  │     if actual_error < y_best: y_best = actual_error                            │    │
│  │                                                                                 │    │
│  │  4. Incremental retrain (_incremental_retrain in gp.py:608):                   │    │
│  │     - PRESERVE input normalization (X_min, X_max) - prevents drift             │    │
│  │     - RECOMPUTE output normalization (y_mean, y_std) for new data              │    │
│  │     - KEEP existing feature_extractor (critical for VAE mode!)                 │    │
│  │     - WARM-START: restore GP kernel hyperparameters                            │    │
│  │     - Lower learning rate (0.001) for stability                                │    │
│  │     - Re-register Standardize outcome transform                                │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                           │
│                              ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Step 7: Visualization (--visualize, only first 10 iterations)                │    │
│  │  ─────────────────────────────────────────────────────────────                   │    │
│  │                                                                                 │    │
│  │  UMAP(10D → 2D) projection showing:                                            │    │
│  │    - EI surface (contour plot)                                                 │    │
│  │    - z_opt (red star)                                                          │    │
│  │    - z_realized (white X)                                                      │    │
│  │    - Inversion gap (dashed line between z_opt and z_realized)                  │    │
│  │    - Trust region boundary (if enabled)                                        │    │
│  │                                                                                 │    │
│  │  Saved to: results/ei_landscape_iter_N.png                                     │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
│  End of iteration: Record to iteration_history, continue to next                        │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Komponenty - Detailní Popis

### 1. GTRInstructionEncoder (`encoder.py:16`)

Wrapper kolem SentenceTransformer GTR-T5-Base pro embedding instrukcí.

```python
GTRInstructionEncoder(
    model_name="sentence-transformers/gtr-t5-base",
    normalize=True,     # L2 normalization (required for Vec2Text)
    device="auto",      # cuda > mps > cpu
)
```

**Metody:**
| Metoda | Input | Output | Popis |
|--------|-------|--------|-------|
| `encode(text)` | str | np.array (768,) | Single text → numpy |
| `encode_tensor(text)` | str | Tensor (768,) | Single text → tensor |
| `encode_batch(texts)` | List[str] | np.array (N, 768) | Batch → numpy |
| `encode_batch_tensor(texts)` | List[str] | Tensor (N, 768) | Batch → tensor |

**Poznámky:**
- L2 normalizace je **kritická** pro Vec2Text kompatibilitu
- Batch metody mají progress bar pro >100 vzorků
- Device auto-detection: CUDA → MPS → CPU

---

### 2. InstructionFeatureExtractor (`encoder.py:94`)

Deep kernel encoder pro GP (používá se v **non-VAE mode**).

```
Architecture:
768D GTR embedding
    │
Linear(768, 128) → ReLU → LayerNorm(128)
    │
Linear(128, 32) → ReLU → LayerNorm(32)    [32D bottleneck]
    │
Linear(32, 10)
    │
10D latent
```

**Klíčové vlastnosti:**
- **LayerNorm** (ne BatchNorm) pro stabilitu s batch_size=1
- Podporuje 3D input (batch, n, dim) pro BoTorch posterior computation
- `get_latent(embedding)` - convenience metoda pro single embeddings

---

### 3. InstructionVAE (`encoder.py:174`)

Variational Autoencoder pro hladký latentní prostor (**default mode**).

```
Encoder:
768D → Linear(768, 64) → ReLU → LayerNorm(64)
    → Linear(64, 32) → ReLU → LayerNorm(32)
    → Linear(32, 2×latent_dim)  # μ, log_var
              ↓
    Reparameterization: z = μ + σ × ε

Decoder:
10D → Linear(10, 32) → ReLU → LayerNorm(32)
   → Linear(32, 64) → ReLU → LayerNorm(64)
   → Linear(64, 256) → ReLU → LayerNorm(256)
   → Linear(256, 768) → L2 Normalize
              ↓
         768D output
```

**Loss:**
```python
L = (1 - cosine_sim(x, x_recon)) + β × KL(q(z|x) || N(0,1))
# KL = -0.5 × Σ(1 + log_var - μ² - exp(log_var))
```

**Metody:**
| Metoda | Popis |
|--------|-------|
| `encode(x) → (μ, log_var)` | Encode to distribution parameters |
| `reparameterize(μ, log_var) → z` | Sample using reparameterization trick |
| `decode(z) → x_recon` | Decode to L2-normalized embedding |
| `forward(x) → (x_recon, μ, log_var, z)` | Full forward pass |
| `encode_mu(x) → z` | Deterministic encode (for GP) |
| `sample_latent(n) → z` | Sample from prior N(0,1) |
| `interpolate(z1, z2, steps)` | Linear interpolation |

**Doporučené hyperparametry:**
```bash
--vae-beta 0.02         # KL weight (0.02 optimal)
--vae-epochs 10000      # Long training
--vae-annealing 500     # Slow KL ramp-up
--vae-patience 500      # High patience
```

---

### 4. VAEWithAdapter (`training.py:27`)

Wrapper kombinující **frozen VAE encoder** s **trainable adapter** pro GP.

```python
class VAEWithAdapter(nn.Module):
    def __init__(self, vae, latent_dim):
        # Freeze VAE
        for param in vae.parameters():
            param.requires_grad = False

        # Trainable adapter
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),  # 10 → 20
            nn.ReLU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),  # 20 → 10
        )

    def forward(self, x):
        z = self._vae.encode_mu(x)  # Frozen
        return self.adapter(z)       # Trainable
```

**Účel:** GP může naučit lepší reprezentaci při zachování smooth latent space z VAE.

---

### 5. InstructionDeepKernelGP (`gp.py:127`)

Gaussian Process s deep kernel pro instruction optimization.

```python
class InstructionDeepKernelGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # Required for BoTorch

    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        # Mean: ZeroMean
        # Kernel: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=10))
        #   - Lengthscale prior: Gamma(3.0, 6.0)
        #   - Outputscale prior: Gamma(2.0, 0.15)
```

**Klíčové vlastnosti:**
- Inherits from `GPyTorchModel` → BoTorch compatible (qLogEI, etc.)
- ARD (10 lengthscales) pro per-dimension weighting
- Matérn 5/2 - smooth but flexible

---

### 6. GPWithEI (`gp.py:198`)

Wrapper pro GP s Expected Improvement a incremental retraining.

```python
GPWithEI(device="cuda", latent_dim=10)
```

**Klíčové metody:**

| Metoda | Popis |
|--------|-------|
| `set_training_data(embeddings, error_rates)` | Set training data |
| `train(epochs, lr, patience)` | Train GP with noise constraint 1e-4 |
| `predict(embedding) → (mean, std)` | GP prediction |
| `expected_improvement(embedding, xi)` | Standard EI (no clipping) |
| `log_expected_improvement(embedding, xi)` | Scalar LogEI |
| `log_expected_improvement_tensor(embedding, xi)` | Differentiable LogEI tensor |
| `add_observation_and_retrain(embedding, error)` | Incremental update |
| `get_training_size()` | Number of training samples |

**Training details:**
- Unit-cube normalization: `X_norm = (X - X_min) / (X_max - X_min)`
- Standardize outcome transform: `y_transformed = Standardize(m=1)(y)`
- Noise constraint: `GaussianLikelihood(noise_constraint=GreaterThan(1e-4))`
- Cholesky jitter: 1e-4

**Incremental retraining (`_incremental_retrain`):**
- **Preserves** X_min, X_max (input normalization) - prevents drift
- **Recomputes** y_mean, y_std (output normalization) for new data
- **Keeps** existing feature_extractor (critical for VAE mode!)
- **Warm-starts** GP kernel hyperparameters
- Lower learning rate (0.001) for stability

---

### 7. LogEI Implementation (`gp.py:26-125`)

Numericky stabilní Log Expected Improvement z článku "Unexpected Improvements to Expected Improvement" (NeurIPS 2023).

```python
def log_h(z: float) -> float:
    """log(h(z)) where h(z) = φ(z) + z·Φ(z)

    Three branches for numerical stability:
    1. z > -1: Direct computation φ(z) + z·Φ(z)
    2. -1/√ε < z ≤ -1: erfcx-based computation
    3. z ≤ -1/√ε: Asymptotic approximation -z²/2 - log(2π)/2 - 2·log(|z|)
    """

def log_h_tensor(z: torch.Tensor) -> torch.Tensor:
    """Tensor version with autograd support.

    Uses piecewise approximation with smooth sigmoid transition
    between direct and asymptotic branches at z = -5.
    """

# LogEI formula:
LogEI(x) = log_h(z) + log(σ)
# where z = (y_best - μ - ξ) / σ
```

**Výhody:**
- Numericky stabilní i pro velmi malé EI hodnoty
- Umožňuje gradient-based optimalizaci (BoTorch)
- Nedochází k underflow (log místo tiny floats)

---

### 8. BoTorch Acquisition (`botorch_acq.py`)

**CompositeLogEI** (`botorch_acq.py:28`) - Acquisition function pro optimalizaci přes decoder:

```python
class CompositeLogEI(AcquisitionFunction):
    """qLogEI that works through decoder transformation.

    Pipeline: z (10D) → decoder → embedding (768D) → GP → qLogEI
    """

    def forward(self, X):  # X: (batch, q, d)
        X_flat = X.reshape(-1, d)
        embeddings = self.decoder(X_flat)
        embeddings = embeddings.view(*batch_shape, q, 768)
        return self._base_acq(embeddings)
```

**LatentSpaceAcquisition** (`botorch_acq.py:102`) - Multi-start L-BFGS-B optimizer:

```python
class LatentSpaceAcquisition:
    def optimize(
        self,
        best_f: float,
        num_restarts: int = 20,
        raw_samples: int = 512,
        options: dict = {"maxiter": 200, "batch_limit": 5},
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Sample raw_samples random points in bounds
        2. Evaluate CompositeLogEI on all
        3. Select top num_restarts as starting points
        4. Run L-BFGS-B from each
        5. Return (best_candidate, log_ei_value)
        """
```

**get_latent_bounds** (`botorch_acq.py:207`):
```python
def get_latent_bounds(encoder, X_train, X_min, X_max, margin=0.2):
    """Compute latent space bounds from training data.

    bounds = [z_min - margin*range, z_max + margin*range]
    """
```

---

### 9. LatentDecoder (`decoder.py:18`)

Decoder z 10D latent do 768D embedding (**non-VAE mode only**).

```
Architecture:
10D latent
    │
Linear(10, 32) → ReLU → BatchNorm(32)
    │
Linear(32, 64) → ReLU → BatchNorm(64)
    │
Linear(64, 256) → ReLU → BatchNorm(256)
    │
Linear(256, 768)
    │
L2 Normalize (if normalize=True)
    │
768D Vec2Text-compatible embedding
```

**Poznámka:**
- Používá **BatchNorm** (ne LayerNorm)
- Pro single-sample inference: dočasně přepne do eval mode

**V VAE mode se místo toho používá `VAE.decode()` wrapped v `VAEDecoderWrapper`.**

---

### 10. DecoderCyclicLoss (`decoder.py:104`)

Loss funkce pro decoder training (non-VAE mode).

```python
DecoderCyclicLoss(
    lambda_cycle=1.0,     # Weight for cyclic loss
    lambda_embedding=0.5, # Weight for cosine embedding loss
    lambda_recon=1.0,     # Weight for MSE reconstruction loss
    tolerance=0.1,        # Soft tolerance for cyclic loss
)
```

**Loss components:**
```python
# Cyclic loss with soft tolerance
cycle_dist = ||z - z_recon||
L_cycle = max(0, cycle_dist - tolerance)²

# Cosine embedding loss
L_embedding = 1 - cosine_sim(decoded, target)

# MSE reconstruction loss
L_recon = MSE(decoded, target)

# Total
L = λ_cycle × L_cycle + λ_embedding × L_embedding + λ_recon × L_recon
```

---

### 11. Vec2TextInverter (`inference.py:64`)

Lazy-loaded Vec2Text pro embedding-to-text inverzi.

```python
Vec2TextInverter(
    num_steps=50,           # Correction iterations (32_tokens) / max_new_tokens (512_tokens)
    beam_width=8,           # Beam search width
    max_length=128,         # Max output tokens
    device="auto",
    model_type="32_tokens", # or "512_tokens"
)
```

**Model types:**

| Type | Models | Limit | Method |
|------|--------|-------|--------|
| `32_tokens` | ielabgroup InversionModel + CorrectorEncoderModel | ~32 tokens | `vec2text.invert_embeddings()` |
| `512_tokens` | vec2text/gtr-512-noise-0.00001 | ~512 tokens | Direct `generate()` |

**Lazy loading:** Models loaded on first `invert()` call.

---

### 12. InvBOInference (`inference.py:248`)

Kompletní inference pipeline.

```python
InvBOInference(
    gp=trained_gp,
    decoder=trained_decoder,
    gtr=gtr_encoder,            # Optional, created if None
    vec2text_steps=50,
    vec2text_beam=4,
    vec2text_model="32_tokens", # or "512_tokens"
    trust_region_config=None,   # TRConfig or None
    seed=None,                  # For reproducibility
)
```

**Klíčové metody:**

| Metoda | Popis |
|--------|-------|
| `get_best_training_latent()` | → (latent, idx, error) from training data |
| `optimize_latent_botorch(num_restarts, raw_samples)` | → (z_opt, log_ei) **[HLAVNÍ]** |
| `optimize_latent_lbfgs(n_restarts, max_iter, xi)` | Scipy L-BFGS-B (deprecated) |
| `optimize_latent_random(n_candidates, xi)` | Random sampling |
| `optimize_latent_adaptive(n_candidates, xi, use_trust_region)` | Trust region sampling |
| `run_single_iteration(...)` | Complete iteration with BoTorch + inversion |
| `inversion_step(text, n_steps, lr)` | InvBO-style inversion |
| `invert_latent(latent, validate)` | z → text with validation |
| `run_optimization(method, ...)` | Deprecated, use run_single_iteration |
| `optimize_with_inversion(method, ...)` | Deprecated, use run_single_iteration |
| `validate_inversion_gap(n_samples)` | Measure inversion gap statistics |

---

### 13. TrustRegionManager (`trust_region.py:54`)

TuRBO-style trust region pro 10D latent space.

```python
TrustRegionManager(
    anchor=best_latent,    # 10D tensor
    config=TRConfig(),     # Configuration
    device="cuda",
)
```

**TRConfig defaults:**
```python
TRConfig(
    initial_radius=0.5,
    min_radius=0.05,
    max_radius=2.0,
    expand_factor=1.5,
    contract_factor=0.5,
    success_threshold=2,
    failure_threshold=3,
    n_restarts_max=5,
)
```

**Metody:**
| Metoda | Popis |
|--------|-------|
| `is_within_region(z)` | Check if point in L∞ ball |
| `distance_to_boundary(z)` | Signed distance (positive = inside) |
| `project_to_region(z)` | Clip to bounds |
| `sample_in_region(n_samples)` | Uniform sampling in L∞ ball |
| `sample_with_perturbation(center, n, scale)` | Gaussian around center, clipped |
| `update(z, error, best_y)` | Update radius based on result |
| `get_status()` | Current state dict |

---

### 14. InvBOTrainer (`training.py:100`)

Two-phase trainer pro InvBO decoder.

```python
trainer = InvBOTrainer(TrainingConfig(...))
gp, decoder = trainer.train(verbose=True)
```

**TrainingConfig defaults:**
```python
TrainingConfig(
    # Data
    instructions_path="datasets/inversion/instructions_100.txt",
    grid_path="datasets/inversion/grid_100_qend.jsonl",
    diverse_instructions_path="datasets/inversion/diverse_instructions_1000.json",

    # Architecture
    latent_dim=10,
    embedding_dim=768,

    # Phase 1: GP
    gp_epochs=5000,
    gp_lr=0.01,
    gp_patience=50,
    gp_initial_top_k=25,

    # Phase 2: Decoder (non-VAE)
    decoder_epochs=500,
    decoder_lr=0.001,
    decoder_patience=30,
    decoder_batch_size=64,
    lambda_cycle=1.0,
    lambda_cosine=5.0,
    cycle_tolerance=0.0,

    # VAE mode
    use_vae=False,  # CLI default is True
    vae_beta=0.1,   # CLI default is 0.02
    vae_epochs=10000,
    vae_lr=0.0003,
    vae_annealing_epochs=500,
    vae_patience=500,

    device="cuda",
)
```

**Training flow:**

1. `load_data()` - Load instructions and error rates from grid
2. `train()`:
   - **VAE mode**: `train_vae()` → `train_gp_with_vae()` → decoder = VAE.decode()
   - **Standard mode**: `train_phase1()` → `train_phase2()`
3. `save(path)` / `load(path)` - Persistence

---

## Datové Soubory

| Soubor | Popis | Samples | Použití |
|--------|-------|---------|---------|
| `datasets/inversion/instructions_100.txt` | 100 instrukcí v 10 kategoriích | 100 | Deprecated (grid obsahuje instrukce) |
| `datasets/inversion/grid_100_qend.jsonl` | Instrukce + error rates (Q_end format) | 100 | GP training, baseline |
| `datasets/inversion/diverse_instructions_1000.json` | Diverse instrukce (APE generated) | 1000 | VAE training / Decoder training |

---

## CLI Parametry - Kompletní Reference

### Základní

| Parametr | Default | Popis |
|----------|---------|-------|
| `--iterations` | 1 | Počet optimalizačních iterací |
| `--skip-eval` | False | Použít GP predikci místo LLM evaluace |
| `--no-vae` | False | Vypnout VAE mode (**VAE je default**) |
| `--visualize` | False | Generovat EI landscape vizualizace |
| `--seed` | 42 | Random seed pro reprodukovatelnost |

### Data Paths

| Parametr | Default | Popis |
|----------|---------|-------|
| `--instructions` | datasets/inversion/instructions_100.txt | Path to instructions |
| `--grid` | datasets/inversion/grid_100_qend.jsonl | Path to grid JSONL |
| `--validation` | hbbops_improved_2/data/validation.json | Validation data for LLM eval |
| `--ape-cache` | datasets/inversion/diverse_instructions_1000.json | APE instructions |
| `--skip-ape` | False | Skip APE, use only grid_100 for VAE |

### BoTorch Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--n-restarts` | 64 | Počet L-BFGS-B restarts pro BoTorch qLogEI |
| `--raw-samples` | 512 | Raw samples pro inicializaci optimalizace |

### VAE Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--vae-beta` | 0.02 | KL regularization weight |
| `--vae-epochs` | 10000 | VAE training epochs |
| `--vae-annealing` | 500 | KL annealing epochs (0 → beta) |
| `--vae-patience` | 500 | Early stopping patience |

### Inversion Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--use-inversion` | True | Použít InvBO inversion loop |
| `--no-inversion` | False | Vypnout inversion loop |
| `--max-inversion-iters` | 10 | Maximum inversion loop iterations |
| `--gap-threshold` | 0.1 | Cosine distance threshold pro accept |

### Vec2Text Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--vec2text-steps` | 50 | Correction steps |
| `--vec2text-beam` | 8 | Beam width |
| `--vec2text-model` | 32_tokens | Model type (32_tokens / 512_tokens) |

### GP Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--gp-epochs` | 5000 | GP training epochs |
| `--gp-patience` | 50 | GP early stopping patience |
| `--latent-dim` | 10 | Latent space dimension |
| `--retrain-interval` | 1 | Retrain GP every N iterations |
| `--retrain-epochs` | 1000 | Epochs pro GP retraining |

### LLM Evaluation

| Parametr | Default | Popis |
|----------|---------|-------|
| `--model` | Qwen/Qwen2.5-7B-Instruct | Model pro evaluaci |
| `--eval-samples` | 1319 | Samples (1319 = plný GSM8K validation) |

### Trust Region (DISABLED BY DEFAULT)

| Parametr | Default | Popis |
|----------|---------|-------|
| `--trust-region` | False | Zapnout trust region |
| `--tr-initial` | 0.5 | Initial trust region radius |
| `--tr-min` | 0.05 | Minimum radius (triggers restart) |
| `--tr-max` | 2.0 | Maximum radius |

### Save/Load

| Parametr | Default | Popis |
|----------|---------|-------|
| `--save` | None | Save trained models to directory |
| `--load` | None | Load pre-trained models from directory |

### Validation

| Parametr | Default | Popis |
|----------|---------|-------|
| `--validate-gap` | False | Validate inversion gap on random samples |
| `--gap-samples` | 10 | Number of samples for gap validation |

---

## Výstupní Formáty

### Results JSON

```json
{
  "timestamp": "20251229_120000",
  "seed": 42,
  "method": "InvBO Decoder",
  "args": { "iterations": 10, "vae_beta": 0.02, ... },
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
      "best_error_so_far": 0.1077,
      "trust_region_radius": null,
      "gp_samples": 101
    }
  ],
  "improvement": 0.0127,
  "vae_quality_metrics": {
    "cosine_mean": 0.98,
    "cosine_std": 0.01,
    "cosine_min": 0.95,
    "cosine_max": 0.99,
    "mse_mean": 0.001,
    "l2_relative_error": 0.02,
    "latent_norm_mean": 2.5,
    "latent_var_mean": 0.45,
    "latent_var_min": 0.1,
    "latent_var_max": 0.8,
    "active_dims": 10,
    "total_dims": 10,
    "kld_mean": 5.2,
    "posterior_collapsed": false
  }
}
```

### Log File

Automaticky ukládán do `generation/invbo_decoder/results/run_TIMESTAMP.log`

Obsahuje kompletní výstup včetně:
- Timestamp a seed
- Konfigurace
- Training progress
- Iteration details
- Final results

---

## Známé Problémy a Řešení

### 1. Misalignment Problem (InvBO paper)

**Problém:** Decoder může produkovat embeddingy, které po Vec2Text inverzi dají odlišný text.

**Řešení:**
- **Inversion loop:** `z_inv = argmin ||decoder(z) - GTR(text)||²`
- Gap měřen jako **cosine distance v embedding space** (stabilnější než L2 v latentu)
- Threshold 0.1 pro accept/reject
- Max 10 iterations

### 2. GP Overconfidence

**Problém:** GP může být příliš jistý svými predikcemi, což vede k LogEI = -5000.

**Řešení:**
- **Noise constraint** `GreaterThan(1e-4)` jako per BoTorch recommendation
- Zajišťuje dostatečnou uncertainty pro exploration

### 3. Posterior Collapse (VAE)

**Problém:** VAE může ignorovat latenty (z → konstanta).

**Řešení:**
- **KL annealing:** β = 0 → target přes `vae_annealing_epochs`
- Nižší `vae_beta` (0.02)
- Early stopping sleduje **reconstruction loss** (ne total)
- Delší trénink s vysokou patience (10000 epochs, patience 500)

### 4. Vec2Text Limitations

**Problém:** Selhává pro delší texty (>30 tokenů).

**Řešení:**
- Instruction-only přístup (ne celé prompty)
- `max_length=128` v Vec2Text
- Alternativa: `model_type="512_tokens"` pro delší texty

### 5. Numerical Instability in EI

**Problém:** Standard EI underflows pro velmi malé hodnoty.

**Řešení:**
- BoTorch `qLogExpectedImprovement` - numericky stabilní
- `log_h_tensor()` s autograd pro gradient optimization
- Tři branches pro různé rozsahy z-score

### 6. GP Incremental Retraining Drift

**Problém:** Normalization drift při přidávání nových observací.

**Řešení:**
- **Preserve** X_min, X_max (input normalization)
- **Recompute** y_mean, y_std (output normalization)
- **Keep** existing feature_extractor (critical for VAE mode)
- **Warm-start** GP kernel hyperparameters
- Lower learning rate (0.001)

### 7. VAEWithAdapter State Dict Mismatch

**Problém:** Při loading modelu se feature_extractor musí správně rekonstruovat.

**Řešení:**
- V `load()` se používá `VAEWithAdapter(vae, latent_dim)` stejně jako při training
- Adapter state dict se loaduje z `gp.pt["gp_model"]` přes GP model state

---

## Porovnání s COWBOYS

| Feature | InvBO Decoder | COWBOYS |
|---------|---------------|---------|
| Latent Dim | 10D | 32D |
| Encoder | VAE (default) / Deep Kernel | VAE only |
| VAE Beta | 0.02 (hladký prostor) | 0.01 |
| Optimizer | BoTorch qLogEI (gradient) | pCN MCMC (sampling) |
| Trust Region | Disabled (optional) | Custom (L2) |
| Inversion | InvBO-style loop (10 iters) | Direct Vec2Text |
| GP Retraining | Incremental (preserved norm) | Batch |
| Noise Constraint | 1e-4 (BoTorch recommendation) | Default |
| Gap Metric | Cosine in embedding space | L2 in latent |
| Standardize | BoTorch outcome transform | Manual |
| Reproducibility | Seed 42 (Python, NumPy, PyTorch, cuDNN) | N/A |

---

## Reference

- **InvBO**: Deshwal et al., 2024 - "Inversion-Based BO with Structured Inputs" (NeurIPS 2024)
- **LogEI**: Ament et al., 2023 - "Unexpected Improvements to Expected Improvement" (NeurIPS 2023)
- **Vec2Text**: Morris et al., 2023 - "Text Embeddings Reveal (Almost) As Much As Text"
- **BoTorch**: Balandat et al., 2020 - "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization"
- **TuRBO**: Eriksson et al., 2019 - "Scalable Global Optimization via Local Bayesian Optimization"
