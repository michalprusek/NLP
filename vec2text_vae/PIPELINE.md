# Cascading Matryoshka Funnel Flow

Dimensionality reduction for GTR embeddings (768D → 128D) using Cascading Matryoshka architecture for Staged LSBO (Latent Space Bayesian Optimization).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   Cascading Matryoshka Flow Architecture                         │
│                        768D → 128D (8 stages × 16D)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ENCODER (Hierarchical - each level conditions on previous)                     │
│   ════════════════════════════════════════════════════════                       │
│                                                                                  │
│   768D GTR embedding (x)                                                         │
│       │                                                                          │
│       ▼                                                                          │
│   ┌────────────────────┐                                                         │
│   │ encoder_16(x)      │ → z[0:16]    ← Coarsest: hlavní téma/styl              │
│   └────────────────────┘                                                         │
│       │                                                                          │
│       ▼                                                                          │
│   ┌────────────────────┐                                                         │
│   │ encoder_32(x,z[:16])│ → z[16:32]  ← Doplňující detaily                      │
│   └────────────────────┘                                                         │
│       │                                                                          │
│       ▼                                                                          │
│   ... (encoder_48, encoder_64, ..., encoder_128)                                 │
│       │                                                                          │
│       ▼                                                                          │
│   z[0:128] = plný latent (importance-ordered)                                    │
│                                                                                  │
│   DECODER (Cascade Prediction + Final Reconstruction)                            │
│   ════════════════════════════════════════════════════                           │
│                                                                                  │
│   z[0:k] (částečný latent)                                                       │
│       │                                                                          │
│       ▼                                                                          │
│   ┌────────────────────┐                                                         │
│   │ predictor_k→k+16   │ → z[k:k+16]  ← Predikce dalších 16D                    │
│   └────────────────────┘                                                         │
│       │                                                                          │
│       ▼                                                                          │
│   ┌────────────────────┐                                                         │
│   │ final_decoder      │ → 768D embedding                                        │
│   └────────────────────┘                                                         │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Parameters: ~16M   |   Latent: 128D   |   Stages: 8×16D                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Current Results (Epoch 99/100)

**Checkpoint**: `vec2text_vae/checkpoints/cascading_matryoshka_funnel_best.pt` (66MB)

| Dimenze | Val Cosine Similarity |
|---------|----------------------|
| 16D | 0.6830 |
| 32D | 0.7300 |
| 48D | 0.7642 |
| 64D | 0.7906 |
| 80D | 0.8121 |
| 96D | 0.8297 |
| 112D | 0.8432 |
| **128D** | **0.8482** |

**Average val cos_sim**: 0.7876

## Directory Structure

```
vec2text_vae/
├── matryoshka_funnel.py         # CascadingMatryoshkaGTRFunnelFlow model
├── train_matryoshka_funnel.py   # Training script
├── test_reconstruction.py       # Evaluation utilities
├── staged_lsbo.py               # Staged LSBO optimizer
├── checkpoints/
│   └── cascading_matryoshka_funnel_best.pt  # Best checkpoint (66MB)
└── cache/
    ├── combined_texts.json      # 1.5M instruction texts
    └── gtr_embeddings_full.pt   # 1.5M GTR embeddings (4.7GB)
```

## Training

```bash
# Full training on A100 80GB
tmux new-session -d -s cascade_train \
    "CUDA_VISIBLE_DEVICES=0 uv run python vec2text_vae/train_matryoshka_funnel.py \
    --epochs 100 --batch-size 16384 --latent-dim 128 \
    2>&1 | tee results/cascade_128d_$(date +%Y%m%d_%H%M%S).log"
```

| Parameter | Value |
|-----------|-------|
| Batch size | 8192-16384 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| LR Schedule | Cosine with 10 epoch warmup |
| Dataset | 1.5M GTR embeddings |
| Split | 80/10/10 (train/val/test) |

## Usage

### Load Trained Model

```python
import torch
from vec2text_vae.matryoshka_funnel import CascadingMatryoshkaGTRFunnelFlow

# Load checkpoint
ckpt = torch.load(
    "vec2text_vae/checkpoints/cascading_matryoshka_funnel_best.pt",
    weights_only=False
)

# Create model
flow = CascadingMatryoshkaGTRFunnelFlow(
    input_dim=768,
    latent_dim=128,
    matryoshka_dims=(16, 32, 48, 64, 80, 96, 112, 128),
).cuda()
flow.load_state_dict(ckpt['model_state_dict'])
flow.eval()
```

### Encode/Decode

```python
# Encode GTR embedding to 128D latent
z = flow.encode(gtr_embedding)  # (batch, 128)

# Decode from partial latent (e.g., only first 64D)
z_partial = torch.zeros(batch, 128)
z_partial[:, :64] = z[:, :64]
x_recon = flow.decode(z_partial, active_dim=64)

# Full decode
x_recon_full = flow.decode(z, active_dim=128)
```

## Staged LSBO Algorithm

### Why Staged?

| Approach | Evaluations | GP Performance |
|----------|-------------|----------------|
| Standard BO (128D) | ~10,000+ | Poor (curse of dimensionality) |
| **Staged LSBO (8×16D)** | ~225 | Excellent (GP works well in 16D) |

### Algorithm

```
Stage 1: GP optimizes z[0:16]     (50 trials)  → cos_sim ~0.68
Stage 2: GP optimizes z[16:32]    (40 trials)  → cos_sim ~0.73
Stage 3: GP optimizes z[32:48]    (35 trials)  → cos_sim ~0.76
Stage 4: GP optimizes z[48:64]    (30 trials)  → cos_sim ~0.79
Stage 5: GP optimizes z[64:80]    (25 trials)  → cos_sim ~0.81
Stage 6: GP optimizes z[80:96]    (20 trials)  → cos_sim ~0.83
Stage 7: GP optimizes z[96:112]   (15 trials)  → cos_sim ~0.84
Stage 8: GP optimizes z[112:128]  (10 trials)  → cos_sim ~0.85
─────────────────────────────────────────────────────────────
Total: 225 evaluations (vs 10,000+ for full 128D)
```

Each stage uses warm-start from cascade predictor for next stage initialization.

## GTR Encoding (Vec2text-compatible)

```python
import vec2text.models.model_utils as model_utils
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# Load GTR-T5-Base encoder
model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base")
encoder = model.encoder.to(device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

# Encode text
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
embeddings = model_utils.mean_pool(model_output.last_hidden_state, inputs['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=-1)  # L2 normalize
```

## References

- **Matryoshka Representation Learning** (NeurIPS 2022): https://arxiv.org/abs/2205.13147
- **Vec2Text** (Morris et al., 2023): https://arxiv.org/abs/2310.06816
- **GTR** (Ni et al., 2022): https://arxiv.org/abs/2112.07899
- **LSBO** (Maus et al., 2022): https://arxiv.org/abs/2201.00245
