# Soft-Prompt VAE Pipeline (CDP-VAE)

> **C**ontrastive **D**enoising **P**rompt **VAE** - VAE architecture using soft prompts for text generation with Llama-3.1-8B backbone.
> Features InfoNCE contrastive learning, adaptive collapse monitoring, and token-level augmentation.
> Designed for NeurIPS-quality research on 2x NVIDIA L40S (96GB VRAM).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SOFT-PROMPT VAE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ENCODER                                 DECODER                                 │
│  ────────                                ────────                                │
│  Instruction (512 tokens max)            Latent z (64 dims)                      │
│      │                                       │                                   │
│      ▼                                       ▼                                   │
│  ┌────────────────┐                    ┌─────────────────┐                      │
│  │ Llama-3.1-8B   │                    │ SoftPrompt      │                      │
│  │   (frozen)     │                    │  Projector      │                      │
│  │  output: 4096  │                    │ 64→2048→4096→   │                      │
│  └───────┬────────┘                    │ 32×4096         │                      │
│          │ hidden states               └────────┬────────┘                      │
│          ▼                                      │                               │
│  ┌────────────────┐                    ┌────────┴────────┐                      │
│  │   Attention    │                    │   32 Soft       │                      │
│  │    Pooling     │                    │   Prompts       │                      │
│  │  8 heads, 4096 │                    │   (4096 each)   │                      │
│  └───────┬────────┘                    └────────┬────────┘                      │
│          │ pooled (4096)                        │                               │
│          ▼                                      ▼                               │
│  ┌────────────────┐                    ┌─────────────────┐                      │
│  │ Variational    │                    │ Word Dropout    │                      │
│  │   Encoder      │                    │    (40%)        │                      │
│  │ 4096→2048→2048 │                    │ [response_ids]  │                      │
│  │  → μ,logσ²(64) │                    └────────┬────────┘                      │
│  └───────┬────────┘                             │                               │
│          │                                      ▼                               │
│          ▼                             ┌─────────────────┐                      │
│    z ~ N(μ, σ²)                        │ Llama-3.1-8B    │──▶ Response          │
│       [64 dims]                        │   (LoRA r=64)   │   (1024 tokens max)  │
│          │                             └─────────────────┘                      │
│          ▼                                                                      │
│    ┌───────────┐                                                                │
│    │  BoW Head │──▶ Bag-of-Words logits (vocab_size)                           │
│    │  64→128K  │                                                                │
│    └───────────┘                                                                │
│                                                                                  │
│  CDP-VAE ADDITIONS                                                               │
│  ─────────────────                                                               │
│    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐         │
│    │ Text Augmenter  │     │ Collapse Monitor│     │   InfoNCE Loss  │         │
│    │ (span mask,     │────▶│ (AU tracking,   │────▶│ (contrastive    │         │
│    │  word dropout)  │     │  MI estimate)   │     │  regularization)│         │
│    └─────────────────┘     └─────────────────┘     └─────────────────┘         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Model Architecture (`model.py`)

#### AttentionPooling
Learns to pool a sequence of hidden states into a single vector using multi-head self-attention.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 4096 | Llama hidden dimension |
| `num_heads` | 8 | Attention heads |
| `dropout` | 0.1 | Dropout probability |

**Mechanism**:
- Learnable query token: `(1, 1, 4096)` initialized with scale `hidden_dim^(-0.5)`
- `MultiheadAttention(embed_dim=4096, num_heads=8, dropout=0.1)`
- Padding mask: `attention_mask == 0` marks positions to ignore
- LayerNorm applied after squeeze

#### VariationalEncoder
Maps pooled representation to latent distribution parameters.

| Layer | Dimensions | Activation |
|-------|------------|------------|
| Input | 4096 | - |
| Hidden 1 | 4096 → 2048 | GELU + Dropout(0.1) |
| Hidden 2 | 2048 → 2048 | GELU + Dropout(0.1) |
| mu_head | 2048 → 64 | None |
| logvar_head | 2048 → 64 | Clamp[-10, 10] |

**Critical Initialization**:
- `logvar_head.weight`: N(0, 0.01) - small random
- `logvar_head.bias`: constant -2.0 - prevents initial variance explosion
- NOT zeros - would block gradient flow

**Reparameterization Trick**:
```python
if training:
    std = exp(0.5 * logvar)
    eps = randn_like(std)
    z = mu + std * eps
else:
    z = mu  # Deterministic at inference
```

#### SoftPromptProjector
Projects latent z to soft prompt embeddings.

| Layer | Dimensions | Activation |
|-------|------------|------------|
| Input | 64 | - |
| Hidden 1 | 64 → 2048 | GELU + Dropout(0.1) |
| Hidden 2 | 2048 → 4096 | GELU + Dropout(0.1) |
| Output | 4096 → 131072 (32×4096) | Reshape |
| Position Emb | (1, 32, 4096) | N(0, 0.02) init |
| LayerNorm | 4096 | - |

**Output**: `(batch, 32, 4096)` - 32 soft prompt tokens

#### DeepPrefixProjector (Optional)
Injects latent z into all Llama attention layers via past_key_values.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_layers` | 32 | Llama layers |
| `num_heads` | 8 | GQA heads (not 32) |
| `head_dim` | 128 | 4096/32 attention heads |
| `prefix_len` | 32 | Same as num_soft_tokens |
| `hidden_dim` | 2048 | Shared projection hidden |
| `bottleneck_dim` | 256 | Per-layer bottleneck |

**Architecture**:
```
Shared: z(64) → 2048 → 2048 → 32×256 → LayerNorm
Per-layer (×32): bottleneck(256) → key(1024), value(1024)
Layer scales: learnable, init=0.1
```

#### LoRA Configuration
Applied to Llama-3.1-8B decoder:

| Parameter | Value |
|-----------|-------|
| `r` (rank) | 64 |
| `lora_alpha` | 128 |
| `scaling` | 128/64 = 2.0 |
| `lora_dropout` | 0.05 |
| `use_rslora` | True |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

**Trainable params**: ~2-3% of base model (168M of 8B)

#### Word Dropout
Forces decoder to rely on latent z instead of autoregressive context.

```python
def apply_word_dropout(input_ids, attention_mask, dropout_rate=0.4):
    dropout_mask = bernoulli(full_like(input_ids, 1.0 - dropout_rate))
    dropout_mask[:, 0] = True  # Never drop first token
    dropped_ids = where(dropout_mask, input_ids, pad_token_id)
    return dropped_ids
```

**Effect**: Without word dropout, decoder copies previous token → ignores soft prompts. With word dropout, decoder must "ask" soft prompts what came before.

### 2. Loss Function (`loss.py`)

#### ELBO Loss (CDP-VAE Extended)
```
L_total = L_recon + β × L_KL + λ_bow × L_bow + λ_nce × L_InfoNCE
```

| Component | Formula | Default Weight |
|-----------|---------|----------------|
| **Reconstruction** | CrossEntropy(logits[:-1], labels[1:]) | 1.0 |
| **KL Divergence** | 0.5 × mean(μ² + σ² - 1 - logσ²) | β (annealed) |
| **Bag-of-Words** | BCE(bow_logits, token_presence) | 1.5 |
| **InfoNCE** | -log(exp(sim(z,z⁺)/τ) / Σⱼexp(sim(z,zⱼ)/τ)) | 0.1 |

#### Cyclical KL Annealing
Based on Fu et al. (2019): "Cyclical Annealing Schedule"

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_cycles` | 4 | Number of complete cycles |
| `ratio` | 0.5 | 50% ramp, 50% hold |
| `beta_max` | 1.0 | Maximum KL weight |

**Schedule per cycle**:
```
steps_per_cycle = total_steps // 4
ramp_steps = steps_per_cycle * 0.5

if cycle_pos < ramp_steps:
    beta = beta_max * (cycle_pos / ramp_steps)  # 0 → 1.0
else:
    beta = beta_max  # Hold at 1.0
```

#### Free Bits Mechanism
Prevents posterior collapse by maintaining minimum KL per latent dimension.

| Parameter | Value |
|-----------|-------|
| `free_bits` | 0.5 nats |
| `reduction` | sum (then scaled by latent_dim) |

```python
kl_per_dim = 0.5 * (μ² + exp(logvar) - 1 - logvar)  # (batch, 64)
kl_per_dim = kl_per_dim.mean(dim=0)  # (64,)
kl_clamped = clamp(kl_per_dim, min=0.5)  # Free bits threshold
active_dims = (kl_per_dim > 0.5).sum()  # Monitoring
kl_loss = kl_clamped.sum() / latent_dim  # Normalized
```

#### Bag-of-Words Auxiliary Loss
Multi-label classification to predict which tokens appear in target.

```python
bow_target = zeros(batch_size, vocab_size)
for b in range(batch_size):
    valid_ids = target_ids[b][target_ids[b] != -100]
    bow_target[b].scatter_(0, valid_ids, 1.0)

bow_loss = binary_cross_entropy_with_logits(bow_logits, bow_target)
```

#### InfoNCE Contrastive Loss (CDP-VAE)
Forces encoder to produce distinguishable latent representations by maximizing agreement between positive pairs (original and augmented views).

```python
# Similarity matrix between anchor and positive pairs
sim_matrix = mm(z_anchor, z_positive.T) / temperature  # (batch, batch)

# Positive similarities on diagonal, negatives elsewhere
pos_sim = sim_matrix.diag()
neg_sim = sim_matrix.masked_fill(eye_mask, -inf)

# Cross-entropy loss (positive should have highest similarity)
loss = cross_entropy(cat([pos_sim, neg_sim], dim=1), labels=zeros(batch))
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `temperature` | 0.07 | Lower = more discriminative |
| `normalize` | True | L2-normalize latents before similarity |
| `contrastive_weight` | 0.1 | Weight in total loss |

### 3. Text Augmentation (`augmentation.py`)

Token-level augmentation creates positive pairs for InfoNCE loss without expensive detokenization.

#### Augmentation Strategies

| Strategy | Description | Default |
|----------|-------------|---------|
| **Span Masking** | Replace random contiguous spans with UNK | prob=0.15, max_len=5 |
| **Word Dropout** | Randomly replace tokens with UNK | rate=0.1 |
| **Local Shuffle** | Permute tokens within windows (disabled) | window=3 |

```python
# Augmentation pipeline
augmenter = TextAugmenter(AugmentationConfig(
    span_mask_prob=0.15,
    word_dropout_rate=0.1,
))
aug_ids, aug_mask = augmenter.augment(input_ids, attention_mask)
```

**Key Features**:
- All operations work on tokenized inputs (no detokenization)
- Special tokens (BOS, EOS, PAD) are preserved
- Augmentations are stochastic for training diversity
- `augmentation_probability=0.5` - only compute augmented encoding 50% of steps (memory optimization)

### 4. Collapse Monitoring (`collapse_monitor.py`)

Adaptive detection and intervention for posterior collapse during training.

#### Collapse Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| **Active Units (AU)** | Dimensions with variance > 0.01 | 32-64 (50-100%) |
| **AU Ratio** | active_dims / latent_dim | > 0.3 |
| **Mutual Information** | InfoNCE lower bound on I(X; Z) | > 1.0 nats |
| **Mean KL** | Average KL per dimension | > 0.1 |

#### Intervention Levels

| Level | Condition | Intervention |
|-------|-----------|--------------|
| **0 (None)** | AU ratio > 0.3, MI > 1.0 | Gradually relax to defaults |
| **1 (Mild)** | AU ratio < 0.3 | BoW ×1.1, contrastive ×1.1 |
| **2 (Moderate)** | Multiple indicators | BoW ×1.3, contrastive ×1.3, dropout ×0.8 |
| **3 (Severe)** | AU ratio < 0.1 or MI < 0.5 | Max BoW=3.0, max contrastive=0.5, min dropout=0.1 |

```python
# Usage in training loop
collapse_monitor = CollapseMonitor(latent_dim=64)
intervention_scheduler = AdaptiveInterventionScheduler(
    initial_bow_weight=1.5,
    initial_contrastive_weight=0.1,
)

for batch in dataloader:
    metrics = collapse_monitor.update(mu, logvar, mu_augmented)
    if metrics.is_collapsing:
        intervention = intervention_scheduler.intervene(metrics)
        loss_fn.bow_loss_weight = intervention["bow_weight"]
```

### 5. Data Pipeline (`data/`)

```
HuggingFace Datasets
        │
        ▼
┌───────────────┐
│ Format Convert│  formats.py - ShareGPT, Alpaca, Messages, Magicoder
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Filtering   │  filters.py - Length (10-512 instr, 20-1024 resp),
└───────┬───────┘              Language (en, 0.8 conf), Quality checks
        │
        ▼
┌───────────────┐
│ Deduplication │  deduplication.py - MD5 exact + MinHash LSH (0.85 sim)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Tokenization │  tokenization.py - LlamaTokenizerWrapper
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Collation   │  collator.py - VAEBatch with dynamic/fixed padding
└───────────────┘
```

#### Supported Dataset Formats

| Format | Datasets | Structure |
|--------|----------|-----------|
| ShareGPT | FineTome, OpenHermes | `conversations: [{from, value}]` |
| Alpaca | WizardLM | `instruction, input, output` |
| Messages | no_robots | `messages: [{role, content}]` |
| Magicoder | Magicoder-OSS | `problem, solution` |

#### Filtering Pipeline

| Filter | Criteria |
|--------|----------|
| **LengthFilter** | 10 ≤ instruction ≤ 512, 20 ≤ response ≤ 1024 tokens |
| **LanguageFilter** | FastText lid.176.bin, English, confidence ≥ 0.8 |
| **QualityFilter** | No control chars, <10 repeated chars, <5 repeated words, <30% special chars |

#### Deduplication

| Method | Parameters |
|--------|------------|
| **Exact** | MD5 hash of `instruction + "\n" + response` |
| **Near-duplicate** | MinHash LSH, 128 permutations, 5-gram shingles, 0.85 Jaccard threshold |

#### Dataset Phases

| Phase | Datasets | Est. Samples |
|-------|----------|--------------|
| 1 | mlabonne/FineTome-100k | 100,000 |
| 2 | OpenHermes-2.5, WizardLM_evol_instruct | 400,000 |
| 3 | no_robots, Magicoder-OSS-Instruct | 100,000 |

### 6. Training (`train.py`)

#### DDP Configuration

```python
# Environment (before torch import)
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=12,
    log_with="tensorboard",
    dataloader_config=DataLoaderConfiguration(dispatch_batches=False)
)
```

#### Optimizer & Scheduler

| Component | Configuration |
|-----------|---------------|
| **Optimizer** | AdamW(lr=2e-4, weight_decay=0.01) |
| **Scheduler** | CosineAnnealingLR(T_max=total-warmup, eta_min=2e-5) |
| **Warmup** | 10% of total steps |
| **Gradient Clipping** | max_norm=1.0 |

#### Training Configuration

| Parameter | Value | Effective |
|-----------|-------|-----------|
| `per_device_batch_size` | 8 | - |
| `gradient_accumulation_steps` | 12 | - |
| `num_gpus` | 2 | - |
| **Effective Batch Size** | 8 × 12 × 2 | **192** |
| `num_epochs` | 10 | - |
| `save_steps` | 100 | Checkpoint every 100 |
| `logging_steps` | 100 | TensorBoard every 100 |
| `gradient_checkpointing` | True | Memory optimization (incompatible with deep_prefix) |

#### CDP-VAE Training Features

| Feature | Default | Description |
|---------|---------|-------------|
| `enable_collapse_monitoring` | True | Adaptive intervention for posterior collapse |
| `contrastive_weight` | 0.1 | InfoNCE loss weight |
| `augmentation_probability` | 0.5 | Probability of computing augmented encoding |
| `--memory-efficient` | Flag | Disables contrastive & augmentation for lower VRAM |

### 7. Metrics (`metrics.py`)

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| **recon_loss** | Cross-entropy for next-token prediction | 1.0 - 2.5 |
| **kl_loss** | KL divergence after free bits | 0.5 - 2.0 |
| **kl_raw** | Raw KL before free bits (monitoring) | varies |
| **beta** | Current KL annealing weight | 0.0 - 1.0 |
| **active_dims** | Dimensions with KL > free_bits | 32 - 64 |
| **active_unit_ratio** | active_dims / latent_dim | 0.5 - 1.0 |
| **bow_loss** | Bag-of-Words auxiliary loss | 0.1 - 0.5 |
| **BLEU-4** | 4-gram overlap (generation quality) | 0.1 - 0.4 |
| **ROUGE-L** | Longest common subsequence F1 | 0.2 - 0.5 |

#### Active Units Counter
Tracks variance of latent means across batches:
```python
var = E[μ²] - E[μ]²
active = var > 0.01  # Per dimension
```

### 8. Preventing Posterior Collapse

| Technique | Status | Effect |
|-----------|--------|--------|
| Cyclical KL Annealing | ✅ | Gradual KL pressure, 4 cycles |
| Free Bits | ✅ | Min 0.5 nats per dimension |
| Word Dropout | ✅ | 40% token dropout forces latent usage |
| Soft Prompts | ✅ | Strong conditioning signal (32 tokens × 4096 dim) |
| Bag-of-Words Loss | ✅ | λ=1.5, forces semantic content in z |
| Deep Prefix (optional) | ✅ | Injects z into all 32 layers via KV cache |
| **InfoNCE Contrastive** | ✅ | λ=0.1, prevents all inputs mapping to same z |
| **Collapse Monitoring** | ✅ | Adaptive intervention based on AU, MI metrics |
| **Text Augmentation** | ✅ | Creates positive pairs for contrastive learning |

## File Structure

```
soft_prompt_vae/
├── __init__.py              # Package exports
├── config.py                # ModelConfig, TrainingConfig, DataConfig dataclasses
├── model.py                 # LlamaSoftPromptVAE, AttentionPooling, VariationalEncoder,
│                            # SoftPromptProjector, DeepPrefixProjector
├── loss.py                  # SoftPromptVAELoss, CyclicalAnnealingSchedule, FreeBitsKL,
│                            # InfoNCELoss, MomentumQueue, compute_bow_loss
├── metrics.py               # ActiveUnitsCounter, ReconstructionMetrics, LatentStatistics
├── train.py                 # DDP training loop with Accelerate + CDP-VAE features
├── preprocess.py            # Data preprocessing pipeline
├── evaluate_reconstruction.py  # Reconstruction quality evaluation
├── check_latents.py         # Latent space diagnostic tool
├── augmentation.py          # TextAugmenter for CDP-VAE contrastive learning
├── collapse_monitor.py      # CollapseMonitor, AdaptiveInterventionScheduler
├── data/
│   ├── __init__.py          # Data module exports
│   ├── formats.py           # Dataset format converters (ShareGPT, Alpaca, etc.)
│   ├── filters.py           # LengthFilter, LanguageFilter, QualityFilter
│   ├── deduplication.py     # MD5 exact + MinHash LSH deduplication
│   ├── tokenization.py      # LlamaTokenizerWrapper
│   ├── dataset.py           # InstructionDataset, PreprocessedDataset
│   ├── collator.py          # VAEBatch, VAECollator, DynamicPaddingCollator
│   └── loader.py            # DataLoader factory functions
├── configs/                 # Custom configuration files
├── checkpoints/             # Model checkpoints (checkpoint-{step}/)
├── logs/                    # TensorBoard logs
├── processed/               # Preprocessed data cache (.pt, .jsonl)
└── results/                 # Evaluation results
```

## Usage

### Preprocessing
```bash
# Preprocess Phase 1 data
uv run python -m soft_prompt_vae.preprocess --phase 1

# Preprocess with sample limit
uv run python -m soft_prompt_vae.preprocess --phase 1 --max-samples 10000
```

### Training
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 uv run python -m soft_prompt_vae.train --phase 1

# Multi-GPU with DDP (requires NVML fix for L40S)
torchrun --nproc_per_node=2 -m soft_prompt_vae.train --phase 1

# With custom config
uv run python -m soft_prompt_vae.train --phase 1 --config soft_prompt_vae/configs/custom.json
```

### Evaluation
```bash
# Test reconstruction quality
CUDA_VISIBLE_DEVICES=1 uv run python -m soft_prompt_vae.evaluate_reconstruction \
    --checkpoint soft_prompt_vae/checkpoints/checkpoint-1000 \
    --num-samples 10

# Check latent space differentiation
uv run python -m soft_prompt_vae.check_latents \
    --checkpoint soft_prompt_vae/checkpoints/checkpoint-1000
```

### Latent Space Operations
```python
from soft_prompt_vae import LlamaSoftPromptVAE, ModelConfig

model = LlamaSoftPromptVAE(ModelConfig())
model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])

# Encode instruction to latent
z, mu, logvar = model.encode(instruction_ids, attention_mask)

# Generate from latent
generated_ids = model.generate(z, max_length=256, temperature=0.8, top_p=0.9)

# Interpolate between two latents
z_interpolated = model.interpolate(z1, z2, num_steps=10)
```

## VAEOutput Structure

```python
@dataclass
class VAEOutput:
    logits: Tensor          # (batch, seq_len, vocab_size) - decoder output
    loss: Tensor            # Combined loss (if labels provided)
    z: Tensor               # (batch, latent_dim) - sampled latent
    mu: Tensor              # (batch, latent_dim) - latent mean
    logvar: Tensor          # (batch, latent_dim) - latent log variance
    kl_loss: Tensor         # KL divergence component
    recon_loss: Tensor      # Reconstruction component
    bow_logits: Tensor      # (batch, vocab_size) - Bag-of-Words prediction
    mu_augmented: Tensor    # (batch, latent_dim) - augmented mean for InfoNCE (CDP-VAE)
```

## Dimension Summary

| Component | Dimension | Notes |
|-----------|-----------|-------|
| Instruction Input | 512 tokens | Padded/truncated |
| Response Output | 1024 tokens | Padded/truncated |
| Llama Hidden | 4096 | Base model output |
| Attention Pooling | 8 heads × 512 dim | Pool to single vector |
| Latent Space z | 64 | Gaussian prior N(0, I) |
| VAE Hidden Layers | 2048 | Encoder/Projector MLPs |
| Soft Prompts | 32 × 4096 | 32 tokens, 4096 each |
| LoRA Rank | 64 | Applied to 7 modules |
| Vocab Size | ~128,000 | Llama tokenizer |
| Effective Batch | 192 | 8 × 12 × 2 GPUs |
| InfoNCE Temperature | 0.07 | Contrastive similarity scaling |
| Augmentation Prob | 0.5 | Fraction of steps with augmented encoding |

## References

1. **Soft Prompts**: Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)
2. **VAE for Text**: Bowman et al., "Generating Sentences from a Continuous Space" (2016)
3. **Cyclical Annealing**: Fu et al., "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing" (2019)
4. **Free Bits**: Kingma et al., "Improved Variational Inference with Inverse Autoregressive Flow" (2016)
5. **Word Dropout**: Bowman et al., "Generating Sentences from a Continuous Space" (2016)
6. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
7. **RSLoRA**: Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" (2023)
8. **InfoNCE**: Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)
9. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
10. **Collapse Detection**: Dieng et al., "Avoiding Latent Variable Collapse with Generative Skip Models" (2019)
11. **Lagging Inference**: He et al., "Lagging Inference Networks and Posterior Collapse in VAEs" (2019)
12. **MoCo**: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
