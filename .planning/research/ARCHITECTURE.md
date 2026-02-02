# Architecture Research: Velocity Networks for Flow Matching

**Domain:** Flow Matching Generative Models
**Context:** 1024D SONAR embeddings, 1K-10K training samples
**Researched:** 2026-01-31
**Overall Confidence:** MEDIUM-HIGH

---

## Executive Summary

This research surveys velocity network architectures for flow matching in 1024D embedding spaces with limited training data (1K-10K samples). The key finding is that **architecture complexity must scale with dataset size** to avoid overfitting. For small datasets, simpler architectures (MLPs, shallow DiT) outperform deep transformers. The existing DiT-style baseline in `rielbo/velocity_network.py` is well-designed but may need scaling down for <5K samples.

---

## Velocity Network Architectures

### 1. DiT-style (Diffusion Transformers with AdaLN)

**Description:**
DiT replaces U-Net backbones with transformers, using Adaptive Layer Normalization (AdaLN) for time/condition injection. The key innovation is AdaLN-Zero: initializing modulation parameters to zero so each block starts as an identity function.

**Architecture Components:**
```
Input: x [B, 1024] + t [B]
  |
  v
Time Embedding: sinusoidal(t) -> MLP -> c [B, hidden_dim]
  |
  v
Input Projection: Linear(1024 -> hidden_dim)
  |
  v
N x AdaLN Blocks:
  - LayerNorm (no affine) + AdaLN modulation (shift, scale, gate)
  - Multi-head Self-Attention
  - LayerNorm (no affine) + AdaLN modulation
  - MLP (hidden_dim -> 4*hidden_dim -> hidden_dim)
  |
  v
Final AdaLN + Linear(hidden_dim -> 1024)
  |
  v
Output: velocity [B, 1024]
```

**Parameter Count Formula:**
```
Per AdaLN Block:
  - Attention: 4 * hidden_dim^2 (Q, K, V, O projections)
  - MLP: 2 * hidden_dim * (4 * hidden_dim) = 8 * hidden_dim^2
  - AdaLN modulation: 2 * hidden_dim * (6 * hidden_dim) = 12 * hidden_dim^2
  Total per block: ~24 * hidden_dim^2

Full Model:
  - Time embed MLP: 2 * time_embed_dim * hidden_dim
  - Input projection: 1024 * hidden_dim
  - N blocks: N * 24 * hidden_dim^2
  - Output projection: hidden_dim * 1024
```

**Existing Implementation Configurations (from rielbo/):**

| Config | hidden_dim | layers | heads | Parameters | Use Case |
|--------|------------|--------|-------|------------|----------|
| DiT-Tiny | 256 | 4 | 4 | ~1.6M | <1K samples |
| DiT-Small | 384 | 6 | 6 | ~5.3M | 1K-5K samples |
| DiT-Base | 512 | 6 | 8 | ~9.4M | 5K-10K samples (current default) |
| DiT-Large | 768 | 8 | 12 | ~28M | >50K samples |

**Overfitting Risk:**
- HIGH for 1K samples with DiT-Base (9.4M params >> 1K * 1024 features)
- MEDIUM for 5K samples with DiT-Small
- LOW for 10K samples with DiT-Base with regularization

**Advantages:**
- Excellent for learning complex, non-linear velocity fields
- AdaLN-Zero provides stable training from scratch
- Modular time conditioning

**Disadvantages:**
- Computationally expensive (quadratic attention)
- Prone to overfitting on small datasets
- Sequence length of 1 (for 1024D vector) underutilizes attention

**Sources:**
- [Scalable Diffusion Models with Transformers (DiT)](https://www.wpeebles.com/DiT.html)
- [Diffusion Transformer Explained](https://towardsdatascience.com/diffusion-transformer-explained-e603c4770f7e/)
- [Facebook DiT GitHub](https://github.com/facebookresearch/DiT)

---

### 2. U-Net Style MLP with Skip Connections

**Description:**
An MLP with encoder-decoder structure and skip connections, inspired by U-Net. Features down-projection (bottleneck) and up-projection with residual connections at matching resolutions.

**Architecture Components:**
```
Input: x [B, 1024] + t [B]
  |
  v
Time Embedding: sinusoidal(t) -> MLP -> t_emb [B, time_dim]
  |
  v
Encoder Path (contracting):
  h0 = Linear(1024 -> 512) + FiLM(t_emb) + activation
  h1 = Linear(512 -> 256) + FiLM(t_emb) + activation
  h2 = Linear(256 -> 128) + FiLM(t_emb) + activation  (bottleneck)
  |
  v
Decoder Path (expanding) with skip connections:
  d2 = Linear(128 -> 256) + FiLM(t_emb) + activation
  d1 = Linear(256 + 256 -> 512) + FiLM(t_emb) + activation  [skip from h1]
  d0 = Linear(512 + 512 -> 1024) + FiLM(t_emb) + activation [skip from h0]
  |
  v
Output: Linear(1024 -> 1024)
```

**Parameter Count Estimate:**
```
Encoder: 1024*512 + 512*256 + 256*128 = 688K
Decoder: 128*256 + 512*512 + 1024*1024 = 1.3M
Skip projections: ~200K
FiLM layers: ~100K
Time embed: ~50K
Total: ~2.3M parameters
```

**Configurations:**

| Config | Bottleneck | Depth | Parameters | Use Case |
|--------|------------|-------|------------|----------|
| UNet-MLP-Tiny | 64 | 3 | ~0.5M | <1K samples |
| UNet-MLP-Small | 128 | 4 | ~1.5M | 1K-5K samples |
| UNet-MLP-Base | 256 | 5 | ~3.5M | 5K-10K samples |

**Overfitting Risk:**
- LOW-MEDIUM for most configurations due to bottleneck regularization
- Skip connections preserve gradient flow, enabling training on limited data
- FiLM conditioning particularly effective in low-data regimes

**Advantages:**
- Bottleneck acts as implicit regularization
- Skip connections preserve spatial information and gradient flow
- FiLM conditioning works well with small datasets
- Lower computational cost than transformers

**Disadvantages:**
- Less expressive than transformers for complex velocity fields
- Fixed architecture (less flexible than attention)
- Manual design of skip connection points

**Sources:**
- [1D U-Net with FiLM Conditioning](https://www.emergentmind.com/topics/1d-u-net-architecture-with-feature-wise-linear-modulation)
- [UNet++ Skip Connections](https://pmc.ncbi.nlm.nih.gov/articles/PMC7357299/)

---

### 3. Simple MLP (Ablation Baseline)

**Description:**
A straightforward feedforward network with time concatenation. Essential baseline for ablation studies to quantify the benefit of architectural complexity.

**Architecture Components:**
```
Input: concat(x, t_emb) [B, 1024 + time_dim]
  |
  v
N x (Linear -> LayerNorm -> Activation)
  |
  v
Output: Linear -> velocity [B, 1024]
```

**Parameter Count Formula:**
```
Input layer: (1024 + time_dim) * hidden_dim
Hidden layers: (N-1) * hidden_dim^2
Output layer: hidden_dim * 1024
Total: ~N * hidden_dim^2 + 2 * 1024 * hidden_dim
```

**Configurations:**

| Config | hidden_dim | layers | Parameters | Use Case |
|--------|------------|--------|------------|----------|
| MLP-Tiny | 256 | 3 | ~0.4M | Baseline, <500 samples |
| MLP-Small | 512 | 4 | ~1.3M | Baseline, 1K-2K samples |
| MLP-Base | 512 | 6 | ~1.8M | Baseline, 5K-10K samples |
| MLP-Wide | 1024 | 4 | ~5.2M | Capacity testing |

**Overfitting Risk:**
- LOW for small configs due to limited capacity
- MEDIUM for MLP-Wide (5.2M params can overfit 1K samples)

**Advantages:**
- Fast training and inference
- Easy to implement and debug
- Good baseline for measuring architectural improvements
- Works well for simple, smooth velocity fields

**Disadvantages:**
- Limited expressiveness
- Struggles with complex, multi-modal velocity fields
- No explicit structure for handling time conditioning

**Sources:**
- [TorchCFM Examples](https://github.com/atong01/conditional-flow-matching)
- [Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t)

---

### 4. Mamba/SSM Architectures

**Description:**
State Space Models (Mamba) adapted for generative modeling. Originally designed for sequences, these require adaptation for 1D embedding inputs.

**Architecture Components (Diffusion Mamba / DiM):**
```
Input: x [B, 1024] -> patchify -> [B, num_patches, patch_dim]
  |
  v
Time Embedding: sinusoidal(t) -> MLP -> t_emb
  |
  v
N x DiM Blocks:
  - Bidirectional SSM (forward + backward scan)
  - Local feature enhancement (MLP)
  - Time conditioning via AdaLN
  |
  v
Unpatchify + Output projection
  |
  v
Output: velocity [B, 1024]
```

**Parameter Count Estimate:**
For 1024D vectors treated as sequence of 16 patches of 64D:
```
Per DiM Block:
  - SSM (bidirectional): ~4 * hidden_dim * state_dim * 2 = 8 * hidden_dim * state_dim
  - MLP: 8 * hidden_dim^2
  - AdaLN: 12 * hidden_dim^2

With hidden_dim=256, state_dim=16, 6 blocks:
  Total: ~2.5M parameters
```

**Configurations:**

| Config | hidden_dim | state_dim | layers | Parameters |
|--------|------------|-----------|--------|------------|
| DiM-Tiny | 192 | 16 | 4 | ~1.2M |
| DiM-Small | 256 | 16 | 6 | ~2.5M |
| DiM-Base | 384 | 32 | 8 | ~6.0M |

**Overfitting Risk:**
- MEDIUM - similar parameter efficiency to U-Net MLP
- Bidirectional scanning adds regularization through structured computation

**Advantages:**
- Linear complexity O(n) vs quadratic O(n^2) for attention
- Efficient long-range dependency modeling
- Memory efficient for larger embedding dimensions

**Disadvantages:**
- Designed for sequences, awkward fit for 1D vectors
- Less mature ecosystem than transformers
- Complex implementation (custom CUDA kernels)
- Limited empirical evidence for flow matching specifically

**Suitability for 1024D Embeddings:**
- EXPERIMENTAL - Mamba's advantages shine for long sequences
- For single 1024D vectors, transformer/MLP may be more straightforward
- Consider if patchification improves learning (treating 1024D as 16x64 or 32x32)

**Sources:**
- [Mamba SSM Architecture](https://github.com/state-spaces/mamba)
- [DiM: Diffusion Mamba](https://arxiv.org/abs/2405.14224)
- [DiMSUM: Diffusion Mamba Unified](https://proceedings.neurips.cc/paper_files/paper/2024/file/39bc6e3cbf5a1991d33dc10ebff9a9cf-Paper-Conference.pdf)

---

## Time Conditioning Methods

### Comparison Matrix

| Method | Mechanism | Parameters | Best For | Complexity |
|--------|-----------|------------|----------|------------|
| Sinusoidal + Concat | Fixed Fourier features, concat to input | ~0 | Simple MLPs | Low |
| Sinusoidal + MLP | Fourier features -> learned MLP | dim * hidden | All architectures | Low |
| FiLM | Scale and shift activations | 2 * hidden per layer | U-Net MLPs, small data | Medium |
| AdaLN | Learned LayerNorm params | 2-6 * hidden per layer | Transformers, DiT | Medium |
| AdaLN-Zero | AdaLN with zero init | Same as AdaLN | Best for stable training | Medium |
| Learned Embedding | Learned lookup table | num_steps * dim | Discrete timesteps | Low |

### Detailed Analysis

**1. Sinusoidal Embeddings (Positional Encoding Style)**
```python
def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    freqs = exp(-log(10000) * arange(half_dim) / half_dim)
    args = t * freqs
    return concat([cos(args), sin(args)])
```
- Multi-scale representation of continuous time
- No learnable parameters (fixed basis)
- Vulnerable to cancellation by subsequent normalization layers

**2. FiLM (Feature-wise Linear Modulation)**
```python
gamma, beta = MLP(t_emb).chunk(2)  # [B, hidden_dim] each
h = gamma * h + beta  # element-wise modulation
```
- Proven effective in low-data regimes (ACDC dataset: 0.39 -> 0.55 Dice at 6% data)
- Lightweight: only 2 * hidden_dim parameters per layer
- Works well with U-Net style architectures

**3. AdaLN (Adaptive Layer Normalization)**
```python
gamma, beta = MLP(t_emb).chunk(2)
h_norm = (h - mean(h)) / std(h)  # no affine params
h = gamma * h_norm + beta
```
- More expressive than FiLM (operates on normalized features)
- Standard in DiT architectures

**4. AdaLN-Zero**
```python
# Same as AdaLN but initialize MLP output to zeros
nn.init.zeros_(adaLN_modulation.weight)
nn.init.zeros_(adaLN_modulation.bias)
```
- Critical for stable training from scratch
- Each block starts as identity function
- Allows deeper networks without careful initialization

**Recommendation by Architecture:**

| Architecture | Recommended Method | Reason |
|--------------|-------------------|--------|
| Simple MLP | Sinusoidal + Concat | Simplicity, adequate for shallow networks |
| U-Net MLP | FiLM | Effective in low-data, natural fit for encoder-decoder |
| DiT | AdaLN-Zero | Proven best for transformers, stable training |
| Mamba/DiM | AdaLN-Zero | Follows DiT patterns |

**Sources:**
- [Conditioning in Diffusion Models](https://medium.com/@spoorthisetty99/conditioning-in-diffusion-models-a-deep-dive-into-ldms-dits-and-vanilla-u-nets-610262d3fe71)
- [The Disappearance of Timestep Embedding](https://arxiv.org/html/2405.14126v1)
- [DiT Conditioning Mechanisms](https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/dit-conditioning)

---

## Normalization Strategies

### Comparison

| Norm | Formula | Per-Sample | Parameters | Use Case |
|------|---------|------------|------------|----------|
| LayerNorm | (x - mean) / std * gamma + beta | Yes | 2 * dim | Traditional transformers |
| RMSNorm | x / rms(x) * gamma | Yes | dim | Modern LLMs (LLaMA, Mistral) |
| GroupNorm | per-group normalization | Yes | 2 * dim | Vision, small batches |
| BatchNorm | across batch | No | 2 * dim | Large batch vision |

### Recommendations for Flow Matching

**1. RMSNorm (Recommended for new implementations)**
- 7-64% speedup over LayerNorm (hardware/framework dependent)
- Equivalent quality for most tasks
- Used by LLaMA, Mistral, Qwen, DeepSeek

**2. LayerNorm (Current standard, proven)**
- Well-understood behavior
- Default in DiT implementations
- Use with elementwise_affine=False when combined with AdaLN

**3. No Normalization (Emerging alternative)**
- Dynamic Tanh (DyT) matches RMSNorm quality
- Removes batch/normalization dependencies
- Experimental, less empirical validation

### Special Considerations for AdaLN

When using AdaLN, the base normalization should have `elementwise_affine=False`:
```python
self.norm = nn.LayerNorm(dim, elementwise_affine=False)
# gamma, beta provided by AdaLN modulation instead
```

**Sources:**
- [LayerNorm and RMSNorm in Transformers](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/)
- [Why Modern Transformers Use RMSNorm](https://medium.com/@ashutoshs81127/why-modern-transformers-use-rmsnorm-instead-of-layernorm-5f386be7156c)

---

## Regularization Strategies for Small Datasets

### Critical for 1K-10K Sample Regimes

| Technique | Effect | When to Use | Implementation |
|-----------|--------|-------------|----------------|
| Weight Decay (L2) | Penalizes large weights | Always | `weight_decay=0.01` in optimizer |
| Dropout | Random neuron masking | MLP layers, NOT attention | `nn.Dropout(0.1-0.3)` |
| EMA | Smooth parameter averaging | Always for diffusion/flow | `decay=0.9999` |
| Early Stopping | Halt at validation minimum | Always | Monitor val loss |
| Spectral Norm | Bounds Lipschitz constant | GANs, can help flow models | `spectral_norm(layer)` |
| Gradient Clipping | Prevents exploding gradients | Always | `clip_grad_norm_(1.0)` |

### EMA Best Practices

**Decay Rate Selection:**
- Large datasets (>100K): `decay=0.9999`
- Medium datasets (10K-100K): `decay=0.999`
- Small datasets (1K-10K): `decay=0.99-0.999`

**Warmup Strategy:**
```python
effective_decay = min(decay, (step + 1) / (step + 10))
```

**Post-hoc EMA (EDM2):**
- Store periodic snapshots with shorter EMA
- Reconstruct optimal EMA profile after training
- Improved ImageNet-512 FID: 2.41 -> 1.81

### Dropout Considerations

**For small datasets (<5K samples):**
- Use conservative dropout rates: 0.1-0.2
- Apply to MLP layers only, NOT attention
- Too much dropout can hurt when data is scarce

**For larger datasets:**
- Standard dropout 0.1-0.5
- Can apply to both MLP and attention outputs

### Architecture-Specific Recommendations

| Data Size | Architecture | Regularization |
|-----------|--------------|----------------|
| <1K | MLP-Tiny or DiT-Tiny | weight_decay=0.05, EMA=0.99, no dropout |
| 1K-5K | U-Net MLP or DiT-Small | weight_decay=0.01, EMA=0.999, dropout=0.1 |
| 5K-10K | DiT-Base | weight_decay=0.01, EMA=0.9999, dropout=0.1-0.2 |
| >10K | DiT-Base/Large | weight_decay=0.01, EMA=0.9999, dropout=0.1 |

**Sources:**
- [EMA in Diffusion Models](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/)
- [Exponential Moving Average Dynamics](https://arxiv.org/html/2411.18704v1)
- [How We Trained Stable Diffusion](https://www.databricks.com/blog/diffusion)
- [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)

---

## Recommended Configurations

### For 1K Samples (HIGH overfitting risk)

**Primary: Simple MLP with FiLM**
```python
MLP-Small:
  input_dim: 1024
  hidden_dims: [512, 256, 256, 512]  # bottleneck structure
  time_dim: 128
  conditioning: FiLM
  activation: SiLU
  dropout: 0.0  # too little data for dropout
  weight_decay: 0.05

Parameters: ~1.0M
```

**Secondary: DiT-Tiny**
```python
DiT-Tiny:
  input_dim: 1024
  hidden_dim: 256
  num_layers: 4
  num_heads: 4
  time_embed_dim: 128

Parameters: ~1.6M
Regularization: weight_decay=0.05, EMA=0.99
```

### For 5K Samples (MEDIUM overfitting risk)

**Primary: U-Net MLP**
```python
UNet-MLP-Small:
  input_dim: 1024
  encoder_dims: [768, 512, 256, 128]
  decoder_dims: [256, 512, 768, 1024]
  skip_connections: [0->3, 1->2]  # encoder->decoder
  time_dim: 256
  conditioning: FiLM

Parameters: ~2.5M
```

**Secondary: DiT-Small**
```python
DiT-Small:
  input_dim: 1024
  hidden_dim: 384
  num_layers: 6
  num_heads: 6
  dropout: 0.1

Parameters: ~5.3M
Regularization: weight_decay=0.01, EMA=0.999
```

### For 10K Samples (LOW-MEDIUM overfitting risk)

**Primary: DiT-Base (current rielbo/ default)**
```python
DiT-Base:
  input_dim: 1024
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  time_embed_dim: 256

Parameters: ~9.4M
```

**Experimental: DiM-Small (Mamba)**
```python
DiM-Small:
  input_dim: 1024
  patch_size: 64  # 16 patches
  hidden_dim: 256
  state_dim: 16
  num_layers: 6

Parameters: ~2.5M
```

---

## Suggested Build Order

Based on complexity and data requirements:

### Phase 1: Baselines (Essential)
1. **Simple MLP** - Establishes lower bound, fastest to implement
2. **Current DiT** - Already implemented, establishes current performance

### Phase 2: Architecture Variants (Core Study)
3. **U-Net MLP with FiLM** - Expected best for small data
4. **DiT-Tiny/Small variants** - Scaling study within DiT family
5. **Ablation: time conditioning methods** - Compare FiLM vs AdaLN

### Phase 3: Advanced Architectures (Experimental)
6. **Mamba/DiM adaptation** - Novel, may not outperform DiT
7. **Hybrid approaches** - Combine best elements

### Phase 4: Regularization Study (Cross-cutting)
8. **EMA decay sweeps** - Essential for all architectures
9. **Dropout/weight decay ablations** - Per-architecture tuning

---

## Overfitting Analysis Summary

| Architecture | 1K Samples | 5K Samples | 10K Samples |
|--------------|------------|------------|-------------|
| MLP-Tiny (~0.4M) | LOW | LOW | LOW |
| MLP-Base (~1.8M) | MEDIUM | LOW | LOW |
| U-Net MLP (~2.5M) | LOW-MEDIUM | LOW | LOW |
| DiT-Tiny (~1.6M) | MEDIUM | LOW | LOW |
| DiT-Small (~5.3M) | HIGH | MEDIUM | LOW |
| DiT-Base (~9.4M) | CRITICAL | HIGH | MEDIUM |
| DiM-Small (~2.5M) | MEDIUM | LOW | LOW |

**Rule of Thumb:** Parameters should be < 10x the number of training samples * input dimension for safe training without heavy regularization.

- 1K samples * 1024D = 1M effective data points -> ~1-2M parameters
- 5K samples * 1024D = 5M effective data points -> ~5-10M parameters
- 10K samples * 1024D = 10M effective data points -> ~10-20M parameters

---

## Key Takeaways

1. **Scale architecture with data:** DiT-Base is likely too large for <5K samples
2. **AdaLN-Zero is essential** for transformer-based velocity networks
3. **FiLM conditioning** shows particular benefit for small datasets
4. **U-Net MLP structure** provides implicit regularization via bottleneck
5. **Mamba/SSM is experimental** for 1D embedding vectors (designed for sequences)
6. **RMSNorm** is the modern choice, equivalent quality with better efficiency
7. **EMA is critical** with decay tuned to dataset size (0.99 for small, 0.9999 for large)
8. **Skip connections** help in both gradient flow and preserving input information

---

## Sources Index

### Architecture
- [DiT Official Paper](https://www.wpeebles.com/DiT.html) - HIGH confidence
- [DiT GitHub](https://github.com/facebookresearch/DiT) - HIGH confidence
- [Mamba SSM](https://github.com/state-spaces/mamba) - HIGH confidence
- [DiM: Diffusion Mamba](https://arxiv.org/abs/2405.14224) - MEDIUM confidence

### Time Conditioning
- [Conditioning in Diffusion Models](https://medium.com/@spoorthisetty99/conditioning-in-diffusion-models-a-deep-dive-into-ldms-dits-and-vanilla-u-nets-610262d3fe71) - MEDIUM confidence
- [Timestep Embedding Analysis](https://arxiv.org/html/2405.14126v1) - HIGH confidence

### Normalization
- [RMSNorm Paper](https://github.com/bzhangGo/rmsnorm) - HIGH confidence
- [RMSNorm vs LayerNorm](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/) - MEDIUM confidence

### Training Stability
- [EMA in Diffusion Models](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/) - HIGH confidence
- [EDM2 Post-hoc EMA](https://arxiv.org/abs/2312.02696) - HIGH confidence

### Flow Matching
- [Flow Matching Original Paper](https://arxiv.org/abs/2210.02747) - HIGH confidence
- [TorchCFM](https://github.com/atong01/conditional-flow-matching) - HIGH confidence
- [Cambridge MLG Introduction](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) - MEDIUM confidence
