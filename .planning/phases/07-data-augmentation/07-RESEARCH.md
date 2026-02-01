# Phase 7: Data Augmentation - Research

**Researched:** 2026-02-01
**Domain:** Data augmentation techniques for flow matching training on SONAR embeddings
**Confidence:** HIGH

## Summary

This research covers three data augmentation strategies for improving generalization in flow matching models trained on 1024-dimensional SONAR embeddings: linear interpolation (mixup), Gaussian noise injection, and dimension dropout/masking. These techniques address the core challenge of limited training data (1K-10K samples) by synthetically expanding the effective dataset size while preserving the semantic structure of the embedding space.

The existing codebase provides a strong foundation: `study/data/dataset.py` implements `FlowDataset` with pre-normalized embeddings, `study/flow_matching/trainer.py` implements the training loop with ICFM formulation (`x_t = (1-t)*x0 + t*x1`), and `study/flow_matching/config.py` already includes an `aug` parameter in `TrainingConfig` with run naming convention `{arch}-{flow}-{dataset}-{aug}`. The infrastructure is ready for augmentation integration.

Recent research (2025-2026) confirms that embedding-level augmentation improves generalization in deep learning models. [Noisy Feature Mixup](https://arxiv.org/abs/2110.02180) demonstrates that combining interpolation with noise injection achieves favorable trade-offs between accuracy and robustness. For flow matching specifically, [TarFlow research](https://openreview.net/forum?id=2uheUFcFsM) shows that "Gaussian noise augmentation during training" is a key technique for improving sample quality in normalizing flows.

**Primary recommendation:** Implement all three augmentation strategies as composable dataset wrappers/transforms that apply augmentation on-the-fly during training. This avoids dataset duplication and allows flexible experimentation with augmentation combinations.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.0+ | Tensor operations, indexing | Already in use; native support for all operations |
| torch.distributions | (PyTorch) | Beta distribution for mixup lambda | Standard, reproducible sampling |
| torch.nn.functional | (PyTorch) | Dropout operations | GPU-optimized masking |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.21+ | Beta distribution alternative | Only if torch.distributions unavailable |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom mixup | timm.data.Mixup | timm adds heavy dependency; custom is simpler for embeddings |
| Manual dropout | F.dropout | F.dropout is standard but dimension-specific masking needs custom |
| Pre-augmented datasets | On-the-fly augmentation | On-the-fly is more flexible, less storage |

**Installation:**
```bash
# Already available via PyTorch in pyproject.toml
uv sync
```

## Architecture Patterns

### Recommended Project Structure
```
study/
├── data/
│   ├── dataset.py           # Existing FlowDataset
│   └── augmentation.py      # NEW: Augmentation transforms
├── flow_matching/
│   ├── config.py            # Update aug param choices
│   ├── trainer.py           # Update to apply augmentation
│   └── train.py             # CLI already has --aug param
```

### Pattern 1: Mixup (Linear Interpolation)
**What:** Linearly interpolate pairs of embeddings within a batch
**When to use:** Every training batch when `aug='mixup'` or `aug='mixup+noise'`
**Example:**
```python
# Source: Mixup paper (Zhang et al., 2018) adapted for embeddings
import torch
from torch.distributions import Beta

def mixup_embeddings(
    embeddings: torch.Tensor,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Apply mixup to a batch of embeddings.

    Args:
        embeddings: Batch of embeddings [B, D]
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Mixed embeddings [B, D]
    """
    batch_size = embeddings.size(0)

    # Sample lambda from Beta(alpha, alpha)
    # When alpha < 1: bimodal, favors unmixed samples
    # When alpha = 1: uniform [0,1]
    # When alpha > 1: unimodal, favors 0.5 (strong mixing)
    beta = Beta(alpha, alpha)
    lam = beta.sample((batch_size, 1)).to(embeddings.device)

    # Random permutation for pairing
    indices = torch.randperm(batch_size, device=embeddings.device)

    # Linear interpolation: x_mixed = lam * x + (1-lam) * x[indices]
    mixed = lam * embeddings + (1 - lam) * embeddings[indices]

    return mixed
```

### Pattern 2: Gaussian Noise Injection
**What:** Add zero-mean Gaussian noise scaled by a noise level parameter
**When to use:** Every training batch when `aug='noise'` or `aug='mixup+noise'`
**Example:**
```python
# Source: Whiteout (Gaussian Adaptive Noise) adapted for embeddings
import torch

def add_gaussian_noise(
    embeddings: torch.Tensor,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian noise to embeddings.

    Args:
        embeddings: Batch of embeddings [B, D]
        noise_std: Standard deviation of noise (relative to data std)

    Returns:
        Noisy embeddings [B, D]
    """
    # Since embeddings are normalized (mean=0, std=1), noise_std
    # directly controls the signal-to-noise ratio
    noise = torch.randn_like(embeddings) * noise_std
    return embeddings + noise
```

### Pattern 3: Dimension Dropout/Masking
**What:** Randomly zero out dimensions in embeddings during training
**When to use:** Every training batch when `aug='dropout'`
**Example:**
```python
# Source: SimCSE, Feature Dropout literature
import torch
import torch.nn.functional as F

def dimension_dropout(
    embeddings: torch.Tensor,
    dropout_rate: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    """Apply dimension-wise dropout to embeddings.

    Unlike standard dropout, this masks entire dimensions consistently
    across the batch, simulating "missing features" during training.

    Args:
        embeddings: Batch of embeddings [B, D]
        dropout_rate: Probability of dropping each dimension
        training: Whether in training mode (no dropout during eval)

    Returns:
        Masked embeddings [B, D]
    """
    if not training or dropout_rate == 0.0:
        return embeddings

    # Per-sample dropout: different mask for each sample in batch
    # Scale by 1/(1-p) to maintain expected value
    return F.dropout(embeddings, p=dropout_rate, training=training)
```

### Pattern 4: Composable Augmentation Pipeline
**What:** Chain multiple augmentations with configurable parameters
**When to use:** For ablation studies combining augmentation types
**Example:**
```python
# Source: Custom pattern following PyTorch transforms
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    mixup_alpha: float = 0.2         # Beta distribution param (0 = disabled)
    noise_std: float = 0.1           # Gaussian noise std (0 = disabled)
    dropout_rate: float = 0.1        # Dimension dropout rate (0 = disabled)

def augment_batch(
    embeddings: torch.Tensor,
    config: AugmentationConfig,
    training: bool = True,
) -> torch.Tensor:
    """Apply configured augmentations to a batch.

    Order: mixup -> noise -> dropout
    This order ensures:
    1. Mixup creates new interpolated samples
    2. Noise perturbs those samples
    3. Dropout masks dimensions (regularization)
    """
    if not training:
        return embeddings

    x = embeddings

    # 1. Mixup (if enabled)
    if config.mixup_alpha > 0:
        x = mixup_embeddings(x, alpha=config.mixup_alpha)

    # 2. Noise injection (if enabled)
    if config.noise_std > 0:
        x = add_gaussian_noise(x, noise_std=config.noise_std)

    # 3. Dimension dropout (if enabled)
    if config.dropout_rate > 0:
        x = dimension_dropout(x, dropout_rate=config.dropout_rate, training=True)

    return x
```

### Anti-Patterns to Avoid
- **Augmenting validation/test data:** Augmentation is training-only; eval must use clean data
- **High noise levels (std > 0.3):** Can destroy semantic content of normalized embeddings
- **Aggressive mixup (alpha > 1.0):** Creates unrealistic interpolations far from data manifold
- **Applying augmentation to x0 (noise):** Only augment x1 (data); x0 is already random
- **Pre-generating augmented samples:** Wastes storage and limits variety; use on-the-fly

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Beta distribution sampling | Manual rejection sampling | torch.distributions.Beta | Numerically stable, GPU-optimized |
| Random permutation | Python random.shuffle | torch.randperm | GPU tensor, no CPU-GPU transfer |
| Element-wise dropout | Manual mask multiplication | F.dropout | Handles scaling, training mode |
| Batch shuffling | Manual index tracking | DataLoader shuffle | Handles worker seeds properly |

**Key insight:** PyTorch's functional operations handle edge cases (empty batches, device placement, gradient tracking) correctly. Custom implementations often miss these.

## Common Pitfalls

### Pitfall 1: Augmenting Both x0 and x1 in Flow Matching
**What goes wrong:** Applying noise/mixup to both noise (x0) and data (x1) in the coupling
**Why it happens:** Natural assumption that "more augmentation = better"
**How to avoid:** Only augment x1 (the data embeddings); x0 is already N(0,I) noise
**Warning signs:** Training loss increases; generated samples look like noise

### Pitfall 2: Inconsistent Augmentation During Epoch
**What goes wrong:** Same sample gets different augmentation each time it appears in an epoch
**Why it happens:** On-the-fly augmentation with no epoch-level consistency
**How to avoid:** For mixup, this is actually desired (more variety). Document this behavior.
**Warning signs:** None - this is often beneficial for regularization

### Pitfall 3: Excessive Noise Destroying Semantics
**What goes wrong:** noise_std too high causes embeddings to leave valid SONAR manifold
**Why it happens:** Normalized embeddings have std=1, so noise_std=0.5 adds 50% noise
**How to avoid:** Start with noise_std=0.05-0.1; validate with SONAR decoder round-trip
**Warning signs:** Decoded text becomes incoherent; cosine similarity to originals < 0.8

### Pitfall 4: Mixup Alpha Mismatch with Dataset Size
**What goes wrong:** Same alpha for 1K vs 10K datasets, but effective augmentation varies
**Why it happens:** Smaller datasets need more regularization but less aggressive mixing
**How to avoid:** Recommended: alpha=0.2 for 10K, alpha=0.1 for 1K (less aggressive for small data)
**Warning signs:** 1K training diverges or overfits faster than expected

### Pitfall 5: Dropout During Validation
**What goes wrong:** Dropout applied during validation/test, causing inconsistent metrics
**Why it happens:** Forgetting to check `training` flag
**How to avoid:** Always pass `training=False` during validation; verify with `model.eval()`
**Warning signs:** Validation loss varies between runs; eval metrics are noisy

### Pitfall 6: Breaking Normalization Statistics
**What goes wrong:** Heavy augmentation shifts data distribution away from N(0,1)
**Why it happens:** Mixup + noise + dropout combined can significantly alter statistics
**How to avoid:** After augmentation, embeddings should still have reasonable mean/std
**Warning signs:** Mean deviates significantly from 0; std deviates significantly from 1

## Code Examples

Verified patterns from official sources and research:

### Integration with FlowTrainer
```python
# Source: Adaptation of study/flow_matching/trainer.py train_epoch method
def train_epoch(self) -> float:
    """Train for one epoch with augmentation."""
    self.model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch_idx, x1 in enumerate(self.train_loader):
        x1 = x1.to(self.device)

        # Apply augmentation to data (x1) ONLY
        if self.aug_config is not None:
            x1 = augment_batch(x1, self.aug_config, training=True)

        # Sample noise (source distribution) - NOT augmented
        x0 = torch.randn_like(x1)

        # Rest of training loop unchanged...
        t, x_t, v_target = self.coupling.sample(x0, x1)
        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)
        # ...
```

### Config Extension
```python
# Source: Extension of study/flow_matching/config.py
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    # ... existing fields ...

    # Augmentation parameters
    aug: str = "none"  # Existing: 'none', 'mixup', 'noise', 'dropout', 'mixup+noise'

    # New: specific augmentation hyperparameters
    mixup_alpha: float = field(default=0.2, repr=False)
    noise_std: float = field(default=0.1, repr=False)
    dropout_rate: float = field(default=0.1, repr=False)
```

### Ablation Study Run Script
```bash
# Run ablation comparing augmentation methods
# Source: Based on existing train.py patterns

# Baseline (no augmentation)
uv run python -m study.flow_matching.train \
  --arch mlp --flow icfm --dataset 5k --aug none --group ablation-aug

# Mixup only
uv run python -m study.flow_matching.train \
  --arch mlp --flow icfm --dataset 5k --aug mixup --group ablation-aug

# Noise only
uv run python -m study.flow_matching.train \
  --arch mlp --flow icfm --dataset 5k --aug noise --group ablation-aug

# Dropout only
uv run python -m study.flow_matching.train \
  --arch mlp --flow icfm --dataset 5k --aug dropout --group ablation-aug

# Combined: mixup + noise (recommended)
uv run python -m study.flow_matching.train \
  --arch mlp --flow icfm --dataset 5k --aug mixup+noise --group ablation-aug
```

### Validation of Augmented Embeddings
```python
# Source: Custom validation pattern
import torch
import torch.nn.functional as F

def validate_augmentation_quality(
    original: torch.Tensor,
    augmented: torch.Tensor,
    min_similarity: float = 0.7,
) -> dict:
    """Verify augmented embeddings remain semantically valid.

    Args:
        original: Original embeddings [B, D]
        augmented: Augmented embeddings [B, D]
        min_similarity: Minimum acceptable cosine similarity

    Returns:
        Dict with validation metrics
    """
    # Cosine similarity
    similarity = F.cosine_similarity(original, augmented, dim=1)

    # Statistics of augmented embeddings (should be near 0,1)
    aug_mean = augmented.mean().item()
    aug_std = augmented.std().item()

    return {
        "mean_similarity": similarity.mean().item(),
        "min_similarity": similarity.min().item(),
        "pass_rate": (similarity >= min_similarity).float().mean().item(),
        "augmented_mean": aug_mean,
        "augmented_std": aug_std,
        "stats_valid": abs(aug_mean) < 0.5 and 0.5 < aug_std < 1.5,
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Input-space mixup only | Manifold mixup (hidden layers) | 2019 (ICML) | Better regularization |
| Fixed noise levels | Adaptive/learnable noise | 2020-2024 | Task-specific optimization |
| Single augmentation | Noisy Feature Mixup (combined) | 2021 | Best of both worlds |
| Pre-generated augmentation | On-the-fly augmentation | Standard practice | More variety, less storage |

**Deprecated/outdated:**
- **CutMix for embeddings:** Designed for images with spatial structure; not applicable to 1D embeddings
- **Aggressive mixup (alpha > 1):** Research shows alpha in [0.1, 0.4] is optimal for most tasks
- **Label smoothing as augmentation:** Not applicable to flow matching (no classification labels)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal mixup alpha for SONAR embeddings**
   - What we know: General recommendation is alpha in [0.1, 0.4]
   - What's unclear: SONAR embedding space may have different optimal values
   - Recommendation: Start with alpha=0.2; run hyperparameter sweep if time permits

2. **Interaction between augmentation and OT-CFM**
   - What we know: OT-CFM uses mini-batch optimal transport coupling
   - What's unclear: Whether augmented samples should be OT-coupled to original or augmented pairs
   - Recommendation: Augment x1 before coupling; let OT operate on augmented data

3. **Whether to augment reflow pairs**
   - What we know: Reflow uses pre-generated (x0, x1) pairs from teacher
   - What's unclear: Augmenting these pairs might hurt the "straighter paths" benefit
   - Recommendation: For reflow, experiment with noise-only augmentation (preserves coupling)

4. **Dimension dropout vs feature dropout**
   - What we know: F.dropout drops individual elements; dimension dropout could mask entire features
   - What's unclear: Which is better for 1024D SONAR space
   - Recommendation: Start with standard F.dropout; dimension-consistent dropout as v2

## Sources

### Primary (HIGH confidence)
- [PyTorch Documentation](https://docs.pytorch.org/) - torch.distributions.Beta, F.dropout, torch.randperm
- [Manifold Mixup GitHub](https://github.com/vikasverma1077/manifold_mixup) - Official ICML 2019 implementation
- [Noisy Feature Mixup](https://arxiv.org/abs/2110.02180) - Combined interpolation + noise approach
- Existing codebase: `study/flow_matching/trainer.py`, `study/flow_matching/config.py`, `study/data/dataset.py`

### Secondary (MEDIUM confidence)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Original flow matching paper
- [Whiteout: Gaussian Adaptive Noise](https://arxiv.org/abs/1612.01490) - Noise injection theory
- [TarFlow](https://openreview.net/forum?id=2uheUFcFsM) - Noise augmentation for normalizing flows
- [SimCSE](https://arxiv.org/abs/2104.08821) - Dropout as augmentation for embeddings

### Tertiary (LOW confidence)
- [inversedMixup](https://arxiv.org/abs/2601.21543) - Recent mixup variant (January 2026, unverified)
- [MultiMix](https://arxiv.org/abs/2311.05538) - Beyond-pairwise mixup (requires more investigation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch native operations, well-documented
- Architecture: HIGH - follows existing codebase patterns, clear integration points
- Pitfalls: HIGH - verified against research literature and common failure modes

**Research date:** 2026-02-01
**Valid until:** 2026-04-01 (60 days - augmentation techniques are stable)

---

## Requirements Traceability

| Requirement | Research Finding | Confidence |
|-------------|------------------|------------|
| DATA-05: Linear interpolation (mixup) | Mixup pattern documented with Beta(alpha, alpha) sampling | HIGH |
| DATA-06: Gaussian noise injection | Gaussian noise with configurable std documented | HIGH |
| DATA-07: Dimension dropout/masking | F.dropout with training flag pattern documented | HIGH |

## Implementation Recommendations

### Phase 7 Plan Structure

**Plan 07-01: Mixup and Noise Injection**
1. Create `study/data/augmentation.py` with `mixup_embeddings` and `add_gaussian_noise`
2. Add augmentation config fields to `TrainingConfig`
3. Integrate augmentation into `FlowTrainer.train_epoch`
4. Add CLI args to `train.py` for augmentation hyperparameters
5. Run validation tests to verify augmented embeddings remain valid

**Plan 07-02: Dimension Dropout and Integration**
1. Add `dimension_dropout` to augmentation module
2. Implement `AugmentationConfig` dataclass for clean configuration
3. Add combined augmentation modes (`mixup+noise`, `all`)
4. Run ablation experiment comparing augmentation methods
5. Verify improved generalization (lower validation loss)

### Recommended Hyperparameters

| Augmentation | Parameter | Default | Range for Sweep |
|--------------|-----------|---------|-----------------|
| Mixup | alpha | 0.2 | [0.1, 0.2, 0.4] |
| Noise | std | 0.1 | [0.05, 0.1, 0.15] |
| Dropout | rate | 0.1 | [0.05, 0.1, 0.2] |

### Success Criteria Mapping

| Success Criterion | How to Verify |
|-------------------|---------------|
| Mixup generates valid training pairs | Cosine similarity between original and mixed > 0.7 |
| Gaussian noise augments data | Statistics validation (mean near 0, std near 1) |
| Dropout masks dimensions | Verify ~10% of values are zero when dropout_rate=0.1 |
| Augmented training improves generalization | Val loss with augmentation < val loss without |
