# Phase 1: Data Pipeline - Research

**Researched:** 2026-01-31
**Domain:** SONAR embeddings data pipeline for flow matching training
**Confidence:** HIGH

## Summary

This research covers the technical implementation details for establishing a data pipeline for flow matching training on SONAR embeddings. The phase involves generating a 10K verbosed sampling (VS) dataset with SONAR embeddings, creating nested train/val/test splits for 1K/5K/10K sizes, implementing per-dimension normalization with stored statistics, and verifying SONAR decoder round-trip fidelity.

The existing codebase provides a strong foundation: `datasets/gsm8k_instructions_vs.pt` contains 4070 VS instructions with SONAR embeddings (1024D) as a reference format, and `rielbo/decoder.py` implements the `SonarDecoder` class correctly. The VS pipeline config shows the generation approach (LLM-based instruction generation with SONAR encoding). The key technical decisions are locked: nested splits (1K subset of 5K subset of 10K), per-dimension normalization from 10K training set only, and cosine similarity >= 0.9 as the round-trip verification threshold.

**Primary recommendation:** Follow the existing `gsm8k_instructions_vs.pt` format (dict with `embeddings`, `instructions`, `sources`, `config`, `stats` keys), implement normalization as z-score transform with per-dimension mean/std, and verify round-trip fidelity using `torch.nn.functional.cosine_similarity`.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.0+ | Tensor operations, data loading | Core framework already in use |
| sonar-space | 0.5.0+ | SONAR encoding/decoding | Official Meta library for SONAR |
| fairseq2 | 0.5.2+ | Backend for SONAR pipelines | Required dependency for sonar-space |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.utils.data | (PyTorch) | Dataset/DataLoader | Data loading pipeline |
| tqdm | 4.65+ | Progress bars | Long embedding generation |
| numpy | 1.21+ | Random seed management | Reproducible splits |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual split logic | sklearn.model_selection | sklearn adds dependency for simple operation |
| torch.save/load | HDF5/Parquet | torch.save is simpler for single-machine workflows |

**Installation:**
```bash
# Already in pyproject.toml
uv sync
```

## Architecture Patterns

### Recommended Project Structure
```
study/
├── data/
│   ├── generate_vs_dataset.py     # VS dataset generation script
│   ├── create_splits.py           # Split creation script
│   ├── normalize.py               # Normalization utilities
│   └── verify_decoder.py          # SONAR round-trip verification
├── datasets/
│   ├── vs_10k.pt                  # Full 10K dataset
│   ├── splits/
│   │   ├── 1k/
│   │   │   ├── train.pt           # 800 samples
│   │   │   ├── val.pt             # 100 samples
│   │   │   └── test.pt            # 100 samples
│   │   ├── 5k/
│   │   │   ├── train.pt           # 4000 samples
│   │   │   ├── val.pt             # 500 samples
│   │   │   └── test.pt            # 500 samples
│   │   └── 10k/
│   │       ├── train.pt           # 8000 samples
│   │       ├── val.pt             # 1000 samples
│   │       └── test.pt            # 1000 samples
│   └── normalization_stats.pt     # mean/std tensors
```

### Pattern 1: SONAR Encoding Pipeline
**What:** Batch encoding of text prompts to 1024D SONAR embeddings
**When to use:** Generating new dataset from text prompts
**Example:**
```python
# Source: https://github.com/facebookresearch/SONAR
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Initialize encoder with GPU and half precision for speed
encoder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=torch.device("cuda:0"),
    dtype=torch.float16,  # fp16 for faster encoding
)

# Batch encode texts
texts = ["instruction 1", "instruction 2", "instruction 3"]
embeddings = encoder.predict(texts, source_lang="eng_Latn")
# Returns: torch.Size([3, 1024]), dtype=torch.float32

# Convert to float32 for storage
embeddings = embeddings.float()
```

### Pattern 2: Per-Dimension Normalization
**What:** Z-score normalization computed per embedding dimension
**When to use:** Preparing data for flow model training
**Example:**
```python
# Source: PyTorch documentation
import torch

def compute_normalization_stats(embeddings: torch.Tensor) -> dict:
    """Compute mean/std per dimension from training data only."""
    # embeddings: [N, 1024]
    mean = embeddings.mean(dim=0)  # [1024]
    std = embeddings.std(dim=0) + 1e-8  # [1024], epsilon for stability
    return {"mean": mean, "std": std}

def normalize(embeddings: torch.Tensor, stats: dict) -> torch.Tensor:
    """Apply z-score normalization."""
    return (embeddings - stats["mean"]) / stats["std"]

def denormalize(embeddings: torch.Tensor, stats: dict) -> torch.Tensor:
    """Reverse normalization before SONAR decoder."""
    return embeddings * stats["std"] + stats["mean"]
```

### Pattern 3: Reproducible Nested Splits
**What:** Creating 1K/5K/10K splits where smaller sets are subsets of larger
**When to use:** Ensuring ablation studies on dataset size are fair comparisons
**Example:**
```python
# Source: PyTorch documentation
import torch
import numpy as np

def create_nested_splits(n_samples: int, seed: int = 42) -> dict:
    """Create nested train/val/test splits for 1K/5K/10K."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Shuffle indices once
    indices = np.random.permutation(n_samples)

    splits = {}
    for size in [1000, 5000, 10000]:
        if size > n_samples:
            raise ValueError(f"Dataset has {n_samples} samples, need {size}")

        # Take first `size` indices from shuffled array
        subset_indices = indices[:size]

        # Split 80/10/10
        n_train = int(size * 0.8)
        n_val = int(size * 0.1)

        splits[f"{size//1000}k"] = {
            "train": subset_indices[:n_train].tolist(),
            "val": subset_indices[n_train:n_train + n_val].tolist(),
            "test": subset_indices[n_train + n_val:size].tolist(),
        }

    return splits
```

### Pattern 4: Round-Trip Verification
**What:** Verify SONAR decoder quality via encode-decode-encode similarity
**When to use:** Validating pipeline correctness before training
**Example:**
```python
# Source: PyTorch documentation, SONAR GitHub
import torch
import torch.nn.functional as F
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

def verify_round_trip(
    original_texts: list[str],
    original_embeddings: torch.Tensor,
    encoder: TextToEmbeddingModelPipeline,
    decoder: EmbeddingToTextModelPipeline,
    threshold: float = 0.9,
) -> dict:
    """Verify round-trip fidelity: embed -> decode -> re-embed."""
    # Decode embeddings to text
    decoded_texts = decoder.predict(
        original_embeddings,
        target_lang="eng_Latn",
        max_seq_len=256,
    )

    # Re-encode decoded texts
    reencoded = encoder.predict(decoded_texts, source_lang="eng_Latn")

    # Compute cosine similarity
    similarities = F.cosine_similarity(
        original_embeddings, reencoded, dim=1
    )

    # Check threshold
    passed = (similarities >= threshold).float().mean().item()

    return {
        "similarities": similarities,
        "mean_similarity": similarities.mean().item(),
        "pass_rate": passed,
        "threshold": threshold,
        "failures": (similarities < threshold).nonzero().squeeze().tolist(),
    }
```

### Anti-Patterns to Avoid
- **Unit-normalizing SONAR embeddings:** SONAR decoder expects original distribution, not L2-normalized vectors
- **Computing stats on val/test:** Data leakage; always compute normalization from training set only
- **Using different seeds for different split sizes:** Breaks nested property; use single shuffle then subset
- **Storing normalized embeddings only:** Must store original embeddings; normalize at load time

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SONAR encoding | Custom transformer | `TextToEmbeddingModelPipeline` | Pre-trained weights, tokenizer compatibility |
| SONAR decoding | Custom seq2seq | `EmbeddingToTextModelPipeline` | Beam search, n-gram blocking built-in |
| Dataset shuffling | Python random | `np.random.permutation` + seed | Reproducibility across runs |
| Cosine similarity | Manual dot/norm | `F.cosine_similarity` | Numerical stability, GPU optimized |

**Key insight:** The SONAR pipeline is non-trivial (specific encoder/decoder checkpoints, fairseq2 integration). Use the official `sonar-space` library exclusively.

## Common Pitfalls

### Pitfall 1: Normalization/Denormalization Mismatch
**What goes wrong:** Training in normalized space, then passing normalized embeddings to SONAR decoder without denormalization
**Why it happens:** Easy to forget denormalize step when the flow model outputs look reasonable (valid tensor shapes)
**How to avoid:** Always call `denormalize()` before any SONAR decoder operation; add assertion in decode wrapper
**Warning signs:** Decoded text is gibberish, repeated tokens, or empty strings despite valid embedding shapes

### Pitfall 2: Data Leakage via Normalization Stats
**What goes wrong:** Computing mean/std from entire dataset (including val/test), leading to optimistic evaluation
**Why it happens:** Natural to call `embeddings.mean()` on full tensor
**How to avoid:** Split data FIRST, then compute stats from training split only
**Warning signs:** Validation loss suspiciously low; results don't reproduce on held-out data

### Pitfall 3: Non-Nested Splits
**What goes wrong:** Each dataset size uses completely different samples, making size ablation invalid
**Why it happens:** Calling `random_split()` separately for each size
**How to avoid:** Single shuffle, then prefix-subset (1K = first 1000 of shuffled, 5K = first 5000, etc.)
**Warning signs:** Some high-quality samples only appear in 1K split, causing anomalous 1K > 5K results

### Pitfall 4: fp16 Precision Loss During Encoding
**What goes wrong:** Encoding in fp16, storing in fp16, accumulating numerical errors
**Why it happens:** Using fp16 for speed without conversion
**How to avoid:** Encode in fp16 for speed, but convert to fp32 before storage
**Warning signs:** Embedding values have suspiciously low precision; round-trip similarity lower than expected

### Pitfall 5: CUDA/CPU Device Mismatch
**What goes wrong:** Encoder on GPU, decoder on CPU (or vice versa), causing slow transfers or errors
**Why it happens:** Default device varies; not explicitly setting device
**How to avoid:** Explicitly pass `device=torch.device("cuda:0")` to both encoder and decoder
**Warning signs:** Encoding is fast but decoding is slow (or vice versa); device mismatch errors

## Code Examples

Verified patterns from official sources:

### SONAR Decoder with N-gram Blocking
```python
# Source: rielbo/decoder.py (existing implementation)
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from fairseq2.generation.step_processor import NGramRepeatBlockProcessor

decoder = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder",
    device=torch.device("cuda:0"),
)

# Decode with n-gram blocking to prevent repetition
step_processors = [NGramRepeatBlockProcessor(ngram_size=3)]
decoded_texts = decoder.predict(
    embeddings,
    target_lang="eng_Latn",
    max_seq_len=256,
    beam_size=5,
    step_processors=step_processors,
)
```

### Dataset Format (Reference Implementation)
```python
# Source: datasets/gsm8k_instructions_vs.pt format
dataset = {
    "embeddings": torch.Tensor,  # [N, 1024], float32
    "instructions": list[str],   # N text prompts
    "sources": dict,             # {"verbalized_sampling": count, ...}
    "config": dict,              # Generation config for reproducibility
    "stats": dict,               # {"n_text_instructions": ..., "total": ...}
}
torch.save(dataset, "path/to/dataset.pt")
```

### Reproducible DataLoader
```python
# Source: PyTorch documentation
import torch
from torch.utils.data import Dataset, DataLoader

def seed_worker(worker_id):
    """Ensure reproducibility in multi-worker DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SONAR with fairseq | SONAR with fairseq2 | 2024 | New API, better performance |
| Manual OT coupling | torchcfm OT-CFM | 2023-2024 | Built-in mini-batch OT |
| Full dataset normalization | Training-set-only normalization | Standard practice | Prevents data leakage |

**Deprecated/outdated:**
- `fairseq` (original): Replaced by `fairseq2` for SONAR
- Unit normalization of embeddings: SONAR decoder expects original scale

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal batch size for SONAR encoding**
   - What we know: fp16 encoding is faster, batch processing is efficient
   - What's unclear: Exact memory requirements for batch_size on L40S (48GB)
   - Recommendation: Start with batch_size=256, increase until OOM

2. **Whether to parallelize SONAR encoding**
   - What we know: Single GPU encoding is standard in existing codebase
   - What's unclear: Whether to use DataParallel for 2x L40S
   - Recommendation: Start single-GPU; parallelize only if encoding is bottleneck

3. **Exact VS generation procedure for 10K**
   - What we know: Existing pipeline produces 4070 samples; config shows VS + EvolInstruct + SLERP
   - What's unclear: Whether to use same mixed approach or pure VS
   - Recommendation: Use pure VS to simplify; existing config can guide if diversity needed

## Sources

### Primary (HIGH confidence)
- [SONAR GitHub](https://github.com/facebookresearch/SONAR) - TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline API
- [PyTorch Documentation](https://docs.pytorch.org/) - torch.save, torch.load, cosine_similarity, DataLoader
- [fairseq2 Documentation](https://facebookresearch.github.io/fairseq2/stable/) - NGramRepeatBlockProcessor, BeamSearchSeq2SeqGenerator
- Existing codebase: `rielbo/decoder.py`, `rielbo/data.py`, `datasets/gsm8k_instructions_vs.pt`

### Secondary (MEDIUM confidence)
- [PyTorch Reproducibility Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) - Seed management best practices
- [sonar-space PyPI](https://pypi.org/project/sonar-space/) - Version compatibility (requires fairseq2>=0.5.2)

### Tertiary (LOW confidence)
- None - all key patterns verified with official sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - sonar-space and PyTorch well-documented
- Architecture: HIGH - follows existing codebase patterns
- Pitfalls: HIGH - verified against existing implementation and research docs

**Research date:** 2026-01-31
**Valid until:** 2026-03-01 (60 days - stable domain)
