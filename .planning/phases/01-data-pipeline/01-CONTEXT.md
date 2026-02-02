# Phase 1: Data Pipeline - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate a 10K verbosed sampling dataset with SONAR embeddings, create nested train/val/test splits for 1K/5K/10K sizes, compute normalization statistics, and verify SONAR decoder produces coherent text from denormalized embeddings.

</domain>

<decisions>
## Implementation Decisions

### Dataset Generation
- Generate fresh 10K prompts using the VS (verbosed sampling) pipeline from rielbo
- Each sample includes: prompt text, SONAR embedding (1024D), and metadata (source, timestamp)
- Store as PyTorch tensors with accompanying JSON metadata

### Split Strategy
- **Nested structure**: 1K ⊂ 5K ⊂ 10K (same samples, just more added at larger sizes)
- Ensures clean comparison across dataset scales — "more data" is the only variable
- **Ratio**: 80% train / 10% val / 10% test
  - 1K: 800 train, 100 val, 100 test
  - 5K: 4000 train, 500 val, 500 test
  - 10K: 8000 train, 1000 val, 1000 test
- Random seed fixed for reproducibility

### Normalization Design
- **Per-dimension** normalization (mean/std computed per embedding dimension)
- Statistics computed on the **10K training set only** (not val/test)
- Same stats applied to 1K/5K subsets (since they're nested)
- Stats saved as `normalization_stats.pt` with `mean` and `std` tensors
- Unnormalized embeddings required for SONAR decoder — denormalize before decoding

### Verification Criteria
- **Automated semantic similarity gate**: cosine similarity ≥ 0.9 between original and round-trip embeddings
- Round-trip = encode(original_text) → denormalize → decode → encode(decoded_text)
- Pipeline passes if 95%+ of samples meet the threshold
- Log failure cases for manual review

### Claude's Discretion
- Exact file naming conventions and directory structure
- Batch size for embedding generation
- Whether to use multiprocessing for SONAR encoding
- Format of metadata JSON schema

</decisions>

<specifics>
## Specific Ideas

- Nested splits ensure ablation studies on dataset size are valid comparisons
- Semantic similarity is more robust than BLEU for paraphrase-tolerant verification
- Normalization stats from 10K only prevents data leakage to smaller subsets

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-data-pipeline*
*Context gathered: 2026-01-31*
