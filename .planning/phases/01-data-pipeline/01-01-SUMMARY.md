---
phase: 01-data-pipeline
plan: 01
subsystem: data
tags: [sonar, embeddings, vs-prompting, dataset-generation, train-val-test-splits]

# Dependency graph
requires: []
provides:
  - 10K VS dataset with SONAR embeddings (study/datasets/vs_10k.pt)
  - Nested train/val/test splits for 1K/5K/10K sizes
  - Dataset generation and split creation scripts
affects: [02-flow-architecture, 03-training-loop]

# Tech tracking
tech-stack:
  added: [sonar, vllm, tqdm]
  patterns: [verbosed-sampling-prompting, nested-splits-for-ablation]

key-files:
  created:
    - study/data/generate_dataset.py
    - study/data/create_splits.py
    - study/datasets/vs_10k.pt
    - study/datasets/splits/{1k,5k,10k}/{train,val,test}.pt
  modified: []

key-decisions:
  - "Split pools first then subset for nested property across all splits"
  - "Use SONAR text_sonar_basic_encoder with float16 encoding, convert to float32 for storage"
  - "Semantic deduplication at cosine similarity 0.95 threshold"
  - "Checkpoint every 1000 samples for long-running generation"

patterns-established:
  - "VS prompting: sample 3-5 GSM8K problems, ask LLM for meta-cognitive instructions"
  - "Nested splits: 1K subset of 5K subset of 10K for fair ablation comparisons"
  - "Dataset format: {embeddings, instructions, sources, config, stats}"

# Metrics
duration: 124min
completed: 2026-02-01
---

# Phase 01 Plan 01: Data Pipeline Summary

**10K verbosed sampling dataset with SONAR embeddings (1024D, float32) and nested 80/10/10 train/val/test splits for 1K/5K/10K ablation studies**

## Performance

- **Duration:** 2h 4min
- **Started:** 2026-01-31T23:12:27Z
- **Completed:** 2026-02-01T01:16:00Z
- **Tasks:** 2/2
- **Files created:** 12 (2 scripts, 1 full dataset, 9 split files)

## Accomplishments
- Generated 10,000 unique meta-cognitive instructions using VS prompting from GSM8K problems
- Encoded all instructions with SONAR to 1024D embeddings (float32, unnormalized)
- Created nested splits where smaller sizes are proper subsets of larger ones
- Verified no data leakage between train/val/test within each size
- Implemented checkpointing for long-running generation (saved every 1000 samples)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create VS dataset generation script** - `54a6a75` (feat)
2. **Task 2: Generate 10K dataset and create splits** - `0337048` (feat)

## Files Created/Modified
- `study/data/generate_dataset.py` - VS dataset generation with SONAR encoding, deduplication, checkpointing
- `study/data/create_splits.py` - Nested split creation with verification
- `study/datasets/vs_10k.pt` - Full 10K dataset (embeddings, instructions, config, stats)
- `study/datasets/splits/1k/{train,val,test}.pt` - 1K splits (800/100/100)
- `study/datasets/splits/5k/{train,val,test}.pt` - 5K splits (4000/500/500)
- `study/datasets/splits/10k/{train,val,test}.pt` - 10K splits (8000/1000/1000)

## Decisions Made
- **Split strategy fix:** Original implementation split after selecting size indices, breaking nested property for val/test. Fixed by creating train/val/test pools first from full 10K, then subsetting each pool for smaller sizes.
- **GPU selection:** Used GPU 1 (A5000, 24GB) as GPU 0 (A100) was occupied by another process.
- **Deduplication threshold:** Used 0.95 cosine similarity to filter near-duplicate instructions.
- **LLM temperature:** Used 1.0 for diverse instruction generation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed tensor device mismatch in generate_dataset.py**
- **Found during:** Task 1 dry-run
- **Issue:** SONAR encoder returns GPU tensors, concatenation with CPU tensor failed
- **Fix:** Added `.cpu()` call after encoding to move embeddings to CPU before concatenation
- **Files modified:** study/data/generate_dataset.py
- **Verification:** Dry-run completed successfully
- **Committed in:** 54a6a75

**2. [Rule 1 - Bug] Fixed nested split logic in create_splits.py**
- **Found during:** Task 2 verification
- **Issue:** Val/test splits not properly nested (1K val not subset of 5K val)
- **Fix:** Changed to split full dataset into train/val/test pools first, then subset from each pool
- **Files modified:** study/data/create_splits.py
- **Verification:** All 6 nested property checks passed
- **Committed in:** 0337048

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes were necessary for correct operation. No scope creep.

## Issues Encountered
- GPU 0 was occupied, switched to GPU 1 (A5000) which worked but slower (~1.5 samples/sec vs estimated 2.5)
- 10K generation took ~114 minutes (vs estimated ~67 minutes) due to semantic deduplication overhead

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- 10K dataset ready for flow matching training
- All split sizes (1K/5K/10K) ready for dataset scaling ablation
- Embeddings are unnormalized float32, compatible with SONAR decoder
- Ready for Phase 2: Flow Architecture implementation

---
*Phase: 01-data-pipeline*
*Completed: 2026-02-01*
