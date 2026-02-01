---
phase: 06-advanced-architectures
plan: 02
subsystem: models
tags: [mamba, ssm, velocity-network, flow-matching, bidirectional]

# Dependency graph
requires:
  - phase: 03-baseline-architectures
    provides: timestep_embedding function in mlp.py
  - phase: 06-01
    provides: model factory pattern with create_model()
provides:
  - MambaVelocityNetwork class (experimental SSM velocity network)
  - MAMBA_AVAILABLE flag for graceful fallback
  - Extended model factory with arch="mamba" support
affects: [07-architecture-training, 08-gp-guided-sampling, 10-ablation]

# Tech tracking
tech-stack:
  added: [mamba-ssm (attempted, failed CUDA mismatch)]
  patterns: [graceful-fallback, conditional-import, bidirectional-ssm]

key-files:
  created:
    - study/flow_matching/models/mamba_velocity.py
  modified:
    - study/flow_matching/models/__init__.py

key-decisions:
  - "mamba-ssm installation failed due to CUDA 13.1 vs PyTorch CUDA 12.8 mismatch"
  - "Graceful fallback: MAMBA_AVAILABLE=False, ImportError on instantiation"
  - "Bidirectional Mamba with chunk_size=64 (16 chunks of 64 dims)"
  - "Time conditioning added after each Mamba layer"
  - "Zero-init output layer for stable training start"

patterns-established:
  - "Conditional import pattern: try/except with AVAILABLE flag"
  - "Factory fallback: check availability before instantiation"
  - "Dynamic error messages: list available options based on installed packages"

# Metrics
duration: 4min
completed: 2026-02-01
---

# Phase 6 Plan 2: Mamba Velocity Network Summary

**Experimental Mamba velocity network with bidirectional SSM treating 1024-dim embedding as 16 chunks of 64 dims, with graceful fallback when mamba-ssm unavailable**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-01T12:27:45Z
- **Completed:** 2026-02-01T12:32:00Z
- **Tasks:** 3 (1 no-op, 2 with commits)
- **Files modified:** 2

## Accomplishments
- Implemented MambaVelocityNetwork with bidirectional SSM processing
- Created graceful fallback pattern for unavailable mamba-ssm
- Extended model factory to support arch="mamba" with dynamic availability

## Task Commits

Each task was committed atomically:

1. **Task 1: Install mamba-ssm dependency** - no commit (installation failed, expected)
2. **Task 2: Implement MambaVelocityNetwork** - `d78a724` (feat)
3. **Task 3: Integrate Mamba into model factory** - `27e826e` (feat)

## Files Created/Modified
- `study/flow_matching/models/mamba_velocity.py` - Experimental Mamba velocity network with bidirectional SSM
- `study/flow_matching/models/__init__.py` - Extended factory with mamba support and MAMBA_AVAILABLE export

## Decisions Made

1. **CUDA version mismatch handled gracefully:** mamba-ssm requires matching CUDA versions (PyTorch CUDA 12.8 vs system CUDA 13.1). Implementation uses try/except pattern so code works regardless of installation status.

2. **Bidirectional processing:** Mamba is inherently causal; for non-causal flow matching, use bidirectional (forward + backward) with concatenated outputs. Pattern from PointMamba literature.

3. **Chunk size 64:** 1024-dim embedding -> 16 chunks of 64 dims. This creates a 16-step "virtual sequence" for Mamba to process. Verified 1024 % chunk_size == 0 in __init__.

4. **Time conditioning after each layer:** Add time embedding projection to both forward and backward hidden states after each Mamba layer.

5. **Import timestep_embedding from mlp.py:** No code duplication; reuse existing function.

## Deviations from Plan

None - plan executed exactly as written. The plan anticipated potential installation failure and specified graceful fallback behavior, which was followed.

## Issues Encountered

1. **CUDA version mismatch:** mamba-ssm requires CUDA toolkit matching PyTorch CUDA version. System has CUDA 13.1 but PyTorch was compiled with CUDA 12.8. This is a known limitation documented in the research phase.

2. **Package import chain issue:** The study/flow_matching package's __init__.py imports from study/data/dataset.py which has a Python 3.9+ type hint syntax error (tuple[Tensor, str] instead of Tuple[Tensor, str]). Worked around by using direct module imports for verification.

## User Setup Required

None - no external service configuration required.

To enable Mamba velocity network in the future:
1. Ensure PyTorch CUDA version matches system CUDA toolkit
2. Run: `pip install mamba-ssm`
3. Verify: `python -c "from study.flow_matching.models import MAMBA_AVAILABLE; print(MAMBA_AVAILABLE)"`

## Next Phase Readiness
- MambaVelocityNetwork class ready for future use when mamba-ssm is installable
- MAMBA_AVAILABLE flag allows conditional experiment execution
- Factory pattern enables seamless architecture switching in training scripts
- Phase 6 Plan 3 (architecture scaling) can proceed

---
*Phase: 06-advanced-architectures*
*Completed: 2026-02-01*
