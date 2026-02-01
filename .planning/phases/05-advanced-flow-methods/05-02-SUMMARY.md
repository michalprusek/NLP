---
phase: 05-advanced-flow-methods
plan: 02
subsystem: flow-matching
tags: [stochastic-interpolants, gvp-schedule, variance-preserving, flow-comparison]

# Dependency graph
requires:
  - phase: 04-flow-matching-baselines
    provides: I-CFM, OT-CFM training infrastructure and checkpoints
  - phase: 05-01
    provides: Reflow (Rectified Flow) checkpoint for comparison
provides:
  - Schedule module with linear and GVP interpolation schedules
  - StochasticInterpolantCoupling class with configurable schedule
  - SI-GVP trained model checkpoint
  - Comprehensive comparison of all 4 flow methods
affects: [08-gp-guided-sampling, phase-5-summary]

# Tech tracking
tech-stack:
  added: []
  patterns: [schedule-based-interpolation, variance-preserving-flows]

key-files:
  created:
    - study/flow_matching/schedules.py
    - study/flow_matching/coupling/stochastic.py
    - study/flow_matching/compare_flow_methods.py
  modified:
    - study/flow_matching/coupling/__init__.py
    - study/flow_matching/config.py
    - study/flow_matching/train.py

key-decisions:
  - "GVP schedule (cos/sin) is variance-preserving: alpha^2 + sigma^2 = 1"
  - "SI velocity target is alpha_dot*x0 + sigma_dot*x1, NOT x1-x0"
  - "Map 'si' to 'si-gvp' for consistent checkpoint naming"
  - "Reflow produces 3x straighter paths than other methods"

patterns-established:
  - "Schedule abstraction: get_schedule(name) returns callable for interpolation"
  - "SI coupling uses time-varying velocity, different from I-CFM constant velocity"

# Metrics
duration: 5min
completed: 2026-02-01
---

# Phase 5 Plan 02: Stochastic Interpolants Summary

**GVP schedule (cos/sin variance-preserving) with time-varying velocity targets; Reflow produces 3x straighter paths than all other methods**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-01T11:30:23Z
- **Completed:** 2026-02-01T11:35:20Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Implemented schedule module with linear and GVP (trigonometric) interpolation
- Created StochasticInterpolantCoupling with correct time-varying velocity target
- Trained SI-GVP model: val loss 2.446, comparable to baselines
- Comprehensive comparison of all 4 flow methods (I-CFM, OT-CFM, Reflow, SI-GVP)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement schedule module and StochasticInterpolantCoupling** - `a4b2515` (feat)
2. **Task 2: Train SI-GVP model** - `f2f78ac` (feat)
3. **Task 3: Flow method comparison** - `6546f56` (feat)

**Plan metadata:** pending

## Files Created/Modified
- `study/flow_matching/schedules.py` - Linear and GVP interpolation schedules with derivatives
- `study/flow_matching/coupling/stochastic.py` - StochasticInterpolantCoupling class
- `study/flow_matching/coupling/__init__.py` - Updated factory with si/si-gvp/si-linear
- `study/flow_matching/config.py` - Added si_schedule config field
- `study/flow_matching/train.py` - Added --schedule argument for SI variants
- `study/flow_matching/compare_flow_methods.py` - Systematic comparison script

## Decisions Made
- GVP schedule uses cos(pi*t/2) and sin(pi*t/2) for variance-preserving property
- SI velocity target is derivative of interpolation: alpha_dot*x0 + sigma_dot*x1
- This differs from I-CFM which uses constant velocity x1-x0
- Map 'si' to 'si-gvp' in train.py for consistent checkpoint naming

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- GPU 1 out of memory for SONAR decoder during evaluation
- Resolution: Used GPU 0 for text generation, GPU 1 for training/metrics

## Quantitative Comparison Results

| Method | Dist MSE | Path Dev | Path Max |
|--------|----------|----------|----------|
| I-CFM | 0.9979 | 0.001541 | 0.003143 |
| OT-CFM | 0.9986 | 0.001553 | 0.003250 |
| Reflow | 0.9923 | 0.000521 | 0.001036 |
| SI-GVP | 1.0006 | 0.001556 | 0.003182 |

**Key Findings:**
1. **Reflow has 3x straighter paths** (0.0005 vs 0.0015-0.0016)
2. **All methods have similar distribution MSE** (~1.0)
3. **SI-GVP offers no advantage** over I-CFM for SONAR embeddings
4. **All methods generate coherent text**

## Phase 5 Success Criteria Verification

1. [x] Rectified Flow reflow procedure runs on trained I-CFM model (05-01)
2. [x] Reflow produces straighter paths (3x improvement!)
3. [x] Stochastic Interpolants with GVP schedule trains successfully
4. [x] All flow methods produce comparable sample quality

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 4 flow methods trained and evaluated
- Reflow recommended for applications requiring straight paths
- Ready for Phase 8: GP-Guided Sampling with flow-based generation
- DiT architecture comparison pending (Phase 6)

---
*Phase: 05-advanced-flow-methods*
*Completed: 2026-02-01*
