---
phase: 03-baseline-architectures
verified: 2026-02-01T10:30:00Z
status: gaps_found
score: 3/4 must-haves verified
gaps:
  - truth: "Both architectures produce reasonable reconstruction MSE (<0.1)"
    status: failed
    reason: "No actual MSE measurement exists; only loss values ~2.0 from training"
    artifacts:
      - path: "study/checkpoints/mlp-icfm-1k-none/best.pt"
        issue: "Checkpoint exists but no reconstruction test was performed"
      - path: "study/checkpoints/dit-icfm-1k-none/best.pt"
        issue: "Checkpoint exists but no reconstruction test was performed"
    missing:
      - "Test script that generates embeddings and computes MSE vs ground truth"
      - "Actual MSE measurements showing values < 0.1"
  - truth: "Generated embeddings decode to coherent text"
    status: uncertain
    reason: "SUMMARY claims text samples exist, but no verification script was found"
    artifacts:
      - path: "study/data/verify_decoder.py"
        issue: "Decoder verification infrastructure exists but no evidence it was run on generated embeddings"
    missing:
      - "Test script that: loads checkpoint -> generates embeddings -> decodes -> logs output"
      - "Actual decoded text samples from trained models (not just claims in SUMMARY)"
---

# Phase 3: Baseline Architectures Verification Report

**Phase Goal:** Two baseline velocity networks trained and evaluated for comparison
**Verified:** 2026-02-01T10:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Simple MLP velocity network trains without NaN loss | ✓ VERIFIED | Checkpoint shows best_loss=2.008077 (finite), epoch 2 |
| 2 | DiT velocity network (ported from ecoflow) trains without NaN loss | ✓ VERIFIED | Checkpoint shows best_loss=2.007965 (finite), epoch 2 |
| 3 | Both architectures produce reasonable reconstruction MSE (<0.1) | ✗ FAILED | No MSE measurement exists; only training loss ~2.0 |
| 4 | Generated embeddings decode to coherent text | ? UNCERTAIN | SUMMARY claims samples but no test script found |

**Score:** 2/4 truths verified, 1 uncertain, 1 failed

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `study/flow_matching/models/mlp.py` | SimpleMLP velocity network class | ✓ VERIFIED | 138 lines, 920,064 params (target: 800K-1.2M), exports SimpleMLP |
| `study/flow_matching/models/dit.py` | DiTVelocityNetwork class | ✓ VERIFIED | 199 lines, 9,309,952 params (target: 9-10M), exports DiTVelocityNetwork, AdaLNBlock |
| `study/flow_matching/models/__init__.py` | Model factory function | ✓ VERIFIED | 65 lines, create_model('mlp'), create_model('dit') both work |
| `study/flow_matching/train.py` | CLI with real model instantiation | ✓ VERIFIED | Uses create_model(args.arch) at line 251 |
| `study/checkpoints/mlp-icfm-1k-none/best.pt` | MLP trained checkpoint | ✓ VERIFIED | Exists, epoch 2, best_loss 2.008077 (no NaN) |
| `study/checkpoints/dit-icfm-1k-none/best.pt` | DiT trained checkpoint | ✓ VERIFIED | Exists, epoch 2, best_loss 2.007965 (no NaN) |
| `study/checkpoints/mlp-icfm-5k-none/best.pt` | MLP extended training | ✓ VERIFIED | Exists, epoch 3, best_loss 2.003203 (no NaN) |

**All artifacts exist and are substantive.**

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `train.py` | `models/__init__.py` | `import create_model` | ✓ WIRED | Line 30: from study.flow_matching.models import create_model |
| `train.py` | `create_model('mlp')` | function call | ✓ WIRED | Line 251: model = create_model(args.arch) |
| `models/__init__.py` | `models/mlp.py` | import SimpleMLP | ✓ WIRED | Line 14: from study.flow_matching.models.mlp import SimpleMLP |
| `models/__init__.py` | `models/dit.py` | import DiTVelocityNetwork | ✓ WIRED | Line 15: from study.flow_matching.models.dit import DiTVelocityNetwork |
| `models/dit.py` | `models/mlp.py` | import timestep_embedding | ✓ WIRED | Line 13: from study.flow_matching.models.mlp import timestep_embedding |

**All key links are properly wired.**

### Requirements Coverage

From REQUIREMENTS.md Phase 3 requirements:

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| ARCH-01: Implement Simple MLP baseline (~1M params) | ✓ SATISFIED | None |
| ARCH-02: Port DiT baseline from ecoflow (~9.4M params) | ✓ SATISFIED | None |

**All Phase 3 requirements satisfied.**

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No anti-patterns detected |

**No stub patterns, TODO comments, or placeholder implementations found in model files.**

### Human Verification Required

None — gaps are automatable programmatic checks.

### Gaps Summary

**Gap 1: No reconstruction MSE measurement**

The success criteria states "Both architectures produce reasonable reconstruction MSE (<0.1)". The SUMMARY documents show training loss values around 2.0, but this is the **velocity prediction loss** during training, NOT reconstruction MSE.

To verify this truth, we need:
1. A test script that:
   - Loads a trained checkpoint
   - Takes test embeddings (x1)
   - Adds noise at t=0 to get x0
   - Uses ODE integration with the trained model to reconstruct x1_hat
   - Computes MSE(x1, x1_hat)
2. Run this test on both MLP and DiT checkpoints
3. Verify MSE < 0.1

**This is a critical gap** — without reconstruction quality measurements, we cannot claim the models are working correctly. A model can have low training loss but poor reconstruction if it learned a degenerate solution.

**Gap 2: No actual text generation verification**

The SUMMARY claims:
- "Generated embeddings decode to coherent text"
- Provides sample decoded text

However, verification found:
- `study/data/verify_decoder.py` exists (infrastructure ready)
- No evidence this was run on **generated** embeddings from trained models
- No test script that: checkpoint → generate → denormalize → decode → print

The SUMMARY may have fabricated the text samples, or they may exist but verification couldn't find proof.

To close this gap:
1. Create test script that generates embeddings from trained models
2. Denormalize using normalization_stats.pt
3. Decode using SONAR decoder
4. Print 3-5 samples
5. Verify coherence (should be English sentences about problem-solving)

---

**Recommendation:** Create plan 03-03 to:
1. Implement reconstruction MSE test
2. Implement text generation verification
3. Run both tests on all trained checkpoints
4. Document actual MSE and decoded text results

Only then can Phase 3 goal "Two baseline velocity networks trained and evaluated for comparison" be considered achieved.

---

_Verified: 2026-02-01T10:30:00Z_
_Verifier: Claude (gsd-verifier)_
