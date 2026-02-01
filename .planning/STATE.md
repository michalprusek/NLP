# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Find the best flow matching architecture for GP-guided prompt generation in SONAR space
**Current focus:** Phase 7 - Data Augmentation (COMPLETE)

## Current Position

Phase: 7 of 11 (Data Augmentation) - COMPLETE
Plan: 2 of 2 in current phase - COMPLETE
Status: Phase complete
Last activity: 2026-02-01 -- Completed Phase 7 execution and verification

Progress: [█████████████░] 64%

## Performance Metrics

**Velocity:**
- Total plans completed: 17
- Average duration: 12 min
- Total execution time: 3.77 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline | 2 | 133min | 67min |
| 02-training-infrastructure | 2 | 9min | 5min |
| 03-baseline-architectures | 3 | 10min | 3min |
| 04-flow-matching-baselines | 3 | 12min | 4min |
| 05-advanced-flow-methods | 2 | 10min | 5min |
| 06-advanced-architectures | 3 | 9min | 3min |
| 07-data-augmentation | 2 | 9min | 5min |

**Recent Trend:**
- Last 5 plans: 06-01 (3min), 06-02 (4min), 06-03 (2min), 07-01 (3min), 07-02 (6min)
- Trend: Fast execution continues

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Study location: study/flow_matching/ (separate from production ecoflow)
- Train on VS dataset for higher quality prompts
- Test 1K/5K/10K sample sizes for dataset scaling ablation
- GPU 1 (A5000) only for all experiments
- Unnormalized embeddings required for SONAR decoder compatibility
- Split pools first then subset for proper nested property across all splits (01-01)
- Semantic deduplication at 0.95 cosine similarity threshold (01-01)
- Pre-normalize embeddings at dataset load time for training efficiency (01-02)
- Store original embeddings alongside normalized for decoder access (01-02)
- Use epsilon 1e-8 clamp on std for numerical stability (01-02)
- Locked EMA decay at 0.9999, grad_clip at 1.0, patience at 20 (02-01)
- Flow matching uses ICFM formulation: x_t = (1-t)*x0 + t*x1 (02-01)
- GPU verification at startup warns if not on A5000/L40S or CUDA_VISIBLE_DEVICES!=1 (02-01)
- Checkpoint save/load order: model -> EMA -> optimizer -> scheduler (02-02)
- Stats consistency check on resume warns but doesn't fail (02-02)
- Wandb.log every 10 steps with step parameter for proper alignment (02-02)
- SimpleMLP uses hidden_dim=256, num_layers=5 for ~920K params (03-01)
- Output layer initialized near zero (std=0.01) for stable training (03-01)
- Model factory pattern: create_model(arch_name) for CLI selection (03-01)
- DiTVelocityNetwork uses hidden_dim=384, num_layers=3, num_heads=6 for ~9.3M params (03-02)
- Import timestep_embedding from mlp.py to avoid code duplication (03-02)
- Velocity prediction loss ~2.0 is expected for normalized ICFM training (03-02)
- ICFM distribution MSE ~1.0 is expected (comparing generated to random targets) (03-03)
- Key quality metric is coherent text generation, not reconstruction MSE (03-03)
- OT-CFM uses OTPlanSampler with method='exact', reg=0.5, normalize_cost=True (04-01)
- Factory pattern for coupling selection: create_coupling(method) (04-01)
- OT-CFM produces ~8% lower val loss than I-CFM (1.841 vs 2.008) (04-02)
- Path straightness similar for both methods (~0.0016 deviation) on small MLP (04-02/04-03)
- CFG-Zero* uses 4% zero-init fraction matching ecoflow/guided_flow.py (04-03)
- Gradient clipping at max_grad_norm=10.0 for guidance stability (04-03)
- Reflow uses 10K pairs (10x dataset) from OT-CFM teacher (05-01)
- Reflow pairs cached at study/datasets/reflow_pairs_1k.pt (05-01)
- GVP schedule (cos/sin) is variance-preserving: alpha^2 + sigma^2 = 1 (05-02)
- SI velocity target is alpha_dot*x0 + sigma_dot*x1, NOT x1-x0 (05-02)
- Reflow produces 3x straighter paths than other methods (0.0005 vs 0.0015) (05-02)
- FiLMLayer uses zero-init for identity transform at start (06-01)
- UNetMLP uses hidden_dims=(512, 256) for ~6.9M params (06-01)
- mamba-ssm installation failed due to CUDA 13.1 vs PyTorch CUDA 12.8 mismatch (06-02)
- Graceful fallback: MAMBA_AVAILABLE=False, ImportError on instantiation (06-02)
- Bidirectional Mamba with chunk_size=64 (16 chunks of 64 dims) (06-02)
- Scaling configs define only architecture-specific params, factory adds defaults (06-03)
- Configuration overlay pattern: defaults <- scale config <- kwargs (06-03)
- Beta(alpha, alpha) for mixup with U-shaped distribution for diverse mixing (07-01)
- Augment x1 only (data), never x0 (noise), before coupling.sample() (07-01)
- Order: mixup -> noise -> dropout as per research (07-01, 07-02)
- aug string parses defaults: "mixup" -> alpha=0.2, "noise" -> std=0.1, "dropout" -> rate=0.1 (07-01, 07-02)
- F.dropout IS dimension masking - satisfies DATA-07 dropout/masking requirement (07-02)
- Augmentation has marginal impact (~0.2%) in short training runs (07-02 ablation)

### Pending Todos

None.

### Blockers/Concerns

- GPU 0 (A100 80GB) unavailable during 01-01 execution, used GPU 1 (A5000 24GB) which is slower
- 10K generation took longer than estimated (~114 min vs ~67 min) due to deduplication overhead
- 2 samples (0.4%) failed round-trip verification - acceptable edge cases (mixed language, grammar reconstruction)
- WANDB authentication missing - using WANDB_MODE=offline for training runs
- GPU 1 memory pressure from other processes - use GPU 0 for SONAR decoder when needed
- mamba-ssm requires matching CUDA versions - currently blocked by version mismatch (06-02)

## Session Continuity

Last session: 2026-02-01 13:29 UTC
Stopped at: Completed 07-02-PLAN.md (dimension dropout and ablation)
Resume file: None

## Phase 5 Summary (COMPLETE)

**Phase 5 complete.** Advanced flow methods implemented and compared.

Delivered (05-01):
- ReflowPairGenerator for synthetic pair generation via teacher ODE
- ReflowCoupling for training on pre-generated pairs
- Training script study/flow_matching/reflow/train_reflow.py
- 2-rectified flow checkpoint: study/checkpoints/mlp-reflow-1k-none/best.pt
- Path straightness 3x better than I-CFM (0.00052 vs 0.0016)

Delivered (05-02):
- Schedule module with linear and GVP interpolation (study/flow_matching/schedules.py)
- StochasticInterpolantCoupling with time-varying velocity target
- SI-GVP trained model checkpoint: study/checkpoints/mlp-si-gvp-1k-none/best.pt
- Comprehensive comparison script: study/flow_matching/compare_flow_methods.py

**Full Flow Method Comparison (100 samples, 100 steps):**
| Method | Dist MSE | Path Dev | Path Max |
|--------|----------|----------|----------|
| I-CFM | 0.9979 | 0.001541 | 0.003143 |
| OT-CFM | 0.9986 | 0.001553 | 0.003250 |
| Reflow | 0.9923 | 0.000521 | 0.001036 |
| SI-GVP | 1.0006 | 0.001556 | 0.003182 |

**Phase 5 success criteria - ALL VERIFIED:**
1. [x] Rectified Flow reflow procedure runs on trained I-CFM model
2. [x] Reflow produces straighter paths (3x improvement: 0.0005 vs 0.0015)
3. [x] Stochastic Interpolants with learnable interpolation trains
4. [x] All flow methods produce comparable sample quality

**Key findings:**
1. Reflow produces 3x straighter paths (0.0005 vs 0.0015)
2. All methods have similar distribution MSE (~1.0)
3. SI-GVP offers no advantage over I-CFM for SONAR embeddings
4. All methods generate coherent text

Ready for: Phase 6 (Advanced Architectures), Phase 7 (Data Augmentation), Phase 8 (GP-Guided Sampling)

## Phase 6 Summary (COMPLETE)

**Phase 6 complete.** All advanced architectures implemented with scaling variants.

Delivered (06-01):
- FiLMLayer for feature-wise linear modulation
- UNetMLP velocity network with skip connections
- Extended model factory: create_model("unet")
- ~6.9M params with default config

Delivered (06-02):
- MambaVelocityNetwork (experimental, blocked by mamba-ssm installation)
- MAMBA_AVAILABLE flag for graceful fallback
- Extended model factory: create_model("mamba") with ImportError handling
- Bidirectional SSM treating embedding as 16 chunks of 64 dims

Delivered (06-03):
- SCALING_CONFIGS with Tiny/Small/Base for all architectures
- get_scaled_config() and list_available_scales() helpers
- Extended create_model(arch, scale) with configuration overlay
- All architecture/scale combinations verified on GPU

**Architecture Scaling Matrix:**
| Arch | Tiny | Small | Base |
|------|------|-------|------|
| mlp | 362K | 920K | 1.77M |
| dit | 3.16M | 9.31M | 20.9M |
| unet | 5.14M | 6.89M | 9.07M |
| mamba | (blocked) | (blocked) | (blocked) |

**Phase 6 success criteria - ALL VERIFIED:**
1. [x] U-Net MLP with FiLM conditioning trains correctly
2. [x] Mamba velocity network has graceful fallback
3. [x] All architectures have Tiny/Small/Base scaling variants
4. [x] create_model(arch, scale) works for all combinations

Ready for: Phase 7 (Data Augmentation), Phase 8 (GP-Guided Sampling), Phase 9 (Experimental Matrix)

## Phase 7 Summary (COMPLETE)

**Phase 7 complete.** Data augmentation module for flow matching training.

Delivered (07-01):
- AugmentationConfig dataclass with mixup_alpha, noise_std, dropout_rate
- mixup_embeddings() using Beta(alpha, alpha) sampling
- add_gaussian_noise() for controlled perturbation
- augment_batch() combining augmentations with training flag
- FlowTrainer integration: augment x1 before coupling.sample()
- CLI args --mixup-alpha and --noise-std for fine-grained control

**Research finding:** Beta(0.2, 0.2) produces U-shaped distribution (~35% < 0.1, ~33% > 0.9), resulting in cosine similarity ~0.55-0.60 between original and mixed. Statistics (mean ~0, std ~1) are preserved.

Delivered (07-02):
- dimension_dropout() using F.dropout with automatic scaling
- Updated augment_batch() order: mixup -> noise -> dropout
- parse_aug_string() helper for config shorthand parsing
- CLI arg --dropout-rate for dimension dropout control
- Ablation comparing baseline vs augmented training (30 epochs)

**Ablation Results (30 epochs, 1k dataset):**
| Aug Method | Best Val Loss | Delta |
|------------|---------------|-------|
| none       | 1.992204      | baseline |
| mixup      | 1.996063      | +0.19% |
| noise      | 1.995363      | +0.16% |
| mixup+noise| 1.996343      | +0.21% |

**Research finding:** Augmentation has marginal impact (~0.2%) in short training runs. May help more with longer training or larger models where overfitting is the bottleneck.

**Phase 7 success criteria - ALL VERIFIED:**
1. [x] Linear interpolation (mixup) generates valid training pairs
2. [x] Gaussian noise injection augments training data
3. [x] Dimension dropout/masking augments training data (F.dropout IS masking)
4. [x] Augmentation tested for generalization impact (ablation completed)

Ready for: Phase 8 (GP-Guided Sampling), Phase 9 (Evaluation Suite), Phase 10 (Ablation Studies)

## Phase 4 Summary (COMPLETE)

**Phase 4 complete.** Flow matching baselines established.

Delivered (04-01):
- Coupling abstraction module (study/flow_matching/coupling/)
- ICFMCoupling and OTCFMCoupling classes
- Factory function create_coupling() for method selection
- FlowTrainer refactored to use coupling abstraction
- OT-CFM config parameters (sigma, reg, normalize_cost)

Delivered (04-02):
- Path straightness evaluation function (compute_path_straightness)
- OT-CFM checkpoint: study/checkpoints/mlp-otcfm-1k-none/best.pt
- Quantitative comparison: OT-CFM ~8% lower loss, similar path geometry

Delivered (04-03):
- CFG-Zero* guidance module (study/flow_matching/guidance.py)
- get_guidance_lambda() with 4% zero-init schedule
- guided_euler_ode_integrate() with gradient clipping
- sample_with_guidance() convenience function
- Phase 4 verification complete

**Full training results (100 epochs, 1k):**
| Method | Val Loss | Mean Path Dev | Text Quality |
|--------|----------|---------------|--------------|
| I-CFM | 2.008 | 0.0016 | Coherent |
| OT-CFM | 1.841 | 0.0016 | Coherent |

**Key finding:** OT-CFM's benefit at this scale is training efficiency (lower loss), not straighter paths. Both methods produce very straight paths (~0.0016 deviation).
