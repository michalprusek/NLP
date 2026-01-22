# FlowPO-HD: GP Benchmark Findings & Solutions

## Executive Summary

SAAS GP v 1024D SONAR prostoru selhával kvůli **overconfident predikcím** (3-6σ chyby).
Benchmark odhalil, že problém je v **extrapolaci vs interpolaci**:
- Benchmark (LOOCV) testoval interpolaci → SAAS fungoval (Spearman 0.87)
- FlowPO-HD dělá extrapolaci do neznámých regionů → SAAS selhával

**Řešení**: VanillaGP s dimension-scaled LogNormal prior (Hvarfner 2024)

---

## Benchmark Results (26 HbBoPs points, fidelity >= 600)

| Model | RMSE | Spearman | Coverage@90% | σ-error | Status |
|-------|------|----------|--------------|---------|--------|
| **VanillaGP** | **0.0182** | 0.8020 | **96%** | **0.69** | ✅ BEST |
| ImprovedSAAS | 0.0187 | **0.8191** | 73% | 1.02 | ✅ Good |
| DKL-10D | 0.0195 | 0.7035 | 0% | 27.25 | ❌ Overconfident |
| Original SAAS | ~0.02 | ~0.73 | ~75% | 3-6 | ❌ Overconfident |

### Key Metrics Explained

- **RMSE**: Root Mean Squared Error - nižší = lepší predikce
- **Spearman**: Rank correlation - vyšší = lepší ranking pro BO
- **Coverage@90%**: % predikcí v 90% CI - cíl ~90%
- **σ-error**: |predikce - skutečnost| / std - cíl ~1.0

---

## Root Cause Analysis

### Proč SAAS selhával v FlowPO-HD

1. **Benchmark testoval INTERPOLACI** (LOOCV)
   - Predikce na bodech Z TÉŽE distribuce
   - GP má data pro spolehlivou predikci
   - Velké lengthscales (25-46) → "hladká funkce"

2. **FlowPO-HD dělá EXTRAPOLACI** (Active Learning)
   - Acquisition function navrhuje body MIMO training distribuci
   - V 1024D je téměř každý nový bod daleko od všech
   - Malé lengthscales (0.5-2.5) → overfit na training data
   - GP extrapoluje = náhodný šum

### Curse of Dimensionality

V 1024D s 26 body:
- Každý bod pokrývá ~0 objemu prostoru
- Nový kandidát je vždy "daleko" od všech
- GP nemá data pro interpolaci → musí extrapolovat
- Extrapolace v 1024D = overconfident garbage

---

## Lengthscale Analysis

### SAAS v Benchmarku (fungující)
```
lengthscales: [25.620, 31.088, 31.974, 33.812, 33.814]  # VELKÉ!
→ GP předpokládá hladkou funkci
→ Rozumná nejistota při extrapolaci
→ Spearman 0.87
```

### SAAS v FlowPO-HD (nefungující)
```
lengthscales: [0.558, 0.561, 0.564, 0.569, 0.572]  # MALÉ!
→ GP se snaží přesně fitovat training data
→ Overconfidence při extrapolaci
→ Prediction error: +5.64σ
```

### VanillaGP (Hvarfner 2024)
```
lengthscales: [77.87, 78.65]  # VELKÉ!
→ Dimension-scaled LogNormal prior
→ loc = sqrt(2) + log(1024)/2 = 4.88
→ Median lengthscale = exp(4.88) ≈ 132
→ Správná kalibrace (96% coverage)
```

---

## Solution: Dimension-Scaled Lengthscale Prior

### Hvarfner et al. 2024 (ICML)

**Paper**: "Vanilla Bayesian Optimization Performs Great in High Dimensions"

**Key Insight**: Default lengthscale priors (Gamma(3,6)) jsou příliš malé pro high-D.

**Solution**: LogNormal prior škálovaný dimenzionalitou:
```python
loc = sqrt(2) + log(dim) / 2  # Pro 1024D: 4.88
scale = sqrt(3)               # Konstantní

# Median lengthscale = exp(loc) ≈ 132 pro 1024D
```

**Proč to funguje**:
- Velké lengthscales = GP předpokládá hladkou funkci
- Hladká funkce = rozumná nejistota v neznámých regionech
- Rozumná nejistota = správná kalibrace = lepší BO

---

## Implementation

### VanillaGP (flowpo_hd/improved_gp.py)

```python
class VanillaGPModel(ExactGP):
    def __init__(self, ..., dim, min_lengthscale=0.1):
        # Dimension-scaled LogNormal prior
        ls_loc = SQRT2 + math.log(dim) * 0.5  # 4.88 pro 1024D
        ls_scale = SQRT3

        # RBF kernel s ARD
        self.covar_module = ScaleKernel(RBFKernel(
            ard_num_dims=dim,
            lengthscale_constraint=GreaterThan(min_lengthscale),
        ))

        # Initialize to prior median
        self.covar_module.base_kernel.lengthscale = math.exp(ls_loc)
```

### Integration do FlowPO-HD

```python
# run_flowpo_hd.py --gp-type=vanilla
if config.gp_type == "vanilla":
    from flowpo_hd.improved_gp import create_vanilla_flow_guided_acquisition
    self.fga = create_vanilla_flow_guided_acquisition(config, manifold_keeper)
```

---

## DKL-10D Failure Analysis

DKL-10D mělo 0% coverage (VŠECHNY predikce mimo 90% CI).

**Příčina**: Neural network feature extractor overfit:
- NN komprimuje 1024D → 10D
- Komprese ztrácí informaci o nejistotě
- GP v 10D má příliš malou variance
- Výsledek: σ-error = 27.25 (cíl: ~1.0)

**Řešení**:
- Více dat pro trénink NN
- Regularizace (dropout, weight decay)
- Nebo použít VanillaGP přímo (bez NN)

---

## Recommendations

### Pro FlowPO-HD Optimalizaci

1. **Použij VanillaGP** místo SAAS
   - Lepší kalibrace (96% vs 73%)
   - Rychlejší fit (~2s vs ~30s)
   - Srovnatelný Spearman (0.80 vs 0.82)

2. **Nepoužívej DKL** s málo daty
   - NN overfit s <30 body
   - Potřeba min. 100+ bodů

3. **Trust region constraints**
   - Omeť exploration na okolí training dat
   - Penalizuj kandidáty příliš daleko

### Pro Budoucí Výzkum

1. **COWBOYS approach** (ICML 2025)
   - Nefit GP v latent space
   - Použij VAE jen pro generování kandidátů

2. **Multi-fidelity BO**
   - Využij low-fidelity evaluace
   - Korelace mezi fidelity levels

3. **Ensemble methods**
   - Kombinuj VanillaGP + SAAS
   - Uncertainty estimation přes ensemble

---

## Key References

1. **Hvarfner et al. 2024**: "Vanilla Bayesian Optimization Performs Great in High Dimensions" (ICML)
2. **Eriksson & Jankowiak 2021**: "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces" (SAAS)
3. **Eriksson et al. 2019**: "Scalable Global Optimization via Local Bayesian Optimization" (TuRBO)
4. **COWBOYS 2025**: "Return of the Latent Space COWBOYS" (ICML)
5. **Understanding HDBO 2025**: "Understanding High-Dimensional Bayesian Optimization"

---

## Files Created/Modified

- `flowpo_hd/improved_gp.py` - VanillaGP, DKL-10D, ImprovedSAAS
- `flowpo_hd/scripts/benchmark_gp_models.py` - LOOCV benchmark
- `flowpo_hd/scripts/run_flowpo_hd.py` - Integration (--gp-type=vanilla)
- `flowpo_hd/results/gp_benchmark_results.json` - Benchmark results

---

*Last updated: 2026-01-21*

---

## FlowDiT Checkpoint Comparison (2026-01-22)

### Available Checkpoints

| Checkpoint | Epoch | Step | Val Loss | Notes |
|------------|-------|------|----------|-------|
| `checkpoints_mega/best.pt` | 47 | 86,000 | **0.0724** | Best - longest training |
| `checkpoints_mega_aux2/best.pt` | 17 | 62,000 | 0.0820 | Second best |
| `checkpoints_mega_aux/best.pt` | 6 | 22,000 | 0.3190 | Worst - early stopping |

### FlowDiT Architecture

All checkpoints use **FlowDiT** architecture:
- `latent_dim=1024` (SONAR native)
- `hidden_dim=1024`
- `num_layers=4` (DiT blocks with AdaLN)
- `time_embed_dim=256`
- `mlp_ratio=2.0`
- ~42M parameters

### Generation Test Results (2026-01-22)

**CRITICAL FINDING: Mode Collapse**

All checkpoints suffer from mode collapse when generating from noise:

```
Seed 1: "So far as I am aware, I have asked myself if I am aware..."
Seed 2: "So far as I am aware, I have asked myself if I am aware..."
Seed 3: "So the question is: have you raised the question..."
```

**Key observations:**
1. Noise must be scaled to match training data norm (~0.26), not standard N(0,I) norm (~32)
2. Different random seeds produce nearly identical outputs → mode collapse
3. Standard noise (wrong scale) produces garbage: "elektron elektron jed jed..."

**Implications:**
- FlowDiT CANNOT be used for diverse instruction generation from noise
- FlowDiT projection also FAILS - collapses to same mode

### Projection Test Results

FlowDiT projection makes things WORSE:
- Original CosSim after perturbation: 0.30
- After FlowDiT projection: 0.06-0.11 (collapsed to "So far as I am aware...")

### Direct SONAR BO Feasibility (NO FlowDiT)

**Key finding: SONAR space is semantically smooth for small perturbations!**

| Noise Scale | CosSim | Result |
|-------------|--------|--------|
| 1% | 0.95 | Perfect preservation |
| 2% | 0.84 | Perfect preservation |
| 5% | 0.53 | Minor semantic changes, still valid |
| 10% | 0.28 | Semantics break |
| 20%+ | <0.16 | Garbage output |

**Interpolation works beautifully:**
```
α=0.00: "Let's think step by step."
α=0.50: "Let's see all steps clearly."  ← Natural blend!
α=1.00: "Show all calculations clearly."
```

**Recommendation for BO:**
- Trust region: **max 5% of embedding norm**
- NO FlowDiT projection needed
- Direct operation in SONAR space

---

## FlowDiT Retraining (2026-01-22)

### Root Cause of Mode Collapse

Original checkpoints were trained WITHOUT auxiliary losses:
- Only flow matching loss: MSE(v_pred, x_1 - x_0)
- No enforcement of semantic preservation
- Result: Model collapses to generating single mode

### Solution: Auxiliary Losses

New training script `flowpo_hd/scripts/train_flow_dit_aux.py` adds:

1. **Semantic Preservation Loss** (weight=0.5):
   ```
   L_sem = 1 - cos_sim(x_1, project(x_1 + noise))
   ```
   Forces model to preserve semantics when projecting perturbed embeddings

2. **Norm Preservation Loss** (weight=0.1):
   ```
   L_norm = MSE(||x_1||, ||project(x_1 + noise)||)
   ```
   Ensures projected embeddings have similar norm to input

### Training Run

Started: 2026-01-22 00:56
- Data: 1.8M unnormalized embeddings (norm ~0.26)
- Model: FlowDiT 40.4M params
- Batch size: 512
- LR: 1e-4
- Aux weights: semantic=0.5, norm=0.1

Monitoring: `tmux attach -t flowdit_aux`

### Test Results After ~4 Epochs (Step 14000)

**Generation WORKS!**
- 10/10 unique outputs from different seeds
- Mode collapse is FIXED
- Diverse topics: health, tourism, behavior, plans

**Projection DOES NOT WORK** (fundamental limitation):
- CosSim after projection is WORSE than before (0.53 → 0.24-0.37)
- Flow matching learns to map **interpolations** (noise↔data), not arbitrary perturbations
- Perturbed embeddings are NOT on the learned flow path
- The model doesn't know what to do with them

### Conclusion

**FlowDiT can be used for:**
- ✅ Generating diverse candidates from noise
- ✅ Sampling from instruction manifold

**FlowDiT CANNOT be used for:**
- ❌ Projecting perturbed embeddings back to manifold
- ❌ Correcting GP candidates

### Training Data Analysis

**mega_raw_encoded.pt** (1.8M samples):
- Very diverse: questions, facts, various text types
- Only ~21% have instruction-like patterns ("step by step", "calculate", etc.)
- FlowDiT learns broad text distribution, not math instructions

**ape_sonar_unnorm.pt** (1,645 samples):
- Focused on reasoning instructions
- Too small for FlowDiT training

### Final Conclusion

**FlowDiT is NOT suitable for GSM8K prompt optimization** because:
1. ❌ Cannot project arbitrary embeddings to manifold
2. ❌ Training data is too diverse (generates terrorism/religious text)
3. ❌ APE data is too small for flow matching

### Recommended Strategy for BO

**Use direct SONAR space optimization WITHOUT FlowDiT:**
1. GP operates directly in 1024D SONAR space
2. Use **small perturbations** (1-2% of norm) for exploration - these stay valid
3. Seed from high-quality instructions (warm start)
4. TuRBO trust region prevents straying too far from valid embeddings

This approach:
- ✅ Works with existing space_mapping data
- ✅ No FlowDiT dependency
- ✅ Perturbations up to 5% preserve semantics

---

## GP Optimization Run Log

### Run 1: Perturbation-based GP (2026-01-22)
- **Timestamp**: 2026-01-22 01:43
- **Script**: `run_gp_perturbation.py`
- **GP Type**: BetaHeteroscedasticGP with Hvarfner prior
- **Warm Start**: space_mapping_100x100.json (100 points)
- **Perturbation**: 2% of embedding norm
- **Temperature**: 0.5 (exploitation-focused)

**Key Innovation**: Instead of generating arbitrary candidates via TuRBO bounds:
1. Select parent embedding from training data (Thompson Sampling weighted)
2. Add perturbation (2% of norm) in random direction
3. CosSim to parent ≈ 0.9998 → stays semantically close

**Final Results**:
| Iteration | Accuracy | Notes |
|-----------|----------|-------|
| 1 | 80.0% | |
| 2 | 77.7% | |
| 3 | 61.2% | Drop |
| 4 | 74.4% | |
| 5 | 88.0% | Jump! |
| 6 | **88.1%** | **Best from optimization** |
| 7 | 86.1% | |
| 8 | 73.8% | |
| 9 | 84.7% | |
| 10 | 65.7% | |
| 11 | 86.5% | |
| 12 | 87.0% | |
| 13 | 83.6% | |
| 14 | 86.1% | |
| 15 | 87.7% | |
| 16 | 88.0% | |
| 17 | 87.7% | |
| 18 | 87.1% | |
| 19 | 87.7% | |
| 20 | - | Final |

**Summary**:
- Warm start best: **94.0%** (err=0.060)
- Best from optimization: **88.1%** (err=0.119, iteration 6)
- Mean accuracy: ~82%
- Total runtime: ~1 hour (20 iterations × ~3 min/iter)

**Key Observations**:
1. ✅ Instructions are valid text (not garbage like "I'm sorry, I'm sorry...")
2. ✅ Perturbation preserves semantic structure (CosSim ≈ 0.9998)
3. ✅ Consistent quality - all 20 iterations produced valid instructions
4. ⚠️ GP overconfident (predicting ~10% error, actual ~20%)
5. ⚠️ High variance in accuracy (61-88%) - typical for prompt engineering
6. ❌ Did NOT beat warm start best (94% > 88.1%)

**Why we didn't beat warm start**:
- Warm start already had excellent 94% accuracy instruction
- Perturbation-based exploration is local (2% of norm)
- Would need more iterations or larger perturbations to find better regions
- GP acquisition is overconfident → not exploring effectively

**Comparison to previous approaches**:
- Direct TuRBO (uniform sampling in bounds): Generated garbage ("I'm sorry, I'm sorry...")
- FlowDiT projection: Failed due to mode collapse
- Perturbation-based: **Works!** Valid instructions with reasonable accuracy

**Conclusion**:
The perturbation-based approach SOLVES the manifold problem in 1024D SONAR space:
1. Generate candidates as small perturbations of known-good embeddings
2. No need for FlowDiT projection
3. All generated instructions are valid and evaluable
4. Future work: larger perturbations, more iterations, better acquisition
