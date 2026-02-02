# EcoFlow Architecture: Analysis & Improvements

## Executive Summary

This document tracks improvement opportunities for the EcoFlow spherical flow matching + Riemannian GP architecture for NeurIPS submission.

**Current Status (2026-02-02):**
- Spherical-OT U-Net trained and running
- Mode collapse bug FIXED (denormalization + unit sphere initialization)
- BO running with 50k LLM budget, generating diverse prompts

---

## Part 1: Remaining Technical Issues

### 1.1 Minibatch OT with Only 50 Sinkhorn Iterations

**Issue in `spherical.py:193`:**
```python
for _ in range(50):  # Sinkhorn iterations
```

For 1024D with reg=0.5, 50 iterations may not converge properly. Research shows you need `O(log(n)/ε²)` iterations for ε-convergence.

**Fix:** Add convergence check or increase to 100+ iterations.

### 1.2 No Exploitation of Best Embedding Neighborhood

Current approach: Pure GP-UCB optimization across entire space.

Missing: Local refinement around best-known embedding using `sample_from_best()` with perturbation.

**Implementation:**
```python
class ExploitationStrategy:
    def __init__(self, exploit_freq=3, perturbation_std=0.05):
        self.exploit_freq = exploit_freq

    def should_exploit(self, iteration: int) -> bool:
        return iteration % self.exploit_freq == 0
```

### 1.3 Single Candidate Per Iteration

Currently evaluating 1 candidate per iteration = 37 iterations for 50k budget.

**Improvement:** Batch evaluate top-K candidates (K=3-5) to better utilize LLM throughput.

---

## Part 2: High-Impact Improvements from Research

### 2.1 **Latent Space BO** ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**Core Insight:** Do BO in flow's noise space z ~ N(0, I), not embedding space x.

**Why it's the best approach:**
1. z-space is Gaussian by construction - GP works naturally
2. No weird manifold geometry to handle
3. Standard RBF/Matern kernels are designed for Gaussian spaces
4. Lengthscale = 1.0 is reasonable default (no MSR needed)

**Implementation:**
```python
class LatentSpaceBO:
    def __init__(self, flow, gp):
        self.flow = flow
        self.gp = gp  # GP operates in noise space z ~ N(0, I)

    def suggest(self):
        # GP proposes z in noise space (Gaussian - GP's home turf!)
        z_cand = self.gp.optimize_acquisition()
        # Transform z → x via flow ODE (forward only, fast!)
        x_cand = self.flow.sample(z_cand)
        return z_cand, x_cand

    def update(self, z, y):
        # Store z (not x!) - NO INVERSION NEEDED during BO
        self.Z_history.append(z)
        self.Y_history.append(y)
        self.gp.update(torch.stack(self.Z_history), torch.tensor(self.Y_history))
```

**Complexity:** Only initial inversion needed for warm start points. No inversion during BO loop.

### 2.2 **Gradient-Enhanced BO (GEBO)** ⭐⭐⭐⭐

**Advantage:** SONAR decoder is differentiable! Each evaluation gives (y, ∇y).

**Implementation (directional only - full gradient too expensive in 1024D):**
```python
class DirectionalGEBO:
    def __init__(self, n_directions=10):
        self.n_directions = n_directions

    def compute_directional_derivatives(self, x, y_grad):
        # Direction toward best point
        directions = torch.randn(self.n_directions, 1024)
        directions[0] = F.normalize(self.best_x - x, dim=-1)

        # Directional derivatives: d_j = <∇y, d_j>
        dir_derivs = (y_grad.unsqueeze(0) * directions).sum(dim=-1)
        return directions, dir_derivs
```

### 2.3 **TuRBO Trust Regions** ⭐⭐⭐⭐

**Issue:** Global GP in 1024D is inefficient - uncertainty is high everywhere.

**Solution:** Local trust regions that expand/shrink based on success.

```python
class GeodesicTrustRegion:
    """Trust region that respects spherical geometry."""
    def __init__(self, center, geodesic_radius=0.5):
        self.center = F.normalize(center, dim=-1)
        self.radius = geodesic_radius  # In radians

    def sample_in_region(self, n_samples):
        # Sample uniformly from spherical cap around center
        tangent = torch.randn(n_samples, 1024)
        tangent = tangent - (tangent @ self.center) * self.center
        tangent = F.normalize(tangent, dim=-1) * self.radius
        return self.exp_map(self.center, tangent)
```

### 2.4 **Flow Velocity as GP Prior** ⭐⭐⭐

**Insight:** Flow velocity at t=0 points toward data manifold. High ||v|| = far from data.

```python
class FlowPriorGP:
    def prior_mean(self, x):
        v = self.flow.velocity(x, t=0.1)
        # Negative: lower velocity = closer to data = better prior
        return -torch.norm(v, dim=-1)
```

### 2.5 **CFG-Zero* Optimized Scale** ⭐⭐⭐

**Paper:** [arxiv.org/abs/2503.18886](https://arxiv.org/abs/2503.18886)

Already partially implemented (zero_init_fraction, cutoff). Could add learned scale correction s*(t).

### 2.6 **Flow Map Distillation** ⭐⭐⭐

**Goal:** Distill 50-step flow to 2-4 step model for faster candidate generation.

```python
class FlowMapDistiller:
    def distill(self, noise_samples, num_epochs=100):
        with torch.no_grad():
            z_1_teacher = self.teacher.sample(z_0, num_steps=50)
        z_1_student = self.student(z_0)  # Single forward pass
        loss = F.mse_loss(z_1_student, z_1_teacher)
```

**Benefits:** 10-20x faster sampling → can generate 5000+ candidates per iteration.

### 2.7 **Metric Flow Matching** ⭐⭐

**Paper:** [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf)

Learn data-dependent metric that keeps interpolants on semantic manifold.

### 2.8 **Newton-BO for High-D** ⭐⭐

**Paper:** [arxiv.org/abs/2508.18423](https://arxiv.org/abs/2508.18423)

Use Newton's method with GP Hessian for local UCB optimization instead of gradient ascent.

---

## Part 3: Semantic Coherence Pipeline

### 3.1 Round-Trip Fidelity Filter (L2-r)

```python
def compute_coherence_score(self, embeddings: torch.Tensor) -> torch.Tensor:
    """Compute semantic coherence via encode-decode round-trip."""
    decoded_texts = self.decode(embeddings)
    re_encoded = self.encoder.encode(decoded_texts)
    l2_distances = (embeddings - re_encoded).norm(dim=-1)
    return l2_distances

def filter_coherent(self, embeddings, threshold=0.3):
    scores = self.compute_coherence_score(embeddings)
    mask = scores < threshold
    return embeddings[mask], scores[mask]
```

### 3.2 Integration with Optimization Loop

```python
def step(self):
    candidates = self.sampler.sample(n=512)
    coherent_candidates, scores = self.decoder.filter_coherent(candidates, threshold=0.3)
    ucb_values = self.gp.compute_acquisition(coherent_candidates)
    best_idx = ucb_values.argmax()
    # Evaluate only the best coherent candidate
```

---

## Part 4: Implementation Priority

### Week 1-2: Latent Space BO
- [ ] Implement flow inversion for initial 100 points
- [ ] Standard GP in z-space with Matern kernel
- [ ] Compare with current embedding-space GP

### Week 2-3: Exploitation Interleaving
- [ ] Add sample_from_best() perturbation mode
- [ ] Alternate exploration/exploitation every 3 iterations
- [ ] Batch evaluate top-K candidates

### Week 3-4: GEBO
- [ ] Compute directional derivatives via SONAR decoder
- [ ] GP with (y, dir_deriv) observations
- [ ] Compare sample efficiency vs standard GP

### Week 4+: Advanced Methods
- [ ] Flow Map Distillation for faster sampling
- [ ] TuRBO trust regions with geodesic balls
- [ ] Metric Flow Matching

---

## Part 5: Verification Plan

1. **Metric: Best accuracy after 25 evaluations**
   - Current baseline: ~0.835 (warm start)
   - Target: 0.86+ (statistically significant improvement)

2. **Ablation tests:**
   - Latent Space BO vs Embedding Space BO
   - Exploitation interleaving vs pure exploration
   - GEBO vs standard GP
   - L2-r coherence filtering on/off

3. **Statistical significance:**
   - 5+ seeds minimum
   - Report mean ± std
   - Wilcoxon signed-rank test vs baselines

---

## Sources

- [Metric Flow Matching (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf)
- [CFG-Zero* (March 2025)](https://arxiv.org/abs/2503.18886)
- [Flow Map Matching (NeurIPS 2025)](https://arxiv.org/abs/2406.07507)
- [REI for TuRBO (Dec 2024)](https://arxiv.org/html/2412.11456v1)
- [Newton-BO (Aug 2025)](https://arxiv.org/abs/2508.18423)
- [Standard GP is All You Need (ICLR 2025)](https://arxiv.org/abs/2402.02746)
- [TuRBO (NeurIPS 2019)](https://arxiv.org/abs/1910.01739)
- [SAASBO (NeurIPS 2021)](https://arxiv.org/abs/2103.00349)
