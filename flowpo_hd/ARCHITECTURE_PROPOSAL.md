# FlowPO-HD Architecture Proposal: Two-Phase Space-Mapping + Local BO

## Problem Analysis

Current warm-start data (26 points) is clustered in a tiny region:
- Pairwise distances: 0.05 - 0.35
- GP lengthscale: 78 (from dimension-scaled prior)
- BO exploration distance: ~3.6

**Result**: GP thinks it's interpolating but actually extrapolating → 10-133σ errors

## Proposed Architecture

### Phase 1: Space Mapping (Exploration)

**Goal**: Map the SONAR prompt space with diverse evaluations

```python
class SpaceMapper:
    """Generate diverse prompts to cover SONAR space."""

    def __init__(self, sonar_encoder, manifold_keeper, n_prompts=100):
        self.n_prompts = n_prompts

    def generate_diverse_prompts(self):
        prompts = []

        # 1. Diverse selection from APE instructions (30 prompts)
        ape_diverse = self._diverse_select_ape(30)
        prompts.extend(ape_diverse)

        # 2. LLM-generated variations (30 prompts)
        llm_variations = self._generate_llm_variations(30)
        prompts.extend(llm_variations)

        # 3. Sobol sampling + ManifoldKeeper projection (40 prompts)
        sobol_prompts = self._sobol_manifold_sample(40)
        prompts.extend(sobol_prompts)

        return prompts

    def _diverse_select_ape(self, n):
        """Select maximally diverse prompts from APE using farthest point sampling."""
        # Load APE embeddings
        # Use farthest point sampling to select n diverse points
        pass

    def _generate_llm_variations(self, n):
        """Use LLM to generate variations of existing prompts."""
        # For each seed prompt:
        # - "Rewrite this instruction in a different style: {prompt}"
        # - "Make this instruction more/less detailed: {prompt}"
        pass

    def _sobol_manifold_sample(self, n):
        """Sample Sobol points in SONAR space, project to manifold."""
        # 1. Generate Sobol points in normalized SONAR bounds
        # 2. Use ManifoldKeeper to project to instruction manifold
        # 3. Decode to text with SONAR decoder
        pass

    def evaluate_low_fidelity(self, prompts, n_examples=100):
        """Quick evaluation for ranking."""
        # Evaluate each prompt on n_examples
        # ~15s per prompt with vLLM
        # Returns rough error rates (std ~ 5%)
        pass
```

### Phase 2: Local BO (Exploitation)

**Goal**: Optimize within promising regions using TuRBO

```python
class TuRBOOptimizer:
    """Trust-region BO for local optimization."""

    def __init__(self, initial_data, trust_region_size=0.5):
        self.trust_region_size = trust_region_size
        self.success_count = 0
        self.fail_count = 0

    def optimize(self, n_iterations=20):
        for i in range(n_iterations):
            # 1. Fit local GP within trust region
            local_gp = self._fit_local_gp()

            # 2. Optimize acquisition within trust region
            candidate = self._optimize_in_trust_region(local_gp)

            # 3. Evaluate (high fidelity)
            error_rate = self._evaluate(candidate, n_examples=1319)

            # 4. Update trust region
            self._update_trust_region(error_rate)

    def _update_trust_region(self, error_rate):
        """Adapt trust region based on success/failure."""
        if error_rate < self.best_error_rate:
            # Success - expand trust region
            self.success_count += 1
            self.fail_count = 0
            if self.success_count >= 3:
                self.trust_region_size *= 2
                self.success_count = 0
        else:
            # Failure - shrink trust region
            self.fail_count += 1
            self.success_count = 0
            if self.fail_count >= 3:
                self.trust_region_size /= 2
                self.fail_count = 0

        # Restart if trust region too small
        if self.trust_region_size < 0.01:
            self._restart_from_promising_region()
```

### Multi-Fidelity Strategy

| Phase | Fidelity | Examples | Time/eval | Purpose |
|-------|----------|----------|-----------|---------|
| 1 (Mapping) | Low | 100-200 | ~15s | Space coverage, ranking |
| 2 (BO) | High | 1319 | ~120s | Precise optimization |

**Key insight**: Low fidelity is sufficient for finding promising regions.
High fidelity is only needed for final optimization.

## Expected Benefits

1. **Better GP calibration**: Training data covers the space, not just one cluster
2. **Correct trust region sizing**: Based on actual data spread, not lengthscale
3. **Efficient evaluation budget**: Cheap exploration, expensive exploitation
4. **Local GP advantage**: No extrapolation, only interpolation within trust region

## Implementation Plan

1. **SpaceMapper class**: Generate and evaluate diverse prompts
2. **TuRBOOptimizer class**: Local BO with adaptive trust regions
3. **Multi-fidelity evaluator**: Switch between low/high fidelity
4. **Integration**: Combine with existing ManifoldKeeper and SONAR

## References

- [TuRBO: Trust Region Bayesian Optimization](https://botorch.org/docs/tutorials/turbo_1/)
- [Regional Expected Improvement (REI) 2024](https://arxiv.org/abs/2412.11456)
- [Vanilla BO in High Dimensions (Hvarfner 2024)](https://arxiv.org/abs/2402.02229)
