# Phase 4: Flow Matching Baselines - Research

**Researched:** 2026-02-01
**Domain:** Conditional Flow Matching (I-CFM, OT-CFM, CFG-Zero*)
**Confidence:** HIGH

## Summary

This phase implements two flow matching training methods: I-CFM (Independent Coupling) which is already implemented in Phase 3, and OT-CFM (Optimal Transport Coupling) which uses mini-batch Sinkhorn to create straighter flow paths. The key difference is how source-target pairs are coupled during training.

The existing `study/flow_matching/trainer.py` already implements I-CFM correctly with random noise-data pairing. OT-CFM requires integrating the `torchcfm` library's `ExactOptimalTransportConditionalFlowMatcher` or `OTPlanSampler` to compute optimal transport couplings before computing the velocity matching loss.

CFG-Zero* is already implemented in `rielbo/guided_flow.py` with 4% zero-guidance at initial steps. This schedule should be integrated into the evaluation pipeline for guided sampling.

**Primary recommendation:** Use `torchcfm.ExactOptimalTransportConditionalFlowMatcher` for OT-CFM training with `normalize_cost=True` and higher regularization (0.5) for numerical stability in 1024D SONAR space. Measure path straightness using per-sample trajectory variance.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchcfm | 1.0.4+ | OT-CFM and Sinkhorn coupling | Official TorchCFM library by Tong et al., provides `ExactOptimalTransportConditionalFlowMatcher` |
| POT (via torchcfm) | 0.9+ | Sinkhorn algorithm | Python Optimal Transport, used internally by torchcfm's OTPlanSampler |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchcfm.OTPlanSampler | - | Direct OT plan computation | When needing explicit control over coupling method ('exact' or 'sinkhorn') |
| torchcfm.SchrodingerBridgeConditionalFlowMatcher | - | Stochastic OT-CFM | If sigma > 0 is needed for regularization |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torchcfm | Facebook flow_matching | Facebook's library is newer but less focused on OT-CFM |
| Sinkhorn | Exact OT (Hungarian) | Exact OT is O(n^3) but more accurate; Sinkhorn is faster with soft matching |
| OTPlanSampler | Manual POT calls | OTPlanSampler handles edge cases and numerical stability |

**Installation:**
```bash
# Already in pyproject.toml
uv add torchcfm>=1.0.4
# POT is installed as torchcfm dependency
```

## Architecture Patterns

### Recommended Project Structure
```
study/flow_matching/
├── models/
│   ├── mlp.py           # SimpleMLP (existing)
│   ├── dit.py           # DiTVelocityNetwork (existing)
│   └── __init__.py      # create_model factory (existing)
├── coupling/
│   ├── __init__.py      # Coupling method factory
│   ├── icfm.py          # Independent coupling (random)
│   └── otcfm.py         # OT coupling (Sinkhorn)
├── trainer.py           # FlowTrainer (refactor for coupling injection)
├── train.py             # CLI (add --flow otcfm option)
├── evaluate.py          # Add path straightness metrics
└── config.py            # Add OT-CFM specific config
```

### Pattern 1: Coupling Abstraction

**What:** Abstract the noise-data coupling into interchangeable classes
**When to use:** When training with different CFM variants (I-CFM vs OT-CFM)
**Example:**
```python
# Source: torchcfm documentation
from torchcfm import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher

class ICFMCoupling:
    """Independent coupling: sample x0, x1 independently."""
    def __init__(self, sigma: float = 0.0):
        self.fm = ConditionalFlowMatcher(sigma=sigma)

    def sample(self, x0: torch.Tensor, x1: torch.Tensor):
        """Returns t, xt, ut for velocity matching."""
        return self.fm.sample_location_and_conditional_flow(x0, x1)

class OTCFMCoupling:
    """OT coupling: pair x0, x1 via mini-batch optimal transport."""
    def __init__(self, sigma: float = 0.0):
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    def sample(self, x0: torch.Tensor, x1: torch.Tensor):
        """Returns t, xt, ut with OT-reordered pairs."""
        return self.fm.sample_location_and_conditional_flow(x0, x1)
```

### Pattern 2: OT-CFM Training Loop

**What:** Training loop that uses OT coupling for straighter paths
**When to use:** OT-CFM training
**Example:**
```python
# Source: torchcfm/examples
from torchcfm import ExactOptimalTransportConditionalFlowMatcher

ot_fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

for batch_x1 in dataloader:
    # Sample noise
    x0 = torch.randn_like(batch_x1)

    # OT-CFM samples with optimal transport coupling
    # Internally reorders x0 to minimize transport cost
    t, xt, ut = ot_fm.sample_location_and_conditional_flow(x0, batch_x1)

    # Velocity prediction
    vt = model(xt, t)

    # Loss is same as I-CFM
    loss = F.mse_loss(vt, ut)
    loss.backward()
```

### Pattern 3: CFG-Zero* Guidance Schedule

**What:** Zero guidance for first 4% of ODE steps
**When to use:** Any guided sampling with flow matching
**Example:**
```python
# Source: CFG-Zero* paper (arXiv:2503.18886), existing rielbo/guided_flow.py
def get_guidance_lambda(step: int, total_steps: int,
                        guidance_strength: float,
                        zero_init_fraction: float = 0.04) -> float:
    """CFG-Zero* schedule: zero guidance for first 4% of steps."""
    zero_init_steps = max(1, int(zero_init_fraction * total_steps))
    if step < zero_init_steps:
        return 0.0
    return guidance_strength
```

### Pattern 4: Path Straightness Measurement

**What:** Measure trajectory variance to compare I-CFM vs OT-CFM
**When to use:** Evaluation and ablation studies
**Example:**
```python
def compute_path_straightness(model, x0: torch.Tensor, x1: torch.Tensor,
                               n_steps: int = 50) -> dict:
    """Measure how straight the flow trajectories are.

    Straighter paths = lower variance = easier ODE integration.
    OT-CFM should produce straighter paths than I-CFM.
    """
    dt = 1.0 / n_steps
    trajectories = []  # [n_steps+1, batch, dim]

    x = x0.clone()
    trajectories.append(x.clone())

    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v = model(x, t)
        x = x + v * dt
        trajectories.append(x.clone())

    trajectories = torch.stack(trajectories)  # [n_steps+1, batch, dim]

    # Ideal straight path: linear interpolation from x0 to x1
    # x_ideal_t = (1-t)*x0 + t*x1
    ts = torch.linspace(0, 1, n_steps + 1, device=x0.device).view(-1, 1, 1)
    ideal_path = (1 - ts) * x0.unsqueeze(0) + ts * x1.unsqueeze(0)

    # Path deviation: MSE from ideal straight line
    deviation = ((trajectories - ideal_path) ** 2).mean(dim=-1)  # [steps, batch]

    return {
        "mean_deviation": deviation.mean().item(),
        "max_deviation": deviation.max().item(),
        "path_variance": deviation.var(dim=0).mean().item(),  # variance along path
    }
```

### Anti-Patterns to Avoid

- **Don't mix coupling methods during training:** Use consistent coupling (I-CFM or OT-CFM) throughout training, not randomly switching
- **Don't use small regularization for Sinkhorn in high dimensions:** In 1024D SONAR space, use reg >= 0.5 and normalize_cost=True to avoid numerical instability
- **Don't forget to reorder both x0 and x1:** OT coupling reorders pairs; the velocity target (x1 - x0) must use the reordered pairs
- **Don't apply guidance at t=0:** CFG-Zero* zeros guidance for first 4% of steps; applying guidance too early corrupts trajectories

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Optimal transport coupling | Custom Sinkhorn loop | `torchcfm.ExactOptimalTransportConditionalFlowMatcher` | Handles numerical stability, batch size mismatch, fallback to uniform |
| Sinkhorn algorithm | Manual matrix scaling | `POT.sinkhorn` (via torchcfm) | Numerically stable, supports GPU, log-domain option |
| OT plan sampling | Random from plan | `OTPlanSampler.sample_plan()` | Correct multinomial sampling from transport plan |
| Cost matrix normalization | Manual normalization | `OTPlanSampler(normalize_cost=True)` | Prevents overflow in high dimensions |

**Key insight:** The torchcfm library handles many edge cases: plan sparsity, numerical precision, degenerate plans, and batch size matching. Rolling your own OT coupling will hit these issues.

## Common Pitfalls

### Pitfall 1: Sinkhorn Numerical Instability in High Dimensions

**What goes wrong:** Sinkhorn algorithm produces NaN or diverges in 1024D SONAR space with default regularization (reg=0.05)
**Why it happens:** Squared Euclidean distances grow with dimension; exp(-M/reg) underflows
**How to avoid:**
- Use `normalize_cost=True` in OTPlanSampler
- Increase regularization to reg >= 0.5
- Use log-domain Sinkhorn: `method='sinkhorn_log'` in POT
**Warning signs:** RuntimeWarning about divide by zero, UserWarning about numerical errors reverting to uniform plan

### Pitfall 2: Inconsistent Batch Sizes

**What goes wrong:** torchcfm's OT coupling fails when source and target batch sizes differ
**Why it happens:** OT plan is square matrix; needs same number of source and target samples
**How to avoid:** Always pair noise x0 with data x1 of same batch size; use `drop_last=True` in DataLoader
**Warning signs:** Shape mismatch errors in get_map() or sample_plan()

### Pitfall 3: Forgetting CFG-Zero* During Evaluation

**What goes wrong:** Guided sampling uses full guidance from step 0, producing poor samples
**Why it happens:** Early flow estimates are inaccurate; strong guidance corrupts trajectories
**How to avoid:** Always apply CFG-Zero* schedule with zero_init_fraction=0.04 during guided sampling
**Warning signs:** Guided samples are worse than unguided; samples drift off manifold

### Pitfall 4: Comparing I-CFM and OT-CFM with Different Velocity Targets

**What goes wrong:** I-CFM uses v = x1 - x0 with random pairing; OT-CFM must use reordered pairs
**Why it happens:** OT-CFM internally reorders x0 to minimize transport cost; the velocity target changes
**How to avoid:** Use torchcfm's `sample_location_and_conditional_flow()` which returns correctly computed ut
**Warning signs:** OT-CFM loss is higher than I-CFM (should be lower or equal)

### Pitfall 5: Not Accounting for OT Coupling Overhead

**What goes wrong:** OT-CFM training is much slower than expected
**Why it happens:** Sinkhorn is O(n^2) per batch; exact OT (Hungarian) is O(n^3)
**How to avoid:** Use moderate batch sizes (256-512); Sinkhorn is faster than exact for large batches
**Warning signs:** GPU utilization drops during OT computation (happens on CPU)

## Code Examples

### Complete I-CFM Training Step
```python
# Source: Existing study/flow_matching/trainer.py (verified)
def train_step_icfm(model, x1, optimizer):
    """Independent CFM training step."""
    # Sample noise (source distribution)
    x0 = torch.randn_like(x1)

    # Sample time uniformly
    t = torch.rand(x1.shape[0], device=x1.device)

    # Interpolate: x_t = (1-t)*x0 + t*x1
    t_unsqueeze = t.unsqueeze(-1)
    x_t = (1 - t_unsqueeze) * x0 + t_unsqueeze * x1

    # Target velocity: v = x1 - x0
    v_target = x1 - x0

    # Forward pass
    v_pred = model(x_t, t)

    # MSE loss
    loss = F.mse_loss(v_pred, v_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### Complete OT-CFM Training Step with torchcfm
```python
# Source: torchcfm documentation + numerical stability fixes
from torchcfm import ExactOptimalTransportConditionalFlowMatcher

# Initialize with default sigma=0.0 for deterministic OT
ot_fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

# For better stability in high-D, modify the internal OTPlanSampler
# ot_fm.ot_sampler = OTPlanSampler(method='sinkhorn', reg=0.5, normalize_cost=True)

def train_step_otcfm(model, x1, optimizer, ot_fm):
    """OT-CFM training step with optimal transport coupling."""
    # Sample noise (source distribution)
    x0 = torch.randn_like(x1)

    # OT-CFM: samples with optimal transport coupling
    # Returns t, x_t (interpolated), u_t (target velocity with OT-reordered pairs)
    t, x_t, u_t = ot_fm.sample_location_and_conditional_flow(x0, x1)

    # Forward pass
    v_pred = model(x_t, t)

    # MSE loss (same as I-CFM, but with OT-coupled target)
    loss = F.mse_loss(v_pred, u_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### CFG-Zero* Guided Sampling
```python
# Source: Existing rielbo/guided_flow.py (verified)
@torch.no_grad()
def sample_with_cfg_zero_star(
    model,
    n_samples: int,
    n_steps: int = 50,
    guidance_fn = None,  # Function that returns gradient of guidance objective
    guidance_strength: float = 1.0,
    zero_init_fraction: float = 0.04,
    device: str = "cuda",
) -> torch.Tensor:
    """Sample with CFG-Zero* schedule (zero guidance for first 4% of steps)."""
    dt = 1.0 / n_steps
    zero_init_steps = max(1, int(zero_init_fraction * n_steps))

    # Start from noise
    x = torch.randn(n_samples, 1024, device=device)

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)

        # Base velocity
        v = model(x, t)

        # Add guidance after zero-init period
        if guidance_fn is not None and i >= zero_init_steps:
            grad = guidance_fn(x)
            v = v + guidance_strength * grad

        # Euler step
        x = x + v * dt

    return x
```

### Path Straightness Evaluation
```python
# Source: Derived from OT-CFM paper (arXiv:2302.00482) concepts
@torch.no_grad()
def evaluate_path_straightness(
    model,
    test_x1: torch.Tensor,
    n_steps: int = 50,
    device: str = "cuda",
) -> dict:
    """Compare actual flow trajectories to ideal straight lines.

    OT-CFM should have lower deviation than I-CFM.
    """
    test_x1 = test_x1.to(device)
    x0 = torch.randn_like(test_x1)

    dt = 1.0 / n_steps
    model.eval()

    # Track trajectory
    x = x0.clone()
    trajectory = [x.clone()]

    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.clone())

    trajectory = torch.stack(trajectory)  # [n_steps+1, batch, dim]
    x1_reached = trajectory[-1]

    # Ideal straight path from actual x0 to reached x1
    ts = torch.linspace(0, 1, n_steps + 1, device=device).view(-1, 1, 1)
    ideal = (1 - ts) * x0.unsqueeze(0) + ts * x1_reached.unsqueeze(0)

    # Deviation from straight line
    deviation = ((trajectory - ideal) ** 2).sum(dim=-1).sqrt()  # [steps, batch]

    return {
        "mean_path_deviation": deviation.mean().item(),
        "max_path_deviation": deviation.max().item(),
        "final_mse": ((x1_reached - test_x1) ** 2).mean().item(),
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Independent coupling (I-CFM) | OT coupling (OT-CFM) | Tong et al. 2023 | Straighter paths, lower variance, faster sampling |
| Full CFG guidance | CFG-Zero* schedule | Fan et al. Mar 2025 | Better sample quality by zeroing early guidance |
| Exact OT (Hungarian) | Sinkhorn entropic OT | Standard practice | O(n^2) vs O(n^3), differentiable, GPU-friendly |
| Manual OT implementation | torchcfm library | 2023 | Production-ready, handles edge cases |

**Deprecated/outdated:**
- Naive random coupling: Replaced by OT-CFM for straighter paths
- Full guidance from t=0: Replaced by CFG-Zero* which zeros first 4%

## Open Questions

1. **Optimal Sinkhorn regularization for SONAR embeddings**
   - What we know: reg=0.5 with normalize_cost=True works without warnings
   - What's unclear: Is this optimal? Does higher reg hurt coupling quality?
   - Recommendation: Ablate reg in {0.1, 0.5, 1.0} and measure path straightness

2. **OT coupling computational overhead**
   - What we know: Sinkhorn runs on CPU via numpy/scipy in POT
   - What's unclear: How much does this slow training vs I-CFM?
   - Recommendation: Time per batch and report overhead percentage

3. **CFG-Zero* optimal zero-init fraction**
   - What we know: 4% is recommended default, existing code uses 0.04
   - What's unclear: Is 4% optimal for SONAR flow models?
   - Recommendation: Keep 4% as default, could ablate in Phase 6

## Sources

### Primary (HIGH confidence)
- torchcfm library: `ExactOptimalTransportConditionalFlowMatcher`, `OTPlanSampler` classes - verified working
- [POT documentation (Context7)](/pythonot/pot) - Sinkhorn algorithm solvers
- Existing `rielbo/guided_flow.py` - CFG-Zero* implementation verified
- Existing `study/flow_matching/trainer.py` - I-CFM implementation verified

### Secondary (MEDIUM confidence)
- [TorchCFM GitHub](https://github.com/atong01/conditional-flow-matching) - API documentation and examples
- [CFG-Zero* Paper](https://arxiv.org/abs/2503.18886) - Zero-init guidance schedule (4% default)
- [CFG-Zero* GitHub](https://github.com/WeichenFan/CFG-Zero-star) - Implementation details
- [Cambridge MLG Flow Matching Blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) - OT-CFM vs I-CFM explanation
- [ICLR 2025 Visual CFM Blog](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/) - Mini-batch OT explanation

### Tertiary (LOW confidence)
- WebSearch results on path straightness metrics - no specific implementations found, derived from paper descriptions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - torchcfm and POT verified working, APIs tested
- Architecture: HIGH - patterns derived from existing code and torchcfm docs
- Pitfalls: HIGH - Sinkhorn stability issues reproduced and fixed
- Path straightness metrics: MEDIUM - derived from paper concepts, no standard implementation

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (stable domain, libraries mature)
