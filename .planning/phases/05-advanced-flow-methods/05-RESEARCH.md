# Phase 5: Advanced Flow Methods - Research

**Researched:** 2026-02-01
**Domain:** Rectified Flow (Reflow) and Stochastic Interpolants
**Confidence:** HIGH

## Summary

This phase implements two advanced flow matching methods: Rectified Flow with the reflow procedure (FLOW-03) and Stochastic Interpolants with learnable interpolation (FLOW-04). Both extend the baseline I-CFM/OT-CFM infrastructure from Phase 4.

**Rectified Flow** straightens flow trajectories through iterative refinement. The key insight is that after training an initial flow model, we can generate (x0, x1) pairs by sampling noise and integrating the ODE, then retrain on these synthetic pairs. This "reflow" procedure causally straightens paths, reducing ODE integration steps needed for sampling. Recent research (arXiv:2405.20320) shows a single reflow iteration is sufficient for nearly straight trajectories.

**Stochastic Interpolants** generalize flow matching by allowing different interpolation paths. While I-CFM uses linear interpolation x_t = (1-t)x_0 + tx_1, stochastic interpolants support trigonometric (GVP) or learnable paths. The SiT paper (Scalable Interpolant Transformers) found GVP interpolant (cos/sin schedule) outperforms linear on ImageNet, suggesting learnable schedules may improve SONAR flow quality.

**Primary recommendation:** Implement reflow as a distillation procedure using the existing OT-CFM checkpoint as teacher. For stochastic interpolants, start with fixed GVP schedule then optionally make coefficients learnable. Use existing coupling abstraction to add new coupling classes cleanly.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchcfm | 1.0.4+ | Base flow matching classes | Already in use for OT-CFM, supports rectified flow formulation |
| torch | 2.1+ | ODE integration for reflow pairs | Standard PyTorch, no new dependencies |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchdiffeq | 0.2.4+ | High-quality ODE solvers (dopri5) | For accurate reflow pair generation (optional) |
| lucidrains/rectified-flow-pytorch | latest | Reference implementation | For API inspiration only, don't import |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom reflow | lqiang67/rectified-flow | Their codebase is comprehensive but heavy; we want minimal extension |
| Euler integration | torchdiffeq dopri5 | Dopri5 more accurate for pair generation, but Euler sufficient for 100 steps |
| GVP schedule | Fully learnable alpha_t | GVP is proven effective; learnable adds complexity without proven SONAR benefit |

**Installation:**
```bash
# No new dependencies required - torchcfm already installed
# Optional for higher-accuracy reflow pairs:
uv add torchdiffeq>=0.2.4
```

## Architecture Patterns

### Recommended Project Structure
```
study/flow_matching/
├── coupling/
│   ├── __init__.py         # Add 'reflow' and 'si' to factory
│   ├── icfm.py              # Existing
│   ├── otcfm.py             # Existing
│   ├── reflow.py            # NEW: ReflowCoupling using teacher model
│   └── stochastic.py        # NEW: StochasticInterpolantCoupling (GVP schedule)
├── reflow/
│   ├── __init__.py          # Reflow utilities
│   ├── pair_generator.py    # Generate (x0, x1) pairs from teacher
│   └── distill.py           # Reflow training loop
├── schedules.py             # NEW: Interpolation schedules (linear, gvp, learnable)
├── trainer.py               # Already supports coupling injection
└── evaluate.py              # compute_path_straightness() already exists
```

### Pattern 1: Reflow Pair Generation

**What:** Generate synthetic (noise, endpoint) pairs from trained teacher model
**When to use:** Before reflow training
**Example:**
```python
# Source: Derived from gnobitab/RectifiedFlow and lucidrains/rectified-flow-pytorch
class ReflowPairGenerator:
    """Generate (x0, x1) pairs by integrating teacher ODE."""

    def __init__(self, teacher_model: nn.Module, n_steps: int = 100):
        self.teacher = teacher_model
        self.n_steps = n_steps

    @torch.no_grad()
    def generate_pairs(self, n_pairs: int, device: str = "cuda") -> tuple[Tensor, Tensor]:
        """Generate (noise, endpoint) pairs via ODE integration.

        Returns:
            x0: Random noise [n_pairs, dim]
            x1: ODE endpoints [n_pairs, dim]
        """
        # Sample noise
        x0 = torch.randn(n_pairs, 1024, device=device)

        # Integrate ODE from t=0 to t=1
        dt = 1.0 / self.n_steps
        x = x0.clone()
        for i in range(self.n_steps):
            t = torch.full((x.shape[0],), i / self.n_steps, device=device)
            v = self.teacher(x, t)
            x = x + dt * v

        x1 = x  # Synthetic endpoints
        return x0, x1
```

### Pattern 2: Reflow Coupling

**What:** Coupling class that uses generated pairs for straighter paths
**When to use:** Training 2-rectified flow
**Example:**
```python
# Source: Derived from rectified flow papers
class ReflowCoupling:
    """Coupling using pre-generated (x0, x1) pairs from teacher.

    Unlike I-CFM which pairs random noise with data,
    reflow pairs noise with its ODE endpoint, ensuring
    straighter paths by construction.
    """

    def __init__(self, pair_dataset: ReflowPairDataset):
        self.pairs = pair_dataset

    def sample(self, batch_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Sample from pre-generated pairs."""
        x0, x1 = self.pairs[batch_idx]

        # Sample time uniformly
        t = torch.rand(x1.shape[0], device=x1.device)

        # Linear interpolation (reflow uses straight paths)
        t_unsqueeze = t.unsqueeze(-1)
        x_t = (1 - t_unsqueeze) * x0 + t_unsqueeze * x1

        # Target velocity
        u_t = x1 - x0

        return t, x_t, u_t
```

### Pattern 3: Stochastic Interpolant with GVP Schedule

**What:** Generalized Variance Preserving interpolation (cos/sin schedule)
**When to use:** When experimenting with non-linear interpolation paths
**Example:**
```python
# Source: SiT paper (scalable-interpolant.github.io) and Albergo et al.
import math

class StochasticInterpolantCoupling:
    """Stochastic Interpolant with GVP (trigonometric) schedule.

    GVP schedule: alpha_t = cos(pi*t/2), sigma_t = sin(pi*t/2)
    x_t = alpha_t * x0 + sigma_t * x1

    This is variance-preserving and avoids singularities near endpoints.
    """

    def __init__(self, schedule: str = "gvp"):
        """
        Args:
            schedule: 'linear' or 'gvp' (generalized variance preserving)
        """
        self.schedule = schedule

    def get_alpha_sigma(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Get interpolation coefficients at time t."""
        if self.schedule == "gvp":
            # Trigonometric schedule: variance-preserving
            alpha_t = torch.cos(math.pi * t / 2)  # 1 -> 0
            sigma_t = torch.sin(math.pi * t / 2)  # 0 -> 1
        else:  # linear (same as I-CFM)
            alpha_t = 1 - t
            sigma_t = t
        return alpha_t, sigma_t

    def get_alpha_sigma_dot(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Get time derivatives of interpolation coefficients."""
        if self.schedule == "gvp":
            alpha_dot = -math.pi / 2 * torch.sin(math.pi * t / 2)
            sigma_dot = math.pi / 2 * torch.cos(math.pi * t / 2)
        else:  # linear
            alpha_dot = torch.full_like(t, -1.0)
            sigma_dot = torch.full_like(t, 1.0)
        return alpha_dot, sigma_dot

    def sample(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample t, x_t, u_t for stochastic interpolant training.

        Velocity target: u_t = alpha_dot * x0 + sigma_dot * x1
        """
        t = torch.rand(x1.shape[0], device=x1.device)

        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_dot, sigma_dot = self.get_alpha_sigma_dot(t)

        # Interpolation
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
        x_t = alpha_t * x0 + sigma_t * x1

        # Target velocity
        alpha_dot = alpha_dot.unsqueeze(-1)
        sigma_dot = sigma_dot.unsqueeze(-1)
        u_t = alpha_dot * x0 + sigma_dot * x1

        return t, x_t, u_t
```

### Pattern 4: Learnable Interpolation Coefficients

**What:** Make alpha_t, sigma_t parametric via MLP
**When to use:** Advanced ablation to learn optimal schedule
**Example:**
```python
# Source: Derived from SiT ablations and Neural Flow Diffusion
class LearnableSchedule(nn.Module):
    """Learnable interpolation schedule via MLP.

    Outputs alpha_t, sigma_t given t, with constraints:
    - alpha_0 = 1, sigma_0 = 0 (start at x0)
    - alpha_1 = 0, sigma_1 = 1 (end at x1)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # alpha, sigma
        )
        # Initialize to linear schedule
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Get alpha_t, sigma_t with boundary constraints."""
        out = self.net(t.unsqueeze(-1))
        delta_alpha, delta_sigma = out.unbind(-1)

        # Enforce boundaries: multiply by t*(1-t) to ensure zero at endpoints
        boundary_mask = t * (1 - t)

        # Base: linear schedule
        alpha_base = 1 - t
        sigma_base = t

        # Add learnable correction
        alpha_t = alpha_base + boundary_mask * delta_alpha
        sigma_t = sigma_base + boundary_mask * delta_sigma

        return alpha_t, sigma_t
```

### Anti-Patterns to Avoid

- **Don't generate all reflow pairs upfront for large datasets:** For 1K dataset, generating pairs online during training is more memory-efficient than storing 1M pairs
- **Don't use too few ODE steps for pair generation:** Use at least 100 steps for accurate endpoint estimation; 50 steps may introduce noise
- **Don't forget to freeze teacher during pair generation:** Teacher model should be in eval mode and not updated
- **Don't train reflow on original data:** Reflow training ONLY uses synthetic (noise, endpoint) pairs, not original data
- **Don't expect straighter paths from stochastic interpolants alone:** SI changes the velocity target, not path geometry; combine with OT coupling for straighter paths

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ODE integration for pairs | Custom RK4 | Euler 100 steps or torchdiffeq dopri5 | Edge cases in numerical stability |
| Straightness measurement | Custom metric | Existing compute_path_straightness() | Already validated in Phase 4 |
| Schedule derivatives | Manual calculus | PyTorch autograd on schedule functions | Avoids derivative bugs |
| Teacher model loading | Custom checkpoint parser | Existing load_checkpoint() | Consistent with training infrastructure |

**Key insight:** Reflow is conceptually simple (generate pairs, retrain) but implementation details matter. The existing FlowTrainer and coupling abstraction handle most complexity.

## Common Pitfalls

### Pitfall 1: Reflow Data Quality

**What goes wrong:** Reflow model doesn't converge or produces worse samples
**Why it happens:** Teacher model not fully converged, or too few ODE steps for pair generation
**How to avoid:**
- Use best.pt checkpoint from Phase 4 OT-CFM (lower val loss = better teacher)
- Generate pairs with at least 100 Euler steps
- Verify pair quality: x1 should look like valid SONAR embeddings
**Warning signs:** High training loss, generated pairs have unusual statistics

### Pitfall 2: Stochastic Interpolant Velocity Target

**What goes wrong:** Model trained with wrong velocity target
**Why it happens:** Confusing x1-x0 (I-CFM target) with alpha_dot*x0 + sigma_dot*x1 (SI target)
**How to avoid:**
- For SI with GVP: u_t = -pi/2*sin(pi*t/2)*x0 + pi/2*cos(pi*t/2)*x1
- Verify target velocity analytically matches interpolation derivative
**Warning signs:** Loss much higher than I-CFM baseline

### Pitfall 3: Schedule Numerical Stability

**What goes wrong:** NaN or divergence during training with learnable schedule
**Why it happens:** Schedule outputs extreme values or derivatives
**How to avoid:**
- Clamp alpha_t and sigma_t to [0, 1]
- Initialize learnable schedule to linear (zero corrections)
- Add boundary enforcement (alpha_0=1, sigma_0=0, alpha_1=0, sigma_1=1)
**Warning signs:** NaN loss, extreme gradient norms

### Pitfall 4: Single vs Multiple Reflow Iterations

**What goes wrong:** Wasted compute on multiple reflow iterations
**Why it happens:** Following older papers recommending 2-3 reflow rounds
**How to avoid:**
- Start with single reflow iteration (2-rectified flow)
- Recent research shows single iteration sufficient for nearly straight paths
- Only iterate if path straightness doesn't improve
**Warning signs:** Diminishing returns on path straightness after first reflow

### Pitfall 5: ODE Direction for Sampling

**What goes wrong:** SI model generates noise instead of data
**Why it happens:** Using wrong ODE direction for non-linear schedules
**How to avoid:**
- For linear: sample t=0->1 (noise->data) standard
- For GVP: same direction, but velocity formula differs
- Verify sampling produces valid embeddings before comparing methods
**Warning signs:** Generated samples look like noise, high distribution MSE

## Code Examples

### Complete Reflow Pipeline
```python
# Source: Derived from multiple implementations
from study.flow_matching.coupling import create_coupling
from study.flow_matching.evaluate import load_checkpoint

# 1. Load teacher model (best OT-CFM checkpoint)
teacher, stats = load_checkpoint(
    "study/checkpoints/mlp-otcfm-1k-none/best.pt",
    arch="mlp",
    device="cuda:1"
)
teacher.eval()

# 2. Generate reflow pairs
pair_generator = ReflowPairGenerator(teacher, n_steps=100)
pairs = []
for _ in tqdm(range(100)):  # Generate in batches
    x0, x1 = pair_generator.generate_pairs(n_pairs=100, device="cuda:1")
    pairs.append((x0.cpu(), x1.cpu()))

# 3. Create reflow dataset
x0_all = torch.cat([p[0] for p in pairs])
x1_all = torch.cat([p[1] for p in pairs])
reflow_dataset = TensorDataset(x0_all, x1_all)

# 4. Train 2-rectified flow
# Use ReflowCoupling with the dataset
reflow_model = create_model("mlp")
# Training loop uses ReflowCoupling.sample() instead of random pairing
```

### GVP Schedule Integration
```python
# Source: SiT paper + torchcfm patterns
from study.flow_matching.coupling import StochasticInterpolantCoupling

# Create coupling with GVP schedule
si_coupling = StochasticInterpolantCoupling(schedule="gvp")

# Training step
for x1 in dataloader:
    x0 = torch.randn_like(x1)
    t, x_t, u_t = si_coupling.sample(x0, x1)

    v_pred = model(x_t, t)
    loss = F.mse_loss(v_pred, u_t)

    # Note: u_t is NOT x1-x0 for GVP schedule!
    # u_t = alpha_dot*x0 + sigma_dot*x1 with trigonometric derivatives
```

### Sampling with Stochastic Interpolant
```python
# Source: Derived from SiT sampling
@torch.no_grad()
def sample_si(model, n_samples: int, n_steps: int, schedule: str = "gvp"):
    """Sample from trained stochastic interpolant model.

    For GVP schedule, we still integrate forward t=0->1,
    but the velocity field was trained with different targets.
    """
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 1024, device="cuda:1")

    for i in range(n_steps):
        t = torch.full((x.shape[0],), i / n_steps, device="cuda:1")
        v = model(x, t)
        x = x + dt * v

    return x  # Same sampling, just different trained velocity
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Multiple reflow iterations | Single reflow iteration | arXiv:2405.20320 (May 2024) | 75% FID improvement with less compute |
| Linear interpolation only | GVP/trigonometric schedules | SiT (2024) | 10-15% FID improvement on ImageNet |
| Fixed schedules | Learnable interpolation | Neural Flow Diffusion (2024) | Theoretical flexibility, unclear practical benefit |
| Independent reflow training | Piecewise rectified flow (PeRFlow) | NeurIPS 2024 | Online training without synthetic dataset |

**Deprecated/outdated:**
- Multiple reflow iterations (2-3 rounds): Single iteration is sufficient
- Pure linear interpolation: GVP often better, especially for images

## Open Questions

1. **Reflow benefit for SONAR space**
   - What we know: Reflow produces straighter paths on image data
   - What's unclear: Does SONAR (1024D embeddings) benefit similarly? Our Phase 4 paths are already very straight (~0.0016 deviation)
   - Recommendation: Implement and measure; if I-CFM paths already straight, benefit may be marginal

2. **GVP vs Linear for text embeddings**
   - What we know: GVP outperforms linear on ImageNet
   - What's unclear: Is this benefit transferable to SONAR text embeddings?
   - Recommendation: Ablate both schedules; text may have different optimal interpolation

3. **Learnable schedule overhead**
   - What we know: Adds ~65K parameters for schedule MLP
   - What's unclear: Is the flexibility worth the added complexity for our use case?
   - Recommendation: Implement as optional ablation, not default

4. **Number of reflow pairs needed**
   - What we know: 1M pairs recommended for CIFAR-10
   - What's unclear: What's needed for 1K/5K/10K SONAR dataset?
   - Recommendation: Start with 10x original dataset size, ablate if needed

## Sources

### Primary (HIGH confidence)
- [Rectified Flow Official Documentation](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html) - Mathematical formulation, reflow procedure
- [SiT: Scalable Interpolant Transformers](https://scalable-interpolant.github.io/) - GVP schedule, interpolant comparison
- [TorchCFM GitHub](https://github.com/atong01/conditional-flow-matching) - Verified support for rectified flow and stochastic interpolants
- Phase 4 Research (04-RESEARCH.md) - Existing OT-CFM infrastructure

### Secondary (MEDIUM confidence)
- [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) - PyTorch API patterns for Reflow class
- [gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) - Official ICLR implementation details
- [Improving Rectified Flow Training (arXiv:2405.20320)](https://arxiv.org/abs/2405.20320) - Single reflow sufficiency, U-shaped timestep distribution
- [Stochastic Interpolants Paper (arXiv:2303.08797)](https://arxiv.org/abs/2303.08797) - Theoretical framework
- [Building Normalizing Flows with Stochastic Interpolants](https://hunterheidenreich.com/notes/machine-learning/generative-models/stochastic-interpolants/) - Implementation notes

### Tertiary (LOW confidence)
- [malbergo/stochastic-interpolants](https://github.com/malbergo/stochastic-interpolants) - Simple reference implementation
- WebSearch results on learnable schedules - Limited practical guidance for SONAR domain

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing torchcfm, no new major dependencies
- Architecture: HIGH - Coupling abstraction allows clean extension
- Reflow implementation: HIGH - Well-documented procedure, multiple reference implementations
- Stochastic interpolants: MEDIUM - Theory clear, SONAR-specific tuning unknown
- Learnable schedules: LOW - Unclear benefit for text embeddings

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (stable domain, recent improvements incorporated)
