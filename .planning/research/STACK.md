# Stack Research: Flow Matching in 1024D SONAR Embeddings

**Project:** Flow Matching Architecture Study
**Researched:** 2026-01-31
**Target:** NeurIPS paper quality
**Hardware:** A5000 (24GB VRAM)
**Dataset:** 1K-10K SONAR embeddings (unnormalized)

---

## Executive Summary

For flow matching in high-dimensional embedding spaces (1024D SONAR), the recommended stack is:

- **Flow Framework:** Custom implementation with torchcfm as reference (not direct dependency)
- **ODE Solver:** Custom Euler/Heun (not torchdiffeq for inference; optional for training)
- **GP Surrogate:** BoTorch 0.16.1 + GPyTorch 1.14+ with MSR initialization
- **EMA:** ema-pytorch 0.7.9
- **Experiment Tracking:** Weights & Biases (wandb)
- **Core:** PyTorch 2.8+

**Key insight:** For 1K-10K samples, the velocity network architecture matters more than the flow matching framework. Keep the network small (6 layers, 512 hidden dim) to avoid overfitting.

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| PyTorch | 2.8+ | Core deep learning framework | HIGH |
| Python | 3.10+ | Required by most dependencies | HIGH |

**Rationale:** PyTorch 2.8 (August 2025) adds control flow operators, improved compilation, and better CUDA support. All major flow matching libraries require PyTorch 2.1+ minimum.

### Flow Matching

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| Custom impl | N/A | Velocity network + ODE sampling | HIGH |
| torchcfm | 1.0.7 (reference) | OT-CFM implementation reference | MEDIUM |

**Rationale:** Your existing RieLBO codebase already has a well-structured implementation:
- `velocity_network.py`: DiT-style AdaLN-Zero architecture
- `flow_model.py`: Custom Euler/Heun integration

For 1024D embeddings with 1K-10K samples, this is optimal because:
1. **Control over architecture:** Small datasets need careful regularization
2. **No framework overhead:** Direct PyTorch is faster for small-scale research
3. **GP-guided sampling compatibility:** Custom ODE integration allows gradient injection

**Do NOT use facebookresearch/flow_matching directly** for this project - it's designed for larger-scale image/text generation and adds unnecessary complexity.

### ODE Solvers

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| Custom Euler/Heun | N/A | Inference sampling (20-50 steps) | HIGH |
| torchdiffeq | 0.2.5 | Optional: training with adjoint | MEDIUM |
| torchode | 0.2.x | Alternative: parallel batch solving | LOW |

**Rationale:**

For inference (sampling), custom Euler/Heun with 20-50 steps is standard practice:
- Flow matching creates nearly straight trajectories
- Adaptive solvers add overhead without benefit
- CFG-Zero* recommends zeroing first 4% of steps

For training, you don't need ODE solvers - flow matching uses direct regression loss:
```python
# Flow matching training (no ODE needed)
t = torch.rand(batch_size)
x_t = (1 - t) * x_0 + t * x_1  # Linear interpolation
v_target = x_1 - x_0           # Target velocity
v_pred = model(x_t, t)         # Predicted velocity
loss = F.mse_loss(v_pred, v_target)
```

**torchdiffeq** (0.2.5): Only needed if you want:
- Continuous normalizing flow likelihood computation
- Adjoint method for memory-efficient backprop through ODE
- Adaptive step sizes during training

### Bayesian Optimization / GP Surrogate

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| BoTorch | 0.16.1 | Bayesian optimization framework | HIGH |
| GPyTorch | 1.14+ | GP implementation (BoTorch dependency) | HIGH |
| linear_operator | 0.6+ | Efficient kernel computations | HIGH |

**Rationale:** Your `gp_surrogate.py` already uses BoTorch correctly with:
- MSR initialization (ICLR 2025) for 1024D
- Matern-5/2 kernel with ARD
- LogNormal lengthscale prior scaled by sqrt(D)
- BAxUS random subspace projection option

This is the state-of-the-art approach. Recent research (Feb 2024, confirmed in 2025) shows standard GPs with Matern kernels and UCB can match specialized high-dimensional methods.

### Training Utilities

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| ema-pytorch | 0.7.9 | Exponential moving average | HIGH |
| torch.optim.AdamW | built-in | Optimizer | HIGH |
| torch.amp | built-in | Mixed precision (optional) | MEDIUM |

**EMA Configuration (recommended):**
```python
from ema_pytorch import EMA

ema = EMA(
    model,
    beta=0.9999,           # Decay rate
    update_after_step=100, # Warmup steps
    update_every=10,       # Save compute
)
```

**Learning Rate Schedule (recommended):**
```python
# Cosine annealing with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=1e-6
)
```

**Gradient Clipping (recommended):**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Experiment Tracking

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| wandb | latest | Experiment tracking, hyperparameter sweeps | HIGH |
| tensorboard | built-in | Lightweight alternative | MEDIUM |

**Rationale:**
- **wandb** is the standard for NeurIPS-quality research papers
- Easy team collaboration, artifact versioning, hyperparameter sweeps
- Better visualization than MLflow for generative model metrics

**Cost consideration:** wandb bills by tracked GPU hours. For a research project with 1K-10K samples, costs should be minimal (<$50/month).

---

## Detailed Library Analysis

### Flow Matching Frameworks Comparison

| Library | Pros | Cons | Recommendation |
|---------|------|------|----------------|
| **torchcfm** (1.0.7) | Battle-tested, OT-CFM support, good docs | PyTorch Lightning dependency, more complex than needed | Use as reference only |
| **facebookresearch/flow_matching** | Official Meta library, discrete+continuous | CC BY-NC license, overkill for embeddings | Not recommended |
| **rectified-flow-pytorch** (lucidrains) | Clean API, recent research integrated | Image-focused, less embedding support | Not recommended |
| **Custom implementation** | Full control, minimal dependencies | Requires understanding FM theory | **Recommended** |

**Your existing RieLBO implementation is well-structured.** The DiT-style AdaLN-Zero architecture in `velocity_network.py` follows best practices from Stable Diffusion 3 and SiT papers.

### OT-CFM vs Standard CFM

For small datasets (1K-10K), **OT-CFM (Optimal Transport)** can help:
- Creates straighter trajectories
- Faster inference with fewer ODE steps
- More stable training

Implementation in torchcfm:
```python
from torchcfm.optimal_transport import OTPlanSampler

# Mini-batch OT with Sinkhorn
ot_sampler = OTPlanSampler(method="sinkhorn", reg=0.05)
x0, x1 = ot_sampler.sample_plan(source, target)
```

**Caveat:** OT computation adds O(n^2) to O(n^3) overhead per batch. For batch_size=256, this is ~10ms per step. Worth it for straighter flows.

### CFG-Zero* for Guided Sampling

Your `guided_flow.py` should incorporate CFG-Zero* (March 2025):

```python
# CFG-Zero* implementation
def sample_with_cfg_zero_star(model, n_samples, guidance_scale, n_steps=50):
    x = torch.randn(n_samples, 1024)
    dt = 1.0 / n_steps
    zero_init_steps = int(0.04 * n_steps)  # 4% of steps

    for i in range(n_steps):
        t = i * dt

        # Zero-init: skip first 4% of steps
        if i < zero_init_steps:
            continue

        v_uncond = model(x, t, cond=None)
        v_cond = model(x, t, cond=condition)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        x = x + v * dt

    return x
```

---

## Architecture Recommendations for 1024D

### Velocity Network Sizing

For 1K-10K samples, **keep the network small**:

| Samples | hidden_dim | num_layers | Parameters | Rationale |
|---------|------------|------------|------------|-----------|
| 1K | 256 | 4 | ~2M | Prevent severe overfitting |
| 5K | 512 | 6 | ~8M | Your current default, good balance |
| 10K | 512-768 | 6-8 | ~15M | Can increase slightly |

Your current `VelocityNetwork` with hidden_dim=512, num_layers=6 is appropriate for 5K-10K samples.

### Regularization for Small Datasets

```python
# Recommended configuration
config = {
    "dropout": 0.1,           # Add dropout to MLP blocks
    "weight_decay": 1e-4,     # AdamW regularization
    "ema_decay": 0.9999,      # Use EMA model for inference
    "gradient_clip": 1.0,     # Prevent instability
    "warmup_steps": 100,      # LR warmup
}
```

### Normalization Strategy

**CRITICAL:** SONAR embeddings should NOT be normalized to unit variance.

Your flow model uses:
```python
# Current approach (correct for internal training)
mean = embeddings.mean(dim=0)
std = embeddings.std(dim=0)
normalized = (embeddings - mean) / std
```

For **unnormalized output** requirement, ensure:
```python
# Denormalize after generation
output = generated * std + mean
```

---

## Installation Commands

```bash
# Core dependencies
pip install torch>=2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Bayesian optimization
pip install botorch>=0.16.1  # Includes gpytorch, linear_operator

# Training utilities
pip install ema-pytorch>=0.7.9
pip install wandb

# Optional: ODE solvers for likelihood computation
pip install torchdiffeq>=0.2.5

# Optional: torchcfm for reference/OT utilities
pip install torchcfm>=1.0.7

# Development
pip install pytest black ruff
```

---

## What NOT to Use

### 1. facebookresearch/flow_matching

**Why not:**
- CC BY-NC license (non-commercial) - problematic for some research
- Designed for image/text generation at scale
- Adds unnecessary abstraction for embedding spaces
- No significant advantage over custom implementation for 1024D

### 2. Diffusers library

**Why not:**
- Focused on diffusion, not flow matching
- Image-centric (VAE, UNet architecture)
- Overkill for embedding space optimization

### 3. scipy.integrate for ODE solving

**Why not:**
- CPU only, no GPU support
- Cannot batch process samples
- No gradient support for training
- 10-100x slower than PyTorch for inference

### 4. MLflow for experiment tracking

**Why not:**
- Requires self-hosting infrastructure
- Limited visualization compared to wandb
- Weaker hyperparameter sweep support
- Not standard in NeurIPS-tier publications

### 5. timm.utils.ModelEmaV2

**Why not:**
- Tied to timm ecosystem
- ema-pytorch is more flexible and actively maintained
- timm EMA designed for image classification, not generative models

### 6. Overparameterized networks

**Why not:**
- 1K-10K samples will overfit quickly
- hidden_dim > 768 not recommended
- num_layers > 8 not recommended
- Total parameters should stay under 20M

### 7. torchdiffeq for inference

**Why not:**
- Flow matching creates straight paths
- Adaptive solvers add overhead without benefit
- Custom Euler/Heun with 50 steps is standard
- CFG-Zero* requires manual step control anyway

---

## Confidence Assessment

| Component | Confidence | Verification |
|-----------|------------|--------------|
| PyTorch 2.8+ | HIGH | Official PyTorch release notes |
| BoTorch 0.16.1 | HIGH | PyPI verified, active development |
| GPyTorch 1.14+ | HIGH | BoTorch dependency chain verified |
| ema-pytorch 0.7.9 | HIGH | PyPI verified, lucidrains maintained |
| Custom ODE solvers | HIGH | Standard practice in flow matching literature |
| torchcfm 1.0.7 | MEDIUM | PyPI verified, but recommend custom impl |
| wandb over MLflow | MEDIUM | Community preference, not universal |
| Network sizing | MEDIUM | Based on general deep learning principles, not FM-specific papers |

---

## Sources

### Primary (HIGH confidence)
- [BoTorch Documentation](https://botorch.org/docs/overview/)
- [PyTorch 2.8 Release Blog](https://pytorch.org/blog/pytorch-2-8/)
- [torchcfm GitHub](https://github.com/atong01/conditional-flow-matching)
- [CFG-Zero* Paper](https://arxiv.org/abs/2503.18886)
- [SONAR GitHub](https://github.com/facebookresearch/SONAR)

### Secondary (MEDIUM confidence)
- [Flow Matching Cambridge Blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- [Standard GP in High Dimensions](https://arxiv.org/html/2402.02746v3)
- [OT-CFM Paper](https://arxiv.org/abs/2302.00482)
- [DiT Architecture](https://www.lightly.ai/blog/diffusion-transformers-dit)

### Tertiary (LOW confidence, WebSearch only)
- [Weighted CFM Paper](https://arxiv.org/html/2507.22270v2) - July 2025
- [DeepFlow ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Shin_Deeply_Supervised_Flow-Based_Generative_Models_ICCV_2025_paper.pdf)
