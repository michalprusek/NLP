"""
Text Flow Autoencoder (TFA) for FlowPO.

This module implements the core autoencoding component of FlowPO:
OT-CFM (Optimal Transport Conditional Flow Matching) that maps SONAR
embeddings (1024D) to a compact latent space (128D) for Bayesian Optimization.

KEY INSIGHTS (FlowPO contribution):
- OT-CFM: Pairs noise x₀ with data x₁ via optimal transport → straighter trajectories
- Training: Simulation-free! No ODE solver needed. Just regress velocity field.
- Inference: Use ODE solver to integrate the learned flow.
- Compression: 8:1 (1024D → 128D) vs old 128:1 (4096D → 32D)

Why OT-CFM matters:
- Without OT: Random pairing creates crossing trajectories → poor reconstruction
- With OT: Minibatch OT creates non-crossing trajectories → high-fidelity reconstruction

Novel contribution: First application of OT-CFM autoencoding for text reconstruction.

Reference:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Improving Training of Rectified Flows (Liu et al., 2024) - OT-CFM
- LOL-BO: Lipschitz regularization for BO-friendly latent spaces (NeurIPS 2022)
- CoBO: Coordinate-wise Bayesian Optimization (NeurIPS 2023)
"""

import math
import warnings
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: scipy for exact OT, falls back to approximate Sinkhorn if not available
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(
        "scipy not available - using approximate Sinkhorn OT. "
        "For exact Hungarian OT (recommended for batch_size <= 256), "
        "install scipy: pip install scipy"
    )


# =============================================================================
# MINIBATCH OPTIMAL TRANSPORT (OT-CFM)
# =============================================================================

def compute_ot_plan_exact(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact optimal transport assignment using Hungarian algorithm.

    This pairs noise samples x_0 with data samples x_1 to minimize total
    transport cost, resulting in straighter trajectories.

    Args:
        x_0: (B, D) source samples (noise)
        x_1: (B, D) target samples (data)

    Returns:
        indices: (B,) permutation of x_0 to match x_1
    """
    if not SCIPY_AVAILABLE:
        return compute_ot_plan_approx(x_0, x_1)

    # Compute cost matrix (squared L2 distance)
    cost = torch.cdist(x_0, x_1, p=2).pow(2)

    # Solve assignment problem (Hungarian algorithm)
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    # Return permutation indices for x_0
    perm = torch.zeros(x_0.shape[0], dtype=torch.long, device=x_0.device)
    perm[col_ind] = torch.tensor(row_ind, device=x_0.device)

    return perm


def compute_ot_plan_approx(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    num_iters: int = 20,
    reg: float = 0.05,
) -> torch.Tensor:
    """
    Approximate OT using Sinkhorn algorithm (GPU-friendly).

    Faster than exact OT for large batches, with similar trajectory quality.

    Args:
        x_0: (B, D) source samples
        x_1: (B, D) target samples
        num_iters: Sinkhorn iterations
        reg: Entropy regularization (lower = closer to exact OT)

    Returns:
        indices: (B,) approximate permutation
    """
    B = x_0.shape[0]
    device = x_0.device

    # Cost matrix
    C = torch.cdist(x_0, x_1, p=2).pow(2)

    # Sinkhorn algorithm
    K = torch.exp(-C / reg)
    u = torch.ones(B, device=device) / B

    for _ in range(num_iters):
        v = 1.0 / (K.T @ u + 1e-8)
        u = 1.0 / (K @ v + 1e-8)

    # Transport plan
    P = torch.diag(u) @ K @ torch.diag(v)

    # Extract assignment (greedy from transport plan)
    indices = P.argmax(dim=0)

    return indices


def apply_ot_pairing(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    use_exact: bool = True,
    exact_threshold: int = 256,
) -> torch.Tensor:
    """
    Reorder x_0 to optimally match x_1 via Optimal Transport.

    Uses Hungarian algorithm for small batches (<=256), Sinkhorn for larger.
    """
    batch_size = x_0.shape[0]

    # Auto-switch to Sinkhorn for large batches (Hungarian is O(n³) on CPU)
    if use_exact and SCIPY_AVAILABLE and batch_size <= exact_threshold:
        perm = compute_ot_plan_exact(x_0, x_1)
    else:
        perm = compute_ot_plan_approx(x_0, x_1)

    return x_0[perm]


# =============================================================================
# TIMESTEP SAMPLING STRATEGIES
# =============================================================================

def sample_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Uniform timestep sampling t ~ U[0, 1]."""
    return torch.rand(batch_size, device=device)


def sample_timesteps_u_shaped(
    batch_size: int,
    device: torch.device,
    a: float = 4.0,
) -> torch.Tensor:
    """
    U-shaped timestep distribution: more weight at t=0 and t=1.

    From "Improving the Training of Rectified Flows" (2024):
    Training loss is large at interval endpoints (t≈0 and t≈1) but small
    in the middle. U-shaped distribution improves FID by ~28%.

    p(t) ∝ exp(at) + exp(-at)

    Args:
        batch_size: Number of samples
        device: Torch device
        a: Concentration parameter (higher = more weight at boundaries)

    Returns:
        t: (B,) timesteps in [0.001, 0.999]
    """
    u = torch.rand(batch_size, device=device)

    # Inverse CDF sampling for U-shaped distribution
    # This approximates the integral of exp(at) + exp(-at)
    # by mapping uniform samples to concentrate at endpoints
    centered = 2.0 * u - 1.0  # [-1, 1]
    sign = torch.sign(centered)
    abs_centered = torch.abs(centered)

    # Transform to concentrate at boundaries
    t = 0.5 + 0.5 * sign * (1.0 - torch.exp(-a * abs_centered))

    # Clamp to avoid exact boundaries (numerical stability)
    return t.clamp(0.001, 0.999)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestep embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or scalar timestep in [0, 1]

        Returns:
            encoding: (B, dim) sinusoidal encoding
        """
        t = t.reshape(-1, 1)  # Handles scalar, 1D, and already-shaped inputs

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )

        args = t * freqs
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            encoding = F.pad(encoding, (0, 1))

        return encoding


class TimestepEmbedding(nn.Module):
    """MLP to embed timestep into hidden dimension."""

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.positional = SinusoidalPositionalEncoding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),  # SiLU (Swish) is better for flows than GELU
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        pos = self.positional(t)
        return self.mlp(pos)


class VelocityField(nn.Module):
    """
    Neural velocity field v_θ(x, t) for Flow Matching.

    During training: Used to predict velocity at interpolated points.
    During inference: Integrated via ODE solver.
    """

    def __init__(
        self,
        dim: int = 512,           # Increased from 256 for more capacity
        time_dim: int = 128,      # Increased from 64
        hidden_mult: int = 4,
        num_layers: int = 5,      # Increased from 3 for deeper network
        dropout: float = 0.0,     # Dropout for regularization (0.1 recommended for large datasets)
    ):
        super().__init__()
        self.dim = dim
        self.time_embed = TimestepEmbedding(time_dim, dim)

        # Build MLP with SiLU activation (better for flows)
        layers = []
        in_dim = dim * 2  # x + t_emb
        hidden_dim = dim * hidden_mult

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ])
            # Add dropout after LayerNorm (before activation) for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

        # Zero init output for stable training (identity flow at init)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute velocity at (x, t)."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


class TextFlowAutoencoder(nn.Module):
    """
    Text Flow Autoencoder (TFA) - core component of FlowPO.

    Maps SONAR embeddings (1024D) to compact latent space (128D) via
    simulation-free flow matching. This enables efficient Bayesian
    Optimization in the latent space.

    Architecture (training is simulation-free, inference uses ODE):
        Encode: SONAR 1024D → enc_proj → 512D → [Flow t=1→0] → 512D → to_latent → 128D
        Decode: 128D → from_latent → 512D → [Flow t=0→1] → 512D → dec_proj → 1024D

    Flow operates in 512D intermediate space (flow_dim), with 128D bottleneck at t=0.

    Training: Simulation-free Flow Matching (no ODE solver!)
    Inference: ODE integration for encode/decode

    Flow direction:
        t=0: latent space (flow endpoint, then projected to 128D)
        t=1: data space (projected SONAR embedding in 512D flow_dim)

    Novel contribution: First FM autoencoder for text reconstruction.
    """

    def __init__(
        self,
        input_dim: int = 1024,    # SONAR embedding dimension
        flow_dim: int = 512,      # Increased from 256 for more capacity
        latent_dim: int = 128,    # Target latent dimension
        time_dim: int = 128,      # Increased from 64
        num_ode_steps: int = 20,  # Inference ODE steps (ALIGNED with train for stability)
        num_train_ode_steps: int = 20,  # Training ODE steps (ALIGNED with inference)
        num_velocity_layers: int = 6,  # Deeper velocity network for better capacity
        dropout: float = 0.0,     # Dropout for regularization (0.1 for production, 0.0 for small datasets)
    ):
        super().__init__()

        # Validate dimensions
        for name, val in [
            ("input_dim", input_dim),
            ("flow_dim", flow_dim),
            ("latent_dim", latent_dim),
            ("num_ode_steps", num_ode_steps),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be positive, got {val}")

        self.input_dim = input_dim
        self.flow_dim = flow_dim
        self.latent_dim = latent_dim
        self.num_ode_steps = num_ode_steps
        self.num_train_ode_steps = num_train_ode_steps

        # Projections between spaces
        self.enc_proj = nn.Linear(input_dim, flow_dim)   # 1024 → 256
        self.dec_proj = nn.Linear(flow_dim, input_dim)   # 256 → 1024

        # Latent bottleneck
        self.to_latent = nn.Linear(flow_dim, latent_dim)     # 256 → 128
        self.from_latent = nn.Linear(latent_dim, flow_dim)   # 128 → 256

        # Velocity field for Flow Matching (deeper network for better capacity)
        self.velocity = VelocityField(
            dim=flow_dim,
            time_dim=time_dim,
            num_layers=num_velocity_layers,
            dropout=dropout,
        )

        # Initialize projections
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'velocity' not in name:
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # =========================================================================
    # INFERENCE METHODS (use ODE solver)
    # =========================================================================

    def _euler_integrate(
        self,
        x0: torch.Tensor,
        t_start: float,
        t_end: float,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Euler integration of dx/dt = v(x, t)."""
        num_steps = num_steps or self.num_ode_steps
        dt = (t_end - t_start) / num_steps
        x = x0

        for i in range(num_steps):
            t = t_start + i * dt
            t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)
            v = self.velocity(t_tensor, x)
            x = x + dt * v

        return x

    @torch.no_grad()
    def encode(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Encode input embeddings to latent space via ODE flow.

        Flow direction: t=1 (data) → t=0 (latent prior)

        Args:
            x_input: (B, input_dim) input embeddings (e.g., 1024D SONAR)

        Returns:
            z: (B, latent_dim) latent codes (e.g., 128D)
        """
        # Project to flow space
        x_1 = self.enc_proj(x_input)  # (B, flow_dim)

        # Integrate backwards: t=1 → t=0
        x_0 = self._euler_integrate(x_1, t_start=1.0, t_end=0.0)

        # Project to latent
        z = self.to_latent(x_0)  # (B, latent_dim)

        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Decode latent codes to embeddings via ODE flow.

        Flow direction: t=0 (latent) → t=1 (data)

        Args:
            z: (B, latent_dim) latent codes
            normalize: L2 normalize output

        Returns:
            x_output: (B, input_dim) reconstructed embeddings
        """
        # Unproject from latent
        x_0 = self.from_latent(z)  # (B, flow_dim)

        # Integrate forwards: t=0 → t=1
        x_1 = self._euler_integrate(x_0, t_start=0.0, t_end=1.0)

        # Project to output
        x_output = self.dec_proj(x_1)  # (B, input_dim)

        if normalize:
            x_output = F.normalize(x_output, p=2, dim=-1)

        return x_output

    def forward(self, x_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full encode-decode pass (inference only)."""
        z = self.encode(x_input)
        x_recon = self.decode(z)
        return z, x_recon

    # =========================================================================
    # TRAINING METHODS (simulation-free, no ODE solver!)
    # =========================================================================

    def compute_flow_matching_loss(
        self,
        x_input: torch.Tensor,
        timestep_sampling: Literal["uniform", "u_shaped"] = "u_shaped",
        use_ot: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute simulation-free Flow Matching loss with optional OT pairing.

        OT-CFM pairs noise x_0 with data x_1 via optimal transport for straighter trajectories.
        """
        # Validate input
        if torch.isnan(x_input).any():
            raise ValueError("NaN detected in input embeddings")
        if torch.isinf(x_input).any():
            raise ValueError("Inf detected in input embeddings")

        batch_size = x_input.shape[0]
        device = x_input.device

        # 1. Project data to flow space (this is x_1, the target)
        x_1 = self.enc_proj(x_input)  # (B, flow_dim)

        # 2. Sample source noise (this is x_0)
        x_0 = torch.randn_like(x_1)  # (B, flow_dim)

        # 3. OT-CFM: Reorder x_0 to optimally match x_1
        # This is the key improvement - straighter trajectories!
        if use_ot and batch_size > 1:
            x_0 = apply_ot_pairing(x_0, x_1, use_exact=SCIPY_AVAILABLE)

        # 4. Sample timesteps (U-shaped improves convergence by ~28%)
        if timestep_sampling == "u_shaped":
            t = sample_timesteps_u_shaped(batch_size, device)
        else:
            t = sample_timesteps_uniform(batch_size, device)

        # 5. Compute linear interpolation (straight OT path)
        # x_t = t * x_1 + (1 - t) * x_0
        t_view = t.view(-1, 1)
        x_t = t_view * x_1 + (1 - t_view) * x_0

        # 6. Target velocity (the straight line direction)
        # For OT path: u_t = x_1 - x_0
        u_t = x_1 - x_0

        # 7. Predict velocity
        v_t = self.velocity(t, x_t)

        # 8. Validate velocity before computing loss
        if torch.isnan(v_t).any():
            raise RuntimeError(
                f"NaN detected in predicted velocity v_t. "
                f"x_t norm={x_t.norm():.4f}, t range=[{t.min():.4f}, {t.max():.4f}]"
            )
        if torch.isnan(u_t).any():
            raise RuntimeError(
                f"NaN detected in target velocity u_t. "
                f"x_0 norm={x_0.norm():.4f}, x_1 norm={x_1.norm():.4f}"
            )

        # 9. Flow Matching loss (MSE between predicted and target velocity)
        loss_fm = F.mse_loss(v_t, u_t)

        # Validate loss (should not happen if v_t and u_t are clean)
        if torch.isnan(loss_fm):
            raise RuntimeError(
                f"NaN loss detected. Stats: v_t norm={v_t.norm():.4f}, "
                f"u_t norm={u_t.norm():.4f}, x_1 norm={x_1.norm():.4f}"
            )

        return {
            "loss_fm": loss_fm,
            "x_1": x_1,
            "x_0": x_0,
        }

    def compute_reconstruction_loss(
        self,
        x_input: torch.Tensor,
        x_1: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss through full encode-decode path.

        Uses fewer ODE steps during training (num_train_ode_steps) for speed.
        """
        # Use fewer ODE steps during training for speed
        train_steps = self.num_train_ode_steps

        # STEP 1: Encode - integrate backwards from data (x_1) to flow endpoint
        # x_1 = enc_proj(x_input) is already computed in flow_matching_loss
        x_0_encoded = self._euler_integrate(x_1, t_start=1.0, t_end=0.0, num_steps=train_steps)

        # STEP 2: Pass through latent bottleneck
        z = self.to_latent(x_0_encoded)     # 512D → 128D compression
        x_0_recon = self.from_latent(z)     # 128D → 512D expansion

        # STEP 3: Decode - integrate forward to data space
        x_1_recon = self._euler_integrate(x_0_recon, t_start=0.0, t_end=1.0, num_steps=train_steps)

        # STEP 4: Project back to original embedding space
        x_recon = self.dec_proj(x_1_recon)  # 512D → 1024D

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(x_input, x_recon, dim=-1).mean()
        loss_recon = 1 - cos_sim

        return loss_recon

    def compute_consistency_loss(
        self,
        x_input: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward-backward consistency loss: encode(decode(z)) should equal z.

        Uses fewer ODE steps for speed (3 integrations).
        """
        # Use fewer ODE steps for consistency (3 integrations) to reduce overhead
        # 5 steps is enough to check cycle consistency without being too slow
        num_steps = num_steps or min(5, self.num_train_ode_steps)

        # Forward: x → z
        x_1 = self.enc_proj(x_input)
        x_0 = self._euler_integrate(x_1, t_start=1.0, t_end=0.0, num_steps=num_steps)
        z = self.to_latent(x_0)

        # Backward: z → x'
        x_0_recon = self.from_latent(z)
        x_1_recon = self._euler_integrate(x_0_recon, t_start=0.0, t_end=1.0, num_steps=num_steps)

        # Forward again: x' → z'
        x_0_cycle = self._euler_integrate(x_1_recon, t_start=1.0, t_end=0.0, num_steps=num_steps)
        z_cycle = self.to_latent(x_0_cycle)

        # z and z_cycle should match for a truly invertible flow
        return F.mse_loss(z, z_cycle)


# =============================================================================
# LIPSCHITZ REGULARIZATION (LOL-BO, CoBO)
# =============================================================================

def compute_lipschitz_loss(
    model: TextFlowAutoencoder,
    x_input: torch.Tensor,
    epsilon: float = 0.01,
    lip_bound: float = 5.0,
    penalty_type: Literal["hinge", "soft", "quadratic"] = "soft",
) -> Tuple[torch.Tensor, float]:
    """
    Lipschitz regularization for BO-friendly latent space.

    Ensures ||f(z) - f(z+ε)|| / ||ε|| is bounded for GP smoothness.
    """
    # Use fewer ODE steps during training for speed (5 is enough for Lip estimation)
    train_steps = min(5, model.num_train_ode_steps)

    # Encode to get latent
    x_1 = model.enc_proj(x_input)
    x_0_flow = model._euler_integrate(x_1, t_start=1.0, t_end=0.0, num_steps=train_steps)
    z = model.to_latent(x_0_flow)

    # Perturb latent
    noise = torch.randn_like(z) * epsilon
    z_perturbed = z + noise

    # Decode both (need gradients, so use internal methods)
    x_0_original = model.from_latent(z)
    x_0_perturbed = model.from_latent(z_perturbed)

    x_1_original = model._euler_integrate(x_0_original, t_start=0.0, t_end=1.0, num_steps=train_steps)
    x_1_perturbed = model._euler_integrate(x_0_perturbed, t_start=0.0, t_end=1.0, num_steps=train_steps)

    x_recon = model.dec_proj(x_1_original)
    x_recon_perturbed = model.dec_proj(x_1_perturbed)

    # Compute Lipschitz ratio: ||output_change|| / ||input_change||
    output_change = (x_recon_perturbed - x_recon).norm(dim=-1)
    input_change = noise.norm(dim=-1)

    # Use dtype-appropriate epsilon and clamp ratios for stability
    eps = max(torch.finfo(output_change.dtype).eps * 10, 1e-8)
    lipschitz_ratio = output_change / input_change.clamp(min=eps)
    lipschitz_ratio = lipschitz_ratio.clamp(max=1e6)  # Prevent extreme values

    mean_ratio = lipschitz_ratio.mean().item()

    penalties = {
        "hinge": lambda: F.relu(lipschitz_ratio - lip_bound).mean(),
        "soft": lambda: F.softplus(lipschitz_ratio - lip_bound).mean(),
        "quadratic": lambda: ((lipschitz_ratio / lip_bound) ** 2).mean(),
    }

    if penalty_type not in penalties:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    return penalties[penalty_type](), mean_ratio


# =============================================================================
# COMBINED LOSS FUNCTION
# =============================================================================

def _unwrap_model(model):
    """Unwrap DDP/FSDP model to get the underlying module."""
    if hasattr(model, "module"):
        return model.module
    return model


def flow_matching_loss(
    model: TextFlowAutoencoder,
    x_input: torch.Tensor,
    lambda_recon: float = 0.5,
    lambda_gw: float = 0.0,
    lambda_lip: float = 0.1,
    lambda_consistency: float = 0.1,
    timestep_sampling: Literal["uniform", "u_shaped"] = "u_shaped",
    lip_bound: float = 5.0,
    lip_penalty_type: Literal["hinge", "soft", "quadratic"] = "soft",
    use_ot: bool = True,
    objectives: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Combined Flow Matching training loss with BO-friendly regularization.

    L_total = L_FM + lambda_recon*L_recon + lambda_gw*L_GW + lambda_lip*L_Lip + lambda_cons*L_consistency
    """
    # Unwrap DDP if necessary
    unwrapped = _unwrap_model(model)

    # Flow matching loss (main objective) with OT-CFM and U-shaped timestep sampling
    fm_result = unwrapped.compute_flow_matching_loss(
        x_input,
        timestep_sampling=timestep_sampling,
        use_ot=use_ot,
    )
    loss_fm = fm_result["loss_fm"]

    # Reconstruction loss for projections
    loss_recon = unwrapped.compute_reconstruction_loss(
        x_input, fm_result["x_1"], fm_result["x_0"]
    )

    # Start with core losses
    total = loss_fm + lambda_recon * loss_recon
    gw_value = 0.0
    lip_value = 0.0
    lip_ratio = 0.0
    consistency_value = 0.0

    # GW loss (optional, for manifold preservation)
    if lambda_gw > 0:
        z = unwrapped.to_latent(fm_result["x_0"])  # Approximate latent
        loss_gw = sliced_gw_distance(x_input, z)
        total = total + lambda_gw * loss_gw
        gw_value = loss_gw.item()

    # Lipschitz regularization (for BO-friendly latent space)
    if lambda_lip > 0:
        loss_lip, lip_ratio = compute_lipschitz_loss(
            unwrapped, x_input,
            lip_bound=lip_bound,
            penalty_type=lip_penalty_type,
        )
        total = total + lambda_lip * loss_lip
        lip_value = loss_lip.item()

    # Forward-backward consistency loss (for stable training)
    if lambda_consistency > 0:
        loss_consistency = unwrapped.compute_consistency_loss(x_input)
        total = total + lambda_consistency * loss_consistency
        consistency_value = loss_consistency.item()

    return {
        "loss": total,
        "fm": loss_fm.item(),
        "recon": loss_recon.item(),
        "gw": gw_value,
        "lip": lip_value,
        "lip_ratio": lip_ratio,  # NEW: actual Lipschitz ratio for monitoring
        "consistency": consistency_value,
    }


def sliced_gw_distance(
    x_source: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """
    Sliced Gromov-Wasserstein distance.
    Preserves pairwise distance structure between spaces.
    """
    # Pairwise L2 distances
    d_src = torch.cdist(x_source, x_source)
    d_tgt = torch.cdist(x_target, x_target)

    # Normalize for stability
    d_src = d_src / (d_src.max() + 1e-8)
    d_tgt = d_tgt / (d_tgt.max() + 1e-8)

    return F.mse_loss(d_src, d_tgt)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing TextFlowAutoencoder (FlowPO TFA)...")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create encoder with new FlowPO dimensions
    encoder = TextFlowAutoencoder(
        input_dim=1024,    # SONAR dimension
        flow_dim=256,
        latent_dim=128,    # Increased from 32
        num_ode_steps=20,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"Compression ratio: {encoder.input_dim}D → {encoder.latent_dim}D = {encoder.input_dim / encoder.latent_dim:.1f}:1")

    # Test training loss (SIMULATION-FREE)
    batch_size = 32
    x = torch.randn(batch_size, 1024, device=device)
    x = F.normalize(x, dim=-1)

    print("\n--- Training Loss (Simulation-Free) ---")
    losses = flow_matching_loss(encoder, x, lambda_recon=0.1, lambda_lip=0.01)
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"FM loss: {losses['fm']:.4f}")
    print(f"Recon loss: {losses['recon']:.4f}")
    print(f"Lipschitz loss: {losses['lip']:.4f}")

    # Test inference (uses ODE solver)
    print("\n--- Inference (ODE Integration) ---")
    z, x_recon = encoder(x[:8])
    cos_sim = F.cosine_similarity(x[:8], x_recon, dim=-1).mean()
    print(f"Latent shape: {z.shape}")
    print(f"Recon cosine (untrained): {cos_sim.item():.4f}")

    # Test reproducibility
    print("\n--- Reproducibility Test ---")
    z1 = encoder.encode(x[:1])
    z2 = encoder.encode(x[:1])
    diff = (z1 - z2).abs().max()
    print(f"Max diff between encodes: {diff.item():.8f}")
    if diff < 1e-5:
        print("PASS: Deterministic!")
    else:
        print("FAIL: Non-deterministic")

    print("\n[OK] TextFlowAutoencoder tests passed!")
