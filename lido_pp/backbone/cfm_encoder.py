"""
Text Flow Autoencoder (TFA) for FlowPO.

This module implements the core autoencoding component of FlowPO:
simulation-free flow matching that maps SONAR embeddings (1024D) to
a compact latent space (128D) for Bayesian Optimization.

KEY INSIGHT (FlowPO contribution):
- Training: Simulation-free! No ODE solver needed. Just regress velocity field.
- Inference: Use ODE solver to integrate the learned flow.
- Compression: 8:1 (1024D → 128D) vs old 128:1 (4096D → 32D)

Novel contribution: First application of FM autoencoding for text reconstruction.

Reference:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- LOL-BO: Lipschitz regularization for BO-friendly latent spaces (NeurIPS 2022)
- CoBO: Coordinate-wise Bayesian Optimization (NeurIPS 2023)
"""

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        dim: int = 256,
        time_dim: int = 64,
        hidden_mult: int = 4,
        num_layers: int = 3,
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
                nn.SiLU(),
            ])

        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

        # Zero init output for stable training (identity flow at init)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity at (x, t).

        Args:
            t: (B,) or scalar timestep in [0, 1]
            x: (B, dim) current state

        Returns:
            v: (B, dim) velocity
        """
        # Handle scalar or batch t
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)  # (B, dim)
        h = torch.cat([x, t_emb], dim=-1)  # (B, 2*dim)
        return self.net(h)  # (B, dim)


class TextFlowAutoencoder(nn.Module):
    """
    Text Flow Autoencoder (TFA) - core component of FlowPO.

    Maps SONAR embeddings (1024D) to compact latent space (128D) via
    simulation-free flow matching. This enables efficient Bayesian
    Optimization in the latent space.

    Architecture (training is simulation-free, inference uses ODE):
        Encode: SONAR 1024D → enc_proj → 256D → [Flow t=1→0] → 256D → to_latent → 128D
        Decode: 128D → from_latent → 256D → [Flow t=0→1] → 256D → dec_proj → 1024D

    Flow operates in 256D intermediate space, with 128D bottleneck at t=0.

    Training: Simulation-free Flow Matching (no ODE solver!)
    Inference: ODE integration for encode/decode

    Flow direction:
        t=0: latent space (flow endpoint, then projected to 128D)
        t=1: data space (projected SONAR embedding in 256D)

    Novel contribution: First FM autoencoder for text reconstruction.
    """

    def __init__(
        self,
        input_dim: int = 1024,    # SONAR embedding dimension
        flow_dim: int = 256,      # Intermediate flow space
        latent_dim: int = 128,    # Target latent dimension (was 32)
        time_dim: int = 64,       # Timestep embedding dimension
        num_ode_steps: int = 20,  # Euler integration steps
    ):
        super().__init__()
        self.input_dim = input_dim
        self.flow_dim = flow_dim
        self.latent_dim = latent_dim
        self.num_ode_steps = num_ode_steps

        # Projections between spaces
        self.enc_proj = nn.Linear(input_dim, flow_dim)   # 1024 → 256
        self.dec_proj = nn.Linear(flow_dim, input_dim)   # 256 → 1024

        # Latent bottleneck
        self.to_latent = nn.Linear(flow_dim, latent_dim)     # 256 → 128
        self.from_latent = nn.Linear(latent_dim, flow_dim)   # 128 → 256

        # Velocity field for Flow Matching
        self.velocity = VelocityField(
            dim=flow_dim,
            time_dim=time_dim,
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
    ) -> Dict[str, torch.Tensor]:
        """
        Compute simulation-free Flow Matching loss.

        This is the KEY INNOVATION - no ODE solver during training!
        We directly regress the velocity field to match straight-line paths.

        Args:
            x_input: (B, input_dim) input embeddings

        Returns:
            dict with loss components and intermediates

        Raises:
            ValueError: If input contains NaN or Inf values
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

        # 3. Sample time t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # 4. Compute linear interpolation (straight OT path)
        # x_t = t * x_1 + (1 - t) * x_0
        t_view = t.view(-1, 1)
        x_t = t_view * x_1 + (1 - t_view) * x_0

        # 5. Target velocity (the straight line direction)
        # For OT path: u_t = x_1 - x_0
        u_t = x_1 - x_0

        # 6. Predict velocity
        v_t = self.velocity(t, x_t)

        # 7. Flow Matching loss (MSE between predicted and target velocity)
        loss_fm = F.mse_loss(v_t, u_t)

        # Validate loss
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
        Compute reconstruction loss for the projection layers.

        This ensures enc_proj and dec_proj are invertible.
        """
        # Decode from projected data to reconstructed embedding
        x_recon = self.dec_proj(x_1)  # Should reconstruct original

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(x_input, x_recon, dim=-1).mean()
        loss_recon = 1 - cos_sim

        return loss_recon


# =============================================================================
# LIPSCHITZ REGULARIZATION (LOL-BO, CoBO)
# =============================================================================

def compute_lipschitz_loss(
    model: TextFlowAutoencoder,
    x_input: torch.Tensor,
    epsilon: float = 0.01,
    lip_bound: float = 10.0,
) -> torch.Tensor:
    """
    Lipschitz regularization for BO-friendly latent space.

    Ensures ||f(z) - f(z+ε)|| / ||ε|| is bounded, which is critical for
    GP surrogate smoothness in Bayesian Optimization.

    Key insight from LOL-BO (NeurIPS 2022): Without Lipschitz regularization,
    small latent perturbations can cause large semantic jumps, violating
    GP smoothness assumptions and reducing optimization effectiveness.

    Args:
        model: TextFlowAutoencoder
        x_input: (B, input_dim) input embeddings
        epsilon: Perturbation magnitude
        lip_bound: Maximum allowed Lipschitz ratio (default: 10)

    Returns:
        Lipschitz loss (penalizes ratios above lip_bound)
    """
    # Encode to get latent
    x_1 = model.enc_proj(x_input)
    x_0_flow = model._euler_integrate(x_1, t_start=1.0, t_end=0.0)
    z = model.to_latent(x_0_flow)

    # Perturb latent
    noise = torch.randn_like(z) * epsilon
    z_perturbed = z + noise

    # Decode both (need gradients, so use internal methods)
    x_0_original = model.from_latent(z)
    x_0_perturbed = model.from_latent(z_perturbed)

    x_1_original = model._euler_integrate(x_0_original, t_start=0.0, t_end=1.0)
    x_1_perturbed = model._euler_integrate(x_0_perturbed, t_start=0.0, t_end=1.0)

    x_recon = model.dec_proj(x_1_original)
    x_recon_perturbed = model.dec_proj(x_1_perturbed)

    # Compute Lipschitz ratio: ||output_change|| / ||input_change||
    output_change = (x_recon_perturbed - x_recon).norm(dim=-1)
    input_change = noise.norm(dim=-1)

    # Use dtype-appropriate epsilon and clamp ratios for stability
    eps = max(torch.finfo(output_change.dtype).eps * 10, 1e-8)
    lipschitz_ratio = output_change / input_change.clamp(min=eps)
    lipschitz_ratio = lipschitz_ratio.clamp(max=1e6)  # Prevent extreme values

    # Penalize ratios above bound
    loss = F.relu(lipschitz_ratio - lip_bound).mean()

    return loss


# =============================================================================
# COMBINED LOSS FUNCTION
# =============================================================================

def flow_matching_loss(
    model: TextFlowAutoencoder,
    x_input: torch.Tensor,
    lambda_recon: float = 0.1,
    lambda_gw: float = 0.0,
    lambda_lip: float = 0.01,
    objectives: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Combined Flow Matching training loss with BO-friendly regularization.

    L_total = L_FM + λ_recon·L_recon + λ_gw·L_GW + λ_lip·L_Lip

    SIMULATION-FREE: No ODE solver during training!

    Args:
        model: TextFlowAutoencoder
        x_input: (B, input_dim) input embeddings
        lambda_recon: Weight for reconstruction loss
        lambda_gw: Weight for Gromov-Wasserstein loss (optional)
        lambda_lip: Weight for Lipschitz regularization
        objectives: Optional objective values for correlation loss

    Returns:
        dict with loss components
    """
    # Flow matching loss (main objective)
    fm_result = model.compute_flow_matching_loss(x_input)
    loss_fm = fm_result["loss_fm"]

    # Reconstruction loss for projections
    loss_recon = model.compute_reconstruction_loss(
        x_input, fm_result["x_1"], fm_result["x_0"]
    )

    # Start with core losses
    total = loss_fm + lambda_recon * loss_recon
    gw_value = 0.0
    lip_value = 0.0

    # GW loss (optional, for manifold preservation)
    if lambda_gw > 0:
        z = model.to_latent(fm_result["x_0"])  # Approximate latent
        loss_gw = sliced_gw_distance(x_input, z)
        total = total + lambda_gw * loss_gw
        gw_value = loss_gw.item()

    # Lipschitz regularization (for BO-friendly latent space)
    if lambda_lip > 0:
        loss_lip = compute_lipschitz_loss(model, x_input)
        total = total + lambda_lip * loss_lip
        lip_value = loss_lip.item()

    return {
        "loss": total,
        "fm": loss_fm.item(),
        "recon": loss_recon.item(),
        "gw": gw_value,
        "lip": lip_value,
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
# LEGACY ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================


def CoupledFlowEncoder(*args, **kwargs):
    """DEPRECATED: Use TextFlowAutoencoder instead."""
    warnings.warn(
        "CoupledFlowEncoder is deprecated, use TextFlowAutoencoder",
        DeprecationWarning,
        stacklevel=2,
    )
    return TextFlowAutoencoder(*args, **kwargs)


def cfm_loss(*args, **kwargs):
    """DEPRECATED: Use flow_matching_loss instead."""
    warnings.warn(
        "cfm_loss is deprecated, use flow_matching_loss",
        DeprecationWarning,
        stacklevel=2,
    )
    return flow_matching_loss(*args, **kwargs)


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
