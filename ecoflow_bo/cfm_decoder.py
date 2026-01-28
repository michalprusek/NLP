"""
Rectified Flow Decoder: 1-step deterministic generation from latent z to embedding.

Key features:
- Standard CFM training with OT coupling
- Reflow procedure to straighten trajectories
- After reflow: 1-step Euler decoding (deterministic, fast)

Reference: "Rectified Flow: A Marginal Preserving Approach to Optimal Transport"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm

from .config import DecoderConfig
from .velocity_network import VelocityNetwork


def unwrap_model(model):
    """Unwrap DDP model to get underlying module."""
    if hasattr(model, 'module'):
        return model.module
    return model


class RectifiedFlowDecoder(nn.Module):
    """
    Conditional Flow Matching decoder with Rectified Flow (Reflow) support.

    Training phases:
    1. Standard CFM: Learn v_θ(x_t, t, z) to transform N(0,I) → data
    2. Reflow: Generate (z, x_gen) pairs, retrain to straighten trajectories
    3. After reflow: 1-step Euler decoding

    The reflow procedure makes trajectories nearly straight, enabling:
    - 1-step generation (vs 20-50 ODE steps)
    - Deterministic outputs (crucial for GP which assumes f(x) is fixed)
    """

    def __init__(
        self,
        velocity_net: VelocityNetwork,
        config: Optional[DecoderConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = DecoderConfig()

        self.velocity_net = velocity_net
        self.config = config
        self.sigma = config.sigma

        # Track whether model has been reflowed
        self.is_reflowed = False
        self.n_reflow_iterations = 0

    def compute_cfm_loss(
        self,
        x_target: torch.Tensor,
        z: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Conditional Flow Matching loss.

        Uses optimal transport coupling: straight line from x_0 ~ N(0,I) to x_1 = x_target.
        The target velocity is simply u_t = x_1 - x_0.

        Args:
            x_target: Target embeddings [B, data_dim]
            z: Latent conditions [B, condition_dim]
            t: Optional time values [B]. If None, sampled uniformly.

        Returns:
            loss: MSE between predicted and target velocity
        """
        B, D = x_target.shape
        device = x_target.device

        # Sample x_0 from standard Gaussian (source distribution)
        x_0 = torch.randn_like(x_target)

        # Sample time uniformly from [0, 1]
        if t is None:
            t = torch.rand(B, device=device)

        # OT interpolation: x_t = t * x_1 + (1 - t) * x_0
        # With small sigma noise: x_t = t * x_1 + (1 - t) * x_0 + sigma * eps
        t_expand = t.view(B, 1)
        x_t = t_expand * x_target + (1 - t_expand) * x_0

        # Add small noise for numerical stability
        if self.sigma > 0:
            x_t = x_t + self.sigma * torch.randn_like(x_t)

        # Target velocity for OT: u_t = x_1 - x_0 (constant along trajectory)
        u_t = x_target - x_0

        # Predict velocity
        v_pred = self.velocity_net(x_t, t, z)

        # MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, u_t)

        return loss

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Decode latent z to embedding using ODE integration.

        After reflow training, 1 step is sufficient.

        Args:
            z: Latent codes [B, condition_dim]
            n_steps: Number of Euler steps. If None, uses config.euler_steps.
            return_trajectory: If True, return all intermediate x_t values.

        Returns:
            x: Decoded embeddings [B, data_dim]
            trajectory: (optional) List of x_t at each step
        """
        if n_steps is None:
            n_steps = self.config.euler_steps

        B = z.shape[0]
        device = z.device
        dtype = z.dtype

        # Start from standard Gaussian (unwrap DDP if needed)
        vel_net = unwrap_model(self.velocity_net)
        x = torch.randn(B, vel_net.data_dim, device=device, dtype=dtype)

        trajectory = [x.clone()] if return_trajectory else None

        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device, dtype=dtype)
            v = self.velocity_net(x, t, z)
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def decode_deterministic(
        self, z: torch.Tensor, seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Deterministic decoding with fixed random seed.

        Critical for GP optimization: ensures f(z) returns same x for same z.

        Args:
            z: Latent codes [B, condition_dim]
            seed: Random seed for initial noise

        Returns:
            x: Decoded embeddings [B, data_dim]
        """
        if seed is None:
            seed = 42

        # Save RNG state
        rng_state = torch.get_rng_state()
        if z.is_cuda:
            cuda_rng_state = torch.cuda.get_rng_state(z.device)

        # Set seed for reproducible x_0
        torch.manual_seed(seed)
        if z.is_cuda:
            torch.cuda.manual_seed(seed)

        # Decode
        x = self.decode(z)

        # Restore RNG state
        torch.set_rng_state(rng_state)
        if z.is_cuda:
            torch.cuda.set_rng_state(cuda_rng_state, z.device)

        return x

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        z_samples: torch.Tensor,
        n_ode_steps: int = 50,
        batch_size: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate (x_0, z, x_1) pairs for reflow training.

        Uses multi-step ODE to get accurate x_1, then pairs with x_0.

        Args:
            z_samples: Latent codes to decode [N, condition_dim]
            n_ode_steps: Steps for accurate ODE solve
            batch_size: Batch size for generation

        Returns:
            x_0: Initial noise samples [N, data_dim]
            z: Latent conditions [N, condition_dim]
            x_1: Generated samples [N, data_dim]
        """
        self.velocity_net.eval()

        N = z_samples.shape[0]
        device = z_samples.device
        dtype = z_samples.dtype
        vel_net = unwrap_model(self.velocity_net)
        data_dim = vel_net.data_dim

        x_0_all = []
        x_1_all = []

        # Generate in batches
        for i in tqdm(range(0, N, batch_size), desc="Generating reflow pairs"):
            z_batch = z_samples[i:i + batch_size]
            B = z_batch.shape[0]

            # Sample x_0 and store
            x_0 = torch.randn(B, data_dim, device=device, dtype=dtype)
            x_0_all.append(x_0)

            # Integrate ODE from x_0 to x_1
            x = x_0.clone()
            dt = 1.0 / n_ode_steps

            for step in range(n_ode_steps):
                t = torch.full((B,), step * dt, device=device, dtype=dtype)
                v = self.velocity_net(x, t, z_batch)
                x = x + v * dt

            x_1_all.append(x)

        x_0_all = torch.cat(x_0_all, dim=0)
        x_1_all = torch.cat(x_1_all, dim=0)

        return x_0_all, z_samples, x_1_all

    def compute_reflow_loss(
        self,
        x_0: torch.Tensor,
        z: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reflow loss on (x_0, z, x_1) tuples.

        After reflow, the trajectory should be a straight line from x_0 to x_1,
        so the velocity should be constant: v = x_1 - x_0.

        Args:
            x_0: Initial noise [B, data_dim]
            z: Latent conditions [B, condition_dim]
            x_1: Generated samples [B, data_dim]
            t: Optional time values [B]

        Returns:
            loss: MSE loss
        """
        B = x_0.shape[0]
        device = x_0.device

        if t is None:
            t = torch.rand(B, device=device)

        # Straight line interpolation
        t_expand = t.view(B, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0

        # Target velocity (constant along straight line)
        u_t = x_1 - x_0

        # Predict velocity
        v_pred = self.velocity_net(x_t, t, z)

        # MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, u_t)

        return loss

    def mark_as_reflowed(self):
        """Mark model as having completed reflow training."""
        self.is_reflowed = True
        self.n_reflow_iterations += 1
        # After reflow, 1-step Euler is sufficient
        self.config.euler_steps = 1

    def get_trajectory_straightness(
        self, z: torch.Tensor, n_samples: int = 100
    ) -> float:
        """
        Measure how straight the learned trajectories are.

        Returns ratio of straight-line distance to actual path length.
        After successful reflow, this should be close to 1.0.
        """
        x, trajectory = self.decode(z[:n_samples], n_steps=50, return_trajectory=True)

        # Straight line distance (x_0 to x_1)
        straight_dist = torch.norm(trajectory[-1] - trajectory[0], dim=-1)

        # Path length (sum of step distances)
        path_length = torch.zeros(n_samples, device=z.device)
        for i in range(1, len(trajectory)):
            path_length += torch.norm(trajectory[i] - trajectory[i - 1], dim=-1)

        # Straightness ratio (1.0 = perfectly straight)
        straightness = (straight_dist / (path_length + 1e-8)).mean().item()

        return straightness

    @torch.no_grad()
    def decode_with_refinement(
        self,
        z: torch.Tensor,
        encoder,
        n_refinement_steps: int = 3,
        correction_scale: float = 0.5,
        n_decode_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode with iterative refinement for improved reconstruction.

        Process:
        1. Initial decode: x = decode(z)
        2. For each refinement step:
           a. Re-encode: z_reenc = encoder(x)
           b. Compute correction: correction = z - z_reenc
           c. Apply correction: z_corrected = z + correction_scale * correction
           d. Re-decode: x = decode(z_corrected)

        This iterative process corrects for accumulated errors in the
        encode-decode cycle, improving final reconstruction quality.

        Args:
            z: Latent codes [B, condition_dim]
            encoder: MatryoshkaEncoder with encode_deterministic method
            n_refinement_steps: Number of refinement iterations (default: 3)
            correction_scale: How much correction to apply (default: 0.5)
            n_decode_steps: ODE steps for decoding (uses config if None)

        Returns:
            x: Refined embeddings [B, data_dim]
        """
        # Initial decode
        x = self.decode(z, n_steps=n_decode_steps)

        for step in range(n_refinement_steps):
            # Re-encode current reconstruction
            z_reenc = encoder.encode_deterministic(x)

            # Compute correction (how far off are we in latent space?)
            correction = z - z_reenc

            # Apply scaled correction and re-decode
            z_corrected = z + correction_scale * correction
            x = self.decode(z_corrected, n_steps=n_decode_steps)

        return x
