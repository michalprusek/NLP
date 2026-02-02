"""Flow matching model with ODE-based sampling."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ecoflow.velocity_network import VelocityNetwork


class FlowMatchingModel(nn.Module):
    """Flow matching wrapper with Euler/Heun ODE integration."""

    # Mean SONAR embedding norm (for spherical flow denormalization)
    SONAR_EMBEDDING_NORM = 0.3185

    def __init__(
        self,
        velocity_net: VelocityNetwork,
        norm_stats: Optional[Dict[str, torch.Tensor]] = None,
        is_spherical: bool = False,
    ):
        """
        Args:
            velocity_net: Velocity network.
            norm_stats: Normalization statistics {'mean': [D], 'std': [D]}.
            is_spherical: If True, use spherical denormalization (scale to SONAR norm)
                         instead of mean/std denormalization. Required for spherical
                         flow models (spherical, spherical-ot).
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.input_dim = velocity_net.input_dim
        self.norm_stats = norm_stats
        self.is_spherical = is_spherical

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize samples to training scale."""
        if self.norm_stats is None:
            return x
        mean = self.norm_stats["mean"].to(x.device)
        std = self.norm_stats["std"].to(x.device)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize samples back to original scale.

        For spherical flows: normalizes to unit sphere then scales to SONAR norm.
        For regular flows: applies x * std + mean transform.
        """
        if self.is_spherical:
            # Spherical flow outputs on unit sphere - scale to SONAR embedding norm
            import torch.nn.functional as F
            x_unit = F.normalize(x, p=2, dim=-1)
            return x_unit * self.SONAR_EMBEDDING_NORM

        if self.norm_stats is None:
            return x
        mean = self.norm_stats["mean"].to(x.device)
        std = self.norm_stats["std"].to(x.device)
        return x * std + mean

    def _ode_func(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute velocity: dx/dt = v(x, t)."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.velocity_net(x, t)

    def _integrate(
        self,
        z: torch.Tensor,
        method: str,
        num_steps: int,
        forward: bool,
    ) -> torch.Tensor:
        """
        Integrate ODE using Euler or Heun method.

        Args:
            z: Initial state [B, input_dim]
            method: "euler" or "heun"
            num_steps: Number of integration steps
            forward: True for t=0->1 (sampling), False for t=1->0 (encoding)
        """
        import torch.nn.functional as F

        dt = 1.0 / num_steps
        device = z.device

        if forward:
            step_range = range(num_steps)
        else:
            step_range = range(num_steps - 1, -1, -1)

        for i in step_range:
            if forward:
                t_curr = torch.tensor(i * dt, device=device)
                t_next = torch.tensor((i + 1) * dt, device=device)
                direction = 1.0
            else:
                t_curr = torch.tensor((i + 1) * dt, device=device)
                t_next = torch.tensor(i * dt, device=device)
                direction = -1.0

            v1 = self._ode_func(t_curr, z)

            if method == "heun":
                z_pred = z + direction * v1 * dt
                # For spherical flows, project predictor back to sphere
                if self.is_spherical:
                    z_pred = F.normalize(z_pred, p=2, dim=-1)
                v2 = self._ode_func(t_next, z_pred)
                z = z + direction * 0.5 * (v1 + v2) * dt
            else:
                z = z + direction * v1 * dt

            # For spherical flows, project back to unit sphere after each step
            if self.is_spherical:
                z = F.normalize(z, p=2, dim=-1)

        return z

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: str = "cpu",
        method: str = "heun",
        num_steps: int = 50,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples via ODE integration from noise (t=0) to data (t=1).

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            method: "euler" or "heun" (default: heun)
            num_steps: Number of integration steps
            denormalize: Convert samples to original scale if norm_stats exist
        """
        import torch.nn.functional as F

        self.velocity_net.eval()
        x = torch.randn(n_samples, self.input_dim, device=device)

        # For spherical flows, start on unit sphere (matches SLERP training)
        if self.is_spherical:
            x = F.normalize(x, p=2, dim=-1)

        x = self._integrate(x, method, num_steps, forward=True)

        if denormalize:
            x = self.denormalize(x)
        return x

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        method: str = "heun",
        num_steps: int = 50,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode data to noise space via inverse ODE (t=1 -> t=0).

        Args:
            x: Data points in SONAR space [B, 1024]
            method: "euler" or "heun"
            num_steps: Number of integration steps
            normalize: Normalize input to training scale if norm_stats exist
        """
        self.velocity_net.eval()

        if normalize:
            z = self.normalize(x)
        else:
            z = x.clone()

        return self._integrate(z, method, num_steps, forward=False)
