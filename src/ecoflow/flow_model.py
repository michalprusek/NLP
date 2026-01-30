"""
Flow matching model with ODE-based sampling.

Provides Euler and Heun integration methods for generating samples
from trained velocity network.

Supports denormalization: when trained on normalized data, samples can be
converted back to original scale using saved normalization statistics.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.ecoflow.velocity_network import VelocityNetwork


class FlowMatchingModel(nn.Module):
    """
    Flow matching wrapper with ODE sampling.

    Generates samples by integrating the learned velocity field
    from noise (t=0) to data (t=1).

    Supports denormalization for models trained on normalized data.
    """

    def __init__(
        self,
        velocity_net: VelocityNetwork,
        norm_stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.input_dim = velocity_net.input_dim
        self.norm_stats = norm_stats  # {'mean': [1024], 'std': [1024]}

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize samples back to original scale.

        Args:
            x: Normalized samples [B, 1024]

        Returns:
            Denormalized samples [B, 1024]
        """
        if self.norm_stats is None:
            return x
        mean = self.norm_stats["mean"].to(x.device)
        std = self.norm_stats["std"].to(x.device)
        return x * std + mean

    def ode_func(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        ODE function for integration: dx/dt = v(x, t).

        Args:
            t: Scalar timestep tensor
            x: Current state [B, input_dim]

        Returns:
            Velocity [B, input_dim]
        """
        # Expand scalar t to batch
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.velocity_net(x, t)

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
        Generate samples using ODE integration.

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            method: Integration method ("heun" or "euler", default: heun)
            num_steps: Number of integration steps
            denormalize: If True and norm_stats exist, denormalize samples

        Returns:
            Generated samples [n_samples, input_dim]
        """
        self.velocity_net.eval()

        # Start from noise at t=0
        x = torch.randn(n_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        if method == "heun":
            # Heun's method: predictor-corrector (2nd order)
            for i in range(num_steps):
                t = torch.tensor(i * dt, device=device)
                t_next = torch.tensor((i + 1) * dt, device=device)

                # Predictor: Euler step
                v1 = self.ode_func(t, x)
                x_pred = x + v1 * dt

                # Corrector: evaluate at predicted point
                v2 = self.ode_func(t_next, x_pred)

                # Average velocities
                x = x + 0.5 * (v1 + v2) * dt
        else:
            # Euler integration from t=0 to t=1
            for i in range(num_steps):
                t = torch.tensor(i * dt, device=device)
                v = self.ode_func(t, x)
                x = x + v * dt

        # Denormalize if trained on normalized data
        if denormalize:
            x = self.denormalize(x)

        return x

    def sample_heun(
        self,
        n_samples: int,
        device: str = "cpu",
        num_steps: int = 25,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        DEPRECATED: Use sample(method='heun') instead.

        Generate samples using Heun's method (2nd-order).
        """
        import warnings
        warnings.warn(
            "sample_heun() is deprecated, use sample(method='heun') instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sample(
            n_samples=n_samples,
            device=device,
            method="heun",
            num_steps=num_steps,
            denormalize=denormalize,
        )
