"""
Flow matching model with ODE-based sampling.

Provides Euler and Heun integration methods for generating samples
from trained velocity network.
"""

import torch
import torch.nn as nn

from src.ecoflow.velocity_network import VelocityNetwork


class FlowMatchingModel(nn.Module):
    """
    Flow matching wrapper with ODE sampling.

    Generates samples by integrating the learned velocity field
    from noise (t=0) to data (t=1).
    """

    def __init__(self, velocity_net: VelocityNetwork):
        super().__init__()
        self.velocity_net = velocity_net
        self.input_dim = velocity_net.input_dim

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
        method: str = "euler",
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration.

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            method: Integration method (currently only "euler")
            num_steps: Number of integration steps

        Returns:
            Generated samples [n_samples, input_dim]
        """
        self.velocity_net.eval()

        # Start from noise at t=0
        x = torch.randn(n_samples, self.input_dim, device=device)

        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            v = self.ode_func(t, x)
            x = x + v * dt

        return x

    @torch.no_grad()
    def sample_heun(
        self,
        n_samples: int,
        device: str = "cpu",
        num_steps: int = 25,
    ) -> torch.Tensor:
        """
        Generate samples using Heun's method (2nd-order).

        More accurate than Euler with the same number of function evaluations
        per step (though uses 2 evals per step vs 1 for Euler).

        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            num_steps: Number of integration steps

        Returns:
            Generated samples [n_samples, input_dim]
        """
        self.velocity_net.eval()

        # Start from noise at t=0
        x = torch.randn(n_samples, self.input_dim, device=device)

        # Heun's method: predictor-corrector
        dt = 1.0 / num_steps
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

        return x
