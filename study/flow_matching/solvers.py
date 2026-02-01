"""ODE solvers for flow matching inference.

This module provides various ODE solvers for integrating the flow ODE:
    dx/dt = v(x, t)

from t=0 (noise) to t=1 (data).

Solvers implemented:
- Euler (1st order): Simple, fast, but requires many steps
- Heun (2nd order): Better accuracy with fewer steps
- DPM-Solver++ (adaptive): State-of-the-art for diffusion/flow models

Usage:
    from study.flow_matching.solvers import get_solver, euler_integrate, heun_integrate

    # Use solver by name
    solver = get_solver("heun")
    x1 = solver(model, x0, n_steps=50, device="cuda:0")

    # Or directly
    x1 = heun_integrate(model, x0, n_steps=50, device="cuda:0")
"""

import logging
from typing import Callable, Literal, Optional

import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def euler_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    show_progress: bool = True,
) -> Tensor:
    """Euler method (1st order) ODE integration.

    The simplest ODE solver: x_{t+dt} = x_t + dt * v(x_t, t)

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Number of integration steps.
        device: Computation device.
        show_progress: Whether to show progress bar.

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    dt = 1.0 / n_steps
    x = x0.to(device)

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Euler integration", leave=False)

    for i in iterator:
        t = i / n_steps
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        v = model(x, t_batch)
        x = x + dt * v

    return x


@torch.no_grad()
def heun_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    show_progress: bool = True,
) -> Tensor:
    """Heun's method (2nd order) ODE integration.

    Also known as the improved Euler method or explicit trapezoidal rule.
    Uses predictor-corrector approach for better accuracy:

    1. Predict: x_pred = x_t + dt * v(x_t, t)
    2. Correct: x_{t+dt} = x_t + dt/2 * (v(x_t, t) + v(x_pred, t+dt))

    Requires 2 function evaluations per step but achieves 2nd order accuracy.

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Number of integration steps.
        device: Computation device.
        show_progress: Whether to show progress bar.

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    dt = 1.0 / n_steps
    x = x0.to(device)

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Heun integration", leave=False)

    for i in iterator:
        t = i / n_steps
        t_next = (i + 1) / n_steps

        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        t_next_batch = torch.full((x.shape[0],), t_next, device=device, dtype=x.dtype)

        # Predictor step (Euler)
        v1 = model(x, t_batch)
        x_pred = x + dt * v1

        # Corrector step
        v2 = model(x_pred, t_next_batch)

        # Heun update (average of slopes)
        x = x + dt * 0.5 * (v1 + v2)

    return x


@torch.no_grad()
def rk4_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    show_progress: bool = True,
) -> Tensor:
    """Classical 4th order Runge-Kutta ODE integration.

    The workhorse of numerical ODE solving with 4th order accuracy.
    Requires 4 function evaluations per step.

    k1 = v(x, t)
    k2 = v(x + dt/2 * k1, t + dt/2)
    k3 = v(x + dt/2 * k2, t + dt/2)
    k4 = v(x + dt * k3, t + dt)
    x_{t+dt} = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Number of integration steps.
        device: Computation device.
        show_progress: Whether to show progress bar.

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    dt = 1.0 / n_steps
    x = x0.to(device)

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="RK4 integration", leave=False)

    for i in iterator:
        t = i / n_steps

        t1 = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        t_mid = torch.full((x.shape[0],), t + 0.5 * dt, device=device, dtype=x.dtype)
        t4 = torch.full((x.shape[0],), t + dt, device=device, dtype=x.dtype)

        k1 = model(x, t1)
        k2 = model(x + 0.5 * dt * k1, t_mid)
        k3 = model(x + 0.5 * dt * k2, t_mid)
        k4 = model(x + dt * k3, t4)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


@torch.no_grad()
def dpm_solver_pp_2_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    show_progress: bool = True,
) -> Tensor:
    """DPM-Solver++ 2nd order (multistep) ODE integration.

    Adapted from DPM-Solver++ paper for flow matching.
    Uses exponential integrator formulation with 2nd order accuracy.

    For flow matching with linear interpolation path, this simplifies to:
    - First step: Euler
    - Subsequent steps: Use previous velocity for 2nd order correction

    Reference: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling
    of Diffusion Probabilistic Models", NeurIPS 2022.

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Number of integration steps.
        device: Computation device.
        show_progress: Whether to show progress bar.

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    dt = 1.0 / n_steps
    x = x0.to(device)

    # Store previous velocity for multistep
    v_prev = None
    t_prev = None

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="DPM++ integration", leave=False)

    for i in iterator:
        t = i / n_steps
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)

        # Current velocity
        v = model(x, t_batch)

        if i == 0:
            # First step: Euler (no previous info)
            x = x + dt * v
        else:
            # 2nd order: Use linear combination of current and previous velocity
            # For equally spaced steps, this is equivalent to Adams-Bashforth 2
            # x_{n+1} = x_n + dt * (3/2 * v_n - 1/2 * v_{n-1})
            # But for flow matching, we use midpoint-style correction:
            # x_{n+1} = x_n + dt * v_n + (dt^2 / 2) * (v_n - v_{n-1}) / dt
            #         = x_n + dt * (1.5 * v_n - 0.5 * v_{n-1})
            x = x + dt * (1.5 * v - 0.5 * v_prev)

        v_prev = v
        t_prev = t

    return x


@torch.no_grad()
def adaptive_heun_integrate(
    model: torch.nn.Module,
    x0: Tensor,
    n_steps: int,
    device: torch.device,
    show_progress: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_steps: int = 1000,
) -> Tensor:
    """Adaptive step-size Heun integration with error control.

    Uses local error estimation to adapt step size. Starts with uniform
    steps but can refine in regions where the velocity field changes rapidly.

    Args:
        model: Velocity network with forward(x, t) -> v signature.
        x0: Initial points at t=0, shape [N, D].
        n_steps: Initial/target number of steps (may vary with adaptation).
        device: Computation device.
        show_progress: Whether to show progress bar.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        max_steps: Maximum allowed steps.

    Returns:
        x1: Points at t=1, shape [N, D].
    """
    x = x0.to(device)
    t = 0.0
    dt = 1.0 / n_steps
    min_dt = 1e-6
    steps_taken = 0

    pbar = None
    if show_progress:
        pbar = tqdm(total=1.0, desc="Adaptive Heun", leave=False)

    while t < 1.0 - 1e-8 and steps_taken < max_steps:
        # Don't overshoot t=1
        dt = min(dt, 1.0 - t)

        t_batch = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
        t_next_batch = torch.full((x.shape[0],), t + dt, device=device, dtype=x.dtype)

        # Full Heun step
        v1 = model(x, t_batch)
        x_pred = x + dt * v1
        v2 = model(x_pred, t_next_batch)
        x_heun = x + dt * 0.5 * (v1 + v2)

        # Euler step for error estimation
        x_euler = x + dt * v1

        # Error estimate
        error = (x_heun - x_euler).abs()
        tol = atol + rtol * x_heun.abs()
        err_ratio = (error / tol).max().item()

        if err_ratio <= 1.0:
            # Accept step
            x = x_heun
            t += dt
            steps_taken += 1

            if pbar:
                pbar.update(dt)

            # Increase step size
            if err_ratio > 0:
                dt = dt * min(2.0, 0.9 * (1.0 / err_ratio) ** 0.5)
        else:
            # Reject and reduce step size
            dt = dt * max(0.5, 0.9 * (1.0 / err_ratio) ** 0.5)
            dt = max(dt, min_dt)

    if pbar:
        pbar.close()

    logger.debug(f"Adaptive Heun took {steps_taken} steps (target: {n_steps})")
    return x


# Solver registry
SOLVERS = {
    "euler": euler_integrate,
    "heun": heun_integrate,
    "rk4": rk4_integrate,
    "dpm++": dpm_solver_pp_2_integrate,
    "dpm_solver_pp": dpm_solver_pp_2_integrate,
    "adaptive_heun": adaptive_heun_integrate,
}


def get_solver(
    name: Literal["euler", "heun", "rk4", "dpm++", "dpm_solver_pp", "adaptive_heun"]
) -> Callable:
    """Get solver function by name.

    Args:
        name: Solver name from SOLVERS registry.

    Returns:
        Solver function with signature (model, x0, n_steps, device, show_progress) -> x1.

    Raises:
        ValueError: If solver name is unknown.
    """
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver: {name}. Available: {list(SOLVERS.keys())}")
    return SOLVERS[name]


def list_solvers() -> list[str]:
    """List available solver names."""
    return list(SOLVERS.keys())


@torch.no_grad()
def benchmark_solvers(
    model: torch.nn.Module,
    x0: Tensor,
    device: torch.device,
    step_counts: list[int] = [10, 20, 50, 100, 200],
    solvers_to_test: Optional[list[str]] = None,
    reference_steps: int = 500,
) -> dict:
    """Benchmark solvers at various step counts.

    Uses high-step Euler as reference to compute error.

    Args:
        model: Velocity network.
        x0: Starting points at t=0.
        device: Computation device.
        step_counts: List of step counts to test.
        solvers_to_test: Solver names to test (default: all).
        reference_steps: Steps for reference solution.

    Returns:
        Dict with benchmark results per solver per step count.
    """
    import time

    if solvers_to_test is None:
        solvers_to_test = ["euler", "heun", "rk4", "dpm++"]

    # Compute reference solution
    logger.info(f"Computing reference with {reference_steps} Euler steps...")
    x_ref = euler_integrate(model, x0, reference_steps, device, show_progress=False)

    results = {}

    for solver_name in solvers_to_test:
        solver = get_solver(solver_name)
        results[solver_name] = {}

        for n_steps in step_counts:
            # Time the integration
            start = time.perf_counter()
            x1 = solver(model, x0, n_steps, device, show_progress=False)
            elapsed = time.perf_counter() - start

            # Compute error vs reference
            mse = ((x1 - x_ref) ** 2).mean().item()
            max_error = (x1 - x_ref).abs().max().item()

            results[solver_name][n_steps] = {
                "mse": mse,
                "max_error": max_error,
                "time_seconds": elapsed,
                # NFE varies by solver: euler=1/step, heun=2/step, rk4=4/step, dpm++=1/step (multistep)
                "nfe": n_steps * {"euler": 1, "heun": 2, "rk4": 4, "dpm++": 1, "dpm_solver_pp": 1}.get(solver_name, 2),
            }

            logger.info(
                f"{solver_name:12} steps={n_steps:3d} MSE={mse:.2e} "
                f"max_err={max_error:.2e} time={elapsed:.3f}s"
            )

    return results


if __name__ == "__main__":
    # Quick test with dummy model
    import torch.nn as nn

    class DummyVelocity(nn.Module):
        """Simple linear velocity for testing: v(x,t) = x1 - x0 (constant)."""
        def __init__(self, target: Tensor):
            super().__init__()
            self.register_buffer("target", target)

        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            # For linear interpolation, velocity is constant
            return self.target - torch.randn_like(x)  # Random source

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with random target
    batch_size = 16
    dim = 1024
    x0 = torch.randn(batch_size, dim, device=device)
    target = torch.randn(batch_size, dim, device=device)

    # Simple identity velocity for testing
    class ConstantVelocity(nn.Module):
        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            return torch.ones_like(x)  # Constant velocity = 1

    model = ConstantVelocity().to(device)

    print("\nTesting solvers with constant velocity v=1:")
    print("Expected: x1 = x0 + 1.0 * integral(1, 0, 1) = x0 + 1.0")

    for solver_name in list_solvers():
        if solver_name == "adaptive_heun":
            continue  # Skip adaptive for simple test

        solver = get_solver(solver_name)
        x1 = solver(model, x0, n_steps=10, device=device, show_progress=False)
        expected = x0 + 1.0
        error = (x1 - expected).abs().mean().item()
        print(f"{solver_name:12} error={error:.2e}")
