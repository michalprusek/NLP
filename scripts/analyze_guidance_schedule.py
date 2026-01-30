"""
Analyze optimal guidance schedule λ(t) for LCB-guided flow sampling.

This script evaluates different guidance schedules to determine the optimal
time-dependent guidance weight for Bayesian optimization.

Key questions:
1. How does guidance at different timesteps affect sample quality?
2. What's the optimal schedule shape (constant, linear, cosine, etc.)?
3. How does the CFG-Zero* fraction affect results?

Schedules tested:
- Constant: λ(t) = λ
- Linear increasing: λ(t) = λ * t
- Linear decreasing: λ(t) = λ * (1 - t)
- Cosine: λ(t) = λ * (1 - cos(πt)) / 2
- Sigmoid: λ(t) = λ * sigmoid(k*(t - 0.5))
- Step: λ(t) = 0 for t < t0, else λ

Metrics:
- Sample statistics (mean, std, L2 norm)
- LCB acquisition value (mean - α * std from GP)
- Diversity (cosine similarity)
- Gradient alignment (correlation between guidance direction and improvement)
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for a guidance schedule."""
    name: str
    schedule_fn: Callable[[int, int, float], float]
    description: str


def constant_schedule(step: int, total_steps: int, guidance_strength: float) -> float:
    """Constant guidance: λ(t) = λ"""
    return guidance_strength


def linear_increasing_schedule(step: int, total_steps: int, guidance_strength: float) -> float:
    """Linear increasing: λ(t) = λ * t"""
    t = step / total_steps
    return guidance_strength * t


def linear_decreasing_schedule(step: int, total_steps: int, guidance_strength: float) -> float:
    """Linear decreasing: λ(t) = λ * (1 - t)"""
    t = step / total_steps
    return guidance_strength * (1 - t)


def cosine_schedule(step: int, total_steps: int, guidance_strength: float) -> float:
    """Cosine: λ(t) = λ * (1 - cos(πt)) / 2 (starts at 0, ends at λ)"""
    t = step / total_steps
    return guidance_strength * (1 - np.cos(np.pi * t)) / 2


def inverse_cosine_schedule(step: int, total_steps: int, guidance_strength: float) -> float:
    """Inverse cosine: λ(t) = λ * (1 + cos(πt)) / 2 (starts at λ, ends at 0)"""
    t = step / total_steps
    return guidance_strength * (1 + np.cos(np.pi * t)) / 2


def sigmoid_schedule(step: int, total_steps: int, guidance_strength: float, k: float = 10) -> float:
    """Sigmoid: λ(t) = λ * sigmoid(k*(t - 0.5))"""
    t = step / total_steps
    return guidance_strength / (1 + np.exp(-k * (t - 0.5)))


def bell_schedule(step: int, total_steps: int, guidance_strength: float) -> float:
    """Bell curve: λ(t) = λ * exp(-((t - 0.5) / 0.2)^2) - peaks at t=0.5"""
    t = step / total_steps
    return guidance_strength * np.exp(-((t - 0.5) / 0.2) ** 2)


def step_schedule(step: int, total_steps: int, guidance_strength: float, start_frac: float = 0.2) -> float:
    """Step: λ(t) = 0 for t < start_frac, else λ"""
    t = step / total_steps
    return guidance_strength if t >= start_frac else 0.0


def cfg_zero_star_wrapper(base_schedule: Callable, zero_init_fraction: float = 0.04):
    """Wrap a schedule with CFG-Zero* (zero for first few steps)."""
    def wrapped(step: int, total_steps: int, guidance_strength: float) -> float:
        zero_init_steps = max(1, int(zero_init_fraction * total_steps))
        if step < zero_init_steps:
            return 0.0
        return base_schedule(step, total_steps, guidance_strength)
    return wrapped


# Define all schedules to test
SCHEDULES = [
    ScheduleConfig("constant", constant_schedule, "λ(t) = λ"),
    ScheduleConfig("constant_cfg0", cfg_zero_star_wrapper(constant_schedule, 0.04), "CFG-Zero* + constant"),
    ScheduleConfig("linear_inc", linear_increasing_schedule, "λ(t) = λ * t"),
    ScheduleConfig("linear_inc_cfg0", cfg_zero_star_wrapper(linear_increasing_schedule, 0.04), "CFG-Zero* + linear inc"),
    ScheduleConfig("linear_dec", linear_decreasing_schedule, "λ(t) = λ * (1 - t)"),
    ScheduleConfig("cosine", cosine_schedule, "λ(t) = λ * (1 - cos(πt)) / 2"),
    ScheduleConfig("cosine_cfg0", cfg_zero_star_wrapper(cosine_schedule, 0.04), "CFG-Zero* + cosine"),
    ScheduleConfig("inv_cosine", inverse_cosine_schedule, "λ(t) = λ * (1 + cos(πt)) / 2"),
    ScheduleConfig("sigmoid", lambda s, t, g: sigmoid_schedule(s, t, g, k=10), "λ(t) = λ * sigmoid(10*(t-0.5))"),
    ScheduleConfig("bell", bell_schedule, "λ(t) = λ * exp(-((t-0.5)/0.2)²)"),
    ScheduleConfig("step_20", lambda s, t, g: step_schedule(s, t, g, 0.2), "λ(t) = λ for t ≥ 0.2"),
]


class GuidedFlowSamplerWithSchedule:
    """
    Flow sampler with configurable guidance schedule for analysis.
    """

    def __init__(
        self,
        flow_model,
        gp_surrogate,
        alpha: float = 1.0,
        norm_stats: Optional[dict] = None,
    ):
        self.flow_model = flow_model
        self.gp = gp_surrogate
        self.alpha = alpha
        self.norm_stats = norm_stats or getattr(flow_model, 'norm_stats', None)

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Convert from flow space to SONAR space."""
        if self.norm_stats is None:
            return z
        mean = self.norm_stats["mean"].to(z.device)
        std = self.norm_stats["std"].to(z.device)
        return z * std + mean

    def _compute_lcb_gradient(self, z_sonar: torch.Tensor) -> torch.Tensor:
        """Compute LCB gradient in flow space using GP's built-in method."""
        if self.gp.model is None:
            return torch.zeros_like(z_sonar)

        # Use GP's lcb_gradient which handles embedding internally
        grad_lcb = self.gp.lcb_gradient(z_sonar, alpha=self.alpha)

        # Scale to flow space
        if self.norm_stats is not None:
            norm_std = self.norm_stats["std"].to(z_sonar.device)
            grad_lcb = grad_lcb / (norm_std + 1e-8)

        # Clip gradient norm
        max_grad_norm = 10.0
        grad_norm = grad_lcb.norm(dim=-1, keepdim=True)
        clip_mask = grad_norm > max_grad_norm
        if clip_mask.any():
            grad_lcb = torch.where(clip_mask, grad_lcb * max_grad_norm / grad_norm, grad_lcb)

        return grad_lcb

    @torch.no_grad()
    def sample_with_schedule(
        self,
        n_samples: int,
        schedule_fn: Callable[[int, int, float], float],
        guidance_strength: float,
        device: str = "cuda",
        num_steps: int = 50,
        return_trajectory: bool = False,
        return_metrics: bool = False,
    ) -> Dict:
        """
        Generate samples with a specific guidance schedule.

        Returns dict with:
        - samples: Final samples in SONAR space
        - trajectory: Optional trajectory [steps+1, B, D]
        - metrics: Optional per-step metrics
        """
        self.flow_model.velocity_net.eval()
        device = torch.device(device)
        input_dim = self.flow_model.input_dim

        # Start from noise
        z = torch.randn(n_samples, input_dim, device=device)

        trajectory = [z.clone()] if return_trajectory else None
        metrics = {
            "step_lambdas": [],
            "step_grad_norms": [],
            "step_velocity_norms": [],
            "step_z_norms": [],
        } if return_metrics else None

        dt = 1.0 / num_steps

        # Heun integration with custom schedule
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            t_next = torch.tensor((i + 1) * dt, device=device)

            # Get lambda for this step
            lambda_t = schedule_fn(i, num_steps, guidance_strength)

            # Predictor step
            v1 = self.flow_model.ode_func(t, z)

            # Add guidance
            grad_lcb = torch.zeros_like(z)
            if lambda_t > 0 and self.gp.model is not None:
                z_sonar = self._denormalize(z)
                grad_lcb = self._compute_lcb_gradient(z_sonar)
                v1 = v1 + lambda_t * grad_lcb

            z_pred = z + v1 * dt

            # Corrector step
            lambda_t_next = schedule_fn(i + 1, num_steps, guidance_strength)
            v2 = self.flow_model.ode_func(t_next, z_pred)

            if lambda_t_next > 0 and self.gp.model is not None:
                z_pred_sonar = self._denormalize(z_pred)
                grad_lcb_pred = self._compute_lcb_gradient(z_pred_sonar)
                v2 = v2 + lambda_t_next * grad_lcb_pred

            # Heun step
            z = z + 0.5 * (v1 + v2) * dt

            if return_trajectory:
                trajectory.append(z.clone())

            if return_metrics:
                metrics["step_lambdas"].append(lambda_t)
                metrics["step_grad_norms"].append(grad_lcb.norm(dim=-1).mean().item())
                metrics["step_velocity_norms"].append(v1.norm(dim=-1).mean().item())
                metrics["step_z_norms"].append(z.norm(dim=-1).mean().item())

        # Denormalize final samples
        z_final = self._denormalize(z)

        result = {"samples": z_final}
        if return_trajectory:
            result["trajectory"] = torch.stack(trajectory, dim=0)
        if return_metrics:
            result["metrics"] = metrics

        return result


def compute_sample_statistics(samples: torch.Tensor) -> Dict[str, float]:
    """Compute statistics of generated samples."""
    sample_mean = samples.mean().item()
    sample_std = samples.std().item()
    l2_norms = torch.norm(samples, dim=1)

    return {
        "mean": sample_mean,
        "std": sample_std,
        "l2_norm_mean": l2_norms.mean().item(),
        "l2_norm_std": l2_norms.std().item(),
    }


def compute_diversity(samples: torch.Tensor) -> Dict[str, float]:
    """Compute diversity metrics via pairwise cosine similarity."""
    normalized = samples / (torch.norm(samples, dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.mm(normalized, normalized.t())
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    similarities = sim_matrix[mask]

    return {
        "cosine_sim_mean": similarities.mean().item(),
        "cosine_sim_std": similarities.std().item(),
        "cosine_sim_max": similarities.max().item(),
    }


def compute_lcb_values(samples: torch.Tensor, gp, alpha: float = 1.0) -> Dict[str, float]:
    """Compute LCB acquisition values for samples using GP's predict method."""
    if gp.model is None:
        return {"lcb_mean": 0.0, "lcb_std": 0.0, "lcb_min": 0.0, "lcb_max": 0.0,
                "gp_mean_mean": 0.0, "gp_std_mean": 0.0}

    with torch.no_grad():
        # Use GP's predict method which handles embedding internally
        mean, std = gp.predict(samples)
        lcb = mean - alpha * std

    return {
        "lcb_mean": lcb.mean().item(),
        "lcb_std": lcb.std().item(),
        "lcb_min": lcb.min().item(),
        "lcb_max": lcb.max().item(),
        "gp_mean_mean": mean.mean().item(),
        "gp_std_mean": std.mean().item(),
    }


def evaluate_schedule(
    sampler: GuidedFlowSamplerWithSchedule,
    schedule: ScheduleConfig,
    guidance_strength: float,
    n_samples: int,
    num_steps: int,
    device: str,
    gp,
    alpha: float,
) -> Dict:
    """Evaluate a single guidance schedule."""
    logger.info(f"  Testing schedule: {schedule.name} ({schedule.description})")

    result = sampler.sample_with_schedule(
        n_samples=n_samples,
        schedule_fn=schedule.schedule_fn,
        guidance_strength=guidance_strength,
        device=device,
        num_steps=num_steps,
        return_metrics=True,
    )

    samples = result["samples"]
    metrics = result["metrics"]

    # Compute all metrics
    stats = compute_sample_statistics(samples)
    diversity = compute_diversity(samples)
    lcb = compute_lcb_values(samples, gp, alpha)

    return {
        "schedule_name": schedule.name,
        "schedule_desc": schedule.description,
        **stats,
        **diversity,
        **lcb,
        "step_lambdas": metrics["step_lambdas"],
        "step_grad_norms": metrics["step_grad_norms"],
        "step_velocity_norms": metrics["step_velocity_norms"],
    }


def load_model_and_gp(
    checkpoint_path: str,
    embeddings_path: str,
    device: str,
) -> Tuple:
    """Load flow model and create GP surrogate with training data."""
    from src.ecoflow.velocity_network import VelocityNetwork
    from src.ecoflow.flow_model import FlowMatchingModel
    from src.ecoflow.gp_surrogate import create_surrogate

    # Load flow model
    logger.info(f"Loading flow model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", 512)
    num_layers = args.get("num_layers", 6)
    num_heads = args.get("num_heads", 8)

    velocity_net = VelocityNetwork(
        input_dim=1024,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Load EMA weights
    if "ema_shadow" in checkpoint:
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = {name: ema_shadow[name] for name in velocity_net.state_dict().keys() if name in ema_shadow}
        velocity_net.load_state_dict(state_dict, strict=False)
    else:
        velocity_net.load_state_dict(checkpoint["model_state_dict"])

    velocity_net = velocity_net.to(device).eval()
    norm_stats = checkpoint.get("norm_stats", None)
    flow_model = FlowMatchingModel(velocity_net, norm_stats=norm_stats)

    # Load embeddings and create GP
    logger.info(f"Loading embeddings from {embeddings_path}")
    data = torch.load(embeddings_path, map_location=device, weights_only=False)

    if isinstance(data, dict):
        X = data["embeddings"].to(device)
        # Support both 'scores' and 'accuracies' keys
        if "scores" in data:
            Y = torch.tensor(data["scores"], device=device, dtype=torch.float32)
        elif "accuracies" in data:
            Y = data["accuracies"].to(device).float()
        else:
            raise ValueError("Expected 'scores' or 'accuracies' key in data")
    else:
        raise ValueError("Expected dict with 'embeddings' and 'scores'/'accuracies' keys")

    logger.info(f"Loaded {X.shape[0]} training points")

    # Create and fit GP
    gp = create_surrogate(method="baxus", D=1024, device=device, target_dim=128)
    gp.fit(X, Y)
    logger.info("GP fitted successfully")

    return flow_model, gp, norm_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze guidance schedule λ(t)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Flow model checkpoint")
    parser.add_argument("--embeddings", type=str, required=True, help="Training embeddings with scores")
    parser.add_argument("--n-samples", type=int, default=64, help="Samples per schedule")
    parser.add_argument("--num-steps", type=int, default=50, help="ODE steps")
    parser.add_argument("--guidance-strength", type=float, default=0.3, help="Base guidance λ")
    parser.add_argument("--alpha", type=float, default=1.0, help="LCB alpha")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output", type=str, default="results/guidance_schedule_analysis.json")
    args = parser.parse_args()

    # Load model and GP
    flow_model, gp, norm_stats = load_model_and_gp(args.checkpoint, args.embeddings, args.device)

    # Create sampler
    sampler = GuidedFlowSamplerWithSchedule(flow_model, gp, alpha=args.alpha, norm_stats=norm_stats)

    # Evaluate all schedules
    results = []
    logger.info(f"\nEvaluating {len(SCHEDULES)} guidance schedules...")
    logger.info(f"Parameters: λ={args.guidance_strength}, α={args.alpha}, steps={args.num_steps}, n={args.n_samples}")

    for schedule in tqdm(SCHEDULES, desc="Schedules"):
        result = evaluate_schedule(
            sampler=sampler,
            schedule=schedule,
            guidance_strength=args.guidance_strength,
            n_samples=args.n_samples,
            num_steps=args.num_steps,
            device=args.device,
            gp=gp,
            alpha=args.alpha,
        )
        results.append(result)

        logger.info(f"    L2 norm: {result['l2_norm_mean']:.4f}, LCB: {result['lcb_mean']:.4f}, "
                    f"Diversity: {result['cosine_sim_mean']:.4f}")

    # Sort by LCB value (higher is better for maximization)
    results_sorted = sorted(results, key=lambda x: x["lcb_mean"], reverse=True)

    # Print summary table
    print("\n" + "=" * 80)
    print("GUIDANCE SCHEDULE ANALYSIS RESULTS")
    print("=" * 80)
    print(f"{'Schedule':<20} {'L2 Norm':>10} {'LCB Mean':>10} {'GP Mean':>10} {'Diversity':>10}")
    print("-" * 80)

    for r in results_sorted:
        print(f"{r['schedule_name']:<20} {r['l2_norm_mean']:>10.4f} {r['lcb_mean']:>10.4f} "
              f"{r['gp_mean_mean']:>10.4f} {r['cosine_sim_mean']:>10.4f}")

    print("-" * 80)
    print(f"\nBest schedule by LCB: {results_sorted[0]['schedule_name']}")
    print(f"  Description: {results_sorted[0]['schedule_desc']}")
    print(f"  LCB mean: {results_sorted[0]['lcb_mean']:.4f}")

    # Analyze effect of guidance timing
    print("\n" + "=" * 80)
    print("GUIDANCE TIMING ANALYSIS")
    print("=" * 80)

    # Compare early vs late guidance
    early_schedules = ["linear_dec", "inv_cosine"]
    late_schedules = ["linear_inc", "cosine", "sigmoid"]

    early_lcb = np.mean([r["lcb_mean"] for r in results if r["schedule_name"] in early_schedules])
    late_lcb = np.mean([r["lcb_mean"] for r in results if r["schedule_name"] in late_schedules])

    print(f"Early guidance (linear_dec, inv_cosine) avg LCB: {early_lcb:.4f}")
    print(f"Late guidance (linear_inc, cosine, sigmoid) avg LCB: {late_lcb:.4f}")

    if late_lcb > early_lcb:
        print("\n=> Late guidance appears more effective!")
        print("   Recommendation: Use increasing schedule with CFG-Zero*")
    else:
        print("\n=> Early guidance appears more effective!")
        print("   Recommendation: Use decreasing schedule")

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "guidance_strength": args.guidance_strength,
            "alpha": args.alpha,
            "num_steps": args.num_steps,
            "n_samples": args.n_samples,
        },
        "results": results_sorted,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Plot visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Schedule shapes
        ax1 = axes[0, 0]
        t_values = np.linspace(0, 1, 50)
        for schedule in SCHEDULES[:6]:  # First 6 schedules
            lambdas = [schedule.schedule_fn(int(t * 50), 50, 0.3) for t in t_values]
            ax1.plot(t_values, lambdas, label=schedule.name)
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("λ(t)")
        ax1.set_title("Guidance Schedule Shapes")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: LCB vs Schedule
        ax2 = axes[0, 1]
        names = [r["schedule_name"] for r in results_sorted]
        lcb_values = [r["lcb_mean"] for r in results_sorted]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax2.barh(names, lcb_values, color=colors)
        ax2.set_xlabel("LCB Mean")
        ax2.set_title("LCB by Schedule (higher = better)")
        ax2.invert_yaxis()

        # Plot 3: L2 Norm vs LCB
        ax3 = axes[1, 0]
        l2_norms = [r["l2_norm_mean"] for r in results]
        lcb_means = [r["lcb_mean"] for r in results]
        ax3.scatter(l2_norms, lcb_means, s=100, c=range(len(results)), cmap='viridis')
        for i, r in enumerate(results):
            ax3.annotate(r["schedule_name"], (l2_norms[i], lcb_means[i]), fontsize=7)
        ax3.set_xlabel("L2 Norm Mean")
        ax3.set_ylabel("LCB Mean")
        ax3.set_title("Sample Quality vs Acquisition Value")
        ax3.axhline(y=0.32, color='r', linestyle='--', alpha=0.5, label='Target L2')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Diversity vs LCB
        ax4 = axes[1, 1]
        diversity = [r["cosine_sim_mean"] for r in results]
        ax4.scatter(diversity, lcb_means, s=100, c=range(len(results)), cmap='viridis')
        for i, r in enumerate(results):
            ax4.annotate(r["schedule_name"], (diversity[i], lcb_means[i]), fontsize=7)
        ax4.set_xlabel("Cosine Similarity (lower = more diverse)")
        ax4.set_ylabel("LCB Mean")
        ax4.set_title("Diversity vs Acquisition Value")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_path}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
