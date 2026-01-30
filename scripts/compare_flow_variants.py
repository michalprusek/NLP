#!/usr/bin/env python3
"""
Flow Variants Comparison Script for Guided Sampling Analysis.

This script evaluates the current trained flow model under GP guidance conditions
and compares training costs between standard CFM and OT-CFM.

IMPORTANT: This script compares the CURRENT standard CFM model with and without
guidance. It does NOT compare against reflow models (which don't exist yet -
reflow would be implemented per 09-REFLOW-IMPLEMENTATION-PLAN.md if trajectory
straightness is poor).

Expected Findings (from 09-RESEARCH.md):
========================================

1. STRAIGHTNESS
   - Expected: 0.9+ for well-trained flow (near-straight trajectories)
   - Guidance degradation: Straightness drops ~5-10% with lambda=2.0
   - Interpretation:
     * Straightness > 0.95: Flow is well-trained, no reflow needed
     * Straightness 0.8-0.95: Consider reflow
     * Straightness < 0.8: Reflow strongly recommended

2. GUIDANCE EFFECTIVENESS
   - L2 deviation from unguided samples increases with lambda
   - GP acquisition value (UCB/LCB) should improve with guidance
   - CFG-Zero* (zero guidance for first 4%) prevents early trajectory corruption

3. TIMING (OT-CFM vs CFM)
   - OT-CFM is 10-100x slower than CFM for batch_size >= 256
   - O(n^3) complexity for exact OT (Hungarian algorithm)
   - Standard CFM is O(n) per batch

Key Metrics for Paper:
======================
- Trajectory straightness ratio (1.0 = perfect)
- Guidance-induced deviation (L2 from unguided)
- Training cost ratio (OT-CFM / CFM)

Modes:
======
--mode straightness : Measure trajectory straightness with/without guidance
--mode guidance     : Compare guidance effectiveness at different strengths
--mode timing       : Benchmark CFM vs OT-CFM training cost
--mode all          : Run all comparisons

Usage:
    python scripts/compare_flow_variants.py \
        --checkpoint results/flow_checkpoints/best_flow.pt \
        --mode all \
        --n-samples 500 \
        --output-dir results/flow_variant_comparison/

    # Quick validation of research predictions
    python scripts/compare_flow_variants.py --validate-research

Author: EcoFlow Team
Date: 2026-01-30
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Any

import torch
import numpy as np

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading
# =============================================================================

def load_flow_model(
    checkpoint_path: str,
    device: str = "cuda",
    use_ema: bool = True,
):
    """Load FlowMatchingModel from checkpoint."""
    from src.ecoflow.velocity_network import VelocityNetwork
    from src.ecoflow.flow_model import FlowMatchingModel

    logger.info(f"Loading checkpoint: {checkpoint_path}")
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

    if use_ema and "ema_shadow" in checkpoint:
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = {}
        for name, param in velocity_net.named_parameters():
            if name in ema_shadow:
                state_dict[name] = ema_shadow[name]
        velocity_net.load_state_dict(state_dict, strict=False)
    else:
        velocity_net.load_state_dict(checkpoint["model_state_dict"])

    velocity_net = velocity_net.to(device)
    velocity_net.eval()

    norm_stats = checkpoint.get("norm_stats", None)
    model = FlowMatchingModel(velocity_net, norm_stats=norm_stats)

    return model


def create_dummy_gp(device: str = "cuda"):
    """Create a dummy GP surrogate for testing guidance without real data."""
    from src.ecoflow.gp_surrogate import SonarGPSurrogate

    gp = SonarGPSurrogate(D=1024, device=device)

    # Create synthetic training data
    n_train = 50
    train_X = torch.randn(n_train, 1024, device=device)
    train_Y = torch.randn(n_train, device=device)

    gp.fit(train_X, train_Y)
    return gp


# =============================================================================
# Straightness Comparison
# =============================================================================

def compute_trajectory_with_guidance(
    flow_model,
    gp_surrogate,
    n_samples: int,
    num_steps: int,
    guidance_strength: float,
    alpha: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate trajectories with GP guidance, recording all steps.

    Args:
        flow_model: FlowMatchingModel
        gp_surrogate: GP surrogate for guidance
        n_samples: Number of trajectories
        num_steps: Number of ODE steps
        guidance_strength: Lambda for guidance
        alpha: UCB exploration weight
        device: Device

    Returns:
        trajectory: [num_steps+1, n_samples, input_dim]
    """
    from src.ecoflow.guided_flow import cfg_zero_star_schedule

    flow_model.velocity_net.eval()
    input_dim = flow_model.input_dim

    z = torch.randn(n_samples, input_dim, device=device)
    trajectory = [z.clone()]
    dt = 1.0 / num_steps

    # Get norm_stats for denormalization
    norm_stats = flow_model.norm_stats

    with torch.no_grad():
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)

            # Base velocity
            v = flow_model.ode_func(t, z)

            # Add guidance if not in zero-init phase
            lambda_t = cfg_zero_star_schedule(i, num_steps, guidance_strength, 0.04)
            if lambda_t > 0 and gp_surrogate.model is not None:
                # Denormalize for GP
                if norm_stats is not None:
                    z_sonar = z * norm_stats["std"].to(device) + norm_stats["mean"].to(device)
                else:
                    z_sonar = z

                # Compute UCB gradient
                grad_ucb = gp_surrogate.ucb_gradient(z_sonar, alpha=alpha)

                # Scale to flow space
                if norm_stats is not None:
                    grad_ucb = grad_ucb / (norm_stats["std"].to(device) + 1e-8)

                # Clip gradient
                max_grad_norm = 10.0
                grad_norm = grad_ucb.norm(dim=-1, keepdim=True)
                grad_ucb = torch.where(
                    grad_norm > max_grad_norm,
                    grad_ucb * max_grad_norm / grad_norm,
                    grad_ucb
                )

                v = v + lambda_t * grad_ucb

            z = z + v * dt
            trajectory.append(z.clone())

    return torch.stack(trajectory, dim=0)


def compute_straightness(trajectory: torch.Tensor) -> dict:
    """Compute straightness metrics."""
    diffs = trajectory[1:] - trajectory[:-1]
    step_lengths = diffs.norm(dim=-1)
    path_lengths = step_lengths.sum(dim=0)
    straight_distances = (trajectory[-1] - trajectory[0]).norm(dim=-1)
    straightness = straight_distances / (path_lengths + 1e-8)

    return {
        "mean": float(straightness.mean()),
        "std": float(straightness.std()),
        "min": float(straightness.min()),
        "max": float(straightness.max()),
    }


def run_straightness_comparison(
    checkpoint_path: str,
    n_samples: int = 500,
    num_steps: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Compare trajectory straightness with and without GP guidance.

    Tests guidance strengths: [0.0, 0.5, 1.0, 2.0]
    """
    logger.info("Running straightness comparison...")

    model = load_flow_model(checkpoint_path, device=device)
    gp = create_dummy_gp(device=device)

    results = {"guidance_strengths": [], "straightness": [], "raw_metrics": []}

    for lambda_val in [0.0, 0.5, 1.0, 2.0]:
        logger.info(f"  Testing lambda={lambda_val}...")

        trajectory = compute_trajectory_with_guidance(
            model, gp, n_samples, num_steps, lambda_val, device=device
        )
        metrics = compute_straightness(trajectory.cpu())

        results["guidance_strengths"].append(lambda_val)
        results["straightness"].append(metrics["mean"])
        results["raw_metrics"].append(metrics)

        logger.info(f"    Straightness: {metrics['mean']:.4f} +/- {metrics['std']:.4f}")

    # Compute degradation
    baseline_straightness = results["straightness"][0]  # lambda=0
    results["baseline_straightness"] = baseline_straightness

    if baseline_straightness >= 0.95:
        results["interpretation"] = "EXCELLENT - No reflow needed"
    elif baseline_straightness >= 0.8:
        results["interpretation"] = "GOOD - Consider reflow for improvement"
    else:
        results["interpretation"] = "POOR - Reflow strongly recommended"

    # Degradation at lambda=2.0
    degradation = (baseline_straightness - results["straightness"][-1]) / baseline_straightness * 100
    results["degradation_at_lambda_2"] = degradation
    logger.info(f"  Degradation at lambda=2.0: {degradation:.1f}%")

    return results


# =============================================================================
# Guidance Effectiveness
# =============================================================================

def run_guidance_comparison(
    checkpoint_path: str,
    n_samples: int = 500,
    num_steps: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Compare guidance effectiveness at different strengths.

    Measures:
    - L2 deviation from unguided samples
    - GP acquisition value improvement
    """
    logger.info("Running guidance effectiveness comparison...")

    model = load_flow_model(checkpoint_path, device=device)
    gp = create_dummy_gp(device=device)

    # Generate unguided baseline
    logger.info("  Generating unguided samples...")
    baseline_trajectory = compute_trajectory_with_guidance(
        model, gp, n_samples, num_steps, 0.0, device=device
    )
    baseline_samples = baseline_trajectory[-1].cpu()  # Final samples

    results = {
        "guidance_strengths": [],
        "l2_deviation": [],
        "ucb_values": [],
    }

    for lambda_val in [0.0, 0.5, 1.0, 2.0]:
        logger.info(f"  Testing lambda={lambda_val}...")

        trajectory = compute_trajectory_with_guidance(
            model, gp, n_samples, num_steps, lambda_val, device=device
        )
        final_samples = trajectory[-1]

        # L2 deviation from baseline
        # Match by initial noise (same seed would give same initial)
        # For this comparison, we measure distance from baseline
        l2_deviation = (final_samples.cpu() - baseline_samples).norm(dim=-1).mean()

        # GP UCB value at final samples
        # Need to denormalize first
        if model.norm_stats is not None:
            samples_sonar = final_samples * model.norm_stats["std"].to(device) + model.norm_stats["mean"].to(device)
        else:
            samples_sonar = final_samples

        mean, std = gp.predict(samples_sonar)
        ucb = mean + 1.0 * std  # UCB with alpha=1.0

        results["guidance_strengths"].append(lambda_val)
        results["l2_deviation"].append(float(l2_deviation))
        results["ucb_values"].append({
            "mean": float(ucb.mean()),
            "std": float(ucb.std()),
            "max": float(ucb.max()),
        })

        logger.info(f"    L2 deviation: {l2_deviation:.4f}")
        logger.info(f"    UCB mean: {ucb.mean():.4f}")

    return results


# =============================================================================
# Timing Comparison (CFM vs OT-CFM)
# =============================================================================

def run_timing_comparison(
    batch_sizes: list[int] = [64, 128, 256, 512, 1024],
    n_iters: int = 50,
    dim: int = 1024,
    device: str = "cpu",  # CPU for fair comparison without GPU variability
) -> dict:
    """
    Benchmark standard CFM vs OT-CFM training cost.

    OT-CFM uses ExactOptimalTransportConditionalFlowMatcher which has O(n^3)
    complexity for the Hungarian algorithm.

    Standard CFM uses ConditionalFlowMatcher with O(n) complexity.
    """
    logger.info("Running timing comparison (CFM vs OT-CFM)...")

    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher,
    )

    results = {
        "batch_sizes": [],
        "cfm_times": [],
        "ot_cfm_times": [],
        "speedup_ratios": [],
    }

    cfm = ConditionalFlowMatcher(sigma=0.0)

    for batch_size in batch_sizes:
        logger.info(f"  Batch size {batch_size}...")

        x0 = torch.randn(batch_size, dim)
        x1 = torch.randn(batch_size, dim)

        # Benchmark standard CFM
        start = time.time()
        for _ in range(n_iters):
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
        cfm_time = (time.time() - start) / n_iters

        # Benchmark OT-CFM (only for reasonable batch sizes)
        if batch_size <= 512:
            ot_cfm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
            start = time.time()
            for _ in range(n_iters):
                t, xt, ut = ot_cfm.sample_location_and_conditional_flow(x0, x1)
            ot_cfm_time = (time.time() - start) / n_iters
        else:
            # Estimate based on O(n^3) scaling
            # Time at 512: t_512, Time at batch_size: t_512 * (batch_size/512)^3
            ref_time = results["ot_cfm_times"][-1] if results["ot_cfm_times"] else 1.0
            ot_cfm_time = ref_time * (batch_size / 512) ** 3
            logger.info(f"    OT-CFM estimated (batch > 512): {ot_cfm_time:.4f}s")

        speedup = ot_cfm_time / cfm_time if cfm_time > 0 else float('inf')

        results["batch_sizes"].append(batch_size)
        results["cfm_times"].append(cfm_time)
        results["ot_cfm_times"].append(ot_cfm_time)
        results["speedup_ratios"].append(speedup)

        logger.info(f"    CFM: {cfm_time*1000:.2f}ms, OT-CFM: {ot_cfm_time*1000:.2f}ms")
        logger.info(f"    Speedup: {speedup:.1f}x")

    # Analysis
    results["summary"] = {
        "max_speedup": max(results["speedup_ratios"]),
        "avg_speedup": np.mean(results["speedup_ratios"]),
        "complexity_analysis": "OT-CFM is O(n^3), CFM is O(n)",
    }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_straightness_comparison(results: dict, output_path: str) -> None:
    """Plot straightness vs guidance strength."""
    fig, ax = plt.subplots(figsize=(8, 5))

    lambdas = results["guidance_strengths"]
    straightness = results["straightness"]

    ax.plot(lambdas, straightness, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='Excellent (>0.95)')
    ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='Good (>0.80)')

    ax.set_xlabel('Guidance Strength (lambda)', fontsize=12)
    ax.set_ylabel('Trajectory Straightness', fontsize=12)
    ax.set_title('Trajectory Straightness vs GP Guidance Strength', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_guidance_effect(results: dict, output_path: str) -> None:
    """Plot guidance effectiveness metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    lambdas = results["guidance_strengths"]
    l2_dev = results["l2_deviation"]
    ucb_means = [r["mean"] for r in results["ucb_values"]]

    # L2 deviation
    ax1.plot(lambdas, l2_dev, 'o-', linewidth=2, markersize=8, color='#E94F37')
    ax1.set_xlabel('Guidance Strength (lambda)', fontsize=12)
    ax1.set_ylabel('L2 Deviation from Unguided', fontsize=12)
    ax1.set_title('Guidance-Induced Deviation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # UCB values
    ax2.plot(lambdas, ucb_means, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax2.set_xlabel('Guidance Strength (lambda)', fontsize=12)
    ax2.set_ylabel('Mean UCB Value', fontsize=12)
    ax2.set_title('GP Acquisition Value', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_timing_comparison(results: dict, output_path: str) -> None:
    """Plot CFM vs OT-CFM timing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    batch_sizes = results["batch_sizes"]
    cfm_times = [t * 1000 for t in results["cfm_times"]]  # Convert to ms
    ot_cfm_times = [t * 1000 for t in results["ot_cfm_times"]]
    speedups = results["speedup_ratios"]

    # Timing comparison
    x = np.arange(len(batch_sizes))
    width = 0.35

    ax1.bar(x - width/2, cfm_times, width, label='Standard CFM', color='#2E86AB')
    ax1.bar(x + width/2, ot_cfm_times, width, label='OT-CFM', color='#E94F37')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Time per Batch (ms)', fontsize=12)
    ax1.set_title('Training Cost: CFM vs OT-CFM', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Speedup ratio
    ax2.plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=8, color='#44AF69')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Speedup Ratio (OT-CFM / CFM)', fontsize=12)
    ax2.set_title('CFM Speedup over OT-CFM', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Validation Mode
# =============================================================================

def run_quick_validation(device: str = "cpu") -> dict:
    """
    Run quick validation to check if findings match 09-RESEARCH.md predictions.

    Expected predictions (from 09-RESEARCH.md):
    ==================================================
    1. OT-CFM is 10-100x slower than CFM for batch_size >= 256
    2. OT-CFM shows O(n^3) scaling (time ~8x when batch doubles)
    3. Standard CFM shows O(n) scaling (time ~2x when batch doubles)
    4. CFG-Zero* schedule applies zero guidance for first 4% of steps

    Interpretation thresholds:
    ==================================================
    - Straightness > 0.95: EXCELLENT - No reflow needed
    - Straightness 0.8-0.95: GOOD - Consider reflow
    - Straightness < 0.8: POOR - Reflow strongly recommended

    This validation runs in < 60 seconds using small batch sizes and iterations.
    """
    import time as validation_time
    start_time = validation_time.time()

    logger.info("=" * 60)
    logger.info("RESEARCH VALIDATION MODE")
    logger.info("=" * 60)
    logger.info("Validating predictions from 09-RESEARCH.md\n")

    results = {"predictions": [], "status": [], "details": []}

    # =================================================================
    # Prediction 1: OT-CFM is slower than CFM at batch_size=256
    # =================================================================
    logger.info("[1] Testing: OT-CFM is slower than CFM at large batch sizes")
    timing_results = run_timing_comparison(
        batch_sizes=[64, 128, 256],
        n_iters=10,  # Reduced for speed
        device=device,
    )

    idx_256 = timing_results["batch_sizes"].index(256)
    speedup_256 = timing_results["speedup_ratios"][idx_256]

    if speedup_256 >= 10:
        status = "VALIDATED"
        detail = f"Speedup {speedup_256:.1f}x >= 10x threshold"
    elif speedup_256 >= 5:
        status = "PARTIAL"
        detail = f"Speedup {speedup_256:.1f}x - significant but below 10x threshold"
    else:
        status = "UNEXPECTED"
        detail = f"Speedup only {speedup_256:.1f}x - expected >= 10x"

    results["predictions"].append("OT-CFM >= 10x slower at batch_size=256")
    results["status"].append(status)
    results["details"].append(detail)
    logger.info(f"  Speedup at batch_size=256: {speedup_256:.1f}x")
    logger.info(f"  Status: {status}")

    # =================================================================
    # Prediction 2: OT-CFM shows O(n^3) scaling
    # =================================================================
    logger.info("\n[2] Testing: OT-CFM shows O(n^3) scaling")

    t1 = timing_results["ot_cfm_times"][0]  # batch_size=64
    t2 = timing_results["ot_cfm_times"][1]  # batch_size=128

    # Expected ratio: (128/64)^3 = 8x
    actual_ratio = t2 / t1 if t1 > 0 else float('inf')

    if 4.0 <= actual_ratio <= 16.0:  # Within 2x of expected 8x
        status = "VALIDATED"
        detail = f"Time ratio {actual_ratio:.1f}x matches O(n^3) (expected ~8x)"
    else:
        status = "UNEXPECTED"
        detail = f"Time ratio {actual_ratio:.1f}x deviates from O(n^3) expected ~8x"

    results["predictions"].append("OT-CFM shows O(n^3) scaling (8x for 2x batch)")
    results["status"].append(status)
    results["details"].append(detail)
    logger.info(f"  Time ratio (128/64): {actual_ratio:.1f}x (expected ~8x)")
    logger.info(f"  Status: {status}")

    # =================================================================
    # Prediction 3: Standard CFM shows O(n) scaling
    # =================================================================
    logger.info("\n[3] Testing: Standard CFM shows O(n) scaling")

    cfm_t1 = timing_results["cfm_times"][0]  # batch_size=64
    cfm_t2 = timing_results["cfm_times"][1]  # batch_size=128

    # Expected ratio: (128/64) = 2x for O(n)
    cfm_ratio = cfm_t2 / cfm_t1 if cfm_t1 > 0 else float('inf')

    if 1.0 <= cfm_ratio <= 4.0:  # Within 2x of expected 2x
        status = "VALIDATED"
        detail = f"Time ratio {cfm_ratio:.1f}x matches O(n) (expected ~2x)"
    else:
        status = "UNEXPECTED"
        detail = f"Time ratio {cfm_ratio:.1f}x deviates from O(n) expected ~2x"

    results["predictions"].append("Standard CFM shows O(n) scaling (2x for 2x batch)")
    results["status"].append(status)
    results["details"].append(detail)
    logger.info(f"  Time ratio (128/64): {cfm_ratio:.1f}x (expected ~2x)")
    logger.info(f"  Status: {status}")

    # =================================================================
    # Prediction 4: CFG-Zero* schedule is correctly implemented
    # =================================================================
    logger.info("\n[4] Testing: CFG-Zero* schedule applies zero guidance for first 4%")

    try:
        from src.ecoflow.guided_flow import cfg_zero_star_schedule
    except ModuleNotFoundError:
        # Add project root to path for standalone execution
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.ecoflow.guided_flow import cfg_zero_star_schedule

    # Test schedule at various steps for 50-step ODE
    total_steps = 50
    zero_init_fraction = 0.04
    guidance_strength = 1.0

    # First 2 steps (0, 1) should have zero guidance
    step_0 = cfg_zero_star_schedule(0, total_steps, guidance_strength, zero_init_fraction)
    step_1 = cfg_zero_star_schedule(1, total_steps, guidance_strength, zero_init_fraction)
    # Step 2+ should have full guidance
    step_2 = cfg_zero_star_schedule(2, total_steps, guidance_strength, zero_init_fraction)
    step_49 = cfg_zero_star_schedule(49, total_steps, guidance_strength, zero_init_fraction)

    if step_0 == 0.0 and step_1 == 0.0 and step_2 == guidance_strength and step_49 == guidance_strength:
        status = "VALIDATED"
        detail = f"Steps 0-1 have lambda=0, steps 2+ have lambda={guidance_strength}"
    else:
        status = "UNEXPECTED"
        detail = f"Schedule mismatch: step_0={step_0}, step_1={step_1}, step_2={step_2}"

    results["predictions"].append("CFG-Zero* applies zero guidance for first 4% of steps")
    results["status"].append(status)
    results["details"].append(detail)
    logger.info(f"  Step 0: {step_0} (expected 0.0)")
    logger.info(f"  Step 1: {step_1} (expected 0.0)")
    logger.info(f"  Step 2: {step_2} (expected {guidance_strength})")
    logger.info(f"  Status: {status}")

    # =================================================================
    # Summary
    # =================================================================
    elapsed = validation_time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    validated_count = sum(1 for s in results["status"] if s == "VALIDATED")
    partial_count = sum(1 for s in results["status"] if s == "PARTIAL")
    unexpected_count = sum(1 for s in results["status"] if s == "UNEXPECTED")
    total_count = len(results["status"])

    for pred, stat, detail in zip(results["predictions"], results["status"], results["details"]):
        if stat == "VALIDATED":
            icon = "[OK]"
        elif stat == "PARTIAL":
            icon = "[~]"
        else:
            icon = "[!]"
        logger.info(f"  {icon} {pred}")
        logger.info(f"      -> {detail}")

    logger.info(f"\nResults: {validated_count} validated, {partial_count} partial, {unexpected_count} unexpected")
    logger.info(f"Duration: {elapsed:.1f}s")

    results["summary"] = {
        "validated": validated_count,
        "partial": partial_count,
        "unexpected": unexpected_count,
        "total": total_count,
        "duration_seconds": elapsed,
    }
    results["overall"] = f"{validated_count}/{total_count} validated"

    if unexpected_count == 0:
        logger.info("\nConclusion: Research predictions from 09-RESEARCH.md are confirmed!")
    elif unexpected_count <= 1:
        logger.info("\nConclusion: Most predictions confirmed, minor deviations noted.")
    else:
        logger.info("\nConclusion: Several unexpected results - review 09-RESEARCH.md assumptions.")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare flow variants for guided sampling"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to flow model checkpoint (required for straightness/guidance modes)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["straightness", "guidance", "timing", "all"],
        help="Comparison mode (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples for straightness/guidance comparison (default: 500)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of ODE integration steps (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flow_variant_comparison",
        help="Output directory for results and figures",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="64,128,256,512,1024",
        help="Comma-separated batch sizes for timing comparison",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=50,
        help="Number of iterations for timing benchmark (default: 50)",
    )
    parser.add_argument(
        "--validate-research",
        action="store_true",
        help="Run quick validation of 09-RESEARCH.md predictions",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Quick validation mode
    if args.validate_research:
        validation_results = run_quick_validation(device="cpu")

        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2)

        return

    # Run selected comparisons
    all_results: dict[str, Any] = {}

    if args.mode in ["straightness", "all"]:
        if args.checkpoint is None:
            logger.warning("Skipping straightness comparison - no checkpoint provided")
        else:
            straightness_results = run_straightness_comparison(
                args.checkpoint,
                n_samples=args.n_samples,
                num_steps=args.num_steps,
                device=args.device,
            )
            all_results["straightness"] = straightness_results

            plot_straightness_comparison(
                straightness_results,
                str(output_dir / "straightness_with_guidance.png")
            )

    if args.mode in ["guidance", "all"]:
        if args.checkpoint is None:
            logger.warning("Skipping guidance comparison - no checkpoint provided")
        else:
            guidance_results = run_guidance_comparison(
                args.checkpoint,
                n_samples=args.n_samples,
                num_steps=args.num_steps,
                device=args.device,
            )
            all_results["guidance"] = guidance_results

            plot_guidance_effect(
                guidance_results,
                str(output_dir / "guidance_effect.png")
            )

    if args.mode in ["timing", "all"]:
        timing_results = run_timing_comparison(
            batch_sizes=batch_sizes,
            n_iters=args.n_iters,
            device="cpu",  # Use CPU for consistent timing
        )
        all_results["timing"] = timing_results

        plot_timing_comparison(
            timing_results,
            str(output_dir / "timing_comparison.png")
        )

    # Save all results
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {output_dir / 'comparison_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("FLOW VARIANT COMPARISON COMPLETE")
    print("=" * 60)

    if "straightness" in all_results:
        sr = all_results["straightness"]
        print(f"\nStraightness:")
        print(f"  Baseline (lambda=0): {sr['baseline_straightness']:.4f}")
        print(f"  Degradation at lambda=2.0: {sr['degradation_at_lambda_2']:.1f}%")
        print(f"  Interpretation: {sr['interpretation']}")

    if "guidance" in all_results:
        gr = all_results["guidance"]
        print(f"\nGuidance Effect:")
        print(f"  L2 deviation at lambda=2.0: {gr['l2_deviation'][-1]:.4f}")

    if "timing" in all_results:
        tr = all_results["timing"]
        print(f"\nTiming (CFM vs OT-CFM):")
        print(f"  Max speedup: {tr['summary']['max_speedup']:.1f}x")
        print(f"  Average speedup: {tr['summary']['avg_speedup']:.1f}x")
        print(f"  Complexity: {tr['summary']['complexity_analysis']}")

    print("=" * 60)
    print(f"Figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
