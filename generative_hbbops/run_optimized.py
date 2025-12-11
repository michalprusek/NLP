#!/usr/bin/env python3
"""
Optimized HyLO experiment with advanced hyperparameters.

Goal: Find embedding that maximizes EI (closest to global maximum).
"""
import sys
sys.path.insert(0, '/home/prusek/NLP')

from generative_hbbops.config import HyLOConfig
from generative_hbbops.hylo import HyLO

# Optimized configuration for maximum EI exploration
config = HyLOConfig(
    strategy="coordinate_descent",
    n_initial_samples=20,
    output_dir="/home/prusek/NLP/generative_hbbops/results/cd_optimized",

    # More gradient steps for deeper optimization
    cd_n_steps=1000,           # 2x more steps (was 500)
    cd_lr=0.02,                # Higher LR for faster exploration (was 0.01)
    cd_max_iterations=15,      # More CD cycles (was 10)

    # Gradient stability
    use_log_ei=True,           # Log transform for better gradients
    gradient_clip_norm=1.0,    # Clip gradients
    ei_epsilon=1e-4,           # Larger epsilon for stability

    # Aggressive exploration
    cd_n_restarts=15,          # 3x more restarts (was 5)
    perturbation_scale=0.2,    # Larger perturbations (was 0.1)

    # Advanced optimization
    use_lr_schedule=True,      # Cosine annealing LR
    lr_min_factor=0.05,        # LR decays to 5% of initial
    use_warm_restarts=True,    # Periodically reset LR
    warm_restart_period=200,   # Reset every 200 steps
    adaptive_perturbation=True,# Increase perturbation when stuck
    max_perturbation_scale=0.5,# Max perturbation scale

    # Keep standard settings
    seed=42,
    save_visualizations=True,
)

print("=" * 60)
print("OPTIMIZED HyLO CONFIGURATION")
print("=" * 60)
print(f"Strategy: {config.strategy}")
print(f"Samples: {config.n_initial_samples} (best)")
print(f"Steps: {config.cd_n_steps}, LR: {config.cd_lr}")
print(f"Restarts: {config.cd_n_restarts}")
print(f"LR Schedule: {config.use_lr_schedule}, Warm Restarts: {config.use_warm_restarts}")
print(f"Adaptive Perturbation: {config.adaptive_perturbation}")
print("=" * 60)

hylo = HyLO(config)
results = hylo.run(select_best=True)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Final EI: {results.get('final_ei', 'N/A')}")
print(f"Predicted error rate: {results.get('predicted_error_rate', 'N/A')}")
print(f"Results saved to: {config.output_dir}")
