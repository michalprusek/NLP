"""Benchmark infrastructure for comparing BO methods on GuacaMol.

This module provides a unified framework for running and comparing
multiple Bayesian Optimization methods on GuacaMol molecular optimization tasks.

Methods compared:
- Subspace BO: Projects S^255 → S^15 with ArcCosine kernel
- TuRBO: Trust region BO in full R^256 with Matern kernel
- LOLBO: Deep kernel learning GP with optional VAE fine-tuning

Usage:
    # Run single benchmark
    uv run python -m rielbo.benchmark.runner \
        --methods subspace --tasks adip --seeds 42 --iterations 500

    # Run full benchmark
    uv run python -m rielbo.benchmark.runner \
        --methods subspace,turbo,lolbo --tasks all --seeds 42-51

    # Generate convergence plots
    uv run python -m rielbo.benchmark.plotting \
        --results-dir rielbo/results/benchmark --output-dir rielbo/results/benchmark/plots
"""

from rielbo.benchmark.base import BaseBenchmarkMethod

__all__ = ["BaseBenchmarkMethod"]
