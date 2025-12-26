#!/usr/bin/env python3
"""
CLI entry point for HyLO2: Latent Space Optimization.

Example usage:
    # Basic run with 4 samples
    uv run python generative_hbbops_2/run_hylo2.py

    # Use all samples and select best
    uv run python generative_hbbops_2/run_hylo2.py --use-all-samples --select-best

    # Custom configuration
    uv run python generative_hbbops_2/run_hylo2.py \
        --n-samples 10 \
        --reconstruction-weight 0.5 \
        --warmup-epochs 1000 \
        --latent-lr 0.05
"""
from generative_hbbops_2.hylo2 import main

if __name__ == "__main__":
    main()
