"""
InstructZero: Bayesian Optimization for Prompt Optimization

Based on InstructZero (ICML 2024) - optimizes prompts in low-dimensional
soft prompt space, using an LLM to decode soft prompts to instructions.

Key idea:
- Soft prompt (10D vector) → Random projection → LLM generates instruction
- Bayesian Optimization in low-dimensional space
- Black-box evaluation on target task (GSM8K)

Usage:
    uv run python -m instructzero.run --max-calls 50000
"""

from .loop import InstructZeroLoop

__all__ = ["InstructZeroLoop"]
