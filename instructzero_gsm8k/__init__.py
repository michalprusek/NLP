"""
InstructZero GSM8K: Adapted InstructZero for GSM8K evaluation.

Uses official InstructZero code from GitHub with minimal modifications:
- GSM8K data loading instead of instruction induction
- vLLM for evaluation instead of ChatGPT
- Preserves: soft prompts, instruction-coupled kernel, CMA-ES, BO loop
"""

__all__ = []
