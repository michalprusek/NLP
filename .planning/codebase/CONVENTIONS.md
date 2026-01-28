# Coding Conventions

**Analysis Date:** 2026-01-28

## Naming Patterns

**Files:**
- Snake case: `llm_client.py`, `gsm8k_evaluator.py`, `velocity_network.py`
- Module files in subdirectories follow same pattern: `ecoflow_bo/config.py`, `ecoflow_bo/encoder.py`
- Run scripts: `run_opro.py`, `run_protegi.py`, `run_gepa.py`

**Classes:**
- PascalCase for all classes: `LLMClient`, `VLLMClient`, `OpenAIClient`, `DeepInfraClient`
- PascalCase for configuration dataclasses: `EncoderConfig`, `DiTVelocityNetConfig`, `DecoderConfig`, `PerceiverDecoderConfig`
- PascalCase for exception/special classes: `ResidualBlock`, `MatryoshkaEncoder`, `VelocityNetwork`

**Functions:**
- Snake case: `create_llm_client()`, `bucket_score()`, `normalize_answer()`, `extract_answer()`, `extract_ground_truth()`
- Private functions prefixed with underscore: `_patch_vllm_platform()`, `_format_prompt()`, `_truncate_prompt()`, `_get_random_examples()`
- All lowercase for utility functions: `ensure_active_dims_list()`, `clamp_to_search_bounds()`

**Variables:**
- Snake case for local variables: `model_name`, `tensor_parallel_size`, `gpu_memory_utilization`, `scored_prompts`
- UPPERCASE for module-level constants: `STRICT_PATTERN`, `BOXED_PATTERN`, `FLEXIBLE_PATTERN`, `META_PROMPT_PATH`, `DEFAULT_MODEL`
- Abbreviations used directly: `z` (latent), `mu` (mean), `v` (velocity), `t` (time), `B` (batch size)
- Underscores in multi-word constants: `NUM_CANDIDATES_PER_ITER`, `META_PROMPT_TEMPLATE`

**Types:**
- Type hints use full names: `List[str]`, `Dict[str, Any]`, `Tuple[float, Dict]`, `Optional[str]`
- Generic types from `typing`: `List`, `Dict`, `Any`, `Tuple`, `Optional`

## Code Style

**Formatting:**
- Line length: appears unconstrained, but lines generally 80-100 chars (observed in core files)
- No explicit formatter detected (no .prettierrc, ruff.toml, or black config)
- Imports organized but no strict tool enforcement

**Indentation:**
- 4 spaces consistently throughout codebase
- Python 3.10+ (requires-python = ">=3.10,<3.14" in pyproject.toml)

**Docstrings:**
- Module-level docstrings at file start: `"""Module description and context"""` (triple-quoted)
- Class docstrings: `"""Brief description of class purpose"""` (single line or detailed)
- Function docstrings follow standard format with Args/Returns sections:

```python
def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output using robust methodology.

    Priority order (matches Qwen output formats):
    1. #### (number) - standard GSM8K format
    2. \\boxed{number} - LaTeX format (common in Qwen math outputs)
    3. Last number in text - fallback for "The answer is X"
    """
```

- Parameter types in docstring when not obvious from type hint
- Return type documented in Returns section

**Comments:**
- Inline comments explain "why" not "what": `# Patch vLLM platform before importing LLM`
- Section markers for organizing code: `# ============================================================================`
- Block comments above code sections describe intent
- TODO/FIXME comments are minimal; code assumes completion
- Comments include rationale for non-obvious choices (e.g., NOTE about GPU cleanup)

## Import Organization

**Order:**
1. Standard library: `abc`, `typing`, `os`, `json`, `re`, `random`, `math`, `gc`
2. Third-party: `torch`, `vllm`, `datasets`, `openai`
3. Local imports: `from .config import`, `from .encoder import`

**Pattern:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import os
import json
from pathlib import Path
from datetime import datetime

# Then third-party
import torch
import torch.nn as nn
from vllm import LLM

# Then local
from .config import EncoderConfig
from .losses import KLDivergenceLoss
```

**Path Aliases:**
- No path aliases configured (`baseUrl` not detected in any config)
- Relative imports within packages: `from .config import EncoderConfig`
- Absolute imports from project root for run scripts: `from src.llm_client import create_llm_client`

## Error Handling

**Patterns:**
- Exceptions raised with descriptive messages: `raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}")`
- Specific exception types used: `ValueError`, `KeyboardInterrupt`, `RuntimeError`, `ImportError`, `AttributeError`
- KeyboardInterrupt never swallowed: `except KeyboardInterrupt: raise`
- Fallback patterns in extraction: Try primary method, fall back to secondary, return None if all fail

```python
# Example from gsm8k_evaluator.py - priority-based extraction
match = re.search(STRICT_PATTERN, text)
if match:
    return normalize_answer(match.group(1))

match = re.search(BOXED_PATTERN, text)
if match:
    return normalize_answer(match.group(1))

numbers = re.findall(FLEXIBLE_PATTERN, text)
if numbers:
    return normalize_answer(numbers[-1])

return None
```

- No silent failures; errors are logged or re-raised
- Warning messages printed to console with context: `print(f"[ERROR] OpenAI {error_type} on prompt {i+1}/{len(prompts)}: {e}")`

## Logging

**Framework:** `print()` for normal output, `logging` module in specific cases (e.g., `src/gepa.py`)

**Patterns:**
- Progress printed with iteration counts: `print(f"Iteration {iteration + 1}/{self.num_iterations}")`
- State changes logged: `print(f"Fixed evaluation set: {len(self.fixed_eval_set)} examples")`
- Model loading info: `print(f"Loading model with vLLM: {model_name}")`
- Full prompt logging per CLAUDE.md - prompts NEVER truncated in logs:
  ```python
  # GOOD - always log full prompt
  print(f"Generated:\n{prompt}")

  # BAD - never truncate
  # print(f"Generated: {prompt[:80]}...")
  ```
- Status markers: `print(f"📄 Saved: {filepath}")`, `print(f"🤖 RESPONSE:\n{response}")`
- Formatted separators: `print(f"\n{'='*60}")`

## Functions Design

**Size:** Generally 15-60 lines; larger functions (>100 lines) are optimization loops that naturally span iterations

**Parameters:**
- Positional parameters for required args: `def __init__(self, model_name: str, ...)`
- Keyword-only with defaults for optional: `temperature=0.0`, `use_tqdm=False`
- `**kwargs` used for flexibility in LLM generation: `def generate(self, prompt: str, **kwargs) -> str`

**Return Values:**
- Single values: `-> str`, `-> float`, `-> bool`
- Tuples for multiple related outputs: `-> Tuple[str, List[Dict]]`, `-> Tuple[float, Dict]`
- Dictionaries for complex structured data: `-> Dict[str, Any]`
- None returned explicitly for "not found" cases: `return None`

## Module Design

**Exports:**
- No explicit `__all__` defined; classes and functions are public by naming convention
- Factory functions for polymorphism: `create_llm_client()` returns appropriate LLM client
- Abstract base classes define interfaces: `class LLMClient(ABC)` with `@abstractmethod` decorators

**Class Structure Pattern:**
```python
class ClassName:
    """Docstring explaining purpose."""

    def __init__(self, required_arg: Type, optional_arg: Type = default):
        """Initialize with docstring for complex inits."""
        self.attr = required_arg
        # Additional setup

    def public_method(self) -> ReturnType:
        """Docstring with purpose."""
        pass

    def _private_method(self) -> ReturnType:
        """Underscore prefix for internal methods."""
        pass
```

**Dataclasses:**
- Extensive use of `@dataclass` for configuration and data containers:
  ```python
  @dataclass
  class ScoredPrompt:
      prompt: str
      score: float
      visits: int = 0
      total_reward: float = 0.0

      def __post_init__(self):
          # Validation logic
          if not 0.0 <= self.score <= 1.0:
              raise ValueError(...)
  ```
- Post-init validation for invariants
- Optional `__repr__` override for debugging

**Barrel Files:** Not used; imports are explicit

---

*Convention analysis: 2026-01-28*
