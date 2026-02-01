# Coding Conventions

**Analysis Date:** 2026-01-31

## Naming Patterns

**Files:**
- Module names: lowercase with underscores (`opro.py`, `gp_surrogate.py`, `batch_selection.py`)
- Class files match class names: `decoder.py` for `SonarDecoder`, `flow_model.py` for `FlowMatchingModel`
- Run scripts: `run.py` or `train_flow.py` for executable entry points
- Test files: `test_<module_name>.py` (e.g., `test_gp_surrogate.py`)

**Classes:**
- PascalCase: `OPROOptimizer`, `ProTeGiOptimizer`, `GEPAOptimizer`, `FlowMatchingModel`, `SonarDecoder`, `RealNVP`
- Data classes: `ScoredPrompt`, `ScoredCandidate`, `ReasoningTrace`, `ProTeGiIteration`, `GEPAIteration`
- Base classes: `LLMClient`, `BaseGPSurrogate`

**Functions:**
- snake_case: `bucket_score()`, `select_batch_candidates()`, `normalize_answer()`, `extract_answer()`, `extract_ground_truth()`, `compare_answers()`
- Private methods: underscore prefix `_get_random_examples()`, `_is_budget_exhausted()`, `_can_afford_evaluation()`, `_truncate_prompt()`, `_integrate()`, `_ode_func()`
- Type hints required on all parameters and return values

**Variables:**
- snake_case: `task_llm_client`, `meta_llm_client`, `num_candidates_per_iter`, `minibatch_size`, `keep_top_k`, `total_budget`, `fixed_eval_set`
- Constants: UPPERCASE (`META_PROMPT_TEMPLATE`, `GRADIENT_PROMPT_TEMPLATE`, `EDIT_PROMPT_TEMPLATE`, `DEVICE`, `STRICT_PATTERN`, `BOXED_PATTERN`)
- Private attributes: underscore prefix `self._train_X`, `self._train_Y`, `self.model`

**Types:**
- Type hints use `from typing import List, Dict, Any, Tuple, Optional, Set`
- Modern union syntax occasionally used: `torch.device | str`
- Optional for nullable: `Optional[torch.Tensor]`, `Optional[Dict[str, Any]]`
- Complex types: `Dict[str, Any]`, `Tuple[float, Dict]`, `List[ScoredPrompt]`

## Code Style

**Formatting:**
- No explicit linter/formatter configured (no .pylintrc, .flake8, .isort config found)
- Standard Python conventions appear followed naturally
- 4-space indentation throughout
- Line length typically 80-100 characters (mostly observed)
- Imports organized: stdlib → third-party → local imports

**Linting:**
- No ESLint or Pylint configuration files present
- Code follows PEP 8 conventions informally
- Type hints used consistently across modules (PEP 484 style)
- Module and function docstrings present for public APIs

## Import Organization

**Order:**
1. Standard library (`typing`, `dataclasses`, `pathlib`, `datetime`, `random`, `json`, `os`, `logging`, `re`, `sys`, `math`, `abc`)
2. Third-party packages (`torch`, `numpy`, `tqdm`, `datasets`, `gpytorch`, `botorch`)
3. Local imports (`from shared.llm_client import`, `from shared.gsm8k_evaluator import`)

**Path Aliases:**
- Relative imports used: `from ecoflow.velocity_network import VelocityNetwork`
- Absolute imports from shared modules: `from shared.gsm8k_evaluator import extract_answer, extract_ground_truth, compare_answers`
- No explicit path aliases defined (import paths use package structure directly)

**Load patterns for prompts:**
```python
# Template files loaded at module level
PROMPTS_DIR = Path(__file__).parent / 'prompts'
META_PROMPT_TEMPLATE = (PROMPTS_DIR / 'opro_meta.txt').read_text(encoding='utf-8')
GRADIENT_PROMPT_TEMPLATE = (PROMPTS_DIR / 'protegi_gradient.txt').read_text(encoding='utf-8')
```

## Error Handling

**Patterns:**
- Validation errors with `ValueError`: `raise ValueError(f"score must be in [0, 1], got {self.score}")`
- File I/O with context-specific `FileNotFoundError` and `UnicodeDecodeError` wrapping
- Runtime errors with `RuntimeError`: `raise RuntimeError(f"Pareto front is empty")`
- Exception chaining: `raise FileNotFoundError(...) from None` or `raise ValueError(...) from e`

**Example from `gepa.py` (_load_template):**
```python
def _load_template(filename: str) -> str:
    template_path = PROMPTS_DIR / filename
    try:
        return template_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required prompt template not found: {template_path}\n"
            f"Please ensure the file exists. Expected location: {PROMPTS_DIR}"
        ) from None
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Prompt template {template_path} has invalid encoding: {e}\n"
            "Template files must be UTF-8 encoded."
        ) from e
```

**Invariant validation in dataclasses:**
```python
def __post_init__(self):
    """Validate invariants"""
    if not 0.0 <= self.score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {self.score}")
    if self.visits < 0:
        raise ValueError(f"visits must be non-negative, got {self.visits}")
```

## Logging

**Framework:** `print()` and `logging` module
- Simple optimizers (OPRO, ProTeGi, GEPA) use `print()` for user feedback
- Infrastructure modules use `logging.getLogger(__name__)` (e.g., `decoder.py`, `gp_surrogate.py`)

**Logging usage (`decoder.py`, `gp_surrogate.py`):**
```python
logger = logging.getLogger(__name__)
logger.info(f"Initializing SonarDecoder on device: {self.device}")
logger.info(f"Decoding {embeddings.shape[0]} embeddings...")
```

**Print-based output in optimizers:**
```python
print(f"Fixed evaluation set: {len(self.fixed_eval_set)} examples ({100*len(self.fixed_eval_set)/dataset_size:.1f}%)")
print(f"Score: {score:.1%} | Prompt:\n{prompt}")
```

**Critical requirement from CLAUDE.md:**
- Full prompts are NEVER truncated in logs
- Example: `print(f"Generated:\n{prompt}")` not `print(f"Generated: {prompt[:80]}...")`

## Comments

**When to Comment:**
- Docstrings for all public classes and functions (module level)
- Inline comments for non-obvious mathematical or algorithmic details
- Explaining "why" not "what" (code structure is self-documenting)

**Docstring Examples:**
```python
def normalize_answer(answer: str) -> str:
    """Normalize answer by removing commas, dollar signs, trailing periods.

    Matches lm-eval-harness regexes_to_ignore: [",", "\\$", "\\.(?!\\d)"]
    """

def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output using robust methodology.

    Priority order (matches Qwen output formats):
    1. #### (number) - standard GSM8K format
    2. \\boxed{number} - LaTeX format (common in Qwen math outputs)
    3. Last number in text - fallback for "The answer is X"
    """
```

## Function Design

**Size:** Functions typically 15-50 lines for business logic, methods up to 100+ for optimization loops

**Parameters:**
- Keyword arguments for configuration: `num_iterations: int = 200, num_candidates_per_iter: int = 8`
- Positional for primary inputs: `prompt: str`, `prompts: List[str]`
- Type hints always required: `def evaluate_prompt(self, prompt: str, ...) -> Tuple[float, Dict]:`
- Optional parameters defaulted to None: `meta_llm_client=None, save_eval_json: bool = False`

**Return Values:**
- Single return or Tuple with multiple values: `-> Tuple[float, Dict]`, `-> str`
- No implicit None returns; explicit `return` statements
- Dict returns for structured data: results contain `{'accuracy': score, 'details': [...]}`

**Example from `opro.py`:**
```python
def evaluate_prompt(
    self,
    prompt: str,
    save_eval_json: bool = False,
    eval_output_dir: str = None,
    iteration: int = None,
    candidate_idx: int = None
) -> Tuple[float, Dict]:
    """Evaluate prompt on fixed evaluation set"""
    batch = self.fixed_eval_set

    # Generate answers
    questions = [ex['question'] for ex in batch]
    formatted_prompts = [f"Question: {q}\n\n{prompt}\n\nAnswer:" for q in questions]
    outputs = self.task_llm.generate_batch(
        formatted_prompts, temperature=0.0, max_new_tokens=self.task_max_tokens
    )

    # Evaluate
    indices = [ex['idx'] for ex in batch]
    results = self.evaluator.evaluate_batch(outputs, indices)
    score = results['accuracy']

    return score, results
```

## Module Design

**Exports:**
- Main classes explicitly used: `OPROOptimizer`, `ProTeGiOptimizer`, `GEPAOptimizer`
- Factory functions: `create_llm_client()`, `create_surrogate()`
- Compatibility aliases: `OPRO = OPROOptimizer` for backwards compatibility

**Initialization patterns:**
- Classes initialized with configuration dict or explicit kwargs
- Required dependencies injected: `task_llm_client`, `evaluator`
- Optional overrides: `meta_llm_client=None` defaults to task_llm_client

**Example module exports (`shared/llm_client.py`):**
```python
# Abstract base
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: pass

# Implementations
class VLLMClient(LLMClient): ...
class OpenAIClient(LLMClient): ...

# Factory function
def create_llm_client(model_alias: str, backend: str = "vllm") -> LLMClient:
    ...
```

## Constants and Configuration

**Environment variables:**
- Loaded via `python-dotenv`: `load_dotenv()` at module level
- GPU configuration: `CUDA_VISIBLE_DEVICES`, `VLLM_USE_V1`, `PYTORCH_CUDA_ALLOC_CONF`
- API keys from `.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

**Configuration parameters:**
- Hardcoded defaults in `__init__`: `num_iterations: int = 200, minibatch_size: int = 261`
- Runtime overrideable via CLI arguments
- Magic numbers extracted to named constants: `num_buckets: int = 20`, `bucket_size = 1.0 / num_buckets`

---

*Convention analysis: 2026-01-31*
