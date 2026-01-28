# Codebase Structure

**Analysis Date:** 2026-01-28

## Directory Layout

```
/home/prusek/NLP/
├── src/                        # Core prompt optimization algorithms
│   ├── __init__.py
│   ├── llm_client.py           # LLM client abstraction (vLLM, OpenAI, DeepInfra)
│   ├── gsm8k_evaluator.py      # GSM8K dataset evaluation with answer extraction
│   ├── opro.py                 # OPRO: meta-optimizer with bucketed scoring
│   ├── protegi.py              # ProTeGi: textual gradients + beam search
│   ├── gepa.py                 # GEPA: genetic-Pareto approach with reflection
│   └── prompts/
│       └── gsm8k/              # Prompt templates for meta-optimizers
│           ├── opro_meta.txt   # OPRO meta-optimizer template
│           ├── protegi_gradient.txt
│           ├── protegi_edit.txt
│           ├── protegi_paraphrase.txt
│           ├── gepa_reflect.txt
│           └── gepa_mutate.txt
├── ecoflow_bo/                 # Embedding-space Bayesian optimization
│   ├── __init__.py             # Public API exports
│   ├── config.py               # Configuration dataclasses
│   ├── encoder.py              # MatryoshkaEncoder: 768D → 48D latent
│   ├── velocity_network.py     # DiT-based velocity network for flow matching
│   ├── cfm_decoder.py          # RectifiedFlow decoder: latent → embeddings
│   ├── perceiver_decoder.py    # Perceiver-based alternative decoder
│   ├── losses.py               # KL, InfoNCE, MatryoshkaCFM losses
│   ├── latent_gp.py            # Gaussian Process + CoarseToFineGP
│   ├── density_acquisition.py  # Density-aware acquisition for exploration
│   ├── cycle_consistency.py    # Hallucination detection via encode-decode
│   ├── detail_retriever.py     # Nearest-neighbor z_detail lookup
│   ├── optimizer.py            # Main EcoFlowBO orchestrator
│   ├── train_manifold.py       # Training for manifold learning
│   ├── train_perceiver.py      # Training for Perceiver decoder
│   ├── pipeline.md             # Comprehensive documentation
│   └── __pycache__/
├── tests/
│   ├── test_ecoflow.py         # Unit tests for EcoFlow-BO components
│   └── __pycache__/
├── datasets/
│   ├── gsm8k/
│   │   ├── train/              # GSM8K training set (Arrow format)
│   │   │   ├── data-00000-of-00001.arrow
│   │   │   └── state.json
│   │   ├── test/               # GSM8K test set (Arrow format)
│   │   │   ├── data-00000-of-00001.arrow
│   │   │   └── state.json
│   │   └── dataset_dict.json
│   ├── hbbops/                 # HuggingFace Baseline of Prompts dataset
│   │   ├── ape_instructions_1000.json
│   │   ├── instructions_*.txt
│   │   ├── examples_*.txt
│   │   └── full_grid_combined.jsonl
│   └── tos_local/              # Additional dataset variants
├── results/                    # Experiment outputs (gitignored)
│   ├── baseline_results/       # Reference experiment results
│   ├── ecoflow_checkpoints/    # Trained model checkpoints
│   └── [experiments_*.json]    # Timestamped optimization results
├── papers/                     # Reference papers (PDFs)
├── run_opro.py                 # Entry point: OPRO optimization
├── run_protegi.py              # Entry point: ProTeGi optimization
├── run_gepa.py                 # Entry point: GEPA optimization
├── pyproject.toml              # Project metadata, dependencies
├── uv.lock                     # Lock file for deterministic installs
├── .env                        # API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc)
├── .env.example                # Example .env template
├── .gitignore                  # Excludes results/, .venv/, __pycache__
├── CLAUDE.md                   # Instructions for Claude Code
├── README.md                   # Project overview
├── TODO.md                     # Issue tracking and roadmap
├── .planning/
│   └── codebase/               # GSD codebase analysis documents
└── .venv/                      # Python virtual environment
```

## Directory Purposes

**src/:**
- Purpose: Core optimization algorithms and utilities
- Contains: LLM client abstractions, evaluators, algorithm implementations
- Key files: `llm_client.py` (all backends), `gsm8k_evaluator.py` (scoring)
- Each algorithm has dedicated file (OPRO, ProTeGi, GEPA)

**ecoflow_bo/:**
- Purpose: Embedding-space Bayesian optimization system
- Contains: Neural networks (encoder, decoder, velocity net), GP, acquisition, utilities
- Modular design: Each component in separate file (config, encoder, decoder, etc.)
- Self-contained with comprehensive documentation in `pipeline.md`

**tests/:**
- Purpose: Unit and integration tests
- Contains: Pytest test file (`test_ecoflow.py`) with fixtures for all EcoFlow components
- Coverage: Encoder, decoder, velocity network, losses, GP, acquisition, cycle consistency, Perceiver

**datasets/:**
- Purpose: Training and evaluation data (read-only, version controlled)
- Contents: GSM8K (Arrow format), HbBoPs instructions, local variants
- Format: Hugging Face datasets with train/test splits
- NOT gitignored: Static reference data

**results/:**
- Purpose: Experiment outputs and artifacts (gitignored)
- Contents: Optimization history, best prompts, model checkpoints, logs
- Naming: Timestamped JSON files (e.g., `opro_20260128_205902.json`)
- Subdirs: `baseline_results/`, `ecoflow_checkpoints/`

**src/prompts/gsm8k/:**
- Purpose: Prompt templates for meta-optimizers
- Format: Plain text templates with placeholders for examples/scores
- Usage: Loaded at module init in optimizer files
- Files correspond to algorithms: `opro_meta.txt`, `protegi_*.txt`, `gepa_*.txt`

## Key File Locations

**Entry Points:**
- `run_opro.py`: OPRO command-line interface (model aliasing, training, test evaluation)
- `run_protegi.py`: ProTeGi command-line interface
- `run_gepa.py`: GEPA command-line interface
- All use argparse with sensible defaults; results saved to `results/` with timestamps

**Configuration:**
- `.env`: API keys (required: ANTHROPIC_API_KEY or OPENAI_API_KEY or DEEPINFRA_API_KEY)
- `pyproject.toml`: Dependencies via uv (all Python packages)
- `ecoflow_bo/config.py`: Dataclass configurations for all EcoFlow-BO hyperparameters

**Core Logic:**
- `src/llm_client.py`: Backend abstraction, model loading, generation
- `src/gsm8k_evaluator.py`: Answer extraction, ground truth comparison, batch evaluation
- `src/opro.py`: OPRO optimizer with bucketed scoring
- `src/protegi.py`: ProTeGi optimizer with UCB beam search
- `src/gepa.py`: GEPA optimizer with Pareto frontier
- `ecoflow_bo/optimizer.py`: EcoFlowBO main class

**Testing:**
- `tests/test_ecoflow.py`: All EcoFlow component tests with fixtures
- Commands: `pytest tests/test_ecoflow.py -v` for full suite

**Documentation:**
- `CLAUDE.md`: Claude Code instructions (hardware, workflows, constraints)
- `README.md`: Project overview and setup
- `TODO.md`: Known issues and roadmap
- `ecoflow_bo/pipeline.md`: Comprehensive EcoFlow-BO documentation

## Naming Conventions

**Files:**
- Algorithm implementations: lowercase with underscores (`opro.py`, `protegi.py`, `gepa.py`)
- Entry points: `run_*.py` for command-line scripts
- Tests: `test_*.py` for pytest discovery
- Configs: `config.py` for dataclass definitions
- Prompts: `{algorithm}_{component}.txt` (e.g., `opro_meta.txt`, `protegi_gradient.txt`)

**Directories:**
- Core algorithms: lowercase plural or singular based on scope (`src/`, `ecoflow_bo/`)
- Data: lowercase descriptive (`datasets/`, `results/`, `papers/`)
- Splits: `train/`, `test/` for dataset organization
- Checkpoints: `ecoflow_checkpoints/` (descriptive)

**Classes:**
- Pascal case: `OPRO`, `ProTeGi`, `GEPA`, `EcoFlowBO`, `MatryoshkaEncoder`, `VelocityNetwork`
- Interfaces/ABCs: `LLMClient`, `GSM8KEvaluator`
- Config dataclasses: `EcoFlowConfig`, `EncoderConfig`, `DecoderConfig`

**Functions:**
- Private helpers: `_format_prompt()`, `_patch_vllm_platform()`, `_load_template()`
- Public API: `create_llm_client()`, `optimize()`, `generate()`, `evaluate_batch()`

**Variables:**
- Lowercase with underscores: `max_tokens`, `minibatch_size`, `tensor_parallel_size`
- Single letters for tensors: `z` (latent), `x` (data), `t` (time), `v` (velocity)
- Scores: `accuracy`, `best_score`, `ucb_score`

## Where to Add New Code

**New Optimization Algorithm:**
- Implementation: `src/{algorithm_name}.py` (create ABC-inheriting class for consistency)
- Entry point: `run_{algorithm_name}.py` in root (copy structure from `run_opro.py`)
- Prompts: Create subdirectory `src/prompts/gsm8k/{algorithm_name}_*.txt` as needed
- Tests: Add test class to `tests/test_ecoflow.py` or create `tests/test_{algorithm_name}.py`

**New LLM Backend:**
- Implementation: Add new class inheriting `LLMClient` in `src/llm_client.py`
- Factory: Update `create_llm_client()` factory to recognize backend type
- Env vars: Document required env var in CLAUDE.md
- Models: Add aliases to `MODEL_ALIASES` in respective `run_*.py` files

**New EcoFlow-BO Component:**
- Module file: Create `ecoflow_bo/{component}.py` (e.g., `custom_decoder.py`)
- Config: Add new `@dataclass` to `ecoflow_bo/config.py`
- Export: Add to `ecoflow_bo/__init__.py` `__all__` list
- Tests: Add test class to `tests/test_ecoflow.py`

**New Evaluation Dataset:**
- Location: Create `datasets/{dataset_name}/train/` and `datasets/{dataset_name}/test/` directories
- Format: Use Hugging Face Arrow format (load_from_disk compatible)
- Evaluator: Create `src/{dataset_name}_evaluator.py` inheriting standard interface
- Entry points: Update relevant `run_*.py` to support `--dataset` parameter

**Utilities/Helpers:**
- Shared functions: `src/utils.py` (if > 1 algorithm needs it)
- EcoFlow utilities: Keep in relevant module file (no separate utils)
- Avoid circular imports by careful placement in layer hierarchy

## Special Directories

**datasets/gsm8k/:**
- Purpose: GSM8K (Grade School Math 8K) dataset
- Generated: No (pre-downloaded and committed)
- Committed: Yes (static reference data)
- Format: Hugging Face datasets Arrow format (binary-safe, efficient)
- Structure: `train/` and `test/` splits with state.json and data files

**results/:**
- Purpose: All experiment outputs (optimization history, checkpoints, logs)
- Generated: Yes (created by run scripts)
- Committed: No (gitignored via `.gitignore`)
- Naming: Timestamped per CLAUDE.md (e.g., `opro_20260128_205902.json`)
- Subdirectories: `baseline_results/` (reference), `ecoflow_checkpoints/` (trained models)

**.planning/codebase/:**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by GSD mapping commands)
- Committed: Yes (documentation, not code)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md, STACK.md, INTEGRATIONS.md

**.venv/:**
- Purpose: Python virtual environment (uv managed)
- Generated: Yes (by `uv sync`)
- Committed: No (gitignored)
- Management: Use `uv sync` to install/update, never manually

## Import Patterns

**Algorithm files import hierarchy:**
```
src/{algorithm}.py
├── imports from llm_client.py (LLMClient)
├── imports from gsm8k_evaluator.py (GSM8KEvaluator)
├── imports from prompts/gsm8k/ (template files)
└── root run_{algorithm}.py imports from src/
```

**EcoFlow-BO import hierarchy:**
```
ecoflow_bo/optimizer.py (main orchestrator)
├── from .config import EcoFlowConfig
├── from .encoder import MatryoshkaEncoder
├── from .decoder import RectifiedFlowDecoder
├── from .latent_gp import CoarseToFineGP
├── from .density_acquisition import DensityAwareAcquisition
├── from .cycle_consistency import CycleConsistencyChecker
└── from .detail_retriever import DetailRetriever

ecoflow_bo/__init__.py (public API)
└── imports all major classes for external access
```

**Backend detection:**
```python
# In run_opro.py and src/llm_client.py
create_llm_client(model_name, backend="auto")
# Auto-detects: "gpt" → OpenAI, "google/" → DeepInfra, else vLLM
```

---

*Structure analysis: 2026-01-28*
