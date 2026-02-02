# Codebase Structure

**Analysis Date:** 2026-01-31

## Directory Layout

```
NLP/
├── opro/                        # OPRO: Meta-optimization via LLM feedback
│   ├── __init__.py
│   ├── opro.py                  # OPROOptimizer class (meta-prompt looping)
│   ├── run.py                   # CLI entry point with argparse
│   ├── prompts/
│   │   └── opro_meta.txt        # Meta-prompt template for candidate generation
│   └── results/                 # Output directory for results
│
├── protegi/                     # ProTeGi: Textual gradients + beam search
│   ├── __init__.py
│   ├── protegi.py               # ProTeGiOptimizer class (beam + UCB selection)
│   ├── run.py                   # CLI entry point
│   ├── prompts/
│   │   ├── protegi_gradient.txt # Generate textual gradients from errors
│   │   ├── protegi_edit.txt     # Apply edits to prompt
│   │   └── protegi_paraphrase.txt
│   └── results/
│
├── gepa/                        # GEPA: Genetic-Pareto + LLM reflection
│   ├── __init__.py
│   ├── gepa.py                  # GEPAOptimizer class (Pareto selection + reflection)
│   ├── run.py                   # CLI entry point
│   ├── prompts/
│   │   ├── gepa_reflect.txt     # Analyze failures + propose fixes
│   │   └── gepa_mutate.txt      # Generate mutations from reflection
│   └── results/
│
├── rielbo/                     # RieLBO: Flow matching + Bayesian optimization
│   ├── __init__.py
│   ├── velocity_network.py      # VelocityNetwork: DiT-style transformer (velocity field)
│   ├── flow_model.py            # FlowMatchingModel: Conditional flow matching wrapper
│   ├── train_flow.py            # Training loop with EMA, checkpoint management
│   ├── guided_flow.py           # GuidedFlowSampler: GP-UCB guided ODE sampling
│   ├── gp_surrogate.py          # SonarGPSurrogate, BAxUSGPSurrogate, Heteroscedastic GP
│   ├── decoder.py               # SonarDecoder: Embedding → text conversion
│   ├── optimization_loop.py     # BOOptimizationLoop: Main orchestrator
│   ├── validate.py              # Checkpoint loading utilities
│   ├── batch_selection.py       # Batch selection strategies for BO
│   ├── data.py                  # SonarEmbeddingDataset: Load SONAR embeddings
│   ├── run.py                   # CLI entry point with logging setup
│   └── results/                 # Output: checkpoints, metrics
│
├── nfbo/                        # NFBO: Normalizing Flow Bayesian Optimization
│   ├── __init__.py
│   ├── model.py                 # RealNVP: Invertible normalizing flow
│   ├── sampler.py               # NFBoSampler: Latent BO in Z-space
│   ├── loop.py                  # NFBoLoop: Extends BOOptimizationLoop
│   ├── run.py                   # CLI entry point
│   └── results/
│
├── shared/                      # Shared infrastructure (all methods depend on this)
│   ├── __init__.py
│   ├── llm_client.py            # LLMClient ABC + implementations (vLLM, OpenAI, DeepInfra, Transformers)
│   └── gsm8k_evaluator.py       # GSM8KEvaluator: Load dataset, score prompts
│
├── datasets/                    # Data files (read-only)
│   ├── gsm8k/
│   │   ├── train/               # GSM8K training examples
│   │   └── test/                # GSM8K test examples
│   ├── sonar_embeddings.pt      # Pretrained SONAR embeddings (1.5M × 1024D)
│   ├── evaluated_instructions/  # Previously evaluated prompts
│   ├── hbbops/                  # Hyperband baseline prompts
│   └── tos_local/               # ToS/local evaluation data
│
├── tests/                       # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_batch_selection.py  # Batch selection strategies
│   ├── test_gp_surrogate.py     # GP surrogate models
│   ├── test_nfbo_model.py       # NF-BO flow model
│   ├── test_nfbo_sampler.py     # NF-BO sampler
│   └── test_nfbo_sampler_refined.py
│
├── papers/                      # Paper PDFs and references
├── pyproject.toml               # Project metadata, dependencies
├── CLAUDE.md                    # Development guidelines for Claude
└── .planning/codebase/          # This directory: generated codebase docs
    ├── ARCHITECTURE.md
    ├── STRUCTURE.md
    ├── CONVENTIONS.md
    ├── TESTING.md
    ├── STACK.md
    ├── INTEGRATIONS.md
    └── CONCERNS.md
```

## Directory Purposes

**opro/:**
- Purpose: OPRO optimizer implementation
- Contains: Optimizer class, prompt templates, CLI runner
- Key files: `opro/opro.py` (OPROOptimizer), `opro/run.py` (entry point)

**protegi/:**
- Purpose: ProTeGi optimizer implementation
- Contains: Optimizer with beam search, textual gradients, paraphrasing
- Key files: `protegi/protegi.py` (ProTeGiOptimizer with ScoredPrompt + UCB), `protegi/run.py`

**gepa/:**
- Purpose: GEPA optimizer implementation
- Contains: Optimizer with Pareto selection, reflection-based mutations
- Key files: `gepa/gepa.py` (GEPAOptimizer, ScoredCandidate, ReasoningTrace), `gepa/run.py`

**rielbo/:**
- Purpose: Flow matching + Bayesian optimization for prompt discovery
- Contains: Velocity network, flow model, GP surrogate, decoder, optimization loop, training script
- Key files:
  - Core: `rielbo/optimization_loop.py` (BOOptimizationLoop), `rielbo/flow_model.py`
  - Models: `rielbo/velocity_network.py` (VelocityNetwork), `rielbo/gp_surrogate.py` (SonarGPSurrogate)
  - Utilities: `rielbo/decoder.py` (SonarDecoder), `rielbo/guided_flow.py` (GuidedFlowSampler)

**nfbo/:**
- Purpose: Normalizing Flow Bayesian Optimization
- Contains: RealNVP flow model, latent-space BO sampler, loop extending RieLBO
- Key files: `nfbo/model.py` (RealNVP), `nfbo/sampler.py` (NFBoSampler), `nfbo/loop.py` (NFBoLoop)

**shared/:**
- Purpose: Shared infrastructure used by all optimizer methods
- Contains: LLM client abstraction, GSM8K evaluation harness
- Key files: `shared/llm_client.py` (create_llm_client factory), `shared/gsm8k_evaluator.py`

**datasets/:**
- Purpose: Data files for training and evaluation
- Contains: GSM8K splits, SONAR embeddings, baseline prompts
- Generated: No (static, committed)

**tests/:**
- Purpose: Unit and integration tests
- Contains: Tests for RieLBO components, NFBO components
- Key patterns: Pytest fixtures in `conftest.py`, individual test modules

## Key File Locations

**Entry Points:**
- `opro/run.py`: OPRO CLI, resolves model aliases, initializes LLMClient and GSM8KEvaluator, runs optimization
- `protegi/run.py`: ProTeGi CLI, similar pattern
- `gepa/run.py`: GEPA CLI
- `rielbo/run.py`: RieLBO CLI with logging setup and checkpoint save/resume
- `nfbo/run.py`: NFBO CLI
- `rielbo/train_flow.py`: Flow model training (not a `-m` module, run via `python -m rielbo.train_flow`)

**Configuration:**
- `pyproject.toml`: Project metadata, all dependencies
- `.env` (optional): API keys loaded via `dotenv.load_dotenv()` in `shared/llm_client.py`
- CLAUDE.md (project instructions): Hardware targets, model aliases, running guidelines

**Core Logic:**
- `opro/opro.py`: OPROOptimizer class, meta-prompt looping, scoring
- `protegi/protegi.py`: ProTeGiOptimizer, textual gradients, beam search with UCB
- `gepa/gepa.py`: GEPAOptimizer, Pareto selection, reflection
- `rielbo/optimization_loop.py`: BOOptimizationLoop orchestrator
- `rielbo/flow_model.py`: FlowMatchingModel training and sampling
- `rielbo/gp_surrogate.py`: GP surrogates (SonarGPSurrogate, BAxUSGPSurrogate, Heteroscedastic)
- `nfbo/loop.py`: NFBoLoop with flow fitting
- `shared/llm_client.py`: LLMClient ABC and implementations
- `shared/gsm8k_evaluator.py`: GSM8KEvaluator with answer extraction

**Testing:**
- `tests/conftest.py`: Pytest fixtures
- `tests/test_*.py`: Individual test modules for RieLBO and NFBO

## Naming Conventions

**Files:**
- Optimizer classes: `{method}.py` in method directory (e.g., `opro/opro.py`, `protegi/protegi.py`)
- Entry points: `run.py` in method directory
- Prompt templates: `{method}_{template_type}.txt` in `prompts/` subdirectory
- CLI scripts: `python -m {package}.run` (e.g., `python -m opro.run`)

**Directories:**
- Method directories: lowercase (opro, protegi, gepa, rielbo, nfbo)
- Subdirectories: `prompts/` for templates, `results/` for output
- Shared: `shared/` for cross-cutting code

**Classes:**
- Optimizers: `{METHOD}Optimizer` (OPROOptimizer, ProTeGiOptimizer, GEPAOptimizer, NFBoLoop)
- Scoring: `ScoredPrompt` or `ScoredCandidate` (dataclass with score + metadata)
- Models: `{Model}Network` or `{Model}Model` (VelocityNetwork, FlowMatchingModel, RealNVP)
- Utilities: `{Name}` (SonarDecoder, GSM8KEvaluator, GuidedFlowSampler)

**Functions:**
- Entry: `main()` in `run.py` files
- Factories: `create_{type}()` (create_llm_client, create_surrogate)
- Utilities: `{verb}_{noun}()` (extract_answer, normalize_answer)

## Where to Add New Code

**New Optimization Method:**
- Create directory: `/home/prusek/NLP/{method_name}/`
- Create files:
  - `{method_name}.py`: Core optimizer class
  - `run.py`: CLI entry point
  - `prompts/`: Directory for template files
  - `__init__.py`: Package exports
- Depend on: `shared.llm_client`, `shared.gsm8k_evaluator`
- Pattern: Follow existing methods (OPRO/ProTeGi/GEPA) for initialization and main loop

**New Feature for Existing Method:**
- Implementation: In method's main module (`opro/opro.py`, `rielbo/optimization_loop.py`, etc.)
- Tests: Add to `tests/test_{method_name}.py` or create new test file
- Templates: Add `.txt` file to `{method_name}/prompts/`

**Shared Utility:**
- Location: `shared/`
- Pattern: Follow LLMClient and GSM8KEvaluator; ABC with factory function
- Usage: Import via `from shared.{module} import {class}`

**New Dataset or Benchmark:**
- Location: `datasets/{name}/`
- Evaluator: Add to `shared/gsm8k_evaluator.py` if following GSM8K pattern, else create `shared/{name}_evaluator.py`

**Tests:**
- Location: `tests/test_{module_name}.py`
- Pattern: Use `conftest.py` fixtures, pytest assertions
- Coverage: Unit tests for core classes, integration tests for optimization loops

## Special Directories

**results/ (per method):**
- Purpose: Store optimization outputs (best prompts, scores, metrics)
- Generated: Yes (created by run.py)
- Committed: No (gitignored)
- Contents: JSON results, TXT prompts, checkpoint files

**datasets/:**
- Purpose: Static data (GSM8K splits, SONAR embeddings, baselines)
- Generated: No (pre-computed, committed)
- Committed: Yes (necessary for reproducibility)

**prompts/ (per method):**
- Purpose: Prompt templates for meta-optimization
- Generated: No (hand-written)
- Committed: Yes
- Pattern: Jinja-like templates with `{placeholder}` or simple string formatting

**.planning/codebase/:**
- Purpose: Generated architecture documentation (this directory)
- Generated: Yes (by /gsd:map-codebase)
- Committed: Yes (for reference)

---

*Structure analysis: 2026-01-31*
