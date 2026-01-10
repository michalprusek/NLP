# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Prompt optimization framework implementing **ProTeGi**, **OPRO**, **LIPO**, and **BOLT** algorithms for automatic prompt engineering on GSM8K (math reasoning) dataset.

## Commands

```bash
# Setup
uv sync                          # Install dependencies
cp .env.example .env             # Configure API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)

# OPRO optimization
uv run python run_opro.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# ProTeGi optimization
uv run python run_protegi.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# LIPO (Latent Instruction Prompt Optimization)
uv run python -m lipo.run --skip-hbbops --iterations 50

# BOLT (Bayesian Optimization over Latent Templates)
uv run python -m bolt.run --load-hyperband bolt/data/hyperband_results.json --iterations 30
```

## Architecture

### Core Components (`src/`)

**LLM Client Abstraction** (`llm_client.py`):
- `LLMClient` ABC with `VLLMClient`, `OpenAIClient`, `DeepInfraClient` implementations
- Factory: `create_llm_client(model_name, backend)` - auto-detects backend from model name
- vLLM handles chat templates automatically for `*-Instruct` models

**OPRO Optimizer** (`opro.py`):
- Uses fixed evaluation set (same minibatch for all prompts for fair comparison)
- Bucketed scoring (20 buckets) shows diverse examples to meta-optimizer
- Top-K memory keeps best prompts; meta-LLM generates new candidates at temperature=1.0
- Supports separate task and meta-optimizer models (`--meta-model`)

**GSM8K Evaluator** (`gsm8k_evaluator.py`):
- Answer extraction: **Always extract the last number only** - no format-specific patterns like `####` or `\boxed{}`
- Numerical comparison with 1e-6 tolerance for floats
- **Prompt format: Q_end style** (from OPRO paper) - instruction comes AFTER the question:
  ```
  Q: {question}
  {instruction}
  A:
  ```

## Model Aliases

```python
"haiku" → claude-haiku-4-5-20251001
"sonnet" → claude-sonnet-4-5-20251022
"qwen" → Qwen/Qwen2.5-7B-Instruct
```

## Key Parameters

- `--model` / `--meta-model`: Separate task evaluation and meta-optimization models
- `--backend`: `vllm` (fast, requires CUDA), `openai`, `deepinfra`, `auto`
- `--minibatch-size`: Examples per evaluation (trade-off: stability vs speed)
- `--tensor-parallel-size`: GPU count for vLLM tensor parallelism

## File Conventions

- `datasets/`: Static input data (read-only, version controlled)
- `results/`, `bolt/results/`, `lipo/results/`: Experiment outputs (gitignored)

### LIPO - Latent Instruction Prompt Optimization (`lipo/`)

**Instruction-only optimization with VAE latent space and direct GP on 32D latents:**

```bash
# Skip HbBoPs: Load pre-evaluated results and train GP only (DEFAULT - no LLM needed)
uv run python -m lipo.run --skip-hbbops

# Skip HbBoPs with custom evaluations file
uv run python -m lipo.run --skip-hbbops --hyperband-evals-path path/to/evals.json

# Full run with HbBoPs evaluation (requires LLM)
uv run python -m lipo.run --iterations 10
```

**Datasets:**
- APE instructions: `lipo/data/ape_instructions.json` (2000 instructions for VAE training)
- HbBoPs results: `lipo/data/hbbops_results_{timestamp}.json` (evaluated prompts for GP training only)

**Data separation:**
- **VAE training**: Always uses APE instructions (2000) for diverse latent space coverage
- **GP training**: Uses only HbBoPs-evaluated prompts with accuracy labels

**Architecture:**
- VAE: 768D GTR → 32D latent (frozen during GP training)
- GP: Matern 5/2 kernel with ARD directly on 32D VAE latent (no adapter)

**Optimization flow:**
```
z (32D VAE latent) → GP (ARD kernel) → qLogEI
```

**Skip HbBoPs Mode:**
When `--skip-hbbops` is enabled:
1. Loads HbBoPs evaluation results from `--hyperband-evals-path`
2. Loads APE instructions from `lipo/data/ape_instructions.json` for VAE training
3. Trains GP on evaluated instructions only (with accuracy labels)
4. Runs InvBO inference without additional LLM calls

**IMPORTANT - Documentation:**
- **Always update `lipo/PIPELINE.md`** when making changes to LIPO code (dimensions, parameters, architecture)
- PIPELINE.md is the single source of truth for LIPO architecture and parameters
- Keep dimensions, loss functions, and training parameters in sync with `config.py`
- **Always update `lipo/run.py` CLI argument defaults** when changing `config.py` defaults - CLI overrides config!

### BOLT - Bayesian Optimization over Latent Templates (`bolt/`)

**Joint instruction + exemplar optimization using VAE latent space and GP:**

```bash
# Run BOLT with pre-evaluated Hyperband results (DEFAULT - no LLM needed for initial training)
uv run python -m bolt.run --load-hyperband bolt/data/hyperband_results.json --iterations 30

# Full run with Hyperband evaluation (requires LLM)
uv run python -m bolt.run --iterations 30
```

**Architecture:**
- `StructureAwareVAE`: Joint instruction (16D) + exemplar (8D) VAE = 24D latent
- `InstructionEncoder`: 768D GTR → 256 → 128 → 16D
- `ExemplarSetEncoder`: Set Transformer with ISAB/PMA for permutation-invariant encoding
- `CrossAttentionScorer`: Instruction↔exemplar matching with ListMLE ranking loss
- Deep Kernel Learning GP on 24D joint latent space

**IMPORTANT - Documentation & Code Quality:**
- **Always update `bolt/PIPELINE.md`** when making ANY changes to BOLT code (architecture, parameters, losses, dimensions)
- PIPELINE.md is the single source of truth for BOLT architecture and must stay in sync with code
- **Always run code-simplifier agent** after implementing changes or modifications to BOLT code
- Keep dimensions, loss functions, and training parameters in sync with `bolt/config.py`

## Coding Standards

### Logging Generated Prompts
**NEVER truncate prompts in log output.** Always log the full prompt text, not truncated versions like `"prompt text..."`. This is critical for debugging and reproducibility.

```python
# BAD - truncated prompt is useless for debugging
log(f"Generated: {prompt[:80]}...")

# GOOD - always log full prompt
log(f"Generated:\n{prompt}")
```

### Long-Running Processes
**Always run processes longer than ~30 seconds in tmux** with logging to results directory:

```bash
# Pattern for LIPO runs
tmux new-session -d -s <session_name> "CUDA_VISIBLE_DEVICES=<gpu> uv run python -m lipo.run <args> 2>&1 | tee lipo/results/<descriptive_name>_$(date +%Y%m%d_%H%M%S).log; exec bash"

# Example
tmux new-session -d -s lipo_12d_beta0005 "CUDA_VISIBLE_DEVICES=0 uv run python -m lipo.run --skip-hbbops --latent-dim 12 --vae-beta 0.005 --iterations 50 2>&1 | tee lipo/results/12d_beta0005_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

This ensures:
- Process survives connection drops
- Output is logged for later analysis
- Multiple experiments can run in parallel on different GPUs

## Constraints

- vLLM requires CUDA GPU; Claude API requires `ANTHROPIC_API_KEY`
- Models <3B struggle with meta-optimization tasks
- For 16GB RAM systems, use `--model Qwen/Qwen2.5-3B-Instruct`
- **Always use `--backend vllm`** unless explicitly told otherwise
- **NEVER stop/kill running processes** (tmux sessions, background jobs) unless user explicitly asks to stop them
