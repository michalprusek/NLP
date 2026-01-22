# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hardware Environment

**GPUs: 2x NVIDIA L40S (48GB VRAM each, 96GB total)**

Always optimize for this hardware:
- Use DDP (DistributedDataParallel) for multi-GPU training with `torchrun --nproc_per_node=2`
- Maximize batch sizes to utilize 48GB VRAM per GPU (e.g., batch_size=1024-2048 for embeddings)
- Use `pin_memory=True` and `num_workers=8` for DataLoaders
- Enable mixed precision (fp16/bf16) when appropriate

## Overview

CRITICAL: before each new implementation proposal retrieve more information from context7 MCP and web search - be as much informed and updated as possible. Propose new features to the implementation to keep up with the latest research of 2026.

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

**IMPORTANT - HbBoPs Fidelity:**
- **NEVER use accuracies evaluated on low fidelity (< 600 examples)**
- Evaluations on 10-50 examples are statistically unreliable (high variance)
- Always filter HbBoPs results with `--min-fidelity 600` or higher
- Best accuracy at fidelity 1319: ~90.5% (vs misleading 100% at fidelity 10)

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

### FlowPO-HD - High-Dimensional Flow-based Prompt Optimization (`flowpo_hd/`)

**Direct optimization in 1024D SONAR space with FlowDiT manifold projection:**

```bash
# BetaGP + TuRBO optimization (requires space mapping first)
uv run python -m flowpo_hd.scripts.run_beta_gp_turbo \
    --mapping-results flowpo_hd/results/space_mapping_100x100.json \
    --iterations 20 --acquisition nei
```

**Key Components:**
- **FlowDiT (1024→1024)**: Flow matching model for manifold projection
  - Checkpoint: `flowpo_hd/checkpoints_mega_aux2/best.pt`
  - Architecture: 4 DiT blocks, hidden_dim=1024, no bottleneck
  - Maps noise z ~ N(0,I) → instruction manifold via ODE integration
- **BetaHeteroscedasticGP**: GP with Beta-smoothed observations
  - Beta smoothing: `(k+α)/(n+α+β)` instead of raw `k/n`
  - Heteroscedastic noise: `σ² = p(1-p)/(n+α+β+1)`
  - Hvarfner (2024) dimension-scaled lengthscale prior
- **TuRBO**: Trust Region BO for local optimization
  - Adaptive trust region: expands on success, shrinks on failure
  - Acquisitions: `ts` (Thompson Sampling), `ei`, `nei` (Noisy EI)

**CRITICAL - Manifold & Decoding:**
- **NEVER sample directly in SONAR bounds** - results in garbage text
- **ALWAYS decode latents to text** - vždy dekóduj kandidáty přes SONAR aby vznikly nové instrukce:
```python
from flowpo_hd.utils import SONARHelper
sonar = SONARHelper(device="cuda", normalize=False)
instructions = sonar.decode(candidates)  # Always decode to see actual instructions!
```
- **Perturbation-based candidates** work better than FlowDiT projection:
```python
# Generate candidate as small perturbation of known-good embedding
perturbation = torch.randn_like(parent) * parent.norm() * 0.02  # 2% of norm
candidate = parent + perturbation
instruction = sonar.decode(candidate)  # Decode to get new instruction text
```

**Data Files:**
- Space mapping results: `flowpo_hd/results/space_mapping_*.json`
- FlowDiT checkpoint: `flowpo_hd/checkpoints_mega_aux2/best.pt`

## Coding Standards

### Caching Intermediate Results
**Always implement caching for expensive operations in scripts.** When a script performs time-consuming operations (encoding, downloading, preprocessing), save intermediate results to disk and reload them on re-run. This saves hours of compute time when scripts crash or need to be re-run.

```python
# Pattern for caching expensive operations
cache_path = "data/intermediate_embeddings.pt"
if os.path.exists(cache_path):
    logger.info(f"Loading cached embeddings from {cache_path}")
    data = torch.load(cache_path)
    embeddings = data["embeddings"]
else:
    logger.info("Computing embeddings (will be cached)...")
    embeddings = expensive_encoding_operation(inputs)
    torch.save({"embeddings": embeddings, "metadata": {...}}, cache_path)
    logger.info(f"Saved cache to {cache_path}")
```

Key principles:
- Save intermediate results BEFORE expensive operations that might fail (e.g., save after encoding, before deduplication)
- Use descriptive cache filenames with timestamps or version info
- Include metadata in cache files (source, parameters, timestamp)
- Add `--no-cache` flag to force recomputation when needed

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
