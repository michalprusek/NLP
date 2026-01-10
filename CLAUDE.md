# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Prompt optimization framework implementing **ProTeGi**, **OPRO**, and **HbBoPs** algorithms for automatic prompt engineering on GSM8K (math reasoning) and Claudette (ToS classification) datasets.

## Commands

```bash
# Setup
uv sync                          # Install dependencies
cp .env.example .env             # Configure API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)

# OPRO optimization (main method)
uv run python run_opro.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# HbBoPs multi-fidelity optimization
uv run python hbbops/run_hbbops.py --instructions datasets/hbbops/instructions_25.txt --exemplars datasets/hbbops/examples_25.txt

# Dual GPU with tensor parallelism (for 2x NVIDIA GPUs)
./run_dual_gpu.sh [method] [iterations] [minibatch_size]

# Standalone evaluation
uv run python evaluate_gsm8k.py --prompt "Your prompt" --num-samples 5
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

### HbBoPs (`hbbops/`)

**Core Algorithm** (`hbbops.py`):
- `PromptEncoder`: BERT [CLS] embeddings (768-dim) for prompt representation
- `FeatureExtractor`: Structural-aware MLP (768 → 64 → 32 for each of instruction/exemplar, then joint → 10-dim latent)
- `DeepKernelGP`: GPyTorch GP with Matérn 5/2 kernel on learned features
- Hyperband scheduler with BO proposals using Expected Improvement acquisition

**Key Design Decisions**:
- Validation data shuffled once at init for unbiased fidelity subsets
- Evaluation caching with fidelity extension (reuses lower-fidelity results)
- Unit cube normalization for GP inputs, z-score standardization for outputs

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
- `results/`, `hbbops/results/`: Experiment outputs (gitignored)
- `visualize/`: Analysis and comparison scripts

### Vec2Text-HbBoPs (`vec2text_hbbops/`)

**Latent space optimization with Vec2Text inversion** for generating novel prompts:

- `GTRPromptEncoder`: SentenceTransformer GTR-T5-Base (768D, Vec2Text compatible)
- `PromptAutoencoder`: Regularized AE (1536D → 10D → 1536D) with denoising, dropout, cosine loss
- `HbBoPsVec2Text`: HbBoPs with GTR encoder and `load_from_grid()` support
- `Vec2TextHbBoPsInference`: Complete pipeline with EI optimization and text inversion

**Architecture:**
```
Prompt Text → GTR (768D×2) → AE Encoder → 10D → GP → EI optimization
                                           ↓
10D optimum → AE Decoder → 768D×2 → Vec2Text → Novel Prompt Text
```

## Best Practices

### Default: Use Pre-evaluated Grid (No LLM Required)
**Always load from grid unless explicitly running HbBoPs.** This is the default behavior:

```bash
# Default mode: Load top-25 from grid, train AE+GP, optimize latent, Vec2Text invert
uv run python -m vec2text_hbbops.run_vec2text_hbbops

# Custom top-k
uv run python -m vec2text_hbbops.run_vec2text_hbbops --top-k 50

# Only if you need full HbBoPs with LLM evaluation:
uv run python -m vec2text_hbbops.run_vec2text_hbbops --run-hyperband --model qwen
```

The pre-evaluated grid (`datasets/hbbops/full_grid_combined.jsonl`) contains 625 fully evaluated prompts.
Loading top-25 and training GP takes seconds vs hours for full HbBoPs.

### AE-Only Mode (No Vec2Text)
```bash
# Just train and evaluate autoencoder reconstruction quality
uv run python -m vec2text_hbbops.run_vec2text_hbbops --ae-only
```

### Robust Vec2Text (`robust_vec2text/`)

**VAE-based instruction optimization with GP-guided exemplar selection:**

```bash
# Full pipeline with exemplar selection
uv run python -m robust_vec2text.run --select-exemplar --top-k 25

# Skip APE generation (faster, uses only original 25 instructions)
uv run python -m robust_vec2text.run --skip-ape --select-exemplar
```

**Architecture:**
- `InstructionVAE`: 768D GTR → 32D latent → 768D (cosine-focused loss)
- `LatentGP`: GP on 32D VAE latent for instruction optimization
- `ExemplarSelector`: HbBoPs-style GP for exemplar selection (trains on top-k from grid)

**Key Design Decisions:**
- Uses GTR encoder (not BERT) for consistency with Vec2Text
- **Single GP training**: ExemplarSelector GP trains once in `load_grid()` on top-k prompts, then reused for selection
- APE data augmentation generates 1000 instructions for better VAE training
- Vec2Text max_length increased to 128 tokens for complete instructions

### InvBO Decoder (`generation/invbo_decoder/`)

**LogEI-based Bayesian optimization with VAE latent space and gradient-based acquisition:**

```bash
# Standard run (VAE enabled by default, gradient optimization enabled by default)
uv run python -m generation.invbo_decoder.run --iterations 10

# With custom hyperparameters
uv run python -m generation.invbo_decoder.run \
    --iterations 50 --vae-beta 0.05 --vae-annealing 300
```

**IMPORTANT:**
- **Always evaluate on full validation set (1319 samples)** - this is the default. Never reduce `--eval-samples` for final results.

**Architecture:**
- `InstructionVAE`: 768D GTR → 64D latent → 768D with KL annealing + contrastive loss
- `VAEWithAdapter`: Frozen VAE encoder (64D) with trainable adapter MLP (64→32→10)
- `GPWithEI`: Deep kernel GP with LogEI acquisition on 10D adapter output
- `UncertaintyContinuousContrastiveLoss`: Structures latent space so distances reflect accuracy differences
- Vec2Text inversion for text generation from embeddings

**Key Design Decisions:**
- LogEI instead of EI for numerical stability (avoids underflow with tiny improvement values)
- Noise constraint `Interval(0.001, 0.1)` on GP to balance between over-confidence (too low) and underfitting (too high)
- VAE early stopping tracks reconstruction loss (not total loss) to avoid premature stop during KL annealing
- **Contrastive loss** weighted by evaluation uncertainty (binomial variance from fidelity)

**Training Data Format:**
The file `datasets/inversion/diverse_instructions_1000.json` contains:
```python
{
    "instructions": [...],  # 1053 instruction strings
    "hyperband_evaluations": {
        "source": "path/to/log",
        "num_evaluated": 225,  # Number of evaluated instructions
        "max_fidelity": 1319,  # Maximum possible fidelity (full validation set)
        "results": {
            # Results are indexed by instruction index (string keys)
            # Fidelity varies due to Hyperband's multi-fidelity approach
            "0": {"error_rate": 0.19, "accuracy": 0.81, "fidelity": 1319},  # Full fidelity
            "42": {"error_rate": 0.25, "accuracy": 0.75, "fidelity": 160},  # Partial fidelity
            ...
        }
    }
}
```
VAE trains only on instructions with evaluations (accuracy + fidelity for contrastive loss).
Note: Fidelity indicates how many validation samples were used. Lower fidelity = higher uncertainty.

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
