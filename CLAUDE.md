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
- Answer extraction: `final_answer: NUMBER`, `#### NUMBER`, `\boxed{NUMBER}`, or last number
- Numerical comparison with 1e-6 tolerance for floats

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

## Constraints

- vLLM requires CUDA GPU; Claude API requires `ANTHROPIC_API_KEY`
- Models <3B struggle with meta-optimization tasks
- For 16GB RAM systems, use `--model Qwen/Qwen2.5-3B-Instruct`
