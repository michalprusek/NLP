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

Prompt optimization framework implementing **OPRO** (Optimization by PROmpting) and **ProTeGi** (Prompt Optimization with Textual Gradients) algorithms for automatic prompt engineering on GSM8K (math reasoning) dataset.

## Commands

```bash
# Setup
uv sync                          # Install dependencies
cp .env.example .env             # Configure API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)

# OPRO optimization
uv run python run_opro.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# ProTeGi optimization
uv run python run_protegi.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10
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

**ProTeGi Optimizer** (`protegi.py`):
- Uses textual gradients (LLM-generated critiques) to guide prompt improvement
- Beam search with UCB (Upper Confidence Bound) bandit for exploration/exploitation
- Monte Carlo paraphrasing for local search around promising prompts
- Separate gradient and edit LLMs for modular optimization

**GSM8K Evaluator** (`gsm8k_evaluator.py`):
- Answer extraction: **Always extract the last number only** - no format-specific patterns like `####` or `\boxed{}`
- Numerical comparison with 1e-6 tolerance for floats
- **Prompt format: Q_end style** (from OPRO paper) - instruction comes AFTER the question:
  ```
  Q: {question}
  {instruction}
  A:
  ```

### Prompt Templates (`src/prompts/gsm8k/`)

- `opro_meta.txt`: Meta-optimizer prompt template for OPRO
- `gradient.txt`, `protegi_gradient.txt`: Gradient generation prompts
- `edit.txt`, `protegi_edit.txt`: Prompt editing instructions
- `protegi_paraphrase.txt`: Paraphrasing for exploration

## Model Aliases

```python
"haiku"  → claude-haiku-4-5-20251001
"sonnet" → claude-sonnet-4-5-20251022
"qwen"   → Qwen/Qwen2.5-7B-Instruct
"llama"  → meta-llama/Llama-3.1-8B-Instruct
```

## Key Parameters

- `--model` / `--meta-model`: Separate task evaluation and meta-optimization models
- `--backend`: `vllm` (fast, requires CUDA), `openai`, `deepinfra`, `auto`
- `--minibatch-size`: Examples per evaluation (trade-off: stability vs speed)
- `--tensor-parallel-size`: GPU count for vLLM tensor parallelism
- `--beam-width`: (ProTeGi) Number of candidate prompts to maintain
- `--ucb-c`: (ProTeGi) Exploration parameter for UCB bandit

## Datasets

### GSM8K (Math Reasoning)
- `datasets/gsm8k/train/`: Training set (Arrow format)
- `datasets/gsm8k/test/`: Test set (Arrow format)
- Primary benchmark for evaluating prompt quality

### HbBoPs (Hyperband Baseline of Prompts)
- `datasets/hbbops/ape_instructions_1000.json`: APE instruction dataset
- `datasets/hbbops/instructions_*.txt`: Filtered instruction sets
- `datasets/hbbops/examples_*.txt`: Example sets for few-shot learning
- `datasets/hbbops/full_grid_combined.jsonl`: Grid search results

## File Conventions

- `datasets/`: Static input data (read-only, version controlled)
- `results/`: Experiment outputs (gitignored) - JSON, CSV, logs from OPRO/ProTeGi runs
- `src/prompts/`: Prompt templates for meta-optimization

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
# Pattern for long-running optimization
tmux new-session -d -s <session_name> "CUDA_VISIBLE_DEVICES=<gpu> uv run python <script> <args> 2>&1 | tee results/<descriptive_name>_$(date +%Y%m%d_%H%M%S).log; exec bash"

# Example - OPRO with vLLM
tmux new-session -d -s opro_qwen "CUDA_VISIBLE_DEVICES=0 uv run python run_opro.py --model qwen --backend vllm --iterations 200 2>&1 | tee results/opro_qwen_200it_$(date +%Y%m%d_%H%M%S).log; exec bash"

# Example - ProTeGi with separate meta-model
tmux new-session -d -s protegi_beam10 "CUDA_VISIBLE_DEVICES=1 uv run python run_protegi.py --model qwen --meta-model sonnet --beam-width 10 --iterations 50 2>&1 | tee results/protegi_beam10_$(date +%Y%m%d_%H%M%S).log; exec bash"
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

## Algorithm Comparison

| Algorithm | Type | Strengths | Limitations |
|-----------|------|-----------|-------------|
| **OPRO** | Meta-optimization | Simple, interpretable, good for discrete prompt space | Requires many LLM calls, local optima |
| **ProTeGi** | Gradient-based | Directed search via textual gradients, beam search diversity | Gradient quality depends on critique LLM |

## Typical Workflows

### Running a Quick Experiment
```bash
# 10 iterations with default settings
uv run python run_opro.py --model qwen --iterations 10
```

### Production Run with Logging
```bash
# Create tmux session with full logging
tmux new-session -d -s opro_production \
  "CUDA_VISIBLE_DEVICES=0 uv run python run_opro.py \
  --model qwen --backend vllm --iterations 200 \
  --minibatch-size 512 --tensor-parallel-size 2 \
  2>&1 | tee results/opro_production_$(date +%Y%m%d_%H%M%S).log; \
  exec bash"

# Attach to monitor progress
tmux attach -t opro_production
```

### Comparing Models
```bash
# Run same configuration with different models
tmux new-session -d -s opro_qwen "uv run python run_opro.py --model qwen --iterations 100 2>&1 | tee results/opro_qwen_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s opro_llama "uv run python run_opro.py --model llama --iterations 100 2>&1 | tee results/opro_llama_$(date +%Y%m%d_%H%M%S).log"
```

## Development Guidelines

- **Read before modifying**: Always use `Read` tool on files before suggesting changes
- **Test changes**: Run small-scale experiments (5-10 iterations) before production runs
- **Version control**: Commit working configurations before major refactors
- **Documentation**: Update this file when adding new parameters, models, or workflows
- **Error handling**: Always validate LLM outputs and handle API errors gracefully
