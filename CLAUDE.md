# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Framework for automatic prompt optimization using **ProTeGi** (Prompt Optimization with Textual Gradients) and **OPRO** (Optimization by PROmpting) on the GSM8K math word problem dataset.

- **ProTeGi**: Uses LLM-generated critiques (textual gradients) + beam search + UCB bandit algorithm for prompt refinement
- **OPRO**: Uses LLM as meta-optimizer to generate improved prompts by analyzing previous results

## Common Commands

### Setup

**Install dependencies:**
```bash
uv sync
```

**Configure API keys:**
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your Anthropic API key for Claude models
# ANTHROPIC_API_KEY=your_key_here
```

### Running Optimizations

**Basic optimization (same model for task and meta-optimization):**
```bash
# ProTeGi with Qwen
uv run python main.py --method protegi --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# OPRO with SaulLM
uv run python main.py --method opro --model SaulLM/SaulLM-7B --iterations 10
```

**Separate task and meta-optimizer models:**
```bash
# Use Qwen for task evaluation, Claude Sonnet for meta-optimization
uv run python main.py \
    --method protegi \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --meta-model sonnet \
    --iterations 10

# Use small local model for task, Claude Haiku for meta-optimization (cost-effective)
uv run python main.py \
    --method protegi \
    --model Qwen/Qwen2.5-3B-Instruct \
    --meta-model haiku \
    --iterations 10
```

**Shell scripts (with environment variables and aliases):**
```bash
# Single GPU with separate models (using aliases)
META_MODEL="sonnet" ./run_single_gpu.sh protegi 5 20

# Dual GPU (using aliases)
TASK_MODEL="Qwen/Qwen2.5-7B-Instruct" META_MODEL="haiku" ./run_dual_gpu.sh

# Or with full model names
META_MODEL="claude-sonnet-4-5-20251022" ./run_single_gpu.sh protegi 5 20
```

**Claude Model Aliases:**
- `haiku` → `claude-haiku-4-5-20251001` (latest Haiku 4.5)
- `sonnet` → `claude-sonnet-4-5-20251022` (latest Sonnet 4.5)

**Evaluation only (no optimization):**
```bash
# Evaluate a prompt on GSM8K test set
uv run python evaluate_gsm8k.py --prompt "Your prompt here" --num-samples 5

# With self-consistency (majority voting)
uv run python evaluate_gsm8k.py --prompt "Your prompt" --num-samples 10
```

### Supported Models

**Task Models (--model):**
- `Qwen/Qwen2.5-7B-Instruct` - General-purpose model, good performance
- `Qwen/Qwen2.5-3B-Instruct` - Smaller, faster, lower memory
- `SaulLM/SaulLM-7B` - Legal domain-specialized model
- `meta-llama/Llama-3.1-8B-Instruct` - Meta's Llama 3.1
- `claude-3-haiku-20240307` - Fast Claude model (API)
- `claude-3-5-sonnet-20241022` - Most capable Claude model (API)

**Meta-optimizer Models (--meta-model):**
- Same as task models, plus recommended:
- `claude-3-5-sonnet-20241022` - Best for complex meta-optimization
- `claude-3-haiku-20240307` - Cost-effective for meta-optimization

### Important Parameters

**Model Selection:**
- `--model`: Task model (being optimized)
- `--backend`: Backend for task model (`auto`, `transformers`, `vllm`, `claude`)
- `--meta-model`: Meta-optimizer model for gradient generation/editing (optional, defaults to --model)
- `--meta-backend`: Backend for meta-optimizer model (`auto` detects Claude models)

**Optimization:**
- `--method`: `protegi` or `opro`
- `--iterations`: Number of optimization iterations (default: 10)
- `--minibatch-size`: Examples per evaluation (default: 20)
- `--beam-size`: Beam size for ProTeGi (default: 4)
- `--num-candidates`: Candidates per iteration for OPRO (default: 8, as per paper)

**Evaluation:**
- `--evaluator`: `strict-em` (exact match) or `math-verify` (robust symbolic verification, recommended)
- `--train-split` / `--val-split`: Which GSM8K split to use for training/validation

**Hardware:**
- `--device`: `auto`, `cuda`, `mps`, `cpu` (for transformers backend)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (vLLM only)
- `--gpu-ids`: Comma-separated GPU IDs (e.g., "0,1")

**Debug:**
- `--debug`: Show model outputs and extraction details

## Architecture

### Core Components

**Optimization Algorithms** (src/):
- `protegi.py`: ProTeGi implementation with beam search, UCB selection, and textual gradient application
- `opro.py`: OPRO implementation with meta-prompting for candidate generation

**Evaluation** (src/):
- `evaluator.py`: GSM8K evaluator with exact match (strict string comparison)
- `math_verify_evaluator.py`: Robust symbolic verification (handles equivalent mathematical expressions)
- Answer extraction uses prioritized patterns: `\boxed{NUMBER}`, `#### NUMBER`, `final_answer: NUMBER`, fallback to last number

**LLM Clients** (src/llm_client.py):
- `TransformersClient`: HuggingFace transformers backend (CPU/GPU/MPS)
- `VLLMClient`: vLLM backend (GPU only, much faster)
- `ClaudeClient`: Anthropic Claude API backend (requires ANTHROPIC_API_KEY in .env)
- Auto-detects and applies chat templates for Instruct models
- Handles device selection, dtype optimization, and batch generation
- Supports separate task and meta-optimizer models for hybrid optimization

### Key Design Patterns

**Dual-Model Architecture:**
Both ProTeGi and OPRO now support separate models for different roles:
- **Task Model**: The model being optimized - evaluates prompts on actual task (e.g., solving math problems)
- **Meta-optimizer Model**: Generates gradients, edits prompts, creates new candidates
- **Benefits**:
  - Use powerful API models (Claude Sonnet) for meta-optimization while keeping local models for task evaluation
  - Cost-effective: Small local model for task + Claude Haiku for meta-optimization
  - Better meta-optimization: Claude models excel at critique and prompt engineering
- **Implementation**: Both models share the same LLMClient interface, allowing seamless mixing of backends

**ProTeGi Workflow:**
1. Maintain beam of top-K prompt candidates
2. Select candidate via UCB (exploration/exploitation balance)
3. Evaluate on minibatch → Generate textual gradient (LLM critique)
4. Apply gradient → Generate N improved prompts
5. Evaluate new prompts, update beam with diversity penalty
6. Repeat until convergence or iteration limit

**OPRO Workflow:**
1. Evaluate initial seed prompts
2. Format top-K scored prompts as context for meta-optimizer
3. LLM generates N new candidate prompts
4. Evaluate candidates, keep top-K overall
5. Repeat, allowing LLM to learn patterns from successful prompts

**Stratified Sampling:**
- ProTeGi uses stratified sampling without replacement for better dataset coverage
- Tracks used indices, resets when >80% of dataset seen
- Helps avoid overfitting to specific examples

**Diversity Management:**
- ProTeGi beam selection includes diversity penalty (SequenceMatcher similarity)
- Prevents beam collapse to near-identical prompts
- Threshold: 85% similarity (configurable)

### Meta-Prompts

Both algorithms use carefully engineered meta-prompts:

**ProTeGi GRADIENT_PROMPT** (src/protegi.py:171-244):
- Critiques current prompt based on error examples
- Explains Math-Verify evaluation system (3-step extraction/parsing/verification)
- Requests 2-4 issues with root causes and specific improvements
- Structured JSON output format

**ProTeGi EDIT_PROMPT** (src/protegi.py:252-278):
- Applies critic's feedback to improve prompt
- Hard constraints: brevity (max 3 sentences/150 words), no meta-text, output only improved prompt
- Includes extensive post-processing to remove LLM artifacts

**OPRO META_PROMPT** (src/opro.py:25-64):
- Analyzes previous scored prompts
- Generates N new diverse candidates
- Strict output format (one prompt per line, no numbering/bullets)

### Prompt Cleaning

ProTeGi includes aggressive prompt cleaning (`apply_gradient` method):
- Removes markdown formatting, code fences, preambles
- Deduplicates sentences (consecutive and non-consecutive)
- Enforces length limits (300 words max)
- Strips meta-commentary patterns
- Validates output before accepting

## Dataset Structure

```
datasets/
├── gsm8k/          # GSM8K dataset (7,473 train, 1,319 test)
└── claudette/      # Claudette dataset (ToS analysis, not actively used)
```

## Results Output

```
results/
├── protegi_TIMESTAMP.json    # Full optimization history + config
├── protegi_TIMESTAMP.txt     # Best prompt + validation accuracy
├── opro_TIMESTAMP.json
└── opro_TIMESTAMP.txt
```

JSON includes: method, model, config, best_prompt, history (all iterations), validation scores

## Important Constraints

**API Keys:**
- Claude models require `ANTHROPIC_API_KEY` in `.env` file
- Copy `.env.example` to `.env` and add your API key
- Local models (Qwen, Llama, SaulLM) don't require API keys

**Memory Management:**
- Small models (3B-7B) recommended for 16GB RAM systems
- Use `--device mps` on Apple Silicon, `--torch-dtype bfloat16`
- vLLM requires CUDA GPU
- Claude API models have no local memory requirements

**Answer Format:**
- Models should output final answer as `#### NUMBER` (preferred)
- Math-Verify also handles: `\boxed{NUMBER}`, `final_answer: NUMBER`, `the answer is NUMBER`
- Evaluation is more forgiving than exact string match (handles fractions, decimals, units)

**Chat Templates:**
- Both LLM clients auto-detect Instruct models and apply chat templates
- Raw prompts are wrapped in user/assistant format for Instruct models
- Non-Instruct models use raw text

**Batch Generation:**
- Transformers: uses left padding for causal LM (ensures prompts end at same position)
- Low temperature (<0.3) triggers greedy decoding to avoid numerical instability
- vLLM: handles batching efficiently with repetition penalty

## Common Issues

**Small models (<3B):**
- Often struggle with meta-prompt generation (gradients, critiques)
- May fail at math reasoning
- Script warns when using <2B models

**Prompt artifacts:**
- LLMs sometimes generate meta-text instead of pure instruction
- ProTeGi's `apply_gradient` includes extensive cleaning logic
- If prompts look malformed, check `apply_gradient` post-processing

**Evaluation mismatch:**
- Use `--evaluator math-verify` for robust symbolic verification
- `strict-em` requires exact string match (less reliable)
- Debug with `--debug` flag to see extraction process
