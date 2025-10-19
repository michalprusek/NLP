# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Framework for automatic prompt optimization using **ProTeGi** (Prompt Optimization with Textual Gradients) and **OPRO** (Optimization by PROmpting) on the GSM8K math word problem dataset.

- **ProTeGi**: Uses LLM-generated critiques (textual gradients) + beam search + UCB bandit algorithm for prompt refinement
- **OPRO**: Uses LLM as meta-optimizer to generate improved prompts by analyzing previous results

## Common Commands

### Dependencies
```bash
# Install/sync dependencies
uv sync

# Run with uv
uv run python main.py [args]
```

### Running Optimizations

**Basic optimization:**
```bash
# ProTeGi
uv run python main.py --method protegi --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# OPRO
uv run python main.py --method opro --model Qwen/Qwen2.5-7B-Instruct --iterations 10
```

**Dual GPU optimization (tensor parallelism):**
```bash
# Uses both GPUs for single run (faster inference)
./run_dual_gpu.sh [method] [iterations] [minibatch_size]
# Example: ./run_dual_gpu.sh protegi 5 150
```

**Single GPU optimization:**
```bash
./run_single_gpu.sh
```

**Evaluation only (no optimization):**
```bash
# Evaluate a prompt on GSM8K test set
uv run python evaluate_gsm8k.py --prompt "Your prompt here" --num-samples 5

# With self-consistency (majority voting)
uv run python evaluate_gsm8k.py --prompt "Your prompt" --num-samples 10
```

### Important Parameters

- `--backend`: `transformers` (CPU/GPU) or `vllm` (GPU only, faster)
- `--evaluator`: `strict-em` (exact match) or `math-verify` (robust symbolic verification, recommended)
- `--train-split` / `--val-split`: Which GSM8K split to use for training/validation
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (vLLM only)
- `--gpu-ids`: Comma-separated GPU IDs (e.g., "0,1")
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
- Auto-detects and applies chat templates for Instruct models
- Handles device selection, dtype optimization, and batch generation

### Key Design Patterns

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

**Memory Management:**
- Small models (3B-7B) recommended for 16GB RAM systems
- Use `--device mps` on Apple Silicon, `--torch-dtype bfloat16`
- vLLM requires CUDA GPU

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
