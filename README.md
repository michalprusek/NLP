# Prompt Optimization Framework

Framework for automatic prompt optimization using **ProTeGi** (Prompt Optimization with Textual Gradients) and **OPRO** (Optimization by PROmpting) on multiple datasets:
- **GSM8K**: Math word problems (regression task)
- **Claudette**: Terms of Service fairness classification (classification task)

> **💡 Quick Start for 16GB RAM systems:** Use `--model Qwen/Qwen2.5-3B-Instruct` instead of larger 7B/8B models for better performance on limited RAM.

## Overview

This project implements two state-of-the-art prompt optimization techniques:

### ProTeGi (Prompt Optimization with Textual Gradients)
Based on ["Automatic Prompt Optimization with 'Gradient Descent' and Beam Search"](https://arxiv.org/abs/2305.03495)

- Uses **textual gradients**: LLM-generated critiques that identify prompt weaknesses
- Applies **beam search** to maintain multiple candidate prompts
- Employs **UCB bandit algorithm** for exploration/exploitation balance
- Iteratively refines prompts based on error analysis

### OPRO (Optimization by PROmpting)
Based on ["Large Language Models as Optimizers"](https://arxiv.org/abs/2309.03409)

- Uses **LLM as meta-optimizer**: generates improved prompts by analyzing previous results
- Maintains **top-k scored prompts** in memory
- Evolutionary approach: new candidates are generated based on patterns in successful prompts
- Simple but effective derivative-free optimization

## Installation

This project uses `uv` for dependency management:

```bash
# Dependencies are already installed
# If you need to reinstall:
uv sync
```

## Datasets

The framework supports two datasets. Both are already included in the repository.

### Downloading Datasets (if needed)

If you need to re-download the datasets:

**GSM8K:**
```bash
# GSM8K is included in the repository at datasets/gsm8k/
# To re-download from HuggingFace:
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")
ds.save_to_disk("datasets/gsm8k")
```

**Claudette:**
```bash
# Claudette is included in the repository at datasets/claudette/
# To re-download from HuggingFace:
from datasets import load_dataset
ds = load_dataset("joelniklaus/online_terms_of_service")
ds.save_to_disk("datasets/claudette")
```

### GSM8K (Grade School Math 8K)
Math word problems requiring multi-step reasoning:
- **7,473 training examples**
- **1,319 test examples**
- Task: Extract final numerical answer from step-by-step solutions
- Source: HuggingFace dataset [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
- Dataset location: `datasets/gsm8k/` (already downloaded)

### Claudette (ToS Fairness Classification)
Terms of Service clause classification into 9 potentially unfair categories:
- **19,942 training examples** (2,295 labeled with unfair categories)
- **1,690 validation examples** (191 labeled)
- **4,297 test examples** (417 labeled)
- Task: Multi-label classification of clauses into unfair categories (0-8)
- Categories:
  - 0: Limitation of liability (ltd)
  - 1: Unilateral termination (ter)
  - 2: Unilateral change (ch)
  - 3: Arbitration (a)
  - 4: Content removal (cr)
  - 5: Choice of law (law)
  - 6: Other (pinc)
  - 7: Contract by using (use)
  - 8: Jurisdiction (j)
- Source: HuggingFace dataset [`joelniklaus/online_terms_of_service`](https://huggingface.co/datasets/joelniklaus/online_terms_of_service)
- Dataset location: `datasets/claudette/` (local preprocessed version)

## Usage

### 🚀 Dual GPU Optimization (Recommended for 2x NVIDIA L40S)

Utilize tensor parallelism for faster inference on two GPUs:

```bash
# Dual GPU with tensor parallelism (both GPUs for one run - faster)
./run_dual_gpu.sh [method] [iterations] [minibatch_size]

# Examples:
./run_dual_gpu.sh protegi 5 150
./run_dual_gpu.sh opro 10 100

# Single GPU (if you want to use only one GPU)
./run_single_gpu.sh [method] [iterations] [minibatch_size]
```

**Benefits of dual GPU with tensor parallelism:**
- ⚡ Faster inference (model split across 2 GPUs)
- 💪 Supports larger models
- 🔧 Automatic vLLM configuration

**Note:** Tensor parallelism splits the model across multiple GPUs for faster inference of a single run. If you want to run ProTeGi and OPRO simultaneously, launch two terminals with the `--gpu-ids` parameter.

### Basic Commands

#### GSM8K (Math Problems)

```bash
# ProTeGi optimization with Qwen2.5 3B (recommended for 16GB RAM)
uv run python main.py \
  --task gsm8k \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --minibatch-size 20 \
  --beam-size 4

# OPRO optimization with Qwen2.5 3B
uv run python main.py \
  --task gsm8k \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --num-candidates 4
```

#### Claudette (ToS Classification)

```bash
# ProTeGi optimization for Claudette
uv run python main.py \
  --task claudette \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --minibatch-size 50 \
  --beam-size 4

# OPRO optimization for Claudette
uv run python main.py \
  --task claudette \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --num-candidates 4
```

**Note:** Task-specific prompts are automatically loaded from `src/prompts/<task>/initial.txt`. The framework adapts meta-prompts for classification vs. regression tasks.

### Available Arguments

**Required:**
- `--method`: Optimization method (`protegi` or `opro`)
- `--model`: HuggingFace model name

**Optional:**
- `--task`: Task to optimize (`gsm8k` or `claudette`, default: `gsm8k`)
- `--backend`: LLM backend (`transformers` or `vllm`, default: `transformers`)
- `--evaluator`: Evaluator type for GSM8K (`strict-em` or `math-verify`, default: `math-verify`)
  - `strict-em`: Exact string matching (strict)
  - `math-verify`: Robust symbolic verification (handles equivalent expressions, **recommended**)
  - Note: Claudette always uses classification evaluator
- `--dataset-path`: Path to dataset (defaults: `datasets/gsm8k` for GSM8K, `tommasobonomo/sem_eval_2023_task_4` for Claudette)
- `--train-split`: Dataset split for training/optimization (`train` or `test`, default: `train`)
- `--val-split`: Dataset split for validation (`train` or `test`, default: `test`)
- `--iterations`: Number of optimization iterations (default: 10)
- `--minibatch-size`: Examples per evaluation (default: 20)
- `--beam-size`: Beam size for ProTeGi (default: 4)
- `--num-candidates`: Candidates per iteration for OPRO (default: 8, as per paper)
- `--initial-prompt`: Starting prompt (default: loaded from `src/prompts/<task>/initial.txt`)
- `--output-dir`: Results directory (default: `results`)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (vLLM only, default: 1)
- `--gpu-ids`: Comma-separated GPU IDs (default: `0,1`)
- `--debug`: Enable debug output (shows model outputs and extraction details)
- `--quiet`: Suppress verbose output

## Supported Models

The framework supports any HuggingFace model. This project primarily uses **Qwen/Qwen2.5-7B-Instruct** for optimization tasks.

**Recommended models for systems with 16GB RAM:**

### Qwen2.5 3B Instruct (RECOMMENDED for limited RAM)
- Good math performance (~70-75% on GSM8K)
- Only ~6GB RAM required
- Fast inference on CPU/MPS
```bash
--model Qwen/Qwen2.5-3B-Instruct
```

### Qwen2.5 1.5B Instruct (for very limited RAM)
- Smaller but still capable (~60-65% on GSM8K)
- Only ~3GB RAM required
- Very fast, works even on 8GB systems
```bash
--model Qwen/Qwen2.5-1.5B-Instruct
```

### Qwen2.5 0.5B Instruct (ultra-light)
- Minimal resource requirements (~1.5GB RAM)
- Basic math capabilities (~40-50% on GSM8K)
- Extremely fast, good for testing
```bash
--model Qwen/Qwen2.5-0.5B-Instruct
```

**For systems with 32GB+ RAM (more powerful but need more memory):**
- `Qwen/Qwen2.5-7B-Instruct` (~15GB RAM, ~85% on GSM8K) - **Default model used in this project**
- `meta-llama/Llama-3.1-8B-Instruct` (~16GB RAM, ~84% on GSM8K)

### Claude API Models (via Anthropic API)

The framework supports Anthropic's Claude models via API for both task execution and meta-optimization:

**Setup:**
```bash
# Copy example .env file
cp .env.example .env

# Add your Anthropic API key to .env
# ANTHROPIC_API_KEY=your_key_here
```

**Model Aliases:**
- `haiku` → `claude-haiku-4-5-20251001` (latest Haiku 4.5)
- `sonnet` → `claude-sonnet-4-5-20251022` (latest Sonnet 4.5)

**Usage Examples:**
```bash
# Use Claude Sonnet for both task and meta-optimization
uv run python main.py \
  --method protegi \
  --model sonnet \
  --iterations 10

# Hybrid: Local model for task, Claude for meta-optimization (RECOMMENDED)
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend vllm \
  --meta-model sonnet \
  --iterations 10

# Cost-effective: Small local model + Claude Haiku for meta-optimization
uv run python main.py \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --meta-model haiku \
  --iterations 10
```

**Dual-Model Architecture:**

Both ProTeGi and OPRO support separate models for different roles:
- **Task Model** (`--model`): Evaluates prompts on actual task (solving math problems)
- **Meta-optimizer Model** (`--meta-model`): Generates gradients, edits prompts, creates new candidates

**Benefits:**
- 💡 Use powerful API models (Claude Sonnet) for meta-optimization while keeping local models for task evaluation
- 💰 Cost-effective: Small local model for task + Claude Haiku for meta-optimization
- 🚀 Better meta-optimization: Claude models excel at critique and prompt engineering
- 🔧 No local memory requirements for API models

**Available Parameters:**
- `--meta-model`: Meta-optimizer model (optional, defaults to `--model`)
- `--meta-backend`: Backend for meta-optimizer (`auto`, `transformers`, `vllm`, `claude`)

## How It Works

### ProTeGi Pipeline

1. **Initialize**: Start with an initial prompt
2. **Evaluate**: Test prompt on minibatch, collect errors
3. **Generate Gradient**: LLM critiques the prompt based on errors
4. **Apply Gradient**: LLM generates improved prompt based on critique
5. **Beam Search**: Maintain top-k candidates, select next via UCB
6. **Iterate**: Repeat until convergence or budget exhausted

### OPRO Pipeline

1. **Initialize**: Start with one or more seed prompts
2. **Evaluate**: Score each prompt on minibatch
3. **Generate Candidates**: LLM analyzes scored prompts and generates new candidates
   - Makes N independent calls to meta-optimizer (temperature=1.0 for diversity)
   - Each call uses 3 randomly sampled examples from training set
   - Generates 8 candidates per iteration (default, as per paper)
4. **Update Memory**: Keep top-20 prompts based on scores (paper default)
5. **Iterate**: Repeat, allowing LLM to learn patterns from successful prompts

**Paper-Compliant Implementation:** Our OPRO follows the original paper (arXiv:2309.03409) with temperature=1.0, random examples, and independent generation calls for maximum diversity.

### Answer Evaluation (Math-Verify)

The framework uses a robust 3-step evaluation process:

**Step 1 - Extraction** (prioritizes later matches):
- Highest priority: `\boxed{NUMBER}`, `#### NUMBER`, `final_answer: NUMBER`
- Medium priority: `the answer is NUMBER`, `therefore NUMBER`
- Fallback: last number in output

**Step 2 - Parsing** (normalization):
- Removes: commas (1,234 → 1234), currency symbols ($50 → 50), units
- Handles: percentages (50% → 0.5), fractions (1/3), decimals (3.14)
- Converts to symbolic representation for smart comparison

**Step 3 - Verification** (multiple strategies):
- Numerical equality with tolerance (1/3 ≈ 0.333...)
- Symbolic simplification (checks if predicted - ground_truth = 0)
- Expression equivalence (different forms of same answer)

This is **more forgiving** than exact string matching:
- "1/3" and "0.333" are considered equivalent
- "42 km" and "42" are considered equivalent
- Different mathematical representations of same value are equivalent

## Output

Results are saved to `results/` directory in two formats:

### 1. Text file (.txt) with the optimized prompt:
```
# PROTEGI Optimized Prompt
# Model: meta-llama/Llama-3.1-8B-Instruct
# Timestamp: 20250118_143022
# Initial prompt: Solve the following math problem step by step...

================================================================================
OPTIMIZED PROMPT:
================================================================================

[Optimized prompt text here]

================================================================================
VALIDATION (test split, 1319 examples):
Accuracy: 87.0%
Correct: 1148/1319
================================================================================
```

### 2. JSON file (.json) with full results:
```json
{
  "method": "protegi",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "backend": "vllm",
  "timestamp": "20250118_143022",
  "config": {
    "iterations": 10,
    "minibatch_size": 20,
    "beam_size": 4,
    "train_split": "train",
    "val_split": "test",
    "train_size": 7473,
    "val_size": 1319
  },
  "best_prompt": "Optimized prompt text...",
  "history": [...],
  "validation": {
    "split": "test",
    "accuracy": 0.87,
    "correct": 1148,
    "total": 1319
  }
}
```

## Project Structure

```
.
├── datasets/
│   └── gsm8k/                        # GSM8K dataset (7,473 train, 1,319 test)
├── src/
│   ├── __init__.py
│   ├── evaluator.py                  # GSM8K evaluator (strict exact match)
│   ├── math_verify_evaluator.py      # Math-Verify evaluator (robust symbolic)
│   ├── claudette_evaluator.py        # Claudette evaluator (classification)
│   ├── llm_client.py                 # LLM client abstraction (Transformers + vLLM)
│   ├── protegi.py                    # ProTeGi implementation (multi-task)
│   ├── opro.py                       # OPRO implementation (multi-task)
│   └── prompts/                      # Task-specific prompt templates
│       ├── gsm8k/
│       │   └── initial.txt           # Default GSM8K prompt
│       └── claudette/
│           ├── initial.txt           # Default Claudette prompt
│           ├── gradient.txt          # ProTeGi gradient/critique prompt
│           ├── edit.txt              # ProTeGi edit prompt
│           └── opro_meta.txt         # OPRO meta-optimization prompt
├── results/                          # Optimization results (created on first run)
├── main.py                           # Main CLI entry point for optimization
├── evaluate_gsm8k.py                 # GSM8K evaluation (with self-consistency)
├── evaluate_claudette.py             # Claudette evaluation (with self-consistency)
├── run_dual_gpu.sh                   # Dual GPU optimization with tensor parallelism
├── run_single_gpu.sh                 # Single GPU optimization
├── CLAUDE.md                         # Documentation for Claude Code
├── README.md
└── pyproject.toml                    # Dependencies (managed by uv)
```

## Standalone Evaluation

### GSM8K Evaluation

To evaluate a specific prompt without optimization, use `evaluate_gsm8k.py`.

**Note:** The script now correctly uses evaluator-specific comparison methods:
- With `--evaluator math-verify`: Uses symbolic verification (recommended)
- With `--evaluator standard`: Uses numerical tolerance comparison
- Both are more robust than simple string matching

```bash
# Basic evaluation (single response per question)
uv run python evaluate_gsm8k.py --prompt "Solve this math problem step by step."

# With self-consistency (majority voting over 5 responses)
uv run python evaluate_gsm8k.py \
    --prompt "Solve step by step and provide answer as #### NUMBER" \
    --num-samples 5

# Using vLLM for faster inference
uv run python evaluate_gsm8k.py \
    --backend vllm \
    --num-samples 10 \
    --evaluator math-verify

# Quick test on subset
uv run python evaluate_gsm8k.py \
    --prompt "Your prompt here" \
    --max-examples 100 \
    --num-samples 3
```

**Available prompt templates:**
- `--prompt-name basic` (default)
- `--prompt-name cot` (chain-of-thought)
- `--prompt-name concise`
- `--prompt-name detailed`

Or use `--prompt "Your custom prompt"` for custom prompts.

### Claudette Evaluation

To evaluate prompts on the Claudette dataset, use `evaluate_claudette.py`:

```bash
# Basic evaluation (single response per clause)
uv run python evaluate_claudette.py \
    --prompt "Classify this Terms of Service clause into one of 9 categories (0-8)."

# With self-consistency (majority voting over 5 responses)
uv run python evaluate_claudette.py \
    --prompt "Analyze this clause and classify. Provide: LABEL: <number>" \
    --num-samples 5

# Using vLLM for faster inference
uv run python evaluate_claudette.py \
    --backend vllm \
    --num-samples 10

# Quick test on subset
uv run python evaluate_claudette.py \
    --prompt-name reasoning \
    --max-examples 100 \
    --num-samples 3
```

**Available prompt templates:**
- `--prompt-name basic` (default)
- `--prompt-name detailed` (includes all 9 categories)
- `--prompt-name reasoning` (encourages chain-of-thought)

## Examples

### Quick Test with Smallest Model (GSM8K)

```bash
# Ultra-fast test with 1.5B model
uv run python main.py \
  --task gsm8k \
  --method protegi \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --iterations 5 \
  --minibatch-size 10
```

### Full Optimization Run (Recommended)

```bash
# ProTeGi with 3B model - good balance of speed and performance (GSM8K)
uv run python main.py \
  --task gsm8k \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --minibatch-size 20 \
  --beam-size 4

# OPRO for Claudette ToS classification
uv run python main.py \
  --task claudette \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --minibatch-size 50 \
  --num-candidates 4
```

### Custom Initial Prompt

```bash
# GSM8K with custom prompt
uv run python main.py \
  --task gsm8k \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --initial-prompt "Let's solve this math problem carefully, showing all steps."

# Claudette with custom prompt
uv run python main.py \
  --task claudette \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --initial-prompt "Read the clause, identify its legal implications, and classify."
```


## Performance Tips

### Memory Management (Important for macOS/limited RAM)

**If you have limited RAM (< 32GB):**

1. **Use a smaller model**:
   ```bash
   # Qwen 3B instead of 7B (much less memory)
   --model Qwen/Qwen2.5-3B-Instruct

   # Or even smaller: 1.5B models
   --model Qwen/Qwen2.5-1.5B-Instruct
   ```

2. **Use CPU for large models** (slower but works):
   ```bash
   --device cpu
   ```

3. **Data type optimization** (automatic by default):
   ```bash
   # Auto selects best for your device
   --torch-dtype auto

   # For Apple Silicon (MPS): bfloat16 is recommended
   --torch-dtype bfloat16

   # For CUDA: float16
   --torch-dtype float16

   # Lowest memory (slowest): float32 on CPU
   --device cpu --torch-dtype float32
   ```

### Speed Optimization

1. **Use vLLM for faster inference** (requires CUDA GPU):
   ```bash
   --backend vllm
   ```

2. **Adjust minibatch size** based on your memory:
   - Smaller (10-15): Less memory, faster iterations
   - Larger (30-50): More stable gradients, slower

3. **Start with fewer iterations** for testing:
   ```bash
   --iterations 5
   ```

### Recommended Configurations

**macOS with 16GB RAM (limited memory):**
```bash
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device mps \
  --torch-dtype bfloat16 \
  --iterations 10 \
  --minibatch-size 10
```

**macOS with 32GB+ RAM:**
```bash
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --iterations 10
```

**Linux/Windows with CUDA GPU:**
```bash
uv run python main.py \
  --method opro \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend vllm \
  --iterations 10
```

## References

**Optimization Methods:**
- ProTeGi: [arXiv:2305.03495](https://arxiv.org/abs/2305.03495) - Automatic Prompt Optimization with Gradient Descent and Beam Search
- OPRO: [arXiv:2309.03409](https://arxiv.org/abs/2309.03409) - Large Language Models as Optimizers

**Datasets:**
- GSM8K: [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) - Training Verifiers to Solve Math Word Problems
  - HuggingFace: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
- CLAUDETTE: [arXiv:1805.01217](https://arxiv.org/abs/1805.01217) - Automated Detector of Potentially Unfair Clauses in Online Terms of Service
  - HuggingFace: [`joelniklaus/online_terms_of_service`](https://huggingface.co/datasets/joelniklaus/online_terms_of_service)

## License

MIT
