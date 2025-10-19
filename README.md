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

The framework supports two datasets:

### GSM8K (Grade School Math 8K)
Math word problems requiring multi-step reasoning:
- **7,473 training examples**
- **1,319 test examples**
- Task: Extract final numerical answer from step-by-step solutions
- Dataset location: `datasets/gsm8k/` (already downloaded)

### Claudette (ToS Fairness Classification)
Terms of Service clause classification into 9 potentially unfair categories:
- **9 training examples** (few-shot setting)
- **2,048 test examples**
- Task: Classify clauses into categories (0-8)
- Categories:
  - 0: Limitation of liability
  - 1: Unilateral termination
  - 2: Unilateral change
  - 3: Arbitration
  - 4: Content removal
  - 5: Choice of law
  - 6: Other
  - 7: Contract by using
  - 8: Jurisdiction
- Dataset: Loaded from HuggingFace (`tommasobonomo/sem_eval_2023_task_4`)

## Usage

### 🚀 Dual GPU Optimization (DOPORUČENO pro 2x NVIDIA L40S)

Využijte tensor parallelism pro rychlejší inferenci na dvou GPU:

```bash
# Dual GPU s tensor parallelismem (obě GPU pro jeden běh - rychlejší)
./run_dual_gpu.sh [method] [iterations] [minibatch_size]

# Příklady:
./run_dual_gpu.sh protegi 5 150
./run_dual_gpu.sh opro 10 100

# Single GPU (pokud chcete použít jen jedno GPU)
./run_single_gpu.sh [method] [iterations] [minibatch_size]
```

**Výhody dual GPU s tensor parallelismem:**
- ⚡ Rychlejší inference (model rozdělen na 2 GPU)
- 💪 Podporuje větší modely
- 🔧 Automatická konfigurace vLLM

**Poznámka:** Tensor parallelism rozděluje model na více GPU pro rychlejší inferenci jednoho běhu. Pokud chcete spustit ProTeGi a OPRO současně, spusťte dva terminály s `--gpu-ids` parametrem.

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
- `--num-candidates`: Candidates per iteration for OPRO (default: 4)
- `--initial-prompt`: Starting prompt (default: loaded from `src/prompts/<task>/initial.txt`)
- `--output-dir`: Results directory (default: `results`)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (vLLM only, default: 1)
- `--gpu-ids`: Comma-separated GPU IDs (default: `0,1`)
- `--debug`: Enable debug output (shows model outputs and extraction details)
- `--quiet`: Suppress verbose output

## Supported Models

The framework supports any HuggingFace model. **Recommended models for systems with 16GB RAM:**

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
- `meta-llama/Llama-3.1-8B-Instruct` (~16GB RAM, ~84% on GSM8K)
- `Qwen/Qwen2.5-7B-Instruct` (~15GB RAM, ~85% on GSM8K)

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
4. **Update Memory**: Keep top-k prompts based on scores
5. **Iterate**: Repeat, allowing LLM to learn patterns from successful prompts

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

Pro evaluaci konkrétního promptu bez optimalizace použijte `evaluate_gsm8k.py`:

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

Pro evaluaci promptů na Claudette datasetu použijte `evaluate_claudette.py`:

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

- ProTeGi: [arXiv:2305.03495](https://arxiv.org/abs/2305.03495) - Automatic Prompt Optimization with Gradient Descent and Beam Search
- OPRO: [arXiv:2309.03409](https://arxiv.org/abs/2309.03409) - Large Language Models as Optimizers
- GSM8K: [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) - Training Verifiers to Solve Math Word Problems
- Claudette: SemEval-2023 Task 4 - Legal Extractive Question Answering ([HuggingFace](https://huggingface.co/datasets/tommasobonomo/sem_eval_2023_task_4))

## License

MIT
