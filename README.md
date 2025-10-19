# Prompt Optimization Framework

Framework for automatic prompt optimization using **ProTeGi** (Prompt Optimization with Textual Gradients) and **OPRO** (Optimization by PROmpting) on the GSM8K math word problem dataset.

> **💡 Quick Start for 16GB RAM systems:** Use `./quick_test.sh` or run with `--model Qwen/Qwen2.5-3B-Instruct` instead of larger 7B/8B models.

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

## Dataset

The project uses the **GSM8K** dataset (Grade School Math 8K), containing:
- 7,473 training examples
- 1,319 test examples

Dataset is already downloaded in `datasets/gsm8k/`.

## Usage

### 🚀 Paralelní běh na 2 GPU (DOPORUČENO pro 2x NVIDIA L40S)

Spusťte ProTeGi a OPRO současně na dvou GPU:

```bash
# Základní použití (Transformers backend)
./run_parallel.sh

# S vlastním modelem
./run_parallel.sh Qwen/Qwen2.5-7B-Instruct

# S vLLM backendem (rychlejší inference)
./run_parallel_vllm.sh

# Manuální spuštění s vlastními parametry
python main_parallel.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --iterations 10 \
    --gpu-protegi 0 \
    --gpu-opro 1
```

**Výhody paralelního běhu:**
- ⚡ 2x rychlejší než sekvenční běh
- 🔬 Porovnání obou metod najednou
- 💪 Plné využití obou GPU
- 📊 Automatické porovnání výsledků

**Viz [PARALLEL_OPTIMIZATION.md](PARALLEL_OPTIMIZATION.md) pro kompletní dokumentaci.**

### Basic Commands

```bash
# ProTeGi optimization with Qwen2.5 3B (recommended for 16GB RAM)
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --minibatch-size 20 \
  --beam-size 4

# OPRO optimization with Qwen2.5 3B
uv run python main.py \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --num-candidates 4

# Quick test with smallest model (very fast)
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --iterations 5 \
  --minibatch-size 10
```

### Available Arguments

**Required:**
- `--method`: Optimization method (`protegi` or `opro`)
- `--model`: HuggingFace model name

**Optional:**
- `--backend`: LLM backend (`transformers` or `vllm`, default: `transformers`)
- `--dataset-path`: Path to GSM8K dataset (default: `datasets/gsm8k`)
- `--split`: Dataset split (`train` or `test`, default: `test`)
- `--iterations`: Number of optimization iterations (default: 10)
- `--minibatch-size`: Examples per evaluation (default: 20)
- `--beam-size`: Beam size for ProTeGi (default: 4)
- `--num-candidates`: Candidates per iteration for OPRO (default: 4)
- `--initial-prompt`: Starting prompt (default: basic math solving prompt)
- `--output-dir`: Results directory (default: `results`)
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
FINAL EVALUATION (100 examples):
Accuracy: 87.0%
Correct: 87/100
================================================================================
```

### 2. JSON file (.json) with full results:
```json
{
  "method": "protegi",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "timestamp": "20250118_143022",
  "best_prompt": "Optimized prompt text...",
  "history": [...],
  "final_evaluation": {
    "accuracy": 0.87,
    "correct": 87,
    "total": 100
  }
}
```

## Project Structure

```
.
├── datasets/
│   ├── gsm8k/              # GSM8K dataset
│   └── claudette/          # Claudette dataset (ToS analysis)
├── src/
│   ├── __init__.py
│   ├── evaluator.py        # GSM8K evaluation logic
│   ├── llm_client.py       # LLM client abstraction
│   ├── protegi.py          # ProTeGi implementation
│   └── opro.py             # OPRO implementation
├── results/                # Optimization results (created on first run)
├── main.py                 # CLI entry point
└── README.md
```

## Examples

### Quick Test with Smallest Model

```bash
# Ultra-fast test with 1.5B model
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --iterations 5 \
  --minibatch-size 10
```

### Full Optimization Run (Recommended)

```bash
# ProTeGi with 3B model - good balance of speed and performance
uv run python main.py \
  --method protegi \
  --model Qwen/Qwen2.5-3B-Instruct \
  --iterations 10 \
  --minibatch-size 20 \
  --beam-size 4
```

### Custom Initial Prompt

```bash
uv run python main.py \
  --method opro \
  --model Qwen/Qwen2.5-3B-Instruct \
  --initial-prompt "Let's solve this math problem carefully, showing all steps."
```

### Using Quick Test Script

```bash
# Pre-configured script for testing
./quick_test.sh
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

- ProTeGi: [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
- OPRO: [arXiv:2309.03409](https://arxiv.org/abs/2309.03409)
- GSM8K: [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)

## License

MIT
