# Prompt Optimization Framework

Framework for automatic prompt optimization implementing **ProTeGi**, **OPRO**, **LIPO**, and **BOLT** algorithms for GSM8K math reasoning benchmark.

## Overview

This project implements four prompt optimization techniques:

### ProTeGi (Prompt Optimization with Textual Gradients)
Based on ["Automatic Prompt Optimization with 'Gradient Descent' and Beam Search"](https://arxiv.org/abs/2305.03495)
- Uses **textual gradients**: LLM-generated critiques that identify prompt weaknesses
- Applies **beam search** to maintain multiple candidate prompts
- Employs **UCB bandit algorithm** for exploration/exploitation balance

### OPRO (Optimization by PROmpting)
Based on ["Large Language Models as Optimizers"](https://arxiv.org/abs/2309.03409)
- Uses **LLM as meta-optimizer**: generates improved prompts by analyzing previous results
- Maintains **top-k scored prompts** in memory
- Evolutionary approach with temperature=1.0 for diversity

### LIPO (Latent Instruction Prompt Optimization)
Instruction-only optimization using VAE latent space:
- **VAE encoder**: 768D GTR embeddings → 32D latent space
- **GP with ARD**: Direct Bayesian optimization on latent space
- **Vec2Text inversion**: Converts optimized embeddings back to text

### BOLT (Bayesian Optimization over Latent Templates)
Joint instruction + exemplar optimization:
- **Structure-aware VAE**: Joint instruction (16D) + exemplar (8D) = 24D latent
- **Set Transformer**: Permutation-invariant exemplar encoding
- **Deep Kernel Learning GP**: Bayesian optimization on joint latent space

## Installation

```bash
# Install dependencies with uv
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY, OPENAI_API_KEY
```

## Usage

### OPRO Optimization
```bash
uv run python run_opro.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10
```

### ProTeGi Optimization
```bash
uv run python run_protegi.py --model Qwen/Qwen2.5-7B-Instruct --iterations 10
```

### LIPO Optimization
```bash
# Using pre-evaluated results (no LLM needed)
uv run python -m lipo.run --skip-hbbops --iterations 50

# Full run with LLM evaluation
uv run python -m lipo.run --iterations 10
```

### BOLT Optimization
```bash
# Using pre-evaluated Hyperband results (no LLM needed)
uv run python -m bolt.run --load-hyperband bolt/data/hyperband_results.json --iterations 30

# Full run with Hyperband evaluation
uv run python -m bolt.run --iterations 30
```

## Model Aliases

```python
"haiku" → claude-haiku-4-5-20251001
"sonnet" → claude-sonnet-4-5-20251022
"qwen"  → Qwen/Qwen2.5-7B-Instruct
```

## Key Parameters

- `--model`: Task evaluation model
- `--meta-model`: Meta-optimization model (defaults to --model)
- `--backend`: `vllm` (fast, requires CUDA), `openai`, `deepinfra`, `auto`
- `--iterations`: Number of optimization iterations
- `--minibatch-size`: Examples per evaluation

## Project Structure

```
.
├── src/                    # Core components
│   ├── llm_client.py       # LLM client abstraction (vLLM, OpenAI, etc.)
│   ├── opro.py             # OPRO optimizer
│   ├── gsm8k_evaluator.py  # GSM8K evaluation
│   └── prompts/            # Task-specific prompts
├── lipo/                   # LIPO method
│   ├── run.py              # Main entry point
│   ├── vae.py              # VAE encoder
│   ├── gp.py               # Gaussian Process
│   └── inference.py        # BO inference
├── bolt/                   # BOLT method
│   ├── run.py              # Main entry point
│   ├── vae.py              # Structure-aware VAE
│   ├── gp.py               # Deep kernel GP
│   └── inference.py        # BO inference
├── datasets/               # Static datasets
│   ├── gsm8k/              # GSM8K math problems
│   ├── hbbops/             # HbBoPs data
│   └── tos_local/          # ToS classification
├── results/                # Experiment outputs
├── run_opro.py             # OPRO runner
├── run_protegi.py          # ProTeGi runner
├── CLAUDE.md               # Claude Code instructions
└── pyproject.toml          # Dependencies
```

## Requirements

- Python 3.10+
- CUDA GPU (for vLLM backend)
- API keys for Claude/OpenAI (optional)

## References

- ProTeGi: [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
- OPRO: [arXiv:2309.03409](https://arxiv.org/abs/2309.03409)
- GSM8K: [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)

## License

MIT
