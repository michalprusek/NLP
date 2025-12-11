# CLAUDE.md

Framework pro automatickou optimalizaci promptů pomocí **ProTeGi**, **OPRO** a **HbBoPs** na GSM8K datasetu.

## Quick Start

```bash
uv sync                          # Install dependencies
cp .env.example .env             # Configure API keys

# ProTeGi/OPRO optimization
uv run python main.py --method protegi --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# HbBoPs multi-fidelity optimization
uv run python hbbops/run_hbbops.py --instructions datasets/hbbops/instructions_25.txt --exemplars datasets/hbbops/examples_25.txt
```

## Repository Structure

```
/home/prusek/NLP/
├── src/                         # Core implementation
│   ├── protegi.py              # ProTeGi: beam search + textual gradients
│   ├── opro.py                 # OPRO: meta-optimizer prompting
│   ├── llm_client.py           # LLM backends (transformers, vllm, claude)
│   └── evaluator.py            # GSM8K evaluator
│
├── hbbops/                      # HbBoPs algorithm
│   ├── hbbops.py               # Core multi-fidelity optimization
│   ├── run_hbbops.py           # CLI entry point
│   └── results/                # Experiment outputs
│
├── generative_hbbops/           # Generative HbBoPs + Vec2Text
│   ├── generative_hbbops.py    # Main orchestrator
│   ├── deep_kernel.py          # GP with differentiable kernel
│   ├── acquisition.py          # Differentiable Expected Improvement
│   └── inverter.py             # Vec2Text embedding inversion
│
├── datasets/                    # Input data (read-only)
│   ├── gsm8k/                  # Math word problems
│   ├── tos_local/              # ToS classification
│   └── hbbops/                 # Instruction/exemplar databases
│
├── visualize/                   # Analysis scripts
├── results/                     # ProTeGi/OPRO outputs
└── main.py                      # Main entry point
```

## Best Practices

**File Organization:**
- `datasets/` — static input data, version controlled, never modified
- `results/`, `hbbops/results/` — experiment outputs, gitignored
- `visualize/` — analysis scripts with descriptive names

**Key Parameters:**
- `--model` / `--meta-model` — separate task and meta-optimizer models
- `--backend` — `vllm` (fast, GPU), `transformers` (flexible), `claude` (API)
- Model aliases: `haiku`, `sonnet` for latest Claude versions

**Evaluation:**
- Answer format: `final_answer: NUMBER` or `#### NUMBER`
- Numerical tolerance for float comparisons
- Use `--debug` for extraction details

**Constraints:**
- Claude API requires `ANTHROPIC_API_KEY` in `.env`
- vLLM requires CUDA GPU
- Models <3B struggle with meta-optimization tasks
