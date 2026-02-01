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

## Research Context (NeurIPS Submission)

**Core Problem**: Sample-efficient optimization in high-dimensional latent spaces.

**Key Constraints**:
- **Extremely limited evaluation budget**: 6-25 full-fidelity evaluations (e.g., 25 prompts, 6 protein samples)
- **High-dimensional search space**: 1024D+ from pretrained autoencoders (SONAR, vec2text, ESM-C)
- **Expensive oracle**: Each evaluation costs significant compute (LLM inference, wet lab, simulation)

**Goal**: Outperform SOTA methods (ZeroInstruct, NFBO, GEPA, TuRBO) in low-sample regimes.

**Design Principles**:
- Every method must work with n < 30 observations
- Leverage pretrained encoder structure (manifold, semantics)
- Prioritize sample efficiency over wall-clock time
- Report results with statistical significance across random seeds

**Domains**: Prompt optimization (GSM8K), protein engineering (ESM-C), text-to-text (vec2text)

**Reference Papers**: `papers/` contains 40 PDFs covering flow matching, high-D BO, prompt optimization (OPRO, ProTeGi, MIPRO), TuRBO, SONAR, CFG-Zero*, and related SOTA methods. Read these before proposing new techniques.

This repository contains six prompt optimization methods:

1. **OPRO** (`opro/`) - Optimization by PROmpting for meta-optimization
2. **ProTeGi** (`protegi/`) - Prompt Optimization with Textual Gradients
3. **GEPA** (`gepa_gsm8k/`) - Genetic-Pareto Prompt Adaptation (GSM8K-specific)
4. **EcoFlow** (`ecoflow/`) - Guided flow matching in SONAR embedding space
5. **NFBO** (`nfbo_gsm8k/`) - Normalizing Flow Bayesian Optimization (GSM8K-specific)
6. **InstructZero** (`instructzero/`, `instructzero_gsm8k/`) - Soft prompt optimization with GP

## Repository Structure

```
NLP/
├── opro/                    # OPRO optimization method
│   ├── opro.py              # OPROOptimizer class
│   ├── run.py               # CLI entry point
│   ├── prompts/             # opro_meta.txt
│   └── results/             # Output directory
│
├── protegi/                 # ProTeGi optimization method
│   ├── protegi.py           # ProTeGiOptimizer class
│   ├── run.py               # CLI entry point
│   ├── prompts/             # gradient.txt, edit.txt, etc.
│   └── results/
│
├── gepa_gsm8k/              # GEPA for GSM8K task
│   └── run.py               # CLI entry point with integrated optimizer
│
├── ecoflow/                 # Flow matching + BO
│   ├── velocity_network.py  # DiT-style velocity network
│   ├── flow_model.py        # FlowMatchingModel
│   ├── guided_flow.py       # GuidedFlowSampler with CFG-Zero*
│   ├── gp_surrogate.py      # GP surrogate for 1024D optimization
│   ├── decoder.py           # SonarDecoder
│   ├── optimization_loop.py # BOOptimizationLoop
│   ├── run.py               # CLI entry point
│   └── results/
│
├── nfbo_gsm8k/              # NFBO for GSM8K task
│   ├── run.py               # CLI entry point
│   ├── train_textflow.py    # TextFlow model training
│   └── textflow/            # Discrete normalizing flow implementation
│
├── nfbo_original/           # Reference NFBO implementation from paper
│
├── instructzero/            # InstructZero core implementation
│   ├── gp_optimizer.py      # Gaussian Process optimizer
│   ├── loop.py              # Optimization loop
│   ├── soft_prompt.py       # Soft prompt handling
│   └── run.py               # CLI entry point
│
├── instructzero_gsm8k/      # InstructZero for GSM8K task
│   └── run.py               # CLI entry point
│
├── study/                   # Flow matching architecture study
│   ├── data/                # Data pipeline utilities
│   ├── flow_matching/       # Training, models, evaluation
│   └── run_all_experiments.py
│
├── shared/                  # Shared infrastructure
│   ├── llm_client.py        # LLMClient ABC + implementations
│   └── gsm8k_evaluator.py   # GSM8K evaluation utilities
│
├── datasets/                # Data files (read-only)
├── papers/                  # Reference papers (40 PDFs)
├── tests/                   # Test suite
├── CLAUDE.md                # This file
└── pyproject.toml           # Project configuration
```

---

## Running Optimization Methods

### OPRO

```bash
# Quick run
uv run python -m opro.run --model qwen --iterations 10

# Production run in tmux
tmux new-session -d -s opro_run \
  "CUDA_VISIBLE_DEVICES=0 uv run python -m opro.run \
  --model qwen --backend vllm --iterations 200 \
  2>&1 | tee opro/results/opro_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

### ProTeGi

```bash
# Quick run
uv run python -m protegi.run --model qwen --steps 6

# With separate meta-model
uv run python -m protegi.run --model qwen --meta-model sonnet --steps 10
```

### GEPA (GSM8K)

```bash
# Quick run
uv run python -m gepa_gsm8k.run --model qwen --budget 10000

# Production run
tmux new-session -d -s gepa_run \
  "uv run python -m gepa_gsm8k.run --model qwen --budget 150000 \
  2>&1 | tee gepa_gsm8k/results/gepa_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

### EcoFlow

```bash
# Train flow model
uv run python -m ecoflow.train_flow \
  --data datasets/sonar_embeddings.pt \
  --epochs 50 --batch-size 1024

# Run BO optimization
uv run python -m ecoflow.run \
  --flow-checkpoint path/to/flow.pt \
  --iterations 100
```

### NFBO (GSM8K)

```bash
# Train TextFlow model first
uv run python -m nfbo_gsm8k.train_textflow --data datasets/gsm8k

# Run NF-BO optimization
uv run python -m nfbo_gsm8k.run --iterations 50 --n-initial 20
```

### InstructZero

```bash
# Quick run (GSM8K)
uv run python -m instructzero_gsm8k.run --model qwen --iterations 20

# Production run
tmux new-session -d -s instructzero_run \
  "uv run python -m instructzero_gsm8k.run --model qwen --iterations 100 \
  2>&1 | tee instructzero_gsm8k/results/iz_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

---

## Shared Infrastructure

### LLM Client (`shared/llm_client.py`)

```python
from shared.llm_client import create_llm_client

# Factory function with auto-detection
client = create_llm_client("qwen", backend="vllm")  # Resolves to Qwen/Qwen2.5-7B-Instruct

# Model aliases
# "qwen"   → Qwen/Qwen2.5-7B-Instruct
# "llama"  → meta-llama/Llama-3.1-8B-Instruct
# "haiku"  → claude-haiku-4-5-20251001
# "sonnet" → claude-sonnet-4-5-20251022
```

### GSM8K Evaluator (`shared/gsm8k_evaluator.py`)

```python
from shared.gsm8k_evaluator import GSM8KEvaluator

evaluator = GSM8KEvaluator(dataset_path="datasets/gsm8k", split="test")
results = evaluator.evaluate_batch(outputs, indices)
```

**Prompt format (Q_end style):**
```
Q: {question}
{instruction}
A:
```

---

## Algorithm Comparison

| Algorithm | Type | Strengths | Limitations |
|-----------|------|-----------|-------------|
| **OPRO** | Meta-optimization | Simple, interpretable | Requires many LLM calls |
| **ProTeGi** | Gradient-based | Directed search, beam search | Gradient quality varies |
| **GEPA** | Evolutionary | Pareto diversity, reflection | Complex, slower convergence |
| **EcoFlow** | Flow + BO | Continuous space, GP-guided | Requires pretrained flow |
| **NFBO** | Normalizing Flow | Adaptive density modeling | Training overhead per step |
| **InstructZero** | GP + Soft Prompts | Sample-efficient, continuous | Requires soft prompt support |

---

## Key Parameters

### All Methods
- `--model`: Task model for evaluation
- `--meta-model`: Meta-optimizer model (default: same as --model)
- `--backend`: `vllm` (default), `openai`, `deepinfra`, `auto`
- `--tensor-parallel-size`: GPU count for vLLM

### OPRO
- `--iterations`: Optimization iterations (default: 200)
- `--minibatch-size`: Examples per evaluation (default: 261)
- `--num-candidates`: Candidates per iteration (default: 8)

### ProTeGi
- `--steps`: Optimization steps (default: 6)
- `--beam-size`: Beam width (default: 4)
- `--gradients`: Gradients per error group (default: 4)
- `--mc-samples`: Monte Carlo paraphrases (default: 2)

### GEPA (GSM8K)
- `--budget`: Total task LLM call budget (default: 150000)
- `--pareto-size`: Max Pareto front size (default: 10)
- `--mutations`: Mutations per iteration (default: 4)

### EcoFlow
- `--guidance-strength`: LCB guidance λ (default: 1.0)
- `--alpha`: LCB exploration weight (default: 1.0)
- `--n-candidates`: Candidates to generate (default: 64)

### InstructZero
- `--iterations`: Optimization iterations (default: 100)
- `--n-initial`: Initial random samples (default: 5)
- `--prompt-length`: Soft prompt token length (default: 10)

---

## Datasets

- `datasets/gsm8k/`: GSM8K math reasoning (train/test splits)
- `datasets/sonar_embeddings.pt`: 1.5M SONAR embeddings (1024D)
- `datasets/hbbops/`: Hyperband baseline prompts

---

## Ablation Study Results

### Best Flow Model
- **U-Net + OT-CFM + 10K** - Val loss ~1.07 (after 1000 epochs)
- Checkpoint: `study/checkpoints/unet-otcfm-10k-none/best.pt`
- Continue training: `./scripts/continue_unet_training.sh 500`

### Best GP Configuration
- **RiemannianGP + arccosine kernel** - Spearman=0.44, ECE=0.012
- ArcCosine kernel has NO lengthscale parameter - handle in `get_hyperparameters()`
- Kernel normalizes inputs internally: `k(x,y) = 1 - arccos(x̂·ŷ)/π`

### Riemannian Guided Flow (best for ArcCosine GP)
- **Spherical projection**: Project gradient onto tangent plane (don't change norm)
- **Flow-relative scaling**: Gradient ≤ 50% of flow velocity magnitude
- **Cutoff at 80%**: Stop guidance, let flow smooth last 20%
- **Uncertainty weighting**: Less guidance where GP is uncertain

```bash
# Run EcoFlow BO with best configuration (U-Net + arccosine GP + Riemannian guidance)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/prusek/NLP uv run python -m ecoflow.run_bo_full \
  --gp-kernel arccosine --warm-start-k 10 --llm-budget 50000 --eval-size 1319
```

### Flow Matching Methods

| Method | Coupling | Best For |
|--------|----------|----------|
| `icfm` | Random pairing | Baseline |
| `otcfm` | Optimal Transport | Straight paths, best overall |
| `spherical` | SLERP interpolation | Normalized embeddings |
| `spherical-ot` | OT + SLERP | Best for hypersphere data |
| `si-gvp` | Stochastic Interpolant | Alternative schedule |

**Spherical Flow**: Uses geodesic (great circle) paths instead of Euclidean linear interpolation. Ideal for SONAR embeddings which lie on unit hypersphere.

---

## Training → BO Automation

Use monitoring scripts to auto-start BO after training completes:

```bash
# Pattern: pgrep to detect training, then launch next step
scripts/run_bo_spherical.sh    # Monitors spherical training, auto-starts BO
scripts/run_bo_after_training.sh  # Monitors U-Net training, auto-starts BO

# Manual monitoring pattern
while pgrep -f "study.flow_matching.train.*<model>" > /dev/null; do sleep 30; done
# Then run BO with the trained checkpoint
```

---

## Coding Standards

### Logging Generated Prompts
**NEVER truncate prompts in log output or print statements.** Always output the full prompt text.

```python
# BAD - truncated output
log(f"Generated: {prompt[:80]}...")
print(f"Prompt: {prompt[:100]}...")
logger.info(f"  Prompt: {result['prompt'][:200]}...")

# GOOD - full prompt
log(f"Generated:\n{prompt}")
print(f"Prompt:\n{prompt}")
logger.info(f"Prompt:\n{result['prompt']}")
```

This applies to ALL prompt output: logs, prints, debug messages, result summaries.

### Long-Running Processes
**Always run processes >30 seconds in tmux** with logging:

```bash
tmux new-session -d -s <session_name> \
  "CUDA_VISIBLE_DEVICES=<gpu> uv run python -m <module>.run <args> \
  2>&1 | tee <module>/results/<name>_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

### Running Tests

```bash
uv run pytest tests/ -x -q
```

---

## Constraints

- vLLM requires CUDA GPU; Claude API requires `ANTHROPIC_API_KEY`
- Models <3B struggle with meta-optimization tasks
- **Always use `--backend vllm`** unless explicitly told otherwise
- **NEVER stop/kill running processes** unless user explicitly asks

## Autonomous Work Mode

**Work autonomously without asking for confirmation.** When checkpoints require human verification:
- Run the verification tests yourself
- Approve if tests pass, fix issues if they don't
- Continue to the next step without waiting for user input

## Development Guidelines

- **Read before modifying**: Always use `Read` tool on files before suggesting changes
- **Test changes**: Run small-scale experiments (5-10 iterations) before production runs
- **Run tests**: `uv run pytest tests/ -x -q`
- **Version control**: Commit working configurations before major refactors
- **Documentation**: Update this file when adding new parameters, models, or workflows
