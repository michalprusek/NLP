# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hardware Environment

- **GPU 0**: NVIDIA A100 80GB PCIe (main compute)
- **GPU 1**: NVIDIA RTX A5000 24GB (secondary, used for vLLM task model when GPU 0 is busy)

Notes:
- GPUs are heterogeneous — do NOT use DDP/torchrun across them
- Use `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` to target a specific GPU
- Use `pin_memory=True` and `num_workers=8` for DataLoaders
- Enable mixed precision (bf16) when appropriate

## Research Context (NeurIPS Submission)

**Core Problem**: Sample-efficient optimization in high-dimensional latent spaces.

**Current Focus**:
- **Molecular optimization** (GuacaMol): Subspace BO on SELFIES VAE (256D latent → 16D subspace)
- **Prompt optimization** (GSM8K): OPRO, ProTeGi, GEPA benchmarks

**Goal**: Outperform SOTA methods (TuRBO, NFBO, GEPA) in high-dimensional settings.

**Reference Papers**: `papers/` contains 40 PDFs covering flow matching, high-D BO, prompt optimization, TuRBO, and related methods.

## Repository Structure

```
NLP/
├── opro/                    # OPRO optimization method
│   ├── opro.py              # OPROOptimizer class
│   ├── run.py               # CLI entry point
│   └── results/
│
├── protegi/                 # ProTeGi optimization method
│   ├── protegi.py           # ProTeGiOptimizer class
│   ├── run.py               # CLI entry point
│   └── results/
│
├── gepa_gsm8k/              # GEPA for GSM8K task (with BatchingVLLMWrapper)
│   └── run.py               # CLI entry point with integrated optimizer
│
├── rielbo/                  # Subspace BO for molecular optimization
│   ├── subspace_bo.py       # SphericalSubspaceBO (v1, ArcCosine/Hvarfner)
│   ├── subspace_bo_v2.py    # V2: geodesic/novelty presets
│   ├── subspace_bo_v3.py    # V3: multi-projection rotation
│   ├── subspace_bo_v4.py    # V4: novelty-weighted acquisition
│   ├── subspace_bo_v5.py    # V5: latest variant
│   ├── turbo_baseline.py    # TuRBO baseline (R^256)
│   ├── vanilla_bo.py        # Vanilla BO with Hvarfner priors (256D)
│   ├── kernels.py           # ArcCosineKernel
│   ├── gp_diagnostics.py    # GP health monitoring
│   ├── plot_convergence.py  # Convergence plots
│   ├── run_guacamol_subspace.py    # CLI: Subspace BO v1
│   ├── run_guacamol_subspace_v2.py # CLI: Subspace BO v2
│   ├── run_guacamol_subspace_v3.py # CLI: Subspace BO v3
│   ├── run_guacamol_subspace_v4.py # CLI: Subspace BO v4
│   ├── run_guacamol_subspace_v5.py # CLI: Subspace BO v5
│   ├── run_guacamol_vanilla.py     # CLI: Vanilla BO
│   └── results/
│
├── shared/                  # Shared infrastructure
│   ├── llm_client.py        # LLMClient ABC (vLLM, Anthropic, OpenAI)
│   ├── gsm8k_evaluator.py   # GSM8K evaluation utilities
│   ├── incremental_saver.py # JSON incremental checkpointing
│   └── guacamol/            # GuacaMol codec, oracle, data loaders
│       ├── codec.py         # SELFIES VAE (256D latent)
│       ├── oracle.py        # GuacaMol scoring
│       └── data.py          # Data loaders (GuacaMol CSV, ZINC)
│
├── datasets/                # Data files (read-only)
├── papers/                  # Reference papers (40 PDFs)
├── tests/                   # Test suite
└── pyproject.toml           # Project configuration
```

**Kept but not actively developed**:
- `instructzero/`, `instructzero_gsm8k/` — InstructZero

---

## Running Optimization Methods

### OPRO

```bash
# Quick run
uv run python -m opro.run --model qwen --iterations 10

# With Sonnet meta-model (Qwen evaluates, Sonnet generates candidates)
CUDA_VISIBLE_DEVICES=1 uv run python -m opro.run \
  --model qwen --backend vllm --meta-model sonnet \
  --gpu-ids 1 --iterations 200 --max-prompts 100 --split test
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

# Production run (uses BatchingVLLMWrapper for ~100x speedup)
tmux new-session -d -s gepa_run \
  "uv run python -m gepa_gsm8k.run --model qwen --budget 150000 \
  2>&1 | tee gepa_gsm8k/results/gepa_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

### Subspace BO (GuacaMol)

```bash
# Run Subspace BO v1
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --subspace-dim 16 --task-id adip --n-cold-start 100 --iterations 500

# With Hvarfner kernel (RBF + LogNormal + ARD, BoTorch defaults)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
    --subspace-dim 16 --kernel hvarfner --task-id adip --iterations 500

# Run Subspace BO v2 (geodesic preset, recommended)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
    --preset geodesic --task-id adip --n-cold-start 100 --iterations 500
```

### Vanilla BO (Hvarfner, 256D)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_vanilla \
    --task-id adip --n-cold-start 100 --iterations 500 --acqf ts
```

### TuRBO Baseline

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.turbo_baseline \
    --task-id adip --n-cold-start 100 --iterations 500 --seed 42
```

---

## Shared Infrastructure

### LLM Client (`shared/llm_client.py`)

```python
from shared.llm_client import create_llm_client

# Factory function with auto-detection
client = create_llm_client("qwen", backend="vllm")

# Model aliases
# "qwen"   → Qwen/Qwen2.5-7B-Instruct
# "llama"  → meta-llama/Llama-3.1-8B-Instruct
# "haiku"  → claude-haiku-4-5-20251001
# "sonnet" → claude-sonnet-4-5-20250929

# Backends: vllm, anthropic, openai, deepinfra, auto
# Auto-detect: "claude" → anthropic, "gpt" → openai, else → vllm
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

## RieLBO Subspace BO

### Design Pattern

**Problem**: GP in 256D with ~100 points overfits (correlation=1.0, no generalization).

**Solution**: Orthonormal projection S^(D-1) → S^(d-1) with d=16:
- QR decomposition: `A, _ = torch.linalg.qr(torch.randn(D, d))`
- Project: `v = normalize(u @ A)`
- Lift: `u = normalize(v @ A.T)`
- Magnitude: Use `mean_norm` from training embeddings

**Key files**: `rielbo/subspace_bo.py`, `rielbo/run_guacamol_subspace.py`

### CLI Options

**Kernel options** (`--kernel`):
- `arccosine` (default): k(x,y) = 1 - arccos(x·y)/π (no lengthscale, natural for spherical data)
- `matern`: Matern-5/2 kernel
- `hvarfner`: BoTorch defaults (RBF + LogNormal + ARD + Standardize). Remaps S^(d-1) from [-1,1] to [0,1]

**Acquisition Functions** (`--acqf`):
- `ts` (default): Thompson Sampling
- `ei`: Expected Improvement
- `ucb`: Upper Confidence Bound (with `--ucb-beta`)

### Standard GuacaMol Test Configuration

**Always use this setup for benchmarking:**
- **Cold start**: 100 molecules
- **Iterations**: 500
- **Seeds**: 42, 43, 44, 45, 46 (5 runs)
- **Tasks**: adip, med2 only

**IMPORTANT: Do NOT benchmark on pdop — it is solved. Focus experiments on adip and med2.**

```bash
# Benchmark Subspace BO
for task in adip med2; do
  for seed in 42 43 44 45 46; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace \
      --task-id $task --subspace-dim 16 --acqf ts --trust-region 0.8 \
      --n-cold-start 100 --iterations 500 --seed $seed
  done
done

# Benchmark TuRBO baseline
for task in adip med2; do
  for seed in 42 43 44 45 46; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.turbo_baseline \
      --task-id $task --n-cold-start 100 --iterations 500 --seed $seed
  done
done
```

### Benchmark Results (2026-02-05)

| Task | Cold Start | Subspace v2 (30 seeds) | Subspace v3 (10 seeds) | Vanilla BO (Hvarfner) | TuRBO | Best |
|------|------------|------------------------|------------------------|-----------------------|-------|------|
| adip | 0.4910 | **0.5475 ± 0.018** | 0.5465 ± 0.030 | 0.5022 (20 iter test) | 0.5044 ± 0.003 | **+11.5%** |
| med2 | 0.1856 | 0.1859 ± 0.002 | 0.1856 ± 0.000 | - | - | +0.0%* |

*Med2: only 0.6% of 20K molecules beat cold start best. Score range [0.02, 0.19] is extremely narrow.

**Key findings**:
- Subspace BO (S^15) consistently outperforms TuRBO (R^256)
- Vanilla BO (256D Hvarfner): ~25x slower due to 256D ARD fitting, marginal improvement
- SAASBO: ~25s/iter due to MCMC — impractical
- PCA/Active Subspace: linear projections lose nonlinear VAE structure, no improvement over random
- PLS BO: supervised projection nearly identical to random Subspace BO (0.5576 vs 0.5582)
- Intrinsic dimensionality: TwoNN=16.8, DANCo=11.3, FisherS=18.9 → validates d=16

### Vanilla BO (Hvarfner)

**Full 256D GP with BoTorch default Hvarfner priors** — no subspace projection.
- `rielbo/vanilla_bo.py` / `rielbo/run_guacamol_vanilla.py`
- **CRITICAL**: Must use [0,1]^D min-max normalization, NOT z-score. Z-score causes GP degeneration (pairwise distances ~16 vs median lengthscale 65.8 → singular kernel)
- ~33s/iteration vs ~1.3s for Subspace BO

### Prompt Optimization Benchmark (2026-02-05)

100 prompts evaluated, test split (1319 examples), Qwen2.5-7B:

| Method | Best Score | Prompts | Notes |
|--------|-----------|---------|-------|
| **ProTeGi** | **88.9%** | 100 | Sonnet meta-model |
| OPRO | 81.9% | 100 | Sonnet meta-model |
| GEPA | in progress | - | BatchingVLLMWrapper for ~100x speedup |

### GEPA Batching (Performance Fix)

GEPA's default adapter calls `task_lm(messages)` sequentially per example.
`BatchingVLLMWrapper` in `gepa_gsm8k/run.py` prefetches ALL trainset responses
in one `generate_batch()` call when a new prompt is detected. ~100-200x speedup.
- Use `--no-batch` to disable (for debugging)
- GEPA message format: `[{system: candidate_prompt}, {user: data["input"]}]`

---

## Coding Standards

### Logging Generated Prompts
**NEVER truncate prompts in log output or print statements.** Always output the full prompt text.

```python
# BAD
logger.info(f"Prompt: {result['prompt'][:200]}...")

# GOOD
logger.info(f"Prompt:\n{result['prompt']}")
```

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
- **Always use `--backend vllm`** for task model unless explicitly told otherwise
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
