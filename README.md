# NLP: Prompt & Molecular Optimization Research

Research framework for sample-efficient optimization in high-dimensional latent spaces, targeting NeurIPS submission.

## Methods

### Prompt Optimization (GSM8K)

| Method | Module | Description |
|--------|--------|-------------|
| **OPRO** | `opro/` | Optimization by PROmpting — LLM as meta-optimizer |
| **ProTeGi** | `protegi/` | Textual gradients + beam search |
| **GEPA** | `gepa_gsm8k/` | Genetic-Pareto prompt adaptation (with batching wrapper) |

### Molecular Optimization (GuacaMol)

| Method | Module | Description |
|--------|--------|-------------|
| **Subspace BO v1** | `rielbo/subspace_bo.py` | Spherical subspace projection S^255 → S^15 + GP |
| **Subspace BO v2** | `rielbo/subspace_bo_v2.py` | Geodesic/novelty presets (recommended) |
| **Subspace BO v3** | `rielbo/subspace_bo_v3.py` | Multi-projection rotation |
| **Subspace BO v4** | `rielbo/subspace_bo_v4.py` | Novelty-weighted acquisition |
| **Subspace BO v5** | `rielbo/subspace_bo_v5.py` | Latest variant |
| **TuRBO** | `rielbo/turbo_baseline.py` | Trust region BO in R^256 (baseline) |
| **Vanilla BO** | `rielbo/vanilla_bo.py` | Full 256D Hvarfner GP (baseline) |

## Key Results

### Molecular Optimization (GuacaMol adip)

| Method | Best Score | Seeds | vs Cold Start |
|--------|-----------|-------|---------------|
| Cold Start | 0.4910 | — | — |
| **Subspace BO v2** | **0.5475 ± 0.018** | 30 | **+11.5%** |
| Subspace BO v3 | 0.5465 ± 0.030 | 10 | +11.3% |
| Vanilla BO (256D) | 0.5022 | 20 iter | +2.3% |
| TuRBO (R^256) | 0.5044 ± 0.003 | 5 | +2.7% |

Intrinsic dimensionality analysis validates d=16 subspace: TwoNN=16.8, DANCo=11.3, FisherS=18.9.

### Prompt Optimization (GSM8K, 100 prompts, test split)

| Method | Best Accuracy |
|--------|--------------|
| **ProTeGi** | **88.9%** |
| OPRO | 81.9% |

## Quick Start

```bash
# Install
uv sync

# Prompt optimization
uv run python -m opro.run --model qwen --iterations 10
uv run python -m protegi.run --model qwen --steps 6

# Molecular optimization (Subspace BO v2, recommended)
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
    --preset geodesic --task-id adip --n-cold-start 100 --iterations 500

# TuRBO baseline
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.turbo_baseline \
    --task-id adip --n-cold-start 100 --iterations 500 --seed 42

# Run full benchmark suite
CUDA_VISIBLE_DEVICES=0 bash rielbo/benchmark/run_v2_benchmark.sh
```

## Requirements

- Python 3.10+
- CUDA GPU (for vLLM backend and BO)
- `ANTHROPIC_API_KEY` (for Claude meta-models)

## References

- ProTeGi: [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
- OPRO: [arXiv:2309.03409](https://arxiv.org/abs/2309.03409)
- TuRBO: [arXiv:1910.01739](https://arxiv.org/abs/1910.01739)

See `CLAUDE.md` for detailed documentation, benchmark results, and development guidelines.
