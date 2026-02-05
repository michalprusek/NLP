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
| **RieLBO Subspace BO** | `rielbo/` | Spherical subspace projection S^255 → S^15 + GP |
| **TuRBO** | `rielbo/turbo_baseline.py` | Trust region BO in R^256 (baseline) |
| **Vanilla BO** | `rielbo/vanilla_bo.py` | Full 256D Hvarfner GP (baseline) |

## Quick Start

```bash
# Install
uv sync

# Prompt optimization
uv run python -m opro.run --model qwen --iterations 10
uv run python -m protegi.run --model qwen --steps 6

# Molecular optimization
CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.run_guacamol_subspace_v2 \
    --preset geodesic --task-id adip --n-cold-start 100 --iterations 500
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
