# Technology Stack

**Analysis Date:** 2026-01-28

## Languages

**Primary:**
- Python 3.10+ - All project code, optimization algorithms, neural networks, and CLI tools

## Runtime

**Environment:**
- Python 3.10-3.13 (specified in `pyproject.toml` as `>=3.10,<3.14`)
- venv located in `.venv/` directory

**Package Manager:**
- `uv` - Ultra-fast Python package installer (primary method: `uv sync`, `uv run`)
- `pip` - Traditional fallback (venv located at `/home/prusek/NLP/.venv/`)
- Lockfile: `uv.lock` present (1.25MB)

## Frameworks

**Core ML/Training:**
- PyTorch 2.0+ - Neural network implementations, distributed training
- Transformers 4.57.1+ - HuggingFace models (Qwen, Llama, SaulLM), tokenizers
- vLLM 0.10.0-0.11.0 - Fast LLM inference with tensor parallelism
- Accelerate 1.10.1+ - Multi-GPU training support (DDP)

**Optimization & BO:**
- BoTorch 0.14.0+ - Bayesian optimization primitives (UCB acquisition)
- GPyTorch 1.14.2+ - Gaussian Process models
- Optuna 4.1.0+ - Hyperparameter optimization framework
- torch-cfm 1.0.4+ - Conditional Flow Matching models

**ML Components:**
- Sentence Transformers 2.3.0+ - GTR embedding model (768D) for encoder
- PEFT 0.14.0+ - LoRA adapters for prompt optimization
- Vec2Text 0.0.13 - Text inversion from embeddings
- TorchMetrics 1.0.0+ - BLEU/ROUGE metrics for evaluation
- torch-optimizer 0.3.0+ - Additional optimizer implementations

**Data & Datasets:**
- Datasets 4.2.0+ - HuggingFace datasets library (load_from_disk for GSM8K/HbBoPs)
- Pandas 2.0.0+ - Data manipulation
- PyArrow - Dataset serialization format (Arrow files in `datasets/`)

**XGBoost Variants:**
- CatBoost 1.2.0+ - Gradient boosting
- NGBoost 0.5.1+ - Natural gradient boosting
- scikit-learn 1.3.0+ - ML utilities and metrics

**Deduplication & NLP:**
- DataSketch 1.6.0+ - MinHash for duplicate detection
- fasttext-wheel 0.9.2 - Language detection

**Monitoring & Utilities:**
- TensorBoard 2.20.0+ - Training visualization
- UMAP 0.5.5+ - Dimensionality reduction visualization
- Optuna 4.1.0+ - Optimization framework
- tqdm 4.65.0+ - Progress bars
- python-dotenv 1.0.0+ - Environment variable loading

**API Clients:**
- Anthropic 0.71.0+ - Claude API (haiku, sonnet models)
- OpenAI 2.5.0+ - GPT models
- sonar-space 0.5.0 - Text embedding/reranking (external service integration)

**Notebooks:**
- Jupyter 1.1.1+ - Interactive notebooks
- IPython/IPyKernel 6.31.0+ - Kernel for Jupyter
- Bokeh 3.4.3+ - Interactive visualizations
- Plotly 5.18.0+ - Data visualization

## Key Dependencies

**Critical:**
- `torch>=2.0.0` - Core neural network engine (trainable models require this)
- `transformers>=4.57.1` - Model loading and inference (all LLMs)
- `vllm>=0.10.0,<0.11.0` - Fast batch inference (required for OPRO/ProTeGi/GEPA optimization runs)
- `anthropic>=0.71.0` - Claude API calls (meta-optimization with Claude models)
- `openai>=2.5.0` - OpenAI API calls (GPT models as alternative)
- `datasets>=4.2.0` - GSM8K and HbBoPs dataset loading

**Infrastructure:**
- `botorch>=0.14.0` - Bayesian optimization (EcoFlow-BO uses UCB acquisition)
- `gpytorch>=1.14.2` - GP backends for BO
- `torch-cfm>=1.0.4` - Conditional flow matching for EcoFlow-BO velocity networks
- `sentence-transformers>=2.3.0` - GTR embeddings (768D encoding)

## Configuration

**Environment:**
- `.env` file - API keys configuration (ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPINFRA_API_KEY)
- `.env.example` - Template for `.env` (not present in current scan, but referenced in README)
- Environment variables:
  - `ANTHROPIC_API_KEY` - Claude API authentication
  - `OPENAI_API_KEY` - OpenAI API authentication
  - `DEEPINFRA_API_KEY` - DeepInfra API authentication
  - `CUDA_VISIBLE_DEVICES` - GPU selection (set per experiment in `run_opro.py`, `run_protegi.py`)
  - `VLLM_USE_V1` - Set to "0" in `src/llm_client.py` for vLLM v0.10 compat
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - vLLM memory optimization

**Build:**
- `pyproject.toml` - Python project configuration and dependency specification
- `uv.lock` - Locked dependency versions for reproducibility

## Platform Requirements

**Development:**
- Python 3.10+ with pip/uv
- CUDA GPU (mandatory for vLLM backend; tested on 2x NVIDIA L40S 48GB)
- 48GB+ VRAM per GPU recommended (ecoflow_bo uses DDP with `torchrun --nproc_per_node=2`)
- NVIDIA CUDA Toolkit (vLLM requires CUDA for local inference)

**Production:**
- Deployment target: Linux (tested on 4.18.0 kernel)
- Can run with local vLLM (CUDA) or remote APIs (OpenAI, DeepInfra, Anthropic)
- Multi-GPU support via PyTorch DDP (vLLM tensor parallelism on `--tensor-parallel-size`)

---

*Stack analysis: 2026-01-28*
