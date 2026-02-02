# Technology Stack

**Analysis Date:** 2026-01-31

## Languages

**Primary:**
- Python 3.10+ - All core implementation, optimization algorithms, and utilities

**Secondary:**
- Bash - Shell scripts for running optimization pipelines (e.g., `run_all_optimizers_50k.sh`)

## Runtime

**Environment:**
- Python 3.10 to 3.13 (per pyproject.toml: `requires-python = ">=3.10,<3.14"`)
- CUDA GPU (2x NVIDIA L40S with 48GB VRAM each, 96GB total)
- Linux (running on 4.18.0-553.92.1.el8_10.x86_64)

**Package Manager:**
- uv (ultra-fast Python package installer/resolver)
- Lockfile: `uv.lock` (present and committed)

## Frameworks

**Deep Learning:**
- PyTorch 2.0+ - Core tensor operations, neural networks, distributed training
- TorchDyn 1.0.6+ - ODE solvers for flow matching
- Transformers 4.57.1+ - LLM tokenizers and model loading
- vLLM 0.10.0-0.11.0 - Fast local LLM inference with tensor parallelism
- PEFT 0.14.0+ - LoRA adapter fine-tuning for soft-prompt VAE
- TorchCFM 1.0.4+ - Conditional Flow Matching implementation

**Optimization & Bayesian Methods:**
- BoTorch 0.14.0+ - Bayesian optimization acquisition functions and surrogate models
- GPyTorch 1.14.2+ - Gaussian Process kernels and likelihoods
- Optuna 4.1.0+ - Hyperparameter optimization framework
- NGBoost 0.5.1+ - Natural Gradient Boosting with uncertainty
- CatBoost 1.2.0+ - Gradient boosting on CPU/GPU
- Torch-Optimizer 0.3.0+ - AdamW and extended optimizers

**NLP & Embeddings:**
- Sentence-Transformers 2.3.0+ - SONAR embedding model loading
- SonarSpace 0.5.0+ - SONAR embedding space utilities and decoder
- Vec2Text 0.0.13+ - Embedding inversion techniques
- Transformers-based FastText 0.9.2 - Language detection for soft-prompt VAE
- TorchMetrics 1.0.0+ - BLEU, ROUGE metrics for soft-prompt VAE evaluation

**Data Processing:**
- Datasets 4.2.0+ - HuggingFace datasets loading (GSM8K, etc.)
- Pandas 2.0.0+ - Data manipulation and analysis
- NumPy 1.21.0+ - Numerical computing
- UMAP 0.5.5+ - Dimensionality reduction for embedding visualization
- Datasketch 1.6.0+ - MinHash deduplication for soft-prompt VAE

**Visualization & Monitoring:**
- Plotly 5.18.0+ - Interactive plotting for optimization trajectories
- Bokeh 3.4.3+ - Web-based visualization
- Seaborn 0.13.2+ - Statistical data visualization
- TensorBoard 2.20.0+ - Training monitoring for soft-prompt VAE
- TorchMetrics 1.0.0+ - Built-in metric computation

**Testing:**
- Pytest 9.0.2+ - Unit and integration testing framework

## Key Dependencies

**Critical:**
- torch 2.0+ - Foundational deep learning framework; used in all optimization methods
- vllm 0.10.0+ - Enables fast local inference for Qwen/Llama models; critical for OPRO/ProTeGi/GEPA
- transformers 4.57.1+ - Provides AutoTokenizer, AutoModel for HuggingFace models; used in all LLM client backends
- anthropic 0.71.0+ - Claude API client for claude-haiku and claude-sonnet backends

**Optimization-Specific:**
- botorch 0.14.0+ - Provides SingleTaskGP, Matern kernels for RieLBO GP surrogate
- gpytorch 1.14.2+ - Underlying GP implementation used by botorch
- torchcfm 1.0.4+ - ConditionalFlowMatcher for OT-CFM training in RieLBO
- uncertainty-toolbox 0.1.1+ - Uncertainty quantification tools for RieLBO

**Embedding & Decoding:**
- sonar-space 0.5.0+ - SONAR embedding utilities and text_sonar_basic_decoder
- sentence-transformers 2.3.0+ - For loading SONAR models
- vec2text 0.0.13+ - Embedding inversion fallback techniques

## Configuration

**Environment Variables:**
- `ANTHROPIC_API_KEY` - For Claude model backends (haiku, sonnet)
- `OPENAI_API_KEY` - Optional, for OpenAI GPT models via OpenAI backend
- `DEEPINFRA_API_KEY` - For DeepInfra API (Gemma, Llama alternatives)
- `VLLM_USE_V1` - Set to "0" (disabled), uses vLLM v0.10 API
- `PYTORCH_CUDA_ALLOC_CONF` - Set to "expandable_segments:True" for flexible GPU memory
- `VLLM_TARGET_DEVICE` - Set to "cuda" for GPU inference
- `CUDA_VISIBLE_DEVICES` - GPU selection for distributed training

**Configuration Files:**
- `.env` - API key storage (ANTHROPIC_API_KEY, DEEPINFRA_API_KEY)
- `pyproject.toml` - Dependency specifications, metadata

**Build Configuration:**
- None detected (pure Python package; no compiled extensions required)

## Data Files

**Input Datasets:**
- `datasets/gsm8k/` - GSM8K math reasoning dataset (train/test splits)
- `datasets/sonar_embeddings.pt` - Pre-computed 1.5M SONAR embeddings (1024D)
- `datasets/hbbops/` - Hyperband baseline prompts

## Platform Requirements

**Development:**
- NVIDIA CUDA-capable GPU (2x L40S recommended)
- Python 3.10+
- ~50GB disk for datasets and model checkpoints
- vLLM requires CUDA; Claude API requires internet access

**Production:**
- Same as development; optimized for 2x GPU setup
- Supports tmux for long-running processes
- Single GPU fallback supported with `tensor_parallel_size=1`

---

*Stack analysis: 2026-01-31*
