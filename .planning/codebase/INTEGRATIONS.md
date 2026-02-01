# External Integrations

**Analysis Date:** 2026-01-31

## APIs & External Services

**LLM Backends:**
- **Claude (Anthropic)** - Meta-optimization and task evaluation
  - SDK/Client: `anthropic>=0.71.0`
  - Auth: `ANTHROPIC_API_KEY`
  - Models supported: claude-haiku-4-5-20251001, claude-sonnet-4-5-20251022
  - Implementation: `shared/llm_client.py::AnthropicClient` (if available) or generic OpenAI client

- **OpenAI** - GPT-3.5, GPT-4 task evaluation
  - SDK/Client: `openai>=2.5.0`
  - Auth: `OPENAI_API_KEY`
  - Implementation: `shared/llm_client.py::OpenAIClient`
  - Endpoint: Chat completions API

- **DeepInfra** - Gemma, Llama inference (OpenAI-compatible)
  - SDK/Client: `openai>=2.5.0` (reused client)
  - Auth: `DEEPINFRA_API_KEY`
  - Base URL: `https://api.deepinfra.com/v1/openai`
  - Implementation: `shared/llm_client.py::DeepInfraClient`

- **HuggingFace Models (Local/vLLM)** - Qwen, Llama, SaulLM
  - SDK/Client: `transformers>=4.57.1`, `vllm>=0.10.0,<0.11.0`
  - Auth: None (uses HuggingFace token from `~/.cache/huggingface/`)
  - Implementation: `shared/llm_client.py::VLLMClient`
  - Models: Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct, Equall/Saul-7B-Instruct-v1

**Embedding Models:**
- **SONAR (Meta)** - 1024D embedding space for optimization
  - SDK/Client: `sonar-space>=0.5.0`, `sentence-transformers>=2.3.0`
  - Auth: None (HuggingFace token)
  - Models: text_sonar_basic_encoder (embedding), text_sonar_basic_decoder (decoding)
  - Implementation: `ecoflow/decoder.py::SonarDecoder`, `ecoflow/data.py`
  - Pipeline: `sonar.inference_pipelines.text.EmbeddingToTextModelPipeline`

## Data Storage

**Datasets:**
- **HuggingFace Datasets Hub** - GSM8K loading
  - Client: `datasets>=4.2.0` with `load_from_disk()`
  - Location: `datasets/gsm8k/` (local arrow format)
  - Format: Arrow (.arrow files with metadata)
  - Reference: `shared/gsm8k_evaluator.py::GSM8KEvaluator`

**Local File Storage:**
- **PyTorch Checkpoints** - Model weights and embeddings
  - Location: `datasets/sonar_embeddings.pt` (1.5M pre-computed embeddings)
  - Location: `ecoflow/results/` (flow model checkpoints)
  - Format: PyTorch `.pt` (torch.save/load)

- **Results Output** - Optimization trajectories and prompts
  - Location: `{opro,protegi,gepa,ecoflow,nfbo}/results/`
  - Format: JSON files with scored prompts, timestamps

**Caching:**
- HuggingFace transformers cache: `~/.cache/huggingface/hub/`
- vLLM KV-cache: On-GPU (managed by vLLM)
- No explicit caching service (Redis, Memcached) detected

## Authentication & Identity

**Auth Provider:**
- Custom API key management via environment variables
- Implementation: `shared/llm_client.py` loads keys from `.env` via `python-dotenv>=1.0.0`
- No OAuth, JWT, or service-to-service auth detected

**Per-Backend Auth:**
- **vLLM**: HuggingFace token (auto-loaded from `~/.cache/`)
- **OpenAI**: `OPENAI_API_KEY` env var
- **DeepInfra**: `DEEPINFRA_API_KEY` env var
- **Anthropic**: `ANTHROPIC_API_KEY` env var

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, DataDog, etc.)
- Errors logged to stdout/stderr via print statements and logging module

**Logging:**
- **Built-in Python logging** - Used in `ecoflow/train_flow.py`
  - Format: `%(asctime)s - %(levelname)s - %(message)s`
  - Level: INFO by default
- **stdout/print statements** - Throughout `shared/llm_client.py` and optimization modules
- **TQDM progress bars** - For iteration progress visualization

**TensorBoard:**
- Optional integration via `tensorboard>=2.20.0`
- Used in soft-prompt VAE training for loss tracking

## CI/CD & Deployment

**Hosting:**
- Local GPU workstations (NVIDIA L40S)
- No detected cloud deployment (AWS, GCP, Azure)

**CI Pipeline:**
- Not detected (no GitHub Actions, GitLab CI, Jenkins)

**Testing:**
- Local pytest execution: `uv run pytest tests/ -x -q`
- No continuous integration service configured

## Environment Configuration

**Required Environment Variables:**
- `ANTHROPIC_API_KEY` - Claude model access (CRITICAL if using haiku/sonnet)
- `DEEPINFRA_API_KEY` - Gemma/Llama access (optional, for alternative models)
- `OPENAI_API_KEY` - OpenAI GPT access (optional)
- `CUDA_VISIBLE_DEVICES` - GPU selection (e.g., "0,1" for 2 GPUs)

**Optional Configuration:**
- `VLLM_USE_V1` - Should be "0" (v0.10 API)
- `PYTORCH_CUDA_ALLOC_CONF` - Should be "expandable_segments:True" for flexible memory
- `VLLM_TARGET_DEVICE` - Should be "cuda"

**Secrets Location:**
- `.env` file in repository root (VERSION CONTROLLED with dummy keys!)
- **WARNING**: Actual API keys are committed to `.env` (security risk for production)
- Proper usage: Copy `.env.example` to `.env` and fill in actual keys locally

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected (no reporting to external services)

## Model Registry & Package Repositories

**HuggingFace Hub:**
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- Equall/Saul-7B-Instruct-v1
- google/gemma-3-4b-it (via DeepInfra)
- Meta's SONAR encoder/decoder models

**PyPI:**
- All dependencies installed from PyPI via uv/pip

## Network & Connectivity

**API Endpoints:**
- `api.anthropic.com` - Anthropic API
- `api.openai.com` - OpenAI API
- `api.deepinfra.com/v1/openai` - DeepInfra API
- `huggingface.co` - Model downloading

**Local Inference:**
- vLLM runs locally on GPU (no external network calls)
- SONAR encoder/decoder run locally on GPU

## Distributed Computing

**Multi-GPU Training:**
- vLLM: `tensor_parallel_size` parameter for model parallelism
- PyTorch: `accelerate>=1.10.1` for distributed training setup
- Training command: `torchrun --nproc_per_node=2` for 2-GPU execution
- No inter-node communication (single machine only)

---

*Integration audit: 2026-01-31*
