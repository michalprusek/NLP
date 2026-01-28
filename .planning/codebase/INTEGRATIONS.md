# External Integrations

**Analysis Date:** 2026-01-28

## APIs & External Services

**LLM Inference Backends:**
- **vLLM** (Local CUDA)
  - What it's used for: Fast batch inference for task evaluation and meta-optimization
  - SDK/Client: `vllm` package
  - Models supported: Qwen, Llama, SaulLM (HuggingFace Instruct models)
  - Auto-detection: Preferred for models not starting with "gpt-" or "google/"
  - Implementation: `src/llm_client.py` - `VLLMClient` class

- **OpenAI API (GPT)**
  - What it's used for: Meta-optimization (Claude alternative) or task evaluation with GPT models
  - SDK/Client: `openai>=2.5.0` package
  - Models supported: gpt-3.5-turbo (alias "gpt-3.5"), gpt-4 (auto-detected by "gpt" prefix)
  - Auth: `OPENAI_API_KEY` environment variable
  - Implementation: `src/llm_client.py` - `OpenAIClient` class

- **Anthropic API (Claude)**
  - What it's used for: Meta-optimization (textual gradients, reflection, prompt editing)
  - SDK/Client: `anthropic>=0.71.0` package
  - Models supported: "haiku" → claude-haiku-4-5-20251001, "sonnet" → claude-sonnet-4-5-20251022
  - Auth: `ANTHROPIC_API_KEY` environment variable
  - Implementation: Integrated via OpenAI-compatible wrapper in `src/llm_client.py`
  - Usage: `--model sonnet` or `--meta-model haiku` in run scripts

- **DeepInfra API (Gemma, OpenAI-compatible)**
  - What it's used for: Alternative LLM provider for Gemma and other models
  - SDK/Client: `openai>=2.5.0` package with custom base_url
  - Base URL: `https://api.deepinfra.com/v1/openai`
  - Models supported: google/gemma-3-4b-it (auto-detected by "google/" prefix)
  - Auth: `DEEPINFRA_API_KEY` environment variable
  - Implementation: `src/llm_client.py` - `DeepInfraClient` class

**Embedding Services:**
- **HuggingFace Sentence Transformers (GTR)**
  - What it's used for: 768D embedding generation for EcoFlow-BO encoder
  - SDK/Client: `sentence-transformers>=2.3.0` package
  - Model: GTR (General Text Representation) - loaded locally
  - Implementation: `ecoflow_bo/encoder.py` - `MatryoshkaEncoder` uses GTR embeddings
  - Data source: Pre-computed embeddings in `datasets/gtr_embeddings_full.pt` (4.7GB)

- **Sonar API (sonar-space)**
  - What it's used for: Text embedding and reranking
  - SDK/Client: `sonar-space>=0.5.0` package
  - Implementation: Available as integration but not actively used in current optimization runs
  - Potential use: Prompt representation for semantic search

## Data Storage

**Databases:**
- Not applicable - No persistent database used

**File Storage:**
- **Local filesystem only**
  - Training/test data: `datasets/gsm8k/` (Arrow format, HuggingFace datasets)
  - Additional data: `datasets/hbbops/` (JSON, TXT instruction sets)
  - Embeddings: `datasets/gtr_embeddings_full.pt` (PyTorch tensor, 4.7GB)
  - Text corpus: `datasets/combined_texts.json` (1.7GB JSON)
  - Optimization results: `results/` (JSON, TXT outputs, timestamps)
  - EcoFlow checkpoints: `results/ecoflow_checkpoints/` (PyTorch `.pt` files)
  - Detailed eval JSONs: `results/eval_*_YYYYMMDD_HHMMSS/` (optional, see `--save-eval-json`)

**Caching:**
- PyTorch in-memory caching during runs
- vLLM prefix caching (`enable_prefix_caching=True` in `src/llm_client.py:77`)
- HuggingFace model cache: Standard `~/.cache/huggingface/` location

## Authentication & Identity

**Auth Providers:**
- **Environment variables** (custom implementation)
  - No OAuth or session-based auth
  - Direct API key passing: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPINFRA_API_KEY`
  - Loaded via `python-dotenv` in `src/llm_client.py:7`
  - Implementation: `load_dotenv()` reads `.env` file at startup

**Auth Pattern:**
- Each API client validates key presence at initialization
- OpenAI/DeepInfra: Keys passed to OpenAI() client initialization
- Anthropic: Key managed by anthropic package (via environment or passed)
- No token refresh or expiration handling (delegated to SDK)

## Monitoring & Observability

**Error Tracking:**
- Not detected - No external error tracking service (Sentry, Rollbar, etc.)
- Local error handling via try-catch with console logging

**Logs:**
- Console output (stdout/stderr)
- File logging (optional):
  - tmux sessions with `tee` for long-running processes
  - Pattern: `tmux new-session -d -s <name> "... 2>&1 | tee results/<descriptive>_$(date +%Y%m%d_%H%M%S).log"`
  - TensorBoard logs: `tensorboard>=2.20.0` for training visualization

## CI/CD & Deployment

**Hosting:**
- Not applicable - Framework/library, not a deployed service
- Can be used locally or on custom infrastructure

**CI Pipeline:**
- Pytest configuration exists: `.pytest_cache/` directory, `pytest>=9.0.2` in dev dependencies
- No external CI service detected (no GitHub Actions, GitLab CI, etc. configs)
- Tests run locally via: `pytest` (configuration in `pyproject.toml`)

## Environment Configuration

**Required env vars:**
- `ANTHROPIC_API_KEY` - Claude API access (optional if using vLLM only)
- `OPENAI_API_KEY` - OpenAI API access (optional if using vLLM only)
- `DEEPINFRA_API_KEY` - DeepInfra API access (optional if using vLLM only)
- `CUDA_VISIBLE_DEVICES` - GPU selection (set at runtime, e.g., `--gpu-ids 0,1`)

**Secrets location:**
- `.env` file (gitignored, not committed)
- Template: `.env.example` referenced in README (not found in current repo)
- Keys are loaded at client initialization in `src/llm_client.py`

## Webhooks & Callbacks

**Incoming:**
- Not applicable - This is a framework, not a web service

**Outgoing:**
- Not applicable - No webhook callbacks to external services

## Model Aliases & Routing

**Automatic Backend Detection** (`src/llm_client.py:create_llm_client()`):
```
- "gpt" in model_name → OpenAI backend
- model_name.startswith("google/") → DeepInfra backend
- Otherwise → vLLM backend (default)
```

**Model Aliases** (defined in run scripts):
- `qwen` → `Qwen/Qwen2.5-7B-Instruct`
- `qwen-3b` → `Qwen/Qwen2.5-3B-Instruct`
- `qwen-7b` → `Qwen/Qwen2.5-7B-Instruct`
- `llama` → `meta-llama/Llama-3.1-8B-Instruct`
- `saul` → `Equall/Saul-7B-Instruct-v1`
- `haiku` → `claude-haiku-4-5-20251001`
- `sonnet` → `claude-sonnet-4-5-20251022`
- `gpt-3.5` → `gpt-3.5-turbo`
- `gemma-3-4b` → `google/gemma-3-4b-it`

## Dataset Loading

**GSM8K Math Reasoning:**
- Source: HuggingFace datasets (pre-downloaded to `datasets/gsm8k/`)
- Format: Arrow files (`.arrow` binary format)
- Split: `train/` (7K examples), `test/` (1.3K examples)
- Loading: `datasets.load_from_disk()` in `src/gsm8k_evaluator.py:134`
- Answer extraction: Custom logic matching lm-evaluation-harness standard

**HbBoPs (Hyperband Baseline of Prompts):**
- Location: `datasets/hbbops/`
- Files: `ape_instructions_1000.json`, `instructions_*.txt`, `examples_*.txt`, `full_grid_combined.jsonl`
- Purpose: APE instruction dataset and pre-evaluated prompt results

**GTR Embeddings:**
- Pre-computed: `datasets/gtr_embeddings_full.pt` (4.7GB PyTorch tensor)
- Used in: EcoFlow-BO encoder (`ecoflow_bo/encoder.py`)
- Dimension: 768D embeddings for entire prompt space

**Text Corpus:**
- `datasets/combined_texts.json` (1.7GB)
- Used by: Detail retriever for latent space visualization/retrieval

---

*Integration audit: 2026-01-28*
