# Architecture

**Analysis Date:** 2026-01-28

## Pattern Overview

**Overall:** Multi-algorithm prompt optimization framework with three distinct optimization approaches (OPRO, ProTeGi, GEPA) and an emerging embedding-space Bayesian optimization system (EcoFlow-BO).

**Key Characteristics:**
- Modular LLM client abstraction supporting multiple backends (vLLM, OpenAI, DeepInfra)
- Algorithm-agnostic evaluator framework for GSM8K mathematics reasoning
- Separate task and meta-optimizer models (can use different models for optimization)
- Embedding-space latent optimization (EcoFlow-BO) with flow matching
- Fixed evaluation sets for reproducible prompt scoring

## Layers

**LLM Client Layer:**
- Purpose: Abstract away backend differences (local vLLM, OpenAI, DeepInfra APIs)
- Location: `src/llm_client.py`
- Contains: `LLMClient` ABC, `VLLMClient`, `OpenAIClient`, `DeepInfraClient`, factory function `create_llm_client()`
- Depends on: vLLM or OpenAI SDK, Hugging Face transformers
- Used by: All optimization algorithms (OPRO, ProTeGi, GEPA, EcoFlow-BO)

**Evaluation Layer:**
- Purpose: Assess prompt quality on GSM8K dataset with lm-eval-harness standard metrics
- Location: `src/gsm8k_evaluator.py`
- Contains: Answer extraction (multiple pattern-matching strategies), ground truth comparison, batch evaluation
- Depends on: Hugging Face datasets (Arrow format)
- Used by: OPRO, ProTeGi, GEPA (for scoring), EcoFlow-BO (for objective)

**Prompt Optimization Algorithms:**
- Purpose: Generate and refine instruction prompts via different optimization strategies
- Location: `src/opro.py`, `src/protegi.py`, `src/gepa.py`
- Contains:
  - `OPRO`: Meta-optimizer with bucketed scoring and top-K memory
  - `ProTeGi`: Textual gradients + beam search with UCB bandit
  - `GEPA`: Pareto selection + reflection-based mutation
- Depends on: LLM Client, Evaluator, prompt templates in `src/prompts/gsm8k/`
- Used by: Entry points `run_opro.py`, `run_protegi.py`, `run_gepa.py`

**Embedding-Space Optimization (EcoFlow-BO):**
- Purpose: Bayesian optimization in latent space for prompt embeddings
- Location: `ecoflow_bo/` directory
- Key Components:
  - Encoder (`encoder.py`): Matryoshka VAE (768D → 16D core + 32D detail residual latent)
  - Velocity Network (`velocity_network.py`): Diffusion Transformer for flow matching
  - Decoder (`cfm_decoder.py`, `perceiver_decoder.py`): Rectified flow or Perceiver-based reconstruction
  - GP (`latent_gp.py`): Coarse-to-fine Gaussian Process on z_core (16D)
  - Acquisition (`density_acquisition.py`): Density-aware candidate generation
  - Cycle Consistency (`cycle_consistency.py`): Hallucination detection via encode-decode roundtrip
  - Detail Retriever (`detail_retriever.py`): Nearest-neighbor lookup of z_detail from training set
  - Main Optimizer (`optimizer.py`): Orchestrates all components
- Depends on: torch, gpytorch, botorch, torchcfm
- Used by: Future integration with prompt optimization

**Entry Points:**
- Location: `run_opro.py`, `run_protegi.py`, `run_gepa.py` (algorithms), scripts in root
- Purpose: Command-line interfaces with model/backend selection and experiment configuration
- Responsibilities: Argument parsing, model aliasing, training vs test evaluation, results serialization

## Data Flow

**OPRO Optimization Flow:**

1. Load task LLM and optionally separate meta-optimizer LLM via `create_llm_client()`
2. Initialize `GSM8KEvaluator` with dataset split (train or test)
3. Create fixed evaluation set (same minibatch for all prompt candidates)
4. For each iteration:
   - Generate N candidate prompts via meta-LLM with example few-shots
   - Evaluate each on fixed minibatch
   - Bucket scores into 20 buckets (from OPRO paper)
   - Keep top-K prompts in memory
   - Pass top-K + random examples to meta-LLM for next iteration
5. Test best prompt on full test set
6. Save results as JSON with history

**ProTeGi Optimization Flow:**

1. Initialize beam with diverse candidate prompts
2. For each iteration:
   - Evaluate beam on fixed minibatch
   - Use UCB bandit to select which prompts to edit
   - For selected prompts:
     - Generate "textual gradients" (LLM critiques of failures)
     - Use gradient LLM to propose edits
     - Apply Monte Carlo paraphrasing for exploration
   - Evaluate new candidates
   - Update beam (keep best by UCB score)
3. Test best prompt on full test set

**GEPA Optimization Flow:**

1. Initialize population with seed prompts
2. For each iteration:
   - Evaluate all candidates on minibatch
   - Compute Pareto frontier (non-dominated candidates)
   - For each frontier member:
     - Reflection LLM analyzes failure patterns
     - Mutation LLM applies targeted improvements
   - Evaluate mutations
   - Update population (merge Pareto fronts)
3. Test best prompt on full test set

**EcoFlow-BO Optimization Flow:**

1. Pre-encode initial prompts to embeddings (768D GTR)
2. Run Matryoshka encoder to get z_core (16D) + z_detail (32D)
3. Initialize detail retriever with training set z_details
4. For each BO iteration:
   - Evaluate current candidates (requires decoding back to text)
   - Update GP on z_core with scores
   - Generate acquisition candidates (density-aware UCB)
   - Check cycle consistency (encode-decode roundtrip validity)
   - Propose next z_core, retrieve z_detail from nearest neighbor
   - Decode via flow matching to get new prompt embeddings
5. Return best prompt and final embedding

**State Management:**
- OPRO: `scored_prompts` list, `fixed_eval_set`
- ProTeGi: `beam` (list of ScoredPrompt), `total_visits` counter for UCB
- GEPA: `population` list with separate scoring for speed/quality metrics
- EcoFlow-BO: `best_z_core`, `best_z_detail`, `best_embedding`, `history` dict list

## Key Abstractions

**LLMClient (ABC):**
- Purpose: Unified interface for any LLM backend
- Examples: `VLLMClient`, `OpenAIClient`, `DeepInfraClient`
- Pattern: Factory `create_llm_client()` auto-detects backend from model name
- Methods: `generate(prompt, **kwargs)`, `generate_batch(prompts, **kwargs)`

**ScoredPrompt:**
- Purpose: Immutable record of a prompt and its evaluation score
- Examples: Used in OPRO (`src/opro.py`), ProTeGi (`src/protegi.py`)
- Pattern: Dataclass with `__repr__` for truncated display (first 50 chars)

**Evaluator Interface:**
- Purpose: Standard evaluation API regardless of dataset
- Examples: `GSM8KEvaluator`
- Pattern: `evaluate_batch(outputs, indices) -> Dict[accuracy, correct, total, details]`
- Answer extraction: Multiple fallback patterns (####, \boxed{}, last number)

**CoarseToFineGP:**
- Purpose: Progressive dimension unlocking for tractable Bayesian optimization
- Pattern: Stage-based active dimensions schedule (4D → 8D → 16D)
- Advances automatically when enough training points collected

**MatryoshkaEncoder:**
- Purpose: Hierarchical embedding compression with multi-scale supervision
- Pattern: Output at multiple dimensions [2, 4, 8, 16] with weighted reconstruction loss
- Result: z_core (16D) useful at all scales, z_detail (32D) for final 48D capacity

## Entry Points

**run_opro.py:**
- Location: `/home/prusek/NLP/run_opro.py`
- Triggers: Command line via `uv run python run_opro.py [args]`
- Responsibilities:
  - Model aliasing and backend selection
  - Dataset loading
  - OPRO optimizer instantiation and execution
  - Test set evaluation
  - Results serialization (JSON + text prompt file)

**run_protegi.py:**
- Location: `/home/prusek/NLP/run_protegi.py`
- Triggers: Command line via `uv run python run_protegi.py [args]`
- Responsibilities: ProTeGi-specific optimization

**run_gepa.py:**
- Location: `/home/prusek/NLP/run_gepa.py`
- Triggers: Command line via `uv run python run_gepa.py [args]`
- Responsibilities: GEPA-specific optimization with Pareto frontier management

## Error Handling

**Strategy:** Fail-fast with informative messages; distinguish recoverable (API transients) from fatal errors.

**Patterns:**
- OpenAI/DeepInfra clients catch per-prompt failures and return `None`, not raising
- Authentication errors (401) re-raised immediately (fatal)
- Empty LLM responses logged but continue (distinguishable from failures)
- vLLM platform detection includes fallback logic for broken NVML
- GSM8K evaluator validates dataset splits before loading
- All answer extraction has multiple fallback patterns (strict → flexible)

## Cross-Cutting Concerns

**Logging:**
- No centralized logger; uses `print()` for interactive feedback
- Critical: Full prompt logging (no truncation) per CLAUDE.md
- vLLM setup logs platform detection, tensor parallelism
- Evaluator logs dataset size and accuracy metrics

**Validation:**
- GSM8K answer extraction validates 3 patterns before returning None
- Model alias resolution with helpful warnings
- GPU ID validation via CUDA_VISIBLE_DEVICES
- Minibatch size validation against dataset size

**Authentication:**
- API keys loaded via `.env` and `python-dotenv`
- Defaults: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPINFRA_API_KEY`
- Raises `ValueError` if required key missing for backend

**Optimization Hyperparameters:**
- Temperature: 0.0 for task LLM (deterministic solving), 1.0 for meta-optimizer (diverse candidates)
- Budget tracking: All algorithms track LLM call costs
- Minibatch size: Fixed across all prompts in iteration (reproducible fairness)
- Top-K selection: OPRO keeps 20 prompts; ProTeGi keeps variable beam width

---

*Architecture analysis: 2026-01-28*
