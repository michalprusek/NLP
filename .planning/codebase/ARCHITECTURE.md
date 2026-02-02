# Architecture

**Analysis Date:** 2026-01-31

## Pattern Overview

**Overall:** Modular optimizer framework with pluggable LLM clients and evaluation pipelines

**Key Characteristics:**
- Multi-method research codebase supporting 5 distinct prompt optimization algorithms
- Common shared infrastructure (LLM client abstraction, evaluation harness)
- Each method independently implements its optimization strategy
- Pipeline: evaluate prompts → score → optimize → iterate

## Layers

**Shared Infrastructure Layer:**
- Purpose: Provide pluggable LLM clients and evaluation utilities used by all optimizers
- Location: `shared/`
- Contains: LLMClient (ABC), Concrete implementations (VLLMClient, OpenAIClient, DeepInfraClient, TransformersClient), GSM8KEvaluator
- Depends on: External LLM APIs (vLLM, OpenAI, DeepInfra), HuggingFace Transformers, Datasets
- Used by: All optimizer methods (OPRO, ProTeGi, GEPA, RieLBO, NFBO)

**Prompt Optimization Methods Layer:**
- Purpose: Implement distinct optimization algorithms for discovering better task prompts
- Location: `opro/`, `protegi/`, `gepa/`, `rielbo/`, `nfbo/`
- Contains: Optimizer classes (OPROOptimizer, ProTeGiOptimizer, GEPAOptimizer, BOOptimizationLoop, NFBoLoop), run.py CLI entry points
- Depends on: Shared infrastructure, PyTorch, specialized packages (torchcfm, botorch, gpytorch)
- Used by: CLI entry points for running optimization experiments

**RieLBO-Specific Architecture (Sub-layer):**
- Purpose: Flow matching guided Bayesian optimization in SONAR embedding space
- Components:
  - `velocity_network.py`: DiT-style transformer with AdaLN time conditioning for velocity field modeling
  - `flow_model.py`: Conditional flow matching model wrapping velocity network with training utilities
  - `guided_flow.py`: GuidedFlowSampler combining GP-UCB guidance with ODE sampling
  - `gp_surrogate.py`: GP surrogate models (SonarGPSurrogate, BAxUSGPSurrogate, Heteroscedastic)
  - `decoder.py`: SonarDecoder wrapping Meta SONAR text-to-embedding and embedding-to-text models
  - `optimization_loop.py`: BOOptimizationLoop orchestrating full BO cycle
  - `train_flow.py`: Flow model training with EMA and checkpoint management

**NFBO-Specific Architecture (Sub-layer):**
- Purpose: Normalizing flow Bayesian optimization (RealNVP latent space BO)
- Components:
  - `model.py`: RealNVP normalizing flow with invertible transformations
  - `sampler.py`: NFBoSampler implementing latent BO in Z-space
  - `loop.py`: NFBoLoop extending BOOptimizationLoop with flow fitting and latent optimization

**CLI and Configuration Layer:**
- Purpose: User-facing entry points with argument parsing and logging
- Location: `*/run.py` files for each method
- Contains: Argument parsers, initialization logic, main loop coordination
- Depends on: Method-specific optimizers, shared infrastructure
- Used by: Command-line invocation

## Data Flow

**OPRO (Meta-optimization) Flow:**

1. Initialize task and meta LLM clients (same or different models)
2. Create fixed evaluation set from GSM8K train split (261 examples)
3. For each iteration:
   - Generate K candidates from meta-model using best prompts + example failures as context
   - Score each candidate on fixed eval set with task model
   - Collect feedback (which candidates worked best)
   - Meta-model proposes next candidates based on scored history
4. Return best prompt, evaluate on test set, save results

**ProTeGi (Textual Gradients) Flow:**

1. Initialize task model and meta model
2. Start with random prompt
3. For each step:
   - Evaluate prompt on examples, identify failure cases
   - Meta-model generates "textual gradients" (natural language error analysis)
   - Generate edited candidates from gradients
   - Monte Carlo paraphrasing for local search
   - Beam search + UCB bandit selection to choose best candidates
   - Track top-K prompts in beam
4. Return best prompt with test set evaluation

**GEPA (Genetic-Pareto) Flow:**

1. Initialize population of candidates
2. For each iteration:
   - Evaluate candidates, maintain Pareto front (accuracy vs response length)
   - For low-scoring examples, extract reasoning traces
   - Meta-model reflects on failures ("what would fix this?")
   - Generate mutations based on reflections
   - Combine mutations with parent candidates
   - Select next generation via Pareto dominance
3. Return best prompt from final population

**RieLBO (Flow + BO) Flow:**

1. Load pretrained FlowMatchingModel (trained on SONAR embeddings)
2. Initialize SonarDecoder (embedding → text)
3. Create GP surrogate (BAxUS or Heteroscedastic)
4. For each iteration:
   - Optimize GP-UCB acquisition function in embedding space (bounded to high-confidence region)
   - Sample from flow model using guided ODE with UCB guidance
   - Apply L2-r filtering (round-trip fidelity check via SONAR encoder)
   - Decode selected embedding to text prompt
   - Evaluate prompt on GSM8K subset via LLM
   - Update GP with observation
   - Track best score and prompt
5. Save checkpoints, metrics, best prompt

**NFBO (Normalizing Flow BO) Flow:**

1. Initialize random samples, evaluate
2. Fit RealNVP normalizing flow on observed embeddings
3. For each iteration:
   - Create GP in Z-space (latent of RealNVP)
   - Optimize GP-UCB in Z-space, sample optimized z
   - Transform z → x via flow bijection
   - Apply L2-r filter if enabled
   - Decode embedding to prompt
   - Evaluate, update observations and flow
4. Return best prompt and metrics

**State Management:**

- **OPRO/ProTeGi/GEPA:** Store scored prompts in memory, serialize best prompt to JSON/TXT
- **RieLBO/NFBO:** Use OptimizationState dataclass with save/load methods, checkpoint embeddings (train_X/train_Y), best prompt, iteration counter

## Key Abstractions

**LLMClient:**
- Purpose: Unified interface for different LLM backends
- Examples: `shared/llm_client.py` (VLLMClient, OpenAIClient, DeepInfraClient, TransformersClient)
- Pattern: Abstract base class with `generate()` and `generate_batch()` methods; factory function `create_llm_client()` with auto-detection

**Evaluator:**
- Purpose: Score prompts on task (GSM8K math reasoning)
- Examples: `shared/gsm8k_evaluator.py` (GSM8KEvaluator)
- Pattern: Load dataset, provide batch evaluation with answer extraction and comparison

**ScoredPrompt:**
- Purpose: Encapsulate prompt + score with metadata
- Examples: `opro/opro.py` (ScoredPrompt), `protegi/protegi.py` (ScoredPrompt with UCB stats), `gepa/gepa.py` (ScoredCandidate with traces)
- Pattern: Dataclass holding text and metrics, methods for comparison/selection

**BOOptimizationLoop:**
- Purpose: Orchestrate full Bayesian optimization cycle
- Examples: `rielbo/optimization_loop.py` (base), `nfbo/loop.py` (extends with flow fitting)
- Pattern: Initialize with models/evaluators, expose `initialize()` and `step()` methods, manage state

**OptimizationState:**
- Purpose: Checkpoint-able full state for resumable optimization
- Examples: `rielbo/optimization_loop.py`
- Pattern: Dataclass with torch tensors, save/load via torch.save()

## Entry Points

**OPRO:**
- Location: `opro/run.py`
- Triggers: `python -m opro.run [--model qwen] [--iterations 200]`
- Responsibilities: Parse args, initialize LLM clients/evaluator, run OPROOptimizer.optimize(), evaluate on test set, save results JSON + prompt TXT

**ProTeGi:**
- Location: `protegi/run.py`
- Triggers: `python -m protegi.run [--model qwen] [--steps 6]`
- Responsibilities: Similar to OPRO but coordinates ProTeGiOptimizer with beam search and UCB selection

**GEPA:**
- Location: `gepa/run.py`
- Triggers: `python -m gepa.run [--model qwen] [--budget 150000]`
- Responsibilities: Initialize population, run GEPA loop with Pareto selection and reflection-based mutations

**RieLBO:**
- Location: `rielbo/run.py`
- Triggers: `python -m rielbo.run [--iterations 100] [--flow-checkpoint path]`
- Responsibilities: Load flow model, initialize GP/decoder/evaluator, run BOOptimizationLoop with checkpoint save/resume support

**NFBO:**
- Location: `nfbo/run.py`
- Triggers: `python -m nfbo.run [--iterations 50] [--n-initial 20]`
- Responsibilities: Initialize evaluator, create NFBoSampler + NFBoLoop, run BO with flow fitting per iteration

**Flow Training (RieLBO):**
- Location: `rielbo/train_flow.py`
- Triggers: `python -m rielbo.train_flow [--epochs 50] [--batch-size 1024]`
- Responsibilities: Load SONAR embeddings, train FlowMatchingModel with EMA, save checkpoints

## Error Handling

**Strategy:**
- Early API failures (missing keys, bad auth) → raise immediately
- Generation failures (content filter, API error) → return None, allow downstream to handle
- Numeric edge cases (division by zero in metrics) → log warning, use safe defaults
- Decoder failures (invalid embedding) → fall back to generic placeholder prompt

**Patterns:**

**LLMClient Exception Handling:**
```python
# OpenAI/DeepInfra batch generation
for prompt in prompts:
    try:
        response = client.chat.completions.create(...)
        results.append(response.choices[0].message.content)
    except KeyboardInterrupt:
        raise  # Never swallow
    except Exception as e:
        if "auth" in type(e).__name__.lower():
            raise  # Re-raise auth errors
        results.append(None)  # Return None for recoverable errors
```

**Decoder Safe Decoding:**
```python
# `rielbo/optimization_loop.py` _decode_safe()
try:
    prompts = self.decoder.decode(embedding)
    return prompts
except Exception as e:
    logger.warning(f"Decoder failed: {e}")
    return ["Provide a concise step-by-step solution."]  # Fallback
```

## Cross-Cutting Concerns

**Logging:**
- Method-specific: Print statements for visibility (OPRO/ProTeGi/GEPA)
- RieLBO/NFBO: Python logging module via `logging.getLogger(__name__)`
- CLI: Configure via `--log-level` argument (INFO, DEBUG, WARNING, ERROR)
- CLAUDE.md constraint: Always log full prompt text, never truncate

**Validation:**
- Prompt format: Check non-empty, reasonable length in decoder
- Answer extraction: Regex-based with multiple pattern fallbacks (####, \\boxed{}, last number)
- Embedding validity: L2-r filter checks round-trip fidelity in RieLBO
- Budget checking: OPRO tracks total_budget spent across iterations

**Authentication:**
- Environment variables: OPENAI_API_KEY, DEEPINFRA_API_KEY, ANTHROPIC_API_KEY loaded via dotenv
- Client factory auto-detects based on model name (gpt → OpenAI, google/* → DeepInfra, others → vLLM)

**GPU Management:**
- vLLM: Multiple clients managed via CUDA_VISIBLE_DEVICES, tensor_parallel_size
- RieLBO: Explicit device passing, deterministic random seed for reproducibility
- Cleanup: VLLMClient.cleanup() to free memory between model loads

---

*Architecture analysis: 2026-01-31*
