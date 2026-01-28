# Codebase Concerns

**Analysis Date:** 2026-01-28

## Tech Debt

**OPRO/LIPO Prompt Quality Issues:**
- Issue: LLM-generated prompts often contain Unicode gibberish and malformed instructions
- Files: `src/opro.py`, `run_opro.py`
- Impact: Meta-optimizer (temperature=1.0) generates candidates with mixed languages, incomplete truncations, breaking downstream evaluation. Example from TODO.md: "Solve problem步骤：解析数据，应用计算" (Chinese mixed with English)
- Fix approach:
  1. Add prompt validation to reject outputs with non-ASCII characters outside expected ranges
  2. Implement prompt quality filters before evaluation
  3. Use stricter extraction patterns in `generate_candidates()` - validate bracket extraction results
  4. Consider lowering meta-optimizer temperature or adding beam search with quality scores

**EcoFlow-BO: Fixed Exemplar Selection Bug:**
- Issue: BOLT optimizer returns identical exemplar sets across all 30 iterations, suggesting z_detail never changes
- Files: `ecoflow_bo/optimizer.py`, `ecoflow_bo/detail_retriever.py`, `ecoflow_bo/cycle_consistency.py`
- Impact: GP optimization on z_core produces no meaningful variation in exemplar selection - component is non-functional. From TODO.md: "All 30 iterations vybírá [4046, 3305, 2625, 3826, 4878, 2053, 5164, 1795]"
- Fix approach:
  1. Verify detail_retriever is actually being called and retrieving different neighbors for perturbed z_cores
  2. Check nearest neighbor similarity distribution - if all z_cores map to same nearest neighbor, increase diversity in acquisition
  3. Add logging to `detail_retriever.get_detail()` to track which neighbors are actually selected
  4. Test cycle_consistency retrieval pipeline end-to-end
  5. Consider "k_nearest" mode with averaging instead of "nearest" for smoother transitions

**GP Exploration Failure:**
- Issue: Gaussian Process in BOLT fails to discover that z_core dimensions matter (behaves as if exemplars are irrelevant)
- Files: `ecoflow_bo/latent_gp.py`, `ecoflow_bo/config.py`
- Impact: After 10 iterations, exemplar variance remains zero - GP has not learned to vary z_core values. From TODO.md: "GP ignoruje exemplar dimenze → zamrzlé exempláře"
- Fix approach:
  1. Add ARD lengthscale priors favoring small scales on z_core dimensions (GammaPrior(4.0, 4.0) per TODO.md)
  2. Or enforce explicit constraints: exemplar_constraint = Interval(0.1, 0.5) on exemplar dims
  3. Verify kernel is ExactGP or SpectralDeltaKernel, not approximations that could ignore dimensions
  4. Add diagnostic: log learned lengthscales per dimension to detect frozen dims

**GSM8K Answer Extraction Edge Cases:**
- Issue: Extract logic relies on three cascading regex patterns but some math answers may be extracted incorrectly
- Files: `src/gsm8k_evaluator.py` lines 36-65
- Impact: False positives/negatives on edge cases where numbers appear in reasoning but answer uses non-numeric format. Fallback "last number" rule is fragile.
- Fix approach:
  1. Add unit tests for edge cases: negative numbers, decimals, very large numbers, comma-separated thousands
  2. Consider stricter FLEXIBLE_PATTERN that requires number to be standalone word boundary
  3. Log extraction method used to track pattern effectiveness
  4. Validate against lm-eval-harness reference implementation for consistency

## Known Bugs

**LLM Client None Responses Unhandled:**
- Symptoms: When OpenAI/DeepInfra API returns empty choices (content filter) or errors, client appends None to results list. Downstream code may crash on None
- Files: `src/llm_client.py` lines 205, 217, 255-256, 268
- Trigger: Content filter on OpenAI/DeepInfra; API rate limits; auth failures (auth errors are re-raised, but transient errors return None)
- Workaround: Callers should check for None in batch results, e.g., `outputs = [o for o in outputs if o is not None]`
- Fix approach:
  1. Raise exception instead of returning None for transient errors
  2. Or add retry logic with exponential backoff in generate_batch()
  3. Add explicit None checks in OPRO.evaluate_prompt() and OPRO.generate_candidates()

**VLLMClient Platform Detection Brittle:**
- Symptoms: `_patch_vllm_platform()` monkey-patches internal vLLM modules which may change between versions
- Files: `src/llm_client.py` lines 22-42
- Trigger: vLLM version upgrade where internal module structure changes
- Workaround: Explicitly set VLLM_TARGET_DEVICE=cuda env var before import
- Fix approach:
  1. Check vLLM version and skip patching if > certain version (vLLM fixed NVML handling)
  2. Add try-catch around monkey-patching with fallback to environment variables only
  3. Document minimum vLLM version required

**Cycle Consistency Checker Residual Mode Coupling:**
- Symptoms: CycleConsistencyChecker now requires detailed_retriever for residual mode but API allows initialization without it
- Files: `ecoflow_bo/cycle_consistency.py` lines 114-134, `ecoflow_bo/optimizer.py` lines 284-291
- Trigger: Calling step() before detail_retriever is set; or using z_full input with wrong dimensions
- Workaround: Must call set_detail_retriever() after initialization before calling step()
- Fix approach:
  1. Make detail_retriever mandatory in __init__ if residual_mode=True (fail early)
  2. Or add explicit validation in step() that raises RuntimeError with clear message
  3. Add type hints Union[z_core, z_full] to clarify API expectations

## Security Considerations

**API Keys in Environment Variables:**
- Risk: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPINFRA_API_KEY loaded from .env file
- Files: `src/llm_client.py` line 5 (load_dotenv), line 180, 228
- Current mitigation: .env in .gitignore; keys not logged
- Recommendations:
  1. Add explicit check: if not api_key, raise ValueError with guidance instead of silent None
  2. Never log full API keys, only first 10 chars with asterisks if needed
  3. Add warning if running with VLLM_TARGET_DEVICE=cuda without proper auth (suggests unintended local/API mix)

**Model Weights Not Validated:**
- Risk: from_checkpoint() in EcoFlowBO and other places load PyTorch checkpoint with weights_only=False
- Files: `ecoflow_bo/optimizer.py` line 143 - explicit weights_only=False comment says it's needed for dataclass
- Current mitigation: Checkpoints are version controlled, known source
- Recommendations:
  1. Use weights_only=True with custom unpickler that only allows specific types (config dataclass)
  2. Add hash validation: save SHA256 of checkpoint, validate before loading
  3. Document that checkpoints must come from trusted sources

## Performance Bottlenecks

**Cycle Consistency Re-encoding Expensive:**
- Problem: Every step() call decodes z → x, then re-encodes x → z' for validation. Two full forward passes per candidate evaluation.
- Files: `ecoflow_bo/cycle_consistency.py` lines 161-234 compute_cycle_error, select_valid_from_ranked
- Cause: Necessary for quality checking but adds ~2x latency per optimization step
- Improvement path:
  1. Batch multiple candidates before re-encoding (currently one at a time)
  2. Cache encoder outputs if input z distribution is stable
  3. Use half-precision (float16) for cycle check only, not objective evaluation
  4. Consider probabilistic validation: sample only 10% of candidates for cycle check

**Nearest Neighbor Search Brute Force:**
- Problem: SimpleDetailRetriever uses dense matrix multiplication [B, 16] @ [N, 16].T for every step(). For N>100k, this becomes O(B*N*D) per iteration.
- Files: `ecoflow_bo/detail_retriever.py` lines 207-222 (brute force), lines 88-100 (FAISS fallback)
- Cause: FAISS only used if N>50k AND FAISS_AVAILABLE=True, otherwise falls back to SimpleDetailRetriever
- Improvement path:
  1. Always use FAISS for N>10k (not just 50k) - overhead is negligible
  2. Pre-compute approximate neighbors during initialize(), cache until next stage
  3. Use GPU-resident FAISS index (faiss.index_gpu_to_cpu) if available
  4. Profile actual latency: if <1s per step, optimization is fine

**GSM8K Evaluation Batch Size Bottleneck:**
- Problem: Fixed eval set of 261 examples evaluated on every prompt (OPRO.num_candidates_per_iter=8 candidates × 261 examples = 2088 LLM calls per iteration)
- Files: `src/opro.py` lines 46, 73-82, 126-130
- Cause: minibatch_size=261 gives stable scoring but costs 8x more LLM calls than necessary
- Improvement path:
  1. Use adaptive batch sizing: start with 20, grow to 261 only if std_dev is high
  2. Implement early stopping: evaluate candidates in parallel, stop when top-3 clear
  3. Cache fixed_eval_set embeddings to avoid re-computing prompts: questions are fixed
  4. Reduce to 50 examples instead of 261 for initial 5 iterations, grow later

## Fragile Areas

**OPRO Meta-Prompt Truncation Logic:**
- Files: `src/opro.py` lines 175-179, 195-197
- Why fragile: Prompts are truncated to max_prompt_length=300 chars for meta-context, but no validation that result is a complete, grammatical instruction
- Safe modification:
  1. Add post-truncation validation: result must match regex ^[A-Z].*[\.!?]$ (starts with capital, ends with punctuation)
  2. If invalid, discard and try next prompt instead of passing truncated version
  3. Test with intentionally long prompts to verify no silent breakage
- Test coverage: No unit tests for truncation edge cases

**Residual Latent Architecture Initialization:**
- Files: `ecoflow_bo/optimizer.py` lines 170-238 (initialize method)
- Why fragile: initialize() must be called before step(), and detail_retriever setup happens here. If skipped or called twice, state is inconsistent.
- Safe modification:
  1. Add _initialized flag check at start of step() (exists at line 264)
  2. Prevent double-initialization by raising RuntimeError if _initialized=True
  3. Document initialization contract in class docstring
- Test coverage: Integration test exists (test_encode_decode_cycle) but not full initialize→step cycle

**Matryoshka Dimension Validation:**
- Files: `ecoflow_bo/config.py` lines 264-284 (__post_init__)
- Why fragile: Validation asserts must have exact matches (e.g., final Matryoshka dim must equal latent_dim), but no helpful error messages
- Safe modification:
  1. Replace asserts with explicit ValueError() calls with diagnostic messages
  2. Add example valid configs in docstring
  3. Validate stage continuity: each GP stage must have contiguous prefix of dims
- Test coverage: No unit tests for config validation

## Scaling Limits

**Latent Space Dimensionality Curse:**
- Current capacity: GP operates on 16D z_core (BOLT), task evaluation fine on GSM8K ~1300 questions
- Limit: GP curse of dimensionality becomes severe at >20D. Active dims schedule [4, 8, 16] partially mitigates but exploration is slow.
- Scaling path:
  1. For 32D latent: use coarse-to-fine schedule [8, 16, 24, 32] with longer point collection
  2. Hybrid GP: separate GPs for instruction dims vs exemplar dims, combine acquisitions
  3. Sparse approximation: VarSparseGP or inducing points (GPyTorch has built-in support)

**Training Set Size Bottleneck:**
- Current capacity: detail_retriever stores full z_core/z_detail tensors in memory. For 100k examples: 100k × 16 × 4 bytes = 6.4 MB (fine). For 10M: 640 MB (manageable).
- Limit: FAISS index at 1M+ examples becomes slow on CPU. GPU FAISS needed.
- Scaling path:
  1. Use quantized FAISS index (IndexIVFFlat) for 10M+ examples (5-10x faster, minimal accuracy loss)
  2. Shard retriever across GPUs if available
  3. Online mode: stream training data, don't materialize all z_cores at once

**Fixed Evaluation Set Fixed:**
- Current capacity: 261 examples per OPRO iteration, max 200 iterations = max ~50k LLM calls
- Limit: For large-scale experiments, budget exhaustion at total_budget limit
- Scaling path:
  1. Budget tracking already exists in OPRO (total_budget parameter)
  2. Consider curriculum learning: eval_size grows with iteration (10 → 100 → 261)
  3. Add checkpoint/resume capability to save OPRO state and continue later

## Dependencies at Risk

**vLLM Version Sensitivity:**
- Risk: Code uses internal vLLM modules (_patch_vllm_platform), version pins not enforced
- Files: `src/llm_client.py` lines 22-42, 61 imports
- Impact: vLLM 0.5.x changed platform detection; future versions may further refactor internals
- Migration plan:
  1. Pin vLLM >=0.5.0,<0.7.0 in pyproject.toml (check latest stable)
  2. Add vLLM version check: if version > 0.6, skip patching
  3. Add CI test against multiple vLLM versions
  4. Consider switching to LM Studio or ollama as vLLM alternative (more stable API)

**GPyTorch ExactGP on Large Data:**
- Risk: CoarseToFineGP uses ExactGP which computes Cholesky decomposition O(N^3)
- Files: `ecoflow_bo/latent_gp.py` (inferred from config usage)
- Impact: Slow inference with >1k observations. Current max is ~100 points per stage, manageable.
- Migration plan:
  1. Profile actual inference time: if >100ms, switch to VariationalGP
  2. Use DiagLazyTensor approximation for faster covariance computation
  3. Document performance: "ExactGP suitable for <500 observations"

**Sentence-Transformers Model Loading:**
- Risk: Hard-coded model path "sentence-transformers/gtr-t5-base" in EcoFlowBOWithVec2Text
- Files: `ecoflow_bo/optimizer.py` lines 510-511
- Impact: Model may be deprecated, slow download on first run, no retry logic
- Migration plan:
  1. Make model configurable: add parameter to __init__
  2. Cache model locally: check ~/.cache/huggingface before downloading
  3. Add timeout and retry: model_kwargs={"timeout": 30, "local_files_only": False}
  4. Document: "Requires internet on first run, ~200MB download"

## Missing Critical Features

**No Hyperparameter Tuning for OPRO:**
- Problem: keep_top_k=20, num_candidates_per_iter=8 are fixed, not optimized per task
- Blocks: Can't easily compare different search widths or memory sizes
- Fix: Add command-line arguments --keep-top-k and --num-candidates to run_opro.py

**No Checkpoint/Resume for Long Runs:**
- Problem: If run crashes at iteration 150/200, must restart from scratch
- Blocks: Can't do fault-tolerant optimization on unreliable hardware
- Fix: Save OPRO state every N iterations, add --resume flag to load and continue

**No Multi-Objective Support:**
- Problem: OPRO optimizes single accuracy metric; can't optimize accuracy + latency simultaneously
- Blocks: Can't find Pareto-optimal prompts
- Fix: Add Pareto tracking in OPRO.optimize(), visualize trade-offs

**Missing Error Bounds/Uncertainty Quantification:**
- Problem: No confidence intervals on final prompt quality (accuracy is point estimate)
- Blocks: Can't report statistical significance of improvements
- Fix: Run optimization multiple times, report mean ± std, do t-tests

## Test Coverage Gaps

**OPRO Generation Lacks Unit Tests:**
- What's not tested: generate_candidates() method, meta-prompt formatting, prompt extraction from brackets
- Files: `src/opro.py` lines 181-279
- Risk: Bracket extraction regex may fail silently on edge cases ("[[double brackets]]", missing brackets, nested brackets)
- Priority: High - meta-optimization is core algorithm

**GSM8K Evaluator Edge Cases Untested:**
- What's not tested: answer extraction for negative numbers, decimals, scientific notation, very large numbers
- Files: `src/gsm8k_evaluator.py` lines 36-65
- Risk: False positives/negatives on specific answer formats
- Priority: High - affects all experiments

**LLM Client Error Handling Untested:**
- What's not tested: None response handling, API errors, timeouts, retry logic
- Files: `src/llm_client.py` lines 190-218
- Risk: Pipeline crashes on transient errors instead of gracefully degrading
- Priority: Medium - occurs in production but may be rare

**Cycle Consistency Residual Mode Untested:**
- What's not tested: Full residual latent workflow (z_core → detail_retriever → z_full → decode → re-encode)
- Files: `ecoflow_bo/cycle_consistency.py`, `ecoflow_bo/detail_retriever.py`
- Risk: Integration bug remains undetected (e.g., detail_retriever returns wrong shape)
- Priority: High - was added recently in refactor

**EcoFlow Config Validation Partially Tested:**
- What's not tested: Invalid Matryoshka dims, mismatched encoder/decoder configs
- Files: `ecoflow_bo/config.py` __post_init__
- Risk: Invalid configs raise cryptic AssertionError instead of ValueError
- Priority: Medium - typically caught during development

---

*Concerns audit: 2026-01-28*
