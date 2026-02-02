# Codebase Concerns

**Analysis Date:** 2026-01-31

## Tech Debt

**Bare Exception Handlers:**
- Issue: Multiple modules use `except Exception` or bare `except:` clauses without specific error types, making debugging and error handling harder to trace
- Files: `protegi/protegi.py`, `protegi/run.py`, `rielbo/optimization_loop.py`, `rielbo/run.py`, `shared/llm_client.py`, `nfbo/run.py`
- Impact: Swallows unexpected errors, makes error recovery harder, masks programming mistakes
- Fix approach: Replace broad exception handlers with specific exception types (e.g., `except (RuntimeError, ValueError)`) and re-raise unhandled errors

**vLLM Platform Detection Workaround:**
- Issue: `shared/llm_client.py` lines 22-42 contain monkey-patch for vLLM NVML detection failures with fallback to print warnings
- Files: `shared/llm_client.py:22-42`
- Impact: Fragile dependency on vLLM internals; if vLLM structure changes, platform detection breaks silently
- Fix approach: Document the vLLM version requirement in CLAUDE.md; consider using environment variables for platform forcing instead of runtime patching

**GPU Memory Cleanup Complexity:**
- Issue: `shared/llm_client.py:89-141` contains complex multi-step cleanup with nested exception handling, comments warn about process group destruction
- Files: `shared/llm_client.py:89-141`
- Impact: GPU memory leaks possible if cleanup fails; requires manual restart in edge cases (noted in warning message)
- Fix approach: Add teardown tests to verify memory is released; consider using context managers for automatic cleanup

## Known Bugs

**OpenAI/DeepInfra Content Filter Handling:**
- Symptoms: Returns `None` when API returns empty choices, not distinguishable from actual failure
- Files: `shared/llm_client.py:203-207` (OpenAI), `shared/llm_client.py:254-258` (DeepInfra)
- Trigger: Content filter triggers on prompt during batch evaluation
- Workaround: Callers must check for `None` values in results and handle accordingly; no retry logic

**GSM8K Answer Extraction Edge Cases:**
- Symptoms: May fail on formats not matching regex patterns (STRICT_PATTERN, BOXED_PATTERN, FLEXIBLE_PATTERN)
- Files: `shared/gsm8k_evaluator.py:36-65`
- Trigger: Model outputs with non-standard formatting (e.g., "The answer is approximately 42.5")
- Workaround: Falls back to last numeric value found, but this is fragile for multi-number outputs

**Flow Matching ODE Integration Device Handling:**
- Symptoms: Tensor device mismatches possible if normalization stats not on correct device
- Files: `rielbo/flow_model.py:24-38`
- Trigger: Using flow model on different GPU than where norm_stats were computed
- Workaround: Explicit `.to(x.device)` calls in normalize/denormalize, but not validated at init

## Security Considerations

**API Key Exposure in Error Messages:**
- Risk: Exception messages in `shared/llm_client.py:211-217` may leak context about API failures, though keys are not directly logged
- Files: `shared/llm_client.py:211-217`
- Current mitigation: OpenAI client checks for "auth" in error string before re-raising; not comprehensive
- Recommendations: Add log redaction for API key patterns; use structured logging with sanitization

**Untrusted Prompt Execution:**
- Risk: SONAR decoder in `rielbo/decoder.py` receives generated embeddings that may encode adversarial content
- Files: `rielbo/decoder.py:36-82`
- Current mitigation: None; relies on downstream task model to validate semantic safety
- Recommendations: Add prompt validation filter pre-evaluation; log decoded prompts for audit trails

**Environment Variable Reliance:**
- Risk: `.env` loading in `shared/llm_client.py:7` may fail silently if .env is missing
- Files: `shared/llm_client.py:5-7`
- Current mitigation: Specific checks for API keys on client init, but only for API-based backends
- Recommendations: Centralize .env loading with validation; warn if critical keys missing at startup

## Performance Bottlenecks

**GP Surrogate Fitting in Optimization Loop:**
- Problem: Each BO iteration refits GP from scratch on all historical data (O(n³) cubic complexity in BO observations)
- Files: `rielbo/gp_surrogate.py:258-284` (SonarGPSurrogate.fit/update), `rielbo/optimization_loop.py:270-320`
- Cause: `_fit_model()` calls `fit_gpytorch_mll()` which solves Cholesky decomposition on full data
- Improvement path: Implement incremental GP updates using Sherman-Morrison formula for rank-1 updates; cache Cholesky factorization

**1024D Embedding Space Without Dimensionality Reduction:**
- Problem: GP operates in full 1024D SONAR space, leading to slow posterior computations and high kernel memory
- Files: `rielbo/gp_surrogate.py:231-256` (SonarGPSurrogate), `rielbo/guided_flow.py` (guidance generation)
- Cause: MSR initialization uses full-D lengthscale priors; no automatic dimensionality reduction
- Improvement path: Use BAxUSGPSurrogate with reduced target_dim by default (currently must be explicitly selected); profile to find optimal subspace dimension

**ODE Integration with Small Time Steps:**
- Problem: `rielbo/flow_model.py:46-89` uses default num_steps=50 for 1024D integration; may be slow for large batches
- Files: `rielbo/flow_model.py:46-89`, `rielbo/optimization_loop.py` (guidance loop)
- Cause: No adaptive step size control; trades accuracy for speed without auto-tuning
- Improvement path: Implement tolerance-based adaptive Runge-Kutta; benchmark step count vs accuracy

**Batch Evaluation Without Parallelization:**
- Problem: `protegi/protegi.py` and other optimizers evaluate candidates sequentially with LLM calls
- Files: `protegi/protegi.py:200-250`, `rielbo/optimization_loop.py:240-320`
- Cause: vLLM batch size is single-example per prompt; no prompt fusion/batching
- Improvement path: Batch multiple prompts together using vLLM native batch API; reduce I/O overhead

## Fragile Areas

**Flow Model Checkpoint Loading:**
- Files: `rielbo/run.py:150-170` (checkpoint path resolution)
- Why fragile: Path must be absolute or relative to cwd; no validation that checkpoint exists or has correct architecture before loading
- Safe modification: Add explicit checkpoint validation in `FlowMatchingModel.load_checkpoint()`; check expected input_dim
- Test coverage: Only implicit via integration tests; no unit tests for checkpoint I/O

**BAxUS Embedding Matrix Initialization:**
- Files: `rielbo/gp_surrogate.py:306-315`
- Why fragile: Random sparse embedding matrix S generated with `torch.rand()` without seed control in default path; non-reproducible across runs
- Safe modification: Always accept optional seed parameter and use `torch.manual_seed()` before matrix creation
- Test coverage: `tests/test_gp_surrogate.py` does not test reproducibility of BAxUSGPSurrogate with fixed seed

**ProTeGi Beam Search State Management:**
- Files: `protegi/protegi.py:76-150` (ProTeGiOptimizer class)
- Why fragile: Beam is updated in-place; no state snapshots between steps; if evaluation fails mid-iteration, beam may be in inconsistent state
- Safe modification: Use immutable dataclass updates; snapshot beam before each evaluation step; implement rollback on failure
- Test coverage: No tests for beam consistency under partial failures

**GSM8K Evaluator Index Bounds:**
- Files: `shared/gsm8k_evaluator.py:141-179` (evaluate_batch)
- Why fragile: No validation that indices are within dataset bounds; accessing `self.dataset[idx]` can fail silently with KeyError on out-of-bounds
- Safe modification: Add bounds checking in `evaluate_batch()` before accessing dataset
- Test coverage: No tests for out-of-bounds indices

## Scaling Limits

**1024D GP Kernel Matrix Memory:**
- Current capacity: ~1000 observations in-memory (kernel matrix is 1000x1000 with 1024D features)
- Limit: Exceeding ~2000 observations causes OOM on 48GB GPU; kernel matrix grows as O(n²)
- Scaling path: Implement inducing point approximation (sparse GP); use subset of Regressors or variational inference

**vLLM Single GPU Occupancy:**
- Current capacity: One 7B model on single L40S GPU with batch_size=1-4 for 1024-token prompts
- Limit: Cannot run parallel evaluations without model unloading; switching models requires full cleanup
- Scaling path: Implement model pooling/recycling; use multi-GPU tensor parallelism via `tensor_parallel_size=2` (documented in CLAUDE.md)

**Batch Evaluation With Small Eval Subset:**
- Current capacity: Default eval_subset_size=150 examples for prompt evaluation
- Limit: Variance in accuracy estimates high for small subsets; BO convergence may be slow
- Scaling path: Increase eval_subset_size to 500-1000 for production; implement adaptive sampling based on prompt uncertainty

## Dependencies at Risk

**vLLM Version Pinning (0.10.x):**
- Risk: Pinned to `vllm>=0.10.0,<0.11.0` due to API instability; upgrading may break platform detection and generation API
- Impact: Stuck on older vLLM; cannot use newer features or bug fixes
- Migration plan: Document compatibility matrix for vLLM versions; use feature detection instead of version checks

**vec2text Package Deprecation:**
- Risk: `vec2text>=0.0.13` is archived upstream; decoding via SONAR fallback may not be maintained
- Impact: Decoder may break on new systems; embedding-to-text research has moved to other repos
- Migration plan: Consider switching to sentence-transformers multi-vector decoder or maintaining fork of SONAR

**Soft Prompt VAE Experimental Features:**
- Risk: LoRA adapters via `peft>=0.14.0` and soft prompt VAE are research features not widely tested in production
- Impact: May encounter compatibility issues with newer transformers versions
- Migration plan: Pin feature to specific transformer versions if used; add integration tests

## Missing Critical Features

**Error Recovery and Retries:**
- Problem: No exponential backoff or retry logic for transient API failures (rate limits, network timeouts)
- Blocks: Long-running optimization cannot gracefully handle temporary API outages
- Mitigation: Manual checkpoint resumption is documented, but user must restart manually

**Checkpoint Versioning:**
- Problem: No schema versioning for saved checkpoints; format changes break resume from old checkpoints
- Blocks: Upgrading model architecture requires restarting optimization from scratch
- Mitigation: Document checkpoint format in each release

**Hyperparameter Search:**
- Problem: No automated hyperparameter tuning for GP surrogate (target_dim for BAxUS, kernel priors, optimization learning rates)
- Blocks: No principled way to tune for new tasks; defaults may be suboptimal
- Mitigation: Validation script provided in `rielbo/validate.py` but not integrated into pipeline

## Test Coverage Gaps

**ProTeGi Gradient Generation:**
- What's not tested: Edge cases where gradient prompt formatting fails; malformed feedback handling
- Files: `protegi/protegi.py:200-280` (gradient generation and application)
- Risk: May silently fail to improve prompts or crash with formatting errors mid-optimization
- Priority: Medium - crashes would surface quickly, but silent failures on edge cases possible

**Flow Model ODE Accuracy:**
- What's not tested: Verification that ODE integration produces samples on SONAR manifold; no distribution matching tests
- Files: `rielbo/flow_model.py:46-89`, `rielbo/flow_model.py:92-120` (sample method)
- Risk: Flow may produce out-of-distribution embeddings causing decoder failures
- Priority: High - affects entire RieLBO method validity

**Decoder Robustness:**
- What's not tested: Behavior when decoder fails (OOM, malformed embedding); no fallback handling
- Files: `rielbo/decoder.py:36-82`
- Risk: Single failing embedding halts entire optimization batch
- Priority: High - blocking issue for long runs

**GP Update Consistency:**
- What's not tested: That GP training data remains consistent after incremental updates; no numerical stability checks
- Files: `rielbo/gp_surrogate.py:269-284` (BAxUSGPSurrogate.update)
- Risk: Accumulated numerical errors in iterative updates may degrade model quality
- Priority: Medium - would manifest as poor exploration decisions after many iterations

**LLM Client Edge Cases:**
- What's not tested: Handling of extremely long prompts (>model context), tokenizer edge cases, unicode normalization
- Files: `shared/llm_client.py:143-171` (VLLMClient.generate_batch), transformers backend
- Risk: Evaluation crashes on certain prompt types
- Priority: Low - unusual formats would likely fail deterministically

---

*Concerns audit: 2026-01-31*
