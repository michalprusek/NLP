# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Framework for automatic prompt optimization using **ProTeGi** (Prompt Optimization with Textual Gradients) and **OPRO** (Optimization by PROmpting) on the GSM8K math word problem dataset.

- **ProTeGi**: Uses LLM-generated critiques (textual gradients) + beam search + UCB bandit algorithm for prompt refinement
- **OPRO**: Uses LLM as meta-optimizer to generate improved prompts by analyzing previous results

## Common Commands

### Setup

**Install dependencies:**
```bash
uv sync
```

**Configure API keys:**
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your Anthropic API key for Claude models
# ANTHROPIC_API_KEY=your_key_here
```

### Running Optimizations

**Basic optimization (same model for task and meta-optimization):**
```bash
# ProTeGi with Qwen
uv run python main.py --method protegi --model Qwen/Qwen2.5-7B-Instruct --iterations 10

# OPRO with SaulLM (legal domain model)
uv run python main.py --method opro --model Equall/Saul-7B-Instruct-v1 --iterations 10
```

**Separate task and meta-optimizer models:**
```bash
# Use Qwen for task evaluation, Claude Sonnet for meta-optimization
uv run python main.py \
    --method protegi \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --meta-model sonnet \
    --iterations 10

# Use small local model for task, Claude Haiku for meta-optimization (cost-effective)
uv run python main.py \
    --method protegi \
    --model Qwen/Qwen2.5-3B-Instruct \
    --meta-model haiku \
    --iterations 10
```

**Shell scripts (with environment variables and aliases):**
```bash
# Single GPU with separate models (using aliases)
META_MODEL="sonnet" ./run_single_gpu.sh protegi 5 20

# Dual GPU (using aliases)
TASK_MODEL="Qwen/Qwen2.5-7B-Instruct" META_MODEL="haiku" ./run_dual_gpu.sh

# Or with full model names
META_MODEL="claude-sonnet-4-5-20251022" ./run_single_gpu.sh protegi 5 20
```

**Claude Model Aliases:**
- `haiku` → `claude-haiku-4-5-20251001` (latest Haiku 4.5)
- `sonnet` → `claude-sonnet-4-5-20251022` (latest Sonnet 4.5)

**Evaluation only (no optimization):**
```bash
# Evaluate a prompt on GSM8K test set
uv run python evaluate_gsm8k.py --prompt "Your prompt here" --num-samples 5

# With self-consistency (majority voting)
uv run python evaluate_gsm8k.py --prompt "Your prompt" --num-samples 10
```

### Supported Models

**Task Models (--model):**
- `Qwen/Qwen2.5-7B-Instruct` - **Default model**, general-purpose, good performance (~85% GSM8K)
- `Qwen/Qwen2.5-3B-Instruct` - Smaller, faster, lower memory
- `Equall/Saul-7B-Instruct-v1` - Legal domain-specialized model (based on Mistral-7B, trained on 30B legal tokens)
- `meta-llama/Llama-3.1-8B-Instruct` - Meta's Llama 3.1
- `claude-3-haiku-20240307` - Fast Claude model (API)
- `claude-3-5-sonnet-20241022` - Most capable Claude model (API)

**Meta-optimizer Models (--meta-model):**
- Same as task models, plus recommended:
- `claude-3-5-sonnet-20241022` - Best for complex meta-optimization
- `claude-3-haiku-20240307` - Cost-effective for meta-optimization

### Important Parameters

**Model Selection:**
- `--model`: Task model (being optimized)
- `--backend`: Backend for task model (`auto`, `transformers`, `vllm`, `claude`)
- `--meta-model`: Meta-optimizer model for gradient generation/editing (optional, defaults to --model)
- `--meta-backend`: Backend for meta-optimizer model (`auto` detects Claude models)

**Optimization:**
- `--method`: `protegi` or `opro`
- `--iterations`: Number of optimization iterations (default: 10)
- `--minibatch-size`: Examples per evaluation (default: 20)
- `--beam-size`: Beam size for ProTeGi (default: 4)
- `--num-candidates`: Candidates per iteration for OPRO (default: 8, as per paper)

**Evaluation:**
- `--evaluator`: GSM8K evaluator with numerical tolerance (exact match with tolerance for floating point comparisons)
- `--train-split` / `--val-split`: Which GSM8K split to use for training/validation

**Hardware:**
- `--device`: `auto`, `cuda`, `mps`, `cpu` (for transformers backend)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (vLLM only)
- `--gpu-ids`: Comma-separated GPU IDs (e.g., "0,1")

**Debug:**
- `--debug`: Show model outputs and extraction details

## Architecture

### Core Components

**Optimization Algorithms** (src/):
- `protegi.py`: ProTeGi implementation with beam search, UCB selection, and textual gradient application
- `opro.py`: OPRO implementation with meta-prompting for candidate generation

**Evaluation** (src/):
- `evaluator.py`: Simplified GSM8K evaluator with numerical tolerance
- `claudette_evaluator.py`: Multi-label ToS classification evaluator
- `claudette_binary_evaluator.py`: Binary ToS classification evaluator (fair vs unfair)
- Answer extraction uses prioritized patterns: `final_answer: NUMBER`, `#### NUMBER`, `\boxed{NUMBER}`, fallback to last number

**LLM Clients** (src/llm_client.py):
- `TransformersClient`: HuggingFace transformers backend (CPU/GPU/MPS)
- `VLLMClient`: vLLM backend (GPU only, much faster)
- `ClaudeClient`: Anthropic Claude API backend (requires ANTHROPIC_API_KEY in .env)
- Auto-detects and applies chat templates for Instruct models
- Handles device selection, dtype optimization, and batch generation
- Supports separate task and meta-optimizer models for hybrid optimization

### Key Design Patterns

**Dual-Model Architecture:**
Both ProTeGi and OPRO now support separate models for different roles:
- **Task Model**: The model being optimized - evaluates prompts on actual task (e.g., solving math problems)
- **Meta-optimizer Model**: Generates gradients, edits prompts, creates new candidates
- **Benefits**:
  - Use powerful API models (Claude Sonnet) for meta-optimization while keeping local models for task evaluation
  - Cost-effective: Small local model for task + Claude Haiku for meta-optimization
  - Better meta-optimization: Claude models excel at critique and prompt engineering
- **Implementation**: Both models share the same LLMClient interface, allowing seamless mixing of backends

**ProTeGi Workflow:**
1. Maintain beam of top-K prompt candidates
2. Select candidate via UCB (exploration/exploitation balance)
3. Evaluate on minibatch → Generate textual gradient (LLM critique)
4. Apply gradient → Generate N improved prompts
5. Evaluate new prompts, update beam with diversity penalty
6. Repeat until convergence or iteration limit

**OPRO Workflow:**
1. **Fixed Evaluation Set**: Create small fixed subset of training data (~3.5% as in paper)
2. Evaluate initial seed prompts on this SAME fixed set
3. Format top-K scored prompts as context for meta-optimizer
4. LLM generates N new candidate prompts
5. Evaluate candidates on the SAME fixed set (ensures comparable scores)
6. Keep top-K overall, repeat
7. **Key advantage**: All prompts evaluated on identical data → directly comparable scores, no noise from different samples

**Evaluation Sampling Strategies:**
- **OPRO**: Fixed evaluation set (same examples for all prompts) - ensures comparable scores, avoids noisy signals
- **ProTeGi**: Stratified sampling without replacement for better dataset coverage
  - Tracks used indices, resets when >80% of dataset seen
  - Helps avoid overfitting to specific examples

**Diversity Management:**
- ProTeGi beam selection includes diversity penalty (SequenceMatcher similarity)
- Prevents beam collapse to near-identical prompts
- Threshold: 85% similarity (configurable)

### Meta-Prompts

Both algorithms use carefully engineered meta-prompts:

**IMPORTANT:** This implementation uses **significantly enhanced meta-prompts** compared to the original paper (Appendix 1.1). See `src/prompts/README.md` for detailed comparison and rationale. Key differences:
- Paper uses minimal prompts (~5 lines), we use structured prompts (26-74 lines)
- We add brevity constraints, output format rules, anti-artifact measures
- We explain evaluation system to generate better critiques
- Trade-off: Better quality prompts vs. less faithful to paper

**ProTeGi GRADIENT_PROMPT** (src/prompts/gsm8k/gradient.txt):
- Critiques current prompt based on error examples
- Explains evaluation system (extraction patterns and numerical comparison)
- Requests 2-4 issues with root causes and specific improvements
- Structured output format: ISSUE / Root cause / Suggested improvements

**ProTeGi EDIT_PROMPT** (src/prompts/gsm8k/edit.txt):
- Applies critic's feedback to improve prompt
- Hard constraints: brevity (max 3 sentences/150 words), no meta-text, output only improved prompt
- Includes extensive post-processing to remove LLM artifacts (see `apply_gradient` method)

**OPRO META_PROMPT** (src/opro.py:25-64):
- Analyzes previous scored prompts
- Generates N new diverse candidates
- Strict output format (one prompt per line, no numbering/bullets)

### Prompt Cleaning

ProTeGi includes aggressive prompt cleaning (`apply_gradient` method):
- Removes markdown formatting, code fences, preambles
- Deduplicates sentences (consecutive and non-consecutive)
- Enforces length limits (300 words max)
- Strips meta-commentary patterns
- Validates output before accepting

## Repository Structure

```
/home/prusek/NLP/
├── datasets/                    # Input data and ground truth
│   ├── gsm8k/                  # GSM8K dataset (7,473 train, 1,319 test)
│   ├── tos_local/              # ToS fairness classification (6,589 train, 1,412 val, 1,413 test)
│   │                           # 50 companies, 8 unfair categories, ~11% unfair clauses
│   │                           # Supports multi-label (claudette) and binary (claudette_binary) tasks
│   └── hbbops/                 # HbBoPs prompt datasets
│       ├── instructions_5.txt        # 5 curated instructions (selected)
│       ├── instructions_10.txt       # 10 uniform instructions
│       ├── instructions_25.txt       # 25 diverse instructions (full)
│       ├── examples_5.txt            # 5 selected exemplars
│       ├── examples_25.txt           # 25 diverse exemplars (K-means clustering)
│       └── full_grid_combined.jsonl  # Ground truth: all 625 prompt combinations
│
├── hbbops/                      # HbBoPs (Hyperband Bayesian Optimization for Prompts)
│   ├── hbbops.py               # Core HbBoPs algorithm implementation
│   ├── run_hbbops.py           # Main entry point for running experiments
│   ├── prompt_inverse.py       # Gradient-based prompt optimization
│   ├── data/                   # ToS dataset splits (train.json, validation.json, test.json)
│   └── results/                # Experiment outputs
│       ├── 10x25_*.json              # 10×25 grid search results
│       ├── 25x25_*.json              # 25×25 grid search results
│       ├── *.log                     # Detailed execution logs
│       └── grid_size_comparison.png  # Visualization comparing grid sizes
│
├── visualize/                   # Visualization scripts and plots
│   ├── compare_grid_sizes.py   # Compare 10×25 vs 25×25 experiments
│   ├── visualize_split.py      # Analyze train/val/test splits
│   ├── compare_all_hbbops.py   # Compare multiple HbBoPs runs
│   ├── full_gp_visualization.png     # Full GP visualization
│   └── split_visualization.png       # Dataset split visualization
│
├── results/                     # ProTeGi/OPRO experiment outputs
│   ├── protegi_TIMESTAMP.json  # Full optimization history + config
│   ├── protegi_TIMESTAMP.txt   # Best prompt + validation accuracy
│   ├── opro_TIMESTAMP.json     # OPRO results
│   └── opro_prompts_200it.*    # OPRO prompt datasets (CSV/JSON)
│
└── src/                         # Core implementation (ProTeGi, OPRO, evaluators)
```

**Key Directories:**
- `datasets/`: Static input data (never modified by experiments)
- `hbbops/results/`: HbBoPs experiment outputs and visualizations
- `results/`: ProTeGi/OPRO optimization results
- `visualize/`: Analysis and visualization scripts

**Best Practices:**
1. **Input data** goes in `datasets/` (version controlled, read-only)
2. **Experiment outputs** go in `hbbops/results/` or `results/` (gitignored large files)
3. **Visualization scripts** go in `visualize/` with descriptive names
4. **Generated plots** saved alongside experiment results for context

## HbBoPs (Hyperband Bayesian Optimization for Prompts)

HbBoPs is a multi-fidelity optimization algorithm that efficiently searches through large prompt spaces (instructions × exemplars) using adaptive resource allocation.

### Running HbBoPs

**Basic usage:**
```bash
# Run 10×25 grid search (10 instructions, 25 exemplars)
uv run python hbbops/run_hbbops.py \
    --instructions datasets/hbbops/instructions_10.txt \
    --exemplars datasets/hbbops/examples_25.txt \
    --output-dir hbbops/results

# Run full 25×25 grid search
uv run python hbbops/run_hbbops.py \
    --instructions datasets/hbbops/instructions_25.txt \
    --exemplars datasets/hbbops/examples_25.txt
```

**Key parameters:**
- `--bmin`: Minimum validation instances (default: 10)
- `--eta`: Halving parameter for Hyperband (default: 2.0)
- `--backend`: LLM backend (`vllm`, `transformers`, `claude`)
- `--model`: Task model to use
- `--use-test-set`: Use test set for evaluation (for GT comparison)
- `--ground-truth`: Path to ground truth JSONL for comparison

### HbBoPs Output Files

**JSON results (`hbbops/results/*.json`):**
```json
{
  "ground_truth_path": "...",
  "instruction_mapping": [8, 20, 15, ...],  // Selected instruction IDs
  "exemplar_mapping": [0, 1, 2, ...],        // Selected exemplar IDs
  "total_prompts_in_grid": 250,
  "llm_calls": {
    "hbbops_total": 67912,
    "gt_total": 329750,
    "efficiency_ratio": 4.86
  },
  "all_evaluated_prompts": [
    {
      "sel_inst": 0, "sel_ex": 7,
      "orig_inst": 8, "orig_ex": 7,
      "max_fidelity": 1319,
      "hbbops_error": 0.1145,
      "gt_error": 0.1304,
      "diff_pp": -1.59
    },
    ...
  ]
}
```

**Text results (`hbbops/results/*.txt`):**
- Best prompts at each fidelity level
- Error rates and rankings
- Summary statistics

### Visualizing HbBoPs Results

**Compare different grid sizes:**
```bash
# Generate comparison plot (10×25 vs 25×25)
python3 visualize/compare_grid_sizes.py
```

**Output:** `hbbops/results/grid_size_comparison.png` with:
1. **Accuracy Difference by Fidelity**: HbBoPs vs Ground Truth error (in pp)
2. **Computational Efficiency**: LLM calls comparison (11.3x vs 4.9x)
3. **Rank Correlation**: Spearman & Kendall with ground truth
4. **Top-K Overlap**: Agreement on best prompts at various K

**Key Insights from Grid Comparison:**
- Larger grids (25×25) achieve **2.3× better efficiency** (11.3x vs 4.9x)
- Rank correlation improves with grid size (Spearman 0.961 vs 0.945)
- Top-K overlap is consistently higher for larger grids (90% vs 80% at K=5)
- Similar number of HbBoPs calls (~70K) but vastly different GT coverage

### HbBoPs Algorithm Overview

**Multi-fidelity approach:**
1. Start with many prompts at low fidelity (few validation examples)
2. Progressively eliminate poor performers
3. Allocate more resources (higher fidelity) to promising prompts
4. Final evaluation at max fidelity (1,319 examples for ToS dataset)

**Efficiency gains:**
- Avoids full evaluation of all prompts
- Focuses compute on most promising candidates
- 5-11× fewer LLM calls vs. exhaustive grid search
- Quality: High rank correlation with ground truth (ρ > 0.94)

## Important Constraints

**API Keys:**
- Claude models require `ANTHROPIC_API_KEY` in `.env` file
- Copy `.env.example` to `.env` and add your API key
- Local models (Qwen, Llama, Saul) don't require API keys

**Memory Management:**
- Small models (3B-7B) recommended for 16GB RAM systems
- Use `--device mps` on Apple Silicon, `--torch-dtype bfloat16`
- vLLM requires CUDA GPU
- Claude API models have no local memory requirements

**Answer Format:**
- Models should output final answer as `final_answer: NUMBER` (preferred) or `#### NUMBER`
- Evaluator also handles: `\boxed{NUMBER}` and fallback to last number in output
- Numerical comparison with tolerance for floating point values

**Chat Templates:**
- Both LLM clients auto-detect Instruct models and apply chat templates
- Raw prompts are wrapped in user/assistant format for Instruct models
- Non-Instruct models use raw text

**Batch Generation:**
- Transformers: uses left padding for causal LM (ensures prompts end at same position)
- Low temperature (<0.3) triggers greedy decoding to avoid numerical instability
- vLLM: handles batching efficiently with repetition penalty

## Common Issues

**Small models (<3B):**
- Often struggle with meta-prompt generation (gradients, critiques)
- May fail at math reasoning
- Script warns when using <2B models

**Prompt artifacts:**
- LLMs sometimes generate meta-text instead of pure instruction
- ProTeGi's `apply_gradient` includes extensive cleaning logic
- If prompts look malformed, check `apply_gradient` post-processing

**Evaluation issues:**
- Evaluator uses numerical tolerance for floating point comparisons
- Debug with `--debug` flag to see extraction process and exact matching logic
