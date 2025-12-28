# COWBOYS Vec2Text - Roadmap

### A. Weak Baseline Models

Our code (apparently) only compares our method with grid search or HbBoPs. That's not enough. We must beat the current state-of-the-art in generative optimization:

- [ ] **APE (Automatic Prompt Engineer)**: Instruction generation using LLM
- [ ] **OPRO (Optimization by PROmpting)**: Google DeepMind method that iteratively improves prompts using meta-prompts
- [ ] **PromptBreeder**: Evolutionary algorithm
- [ ] **ProTeGi (Prompt Optimization with Textual Gradients)**: Gradient-based method that uses textual gradients to improve prompts

- We must prove that our method finds prompts that pure language methods cannot find

### B. Benchmarks (GSM8K Is Not Enough)

The standard in the field is to test on **Big-Bench Hard (BBH)** or at least on a mix of tasks:

- [ ] **Arithmetic**: GSM8K, SVAMP
- [ ] **Reasoning**: BBH (Date understanding, Logical deduction)
- [ ] **Instruction Following**: Some NLP tasks (Summarization, Translation)

- If our method only works on math, it will be considered "domain-specific"
- We must prove that our method works on a variety of tasks to show that VAE latent space works universally

### C. "Vec2Text Hallucination" Problem

**This is the biggest risk of our method.**

- **Hypothesis**: VAE finds a point `z*` in latent space that has great GP score
- **Reality**: When we decode `z*` via Vec2Text, we get "word salad" (nonsensical text), or text that looks good but lost the semantic meaning of the instruction

**Required analysis:**
- [ ] Add **semantic drift** metric - how much does the optimized prompt semantically differ from the original?
- [ ] If it differs too much, the method is "hacking" the benchmark rather than optimizing the instruction

### D. Ablation Studies (Component Breakdown)

We must prove that each part of our complex system makes sense.

- [ ] **Full Method** (COWBOYS + TuRBO + Retrain)
- [ ] **No TuRBO**: Pure pCN MCMC only (shows that without Trust Region it diverges)
- [ ] **No Retrain**: VAE fixed throughout (shows that latent space adaptation is necessary)
- [ ] **No Decoded Embeddings**: Training GP on inputs (shows that our "Alignment Fix" is key)

### Roadmap

#### Phase 1: Baseline Implementation
- [ ] Implement OPRO (simple - just a loop of LLM calls)
- [ ] Implement APE baseline
- [ ] Set up fair comparison setup

#### Phase 2: Scale Evaluation
- [ ] Run on 5-10 tasks from Big-Bench Hard
- [ ] Add SVAMP dataset
- [ ] Test on reasoning tasks

#### Phase 3: Cost Analysis
- [ ] Measure **token cost** for each method
- [ ] Measure **wall-clock time**
- [ ] Count number of evaluations on target model

**Key argument:**
> "OPRO needs 1000 calls to an expensive LLM. We need 1000 steps in cheap latent space and only 20 calls to expensive LLM for verification."

- **If this holds, we win.**

#### Phase 4: Ablation & Analysis
- [ ] Prepare ablation table
- [ ] Semantic drift analysis
- [ ] Vec2Text quality vs. optimization performance correlation

---

## 4. Technical TODO

### Immediate Fixes
- [x] Fix `patience` argument in `add_observation_and_retrain`
- [x] Add VAE quality metrics to logs

### Code Quality
- [ ] Unified logging (no truncated prompts)
- [ ] Checkpoint saving for long runs
- [ ] Reproducibility (seed handling)

### Experiments Infrastructure
- [ ] Multi-task evaluation script
- [ ] Results aggregation and visualization
- [ ] Cost tracking (tokens, time, GPU hours)
