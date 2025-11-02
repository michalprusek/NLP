# ProTeGi Meta-Prompts Documentation

## Overview

This implementation uses **significantly enhanced meta-prompts** compared to the original paper (Appendix 1.1). The enhancements improve prompt quality and reduce LLM artifacts, but deviate from the paper's minimal prompts.

## Differences from Paper

### 1. Gradient Generation Prompt

**Paper Version (Appendix 1.1):**
```
I'm trying to write a zero-shot classifier prompt.
My current prompt is: "{prompt}"
But this prompt gets the following examples wrong: {error_string}
give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
Wrap each reason with <START> and <END>
```

**Our Implementation (`gradient.txt`):**
- 74 lines with structured format
- Explains Math-Verify 3-step evaluation system (extraction → parsing → verification)
- Requests 2-4 issues with root causes and specific improvements
- Structured output: ISSUE / Root cause / Suggested improvements / GLOBAL NOTES
- Emphasizes systematic issues over individual examples

**Why enhanced:**
- Provides context about the evaluation system to generate better critiques
- Structured output makes parsing more reliable
- Focuses on actionable, testable improvements
- Reduces vague feedback like "the prompt is unclear"

### 2. Edit/Application Prompt

**Paper Version (Appendix 1.1):**
```
I'm trying to write a zero-shot classifier.
My current prompt is: "{prompt}"
But it gets the following examples wrong: {error_str}
Based on these examples the problem with this prompt is that {gradient}
Based on the above information, I wrote {steps_per_gradient} different improved prompts.
Each prompt is wrapped with <START> and <END>.
The {steps_per_gradient} new prompts are:
```

**Our Implementation (`edit.txt`):**
- 26 lines with HARD RULES
- Brevity constraint: "MAX 3 sentences OR less than 150 words"
- Explicit output format guidance: "#### NUMBER" for answers
- Anti-artifact rules: no preambles, quotes, meta-text, code fences
- Single prompt generation (not multiple wrapped in <START>/<END>)

**Why enhanced:**
- Paper version often generates verbose, rambling prompts
- LLMs tend to add preambles ("Here's the improved prompt:") without constraints
- Brevity constraint prevents prompt bloat over iterations
- Single prompt generation + separate paraphrasing step (cleaner pipeline)

### 3. Paraphrasing Prompt

**Paper Version (Appendix 1.1):**
```
Generate a variation of the following instruction while keeping the semantic meaning.
Input: {prompt_instruction}
Output:
```

**Our Implementation (`protegi.py:680-684`):**
Same as paper ✓ (uses Zhou et al. 2022 prompt)

---

## Task-Specific Prompts

### GSM8K (Math Word Problems)

**Files:**
- `src/prompts/gsm8k/gradient.txt` - Gradient generation with Math-Verify explanation
- `src/prompts/gsm8k/edit.txt` - Edit prompt with brevity constraints

**Key enhancements:**
- Explains Math-Verify's 3-step evaluation (extraction priorities, parsing, verification)
- Emphasizes that Math-Verify is more forgiving than strict string matching
- Guides models to provide clear final answers in "#### NUMBER" format

### Claudette (Binary Classification)

**Files:**
- `src/prompts/claudette_binary/gradient.txt`
- `src/prompts/claudette_binary/edit.txt`

**Key differences:**
- Uses binary classification metrics (F1, precision, recall for positive class)
- Explains class imbalance (~90% fair, ~10% unfair clauses)

### Claudette (Multi-label Classification)

**Files:**
- `src/prompts/claudette/gradient.txt`
- `src/prompts/claudette/edit.txt`

**Key differences:**
- Uses multi-label metrics (micro-F1, macro-F1)
- Explains 8 unfairness categories

---

## Rationale for Not Using Paper Prompts

### Problem 1: LLM Artifacts
Paper's minimal prompts lead to outputs like:
```
Here's the improved prompt: "Solve the problem step by step. ..."

This new prompt addresses the issues by:
1. Being more specific
2. Adding structure
...
```

Our enhanced prompts with "OUTPUT ONLY THE IMPROVED PROMPT" rules eliminate this.

### Problem 2: Prompt Bloat
Without length constraints, prompts grow exponentially:
- Iteration 1: 50 words
- Iteration 2: 150 words (LLM adds details)
- Iteration 3: 400 words (more elaboration)
- Iteration 6: 1200+ words (unusable)

Our "MAX 3 sentences OR 150 words" prevents this.

### Problem 3: Vague Critiques
Paper prompt generates feedback like:
```
The prompt is unclear and doesn't specify the output format.
```

Our structured format enforces:
```
ISSUE 1: Missing output format specification
- Root cause: Model doesn't know to use #### NUMBER format
- Suggested improvements:
  * Add "Provide final answer as #### NUMBER"
  * Show example output format
```

---

## Trade-offs

### Advantages of Enhanced Prompts
✅ Higher quality prompts with fewer artifacts
✅ Consistent formatting across iterations
✅ More actionable, specific feedback
✅ Better alignment with evaluation system (Math-Verify)

### Disadvantages of Enhanced Prompts
❌ Not reproducible from paper alone (requires our prompt files)
❌ More complex / harder to understand
❌ Longer meta-prompts → higher API costs
❌ May be tuned to specific LLM behaviors (GPT-3.5/4)

---

## Using Paper-Original Prompts

If you want to replicate paper exactly, create:

**`src/prompts/gsm8k/gradient_paper.txt`:**
```
I'm trying to write a zero-shot classifier prompt.
My current prompt is: "{prompt}"
But this prompt gets the following examples wrong:
{results}

give 4 reasons why the prompt could have gotten these examples wrong.
Wrap each reason with <START> and <END>
```

**`src/prompts/gsm8k/edit_paper.txt`:**
```
I'm trying to write a zero-shot classifier.
My current prompt is: "{prompt}"
But it gets the following examples wrong: {results}
Based on these examples the problem with this prompt is that {gradient}

Based on the above information, I wrote 1 improved prompt.
The improved prompt is:
```

Then modify `protegi.py:335-336` to load these files.

---

## References

- **Original paper:** Pryzant et al. 2023, "Automatic Prompt Optimization with Gradient Descent and Beam Search"
- **Paper prompts:** Appendix 1.1 (page 10)
- **Zhou et al. 2022:** "Large Language Models are Human-Level Prompt Engineers" (paraphrasing prompt)
