"""
ProTeGi: Prompt Optimization with Textual Gradients

Based on "Automatic Prompt Optimization with Gradient Descent and Beam Search"
https://arxiv.org/abs/2305.03495

Key differences from OPRO:
1. Uses "textual gradients" - natural language feedback from errors
2. Beam search with UCB bandit selection for exploration-exploitation
3. Monte Carlo paraphrasing for local search exploration
"""
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import random
import math
import json
import os
import re

if TYPE_CHECKING:
    from shared.incremental_saver import IncrementalPromptSaver


# Load prompt templates
PROMPTS_DIR = Path(__file__).parent / 'prompts'
GRADIENT_PROMPT_TEMPLATE = (PROMPTS_DIR / 'protegi_gradient.txt').read_text(encoding='utf-8')
EDIT_PROMPT_TEMPLATE = (PROMPTS_DIR / 'protegi_edit.txt').read_text(encoding='utf-8')
PARAPHRASE_PROMPT_TEMPLATE = (PROMPTS_DIR / 'protegi_paraphrase.txt').read_text(encoding='utf-8')


@dataclass
class ScoredPrompt:
    """A prompt with its evaluation score and UCB statistics"""
    prompt: str
    score: float
    visits: int = 0
    total_reward: float = 0.0

    def __post_init__(self):
        """Validate invariants"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.visits < 0:
            raise ValueError(f"visits must be non-negative, got {self.visits}")
        if self.total_reward < 0:
            raise ValueError(f"total_reward must be non-negative, got {self.total_reward}")

    def ucb_score(self, total_visits: int, c: float = 2.0) -> float:
        """Calculate UCB score for bandit selection"""
        if self.visits == 0:
            return float('inf')  # Explore unvisited first
        mean = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(total_visits + 1) / self.visits)
        return mean + exploration

    def __repr__(self):
        # Full prompt logging per CLAUDE.md - never truncate
        return f"Score: {self.score:.1%} | Visits: {self.visits} | Prompt:\n{self.prompt}"


@dataclass
class ProTeGiIteration:
    """Record of one ProTeGi optimization step"""
    iteration: int
    parent_prompt: str
    parent_score: float
    gradients: List[str]
    edited_prompts: List[str]
    paraphrased_prompts: List[str]
    candidates_evaluated: int
    best_score: float
    beam_top3: List[Dict[str, Any]]
    meta_calls: int
    task_calls: int


class ProTeGiOptimizer:
    """
    ProTeGi: Prompt Optimization with Textual Gradients

    Algorithm:
    1. Initialize beam with initial prompt(s)
    2. For each optimization step:
       a. Select parent using UCB bandit
       b. Evaluate parent, collect errors
       c. Generate gradients from errors (textual feedback)
       d. Apply gradients to edit prompt
       e. Paraphrase edits for MC exploration
       f. Evaluate candidates
       g. Update beam with top-b
    3. Return best prompt from beam
    """

    def __init__(
        self,
        task_llm_client,
        evaluator,
        meta_llm_client=None,
        # ProTeGi parameters (from paper)
        beam_size: int = 4,              # b=4 in paper
        num_steps: int = 6,              # r=6 optimization steps
        gradients_per_group: int = 4,    # m=4 gradients per error group
        errors_per_group: int = 4,       # Group 4 errors at a time
        mc_samples: int = 2,             # p=2 paraphrases per edited prompt
        max_successors: int = 8,         # Max candidates before selection
        minibatch_size: int = 64,        # Paper uses 64
        # LLM generation params
        task_max_tokens: int = 2048,
        meta_max_tokens: int = 800,
        # UCB parameters
        ucb_c: float = 2.0,              # Exploration parameter
        # Budget tracking
        total_budget: int = None,        # Total task LLM eval budget
        # Benchmarking parameters
        max_prompts: Optional[int] = None,  # Max prompts to evaluate
        incremental_saver: Optional["IncrementalPromptSaver"] = None,  # Save prompts incrementally
    ):
        self.task_llm = task_llm_client
        self.meta_llm = meta_llm_client if meta_llm_client is not None else task_llm_client
        self.evaluator = evaluator

        # ProTeGi-specific params
        self.beam_size = beam_size
        self.num_steps = num_steps
        self.gradients_per_group = gradients_per_group
        self.errors_per_group = errors_per_group
        self.mc_samples = mc_samples
        self.max_successors = max_successors
        self.minibatch_size = minibatch_size

        self.task_max_tokens = task_max_tokens
        self.meta_max_tokens = meta_max_tokens
        self.ucb_c = ucb_c
        self.total_budget = total_budget
        self.max_prompts = max_prompts
        self.incremental_saver = incremental_saver

        # State
        self.beam: List[ScoredPrompt] = []
        self.history: List[ProTeGiIteration] = []

        # Budget tracking (separate for transparency)
        self.task_budget_used = 0       # Task LLM evaluations (counts against budget)
        self.meta_calls_gradient = 0    # Gradient generation calls
        self.meta_calls_edit = 0        # Edit prompt calls
        self.meta_calls_paraphrase = 0  # Paraphrase calls
        self.prompts_evaluated = 0      # Track number of prompts evaluated (for max_prompts)
        self._current_iteration = 0     # Track current iteration for saver

        # Create fixed evaluation set (same as OPRO - consistent evaluation)
        dataset_size = len(evaluator)
        eval_size = min(minibatch_size, dataset_size)
        eval_indices = random.sample(range(dataset_size), eval_size)

        self.fixed_eval_set = []
        for idx in eval_indices:
            batch = evaluator.get_batch(idx, 1)
            if batch:
                self.fixed_eval_set.append(batch[0])

        print(f"Fixed evaluation set: {len(self.fixed_eval_set)} examples ({100*len(self.fixed_eval_set)/dataset_size:.1f}%)")

    # =========================================================================
    # BUDGET MANAGEMENT
    # =========================================================================

    def _is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        if self.total_budget is None:
            return False
        return self.task_budget_used >= self.total_budget

    def _can_afford_evaluation(self) -> bool:
        """Check if we can afford one more evaluation."""
        if self.total_budget is None:
            return True
        return (self.task_budget_used + len(self.fixed_eval_set)) <= self.total_budget

    def _is_max_prompts_reached(self) -> bool:
        """Check if max prompts limit is reached."""
        if self.max_prompts is None:
            return False
        return self.prompts_evaluated >= self.max_prompts

    def _can_evaluate_more_prompts(self) -> bool:
        """Check if we can evaluate more prompts (budget and max_prompts)."""
        if not self._can_afford_evaluation():
            return False
        if self._is_max_prompts_reached():
            return False
        return True

    @property
    def total_meta_calls(self) -> int:
        """Total meta-LLM calls made"""
        return self.meta_calls_gradient + self.meta_calls_edit + self.meta_calls_paraphrase

    # =========================================================================
    # PROMPT VALIDATION AND CLEANING
    # =========================================================================

    def _is_valid_prompt(self, prompt: str) -> bool:
        """
        Validate that a generated prompt is reasonable.

        Rejects:
        - Empty or too short prompts (< 20 chars)
        - Template artifacts (Input:, Output:, <START>, etc.)
        - Prompts with too few alphabetic characters
        """
        if not prompt:
            return False

        stripped = prompt.strip()
        if len(stripped) < 20:
            return False

        # Reject template artifacts
        invalid_prefixes = [
            'Input:', 'Output:', '<START>', '<END>',
            'Variation:', '```', 'Here\'s', 'Here is'
        ]
        stripped_lower = stripped.lower()
        for prefix in invalid_prefixes:
            if stripped_lower.startswith(prefix.lower()):
                return False

        # Reject if mostly special characters (need at least 10 alphabetic chars)
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if alpha_count < 10:
            return False

        return True

    def _clean_meta_commentary(self, text: str) -> str:
        """
        Remove common LLM meta-commentary prefixes from generated prompts.

        LLMs often add preambles like "Certainly! Here's an improved version..."
        This function strips those to get cleaner prompts.
        """
        if not text:
            return text

        # Common meta-commentary patterns to remove
        patterns_to_remove = [
            r'^Certainly!?\s*',
            r'^Sure!?\s*',
            r'^Of course!?\s*',
            r'^Absolutely!?\s*',
            r'^Great!?\s*',
            r'^Here\'s?\s+(a\s+)?(the\s+)?(an\s+)?(improved|revised|updated|refined|new|variation|better)\s+[^:]+:\s*',
            r'^Based on\s+(the\s+)?(your\s+)?(feedback|issues?|problems?|errors?)[^:]*:\s*',
            r'^To\s+(address|fix|improve|solve)[^:]+:\s*',
            r'^Let me\s+[^:]+:\s*',
            r'^I\'ve\s+[^:]+:\s*',
        ]

        cleaned = text.strip()
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def evaluate_prompt(self, prompt: str, is_parent_eval: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate prompt on fixed evaluation set.

        Args:
            prompt: The prompt to evaluate
            is_parent_eval: If True, this is a parent re-evaluation (for fresh errors),
                           not counted towards prompts_evaluated

        Returns:
            score: Accuracy on evaluation set
            results: Detailed results including per-example correctness
        """
        batch = self.fixed_eval_set

        # Track budget
        self.task_budget_used += len(batch)

        # Generate answers using Q_end format (instruction after question)
        # Per CLAUDE.md: Q: {question}\n{instruction}\nA:
        questions = [ex['question'] for ex in batch]
        formatted_prompts = [f"Q: {q}\n{prompt}\nA:" for q in questions]
        outputs = self.task_llm.generate_batch(
            formatted_prompts, temperature=0.0, max_new_tokens=self.task_max_tokens
        )

        # Evaluate
        indices = [ex['idx'] for ex in batch]
        results = self.evaluator.evaluate_batch(outputs, indices)
        score = results['accuracy']

        # Track prompts evaluated (only for non-parent evaluations)
        # Parent re-evaluations are for getting fresh errors, not new prompts
        if not is_parent_eval:
            self.prompts_evaluated += 1

            # Save to incremental saver if provided
            if self.incremental_saver is not None:
                self.incremental_saver.save_prompt(
                    prompt=prompt,
                    score=score,
                    iteration=self._current_iteration,
                )

        return score, results

    def _get_failed_examples(self, results: Dict, max_examples: int = 8) -> List[Dict]:
        """Extract failed examples from evaluation results"""
        details = results.get('details', [])
        failed = [d for d in details if not d.get('correct', False)]

        if len(failed) > max_examples:
            failed = random.sample(failed, max_examples)

        return failed

    def _format_error_string(self, failed_examples: List[Dict]) -> str:
        """Format failed examples for gradient prompt"""
        error_strings = []
        for ex in failed_examples:
            error_strings.append(
                f"Question: {ex.get('question', 'N/A')}\n"
                f"Correct Answer: {ex.get('ground_truth', 'N/A')}\n"
                f"Model Output: {ex.get('output', 'N/A')[:200]}\n"
                f"Extracted: {ex.get('predicted', 'N/A')}"
            )
        return "\n\n---\n\n".join(error_strings)

    # =========================================================================
    # GRADIENT GENERATION (Step 1 of expansion)
    # =========================================================================

    def generate_gradients(
        self,
        prompt: str,
        failed_examples: List[Dict],
        verbose: bool = False
    ) -> List[str]:
        """
        Generate textual gradients from error analysis.

        Groups errors and generates m=4 gradients per group.
        """
        gradients = []

        if not failed_examples:
            return gradients

        # Group errors (4 at a time per paper)
        for i in range(0, len(failed_examples), self.errors_per_group):
            error_group = failed_examples[i:i + self.errors_per_group]
            error_string = self._format_error_string(error_group)

            # Format gradient prompt
            gradient_prompt = GRADIENT_PROMPT_TEMPLATE.format(
                prompt=prompt if prompt else "(empty prompt)",
                error_string=error_string,
                num_feedbacks=self.gradients_per_group
            )

            if verbose:
                print(f"\n[Gradient] Generating {self.gradients_per_group} gradients for {len(error_group)} errors...")

            # Generate gradients with error handling
            try:
                response = self.meta_llm.generate(
                    gradient_prompt,
                    temperature=0.7,
                    max_new_tokens=self.meta_max_tokens
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate textual gradients from meta-LLM.\n"
                    f"This step analyzes errors to suggest prompt improvements.\n"
                    f"Error: {type(e).__name__}: {e}\n"
                    f"Suggestions:\n"
                    f"  - Check network connectivity\n"
                    f"  - Verify API key/credentials for meta-model\n"
                    f"  - Try reducing --gradients to lower API usage"
                ) from e
            self.meta_calls_gradient += 1

            # Parse gradients from response (between <START> and <END>)
            parsed_gradients = self._parse_delimited_responses(response, '<START>', '<END>')

            if verbose:
                print(f"[Gradient] Parsed {len(parsed_gradients)} gradients")

            gradients.extend(parsed_gradients)

        return gradients

    def _parse_delimited_responses(
        self,
        response: str,
        start_tag: str,
        end_tag: str
    ) -> List[str]:
        """Parse responses between start and end tags"""
        results = []
        pattern = re.escape(start_tag) + r'(.*?)' + re.escape(end_tag)
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            text = match.strip()
            if text:
                results.append(text)
        return results

    # =========================================================================
    # EDIT APPLICATION (Step 2 of expansion)
    # =========================================================================

    def apply_gradients(
        self,
        prompt: str,
        gradients: List[str],
        failed_examples: List[Dict],
        verbose: bool = False
    ) -> List[str]:
        """
        Apply gradients to generate edited prompts.

        Each gradient produces one or more edited prompts.
        Extracts ONLY content within <PROMPT></PROMPT> tags.
        """
        edited = []
        error_str = self._format_error_string(failed_examples[:4])

        for gradient in gradients:
            # Format edit prompt
            edit_prompt = EDIT_PROMPT_TEMPLATE.format(
                prompt=prompt if prompt else "(empty prompt)",
                error_str=error_str,
                gradient=gradient,
                steps_per_gradient=1  # Generate 1 edit per gradient
            )

            if verbose:
                print(f"\n[Edit] Applying gradient:\n{gradient}")

            # Generate edited prompt with error handling
            try:
                response = self.meta_llm.generate(
                    edit_prompt,
                    temperature=0.7,
                    max_new_tokens=500  # Increased for complete prompts
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply gradient edit via meta-LLM.\n"
                    f"This step generates improved prompts based on feedback.\n"
                    f"Error: {type(e).__name__}: {e}\n"
                    f"Gradient being applied:\n{gradient}"
                ) from e
            self.meta_calls_edit += 1

            # Parse edited prompts from <PROMPT> tags only
            parsed_edits = self._parse_delimited_responses(response, '<PROMPT>', '</PROMPT>')

            # Fallback: try old-style tags for backwards compatibility
            if not parsed_edits:
                parsed_edits = self._parse_delimited_responses(response, '<START>', '<END>')

            valid_count = 0
            for edit in parsed_edits:
                # Validate the extracted prompt
                if edit and edit != prompt and edit not in edited and self._is_valid_prompt(edit):
                    edited.append(edit)
                    valid_count += 1

            if verbose:
                print(f"[Edit] Generated {valid_count} valid edits (parsed {len(parsed_edits)})")

        return edited

    # =========================================================================
    # MONTE CARLO PARAPHRASING (Step 3 of expansion)
    # =========================================================================

    def paraphrase_prompts(
        self,
        prompts: List[str],
        verbose: bool = False
    ) -> List[str]:
        """
        Generate p=2 paraphrases for each edited prompt.

        Explores local search space around edited prompts.
        Extracts ONLY content within <PROMPT></PROMPT> tags.
        """
        paraphrased = []

        for prompt in prompts:
            for _ in range(self.mc_samples):
                # Format paraphrase prompt
                para_prompt = PARAPHRASE_PROMPT_TEMPLATE.format(
                    prompt_instruction=prompt
                )

                # Generate paraphrase with error handling
                try:
                    response = self.meta_llm.generate(
                        para_prompt,
                        temperature=0.9,  # High temperature for diversity
                        max_new_tokens=500  # Increased for complete prompts
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to generate paraphrase via meta-LLM.\n"
                        f"This step explores variations of edited prompts.\n"
                        f"Error: {type(e).__name__}: {e}\n"
                        f"Original prompt being paraphrased:\n{prompt}"
                    ) from e
                self.meta_calls_paraphrase += 1

                # Extract from <PROMPT> tags only
                parsed = self._parse_delimited_responses(response, '<PROMPT>', '</PROMPT>')

                # Fallback: try old-style tags
                if not parsed:
                    parsed = self._parse_delimited_responses(response, '<START>', '<END>')

                for para in parsed:
                    # Validate and add
                    if para and para != prompt and para not in paraphrased and self._is_valid_prompt(para):
                        paraphrased.append(para)

        if verbose:
            print(f"[Paraphrase] Generated {len(paraphrased)} valid paraphrases from {len(prompts)} prompts")

        return paraphrased

    # =========================================================================
    # UCB BANDIT SELECTION
    # =========================================================================

    def select_parent_ucb(self) -> Optional[ScoredPrompt]:
        """
        Select parent prompt from beam using UCB bandit.

        UCB = mean_reward + c * sqrt(ln(N) / n_i)
        """
        if not self.beam:
            return None

        total_visits = sum(p.visits for p in self.beam) + 1

        best_ucb = -float('inf')
        best_prompt = None

        for p in self.beam:
            ucb = p.ucb_score(total_visits, self.ucb_c)
            if ucb > best_ucb:
                best_ucb = ucb
                best_prompt = p

        return best_prompt

    def update_ucb_stats(self, prompt: ScoredPrompt, reward: float):
        """Update UCB statistics after evaluation"""
        prompt.visits += 1
        prompt.total_reward += reward

    # =========================================================================
    # BEAM MANAGEMENT
    # =========================================================================

    def update_beam(self, new_candidates: List[ScoredPrompt]):
        """
        Update beam with new candidates, keep top beam_size.

        Paper: Select top-b prompts by score after each step.
        """
        all_prompts = self.beam + new_candidates

        # Deduplicate (keep highest scoring version)
        seen = {}
        for p in all_prompts:
            if p.prompt not in seen or p.score > seen[p.prompt].score:
                seen[p.prompt] = p

        # Sort by score, keep top beam_size
        sorted_prompts = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        self.beam = sorted_prompts[:self.beam_size]

    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================

    def optimize(
        self,
        initial_prompts: List[str] = None,
        verbose: bool = True,
        save_details: bool = False,
        output_dir: str = None
    ) -> Tuple[str, List[Dict]]:
        """
        Run ProTeGi optimization.

        Algorithm (from paper):
        1. Initialize beam with initial prompts
        2. For each step:
           a. Select parent via UCB
           b. Evaluate parent, get errors
           c. Generate gradients from errors
           d. Apply gradients to get edits
           e. Paraphrase edits for MC exploration
           f. Evaluate all candidates
           g. Update beam with top-b
        3. Return best prompt
        """
        if initial_prompts is None:
            initial_prompts = [""]  # Start with empty prompt

        if verbose:
            print(f"\n{'='*60}")
            print("ProTeGi: Prompt Optimization with Textual Gradients")
            print(f"{'='*60}")
            print(f"Beam size: {self.beam_size}")
            print(f"Optimization steps: {self.num_steps}")
            print(f"Gradients per group: {self.gradients_per_group}")
            print(f"MC samples: {self.mc_samples}")
            print(f"Minibatch size: {self.minibatch_size}")
            if self.total_budget:
                print(f"Budget: {self.total_budget} task LLM evaluations")
            print(f"{'='*60}\n")

        if save_details and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # =====================================================================
        # INITIALIZATION: Evaluate initial prompts
        # =====================================================================
        self._current_iteration = 0

        if verbose:
            print("Evaluating initial prompts...\n")

        for prompt in initial_prompts:
            if not self._can_evaluate_more_prompts():
                if verbose:
                    if self._is_max_prompts_reached():
                        print(f"Max prompts reached ({self.prompts_evaluated}/{self.max_prompts}). Stopping initial evaluation.")
                    else:
                        print(f"Budget exhausted. Stopping initial evaluation.")
                break

            score, results = self.evaluate_prompt(prompt)
            # Initialize UCB stats with initial evaluation (visits=1, total_reward=score)
            scored_prompt = ScoredPrompt(prompt=prompt, score=score, visits=1, total_reward=score)
            self.beam.append(scored_prompt)

            if verbose:
                budget_str = f" [budget: {self.task_budget_used}/{self.total_budget}]" if self.total_budget else ""
                prompts_str = f" [prompts: {self.prompts_evaluated}/{self.max_prompts}]" if self.max_prompts else ""
                # Full prompt logging per CLAUDE.md
                print(f"Initial: Score {score:.1%}{budget_str}{prompts_str}\nPrompt:\n{prompt if prompt else '(empty)'}")

        # Sort beam by score
        self.beam.sort(key=lambda x: x.score, reverse=True)
        self.beam = self.beam[:self.beam_size]

        # =====================================================================
        # MAIN OPTIMIZATION LOOP
        # =====================================================================
        # If max_prompts is set, run until we reach it (ignore num_steps limit)
        # Otherwise, run for num_steps iterations
        max_steps = self.num_steps if self.max_prompts is None else 10000

        for step in range(max_steps):
            self._current_iteration = step + 1

            if self._is_budget_exhausted():
                if verbose:
                    print(f"\nBudget exhausted ({self.task_budget_used}/{self.total_budget}). Stopping.")
                break

            if self._is_max_prompts_reached():
                if verbose:
                    print(f"\nMax prompts reached ({self.prompts_evaluated}/{self.max_prompts}). Stopping.")
                break

            # Also stop at num_steps if max_prompts not set
            if self.max_prompts is None and step >= self.num_steps:
                if verbose:
                    print(f"\nCompleted {self.num_steps} steps. Stopping.")
                break

            if verbose:
                print(f"\n{'='*60}")
                step_display = f"Step {step + 1}" + (f"/{self.num_steps}" if self.max_prompts is None else "")
                print(step_display)
                if self.total_budget:
                    print(f"Budget: {self.task_budget_used}/{self.total_budget} | Meta calls: {self.total_meta_calls}")
                if self.max_prompts:
                    print(f"Prompts evaluated: {self.prompts_evaluated}/{self.max_prompts}")
                print(f"{'='*60}\n")

            step_meta_calls_start = self.total_meta_calls
            step_task_calls_start = self.task_budget_used

            # -----------------------------------------------------------------
            # SELECT PARENT using UCB
            # -----------------------------------------------------------------
            parent = self.select_parent_ucb()
            if parent is None:
                if verbose:
                    print("No parent available. Stopping.")
                break

            if verbose:
                # Full prompt logging per CLAUDE.md
                print(f"Selected parent (UCB) - visits: {parent.visits}, score: {parent.score:.1%}")
                print(f"Prompt:\n{parent.prompt if parent.prompt else '(empty)'}")

            # -----------------------------------------------------------------
            # EVALUATE PARENT to get fresh errors
            # -----------------------------------------------------------------
            if not self._can_afford_evaluation():
                if verbose:
                    print("Budget exhausted before parent evaluation.")
                break

            # Note: is_parent_eval=True means this doesn't count towards prompts_evaluated
            # because it's re-evaluating an existing prompt, not a new one
            parent_score, results = self.evaluate_prompt(parent.prompt, is_parent_eval=True)
            self.update_ucb_stats(parent, parent_score)

            if verbose:
                print(f"Parent score: {parent_score:.1%}")

            # Get failed examples
            failed = self._get_failed_examples(results)
            if not failed:
                if verbose:
                    print("No errors to learn from (perfect score). Moving to next step.")
                continue

            if verbose:
                print(f"Failed examples: {len(failed)}")

            # -----------------------------------------------------------------
            # EXPANSION STEP
            # -----------------------------------------------------------------

            # 1. Generate gradients
            if verbose:
                print(f"\nGenerating gradients...")
            gradients = self.generate_gradients(parent.prompt, failed, verbose=verbose)
            if verbose:
                print(f"Generated {len(gradients)} gradients")

            # 2. Apply gradients (edit prompts)
            if verbose:
                print(f"\nApplying gradients...")
            edited = self.apply_gradients(parent.prompt, gradients, failed, verbose=verbose)
            if verbose:
                print(f"Generated {len(edited)} edited prompts")

            # 3. Monte Carlo paraphrasing
            if verbose:
                print(f"\nParaphrasing...")
            paraphrased = self.paraphrase_prompts(edited, verbose=verbose)
            if verbose:
                print(f"Generated {len(paraphrased)} paraphrases")

            # Combine all candidates
            all_candidates = list(set(edited + paraphrased))

            # Limit to max_successors
            if len(all_candidates) > self.max_successors:
                all_candidates = random.sample(all_candidates, self.max_successors)

            if verbose:
                print(f"\nTotal candidates to evaluate: {len(all_candidates)}")

            # -----------------------------------------------------------------
            # EVALUATE CANDIDATES
            # -----------------------------------------------------------------
            scored_candidates = []
            for i, candidate in enumerate(all_candidates):
                if not self._can_evaluate_more_prompts():
                    if verbose:
                        if self._is_max_prompts_reached():
                            print(f"Max prompts reached. Evaluated {i}/{len(all_candidates)} candidates.")
                        else:
                            print(f"Budget exhausted. Evaluated {i}/{len(all_candidates)} candidates.")
                    break

                score, _ = self.evaluate_prompt(candidate)
                scored_candidates.append(ScoredPrompt(prompt=candidate, score=score))

                if verbose:
                    budget_str = f" [budget: {self.task_budget_used}/{self.total_budget}]" if self.total_budget else ""
                    prompts_str = f" [prompts: {self.prompts_evaluated}/{self.max_prompts}]" if self.max_prompts else ""
                    print(f"  Candidate {i+1}: Score {score:.1%}{budget_str}{prompts_str}")

            # -----------------------------------------------------------------
            # UPDATE BEAM
            # -----------------------------------------------------------------
            self.update_beam(scored_candidates)

            # Record iteration
            iteration_record = ProTeGiIteration(
                iteration=step,
                parent_prompt=parent.prompt,
                parent_score=parent_score,
                gradients=gradients,
                edited_prompts=edited,
                paraphrased_prompts=paraphrased,
                candidates_evaluated=len(scored_candidates),
                best_score=self.beam[0].score if self.beam else 0.0,
                beam_top3=[
                    {'prompt': p.prompt, 'score': p.score, 'visits': p.visits}
                    for p in self.beam[:3]
                ],
                meta_calls=self.total_meta_calls - step_meta_calls_start,
                task_calls=self.task_budget_used - step_task_calls_start
            )
            self.history.append(iteration_record)

            if verbose:
                print(f"\nBeam after step {step + 1}:")
                for i, sp in enumerate(self.beam[:3]):
                    # Full prompt logging per CLAUDE.md
                    print(f"  {i+1}. Score {sp.score:.1%}")
                    print(f"     Prompt:\n{sp.prompt if sp.prompt else '(empty)'}")

            # Save iteration details
            if save_details and output_dir:
                self._save_iteration_details(output_dir, step, iteration_record)

        # =====================================================================
        # RETURN BEST
        # =====================================================================
        if not self.beam:
            if verbose:
                print("No prompts evaluated. Returning empty.")
            if self.incremental_saver is not None:
                self.incremental_saver.finalize("", 0.0)
            return "", self._format_history()

        best = self.beam[0]
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best prompt (score: {best.score:.1%}):")
            print(f"{best.prompt}")
            print(f"\nTask budget used: {self.task_budget_used}")
            if self.max_prompts:
                print(f"Total prompts evaluated: {self.prompts_evaluated}/{self.max_prompts}")
            print(f"Meta calls: {self.total_meta_calls} (gradient: {self.meta_calls_gradient}, edit: {self.meta_calls_edit}, paraphrase: {self.meta_calls_paraphrase})")
            print(f"{'='*60}\n")

        # Finalize incremental saver
        if self.incremental_saver is not None:
            self.incremental_saver.finalize(best.prompt, best.score)

        return best.prompt, self._format_history()

    def _format_history(self) -> List[Dict]:
        """Format history for JSON serialization"""
        return [
            {
                'iteration': h.iteration,
                'parent_prompt': h.parent_prompt,
                'parent_score': h.parent_score,
                'num_gradients': len(h.gradients),
                'num_edited': len(h.edited_prompts),
                'num_paraphrased': len(h.paraphrased_prompts),
                'candidates_evaluated': h.candidates_evaluated,
                'best_score': h.best_score,
                'beam_top3': h.beam_top3,
                'meta_calls': h.meta_calls,
                'task_calls': h.task_calls
            }
            for h in self.history
        ]

    def _save_iteration_details(self, output_dir: str, step: int, record: ProTeGiIteration):
        """Save detailed iteration data to JSON"""
        filepath = os.path.join(output_dir, f"protegi_step{step:02d}.json")
        data = {
            'iteration': record.iteration,
            'parent_prompt': record.parent_prompt,
            'parent_score': record.parent_score,
            'gradients': record.gradients,
            'edited_prompts': record.edited_prompts,
            'paraphrased_prompts': record.paraphrased_prompts,
            'candidates_evaluated': record.candidates_evaluated,
            'best_score': record.best_score,
            'beam_top3': record.beam_top3,
            'meta_calls': record.meta_calls,
            'task_calls': record.task_calls
        }
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {filepath}")
        except (OSError, IOError) as e:
            print(f"  [WARNING] Failed to save iteration details to {filepath}: {e}")
            # Try fallback to temp directory
            import tempfile
            fallback_path = os.path.join(tempfile.gettempdir(), f"protegi_step{step:02d}.json")
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  Saved to fallback: {fallback_path}")
            except Exception as fallback_e:
                print(f"  [WARNING] Fallback save also failed: {fallback_e}")


# Backwards compatibility alias
ProTeGi = ProTeGiOptimizer
