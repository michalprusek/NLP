"""
ProTeGi: Prompt Optimization with Textual Gradients

Based on "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search"
https://arxiv.org/abs/2305.03495
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json
import re
from difflib import SequenceMatcher


@dataclass
class PromptCandidate:
    """A candidate prompt with its score and history"""
    prompt: str
    score: float = 0.0
    num_evals: int = 0
    sum_scores: float = 0.0
    last_score: float = 0.0  # Track last evaluation score
    history: List[str] = None

    def __post_init__(self):
        if self.history is None:
            self.history = [self.prompt]

    def update_score(self, new_score: float):
        """Update running average of score and track last score"""
        self.num_evals += 1
        self.sum_scores += new_score
        self.score = self.sum_scores / self.num_evals
        self.last_score = new_score  # Track last evaluation

    def ucb_score(self, total_evals: int, c: float = 2.0) -> float:
        """
        Compute Upper Confidence Bound score for exploration/exploitation.

        Args:
            total_evals: Total number of evaluations across all candidates
            c: Exploration constant

        Returns:
            UCB score
        """
        if self.num_evals == 0:
            return float('inf')
        return self.score + c * np.sqrt(np.log(total_evals) / self.num_evals)


class ProTeGi:
    """ProTeGi optimizer using textual gradients and beam search"""

    @staticmethod
    def clean_prompt(prompt: str) -> str:
        """
        Clean LLM-generated prompt by removing meta-text and formatting artifacts.

        Extracts the actual instruction from LLM output that may contain:
        - Preambles like "Here's the improved instruction prompt:"
        - Explanations and meta-commentary
        - Formatting artifacts

        Args:
            prompt: Raw LLM-generated prompt

        Returns:
            Cleaned instruction prompt
        """
        # Remove leading/trailing whitespace
        prompt = prompt.strip()

        # Common preamble patterns to remove
        preamble_patterns = [
            r'^Here\'s the improved instruction prompt:\s*',
            r'^Here is the improved instruction prompt:\s*',
            r'^Improved instruction prompt:\s*',
            r'^The improved instruction prompt is:\s*',
            r'^Improved prompt:\s*',
            r'^Here\'s the improved prompt:\s*',
            r'^Here is the improved prompt:\s*',
            r'^The improved prompt is:\s*',
            r'^Instruction prompt:\s*',
            r'^Instruction:\s*',
        ]

        for pattern in preamble_patterns:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)

        # Remove quotes at the beginning and end
        prompt = prompt.strip('"\'')

        # Split by newlines and look for the actual instruction
        lines = prompt.split('\n')

        # Remove meta-commentary (lines that explain what the prompt does)
        # These often start with "This improved instruction prompt:" or similar
        meta_patterns = [
            r'^This improved instruction prompt:',
            r'^This improved prompt:',
            r'^This prompt:',
            r'^The above prompt:',
            r'^Explanation:',
            r'^Note:',
            r'^\d+\.',  # Numbered lists explaining the prompt
        ]

        cleaned_lines = []
        in_meta_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts meta-commentary
            is_meta = any(re.match(pattern, line, re.IGNORECASE) for pattern in meta_patterns)

            if is_meta:
                in_meta_section = True
                continue

            # If we're in meta section and line doesn't look like instruction, skip
            if in_meta_section:
                # Check if this looks like continuation of instruction
                if not (line[0].isupper() or line.startswith('Solve') or line.startswith('Answer')):
                    continue
                else:
                    in_meta_section = False

            cleaned_lines.append(line)

        # Join cleaned lines
        if cleaned_lines:
            prompt = ' '.join(cleaned_lines)

        # Remove trailing quotes and periods that might be artifacts
        prompt = prompt.strip('"\'.')

        # Remove any remaining "To solve..." repetitions (common artifact)
        # Keep only the first occurrence
        if prompt.count('To solve') > 1 or prompt.count('Solve') > 1:
            # Find the first complete instruction sentence
            sentences = re.split(r'[.!?]\s+', prompt)
            # Keep sentences that look like instructions
            instruction_sentences = []
            seen_instruction = False
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                # Check if this is an instruction (starts with verb or "To")
                if re.match(r'^(Solve|Answer|Calculate|Find|Determine|To\s+\w+)', sent, re.IGNORECASE):
                    if not seen_instruction:
                        instruction_sentences.append(sent)
                        seen_instruction = True
                    # Skip duplicate instructions
                elif instruction_sentences:
                    # Add supporting sentences
                    instruction_sentences.append(sent)

            if instruction_sentences:
                prompt = '. '.join(instruction_sentences)
                if not prompt.endswith('.'):
                    prompt += '.'

        return prompt.strip()

    # Meta-prompt for generating textual gradients (criticism)
    GRADIENT_PROMPT = """
You are a CRITIC for Prompt Optimization.
Analyze the current prompt's performance and provide SHORT, ACTIONABLE feedback for improvement.

CONTEXT
- TASK: {task_description}
- CURRENT_PROMPT:
<<<
{prompt}
>>>
- PERFORMANCE:
  * Accuracy: {accuracy:.2%} (correct={correct}, total={total}, minibatch={num_examples})
  * Top failure examples (truncated):
{results}

IMPORTANT - How answers are evaluated (Math-Verify approach):
The evaluation uses a 3-step robust verification process:

STEP 1 - EXTRACTION (prioritized patterns, prefers LATER matches):
  * Highest priority: "\boxed{{NUMBER}}", "#### NUMBER", "final_answer: NUMBER"
  * Medium priority: "the answer is NUMBER", "therefore NUMBER"
  * Fallback: last number in output
  * Units are IGNORED (e.g., "42 km" → "42")

STEP 2 - PARSING (normalization to SymPy):
  * Removes: commas (1,234 → 1234), currency symbols ($50 → 50), units
  * Handles: percentages (50% → 0.5), fractions (1/3), decimals (3.14)
  * Converts to symbolic representation for smart comparison

STEP 3 - VERIFICATION (multiple strategies):
  * Numerical equality with tolerance (1/3 ≈ 0.333...)
  * Symbolic simplification (checks if predicted - ground_truth = 0)
  * Expression equivalence (different forms of same answer)

KEY INSIGHT: This is MORE FORGIVING than strict string matching!
- "1/3" and "0.333" are considered EQUIVALENT
- "42 km" and "42" are considered EQUIVALENT
- Different mathematical representations of same value are EQUIVALENT

Therefore, the prompt should:
1. Guide models to provide CLEAR final answers (preferably "#### NUMBER" format)
2. Ensure step-by-step work comes BEFORE the final answer
3. Avoid ambiguity about which number is the final answer
4. Don't worry too much about exact format - Math-Verify handles various formats

YOUR TASK:
Provide a concise critique identifying 2-4 key issues with the current prompt that led to failures.
For each issue, suggest 1-3 specific, testable improvements.

FORMAT YOUR RESPONSE AS:

ISSUE 1: [Brief description of the problem]
- Root cause: [Why this causes failures]
- Suggested improvements:
  * [Specific action 1]
  * [Specific action 2]

ISSUE 2: [Brief description of the problem]
- Root cause: [Why this causes failures]
- Suggested improvements:
  * [Specific action 1]

[Continue for 2-4 issues total]

GLOBAL NOTES:
- [Important constraint or guideline]
- [Another important note]

CONSTRAINTS:
- Keep suggestions compact (≈300-450 tokens total)
- Do NOT propose changes that break the output format
- Focus on systematic issues, not individual examples
- Ensure the prompt guides models to provide answers in recognizable formats (#### NUMBER)
"""






    # Meta-prompt for applying textual gradients (editing)
    EDIT_PROMPT = """
You are a PROMPT EDITOR. Apply the CRITIC's textual gradients to improve the CURRENT PROMPT.
Make small, targeted edits (≈5–15% of text). Preserve the task's output schema.

CURRENT PROMPT
<<<
{prompt}
>>>

CRITIC JSON (parse it and use the top-priority, highest-impact actions):
<<<
{gradient}
>>>

HARD RULES
- OUTPUT GUIDANCE: Ensure prompt guides models to provide CLEAR final answers (preferably "#### NUMBER")
- STEP-BY-STEP: Final answer should come AFTER reasoning, not before
- NO AMBIGUITY: Make it clear which number is the final answer
- BREVITY: Output ONLY the improved prompt, no preface, no quotes, no lists, no code fences.
- SIZE: MAX 3 sentences OR less than 150 words, whichever comes first.
- NO META: Do not include explanations, “here’s the improved prompt”, or references to the critic.
- DEDUPLICATE: Avoid repeating the same instruction in different words.
- STABILITY: Prefer clarifications, short rules, and a tiny checklist over long prose.
- If critic actions conflict, pick the smallest set that addresses the most frequent errors.

NOW OUTPUT ONLY THE IMPROVED PROMPT TEXT:
"""


    def __init__(
        self,
        llm_client,
        evaluator,
        beam_size: int = 4,
        num_iterations: int = 10,
        minibatch_size: int = 20,
        ucb_constant: float = 2.0,
        num_candidates_per_gradient: int = 4,
        task_description: str = "Solve math word problems step by step and provide the final numerical answer.",
        validation_evaluator = None,
        early_stopping_patience: int = 3,
    ):
        """
        Initialize ProTeGi optimizer.

        Args:
            llm_client: LLM client for generation
            evaluator: GSM8K evaluator for training
            beam_size: Number of prompt candidates to maintain
            num_iterations: Number of optimization iterations
            minibatch_size: Examples per evaluation
            ucb_constant: UCB exploration constant
            num_candidates_per_gradient: Number of new prompts to generate per gradient
            task_description: Description of the task for meta-prompts
            validation_evaluator: Optional separate evaluator for validation set
            early_stopping_patience: Number of iterations without improvement before stopping
        """
        self.llm = llm_client
        self.evaluator = evaluator
        self.validation_evaluator = validation_evaluator
        self.beam_size = beam_size
        self.num_iterations = num_iterations
        self.minibatch_size = minibatch_size
        self.ucb_constant = ucb_constant
        self.num_candidates_per_gradient = num_candidates_per_gradient
        self.task_description = task_description
        self.early_stopping_patience = early_stopping_patience

        self.total_evals = 0
        self.history = []
        self.used_indices = set()  # Track used indices for stratified sampling
        self.validation_history = []  # Track validation scores
        self.best_val_score = 0.0
        self.patience_counter = 0

    def _stratified_sample(self, dataset_size: int, sample_size: int) -> List[int]:
        """
        Stratified sampling to ensure better coverage of the dataset.
        Divides dataset into strata and samples from each without replacement.

        Args:
            dataset_size: Total size of the dataset
            sample_size: Number of samples to draw

        Returns:
            List of sampled indices
        """
        import random

        # Calculate how many strata we can have
        num_strata = min(10, dataset_size // sample_size) if sample_size > 0 else 1
        num_strata = max(1, num_strata)

        strata_size = dataset_size // num_strata
        samples_per_stratum = sample_size // num_strata
        remainder = sample_size % num_strata

        indices = []
        for i in range(num_strata):
            start = i * strata_size
            end = start + strata_size if i < num_strata - 1 else dataset_size

            # Sample from this stratum
            stratum_sample_size = samples_per_stratum + (1 if i < remainder else 0)
            stratum_indices = list(range(start, end))

            # Prefer unused indices
            unused = [idx for idx in stratum_indices if idx not in self.used_indices]
            if len(unused) >= stratum_sample_size:
                sampled = random.sample(unused, stratum_sample_size)
            else:
                # If not enough unused, sample from all
                sampled = random.sample(stratum_indices, min(stratum_sample_size, len(stratum_indices)))

            indices.extend(sampled)
            self.used_indices.update(sampled)

        # Reset used indices if we've used too many (>80% of dataset)
        if len(self.used_indices) > 0.8 * dataset_size:
            self.used_indices.clear()

        return indices[:sample_size]

    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Calculate similarity between two prompts using SequenceMatcher.

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, prompt1.lower(), prompt2.lower()).ratio()

    def _select_diverse_beam(self, candidates: List[PromptCandidate], beam_size: int,
                            diversity_threshold: float = 0.85) -> List[PromptCandidate]:
        """
        Select diverse beam candidates with diversity penalty.
        Ensures beam doesn't converge to very similar prompts.

        Args:
            candidates: List of all candidates
            beam_size: Target beam size
            diversity_threshold: Similarity threshold (prompts above this are considered too similar)

        Returns:
            List of selected diverse candidates
        """
        if len(candidates) <= beam_size:
            return candidates

        # Sort by score first
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        # Always keep the best candidate
        selected = [candidates[0]]

        # Select remaining candidates with diversity check
        for candidate in candidates[1:]:
            if len(selected) >= beam_size:
                break

            # Check similarity with already selected candidates
            max_similarity = max(
                self._calculate_similarity(candidate.prompt, s.prompt)
                for s in selected
            )

            # Only add if sufficiently different OR significantly better score
            if max_similarity < diversity_threshold or candidate.score > selected[-1].score + 0.1:
                selected.append(candidate)

        # If we don't have enough diverse candidates, fill with best remaining
        if len(selected) < beam_size:
            for candidate in candidates:
                if candidate not in selected and len(selected) < beam_size:
                    selected.append(candidate)

        return selected[:beam_size]

    def evaluate_prompt(self, prompt: str, random_sample: bool = True) -> Dict[str, Any]:
        """
        Evaluate a prompt on a minibatch of examples.

        Args:
            prompt: Prompt to evaluate
            random_sample: If True, randomly sample from dataset; if False, use sequential batches

        Returns:
            Evaluation results
        """
        if random_sample:
            # Use stratified sampling for better dataset coverage
            dataset_size = len(self.evaluator)
            indices = self._stratified_sample(dataset_size, min(self.minibatch_size, dataset_size))
            batch = [
                {
                    'idx': idx,
                    'question': self.evaluator.dataset[idx]['question'],
                    'answer': self.evaluator.dataset[idx]['answer']
                }
                for idx in indices
            ]
        else:
            # Sequential sampling (for final evaluation)
            batch = self.evaluator.get_batch(0, self.minibatch_size)

        # Generate answers
        questions = [example['question'] for example in batch]
        prompts = [f"{prompt}\n\nQuestion: {q}\nAnswer:" for q in questions]
        outputs = self.llm.generate_batch(prompts, temperature=0.7)

        # Evaluate
        indices = [example['idx'] for example in batch]
        results = self.evaluator.evaluate_batch(outputs, indices)

        return results

    def generate_gradient(self, candidate: PromptCandidate, results: Dict[str, Any]) -> str:
        """
        Generate textual gradient (critique) for a prompt.

        Args:
            candidate: Prompt candidate
            results: Evaluation results

        Returns:
            Textual gradient (critique)
        """
        # Format results for the gradient prompt
        error_examples = [
            f"Question: {d['question']}\nPredicted: {d['predicted']}\nCorrect: {d['ground_truth']}"
            for d in results['details'][:20]  # Show first 10 examples
            if not d['correct']
        ]

        results_text = "\n\n".join(error_examples) if error_examples else "All examples correct!"

        gradient_prompt = self.GRADIENT_PROMPT.format(
            prompt=candidate.prompt,
            task_description=self.task_description,
            num_examples=results['total'],
            results=results_text,
            accuracy=results['accuracy'],
            correct=results['correct'],
            total=results['total'],
        )

        gradient = self.llm.generate(gradient_prompt, temperature=0.7, max_new_tokens=4000)
        return gradient

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that generated prompt is reasonable.

        Args:
            prompt: Prompt to validate

        Returns:
            True if valid, False otherwise
        """
        # Remove common formatting artifacts
        clean_prompt = prompt.strip().strip('```').strip('"\'')

        # Basic checks
        if len(clean_prompt) < 10:
            return False
            
        return True

    def apply_gradient(self, candidate: PromptCandidate, gradient: str, verbose: bool = False, candidate_num: int = 1) -> Optional[str]:
        """
        Apply textual gradient to generate improved prompt.

        Args:
            candidate: Current prompt candidate
            gradient: Textual gradient (critique)
            verbose: Print debug info
            candidate_num: Candidate number for logging

        Returns:
            New improved prompt, or None if generation/validation fails
        """
        edit_prompt = self.EDIT_PROMPT.format(
            prompt=candidate.prompt,
            gradient=gradient,
        )

        # Lower temperature and max_new_tokens to reduce repetition and verbosity
        new_prompt = self.llm.generate(edit_prompt, temperature=0.5, max_new_tokens=1200)

        # Post-processing: remove common artifacts
        new_prompt = new_prompt.strip().strip('"\'')

        # Remove markdown code blocks (```text, ```, etc.)
        new_prompt = re.sub(r'```\w*\s*', '', new_prompt)  # Remove ```text, ```python, etc.
        new_prompt = re.sub(r'```', '', new_prompt)  # Remove remaining ```

        # Remove repeated empty code blocks pattern
        new_prompt = re.sub(r'(\s*```\s*)+', ' ', new_prompt)

        # Remove common preambles that slip through (including new ones observed)
        preambles = [
            "Here's the improved prompt:",
            "Here is the improved prompt:",
            "The new prompt is:",
            "Improved prompt:",
            "CORRECT OUTPUT:",
            "Certainly.",
            "Sure.",
            "To address the critique,",
            "To address the critique and improve the prompt",
            "The improved instruction prompt is as follows:",
            "To apply the critique",
            "CURRENT PROMPT:",
            "CRITIQUE:",
            "IMPROVED PROMPT:",
        ]
        for preamble in preambles:
            # Case-insensitive removal
            if new_prompt.lower().startswith(preamble.lower()):
                new_prompt = new_prompt[len(preamble):].strip()
                # Remove any leading colon or comma
                new_prompt = new_prompt.lstrip(':,').strip()

        # Remove any text that looks like it's quoting the original prompt or critique
        # Pattern: CURRENT PROMPT: "..." CRITIQUE: "..."
        new_prompt = re.sub(r'CURRENT PROMPT:.*?CRITIQUE:.*?(?=\n|$)', '', new_prompt, flags=re.IGNORECASE | re.DOTALL)

        # Remove lines that contain meta-commentary patterns
        lines = new_prompt.split('\n')
        clean_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that are clearly meta-commentary
            if any(pattern in line_lower for pattern in [
                'issue:',
                'root cause:',
                'concrete action:',
                'priority:',
                'impact:',
                'recommendation:',
                'next steps:',
                'implement the',
                'conduct a/b testing',
                'refine the',
            ]):
                continue
            clean_lines.append(line)
        new_prompt = '\n'.join(clean_lines).strip()

        # Remove markdown separators and structure
        new_prompt = re.sub(r'^---+\s*', '', new_prompt, flags=re.MULTILINE)
        new_prompt = re.sub(r'\n---+\s*\n', '\n', new_prompt)

        # If prompt contains numbered lists or markdown structure, extract just the instruction
        if '**Action:**' in new_prompt or '**Example:**' in new_prompt or re.search(r'^\d+\.\s+\*\*', new_prompt):
            lines = new_prompt.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('---') or '**Action:**' in line or '**Example:**' in line or '**Impact:**' in line:
                    break
                if line and not re.match(r'^\d+\.\s+\*\*', line):
                    clean_lines.append(line)
            if clean_lines:
                new_prompt = ' '.join(clean_lines)

        # IMPROVED: Remove ALL duplicate sentences (both consecutive and non-consecutive)
        sentences = re.split(r'([.!?])\s+', new_prompt)
        # Reconstruct with punctuation
        seen_sentences = set()
        reconstructed = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in '.!?':
                sent = sentences[i] + sentences[i+1]
                sent_normalized = sent.strip().lower()
                # Only add if not seen before (catches both consecutive and non-consecutive duplicates)
                if sent_normalized and sent_normalized not in seen_sentences:
                    seen_sentences.add(sent_normalized)
                    reconstructed.append(sent)
                i += 2
            else:
                if sentences[i].strip():
                    reconstructed.append(sentences[i])
                i += 1
        new_prompt = ' '.join(reconstructed).strip()

        # Remove bold/italic markdown formatting
        new_prompt = re.sub(r'\*\*([^*]+)\*\*', r'\1', new_prompt)  # **text** -> text
        new_prompt = re.sub(r'\*([^*]+)\*', r'\1', new_prompt)  # *text* -> text

        # Final cleanup
        new_prompt = new_prompt.strip().strip('"\'')

        # Enforce length constraint: maximum 300 words
        words = new_prompt.split()
        if len(words) > 300:
            new_prompt = ' '.join(words[:300])
            # Try to end at a sentence boundary
            if not new_prompt.endswith(('.', '!', '?')):
                # Find last sentence ending
                last_period = max(new_prompt.rfind('.'), new_prompt.rfind('!'), new_prompt.rfind('?'))
                if last_period > len(new_prompt) // 2:  # Only truncate if we keep at least half
                    new_prompt = new_prompt[:last_period + 1]

        # Validate the generated prompt
        if not self.validate_prompt(new_prompt):
            if verbose:
                print(f"⚠️  Candidate {candidate_num} REJECTED (invalid)")
                print(f"    Reason: Failed validation checks")
                print(f"    Prompt: {new_prompt[:100]}...")
            return None

        if verbose:
            print(f"✓ Candidate {candidate_num}: {new_prompt}")

        return new_prompt

    def optimize(self, initial_prompt: str, verbose: bool = True) -> Tuple[str, List[Dict]]:
        """
        Run ProTeGi optimization.

        Args:
            initial_prompt: Starting prompt
            verbose: Whether to print progress

        Returns:
            Tuple of (best_prompt, optimization_history)
        """
        # Initialize beam with initial prompt
        beam = [PromptCandidate(prompt=initial_prompt)]

        if verbose:
            print(f"\n{'='*80}")
            print("ProTeGi Optimization")
            print(f"{'='*80}\n")
            print(f"Initial prompt: {initial_prompt}\n")

        for iteration in range(self.num_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.num_iterations} ---")

            # Select candidate using UCB
            ucb_scores = [c.ucb_score(self.total_evals, self.ucb_constant) for c in beam]
            selected_idx = np.argmax(ucb_scores)
            candidate = beam[selected_idx]

            if verbose:
                print(f"Selected candidate (UCB: {ucb_scores[selected_idx]:.3f})")
                print(f"Prompt: {candidate.prompt}")

            # Evaluate on random minibatch from train set
            results = self.evaluate_prompt(candidate.prompt, random_sample=True)

            candidate.update_score(results['accuracy'])
            self.total_evals += 1

            if verbose:
                print(f"Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")

            # Record history
            self.history.append({
                'iteration': iteration,
                'prompt': candidate.prompt,
                'score': results['accuracy'],
                'selected_idx': selected_idx,
            })

            # Generate gradient (critique)
            gradient = self.generate_gradient(candidate, results)

            if verbose:
                print(f"Gradient: {gradient}")

            # Generate multiple new candidates from this gradient
            for i in range(self.num_candidates_per_gradient):
                # Apply gradient to generate new prompt
                new_prompt = self.apply_gradient(candidate, gradient, verbose=verbose, candidate_num=i+1)

                # Skip if validation failed
                if new_prompt is None:
                    continue

                # Skip if duplicate
                if new_prompt in [c.prompt for c in beam]:
                    if verbose:
                        print(f"   Candidate {i+1} skipped (duplicate)")
                    continue

                # CRITICAL FIX: Evaluate BEFORE adding to beam
                new_results = self.evaluate_prompt(new_prompt, random_sample=True)

                # Create candidate with proper score
                new_candidate = PromptCandidate(
                    prompt=new_prompt,
                    history=candidate.history + [new_prompt]
                )
                new_candidate.update_score(new_results['accuracy'])
                self.total_evals += 1

                if verbose:
                    print(f"  Score: {new_results['accuracy']:.1%}")

                # Add to beam
                beam.append(new_candidate)

            # Keep only top beam_size candidates with diversity penalty
            beam = self._select_diverse_beam(beam, self.beam_size)

            if verbose:
                best = beam[0]
                print(f"\nBeam status: {len(beam)} candidates")
                print(f"  Best avg score: {best.score:.1%} (over {best.num_evals} evals)")
                print(f"  Best last score: {best.last_score:.1%}")
                if len(beam) > 1:
                    # Show diversity info
                    avg_similarity = np.mean([
                        self._calculate_similarity(beam[0].prompt, beam[i].prompt)
                        for i in range(1, len(beam))
                    ])
                    print(f"  Average similarity to best: {avg_similarity:.2%}")

            # Validation set monitoring (if validation evaluator provided)
            if self.validation_evaluator is not None:
                # Evaluate best candidate on validation set
                best_candidate = beam[0]

                # Create temporary evaluator context for validation
                original_evaluator = self.evaluator
                self.evaluator = self.validation_evaluator

                val_results = self.evaluate_prompt(best_candidate.prompt, random_sample=True)
                val_accuracy = val_results['accuracy']

                # Restore original evaluator
                self.evaluator = original_evaluator

                # Track validation history
                self.validation_history.append({
                    'iteration': iteration,
                    'val_accuracy': val_accuracy,
                    'train_accuracy': best_candidate.score,
                })

                if verbose:
                    print(f"Validation accuracy: {val_accuracy:.1%}")

                # Early stopping check
                if val_accuracy > self.best_val_score:
                    self.best_val_score = val_accuracy
                    self.patience_counter = 0
                    if verbose:
                        print(f"✓ New best validation score!")
                else:
                    self.patience_counter += 1
                    if verbose:
                        print(f"No improvement ({self.patience_counter}/{self.early_stopping_patience})")

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"\n⚠️  Early stopping triggered after {iteration + 1} iterations")
                        print(f"Best validation score: {self.best_val_score:.1%}")
                    break

        # Final evaluation of all beam candidates on random samples
        if verbose:
            print(f"\n{'='*80}")
            print("Final Beam Evaluation")
            print(f"{'='*80}\n")

        for i, candidate in enumerate(beam):
            results = self.evaluate_prompt(candidate.prompt, random_sample=True)
            candidate.update_score(results['accuracy'])

            if verbose:
                print(f"\nCandidate {i+1}:")
                print(f"Prompt: {candidate.prompt}")
                print(f"Score: {results['accuracy']:.1%}")

        # Return best
        best_candidate = max(beam, key=lambda c: c.score)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Best prompt (score: {best_candidate.score:.1%}):")
            print(best_candidate.prompt)
            print(f"{'='*80}\n")

        return best_candidate.prompt, self.history
