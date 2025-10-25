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
import random
from difflib import SequenceMatcher
from pathlib import Path


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


def load_prompt_template(task_type: str, template_name: str, default_template: str = None) -> str:
    """
    Load prompt template from file.

    Args:
        task_type: Type of task (e.g., 'claudette', 'gsm8k')
        template_name: Name of template file (e.g., 'gradient', 'edit')
        default_template: Optional default template to use if file doesn't exist

    Returns:
        Prompt template string

    Raises:
        FileNotFoundError: If template file doesn't exist and no default provided
    """
    prompt_file = Path(__file__).parent / 'prompts' / task_type / f'{template_name}.txt'
    if prompt_file.exists():
        return prompt_file.read_text(encoding='utf-8')
    if default_template is not None:
        return default_template
    raise FileNotFoundError(f"Prompt template not found: {prompt_file}")


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


    def __init__(
        self,
        task_llm_client,
        evaluator,
        beam_size: int = 4,
        num_iterations: int = 10,
        minibatch_size: int = 20,
        ucb_constant: float = 2.0,
        num_candidates_per_gradient: int = 4,
        task_description: str = "Solve math word problems step by step and provide the final numerical answer.",
        validation_evaluator = None,
        early_stopping_patience: int = 3,
        meta_llm_client = None,
    ):
        """
        Initialize ProTeGi optimizer.

        Args:
            task_llm_client: LLM client for task evaluation (the model being optimized)
            evaluator: GSM8K evaluator for training
            beam_size: Number of prompt candidates to maintain
            num_iterations: Number of optimization iterations
            minibatch_size: Examples per evaluation
            ucb_constant: UCB exploration constant
            num_candidates_per_gradient: Number of new prompts to generate per gradient
            task_description: Description of the task for meta-prompts
            validation_evaluator: Optional separate evaluator for validation set
            early_stopping_patience: Number of iterations without improvement before stopping
            meta_llm_client: Optional separate LLM client for meta-optimization (gradient generation, editing).
                           If None, uses task_llm_client for both.
        """
        self.task_llm = task_llm_client
        self.meta_llm = meta_llm_client if meta_llm_client is not None else task_llm_client
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

        # Determine which metric to optimize based on task
        task_name_for_metric = getattr(evaluator, 'task_name', None)
        if task_name_for_metric == 'claudette':
            self.optimization_metric = 'micro_f1'  # Multi-label: use micro F1
            print(f"  Optimization metric: Micro F1 (multi-label classification)")
        elif task_name_for_metric == 'claudette_binary':
            self.optimization_metric = 'f1'  # Binary: use F1 for unfair class
            print(f"  Optimization metric: F1 (binary classification)")
        else:
            self.optimization_metric = 'accuracy'  # Default: accuracy (GSM8K, etc.)
            print(f"  Optimization metric: Accuracy")

        # Detect task type from evaluator and load appropriate templates
        # Prefer task_name (specific) over task_type (generic) for template selection
        task_name = getattr(evaluator, 'task_name', None)
        task_type = getattr(evaluator, 'task_type', 'regression')

        if task_name:
            # Use specific task name if provided (e.g., 'claudette_binary')
            template_dir = task_name
        elif task_type == 'classification':
            # Default to 'claudette' for generic classification
            template_dir = 'claudette'
        else:
            # Default to 'gsm8k' for regression/other tasks
            template_dir = 'gsm8k'

        # Load templates from appropriate directory
        self.gradient_prompt_template = load_prompt_template(template_dir, 'gradient')
        self.edit_prompt_template = load_prompt_template(template_dir, 'edit')

    def _get_score(self, results: Dict[str, Any]) -> float:
        """
        Get optimization score from evaluation results based on task type.

        Args:
            results: Evaluation results dictionary

        Returns:
            Score to optimize (micro_f1 for claudette, f1 for claudette_binary, accuracy for others)
        """
        return results.get(self.optimization_metric, results.get('accuracy', 0.0))

    def _stratified_sample(self, dataset_size: int, sample_size: int) -> List[int]:
        """
        Stratified sampling to ensure better coverage of the dataset.
        Divides dataset into strata and samples from each without replacement.

        For classification tasks with labels (e.g., Claudette), uses label-based stratification.
        For other tasks (e.g., GSM8K), uses sequential stratification.

        Args:
            dataset_size: Total size of the dataset
            sample_size: Number of samples to draw

        Returns:
            List of sampled indices
        """
        import random

        # Check if evaluator has label-based stratification (Claudette)
        if hasattr(self.evaluator, 'dataset') and hasattr(self.evaluator.dataset, 'column_names'):
            if 'label' in self.evaluator.dataset.column_names:
                return self._stratified_sample_by_label(dataset_size, sample_size)

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

    def _stratified_sample_by_label(self, dataset_size: int, sample_size: int) -> List[int]:
        """
        Stratified sampling by label for classification tasks (e.g., Claudette).
        Samples 50% from neutral examples (no labels) and 50% from unfair examples (1+ labels).

        This ensures balanced representation despite 90% of dataset being neutral.

        Args:
            dataset_size: Total size of dataset
            sample_size: Number of samples to draw

        Returns:
            List of sampled indices stratified by neutral vs unfair
        """
        import random
        from src.claudette_evaluator import get_ground_truth_labels

        if sample_size >= dataset_size:
            return list(range(dataset_size))

        # Group indices into neutral (no labels) vs unfair (1+ labels)
        neutral_indices = []  # NONE labels
        unfair_indices = []   # 1+ labels

        for idx in range(dataset_size):
            example = self.evaluator.dataset[idx]
            labels = get_ground_truth_labels(example)

            if len(labels) == 0:
                neutral_indices.append(idx)
            else:
                unfair_indices.append(idx)

        # Sample 50% from neutral, 50% from unfair
        neutral_sample_size = sample_size // 2
        unfair_sample_size = sample_size - neutral_sample_size

        indices = []

        # Sample from neutral examples
        if len(neutral_indices) > 0:
            # Prefer unused indices
            unused_neutral = [idx for idx in neutral_indices if idx not in self.used_indices]
            if len(unused_neutral) >= neutral_sample_size:
                sampled_neutral = random.sample(unused_neutral, neutral_sample_size)
            else:
                # If not enough unused, sample from all available
                sampled_neutral = random.sample(neutral_indices, min(neutral_sample_size, len(neutral_indices)))

            indices.extend(sampled_neutral)
            self.used_indices.update(sampled_neutral)

        # Sample from unfair examples
        if len(unfair_indices) > 0:
            # Prefer unused indices
            unused_unfair = [idx for idx in unfair_indices if idx not in self.used_indices]
            if len(unused_unfair) >= unfair_sample_size:
                sampled_unfair = random.sample(unused_unfair, unfair_sample_size)
            else:
                # If not enough unused, sample from all available
                sampled_unfair = random.sample(unfair_indices, min(unfair_sample_size, len(unfair_indices)))

            indices.extend(sampled_unfair)
            self.used_indices.update(sampled_unfair)

        # Reset used indices if we've used too many (>80% of dataset)
        if len(self.used_indices) > 0.8 * dataset_size:
            self.used_indices.clear()

        # Shuffle to mix neutral and unfair examples
        random.shuffle(indices)

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
            # Use get_batch to handle different dataset formats (GSM8K vs Claudette)
            batch = []
            for idx in indices:
                example = self.evaluator.dataset[idx]
                # Handle both GSM8K (question/answer) and Claudette (sentence/labels) formats
                question = example.get('question', example.get('sentence', example.get('text', '')))

                # For Claudette, extract labels from boolean fields
                if 'sentence' in example:
                    # Import get_ground_truth_labels from claudette_evaluator
                    from src.claudette_evaluator import get_ground_truth_labels
                    labels = get_ground_truth_labels(example)
                    answer = str(sorted(labels))
                else:
                    # GSM8K format
                    answer = example.get('answer', str(example.get('label', '')))

                batch.append({
                    'idx': idx,
                    'question': question,
                    'answer': answer
                })
        else:
            # Sequential sampling (for final evaluation)
            batch = self.evaluator.get_batch(0, self.minibatch_size)

        # Generate answers
        questions = [example['question'] for example in batch]
        prompts = [f"{prompt}\n\nQuestion: {q}\nAnswer:" for q in questions]
        # Paper uses temperature=0.0 for task evaluation (few-shot classification)
        outputs = self.task_llm.generate_batch(prompts, temperature=0.0)

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
        # Handle both 'question' (GSM8K) and 'text' (Claudette) fields
        # Paper uses groups of 4 errors per gradient (Section 3.2)
        error_examples = []
        for d in results['details'][:4]:  # Use only 4 errors as per paper
            if not d['correct']:
                question_field = d.get('question') or d.get('text', 'N/A')
                error_examples.append(
                    f"Question: {question_field}\nPredicted: {d['predicted']}\nCorrect: {d['ground_truth']}"
                )

        results_text = "\n\n".join(error_examples) if error_examples else "All examples correct!"

        gradient_prompt = self.gradient_prompt_template.format(
            prompt=candidate.prompt,
            task_description=self.task_description,
            num_examples=results['total'],
            results=results_text,
            accuracy=results['accuracy'],
            correct=results['correct'],
            total=results['total'],
        )

        # Paper uses temperature=1.0 for gradient generation (exploration)
        gradient = self.meta_llm.generate(gradient_prompt, temperature=1.0, max_new_tokens=4000)
        return gradient

    def _generate_paraphrases(self, prompt: str, num_paraphrases: int = 2) -> List[str]:
        """
        Generate Monte Carlo paraphrases of a prompt (Algorithm 2 line 5 from paper).

        Args:
            prompt: Prompt to paraphrase
            num_paraphrases: Number of paraphrases to generate

        Returns:
            List of paraphrased prompts
        """
        # Paraphrasing prompt from paper Appendix 1.1
        paraphrase_template = """Generate a variation of the following instruction while keeping the semantic meaning.

Input: {prompt_instruction}

Output:"""

        paraphrases = []
        for _ in range(num_paraphrases):
            paraphrase_prompt = paraphrase_template.format(prompt_instruction=prompt)
            # Use temperature=1.0 for diversity
            paraphrase = self.meta_llm.generate(paraphrase_prompt, temperature=1.0, max_new_tokens=512)

            # Clean the paraphrase
            paraphrase = self.clean_prompt(paraphrase)

            # Validate
            if self.validate_prompt(paraphrase) and paraphrase != prompt:
                paraphrases.append(paraphrase)

        return paraphrases

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
        edit_prompt = self.edit_prompt_template.format(
            prompt=candidate.prompt,
            gradient=gradient,
        )

        # Paper uses temperature=1.0 for prompt editing (exploration)
        new_prompt = self.meta_llm.generate(edit_prompt, temperature=1.0, max_new_tokens=1200)

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

            score = self._get_score(results)
            candidate.update_score(score)
            self.total_evals += 1

            if verbose:
                print(f"Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
                # Show comprehensive metrics for classification tasks (e.g., Claudette)
                if 'micro_f1' in results:
                    print(f"  Micro-F1: {results['micro_f1']:.1%} | Macro-F1: {results['macro_f1']:.1%} | Hamming: {results['hamming_loss']:.3f}")
                if 'f1' in results and self.optimization_metric == 'f1':
                    print(f"  F1: {results['f1']:.1%} | Precision: {results.get('precision', 0):.1%} | Recall: {results.get('recall', 0):.1%}")
                print(f"  Optimization score ({self.optimization_metric}): {score:.1%}")

            # Record history
            self.history.append({
                'iteration': iteration,
                'prompt': candidate.prompt,
                'score': score,
                'selected_idx': selected_idx,
            })

            # Generate multiple gradients (m=4 as per paper Algorithm 2)
            # Sample different error groups for diversity
            error_details = [d for d in results['details'] if not d['correct']]

            num_gradients = min(4, max(1, len(error_details) // 4))  # m=4 gradients

            for grad_idx in range(num_gradients):
                # Sample a small group of errors (4 errors per gradient as per paper)
                if len(error_details) >= 4:
                    error_sample_indices = random.sample(range(len(error_details)), min(4, len(error_details)))
                    error_sample = [error_details[i] for i in error_sample_indices]
                else:
                    error_sample = error_details

                # Create mini results dict for this error group
                mini_results = {
                    'details': error_sample,
                    'total': len(error_sample),
                    'correct': 0,
                    'accuracy': 0.0
                }

                # Generate gradient from this error group
                gradient = self.generate_gradient(candidate, mini_results)

                if verbose:
                    print(f"Gradient {grad_idx+1}/{num_gradients}: {gradient[:100]}...")

                # Apply gradient to generate new prompt (q=1 per gradient as per paper)
                new_prompt = self.apply_gradient(candidate, gradient, verbose=verbose, candidate_num=grad_idx+1)

                # Skip if validation failed
                if new_prompt is None:
                    continue

                # Skip if duplicate
                if new_prompt in [c.prompt for c in beam]:
                    if verbose:
                        print(f"   Candidate {grad_idx+1} skipped (duplicate)")
                    continue

                # Generate Monte Carlo paraphrases (p=2 as per paper Algorithm 2 line 5)
                paraphrases = self._generate_paraphrases(new_prompt, num_paraphrases=2)

                # Evaluate all candidates (original + paraphrases)
                all_new_prompts = [new_prompt] + paraphrases

                for i, prompt_variant in enumerate(all_new_prompts):
                    # Skip duplicates
                    if prompt_variant in [c.prompt for c in beam]:
                        if verbose:
                            print(f"   Variant {i+1} skipped (duplicate)")
                        continue

                    # Evaluate BEFORE adding to beam
                    new_results = self.evaluate_prompt(prompt_variant, random_sample=True)

                    # Create candidate with proper score
                    new_candidate = PromptCandidate(
                        prompt=prompt_variant,
                        history=candidate.history + [prompt_variant]
                    )
                    new_score = self._get_score(new_results)
                    new_candidate.update_score(new_score)
                    self.total_evals += 1

                    if verbose:
                        variant_type = "original" if i == 0 else f"paraphrase {i}"
                        print(f"  Gradient {grad_idx+1} {variant_type}: Acc={new_results['accuracy']:.1%}", end='')
                        if 'micro_f1' in new_results:
                            print(f", Micro-F1={new_results['micro_f1']:.1%}", end='')
                        if 'f1' in new_results and self.optimization_metric == 'f1':
                            print(f", F1={new_results['f1']:.1%}", end='')
                        print(f", Score={new_score:.1%}")

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
                val_score = self._get_score(val_results)

                # Restore original evaluator
                self.evaluator = original_evaluator

                # Track validation history
                self.validation_history.append({
                    'iteration': iteration,
                    'val_score': val_score,
                    'val_accuracy': val_results['accuracy'],  # Keep for backwards compatibility
                    'train_score': best_candidate.score,
                })

                if verbose:
                    print(f"Validation {self.optimization_metric}: {val_score:.1%} (Accuracy: {val_results['accuracy']:.1%})")
                    if 'micro_f1' in val_results:
                        print(f"  Micro-F1: {val_results['micro_f1']:.1%} | Macro-F1: {val_results['macro_f1']:.1%}")

                # Early stopping check
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.patience_counter = 0
                    if verbose:
                        print(f"✓ New best validation {self.optimization_metric}!")
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
            final_score = self._get_score(results)
            candidate.update_score(final_score)

            if verbose:
                print(f"\nCandidate {i+1}:")
                print(f"Prompt: {candidate.prompt}")
                print(f"Optimization score ({self.optimization_metric}): {final_score:.1%}")
                print(f"  Accuracy: {results['accuracy']:.1%}")
                if 'micro_f1' in results:
                    print(f"  Micro-F1: {results['micro_f1']:.1%} | Macro-F1: {results['macro_f1']:.1%}")
                if 'f1' in results and self.optimization_metric == 'f1':
                    print(f"  F1: {results['f1']:.1%} | Precision: {results.get('precision', 0):.1%} | Recall: {results.get('recall', 0):.1%}")

        # Return best
        best_candidate = max(beam, key=lambda c: c.score)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Best prompt (score: {best_candidate.score:.1%}):")
            print(best_candidate.prompt)
            print(f"{'='*80}\n")

        return best_candidate.prompt, self.history
