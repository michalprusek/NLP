"""
OPRO: Optimization by PROmpting

Based on "Large Language Models as Optimizers"
https://arxiv.org/abs/2309.03409
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import random
import json


def load_prompt_template(task_type: str, template_name: str) -> str:
    """
    Load prompt template from file.

    Args:
        task_type: Type of task (e.g., 'claudette', 'gsm8k')
        template_name: Name of template file (e.g., 'opro_meta')

    Returns:
        Prompt template string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    prompt_file = Path(__file__).parent / 'prompts' / task_type / f'{template_name}.txt'
    if prompt_file.exists():
        return prompt_file.read_text(encoding='utf-8')
    raise FileNotFoundError(f"Prompt template not found: {prompt_file}")


def bucket_score(score: float, num_buckets: int = 20) -> float:
    """
    Bucketize score into discrete buckets.

    Args:
        score: Original score (0.0 to 1.0)
        num_buckets: Number of buckets (default: 20, as per OPRO paper)
                    20 buckets = rounding to nearest multiple of 5%

    Returns:
        Bucketed score rounded to nearest bucket

    Example:
        bucket_score(0.73, 20) -> 0.75  # rounds to nearest 5%
        bucket_score(0.72, 20) -> 0.70  # rounds to nearest 5%
    """
    bucket_size = 1.0 / num_buckets
    return round(score / bucket_size) * bucket_size


@dataclass
class ScoredPrompt:
    """A prompt with its evaluation score"""
    prompt: str
    score: float

    def __repr__(self):
        return f"Score: {self.score:.1%} | Prompt: {self.prompt[:50]}..."


def print_failed_examples_opro(results: Dict, num_examples: int = 3):
    """
    Print failed examples from OPRO evaluation results with full model responses.

    Args:
        results: Evaluation results dict with 'details' key
        num_examples: Number of failed examples to show (default: 3)
    """
    if not results or 'details' not in results:
        return

    details = results['details']

    # Find failed examples
    failed = [r for r in details if not r.get('correct', False)]

    if not failed:
        return

    # Sample random failed examples
    num_to_show = min(num_examples, len(failed))
    sampled_failed = random.sample(failed, num_to_show)

    print(f"\n  Failed examples ({num_to_show}/{len(failed)}):")
    print(f"  {'-'*76}")

    for i, result in enumerate(sampled_failed, 1):
        # Get ground truth and prediction
        gt = result.get('ground_truth', 'N/A')
        pred = result.get('predicted', 'N/A')

        # For binary classification, show as FAIR/UNFAIR
        if isinstance(gt, bool):
            gt_str = "FAIR" if gt else "UNFAIR"
        else:
            gt_str = str(gt)

        if isinstance(pred, bool):
            pred_str = "FAIR" if pred else "UNFAIR"
        else:
            pred_str = str(pred)

        # Get text (question/text field)
        text = result.get('text', result.get('question', ''))

        print(f"  {i}. GT: {gt_str:6} | Pred: {pred_str:6} | {text}...")

        # Show full model response (output field contains the full response)
        if result.get('output'):
            response = result['output'].strip()
            print(f"     Response: {response}")

    print(f"  {'-'*76}")


class OPRO:
    """OPRO optimizer using LLM as meta-optimizer"""

    def __init__(
        self,
        task_llm_client,
        evaluator,
        num_iterations: int = 10,
        num_candidates_per_iter: int = 8,  # Paper uses 8 per iteration
        minibatch_size: int = 20,
        keep_top_k: int = 20,  # Paper keeps top 20 prompts in memory
        task_description: str = "Solve math word problems step by step and provide the final numerical answer.",
        meta_llm_client = None,
    ):
        """
        Initialize OPRO optimizer.

        Args:
            task_llm_client: LLM client for task evaluation (the model being optimized)
            meta_llm_client: Optional separate LLM client for meta-optimization (prompt generation).
                           If None, uses task_llm_client for both.
            evaluator: GSM8K evaluator
            num_iterations: Number of optimization iterations
            num_candidates_per_iter: Number of new candidates to generate per iteration
            minibatch_size: Size of fixed evaluation set (paper uses 3.5% of training data)
            keep_top_k: Number of top prompts to keep in memory
            task_description: Description of the task
        """
        self.task_llm = task_llm_client
        self.meta_llm = meta_llm_client if meta_llm_client is not None else task_llm_client
        self.evaluator = evaluator
        self.num_iterations = num_iterations
        self.num_candidates_per_iter = num_candidates_per_iter
        self.minibatch_size = minibatch_size
        self.keep_top_k = keep_top_k
        self.task_description = task_description

        self.scored_prompts: List[ScoredPrompt] = []
        self.history = []

        # Determine which metric to optimize based on task
        task_name_for_metric = getattr(evaluator, 'task_name', None)
        if task_name_for_metric == 'claudette':
            self.optimization_metric = 'micro_f1'  # Multi-label: use micro F1
            print(f"  Optimization metric: Micro F1 (multi-label classification)")
        elif task_name_for_metric == 'claudette_binary':
            # Binary classification: use Macro-F1 for class balance
            # Macro-F1 is standard in CLAUDETTE benchmark due to ~9:1 class imbalance
            # It treats both fair and unfair classes equally, unlike micro-F1 or unfair-only F1
            self.optimization_metric = 'macro_f1'
            print(f"  Optimization metric: Macro-F1 (binary classification, balanced across classes)")
        else:
            self.optimization_metric = 'accuracy'  # Default: accuracy (GSM8K, etc.)
            print(f"  Optimization metric: Accuracy")

        # OPRO paper method: Create fixed evaluation set (same set for all prompts)
        # This ensures comparable scores across all candidates
        # Paper uses small fixed subset (~3.5% of training data for GSM8K)
        dataset_size = len(evaluator)
        eval_size = min(minibatch_size, dataset_size)

        # Sample random indices for fixed evaluation set
        eval_indices = random.sample(range(dataset_size), eval_size)

        # Load fixed evaluation set once
        self.fixed_eval_set = []
        for idx in eval_indices:
            batch = evaluator.get_batch(idx, 1)
            if batch:
                self.fixed_eval_set.append(batch[0])

        print(f"  Fixed evaluation set: {len(self.fixed_eval_set)} examples ({100*len(self.fixed_eval_set)/dataset_size:.1f}% of training data)")
        print(f"  All prompts will be evaluated on the SAME fixed set for comparable scores")

        # Detect task type from evaluator and load appropriate templates
        # Prefer task_name (specific) over task_type (generic) for template selection
        task_name = getattr(evaluator, 'task_name', None)
        task_type = getattr(evaluator, 'task_type', 'regression')

        # Determine template directory
        if task_name:
            # Use specific task name if provided (e.g., 'claudette_binary')
            template_dir = task_name
        elif task_type == 'classification':
            # Default to 'claudette' for generic classification
            template_dir = 'claudette'
        else:
            # Default to 'gsm8k' for math/regression tasks
            template_dir = 'gsm8k'

        # Load meta-prompt template from file (always required)
        self.meta_prompt_template = load_prompt_template(template_dir, 'opro_meta')

    def _get_score(self, results: Dict[str, Any]) -> float:
        """
        Get optimization score from evaluation results based on task type.

        Args:
            results: Evaluation results dictionary

        Returns:
            Score to optimize (micro_f1 for claudette, f1 for claudette_binary, accuracy for others)
        """
        return results.get(self.optimization_metric, results.get('accuracy', 0.0))

    def _get_random_examples(self, num_examples: int = 3) -> str:
        """
        Get random examples from dataset for meta-prompt.
        Paper uses 3 randomly picked exemplars at each step.

        Args:
            num_examples: Number of random examples to pick (default: 3 as in paper)

        Returns:
            Formatted string with example problems
        """
        dataset_size = len(self.evaluator)
        random_indices = random.sample(range(dataset_size), min(num_examples, dataset_size))

        examples = []
        for i, idx in enumerate(random_indices):
            # Get example from dataset
            example_batch = self.evaluator.get_batch(idx, 1)
            if example_batch:
                ex = example_batch[0]
                # Handle both 'question' (GSM8K) and 'text' (Claudette) fields
                question = ex.get('question', ex.get('text', ''))
                answer = ex.get('answer', ex.get('label', ''))
                examples.append(f"Problem:\nQ: {question}\nA: <INS>\nGround truth answer: {answer}")

        return "\n\n".join(examples)

    def evaluate_prompt(self, prompt: str, return_details: bool = False) -> Tuple[float, Any]:
        """
        Evaluate a prompt on the fixed evaluation set.

        OPRO paper method: All prompts are evaluated on the SAME fixed set,
        ensuring scores are directly comparable (no noise from different data samples).

        Args:
            prompt: Prompt to evaluate
            return_details: If True, return (score, results) tuple; if False, return just score

        Returns:
            If return_details=False: Optimization score (micro_f1 for claudette, f1 for claudette_binary, accuracy for others)
            If return_details=True: Tuple of (score, results) where results contains detailed evaluation info
        """
        # Use fixed evaluation set (same for all prompts)
        batch = self.fixed_eval_set

        # Generate answers (handle both 'question' and 'text' fields)
        questions = [example.get('question', example.get('text', '')) for example in batch]
        prompts = [f"Question: {q}\n\n{prompt}\n\nAnswer:" for q in questions]
        outputs = self.task_llm.generate_batch(prompts, temperature=0.0)

        # Evaluate
        indices = [example['idx'] for example in batch]
        results = self.evaluator.evaluate_batch(outputs, indices)

        score = self._get_score(results)

        if return_details:
            return score, results
        return score

    def generate_candidates(self, save_debug_info: bool = False) -> Tuple[List[str], Any]:
        """
        Generate new prompt candidates using LLM as meta-optimizer.

        Paper implementation: Makes N independent calls to the optimizer LLM
        (each at temperature 1.0) to generate N diverse instructions.

        Args:
            save_debug_info: If True, return debug info (meta_prompt, raw responses)

        Returns:
            Tuple of (candidates, debug_info) where debug_info is None if save_debug_info=False
        """
        # Format scored prompts for the meta-prompt (keep top 20 as in paper)
        # Sort in ASCENDING order (worst to best) as per paper - shows progressive improvement
        # Bucketize scores to 20 buckets (round to nearest 5%) as per OPRO paper Section 3.2
        if self.scored_prompts:
            scored_prompts_text = "\n".join([
                f"text: {sp.prompt}\nscore: {int(bucket_score(sp.score, num_buckets=20) * 100)}"
                for sp in sorted(self.scored_prompts, key=lambda x: x.score)  # ascending order
            ])
        else:
            scored_prompts_text = "(No prompts evaluated yet)"

        # Generate N candidates with N independent calls (paper method)
        candidates = []
        seen = set(sp.prompt for sp in self.scored_prompts)

        # For debugging: save first meta-prompt and all raw responses
        debug_info = None
        if save_debug_info:
            debug_info = {
                'meta_prompt': None,
                'raw_responses': [],
            }

        for i in range(self.num_candidates_per_iter):
            # Get 3 random examples for this generation (paper uses 3 random exemplars)
            example_problems = self._get_random_examples(num_examples=3)

            # Format meta-prompt with random examples
            meta_prompt = self.meta_prompt_template.format(
                task_description=self.task_description,
                example_problems=example_problems,
                scored_prompts=scored_prompts_text,
                num_candidates=1,  # Generate 1 candidate per call
            )

            # Save first meta-prompt for debugging
            if save_debug_info and i == 0:
                debug_info['meta_prompt'] = meta_prompt

            # Generate candidate (temperature=1.0 as in paper for diversity)
            response = self.meta_llm.generate(meta_prompt, temperature=1.0, max_new_tokens=500)

            # Save raw response for debugging
            if save_debug_info:
                debug_info['raw_responses'].append(response)

            # Extract the prompt from <INS>...</INS> tags
            candidate = response.strip()

            # Look for <INS>...</INS> tags
            if '<INS>' in candidate and '</INS>' in candidate:
                start_idx = candidate.find('<INS>') + 5
                end_idx = candidate.find('</INS>')
                candidate = candidate[start_idx:end_idx].strip()
            else:
                # Fallback: clean up common artifacts
                for prefix in ['1. ', '2. ', '3. ', '4. ', '- ', '* ', 'Prompt: ', '"']:
                    if candidate.startswith(prefix):
                        candidate = candidate[len(prefix):]
                # Remove trailing quotes
                candidate = candidate.strip().strip('"').strip()

            # Skip if empty or duplicate
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        return candidates, debug_info

    def optimize(
        self,
        initial_prompts: List[str] = None,
        verbose: bool = True,
        save_intermediate_prompts: bool = False
    ) -> Tuple[str, List[Dict]]:
        """
        Run OPRO optimization.

        Args:
            initial_prompts: Starting prompts (if None, generate from scratch)
            verbose: Whether to print progress
            save_intermediate_prompts: If True, save meta-prompts, generated candidates,
                                      and formatted task prompts to history for debugging

        Returns:
            Tuple of (best_prompt, optimization_history)
        """
        if initial_prompts is None:
            initial_prompts = [
                "",
                "Solve the following problem.",
                "Let's solve the problem.",
            ]

        if verbose:
            print(f"\n{'='*80}")
            print("OPRO Optimization")
            print(f"{'='*80}\n")

        # Evaluate initial prompts
        if verbose:
            print("Evaluating initial prompts...\n")

        for prompt in initial_prompts:
            # All prompts evaluated on same fixed set (no data_idx needed)
            score, results = self.evaluate_prompt(prompt, return_details=True)

            self.scored_prompts.append(ScoredPrompt(prompt=prompt, score=score))

            # Record initial prompts in history (iteration -1 to distinguish from generated prompts)
            self.history.append({
                'iteration': -1,  # -1 indicates initial/seed prompt
                'prompt': prompt,
                'score': score,
            })

            if verbose:
                print(f"Score: {score:.1%} | Prompt: {prompt if prompt else '(empty prompt)'}")
                # Show failed examples
                # print_failed_examples_opro(results, num_examples=3)

        # Optimization loop
        for iteration in range(self.num_iterations):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
                print(f"{'='*80}\n")

            # Generate new candidates
            if verbose:
                print("Generating new candidates...\n")

            candidates, debug_info = self.generate_candidates(save_debug_info=save_intermediate_prompts)

            if not candidates:
                if verbose:
                    print("No new candidates generated. Stopping early.")
                break

            # Evaluate new candidates
            for i, candidate in enumerate(candidates):
                if verbose:
                    print(f"Evaluating: {candidate[:]}...")

                # All prompts evaluated on same fixed set (no data_idx needed)
                score, results = self.evaluate_prompt(candidate, return_details=True)

                self.scored_prompts.append(ScoredPrompt(prompt=candidate, score=score))

                if verbose:
                    print(f"Score: {score:.1%}")
                    # Show failed examples
                    # print_failed_examples_opro(results, num_examples=3)

                # Record history
                history_entry = {
                    'iteration': iteration,
                    'prompt': candidate,
                    'score': score,
                }

                # Add debug info for first candidate of this iteration
                if save_intermediate_prompts and i == 0 and debug_info:
                    # Add meta-prompt and generated candidates
                    history_entry['debug'] = {
                        'meta_prompt': debug_info['meta_prompt'],
                        'generated_candidates': candidates,
                        'raw_meta_responses': debug_info['raw_responses'],
                    }

                    # Add example formatted task prompt
                    if self.fixed_eval_set:
                        example = self.fixed_eval_set[0]
                        question = example.get('question', example.get('text', ''))
                        formatted_task_prompt = f"Question: {question}\n\n{candidate}\n\nAnswer:"
                        history_entry['debug']['example_formatted_task_prompt'] = formatted_task_prompt

                self.history.append(history_entry)

            # Keep only top-k prompts
            self.scored_prompts.sort(key=lambda x: x.score, reverse=True)
            self.scored_prompts = self.scored_prompts[:self.keep_top_k]

            if verbose:
                print(f"Top {min(3, len(self.scored_prompts))} prompts so far:")
                for i, sp in enumerate(self.scored_prompts[:3]):
                    print(f"{i+1}. {sp}")

        # Return best prompt
        best = max(self.scored_prompts, key=lambda x: x.score)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Best prompt (score: {best.score:.1%}):")
            print(best.prompt)
            print(f"{'='*80}\n")

        return best.prompt, self.history
