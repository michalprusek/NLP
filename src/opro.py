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


def load_prompt_template(task_type: str, template_name: str, default_template: str) -> str:
    """
    Load prompt template from file if exists, otherwise return default.

    Args:
        task_type: Type of task (e.g., 'claudette', 'gsm8k')
        template_name: Name of the template file (without .txt extension)
        default_template: Default template to use if file doesn't exist

    Returns:
        Loaded or default template string
    """
    prompt_file = Path(__file__).parent / 'prompts' / task_type / f'{template_name}.txt'
    if prompt_file.exists():
        return prompt_file.read_text(encoding='utf-8')
    return default_template


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
        text = result.get('text', result.get('question', ''))[:80]

        print(f"  {i}. GT: {gt_str:6} | Pred: {pred_str:6} | {text}...")

        # Show full model response (output field contains the full response)
        if result.get('output'):
            response = result['output'].strip()
            print(f"     Response: {response}")

    print(f"  {'-'*76}")


class OPRO:
    """OPRO optimizer using LLM as meta-optimizer"""

    # Meta-prompt for generating new prompt candidates
    META_PROMPT = """You are an optimization algorithm generating instruction prompts for solving math word problems.

TASK: {task_description}

EXAMPLE PROBLEMS:
{example_problems}

EVALUATION SYSTEM (INVARIANT):
- Extracts LAST number from model output
- Recognizes: "#### NUMBER", "\\boxed{{NUMBER}}", "answer is NUMBER"
- Normalizes: "1,234" → "1234", "$50" → "50"
- Requires EXACT match (no partial credit)

PREVIOUS PROMPTS AND SCORES:
{scored_prompts}

YOUR TASK:
Generate a NEW instruction prompt that will achieve HIGHER accuracy than previous ones.

REQUIREMENTS:
1. Prompt must be 2-4 sentences
2. Must instruct to provide answer in format "#### NUMBER"
3. Should encourage step-by-step reasoning
4. Should be clear and actionable
5. Must be DIFFERENT from previous prompts
6. Explore different instruction strategies based on what worked well before

OUTPUT FORMAT:
Write EXACTLY ONE prompt.
NO numbering, NO bullet points, NO explanations.
JUST the instruction prompt itself.

EXAMPLE OUTPUT:
Solve the math problem step by step and provide the answer as #### NUMBER.

NOW GENERATE YOUR NEW PROMPT:"""

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
            minibatch_size: Examples per evaluation
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

        # Detect task type from evaluator and load appropriate templates
        # Prefer task_name (specific) over task_type (generic) for template selection
        task_name = getattr(evaluator, 'task_name', None)
        task_type = getattr(evaluator, 'task_type', 'regression')

        if task_name:
            # Use specific task name if provided (e.g., 'claudette_binary')
            self.meta_prompt_template = load_prompt_template(task_name, 'opro_meta', self.META_PROMPT)
        elif task_type == 'classification':
            # Default to 'claudette' for generic classification
            self.meta_prompt_template = load_prompt_template('claudette', 'opro_meta', self.META_PROMPT)
        else:
            # Use default GSM8K template
            self.meta_prompt_template = self.META_PROMPT

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
                question = ex.get('question', ex.get('text', ''))[:150]
                answer = ex.get('answer', ex.get('label', ''))
                examples.append(f"Example {i+1}:\nQ: {question}...\nA: {answer}")

        return "\n\n".join(examples)

    def evaluate_prompt(self, prompt: str, start_idx: int, return_details: bool = False) -> Tuple[float, Any]:
        """
        Evaluate a prompt on a minibatch.

        Args:
            prompt: Prompt to evaluate
            start_idx: Starting index in dataset
            return_details: If True, return (score, results) tuple; if False, return just score

        Returns:
            If return_details=False: Optimization score (micro_f1 for claudette, f1 for claudette_binary, accuracy for others)
            If return_details=True: Tuple of (score, results) where results contains detailed evaluation info
        """
        batch = self.evaluator.get_batch(start_idx, self.minibatch_size)

        # Generate answers (handle both 'question' and 'text' fields)
        questions = [example.get('question', example.get('text', '')) for example in batch]
        prompts = [f"{prompt}\n\nQuestion: {q}\nAnswer:" for q in questions]
        outputs = self.task_llm.generate_batch(prompts, temperature=0.0)

        # Evaluate
        indices = [example['idx'] for example in batch]
        results = self.evaluator.evaluate_batch(outputs, indices)

        score = self._get_score(results)

        if return_details:
            return score, results
        return score

    def generate_candidates(self) -> List[str]:
        """
        Generate new prompt candidates using LLM as meta-optimizer.

        Paper implementation: Makes N independent calls to the optimizer LLM
        (each at temperature 1.0) to generate N diverse instructions.

        Returns:
            List of new prompt candidates
        """
        # Format scored prompts for the meta-prompt (keep top 20 as in paper)
        if self.scored_prompts:
            scored_prompts_text = "\n".join([
                f"Score: {sp.score:.1%} | Prompt: \"{sp.prompt}\""
                for sp in sorted(self.scored_prompts, key=lambda x: x.score, reverse=True)
            ])
        else:
            scored_prompts_text = "(No prompts evaluated yet)"

        # Generate N candidates with N independent calls (paper method)
        candidates = []
        seen = set(sp.prompt for sp in self.scored_prompts)

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

            # Generate candidate (temperature=1.0 as in paper for diversity)
            response = self.meta_llm.generate(meta_prompt, temperature=1.0, max_new_tokens=500)

            # Extract the prompt (remove numbering, bullets, etc.)
            candidate = response.strip()

            # Clean up common artifacts
            for prefix in ['1. ', '2. ', '3. ', '4. ', '- ', '* ', 'Prompt: ', '"']:
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix):]

            # Remove trailing quotes
            candidate = candidate.strip().strip('"').strip()

            # Skip if empty or duplicate
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        return candidates

    def optimize(
        self,
        initial_prompts: List[str] = None,
        verbose: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Run OPRO optimization.

        Args:
            initial_prompts: Starting prompts (if None, generate from scratch)
            verbose: Whether to print progress

        Returns:
            Tuple of (best_prompt, optimization_history)
        """
        if initial_prompts is None:
            initial_prompts = [
                "Solve the math problem step by step.",
                "Let's solve this math problem carefully. Show your work and provide the final answer.",
                "Think through this problem step by step and calculate the answer.",
            ]

        if verbose:
            print(f"\n{'='*80}")
            print("OPRO Optimization")
            print(f"{'='*80}\n")

        data_idx = 0

        # Evaluate initial prompts
        if verbose:
            print("Evaluating initial prompts...\n")

        for prompt in initial_prompts:
            score, results = self.evaluate_prompt(prompt, data_idx, return_details=True)
            data_idx = (data_idx + self.minibatch_size) % len(self.evaluator)

            self.scored_prompts.append(ScoredPrompt(prompt=prompt, score=score))

            if verbose:
                print(f"Score: {score:.1%} | Prompt: {prompt}")
                # Show failed examples
                print_failed_examples_opro(results, num_examples=3)

        # Optimization loop
        for iteration in range(self.num_iterations):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
                print(f"{'='*80}\n")

            # Generate new candidates
            if verbose:
                print("Generating new candidates...\n")

            candidates = self.generate_candidates()

            if not candidates:
                if verbose:
                    print("No new candidates generated. Stopping early.")
                break

            # Evaluate new candidates
            for candidate in candidates:
                if verbose:
                    print(f"Evaluating: {candidate[:80]}...")

                score, results = self.evaluate_prompt(candidate, data_idx, return_details=True)
                data_idx = (data_idx + self.minibatch_size) % len(self.evaluator)

                self.scored_prompts.append(ScoredPrompt(prompt=candidate, score=score))

                if verbose:
                    print(f"Score: {score:.1%}")
                    # Show failed examples
                    print_failed_examples_opro(results, num_examples=3)

                # Record history
                self.history.append({
                    'iteration': iteration,
                    'prompt': candidate,
                    'score': score,
                })

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
