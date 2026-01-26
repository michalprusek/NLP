"""
GEPA: Genetic-Pareto Prompt Optimization

Combines ideas from evolutionary algorithms with LLM-based reflection for prompt optimization.
Inspired by OPRO, ProTeGi, and genetic algorithm approaches.

Key innovations over OPRO/ProTeGi:
1. Pareto selection - maintains non-dominated candidates balancing multiple metrics
2. Reflection-based mutation - LLM analyzes failures and proposes targeted fixes
3. Trace extraction - captures full reasoning chains for better error analysis
"""
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import random
import json
import re
import sys
from tqdm import tqdm

# Import answer extraction/comparison functions from evaluator module
from gsm8k_evaluator import extract_answer, extract_ground_truth, compare_answers


# Load prompt templates with helpful error messages
PROMPTS_DIR = Path(__file__).parent / 'prompts' / 'gsm8k'


def _load_template(filename: str) -> str:
    """Load a prompt template with helpful error messages."""
    template_path = PROMPTS_DIR / filename
    try:
        return template_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required prompt template not found: {template_path}\n"
            f"Please ensure the file exists. Expected location: {PROMPTS_DIR}"
        ) from None
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Prompt template {template_path} has invalid encoding: {e}\n"
            "Template files must be UTF-8 encoded."
        ) from e


REFLECT_PROMPT_TEMPLATE = _load_template('gepa_reflect.txt')
MUTATE_PROMPT_TEMPLATE = _load_template('gepa_mutate.txt')


@dataclass
class ReasoningTrace:
    """Captured reasoning trace from a model response"""
    question: str
    expected_answer: str
    model_response: str
    extracted_answer: Optional[str]
    is_correct: bool

    def format_for_reflection(self) -> str:
        """Format trace for reflection prompt"""
        status = "CORRECT" if self.is_correct else "INCORRECT"
        return f"""Question: {self.question}
Expected Answer: {self.expected_answer}
Model Response: {self.model_response}
Extracted Answer: {self.extracted_answer}
Status: {status}
"""


@dataclass
class ScoredCandidate:
    """A prompt candidate with multiple evaluation metrics"""
    prompt: str
    accuracy: float  # Primary metric
    avg_response_length: float = 0.0  # Secondary metric (shorter = better for ties)
    num_evaluations: int = 0
    failure_traces: List[ReasoningTrace] = field(default_factory=list)

    def dominates(self, other: 'ScoredCandidate') -> bool:
        """Check if this candidate Pareto-dominates another.

        Dominates if: better or equal on all metrics, strictly better on at least one.
        """
        dominated_by_accuracy = self.accuracy >= other.accuracy
        dominated_by_length = self.avg_response_length <= other.avg_response_length
        strictly_better = (
            self.accuracy > other.accuracy or
            self.avg_response_length < other.avg_response_length
        )
        return dominated_by_accuracy and dominated_by_length and strictly_better

    def __repr__(self):
        return f"Accuracy: {self.accuracy:.1%} | Length: {self.avg_response_length:.0f} | Prompt: {self.prompt}"


@dataclass
class GEPAIteration:
    """Record of one GEPA optimization iteration"""
    iteration: int
    parent_prompt: str
    parent_accuracy: float
    reflection: str
    mutations_generated: int
    candidates_evaluated: int
    pareto_front_size: int
    best_accuracy: float
    budget_used: int
    meta_calls: int


class GEPA:
    """
    GEPA: Genetic-Pareto Prompt Optimizer

    Algorithm:
    1. Initialize with seed prompt(s)
    2. Evaluate candidates on training set, capture traces
    3. Select parent from Pareto front (exploit) or random (explore)
    4. Reflect on failures to diagnose problems
    5. Generate mutations based on reflection
    6. Evaluate mutations, update Pareto front
    7. Repeat until budget exhausted
    """

    def __init__(
        self,
        task_llm_client,
        evaluator,
        meta_llm_client=None,
        # GEPA parameters
        pareto_max_size: int = 10,        # Max candidates in Pareto front
        mutations_per_iteration: int = 4,  # Mutations generated per reflection
        exploit_probability: float = 0.8,  # Probability of selecting from Pareto front
        max_failures_for_reflection: int = 5,  # Max failure traces to include
        # Evaluation parameters
        minibatch_size: int = 64,          # Examples per evaluation
        total_budget: int = 150000,        # Total task LLM calls
        # Token limits
        task_max_tokens: int = 2048,
        meta_max_tokens: int = 1500,
    ):
        self.task_llm = task_llm_client
        self.meta_llm = meta_llm_client if meta_llm_client is not None else task_llm_client
        self.evaluator = evaluator

        # GEPA parameters
        self.pareto_max_size = pareto_max_size
        self.mutations_per_iteration = mutations_per_iteration
        self.exploit_probability = exploit_probability
        self.max_failures_for_reflection = max_failures_for_reflection

        # Evaluation parameters
        self.minibatch_size = minibatch_size
        self.total_budget = total_budget
        self.task_max_tokens = task_max_tokens
        self.meta_max_tokens = meta_max_tokens

        # State
        self.pareto_front: List[ScoredCandidate] = []
        self.all_candidates: List[ScoredCandidate] = []
        self.history: List[GEPAIteration] = []
        self.budget_used = 0
        self.meta_calls = 0

        # Create fixed evaluation set
        self._create_fixed_eval_set()

    def _create_fixed_eval_set(self):
        """Create fixed evaluation set for fair comparison"""
        dataset_size = len(self.evaluator)
        eval_size = min(self.minibatch_size, dataset_size)
        eval_indices = random.sample(range(dataset_size), eval_size)

        self.fixed_eval_set = []
        for idx in eval_indices:
            batch = self.evaluator.get_batch(idx, 1)
            if batch:
                # get_batch returns List[Dict], take first item
                item = batch[0]
                # Extract ground truth from GSM8K answer format (has #### marker)
                ground_truth = extract_ground_truth(item['answer'])
                self.fixed_eval_set.append({
                    'question': item['question'],
                    'answer': ground_truth,  # Pre-extracted for efficiency
                })

    def evaluate_prompt(self, prompt: str, show_progress: bool = False) -> ScoredCandidate:
        """Evaluate a prompt and capture reasoning traces"""
        correct = 0
        total_length = 0
        failure_traces = []

        examples = self.fixed_eval_set
        if show_progress:
            examples = tqdm(examples, desc="Evaluating", leave=False, file=sys.stderr)

        for example in examples:
            question = example['question']
            expected = example['answer']

            # Format prompt with question (Q_end style from OPRO paper)
            full_prompt = f"Q: {question}\n{prompt}\nA:"

            # Get model response
            response = self.task_llm.generate(full_prompt, max_new_tokens=self.task_max_tokens)
            self.budget_used += 1

            # Extract answer and check
            extracted = extract_answer(response)
            is_correct = compare_answers(extracted, expected)

            if is_correct:
                correct += 1
            else:
                # Capture failure trace for reflection
                trace = ReasoningTrace(
                    question=question,
                    expected_answer=expected,
                    model_response=response,
                    extracted_answer=extracted,
                    is_correct=False,
                )
                failure_traces.append(trace)

            total_length += len(response)

        if not self.fixed_eval_set:
            raise RuntimeError(
                "Evaluation set is empty - cannot evaluate prompts. "
                "Check that the GSM8K dataset loaded correctly."
            )
        accuracy = correct / len(self.fixed_eval_set)
        avg_length = total_length / len(self.fixed_eval_set)

        candidate = ScoredCandidate(
            prompt=prompt,
            accuracy=accuracy,
            avg_response_length=avg_length,
            num_evaluations=len(self.fixed_eval_set),
            failure_traces=failure_traces[:self.max_failures_for_reflection],
        )

        return candidate

    def update_pareto_front(self, candidate: ScoredCandidate):
        """Add candidate to Pareto front if non-dominated"""
        # Check if candidate is dominated by any existing member
        for existing in self.pareto_front:
            if existing.dominates(candidate):
                return  # Dominated, don't add

        # Remove any existing candidates dominated by new one
        self.pareto_front = [
            c for c in self.pareto_front
            if not candidate.dominates(c)
        ]

        # Add new candidate
        self.pareto_front.append(candidate)

        # If Pareto front too large, remove lowest accuracy
        if len(self.pareto_front) > self.pareto_max_size:
            self.pareto_front.sort(key=lambda c: c.accuracy, reverse=True)
            self.pareto_front = self.pareto_front[:self.pareto_max_size]

    def select_parent(self) -> ScoredCandidate:
        """Select parent for mutation (exploit vs explore)"""
        if not self.pareto_front:
            raise RuntimeError("Pareto front is empty")

        if random.random() < self.exploit_probability:
            # Exploit: select from Pareto front weighted by accuracy
            weights = [c.accuracy for c in self.pareto_front]
            total = sum(weights)
            if total == 0:
                return random.choice(self.pareto_front)
            weights = [w / total for w in weights]
            return random.choices(self.pareto_front, weights=weights, k=1)[0]
        else:
            # Explore: random from all candidates
            if self.all_candidates:
                return random.choice(self.all_candidates)
            return random.choice(self.pareto_front)

    def reflect_on_failures(self, candidate: ScoredCandidate) -> str:
        """Generate reflection on why prompt failed"""
        if not candidate.failure_traces:
            return "No failures to analyze - prompt achieved perfect accuracy on sample."

        # Format failure traces
        traces_text = "\n---\n".join(
            trace.format_for_reflection()
            for trace in candidate.failure_traces[:self.max_failures_for_reflection]
        )

        # Build reflection prompt
        reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
            current_prompt=candidate.prompt,
            failure_traces=traces_text,
        )

        # Get reflection from meta LLM
        reflection = self.meta_llm.generate(reflect_prompt, max_new_tokens=self.meta_max_tokens)
        self.meta_calls += 1

        return reflection

    def generate_mutations(self, parent: ScoredCandidate, reflection: str) -> List[str]:
        """Generate mutated prompts based on reflection"""
        # Format Pareto front prompts for reference
        pareto_text = ""
        for i, c in enumerate(self.pareto_front[:5], 1):
            pareto_text += f"\n{i}. (Accuracy: {c.accuracy:.1%})\n{c.prompt}\n"

        # Build mutation tags
        mutation_tags = "\n".join(
            f"<prompt_{i}>\n[Your improved prompt here]\n</prompt_{i}>"
            for i in range(3, self.mutations_per_iteration + 1)
        )

        # Build mutation prompt
        mutate_prompt = MUTATE_PROMPT_TEMPLATE.format(
            current_prompt=parent.prompt,
            reflection=reflection,
            pareto_prompts=pareto_text if pareto_text else "None yet.",
            num_mutations=self.mutations_per_iteration,
            mutation_tags=mutation_tags,
        )

        # Get mutations from meta LLM
        response = self.meta_llm.generate(mutate_prompt, max_new_tokens=self.meta_max_tokens * 2)
        self.meta_calls += 1

        # Parse mutations with validation
        mutations = []
        for i in range(1, self.mutations_per_iteration + 1):
            pattern = rf'<prompt_{i}>(.*?)</prompt_{i}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                mutation = match.group(1).strip()
                if mutation and len(mutation) > 20:
                    mutations.append(mutation)

        if not mutations:
            import logging
            logging.getLogger(__name__).warning(
                f"No valid mutations parsed from LLM response. "
                f"Response length: {len(response)}, expected {self.mutations_per_iteration} mutations."
            )

        return mutations

    def optimize(
        self,
        initial_prompts: List[str],
        verbose: bool = True,
    ) -> ScoredCandidate:
        """Run GEPA optimization"""

        if verbose:
            print("=" * 60)
            print("GEPA: Genetic-Pareto Prompt Optimization")
            print(f"Budget: {self.total_budget} task LLM calls")
            print(f"Minibatch size: {self.minibatch_size}")
            print(f"Pareto max size: {self.pareto_max_size}")
            print("=" * 60)

        # Evaluate initial prompts
        if verbose:
            print("\nEvaluating initial prompts...", flush=True)

        for i, prompt in enumerate(initial_prompts):
            if self.budget_used >= self.total_budget:
                break

            if verbose:
                print(f"  [{i+1}/{len(initial_prompts)}] Evaluating...", end=" ", flush=True)

            candidate = self.evaluate_prompt(prompt, show_progress=verbose)
            self.all_candidates.append(candidate)
            self.update_pareto_front(candidate)

            if verbose:
                print(f"Accuracy: {candidate.accuracy:.1%}", flush=True)

        iteration = 0
        while self.budget_used < self.total_budget:
            iteration += 1

            # Select parent
            parent = self.select_parent()

            if verbose:
                print(f"\n--- Iteration {iteration} ---")
                print(f"Budget: {self.budget_used}/{self.total_budget}")
                print(f"Parent accuracy: {parent.accuracy:.1%}")
                print(f"Pareto front size: {len(self.pareto_front)}")

            # Reflect on failures
            reflection = self.reflect_on_failures(parent)

            if verbose:
                print(f"Reflection generated ({len(reflection)} chars)")

            # Generate mutations
            mutations = self.generate_mutations(parent, reflection)

            if verbose:
                print(f"Generated {len(mutations)} mutations")

            # Evaluate mutations
            candidates_this_iter = 0
            for mutation in mutations:
                if self.budget_used >= self.total_budget:
                    break

                candidate = self.evaluate_prompt(mutation)
                self.all_candidates.append(candidate)
                self.update_pareto_front(candidate)
                candidates_this_iter += 1

                if verbose:
                    print(f"  Candidate: {candidate.accuracy:.1%}")

            # Record iteration
            best_in_pareto = max(self.pareto_front, key=lambda c: c.accuracy)

            self.history.append(GEPAIteration(
                iteration=iteration,
                parent_prompt=parent.prompt,
                parent_accuracy=parent.accuracy,
                reflection=reflection,
                mutations_generated=len(mutations),
                candidates_evaluated=candidates_this_iter,
                pareto_front_size=len(self.pareto_front),
                best_accuracy=best_in_pareto.accuracy,
                budget_used=self.budget_used,
                meta_calls=self.meta_calls,
            ))

            if verbose:
                print(f"Best in Pareto front: {best_in_pareto.accuracy:.1%}")

        # Return best candidate
        if not self.pareto_front:
            raise RuntimeError(
                "Pareto front is empty - no prompts were successfully evaluated. "
                "Check LLM client connectivity and prompt format."
            )
        best = max(self.pareto_front, key=lambda c: c.accuracy)

        if verbose:
            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETE")
            print(f"Best accuracy: {best.accuracy:.1%}")
            print(f"Budget used: {self.budget_used}")
            print(f"Meta calls: {self.meta_calls}")
            print(f"Best prompt:\n{best.prompt}")
            print("=" * 60)

        return best

    def evaluate_on_test_set(self, prompt: str, verbose: bool = True) -> float:
        """Evaluate best prompt on full test set"""
        if verbose:
            print("\n" + "=" * 60)
            print("TEST SET EVALUATION")
            print("=" * 60)

        # Get full test set using get_batch
        test_set = self.evaluator.get_batch(0, len(self.evaluator))
        if verbose:
            print(f"Test set: {len(test_set)} examples")
            print("Evaluating best prompt on test set...")

        correct = 0
        for example in test_set:
            question = example['question']
            # Extract ground truth from GSM8K answer format (has #### marker)
            expected = extract_ground_truth(example['answer'])

            full_prompt = f"Q: {question}\n{prompt}\nA:"
            response = self.task_llm.generate(full_prompt, max_new_tokens=self.task_max_tokens)

            extracted = extract_answer(response)
            if compare_answers(extracted, expected):
                correct += 1

        if not test_set:
            raise RuntimeError("Test set is empty - cannot evaluate.")
        accuracy = correct / len(test_set)

        if verbose:
            print(f"\nTest accuracy: {accuracy:.2%}")
            print(f"Test error: {1-accuracy:.2%}")
            print("=" * 60)

        return accuracy

    def save_results(self, output_dir: str = "results") -> Tuple[str, str]:
        """Save optimization results with error handling."""
        output_path = Path(output_dir)

        try:
            output_path.mkdir(exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Cannot create output directory {output_path}: {e}") from e

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get best candidate
        if not self.pareto_front:
            raise RuntimeError("Pareto front is empty - nothing to save.")
        best = max(self.pareto_front, key=lambda c: c.accuracy)

        # Save JSON results
        results = {
            "method": "GEPA",
            "timestamp": timestamp,
            "best_prompt": best.prompt,
            "best_accuracy": best.accuracy,
            "budget_used": self.budget_used,
            "meta_calls": self.meta_calls,
            "pareto_front": [
                {"prompt": c.prompt, "accuracy": c.accuracy, "avg_length": c.avg_response_length}
                for c in self.pareto_front
            ],
            "history": [
                {
                    "iteration": h.iteration,
                    "parent_accuracy": h.parent_accuracy,
                    "best_accuracy": h.best_accuracy,
                    "pareto_size": h.pareto_front_size,
                    "budget_used": h.budget_used,
                }
                for h in self.history
            ],
        }

        json_path = output_path / f"gepa_{timestamp}.json"
        txt_path = output_path / f"gepa_{timestamp}.txt"

        try:
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
        except (OSError, PermissionError) as e:
            # Try fallback to home directory
            fallback_json = Path.home() / f"gepa_backup_{timestamp}.json"
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to save to {json_path}: {e}. Trying fallback: {fallback_json}"
            )
            with open(fallback_json, 'w') as f:
                json.dump(results, f, indent=2)
            json_path = fallback_json

        try:
            with open(txt_path, 'w') as f:
                f.write(f"# GEPA Best Prompt\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Accuracy: {best.accuracy:.2%}\n\n")
                f.write(best.prompt)
        except (OSError, PermissionError) as e:
            fallback_txt = Path.home() / f"gepa_backup_{timestamp}.txt"
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to save to {txt_path}: {e}. Trying fallback: {fallback_txt}"
            )
            with open(fallback_txt, 'w') as f:
                f.write(f"# GEPA Best Prompt\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Accuracy: {best.accuracy:.2%}\n\n")
                f.write(best.prompt)
            txt_path = fallback_txt

        return str(json_path), str(txt_path)
