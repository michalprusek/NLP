"""
OPRO: Optimization by PROmpting for GSM8K

Based on "Large Language Models as Optimizers"
https://arxiv.org/abs/2309.03409
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import random
import json
import os


# Load meta-prompt template
META_PROMPT_PATH = Path(__file__).parent / 'prompts' / 'opro_meta.txt'
META_PROMPT_TEMPLATE = META_PROMPT_PATH.read_text(encoding='utf-8')


def bucket_score(score: float, num_buckets: int = 20) -> float:
    """Bucketize score to nearest 5% (20 buckets)"""
    bucket_size = 1.0 / num_buckets
    return round(score / bucket_size) * bucket_size


@dataclass
class ScoredPrompt:
    """A prompt with its evaluation score"""
    prompt: str
    score: float

    def __repr__(self):
        return f"Score: {self.score:.1%} | Prompt: {self.prompt}"


class OPROOptimizer:
    """OPRO optimizer for GSM8K"""

    def __init__(
        self,
        task_llm_client,
        evaluator,
        num_iterations: int = 200,
        num_candidates_per_iter: int = 8,
        minibatch_size: int = 261,
        keep_top_k: int = 20,
        meta_llm_client=None,
        task_max_tokens: int = 2048,
        meta_max_tokens: int = 500,
        total_budget: int = None,  # Total LLM evaluation budget (None = unlimited)
        max_meta_prompts: int = 20,  # Max prompts to show in meta-prompt (prevents context overflow)
        max_prompt_length: int = 300,  # Max chars per prompt in meta-context
    ):
        self.task_llm = task_llm_client
        self.meta_llm = meta_llm_client if meta_llm_client is not None else task_llm_client
        self.evaluator = evaluator
        self.num_iterations = num_iterations
        self.num_candidates_per_iter = num_candidates_per_iter
        self.minibatch_size = minibatch_size
        self.keep_top_k = keep_top_k
        self.task_max_tokens = task_max_tokens
        self.meta_max_tokens = meta_max_tokens
        self.total_budget = total_budget
        self.max_meta_prompts = max_meta_prompts
        self.max_prompt_length = max_prompt_length

        self.scored_prompts: List[ScoredPrompt] = []
        self.history = []
        self.budget_used = 0  # Track LLM evaluations

        # Create fixed evaluation set (same for all prompts)
        dataset_size = len(evaluator)
        eval_size = min(minibatch_size, dataset_size)
        eval_indices = random.sample(range(dataset_size), eval_size)

        self.fixed_eval_set = []
        for idx in eval_indices:
            batch = evaluator.get_batch(idx, 1)
            if batch:
                self.fixed_eval_set.append(batch[0])

        print(f"Fixed evaluation set: {len(self.fixed_eval_set)} examples ({100*len(self.fixed_eval_set)/dataset_size:.1f}%)")

    def _get_random_examples(self, num_examples: int = 3) -> str:
        """Get random examples for meta-prompt (paper uses 3)"""
        dataset_size = len(self.evaluator)
        indices = random.sample(range(dataset_size), min(num_examples, dataset_size))

        examples = []
        for idx in indices:
            batch = self.evaluator.get_batch(idx, 1)
            if batch:
                ex = batch[0]
                examples.append(f"input:\nQ: {ex['question']}\nA: <INS>\noutput:\n {ex['answer']}")

        return "\n\n".join(examples)

    def _is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        if self.total_budget is None:
            return False
        return self.budget_used >= self.total_budget

    def _can_afford_evaluation(self) -> bool:
        """Check if we can afford one more evaluation."""
        if self.total_budget is None:
            return True
        return (self.budget_used + self.minibatch_size) <= self.total_budget

    def evaluate_prompt(
        self,
        prompt: str,
        save_eval_json: bool = False,
        eval_output_dir: str = None,
        iteration: int = None,
        candidate_idx: int = None
    ) -> Tuple[float, Dict]:
        """Evaluate prompt on fixed evaluation set"""
        batch = self.fixed_eval_set

        # Track budget
        self.budget_used += len(batch)

        # Generate answers
        questions = [ex['question'] for ex in batch]
        formatted_prompts = [f"Question: {q}\n\n{prompt}\n\nAnswer:" for q in questions]
        outputs = self.task_llm.generate_batch(
            formatted_prompts, temperature=0.0, max_new_tokens=self.task_max_tokens
        )

        # Evaluate
        indices = [ex['idx'] for ex in batch]
        results = self.evaluator.evaluate_batch(outputs, indices)
        score = results['accuracy']

        # Save detailed JSON if requested
        if save_eval_json and eval_output_dir:
            os.makedirs(eval_output_dir, exist_ok=True)

            if iteration is not None and candidate_idx is not None:
                filename = f"eval_iter{iteration:02d}_cand{candidate_idx:02d}.json"
            else:
                filename = f"eval_{datetime.now().strftime('%H%M%S')}.json"

            eval_data = {
                "prompt": prompt,
                "score": score,
                "iteration": iteration,
                "candidate_idx": candidate_idx,
                "timestamp": datetime.now().isoformat(),
                "examples": []
            }

            for i, (q, fp, output, detail) in enumerate(zip(
                questions, formatted_prompts, outputs, results.get('details', [])
            )):
                eval_data["examples"].append({
                    "idx": detail.get('idx', i),
                    "question": q,
                    "formatted_prompt": fp,
                    "raw_response": output,
                    "extracted_answer": detail.get('predicted'),
                    "ground_truth": detail.get('ground_truth'),
                    "correct": detail.get('correct', False)
                })

            filepath = os.path.join(eval_output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {filepath}")

        return score, results

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to max_prompt_length chars"""
        if len(prompt) <= self.max_prompt_length:
            return prompt
        return prompt[:self.max_prompt_length - 3] + "..."

    def generate_candidates(
        self,
        verbose_meta: bool = False,
        save_meta_json: bool = False,
        meta_output_dir: str = None,
        iteration: int = None
    ) -> List[str]:
        """Generate new prompt candidates"""
        # Format scored prompts context (limited to prevent context overflow)
        if self.scored_prompts:
            # Sort by score ascending (worst first, best last - as OPRO paper recommends)
            sorted_prompts = sorted(self.scored_prompts, key=lambda x: x.score)
            # Take only the top prompts to fit in context
            limited_prompts = sorted_prompts[-self.max_meta_prompts:]
            scored_prompts_text = "\n".join([
                f"text: {self._truncate_prompt(sp.prompt)}\nscore: {int(bucket_score(sp.score) * 100)}"
                for sp in limited_prompts
            ])
        else:
            scored_prompts_text = "(No prompts yet)"

        candidates = []
        seen = set(sp.prompt for sp in self.scored_prompts)

        # For JSON saving
        meta_json_data = None
        if save_meta_json and meta_output_dir:
            meta_json_data = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'num_candidates': self.num_candidates_per_iter,
                'scored_prompts_context': scored_prompts_text,
                'calls': []
            }

        for i in range(self.num_candidates_per_iter):
            # Format meta-prompt
            example_problems = self._get_random_examples(num_examples=3)
            meta_prompt = META_PROMPT_TEMPLATE.format(
                scored_prompts=scored_prompts_text,
                example_problems=example_problems,
            )

            if verbose_meta:
                print(f"\n{'='*60}")
                print(f"META-MODEL CALL {i+1}/{self.num_candidates_per_iter}")
                print(f"{'='*60}")
                print(f"\nMETA-PROMPT:\n{meta_prompt}")
                print(f"\n{'-'*60}")

            # Generate candidate
            response = self.meta_llm.generate(
                meta_prompt, temperature=1.0, max_new_tokens=self.meta_max_tokens
            )

            if verbose_meta:
                print(f"\nRESPONSE:\n{response}")
                print(f"\n{'-'*60}")

            # Extract prompt from [...] brackets
            candidate = response.strip()
            if '[' in candidate and ']' in candidate:
                start = candidate.find('[')
                end = candidate.find(']', start)
                if end > start:
                    candidate = candidate[start + 1:end].strip()

            if verbose_meta:
                print(f"\nEXTRACTED: {candidate if candidate else '(empty)'}")
                print(f"{'='*60}\n")

            # Save to JSON data
            if meta_json_data is not None:
                meta_json_data['calls'].append({
                    'call_idx': i,
                    'formatted_meta_prompt': meta_prompt,
                    'raw_response': response,
                    'extracted_prompt': candidate if candidate else None
                })

            # Add if unique
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        # Save meta JSON
        if meta_json_data is not None and meta_output_dir:
            os.makedirs(meta_output_dir, exist_ok=True)
            filename = f"meta_iter{iteration:02d}.json" if iteration is not None else "meta.json"
            filepath = os.path.join(meta_output_dir, filename)

            meta_json_data['extracted_prompts'] = candidates
            meta_json_data['num_unique_candidates'] = len(candidates)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(meta_json_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {filepath}")

        return candidates

    def optimize(
        self,
        initial_prompts: List[str] = None,
        verbose: bool = True,
        save_eval_json: bool = False,
        eval_output_dir: str = None,
        verbose_meta: bool = False
    ) -> Tuple[str, List[Dict]]:
        """Run OPRO optimization"""
        if initial_prompts is None:
            initial_prompts = [
                "",
                "Solve the following problem.",
                "Let's solve the problem.",
            ]

        if verbose:
            print(f"\n{'='*60}")
            print("OPRO Optimization")
            if self.total_budget:
                print(f"Budget: {self.total_budget} LLM evaluations")
            print(f"{'='*60}\n")

        if save_eval_json and eval_output_dir:
            os.makedirs(eval_output_dir, exist_ok=True)
            print(f"Saving to: {eval_output_dir}\n")

        # Evaluate initial prompts
        if verbose:
            print("Evaluating initial prompts...\n")

        for idx, prompt in enumerate(initial_prompts):
            # Check budget before evaluation
            if not self._can_afford_evaluation():
                if verbose:
                    print(f"Budget exhausted ({self.budget_used}/{self.total_budget}). Stopping initial evaluation.")
                break

            score, _ = self.evaluate_prompt(
                prompt,
                save_eval_json=save_eval_json,
                eval_output_dir=eval_output_dir,
                iteration=-1,
                candidate_idx=idx
            )
            self.scored_prompts.append(ScoredPrompt(prompt=prompt, score=score))
            self.history.append({'iteration': -1, 'prompt': prompt, 'score': score})

            if verbose:
                budget_str = f" [budget: {self.budget_used}/{self.total_budget}]" if self.total_budget else ""
                print(f"Score: {score:.1%} | {prompt if prompt else '(empty)'}{budget_str}")

        # Optimization loop
        budget_exhausted = False
        for iteration in range(self.num_iterations):
            # Check budget at start of iteration
            if self._is_budget_exhausted():
                if verbose:
                    print(f"\nBudget exhausted ({self.budget_used}/{self.total_budget}). Stopping.")
                budget_exhausted = True
                break

            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
                if self.total_budget:
                    print(f"Budget: {self.budget_used}/{self.total_budget}")
                print(f"{'='*60}\n")

            # Generate candidates
            if verbose:
                print("Generating candidates...\n")

            candidates = self.generate_candidates(
                verbose_meta=verbose_meta,
                save_meta_json=save_eval_json,
                meta_output_dir=eval_output_dir,
                iteration=iteration
            )

            if not candidates:
                if verbose:
                    print("No candidates generated. Stopping.")
                break

            # Evaluate candidates
            for i, candidate in enumerate(candidates):
                # Check budget before each evaluation
                if not self._can_afford_evaluation():
                    if verbose:
                        print(f"Budget exhausted ({self.budget_used}/{self.total_budget}). Stopping evaluation.")
                    budget_exhausted = True
                    break

                if verbose:
                    print(f"Evaluating:\n{candidate}")

                score, _ = self.evaluate_prompt(
                    candidate,
                    save_eval_json=save_eval_json,
                    eval_output_dir=eval_output_dir,
                    iteration=iteration,
                    candidate_idx=i
                )
                self.scored_prompts.append(ScoredPrompt(prompt=candidate, score=score))
                self.history.append({'iteration': iteration, 'prompt': candidate, 'score': score})

                if verbose:
                    budget_str = f" [budget: {self.budget_used}/{self.total_budget}]" if self.total_budget else ""
                    print(f"Score: {score:.1%}{budget_str}")

            if budget_exhausted:
                break

            # Keep top-k
            self.scored_prompts.sort(key=lambda x: x.score, reverse=True)
            self.scored_prompts = self.scored_prompts[:self.keep_top_k]

            if verbose:
                print(f"\nTop 3 prompts:")
                for i, sp in enumerate(self.scored_prompts[:3]):
                    print(f"  {i+1}. {sp}")

        # Return best
        if not self.scored_prompts:
            if verbose:
                print("No prompts evaluated. Returning empty.")
            return "", self.history

        best = max(self.scored_prompts, key=lambda x: x.score)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best prompt (score: {best.score:.1%}):")
            print(best.prompt)
            if self.total_budget:
                print(f"Total budget used: {self.budget_used}/{self.total_budget}")
            print(f"{'='*60}\n")

        return best.prompt, self.history


# Backwards compatibility alias
OPRO = OPROOptimizer
