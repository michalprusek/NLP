"""
OPRO: Optimization by PROmpting

Based on "Large Language Models as Optimizers"
https://arxiv.org/abs/2309.03409
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ScoredPrompt:
    """A prompt with its evaluation score"""
    prompt: str
    score: float

    def __repr__(self):
        return f"Score: {self.score:.1%} | Prompt: {self.prompt[:50]}..."


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
Generate {num_candidates} NEW instruction prompts that will achieve HIGHER accuracy.

REQUIREMENTS:
1. Each prompt must be 2-4 sentences
2. Must instruct to provide answer in format "#### NUMBER"
3. Should encourage step-by-step reasoning
4. Should be clear and actionable
5. Each prompt must be DIFFERENT from previous ones
6. Explore different instruction strategies

OUTPUT FORMAT:
Write EXACTLY {num_candidates} prompts, one per line.
NO numbering (1., 2., etc.)
NO bullet points (-, *)
NO explanations or meta-text
JUST the instruction prompts themselves.

EXAMPLE OUTPUT (for num_candidates=3):
Solve the math problem step by step and provide the answer as #### NUMBER.
Break down the problem into steps, show your work, and end with #### NUMBER.
Calculate the solution carefully, verify your answer, and format it as #### NUMBER.

NOW GENERATE {num_candidates} NEW PROMPTS:"""

    def __init__(
        self,
        llm_client,
        evaluator,
        num_iterations: int = 10,
        num_candidates_per_iter: int = 4,
        minibatch_size: int = 20,
        keep_top_k: int = 8,
        task_description: str = "Solve math word problems step by step and provide the final numerical answer.",
    ):
        """
        Initialize OPRO optimizer.

        Args:
            llm_client: LLM client for generation and meta-optimization
            evaluator: GSM8K evaluator
            num_iterations: Number of optimization iterations
            num_candidates_per_iter: Number of new candidates to generate per iteration
            minibatch_size: Examples per evaluation
            keep_top_k: Number of top prompts to keep in memory
            task_description: Description of the task
        """
        self.llm = llm_client
        self.evaluator = evaluator
        self.num_iterations = num_iterations
        self.num_candidates_per_iter = num_candidates_per_iter
        self.minibatch_size = minibatch_size
        self.keep_top_k = keep_top_k
        self.task_description = task_description

        self.scored_prompts: List[ScoredPrompt] = []
        self.history = []

        # CRITICAL FIX: Get example problems from dataset for meta-prompt
        # This helps LLM understand the task better
        example_batch = evaluator.get_batch(0, 5)
        self.example_problems = "\n\n".join([
            f"Example {i+1}:\nQ: {ex['question'][:150]}...\nA: {ex['answer'][:100]}..."
            for i, ex in enumerate(example_batch)
        ])

    def evaluate_prompt(self, prompt: str, start_idx: int) -> float:
        """
        Evaluate a prompt on a minibatch.

        Args:
            prompt: Prompt to evaluate
            start_idx: Starting index in dataset

        Returns:
            Accuracy score
        """
        batch = self.evaluator.get_batch(start_idx, self.minibatch_size)

        # Generate answers
        questions = [example['question'] for example in batch]
        prompts = [f"{prompt}\n\nQuestion: {q}\nAnswer:" for q in questions]
        outputs = self.llm.generate_batch(prompts, temperature=0.1)

        # Evaluate
        indices = [example['idx'] for example in batch]
        results = self.evaluator.evaluate_batch(outputs, indices)

        return results['accuracy']

    def generate_candidates(self) -> List[str]:
        """
        Generate new prompt candidates using LLM as meta-optimizer.

        Returns:
            List of new prompt candidates
        """
        # Format scored prompts for the meta-prompt
        if self.scored_prompts:
            scored_prompts_text = "\n".join([
                f"Score: {sp.score:.1%} | Prompt: \"{sp.prompt}\""
                for sp in sorted(self.scored_prompts, key=lambda x: x.score, reverse=True)
            ])
        else:
            scored_prompts_text = "(No prompts evaluated yet)"

        meta_prompt = self.META_PROMPT.format(
            task_description=self.task_description,
            example_problems=self.example_problems,  # CRITICAL FIX: Add examples
            scored_prompts=scored_prompts_text,
            num_candidates=self.num_candidates_per_iter,
        )

        # Generate candidates
        response = self.llm.generate(meta_prompt, temperature=0.9, max_new_tokens=500)

        # Parse candidates (each on a new line)
        candidates = [
            line.strip()
            for line in response.strip().split('\n')
            if line.strip() and not line.strip().startswith(('#', '-', '*', '1.', '2.', '3.', '4.'))
        ]

        # Clean up any numbered prefixes
        cleaned = []
        for c in candidates:
            # Remove common prefixes
            for prefix in ['1. ', '2. ', '3. ', '4. ', '- ', '* ']:
                if c.startswith(prefix):
                    c = c[len(prefix):]
            cleaned.append(c.strip())

        # Return up to num_candidates unique prompts
        unique_candidates = []
        seen = set(sp.prompt for sp in self.scored_prompts)

        for c in cleaned:
            if c and c not in seen:
                unique_candidates.append(c)
                seen.add(c)
                if len(unique_candidates) >= self.num_candidates_per_iter:
                    break

        return unique_candidates

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
            score = self.evaluate_prompt(prompt, data_idx)
            data_idx = (data_idx + self.minibatch_size) % len(self.evaluator)

            self.scored_prompts.append(ScoredPrompt(prompt=prompt, score=score))

            if verbose:
                print(f"Score: {score:.1%} | Prompt: {prompt}")

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

                score = self.evaluate_prompt(candidate, data_idx)
                data_idx = (data_idx + self.minibatch_size) % len(self.evaluator)

                self.scored_prompts.append(ScoredPrompt(prompt=candidate, score=score))

                if verbose:
                    print(f"Score: {score:.1%}\n")

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
