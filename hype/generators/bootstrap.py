"""
Method C: Few-Shot Bootstrapping Generator

Generate synthetic exemplars using the best instruction.
Focuses on hard examples (frequently failed) for targeted improvement.

Key insight: Models learn better from examples that "sound like" they would
generate them (self-distillation effect).
"""

import re
import random
from typing import List, Dict, Optional, Any
from hype.data_types import (
    Instruction, Exemplar, EvaluationRecord,
    ComponentScore, GenerationResult, ComponentSource
)


BOOTSTRAP_PROMPT_TEMPLATE = """Generate high-quality few-shot examples for a math problem solver.

INSTRUCTION THAT WILL USE THESE EXAMPLES:
<<<
{instruction}
>>>

REFERENCE PROBLEMS (use similar style and difficulty):
{reference_problems}

YOUR TASK:
Generate {num_examples} new Q&A pairs that would help a model learn to solve math problems correctly.

REQUIREMENTS:
1. Each example should show clear step-by-step reasoning
2. The final answer should be a specific number
3. Format each answer to END with the numerical answer
4. Make examples diverse in problem type and difficulty
5. Ensure mathematical accuracy

FORMAT (output exactly this, one example per block):

Q: [Your math word problem here]
A: [Step-by-step solution ending with the final numerical answer]

---

Q: [Next problem]
A: [Solution]

---

(Continue for {num_examples} examples)

Generate the examples now:
"""


class BootstrapGenerator:
    """
    Generate synthetic few-shot exemplars.

    Strategy:
    1. Use best instruction as context
    2. Sample from hard examples (high error rate)
    3. Generate new Q&A pairs using LLM
    4. Filter for quality and correctness
    """

    def __init__(
        self,
        llm_client,
        training_data: List[Dict] = None,  # List of {'question': str, 'answer': str}
        prompt_template: str = None,
        num_exemplars_to_generate: int = 3,
        examples_per_exemplar: int = 5,
        temperature: float = 0.7,
        seed: int = None,
    ):
        """
        Args:
            llm_client: LLM client for generation
            training_data: Training examples to sample references from
            prompt_template: Template for bootstrap generation
            num_exemplars_to_generate: How many new exemplar sets to create
            examples_per_exemplar: Number of Q&A pairs per exemplar
            temperature: Generation temperature
            seed: Random seed for reproducibility
        """
        self.llm = llm_client
        self.training_data = training_data or []
        self.prompt_template = prompt_template or BOOTSTRAP_PROMPT_TEMPLATE
        self.num_exemplars = num_exemplars_to_generate
        self.examples_per_exemplar = examples_per_exemplar
        self.temperature = temperature
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "bootstrap"

    @property
    def requires_llm(self) -> bool:
        return True

    def set_training_data(self, data: List[Dict]) -> None:
        """Set training data (can be called after init)"""
        self.training_data = data

    def generate(
        self,
        instructions: List[Instruction],
        exemplars: List[Exemplar],
        instruction_scores: Dict[int, ComponentScore],
        exemplar_scores: Dict[int, ComponentScore],
        evaluation_records: List[EvaluationRecord],
        generation: int,
        hard_example_indices: List[int] = None,  # Indices of frequently failed examples
        **kwargs
    ) -> GenerationResult:
        """
        Generate new exemplars using bootstrapping.

        Uses the best instruction as context and focuses on hard examples.
        """
        if not self.training_data:
            print("Warning: No training data set for bootstrap generator")
            return GenerationResult(metadata={"error": "no_training_data"})

        # Get best instruction
        best_instruction = self._get_best_instruction(instructions, instruction_scores)
        if not best_instruction:
            return GenerationResult(metadata={"error": "no_scored_instructions"})

        # Get reference problems (focus on hard ones if available)
        reference_problems = self._select_reference_problems(hard_example_indices)

        new_exemplars = []
        metadata = {
            "best_instruction_id": best_instruction.id,
            "reference_indices": [],
            "generated_exemplars": []
        }

        next_id = max((e.id for e in exemplars), default=0) + 1

        for i in range(self.num_exemplars):
            # Sample different reference problems for each exemplar
            refs = self._sample_references(reference_problems, self.examples_per_exemplar)
            metadata["reference_indices"].append([r.get('index', -1) for r in refs])

            # Generate new exemplar
            exemplar_text = self._generate_exemplar(best_instruction, refs)
            if not exemplar_text:
                continue

            # Clean and validate
            exemplar_text = self._clean_exemplar(exemplar_text)
            if not self._is_valid(exemplar_text, exemplars):
                continue

            new_ex = Exemplar(
                id=next_id,
                text=exemplar_text,
                source=ComponentSource.BOOTSTRAP,
                generation=generation,
                parent_ids=[]  # Bootstrapped from training data, not existing exemplars
            )
            new_exemplars.append(new_ex)
            next_id += 1

            metadata["generated_exemplars"].append({
                "id": new_ex.id,
                "length": len(exemplar_text),
                "num_qa_pairs": exemplar_text.count("Q:")
            })

        return GenerationResult(
            new_instructions=[],
            new_exemplars=new_exemplars,
            metadata=metadata
        )

    def _get_best_instruction(
        self,
        instructions: List[Instruction],
        scores: Dict[int, ComponentScore]
    ) -> Optional[Instruction]:
        """Get the best-scoring instruction"""
        best_inst = None
        best_score = -1

        for inst in instructions:
            score = scores.get(inst.id)
            if score and score.score > best_score:
                best_score = score.score
                best_inst = inst

        return best_inst

    def _select_reference_problems(
        self,
        hard_indices: Optional[List[int]]
    ) -> List[Dict]:
        """Select reference problems, prioritizing hard examples"""
        if not self.training_data:
            return []

        references = []

        # Add hard examples first
        if hard_indices:
            for idx in hard_indices:
                if 0 <= idx < len(self.training_data):
                    ref = self.training_data[idx].copy()
                    ref['index'] = idx
                    ref['is_hard'] = True
                    references.append(ref)

        # Add random examples
        remaining_indices = set(range(len(self.training_data)))
        if hard_indices:
            remaining_indices -= set(hard_indices)

        random_indices = self.rng.sample(
            list(remaining_indices),
            min(len(remaining_indices), 20)  # Sample up to 20 random
        )
        for idx in random_indices:
            ref = self.training_data[idx].copy()
            ref['index'] = idx
            ref['is_hard'] = False
            references.append(ref)

        return references

    def _sample_references(
        self,
        all_references: List[Dict],
        n: int
    ) -> List[Dict]:
        """Sample n references, prioritizing hard examples"""
        hard = [r for r in all_references if r.get('is_hard', False)]
        easy = [r for r in all_references if not r.get('is_hard', False)]

        # Take up to half from hard examples
        num_hard = min(len(hard), n // 2)
        num_easy = n - num_hard

        sampled = []
        if hard:
            sampled.extend(self.rng.sample(hard, min(num_hard, len(hard))))
        if easy:
            sampled.extend(self.rng.sample(easy, min(num_easy, len(easy))))

        return sampled[:n]

    def _generate_exemplar(
        self,
        instruction: Instruction,
        references: List[Dict]
    ) -> Optional[str]:
        """Generate a new exemplar using LLM"""
        # Format reference problems
        ref_text = self._format_references(references)

        prompt = self.prompt_template.format(
            instruction=instruction.text,
            reference_problems=ref_text,
            num_examples=self.examples_per_exemplar
        )

        try:
            response = self.llm.generate(
                prompt,
                temperature=self.temperature,
                max_new_tokens=2000
            )
            return response.strip()
        except Exception as e:
            print(f"Warning: Bootstrap generation failed: {e}")
            return None

    def _format_references(self, references: List[Dict]) -> str:
        """Format reference problems for the prompt"""
        formatted = []
        for i, ref in enumerate(references, 1):
            question = ref.get('question', '')
            answer = ref.get('answer', '')
            # Extract just the solution part (before ####)
            if '####' in answer:
                solution = answer.split('####')[0].strip()
                final = answer.split('####')[1].strip()
                formatted.append(f"Example {i}:\nQ: {question}\nA: {solution}\nFinal answer: {final}")
            else:
                formatted.append(f"Example {i}:\nQ: {question}\nA: {answer}")

        return "\n\n".join(formatted)

    def _clean_exemplar(self, text: str) -> str:
        """Clean up generated exemplar text"""
        # Remove common artifacts
        text = re.sub(r'^(Here are|Generated examples:?)[\s:]*\n*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks

        # Normalize separators
        text = re.sub(r'-{3,}', '\n', text)  # Replace --- with newline
        text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines

        # Parse Q&A pairs and reformat
        qa_pairs = self._extract_qa_pairs(text)
        if qa_pairs:
            formatted = []
            for q, a in qa_pairs:
                formatted.append(f"Q: {q}\nA: {a}")
            return "\n\n".join(formatted)

        return text.strip()

    def _extract_qa_pairs(self, text: str) -> List[tuple]:
        """Extract Q&A pairs from text"""
        pairs = []

        # Pattern: Q: ... A: ...
        pattern = r'Q:\s*(.*?)(?=\nA:)\nA:\s*(.*?)(?=\n\nQ:|\n---|\Z)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for q, a in matches:
            q = q.strip()
            a = a.strip()
            if q and a and len(q) > 10 and len(a) > 10:
                pairs.append((q, a))

        return pairs

    def _is_valid(self, text: str, existing: List[Exemplar]) -> bool:
        """Check if exemplar is valid and not duplicate"""
        if len(text) < 50:
            return False
        if len(text) > 5000:
            return False

        # Must have at least 2 Q&A pairs
        if text.count("Q:") < 2:
            return False

        # Check for duplicates (simple text overlap check)
        text_lower = text.lower()
        for ex in existing:
            overlap = len(set(text_lower.split()) & set(ex.text.lower().split()))
            if overlap / max(len(text_lower.split()), 1) > 0.7:
                return False

        return True
