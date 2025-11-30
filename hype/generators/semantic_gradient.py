"""
Method A: Semantic Gradient Generator

LLM-based improvement of instructions using error analysis.
Analyzes failure patterns and generates improved instruction variants.

This is the most expensive method (requires LLM calls) but produces
the highest quality improvements.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from hype.data_types import (
    Instruction, Exemplar, EvaluationRecord,
    ComponentScore, GenerationResult, ComponentSource
)


# Default prompt templates
GRADIENT_PROMPT_TEMPLATE = """You are a CRITIC for Prompt Optimization.
Analyze the current instruction's performance and provide SHORT, ACTIONABLE feedback.

INSTRUCTION BEING EVALUATED:
<<<
{instruction}
>>>

PERFORMANCE SUMMARY:
- Accuracy: {accuracy:.2%}
- Budget (samples): {budget}
- Survival depth: {survival_depth} (higher = survived more Hyperband rounds)

FAILURE EXAMPLES (where this instruction led to wrong answers):
{error_examples}

EVALUATION METHOD:
The system extracts the LAST NUMBER from the model's output and compares to ground truth.
Numbers are normalized (removes commas, handles decimals).

YOUR TASK:
Analyze why this instruction causes these failures. Consider:
1. Is the instruction too vague or too specific?
2. Does it guide the model to show clear reasoning steps?
3. Does it ensure the final answer is clearly formatted?

Provide 2-3 key issues with specific, testable improvements.

FORMAT YOUR RESPONSE AS JSON:
{{
  "issues": [
    {{
      "title": "[Brief description]",
      "root_cause": "[Why this causes failures]",
      "actions": ["[Specific improvement 1]", "[Specific improvement 2]"]
    }}
  ],
  "summary": "[One-sentence summary of main problem]"
}}

Keep response under 500 tokens.
"""

EDIT_PROMPT_TEMPLATE = """You are an EDITOR for Prompt Optimization.
Apply the critic's feedback to improve the instruction.

ORIGINAL INSTRUCTION:
<<<
{instruction}
>>>

CRITIC'S FEEDBACK:
{gradient}

TASK:
Create an IMPROVED version of the instruction that addresses the critic's feedback.

CONSTRAINTS:
- Keep the instruction concise (2-4 sentences max)
- Be specific and actionable
- Ensure it guides the model to show reasoning AND format the final answer clearly
- Do NOT include meta-text like "Here is the improved instruction:"
- Output ONLY the improved instruction text

IMPROVED INSTRUCTION:
"""


class SemanticGradientGenerator:
    """
    Generate improved instructions using semantic gradients.

    Process:
    1. Select low-performing or high-potential instructions
    2. Analyze their failures using LLM (generate gradient/critique)
    3. Apply gradient to create improved versions (edit step)
    """

    def __init__(
        self,
        llm_client,
        gradient_prompt: str = None,
        edit_prompt: str = None,
        num_to_improve: int = 3,
        temperature_gradient: float = 0.8,
        temperature_edit: float = 0.7,
        max_retries: int = 2,
    ):
        """
        Args:
            llm_client: LLM client for generation (with .generate() method)
            gradient_prompt: Template for generating critique
            edit_prompt: Template for applying critique
            num_to_improve: Number of instructions to improve per call
            temperature_gradient: Temperature for gradient generation (higher = more creative)
            temperature_edit: Temperature for edit step (lower = more focused)
            max_retries: Retries on generation failure
        """
        self.llm = llm_client
        self.gradient_prompt = gradient_prompt or GRADIENT_PROMPT_TEMPLATE
        self.edit_prompt = edit_prompt or EDIT_PROMPT_TEMPLATE
        self.num_to_improve = num_to_improve
        self.temperature_gradient = temperature_gradient
        self.temperature_edit = temperature_edit
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return "semantic_gradient"

    @property
    def requires_llm(self) -> bool:
        return True

    def generate(
        self,
        instructions: List[Instruction],
        exemplars: List[Exemplar],
        instruction_scores: Dict[int, ComponentScore],
        exemplar_scores: Dict[int, ComponentScore],
        evaluation_records: List[EvaluationRecord],
        generation: int,
        error_examples: Dict[int, List[Dict]] = None,  # instruction_id -> list of failures
        **kwargs
    ) -> GenerationResult:
        """
        Generate improved instructions using semantic gradient.

        Strategy: Focus on instructions that:
        1. Survived to high fidelity (promising but not perfect)
        2. Have moderate scores (room for improvement)
        """
        # Select instructions to improve
        candidates = self._select_candidates(instructions, instruction_scores)

        new_instructions = []
        metadata = {
            "candidates_selected": [c.id for c in candidates],
            "gradients": [],
            "improvements": []
        }

        next_id = max(i.id for i in instructions) + 1

        for inst in candidates:
            score = instruction_scores.get(inst.id, ComponentScore(inst.id, 0))

            # Collect error examples for this instruction
            inst_errors = self._collect_errors(inst.id, evaluation_records, error_examples)

            # Generate gradient (critique)
            gradient = self._generate_gradient(inst, score, inst_errors)
            if not gradient:
                continue

            metadata["gradients"].append({
                "instruction_id": inst.id,
                "gradient": gradient
            })

            # Apply gradient (edit)
            improved_text = self._apply_gradient(inst, gradient)
            if not improved_text:
                continue

            # Clean and validate
            improved_text = self._clean_instruction(improved_text)
            if not self._is_valid(improved_text, instructions):
                continue

            new_inst = Instruction(
                id=next_id,
                text=improved_text,
                source=ComponentSource.SEMANTIC_GRADIENT,
                generation=generation,
                parent_ids=[inst.id]
            )
            new_instructions.append(new_inst)
            next_id += 1

            metadata["improvements"].append({
                "parent_id": inst.id,
                "new_id": new_inst.id,
                "original": inst.text[:100],
                "improved": improved_text[:100]
            })

        return GenerationResult(
            new_instructions=new_instructions,
            new_exemplars=[],
            metadata=metadata
        )

    def _select_candidates(
        self,
        instructions: List[Instruction],
        scores: Dict[int, ComponentScore]
    ) -> List[Instruction]:
        """
        Select instructions to improve.

        Strategy: Mix of:
        - Mid-tier performers (have potential, need refinement)
        - High survival depth (survived many rounds but not perfect)
        """
        scored = []
        for inst in instructions:
            score = scores.get(inst.id)
            if score is None:
                continue
            # Prioritize: high survival depth + moderate score
            priority = score.max_budget_seen * 0.3 + score.score * 0.7
            scored.append((inst, priority, score))

        # Sort by priority (descending) but skip the very best (they're fine)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take from middle-high range (skip top 10%, take next N)
        start_idx = max(1, len(scored) // 10)
        candidates = [s[0] for s in scored[start_idx:start_idx + self.num_to_improve]]

        return candidates

    def _collect_errors(
        self,
        instruction_id: int,
        records: List[EvaluationRecord],
        error_examples: Optional[Dict[int, List[Dict]]]
    ) -> str:
        """Format error examples for this instruction"""
        if error_examples and instruction_id in error_examples:
            examples = error_examples[instruction_id][:4]
            formatted = []
            for ex in examples:
                formatted.append(
                    f"Question: {ex.get('question', 'N/A')}\n"
                    f"Expected: {ex.get('expected', 'N/A')}\n"
                    f"Got: {ex.get('predicted', 'N/A')}"
                )
            return "\n\n".join(formatted)

        # Fallback: just mention error rate
        inst_records = [r for r in records if r.instruction_id == instruction_id]
        if inst_records:
            avg_error = sum(r.error_rate for r in inst_records) / len(inst_records)
            return f"(Detailed errors not available. Average error rate: {avg_error:.2%})"

        return "(No error data available)"

    def _generate_gradient(
        self,
        instruction: Instruction,
        score: ComponentScore,
        error_examples: str
    ) -> Optional[str]:
        """Generate critique for instruction"""
        prompt = self.gradient_prompt.format(
            instruction=instruction.text,
            accuracy=score.score,
            budget=score.max_budget_seen,
            survival_depth=score.max_budget_seen,
            error_examples=error_examples
        )

        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(
                    prompt,
                    temperature=self.temperature_gradient,
                    max_new_tokens=800
                )
                # Try to parse as JSON for validation
                self._parse_gradient_json(response)
                return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: Gradient generation failed for instruction {instruction.id}: {e}")
                    return None

        return None

    def _parse_gradient_json(self, response: str) -> Dict:
        """Try to extract JSON from gradient response"""
        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"raw": response}

    def _apply_gradient(
        self,
        instruction: Instruction,
        gradient: str
    ) -> Optional[str]:
        """Apply gradient to generate improved instruction"""
        prompt = self.edit_prompt.format(
            instruction=instruction.text,
            gradient=gradient
        )

        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(
                    prompt,
                    temperature=self.temperature_edit,
                    max_new_tokens=500
                )
                return response.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: Edit step failed for instruction {instruction.id}: {e}")
                    return None

        return None

    def _clean_instruction(self, text: str) -> str:
        """Clean up generated instruction text"""
        # Remove common LLM artifacts
        text = re.sub(r'^(Here is|The improved|Improved instruction:?)[\s:]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(IMPROVED INSTRUCTION:?)[\s:]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines
        text = text.strip()

        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text

    def _is_valid(self, text: str, existing: List[Instruction]) -> bool:
        """Check if instruction is valid and not duplicate"""
        if len(text) < 20:
            return False
        if len(text) > 1000:
            return False

        # Check for duplicates
        text_lower = text.lower().strip()
        for inst in existing:
            if inst.text.lower().strip() == text_lower:
                return False
            # Also check for high similarity (simple check)
            if len(set(text_lower.split()) & set(inst.text.lower().split())) / max(len(text_lower.split()), 1) > 0.9:
                return False

        return True
