"""Instruction-only prompt dataclass for Inverse HbBoPs.

Unlike standard HbBoPs which uses (instruction, exemplar) pairs,
Inverse HbBoPs operates on instructions only.
"""

from dataclasses import dataclass


@dataclass
class InstructionOnlyPrompt:
    """A prompt composed of instruction only (no exemplars).

    Attributes:
        instruction: The instruction text
        instruction_id: Unique identifier for this instruction
    """
    instruction: str
    instruction_id: int

    def __str__(self) -> str:
        return self.instruction

    def format_for_eval(self, question: str) -> str:
        """Format prompt for GSM8K Q_end evaluation.

        Uses Q_end style from OPRO paper: instruction comes AFTER the question.

        Args:
            question: The question to solve

        Returns:
            Formatted prompt string
        """
        return f"Q: {question}\n{self.instruction}\nA:"

    def __hash__(self) -> int:
        return hash(self.instruction_id)

    def __eq__(self, other) -> bool:
        if isinstance(other, InstructionOnlyPrompt):
            return self.instruction_id == other.instruction_id
        return False
