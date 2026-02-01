"""
Soft Prompt representation for InstructZero-style optimization.

The soft prompt is a low-dimensional vector (default 10D) that gets
projected to higher dimensions and used to condition an LLM to generate
task-specific instructions.
"""

import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SoftPromptSpace:
    """
    Manages the soft prompt latent space and projections.

    Following InstructZero:
    - Intrinsic dimension d=10 (soft prompt lives here)
    - Random projection matrix A projects to higher dim d'
    - The projected vector conditions the LLM instruction generator
    """

    def __init__(
        self,
        intrinsic_dim: int = 10,
        projection_dim: int = 50,
        init_scale: float = 5.0,
        seed: Optional[int] = None,
        device: str = "cuda"
    ):
        """
        Initialize soft prompt space.

        Args:
            intrinsic_dim: Dimension of soft prompt (default 10)
            projection_dim: Dimension after random projection (default 50)
            init_scale: Scale for uniform initialization [-scale, scale]
            seed: Random seed for reproducibility
            device: Torch device
        """
        self.intrinsic_dim = intrinsic_dim
        self.projection_dim = projection_dim
        self.init_scale = init_scale
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Random projection matrix (fixed during optimization)
        # Each entry sampled from N(0, 1/sqrt(d'))
        self.projection_matrix = torch.randn(
            intrinsic_dim, projection_dim,
            device=device
        ) / np.sqrt(projection_dim)

        logger.info(f"SoftPromptSpace: {intrinsic_dim}D → {projection_dim}D projection")

    def sample_initial(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample initial soft prompts uniformly.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tensor of shape (n_samples, intrinsic_dim)
        """
        return torch.empty(
            n_samples, self.intrinsic_dim, device=self.device
        ).uniform_(-self.init_scale, self.init_scale)

    def project(self, soft_prompt: torch.Tensor) -> torch.Tensor:
        """
        Project soft prompt to higher dimension.

        Args:
            soft_prompt: Tensor of shape (..., intrinsic_dim)

        Returns:
            Tensor of shape (..., projection_dim)
        """
        return soft_prompt @ self.projection_matrix

    def get_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get bounds for BO optimization.

        Returns:
            Tuple of (lower_bounds, upper_bounds) each of shape (intrinsic_dim,)
        """
        lb = torch.full((self.intrinsic_dim,), -self.init_scale, device=self.device)
        ub = torch.full((self.intrinsic_dim,), self.init_scale, device=self.device)
        return lb, ub


class SoftPromptToText:
    """
    Converts projected soft prompt vectors to text instructions using an LLM.

    The LLM is given:
    1. The projected soft prompt as a "context embedding"
    2. A few exemplars from the task
    3. A meta-prompt asking it to generate a task instruction

    This follows InstructZero's approach of using in-context learning
    to decode soft prompts into human-readable instructions.
    """

    # Meta-prompt template for instruction generation
    META_PROMPT = """You are an expert at creating clear, effective instructions for solving math word problems.

Based on the context and examples below, generate a concise instruction that will help solve similar problems.

Examples from the task:
{exemplars}

Context embedding (influences your instruction style):
{embedding_repr}

Generate a clear, step-by-step instruction for solving math word problems.
The instruction should be 1-3 sentences that guide the solver through the reasoning process.

Instruction:"""

    def __init__(
        self,
        llm_client,
        exemplars: list[dict],
        n_exemplars: int = 3,
        temperature: float = 0.7
    ):
        """
        Initialize soft prompt to text converter.

        Args:
            llm_client: LLM client for generating instructions
            exemplars: List of task exemplars with 'question' and 'answer' keys
            n_exemplars: Number of exemplars to include in prompt
            temperature: Generation temperature (higher = more diverse)
        """
        self.llm_client = llm_client
        self.exemplars = exemplars
        self.n_exemplars = n_exemplars
        self.temperature = temperature

    def _format_exemplars(self, indices: Optional[list[int]] = None) -> str:
        """Format exemplars for the meta-prompt."""
        if indices is None:
            indices = list(range(min(self.n_exemplars, len(self.exemplars))))

        formatted = []
        for i, idx in enumerate(indices[:self.n_exemplars]):
            ex = self.exemplars[idx]
            formatted.append(f"Example {i+1}:\nQ: {ex['question']}\nA: {ex['answer']}")

        return "\n\n".join(formatted)

    def _embedding_to_text_repr(self, embedding: torch.Tensor) -> str:
        """
        Convert embedding to text representation for the LLM.

        We use a creative approach: map embedding dimensions to
        instruction style attributes (clarity, detail level, etc.)
        """
        if embedding.dim() > 1:
            embedding = embedding.squeeze()

        # Normalize to [0, 1] range
        emb_np = embedding.detach().cpu().numpy()
        emb_normalized = (emb_np - emb_np.min()) / (emb_np.max() - emb_np.min() + 1e-8)

        # Map to style attributes
        attributes = []

        if len(emb_normalized) >= 5:
            if emb_normalized[0] > 0.6:
                attributes.append("detailed step-by-step")
            elif emb_normalized[0] < 0.4:
                attributes.append("concise")

            if emb_normalized[1] > 0.6:
                attributes.append("use equations")
            elif emb_normalized[1] < 0.4:
                attributes.append("use words")

            if emb_normalized[2] > 0.6:
                attributes.append("check your work")

            if emb_normalized[3] > 0.6:
                attributes.append("identify key information first")

            if emb_normalized[4] > 0.5:
                attributes.append("show reasoning")

        if not attributes:
            attributes = ["clear", "systematic"]

        return f"Style: {', '.join(attributes)}"

    def decode(self, soft_prompt_projected: torch.Tensor) -> str:
        """
        Decode projected soft prompt to instruction text.

        Args:
            soft_prompt_projected: Projected soft prompt vector

        Returns:
            Generated instruction string
        """
        exemplars_text = self._format_exemplars()
        embedding_repr = self._embedding_to_text_repr(soft_prompt_projected)

        prompt = self.META_PROMPT.format(
            exemplars=exemplars_text,
            embedding_repr=embedding_repr
        )

        response = self.llm_client.generate(
            prompt,
            max_new_tokens=150,
            temperature=self.temperature
        )

        # Clean up response
        instruction = response.strip()
        # Remove any leading "Instruction:" if the model repeated it
        if instruction.lower().startswith("instruction:"):
            instruction = instruction[12:].strip()

        # Ensure it's not too long
        sentences = instruction.split(". ")
        if len(sentences) > 4:
            instruction = ". ".join(sentences[:4]) + "."

        return instruction

    def decode_batch(self, soft_prompts_projected: torch.Tensor) -> list[str]:
        """
        Decode batch of projected soft prompts to instructions.

        Args:
            soft_prompts_projected: Tensor of shape (batch, projection_dim)

        Returns:
            List of instruction strings
        """
        instructions = []
        for i in range(len(soft_prompts_projected)):
            instruction = self.decode(soft_prompts_projected[i])
            instructions.append(instruction)
        return instructions
