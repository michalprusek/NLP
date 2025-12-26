"""APE (Automatic Prompt Engineering) Instruction Generator.

Generates diverse instructions using forward pass APE method:
- Show Q/A examples to LLM
- Ask it to generate instruction that would help solve them
- Collect 1000+ unique instructions for VAE training

Supports caching to avoid regenerating instructions each run.
"""

import json
import random
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


class APEInstructionGenerator:
    """Generate instructions using APE forward pass with Qwen.

    The APE forward pass works by:
    1. Sampling Q/A examples from validation data
    2. Asking LLM to generate an instruction that would help solve them
    3. Collecting diverse instructions for VAE training

    This addresses the data scarcity problem (25-39 instructions is too few
    for VAE to learn a good latent manifold).
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "vllm",
    ):
        """Initialize generator.

        Args:
            model: Model name for generation
            backend: LLM backend (vllm, openai, etc.)
        """
        from src.llm_client import create_llm_client

        self.model = model
        self.backend = backend
        self._client = None

    def _get_client(self):
        """Lazy load LLM client."""
        if self._client is None:
            from src.llm_client import create_llm_client
            print(f"Initializing LLM client for APE: {self.model}")
            self._client = create_llm_client(self.model, self.backend)
        return self._client

    def generate_instructions(
        self,
        validation_data: List[dict],
        num_instructions: int = 1000,
        examples_per_prompt: int = 5,
        batch_size: int = 50,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> List[str]:
        """Generate diverse instructions via APE forward pass.

        Args:
            validation_data: List of {"question": ..., "answer": ...} dicts
            num_instructions: Target number of unique instructions
            examples_per_prompt: Q/A examples per generation prompt
            batch_size: Number of prompts to generate in parallel
            temperature: Generation temperature (higher = more diverse)
            verbose: Print progress

        Returns:
            List of unique instruction strings
        """
        client = self._get_client()
        instructions = set()

        if verbose:
            print(f"Generating {num_instructions} instructions via APE forward pass...")
            print(f"  Using {examples_per_prompt} Q/A examples per prompt")
            print(f"  Batch size: {batch_size}")

        # Calculate how many batches we need (with some buffer for duplicates)
        estimated_batches = (num_instructions * 2) // batch_size + 1

        pbar = tqdm(total=num_instructions, desc="APE Generation") if verbose else None

        while len(instructions) < num_instructions:
            # Build batch of prompts
            prompts = []
            for _ in range(batch_size):
                examples = random.sample(validation_data, min(examples_per_prompt, len(validation_data)))
                prompt = self._build_ape_prompt(examples)
                prompts.append(prompt)

            # Generate batch
            responses = client.generate_batch(
                prompts,
                max_tokens=150,
                temperature=temperature,
            )

            # Extract instructions from responses
            for response in responses:
                instruction = self._parse_instruction(response)
                if instruction and len(instruction) > 10:  # Filter very short
                    old_len = len(instructions)
                    instructions.add(instruction)
                    if pbar and len(instructions) > old_len:
                        pbar.update(1)

            if pbar:
                pbar.set_postfix({"unique": len(instructions)})

        if pbar:
            pbar.close()

        result = list(instructions)[:num_instructions]

        if verbose:
            print(f"Generated {len(result)} unique instructions")
            print(f"  Sample: {result[0][:80]}...")

        return result

    def _build_ape_prompt(self, examples: List[dict]) -> str:
        """Build APE prompt from Q/A examples.

        Args:
            examples: List of {"question": ..., "answer": ...} dicts

        Returns:
            Prompt string for instruction generation
        """
        qa_text = "\n\n".join([
            f"Q: {ex['question']}\nA: {ex['answer']}"
            for ex in examples
        ])

        return f"""Given these math problem examples with their step-by-step solutions:

{qa_text}

Generate a single, clear instruction (1-2 sentences) that would help someone solve similar math word problems. Focus on the problem-solving approach and reasoning strategy.

Instruction:"""

    def _parse_instruction(self, response: str) -> Optional[str]:
        """Parse instruction from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned instruction string or None
        """
        if not response:
            return None

        # Clean up response
        instruction = response.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Instruction:",
            "Here is an instruction:",
            "The instruction is:",
            "Answer:",
        ]
        for prefix in prefixes_to_remove:
            if instruction.lower().startswith(prefix.lower()):
                instruction = instruction[len(prefix):].strip()

        # Remove quotes if present
        if instruction.startswith('"') and instruction.endswith('"'):
            instruction = instruction[1:-1]
        if instruction.startswith("'") and instruction.endswith("'"):
            instruction = instruction[1:-1]

        # Take only first paragraph (instruction should be concise)
        instruction = instruction.split("\n\n")[0].strip()

        # Limit length (very long responses are usually garbage)
        if len(instruction) > 500:
            instruction = instruction[:500]

        return instruction if instruction else None

    def save_instructions(self, instructions: List[str], path: str) -> None:
        """Save instructions to JSON file.

        Args:
            instructions: List of instruction strings
            path: Path to save file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(instructions, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(instructions)} instructions to {path}")

    @staticmethod
    def load_instructions(path: str) -> List[str]:
        """Load instructions from JSON file.

        Args:
            path: Path to load file

        Returns:
            List of instruction strings
        """
        with open(path, "r", encoding="utf-8") as f:
            instructions = json.load(f)
        print(f"Loaded {len(instructions)} instructions from {path}")
        return instructions

    def generate_or_load(
        self,
        cache_path: str,
        validation_data: List[dict],
        num_instructions: int = 1000,
        examples_per_prompt: int = 5,
        batch_size: int = 50,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> List[str]:
        """Generate instructions or load from cache if exists.

        Args:
            cache_path: Path to cache file
            validation_data: List of {"question": ..., "answer": ...} dicts
            num_instructions: Target number of unique instructions
            examples_per_prompt: Q/A examples per generation prompt
            batch_size: Number of prompts to generate in parallel
            temperature: Generation temperature
            verbose: Print progress

        Returns:
            List of unique instruction strings
        """
        cache_file = Path(cache_path)

        if cache_file.exists():
            return self.load_instructions(cache_path)

        # Generate new instructions
        instructions = self.generate_instructions(
            validation_data=validation_data,
            num_instructions=num_instructions,
            examples_per_prompt=examples_per_prompt,
            batch_size=batch_size,
            temperature=temperature,
            verbose=verbose,
        )

        # Save to cache
        self.save_instructions(instructions, cache_path)

        return instructions
