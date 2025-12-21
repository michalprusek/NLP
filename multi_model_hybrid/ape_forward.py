"""
APE Forward Pass for instruction generation.

Based on Zhou et al. 2023 "Large Language Models Are Human-Level Prompt Engineers".
Generates instructions by showing input-output examples and asking the LLM to infer
what instruction would produce those outputs.

This is used for initial instruction generation in Phase 1 (HbBoPs),
while OPRO is used for iterative refinement in Phase 2.
"""
import random
from typing import List, Dict, Callable

import numpy as np
from sklearn.cluster import KMeans


# APE Forward meta-prompt template (based on Zhou et al. 2023)
APE_FORWARD_TEMPLATE = """I gave a friend an instruction and some inputs. The friend read the instruction and wrote an output for every input.
Here are the input-output pairs:

{examples}

The instruction was to"""


class APEForwardGenerator:
    """
    Generates instructions using APE forward pass.

    Given input-output examples from the training data, asks the LLM to infer
    what instruction would produce those outputs. Generates many candidates
    and uses K-means clustering on embeddings to select diverse final set.

    Example:
        >>> generator = APEForwardGenerator(meta_llm, encode_fn)
        >>> instructions = generator.generate_instructions(train_data, num_final=25)
    """

    def __init__(
        self,
        meta_llm,
        encode_fn: Callable[[str], np.ndarray],
        num_samples: int = 10,
        num_candidates: int = 100,
        temperature: float = 1.0,
        max_tokens: int = 150,
        seed: int = 42,
    ):
        """
        Args:
            meta_llm: LLM client for generation (must have .generate() method)
            encode_fn: Function to encode text to embedding (e.g., BERT)
            num_samples: Number of input-output examples per generation prompt
            num_candidates: Total candidates to generate before clustering
            temperature: Sampling temperature for diversity
            max_tokens: Maximum tokens per generation
            seed: Random seed for reproducibility
        """
        self.meta_llm = meta_llm
        self.encode_fn = encode_fn
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)

    def generate_instructions(
        self,
        train_data: List[Dict],
        num_final: int = 25,
        verbose: bool = True,
    ) -> List[str]:
        """
        Generate diverse instructions using APE forward pass.

        Args:
            train_data: List of {"question": str, "answer": str} dicts
            num_final: Number of final instructions after clustering
            verbose: Print progress

        Returns:
            List of num_final diverse instruction strings
        """
        if verbose:
            print(f"APE Forward: Generating {self.num_candidates} candidates...")

        # Generate candidates
        candidates = []
        seen = set()
        attempts = 0
        max_attempts = self.num_candidates * 3
        consecutive_failures = 0
        max_consecutive_failures = 10

        while len(candidates) < self.num_candidates and attempts < max_attempts:
            attempts += 1

            # Sample random examples
            examples = self._sample_examples(train_data)

            # Build prompt
            prompt = self._build_prompt(examples)

            # Generate
            try:
                response = self.meta_llm.generate(
                    prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                )
                consecutive_failures = 0  # Reset on success
            except KeyboardInterrupt:
                raise  # Never swallow keyboard interrupt
            except Exception as e:
                consecutive_failures += 1
                if verbose:
                    print(f"  Generation error (attempt {attempts}): {e}")
                if consecutive_failures >= max_consecutive_failures:
                    print(f"  [ERROR] {consecutive_failures} consecutive failures, aborting generation")
                    print(f"  Check model configuration and GPU memory.")
                    break
                continue

            # Extract instruction
            instruction = self._extract_instruction(response)

            if not instruction:
                continue

            # Deduplicate
            instruction_normalized = instruction.lower().strip()
            if instruction_normalized in seen:
                continue

            seen.add(instruction_normalized)
            candidates.append(instruction)

            if verbose and len(candidates) % 20 == 0:
                print(f"  Generated {len(candidates)}/{self.num_candidates} candidates")

        if verbose:
            print(f"  Total unique candidates: {len(candidates)}")

        if len(candidates) < num_final:
            if verbose:
                print(f"  Warning: Only {len(candidates)} candidates, returning all")
            return candidates

        # Cluster to get diverse set
        if verbose:
            print(f"  Clustering to select {num_final} diverse instructions...")

        final_instructions = self._cluster_and_select(candidates, num_final)

        if verbose:
            print(f"  Selected {len(final_instructions)} final instructions")

        return final_instructions

    def _sample_examples(self, train_data: List[Dict]) -> List[Dict]:
        """Sample random input-output examples from training data."""
        indices = self.rng.sample(range(len(train_data)), min(self.num_samples, len(train_data)))
        return [train_data[i] for i in indices]

    def _build_prompt(self, examples: List[Dict]) -> str:
        """Build APE forward prompt from examples."""
        example_str = "\n\n".join([
            f"Input: {ex['question']}\nOutput: {self._format_output(ex['answer'])}"
            for ex in examples
        ])
        return APE_FORWARD_TEMPLATE.format(examples=example_str)

    def _format_output(self, answer: str) -> str:
        """
        Format answer for the prompt.

        For GSM8K, extract just the final answer to keep prompt concise.
        """
        answer = answer.strip()

        # If answer has #### marker, extract final answer
        if "####" in answer:
            parts = answer.split("####")
            if len(parts) > 1:
                final = parts[1].strip()
                return final

        # For short answers, return as-is
        if len(answer) < 100:
            return answer

        # For long answers, truncate
        return answer[:100] + "..."

    def _extract_instruction(self, response: str) -> str:
        """
        Extract instruction from LLM response.

        The prompt ends with "The instruction was to" so the response
        should complete this sentence.
        """
        if not response:
            return ""

        response = response.strip()

        # Clean up common prefixes
        prefixes_to_remove = [
            "The instruction was to",
            "The instruction was",
            ":",
        ]
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # Take first sentence/line if response is too long
        for delimiter in ["\n", ". ", ".\n"]:
            if delimiter in response:
                response = response.split(delimiter)[0]
                break

        # Clean up
        response = response.strip().rstrip(".")

        # Skip if too short or too long
        if len(response) < 10 or len(response) > 500:
            return ""

        # Ensure it starts with a verb (instruction-like)
        # If not, try to fix it
        if response and response[0].islower():
            response = response[0].upper() + response[1:]

        return response

    def _cluster_and_select(
        self,
        candidates: List[str],
        num_final: int,
    ) -> List[str]:
        """
        Use K-means clustering on embeddings to select diverse instructions.

        Args:
            candidates: List of candidate instruction strings
            num_final: Number to select

        Returns:
            List of selected diverse instructions
        """
        # Encode all candidates
        embeddings = np.array([self.encode_fn(c) for c in candidates])

        # K-means clustering
        kmeans = KMeans(n_clusters=num_final, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select one instruction per cluster (closest to centroid)
        selected = []
        for cluster_id in range(num_final):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find instruction closest to cluster centroid
            cluster_embeddings = embeddings[cluster_mask]
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]

            selected.append(candidates[closest_idx])

        return selected

    def generate_single(
        self,
        train_data: List[Dict],
    ) -> str:
        """
        Generate a single instruction.

        Args:
            train_data: Training data for examples

        Returns:
            Single instruction string
        """
        examples = self._sample_examples(train_data)
        prompt = self._build_prompt(examples)

        try:
            response = self.meta_llm.generate(
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )
        except KeyboardInterrupt:
            raise  # Never swallow keyboard interrupt
        except Exception as e:
            print(f"[WARNING] Single instruction generation failed: {e}")
            return ""

        return self._extract_instruction(response)
