"""Diversity Instruction Generator - Task-Agnostic.

Generates diverse instructions using:
1. Style taxonomy (12+ categories) with explicit style constraints
2. Paraphrasing module for base instruction variants
3. Diversity validation before caching
4. Near-duplicate filtering (cosine > 0.95)

Replaces the original APE generator which produced homogeneous instructions.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm


# =============================================================================
# STYLE TAXONOMY (12+ categories)
# =============================================================================

STYLE_CATEGORIES = {
    "minimalist": {
        "description": "Ultra-short, terse (1-5 words only)",
        "examples": ["Solve:", "Answer:", "?", "Output:", "Result:", "Calculate."],
        "length_hint": "1-5 words MAXIMUM. Single word or short phrase only.",
        "temperature": 1.0,
    },
    "direct_command": {
        "description": "Short imperative commands (6-15 words)",
        "examples": [
            "Find the answer to this problem.",
            "Calculate the result step by step.",
            "Determine the solution.",
        ],
        "length_hint": "6-15 words",
        "temperature": 0.9,
    },
    "chain_of_thought": {
        "description": "Step-by-step reasoning triggers",
        "examples": [
            "Let's think step by step.",
            "Let's work through this carefully.",
            "Think about this problem systematically.",
            "Break this down step by step.",
        ],
        "length_hint": "5-25 words",
        "temperature": 0.8,
    },
    "socratic": {
        "description": "Guiding questions that prompt analysis",
        "examples": [
            "What information is given? What are we looking for?",
            "What are the key elements here?",
            "How would you approach this?",
        ],
        "length_hint": "10-40 words, use question format",
        "temperature": 0.9,
    },
    "pedagogical": {
        "description": "Patient teacher/tutor persona",
        "examples": [
            "You are a patient tutor. Walk through this problem step by step.",
            "As a helpful teacher, guide the solution process.",
            "Explain your reasoning as you work through this.",
        ],
        "length_hint": "15-50 words",
        "temperature": 0.9,
    },
    "strict_expert": {
        "description": "Rigorous, formal expert style",
        "examples": [
            "Apply rigorous analysis to determine the solution.",
            "Execute formal procedures systematically.",
            "Provide a precise, well-reasoned answer.",
        ],
        "length_hint": "15-40 words, formal tone",
        "temperature": 0.8,
    },
    "child_friendly": {
        "description": "Simple language, friendly and encouraging tone",
        "examples": [
            "Can you help me figure this out? Use simple words!",
            "Pretend you're explaining to a friend who's just learning.",
            "Let's solve this together, nice and easy!",
        ],
        "length_hint": "10-40 words, simple vocabulary",
        "temperature": 1.0,
    },
    "algorithmic": {
        "description": "Programming-style, structured format",
        "examples": [
            "def solve(x): return process(parse(x))",
            "INPUT -> PARSE -> COMPUTE -> OUTPUT",
            "[1] Read [2] Analyze [3] Calculate [4] Verify",
            "PROCEDURE: Extract data, Apply logic, Return result",
        ],
        "length_hint": "10-50 words, code-like or structured format",
        "temperature": 1.1,
    },
    "motivational": {
        "description": "Supportive, encouraging tone",
        "examples": [
            "You've got this! Work through it and find the answer.",
            "Take your time, you can do this. What's the solution?",
            "Great job tackling this! Now find the answer.",
        ],
        "length_hint": "10-30 words, positive tone",
        "temperature": 0.9,
    },
    "analytical": {
        "description": "Focus on logical structure and analysis",
        "examples": [
            "Analyze the logical structure of this problem.",
            "Use deductive reasoning to solve this.",
            "Identify the key variables and their relationships.",
        ],
        "length_hint": "10-40 words, logic-focused",
        "temperature": 0.8,
    },
    "creative_narrative": {
        "description": "Metaphorical, story-like framing",
        "examples": [
            "Imagine you're a detective investigating this puzzle...",
            "This is a mystery waiting to be solved. What's the answer?",
            "Think of the numbers as characters in a story.",
        ],
        "length_hint": "15-50 words, imaginative",
        "temperature": 1.2,
    },
    "experimental": {
        "description": "Unconventional, creative formats",
        "examples": [
            "<SYSTEM> Execute: solve_task() -> result </SYSTEM>",
            "BEEP BOOP. PROCESS INPUT. OUTPUT ANSWER.",
            "MODE: SOLUTION | ACTION: COMPUTE | OUTPUT: VALUE",
            ">>> TASK.solve() <<<",
        ],
        "length_hint": "5-40 words, unconventional format",
        "temperature": 1.2,
    },
}


# =============================================================================
# PARAPHRASE STRATEGIES
# =============================================================================

PARAPHRASE_STRATEGIES = [
    "Rephrase using completely different words while keeping the same meaning",
    "Change the sentence structure (e.g., active to passive, or statement to command)",
    "Make it significantly shorter while preserving the core idea",
    "Make it longer by adding helpful detail or context",
    "Use more formal, academic language",
    "Use more casual, conversational language",
    "Convert to a question format",
    "Start with a completely different word or phrase",
    "Use a metaphor or analogy to express the same idea",
    "Add emphasis on a specific aspect or step",
]


# =============================================================================
# DIVERSITY METRICS
# =============================================================================

@dataclass
class DiversityMetrics:
    """Metrics for evaluating instruction diversity."""
    mean_cosine: float
    std_cosine: float
    min_cosine: float
    max_cosine: float
    percentile_25: float
    percentile_75: float
    num_near_duplicates: int  # pairs with cosine > 0.95
    total_pairs: int
    length_mean: float
    length_std: float
    length_min: int
    length_max: int
    length_histogram: Dict[str, int]
    passed_validation: bool
    validation_message: str

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class DiversityInstructionGenerator:
    """Task-agnostic diverse instruction generator.

    Generates instructions by:
    1. Inferring the task from Q/A examples (task-agnostic)
    2. Using 12+ style categories with explicit constraints
    3. Paraphrasing base instructions for additional diversity
    4. Filtering near-duplicates and validating diversity

    Replaces the original APE generator which produced homogeneous output.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "vllm",
        base_instructions_path: Optional[str] = None,
    ):
        """Initialize generator.

        Args:
            model: Model name for generation
            backend: LLM backend (vllm, openai, etc.)
            base_instructions_path: Path to base instructions file for paraphrasing
        """
        self.model = model
        self.backend = backend
        self._client = None
        self._gtr = None

        # Load base instructions if provided
        self.base_instructions = []
        if base_instructions_path and Path(base_instructions_path).exists():
            self.base_instructions = self._load_base_instructions(base_instructions_path)

    def _get_client(self):
        """Lazy load LLM client."""
        if self._client is None:
            from src.llm_client import create_llm_client
            print(f"Initializing LLM client: {self.model}")
            self._client = create_llm_client(self.model, self.backend)
        return self._client

    def _get_gtr(self):
        """Lazy load GTR encoder for diversity validation."""
        if self._gtr is None:
            from sentence_transformers import SentenceTransformer
            print("Loading GTR encoder for diversity validation...")
            self._gtr = SentenceTransformer("sentence-transformers/gtr-t5-base")
        return self._gtr

    def _load_base_instructions(self, path: str) -> List[str]:
        """Load base instructions from file.

        Supports formats:
        - Plain text: one instruction per line
        - Numbered: "N. instruction text"
        """
        instructions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Remove numbering if present (e.g., "1. instruction")
                if line[0].isdigit() and ". " in line:
                    line = line.split(". ", 1)[1]

                instructions.append(line)

        print(f"Loaded {len(instructions)} base instructions from {path}")
        return instructions

    # =========================================================================
    # STYLE-CONSTRAINED GENERATION
    # =========================================================================

    def _build_style_prompt(
        self,
        style_name: str,
        style_config: dict,
        examples: List[dict],
    ) -> str:
        """Build task-agnostic prompt with style constraints.

        The LLM infers the task from Q/A examples, not from hardcoded text.
        """
        # Show Q/A examples - let LLM infer the task
        qa_text = "\n\n".join([
            f"Input: {ex['question']}\nOutput: {ex['answer']}"
            for ex in examples[:3]
        ])

        # Format style examples if available
        style_examples = style_config.get("examples", [])
        style_examples_text = ""
        if style_examples:
            examples_list = "\n".join([f"  - {ex}" for ex in style_examples[:3]])
            style_examples_text = f"\nEXAMPLE INSTRUCTIONS IN THIS STYLE:\n{examples_list}"

        return f"""Below are examples of a task. Study them to understand what the task involves.

TASK EXAMPLES:
{qa_text}

---

Generate an instruction that would help someone complete similar tasks.

STYLE REQUIREMENTS:
- Style: {style_config["description"]}
- Length: {style_config.get("length_hint", "flexible")}
{style_examples_text}

IMPORTANT:
- Do NOT mention specific details from the examples above
- Create a GENERAL instruction applicable to this type of task
- Match the requested style EXACTLY
- Output ONLY the instruction, nothing else

Your instruction:"""

    def _generate_with_style(
        self,
        style_name: str,
        style_config: dict,
        validation_data: List[dict],
        num_instructions: int,
        batch_size: int = 20,
    ) -> List[str]:
        """Generate instructions with specific style constraints."""
        client = self._get_client()
        instructions = set()
        temperature = style_config.get("temperature", 1.0)

        attempts = 0
        max_attempts = (num_instructions * 3) // batch_size + 2

        while len(instructions) < num_instructions and attempts < max_attempts:
            attempts += 1

            # Build batch of prompts with different examples each time
            prompts = []
            for _ in range(batch_size):
                examples = random.sample(
                    validation_data,
                    min(3, len(validation_data))
                )
                prompt = self._build_style_prompt(style_name, style_config, examples)
                prompts.append(prompt)

            # Generate batch
            responses = client.generate_batch(
                prompts,
                max_tokens=100,
                temperature=temperature,
            )

            # Parse and filter responses
            for response in responses:
                instruction = self._parse_instruction(response)
                if instruction:
                    # Style-specific length validation
                    if style_name == "minimalist" and len(instruction.split()) > 8:
                        continue  # Too long for minimalist
                    instructions.add(instruction)

        return list(instructions)[:num_instructions]

    def _parse_instruction(self, response: str) -> Optional[str]:
        """Parse and clean instruction from LLM response."""
        if not response:
            return None

        instruction = response.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Instruction:",
            "Here is an instruction:",
            "The instruction is:",
            "Answer:",
            "Output:",
            "Your instruction:",
        ]
        for prefix in prefixes_to_remove:
            if instruction.lower().startswith(prefix.lower()):
                instruction = instruction[len(prefix):].strip()

        # Remove quotes if wrapped
        if len(instruction) > 2:
            if (instruction.startswith('"') and instruction.endswith('"')) or \
               (instruction.startswith("'") and instruction.endswith("'")):
                instruction = instruction[1:-1]

        # Take only first line/paragraph (instruction should be concise)
        instruction = instruction.split("\n")[0].strip()

        # Filter by length
        if len(instruction) < 1 or len(instruction) > 500:
            return None

        return instruction

    # =========================================================================
    # PARAPHRASING MODULE
    # =========================================================================

    def _build_paraphrase_prompt(self, instruction: str, strategy: str) -> str:
        """Build prompt for paraphrasing an instruction."""
        return f"""Paraphrase this instruction using the strategy below.

ORIGINAL INSTRUCTION:
{instruction}

PARAPHRASE STRATEGY:
{strategy}

IMPORTANT:
- Output ONLY the paraphrased instruction, nothing else
- Keep it as a valid instruction (not a question about the instruction)
- Apply the strategy while preserving the core meaning

PARAPHRASED INSTRUCTION:"""

    def _paraphrase_base_instructions(
        self,
        num_paraphrases_per_instruction: int = 5,
        batch_size: int = 20,
        verbose: bool = True,
    ) -> List[str]:
        """Generate paraphrases of base instructions."""
        if not self.base_instructions:
            return []

        client = self._get_client()
        paraphrases = []

        if verbose:
            print(f"Paraphrasing {len(self.base_instructions)} base instructions "
                  f"({num_paraphrases_per_instruction} variants each)...")

        # Prepare all paraphrase tasks
        tasks = []
        for instruction in self.base_instructions:
            strategies = random.sample(
                PARAPHRASE_STRATEGIES,
                min(num_paraphrases_per_instruction, len(PARAPHRASE_STRATEGIES))
            )
            for strategy in strategies:
                tasks.append((instruction, strategy))

        # Process in batches
        pbar = tqdm(range(0, len(tasks), batch_size), desc="Paraphrasing") if verbose else None

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            prompts = [
                self._build_paraphrase_prompt(inst, strat)
                for inst, strat in batch
            ]

            responses = client.generate_batch(
                prompts,
                max_tokens=150,
                temperature=1.0,
            )

            for response in responses:
                parsed = self._parse_instruction(response)
                if parsed:
                    paraphrases.append(parsed)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        if verbose:
            print(f"Generated {len(paraphrases)} paraphrases")

        return paraphrases

    # =========================================================================
    # DIVERSITY VALIDATION
    # =========================================================================

    def validate_diversity(
        self,
        instructions: List[str],
        verbose: bool = True,
    ) -> DiversityMetrics:
        """Compute diversity metrics using GTR embeddings."""
        gtr = self._get_gtr()

        if verbose:
            print("Computing GTR embeddings for diversity validation...")

        # Encode all instructions
        embeddings = gtr.encode(
            instructions,
            show_progress_bar=verbose,
            normalize_embeddings=True,
        )

        # Compute pairwise cosine similarity (upper triangular only)
        n = len(instructions)
        if verbose:
            print(f"Computing pairwise similarities for {n} instructions...")

        # For normalized vectors, cosine = dot product
        sim_matrix = embeddings @ embeddings.T

        # Get upper triangular values (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        similarities = sim_matrix[triu_indices]
        total_pairs = len(similarities)

        # Length analysis (word count)
        lengths = [len(inst.split()) for inst in instructions]

        # Bucket lengths
        length_histogram = {
            "1-5": sum(1 for l in lengths if 1 <= l <= 5),
            "6-15": sum(1 for l in lengths if 6 <= l <= 15),
            "16-30": sum(1 for l in lengths if 16 <= l <= 30),
            "31-50": sum(1 for l in lengths if 31 <= l <= 50),
            "51+": sum(1 for l in lengths if l > 50),
        }

        # Count near-duplicates
        near_duplicate_threshold = 0.95
        num_near_duplicates = int((similarities > near_duplicate_threshold).sum())

        # Validation criteria
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        length_std = float(np.std(lengths))

        validation_checks = []

        # Check 1: Mean similarity in acceptable range
        if mean_sim > 0.75:
            validation_checks.append(f"Mean similarity too high: {mean_sim:.3f} > 0.75")
        elif mean_sim < 0.25:
            validation_checks.append(f"Mean similarity too low: {mean_sim:.3f} < 0.25")

        # Check 2: Few near-duplicates
        near_dup_pct = num_near_duplicates / total_pairs * 100 if total_pairs > 0 else 0
        if near_dup_pct > 1.0:
            validation_checks.append(f"Too many near-duplicates: {near_dup_pct:.2f}% > 1%")

        # Check 3: Length variation
        if length_std < 8:
            validation_checks.append(f"Length variation too low: std={length_std:.1f} < 8")

        passed = len(validation_checks) == 0
        message = "PASSED" if passed else "; ".join(validation_checks)

        metrics = DiversityMetrics(
            mean_cosine=mean_sim,
            std_cosine=std_sim,
            min_cosine=float(np.min(similarities)),
            max_cosine=float(np.max(similarities)),
            percentile_25=float(np.percentile(similarities, 25)),
            percentile_75=float(np.percentile(similarities, 75)),
            num_near_duplicates=num_near_duplicates,
            total_pairs=total_pairs,
            length_mean=float(np.mean(lengths)),
            length_std=length_std,
            length_min=int(np.min(lengths)),
            length_max=int(np.max(lengths)),
            length_histogram=length_histogram,
            passed_validation=passed,
            validation_message=message,
        )

        if verbose:
            print(f"\n--- Diversity Metrics ---")
            print(f"Cosine Similarity:")
            print(f"  Mean: {metrics.mean_cosine:.4f} | Std: {metrics.std_cosine:.4f}")
            print(f"  Range: [{metrics.min_cosine:.4f}, {metrics.max_cosine:.4f}]")
            print(f"  P25/P75: [{metrics.percentile_25:.4f}, {metrics.percentile_75:.4f}]")
            print(f"Near-duplicates (>{near_duplicate_threshold}): "
                  f"{metrics.num_near_duplicates}/{metrics.total_pairs} "
                  f"({near_dup_pct:.3f}%)")
            print(f"Length (words):")
            print(f"  Mean: {metrics.length_mean:.1f} | Std: {metrics.length_std:.1f}")
            print(f"  Range: [{metrics.length_min}, {metrics.length_max}]")
            print(f"  Histogram: {metrics.length_histogram}")
            print(f"Validation: {metrics.validation_message}")
            print(f"-------------------------\n")

        return metrics

    def filter_near_duplicates(
        self,
        instructions: List[str],
        threshold: float = 0.95,
        verbose: bool = True,
    ) -> List[str]:
        """Remove near-duplicate instructions using greedy selection."""
        if len(instructions) <= 1:
            return instructions

        gtr = self._get_gtr()

        if verbose:
            print(f"Filtering near-duplicates (threshold={threshold})...")

        # Encode all
        embeddings = gtr.encode(
            instructions,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Greedy selection: keep instruction if not too similar to any kept
        kept_indices = [0]
        kept_embeddings = [embeddings[0]]

        for i in range(1, len(instructions)):
            emb = embeddings[i]

            # Check similarity with all kept embeddings
            sims = [float(np.dot(emb, kept)) for kept in kept_embeddings]
            max_sim = max(sims) if sims else 0

            if max_sim < threshold:
                kept_indices.append(i)
                kept_embeddings.append(emb)

        result = [instructions[i] for i in kept_indices]

        if verbose:
            removed = len(instructions) - len(result)
            print(f"Removed {removed} near-duplicates, kept {len(result)}")

        return result

    # =========================================================================
    # MAIN GENERATION METHODS
    # =========================================================================

    def generate_diverse_instructions(
        self,
        validation_data: List[dict],
        num_instructions: int = 1000,
        batch_size: int = 20,
        verbose: bool = True,
    ) -> Tuple[List[str], DiversityMetrics]:
        """Generate diverse instructions via style taxonomy + paraphrasing.

        Args:
            validation_data: Q/A examples for task inference
            num_instructions: Target number of unique instructions
            batch_size: Batch size for LLM generation
            verbose: Print progress

        Returns:
            Tuple of (instructions list, diversity metrics)
        """
        all_instructions = []

        # PHASE 1: Style-constrained generation
        num_styles = len(STYLE_CATEGORIES)
        num_per_style = max(10, (num_instructions // 2) // num_styles)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"PHASE 1: Style-constrained generation")
            print(f"{'=' * 60}")
            print(f"Generating ~{num_per_style} instructions per style ({num_styles} styles)")

        for style_name, style_config in STYLE_CATEGORIES.items():
            if verbose:
                print(f"\n[{style_name}] Generating {num_per_style} instructions...")

            style_instructions = self._generate_with_style(
                style_name=style_name,
                style_config=style_config,
                validation_data=validation_data,
                num_instructions=num_per_style,
                batch_size=batch_size,
            )
            all_instructions.extend(style_instructions)

            if verbose:
                print(f"  Generated: {len(style_instructions)}")
                if style_instructions:
                    print(f"  Sample: {style_instructions[0][:80]}...")

        # PHASE 2: Paraphrase base instructions
        if self.base_instructions:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"PHASE 2: Paraphrasing base instructions")
                print(f"{'=' * 60}")

            num_paraphrases = max(3, (num_instructions // 2) // len(self.base_instructions))
            paraphrased = self._paraphrase_base_instructions(
                num_paraphrases_per_instruction=num_paraphrases,
                batch_size=batch_size,
                verbose=verbose,
            )
            all_instructions.extend(paraphrased)

        # PHASE 3: Add original base instructions
        all_instructions.extend(self.base_instructions)

        if verbose:
            print(f"\nTotal before deduplication: {len(all_instructions)}")

        # PHASE 4: Deduplicate exact matches
        unique_instructions = list(set(all_instructions))
        if verbose:
            print(f"After exact deduplication: {len(unique_instructions)}")

        # PHASE 5: Filter near-duplicates
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"PHASE 3: Near-duplicate filtering")
            print(f"{'=' * 60}")

        filtered = self.filter_near_duplicates(
            unique_instructions,
            threshold=0.95,
            verbose=verbose,
        )

        # PHASE 6: Validate diversity
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"PHASE 4: Diversity validation")
            print(f"{'=' * 60}")

        metrics = self.validate_diversity(filtered, verbose=verbose)

        # Trim to requested size
        result = filtered[:num_instructions]

        if len(result) < num_instructions:
            print(f"WARNING: Only generated {len(result)} unique instructions "
                  f"(target: {num_instructions})")

        return result, metrics

    # =========================================================================
    # CACHING
    # =========================================================================

    def save_instructions(
        self,
        instructions: List[str],
        metrics: DiversityMetrics,
        path: str,
    ) -> None:
        """Save instructions and metrics to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "instructions": instructions,
            "metrics": metrics.to_dict(),
            "generator": {
                "model": self.model,
                "backend": self.backend,
                "num_base_instructions": len(self.base_instructions),
                "num_styles": len(STYLE_CATEGORIES),
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(instructions)} instructions to {path}")

    @staticmethod
    def load_instructions(path: str) -> Tuple[List[str], Optional[DiversityMetrics]]:
        """Load instructions and metrics from JSON file.

        Returns:
            Tuple of (instructions list, metrics or None)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(data, list):
            print(f"Loaded {len(data)} instructions from {path} (legacy format)")
            return data, None

        instructions = data.get("instructions", data)
        metrics_dict = data.get("metrics")

        metrics = None
        if metrics_dict:
            metrics = DiversityMetrics(**metrics_dict)

        print(f"Loaded {len(instructions)} instructions from {path}")
        if metrics:
            print(f"  Diversity: mean_cosine={metrics.mean_cosine:.3f}, "
                  f"passed={metrics.passed_validation}")

        return instructions, metrics

    def generate_or_load(
        self,
        cache_path: str,
        validation_data: List[dict],
        num_instructions: int = 1000,
        batch_size: int = 20,
        force_regenerate: bool = False,
        verbose: bool = True,
    ) -> Tuple[List[str], DiversityMetrics]:
        """Generate instructions or load from cache if exists.

        Args:
            cache_path: Path to cache file
            validation_data: Q/A examples for task inference
            num_instructions: Target number of unique instructions
            batch_size: Batch size for LLM generation
            force_regenerate: If True, regenerate even if cache exists
            verbose: Print progress

        Returns:
            Tuple of (instructions list, diversity metrics)
        """
        cache_file = Path(cache_path)

        if cache_file.exists() and not force_regenerate:
            instructions, metrics = self.load_instructions(cache_path)

            # If no metrics (legacy format), compute them
            if metrics is None:
                print("Computing diversity metrics for cached instructions...")
                metrics = self.validate_diversity(instructions, verbose=verbose)

            return instructions, metrics

        # Generate new instructions
        instructions, metrics = self.generate_diverse_instructions(
            validation_data=validation_data,
            num_instructions=num_instructions,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Save to cache
        self.save_instructions(instructions, metrics, cache_path)

        return instructions, metrics


# =============================================================================
# BACKWARDS COMPATIBILITY ALIAS
# =============================================================================

# Keep old class name as alias for backwards compatibility
APEInstructionGenerator = DiversityInstructionGenerator
