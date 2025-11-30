"""
HYPE - Hyperband Prompt Evolution

Main orchestrator that combines:
1. HbBoPs for multi-fidelity evaluation (Hyperband)
2. Component scoring (S_I, S_E)
3. Generation methods (Semantic Gradient + Bootstrap)
4. Evolutionary loop for iterative improvement
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs, Prompt
from hype.data_types import (
    Instruction, Exemplar, EvaluationRecord,
    ComponentScore, ComponentSource
)
from hype.scoring import ComponentScorer
from hype.generators.semantic_gradient import SemanticGradientGenerator
from hype.generators.bootstrap import BootstrapGenerator
from hype.generators.recombination import RecombinationGenerator


@dataclass
class HYPEConfig:
    """Configuration for HYPE evolution"""
    # Evolution settings
    num_generations: int = 5

    # Generation settings (Method A + C every iteration)
    num_new_instructions: int = 8  # Method A: semantic gradient
    num_new_exemplars: int = 8     # Method C: bootstrap
    use_recombination: bool = True # Method B: optional, zero cost

    # HbBoPs settings (passed through)
    bmin: int = 10
    eta: float = 2.0
    encoder_name: str = "bert-base-uncased"

    # Pool management
    max_instructions: int = 100
    max_exemplars: int = 100
    retire_threshold: float = 0.1  # Bottom 10% retired each generation

    # Output
    output_dir: str = "results"
    save_checkpoints: bool = True


@dataclass
class GenerationHistory:
    """Track evolution progress"""
    generation: int
    num_instructions: int
    num_exemplars: int
    best_error: float
    best_instruction_id: int
    best_exemplar_id: int
    new_instructions_added: int = 0
    new_exemplars_added: int = 0
    instructions_retired: int = 0
    exemplars_retired: int = 0
    hyperband_evaluations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HYPE:
    """
    Hyperband Prompt Evolution

    Wraps HbBoPs with an evolutionary outer loop that:
    1. Runs Hyperband to evaluate prompts
    2. Computes component scores (S_I, S_E)
    3. Generates new components (Method A: instructions, Method C: exemplars)
    4. Restocks pools and iterates
    """

    def __init__(
        self,
        instructions: List[str],
        exemplars: List[str],
        validation_data: List[Dict],
        llm_client,
        meta_llm_client=None,
        training_data: List[Dict] = None,  # For bootstrap generator
        config: HYPEConfig = None,
        device: str = "auto",
        verbose: bool = True,
    ):
        """
        Args:
            instructions: Initial instruction pool
            exemplars: Initial exemplar pool
            validation_data: Validation examples for evaluation
            llm_client: LLM client for task evaluation
            meta_llm_client: LLM client for generation (optional, defaults to llm_client)
            training_data: Training data for bootstrap generator
            config: HYPE configuration
            device: Device for computation
            verbose: Print progress
        """
        self.config = config or HYPEConfig()
        self.validation_data = validation_data
        self.training_data = training_data or validation_data  # Fallback to validation
        self.llm_client = llm_client
        self.meta_llm = meta_llm_client or llm_client
        self.device = device
        self.verbose = verbose

        # Initialize instruction pool with IDs
        self.instructions: List[Instruction] = [
            Instruction(id=i, text=text, source=ComponentSource.INITIAL, generation=0)
            for i, text in enumerate(instructions)
        ]

        # Initialize exemplar pool with IDs
        self.exemplars: List[Exemplar] = [
            Exemplar(id=i, text=text, source=ComponentSource.INITIAL, generation=0)
            for i, text in enumerate(exemplars)
        ]

        # Component scorer
        self.scorer = ComponentScorer()

        # Initialize generators
        self.semantic_gradient_gen = SemanticGradientGenerator(
            llm_client=self.meta_llm,
            num_to_improve=self.config.num_new_instructions
        )

        self.bootstrap_gen = BootstrapGenerator(
            llm_client=self.meta_llm,
            training_data=self.training_data,
            num_exemplars_to_generate=self.config.num_new_exemplars
        )

        self.recombination_gen = RecombinationGenerator(
            max_new_instructions=self.config.num_new_instructions
        )

        # Track evolution
        self.history: List[GenerationHistory] = []
        self.best_prompt: Optional[Prompt] = None
        self.best_error: float = float('inf')
        self.best_generation: int = 0

        # Track error examples for gradient generation
        self.error_examples: Dict[int, List[Dict]] = {}

        # Track hard examples for bootstrap
        self.hard_example_indices: List[int] = []

        if self.verbose:
            print(f"HYPE initialized:")
            print(f"  Instructions: {len(self.instructions)}")
            print(f"  Exemplars: {len(self.exemplars)}")
            print(f"  Validation samples: {len(validation_data)}")
            print(f"  Generations: {self.config.num_generations}")

    def _create_hbbops(self, evaluator) -> HbBoPs:
        """Create a fresh HbBoPs instance for this generation"""
        # Convert to plain text lists
        instruction_texts = [i.text for i in self.instructions]
        exemplar_texts = [e.text for e in self.exemplars]

        return HbBoPs(
            instructions=instruction_texts,
            exemplars=exemplar_texts,
            validation_data=self.validation_data,
            llm_evaluator=evaluator,
            encoder_name=self.config.encoder_name,
            bmin=self.config.bmin,
            eta=self.config.eta,
            device=self.device
        )

    def _extract_evaluation_records(self, hbbops: HbBoPs, generation: int) -> List[EvaluationRecord]:
        """Extract evaluation records from HbBoPs design_data"""
        records = []

        for prompt_idx, inst_emb, ex_emb, val_err, fidelity in hbbops.design_data:
            prompt = hbbops.prompts[prompt_idx]

            # Map back to our instruction/exemplar IDs
            # HbBoPs uses indices into its lists, we need to map to our IDs
            inst_id = self.instructions[prompt.instruction_id].id
            ex_id = self.exemplars[prompt.exemplar_id].id

            record = EvaluationRecord(
                instruction_id=inst_id,
                exemplar_id=ex_id,
                budget=fidelity,
                error_rate=val_err,
                generation=generation
            )
            records.append(record)

        return records

    def _update_error_examples(self, hbbops: HbBoPs, generation: int) -> None:
        """
        Collect error examples for semantic gradient generation.

        Note: This requires evaluator to track individual failures.
        For now, we just track which instructions had high error rates.
        """
        # Group by instruction
        inst_errors: Dict[int, List[float]] = {}

        for prompt_idx, _, _, val_err, fidelity in hbbops.design_data:
            prompt = hbbops.prompts[prompt_idx]
            inst_id = self.instructions[prompt.instruction_id].id

            if inst_id not in inst_errors:
                inst_errors[inst_id] = []
            inst_errors[inst_id].append(val_err)

        # For now, just store average error rates
        # In a full implementation, we'd track actual failure examples
        self.error_examples = {
            inst_id: [{"error_rate": sum(errors) / len(errors)}]
            for inst_id, errors in inst_errors.items()
        }

    def _update_hard_examples(self) -> None:
        """
        Identify hard examples (frequently failed) for bootstrap.

        This would require tracking per-example failure rates across evaluations.
        For now, we use a simple heuristic.
        """
        # Placeholder: use random subset
        # In full implementation, track which validation indices are frequently wrong
        import random
        n_hard = min(20, len(self.training_data))
        self.hard_example_indices = random.sample(range(len(self.training_data)), n_hard)

    def _generate_new_components(self, generation: int) -> Tuple[List[Instruction], List[Exemplar]]:
        """
        Generate new components using Method A (instructions) and Method C (exemplars).
        """
        new_instructions = []
        new_exemplars = []

        # Get current scores
        inst_scores = self.scorer.get_all_instruction_scores()
        ex_scores = self.scorer.get_all_exemplar_scores()

        # Method A: Semantic Gradient for new instructions
        if self.verbose:
            print("  Generating new instructions (Method A: Semantic Gradient)...")

        try:
            result_a = self.semantic_gradient_gen.generate(
                instructions=self.instructions,
                exemplars=self.exemplars,
                instruction_scores=inst_scores,
                exemplar_scores=ex_scores,
                evaluation_records=self.scorer.records,
                generation=generation,
                error_examples=self.error_examples
            )
            new_instructions.extend(result_a.new_instructions)
            if self.verbose:
                print(f"    Generated {len(result_a.new_instructions)} new instructions")
        except Exception as e:
            print(f"  Warning: Semantic gradient generation failed: {e}")

        # Method C: Bootstrap for new exemplars
        if self.verbose:
            print("  Generating new exemplars (Method C: Bootstrap)...")

        try:
            result_c = self.bootstrap_gen.generate(
                instructions=self.instructions,
                exemplars=self.exemplars,
                instruction_scores=inst_scores,
                exemplar_scores=ex_scores,
                evaluation_records=self.scorer.records,
                generation=generation,
                hard_example_indices=self.hard_example_indices
            )
            new_exemplars.extend(result_c.new_exemplars)
            if self.verbose:
                print(f"    Generated {len(result_c.new_exemplars)} new exemplars")
        except Exception as e:
            print(f"  Warning: Bootstrap generation failed: {e}")

        # Method B: Recombination (optional, zero cost)
        if self.config.use_recombination and len(new_instructions) < self.config.num_new_instructions:
            if self.verbose:
                print("  Generating additional instructions (Method B: Recombination)...")

            try:
                result_b = self.recombination_gen.generate(
                    instructions=self.instructions,
                    exemplars=self.exemplars,
                    instruction_scores=inst_scores,
                    exemplar_scores=ex_scores,
                    evaluation_records=self.scorer.records,
                    generation=generation
                )
                # Only add if we need more
                needed = self.config.num_new_instructions - len(new_instructions)
                new_instructions.extend(result_b.new_instructions[:needed])
                if self.verbose:
                    print(f"    Generated {min(len(result_b.new_instructions), needed)} additional instructions")
            except Exception as e:
                print(f"  Warning: Recombination generation failed: {e}")

        return new_instructions, new_exemplars

    def _add_new_components(
        self,
        new_instructions: List[Instruction],
        new_exemplars: List[Exemplar]
    ) -> Tuple[int, int]:
        """Add new components to pools, respecting max limits"""
        added_inst = 0
        added_ex = 0

        # Assign new IDs
        next_inst_id = max(i.id for i in self.instructions) + 1 if self.instructions else 0
        next_ex_id = max(e.id for e in self.exemplars) + 1 if self.exemplars else 0

        # Add instructions
        for inst in new_instructions:
            if len(self.instructions) >= self.config.max_instructions:
                break
            # Check for duplicates
            if any(i.text == inst.text for i in self.instructions):
                continue
            inst.id = next_inst_id
            self.instructions.append(inst)
            next_inst_id += 1
            added_inst += 1

        # Add exemplars
        for ex in new_exemplars:
            if len(self.exemplars) >= self.config.max_exemplars:
                break
            # Check for duplicates
            if any(e.text == ex.text for e in self.exemplars):
                continue
            ex.id = next_ex_id
            self.exemplars.append(ex)
            next_ex_id += 1
            added_ex += 1

        return added_inst, added_ex

    def _retire_poor_performers(self) -> Tuple[int, int]:
        """Remove bottom performers from pools"""
        retired_inst = 0
        retired_ex = 0

        if len(self.instructions) <= 5 or len(self.exemplars) <= 5:
            return 0, 0  # Keep minimum pool size

        inst_scores = self.scorer.get_all_instruction_scores()
        ex_scores = self.scorer.get_all_exemplar_scores()

        # Calculate retirement threshold
        num_to_retire = int(len(self.instructions) * self.config.retire_threshold)

        if num_to_retire > 0 and inst_scores:
            # Sort by score, get bottom N
            sorted_inst = sorted(
                [(i, inst_scores.get(i.id, ComponentScore(i.id, 0)).score)
                 for i in self.instructions],
                key=lambda x: x[1]
            )
            to_retire = [i for i, s in sorted_inst[:num_to_retire]]

            # Don't retire initial instructions in first few generations
            to_retire = [i for i in to_retire if i.generation > 0]

            for inst in to_retire:
                self.instructions.remove(inst)
                retired_inst += 1

        num_to_retire = int(len(self.exemplars) * self.config.retire_threshold)

        if num_to_retire > 0 and ex_scores:
            sorted_ex = sorted(
                [(e, ex_scores.get(e.id, ComponentScore(e.id, 0)).score)
                 for e in self.exemplars],
                key=lambda x: x[1]
            )
            to_retire = [e for e, s in sorted_ex[:num_to_retire]]
            to_retire = [e for e in to_retire if e.generation > 0]

            for ex in to_retire:
                self.exemplars.remove(ex)
                retired_ex += 1

        return retired_inst, retired_ex

    def evolve(self, evaluator) -> Tuple[Prompt, float]:
        """
        Run the evolutionary optimization.

        Args:
            evaluator: Function that evaluates prompts on validation data

        Returns:
            (best_prompt, best_error)
        """
        print(f"\n{'='*60}")
        print("HYPE - Hyperband Prompt Evolution")
        print(f"{'='*60}\n")

        for gen in range(1, self.config.num_generations + 1):
            print(f"\n{'='*40}")
            print(f"Generation {gen}/{self.config.num_generations}")
            print(f"{'='*40}")
            print(f"Pool size: {len(self.instructions)} instructions, {len(self.exemplars)} exemplars")

            # Phase 1: Run Hyperband
            print("\nPhase 1: Running Hyperband...")
            hbbops = self._create_hbbops(evaluator)
            gen_best_prompt, gen_best_error = hbbops.run_hyperband(verbose=self.verbose)

            # Update global best
            if gen_best_error < self.best_error:
                self.best_error = gen_best_error
                self.best_prompt = gen_best_prompt
                self.best_generation = gen
                print(f"  New best! Error: {gen_best_error:.4f}")

            # Phase 2: Knowledge Extraction
            print("\nPhase 2: Extracting knowledge...")
            records = self._extract_evaluation_records(hbbops, gen)
            self.scorer.add_records(records)
            self._update_error_examples(hbbops, gen)
            self._update_hard_examples()

            summary = self.scorer.summary()
            print(f"  Total evaluations: {summary['num_records']}")
            print(f"  Instruction scores: mean={summary['instruction_scores']['mean']:.3f}")
            print(f"  Exemplar scores: mean={summary['exemplar_scores']['mean']:.3f}")

            # Phase 3: Generation (skip on last iteration)
            new_inst_added = 0
            new_ex_added = 0
            retired_inst = 0
            retired_ex = 0

            if gen < self.config.num_generations:
                print("\nPhase 3: Generating new components...")
                new_instructions, new_exemplars = self._generate_new_components(gen)
                new_inst_added, new_ex_added = self._add_new_components(new_instructions, new_exemplars)

                # Phase 4: Retire poor performers
                print("\nPhase 4: Retiring poor performers...")
                retired_inst, retired_ex = self._retire_poor_performers()
                print(f"  Retired: {retired_inst} instructions, {retired_ex} exemplars")

            # Record history
            history_entry = GenerationHistory(
                generation=gen,
                num_instructions=len(self.instructions),
                num_exemplars=len(self.exemplars),
                best_error=self.best_error,
                best_instruction_id=self.best_prompt.instruction_id if self.best_prompt else -1,
                best_exemplar_id=self.best_prompt.exemplar_id if self.best_prompt else -1,
                new_instructions_added=new_inst_added,
                new_exemplars_added=new_ex_added,
                instructions_retired=retired_inst,
                exemplars_retired=retired_ex,
                hyperband_evaluations=len(hbbops.design_data)
            )
            self.history.append(history_entry)

            print(f"\nGeneration {gen} complete:")
            print(f"  Best error so far: {self.best_error:.4f} (gen {self.best_generation})")
            print(f"  Pool: {len(self.instructions)} inst (+{new_inst_added}/-{retired_inst}), "
                  f"{len(self.exemplars)} ex (+{new_ex_added}/-{retired_ex})")

            # Save checkpoint after each generation to prevent data loss
            checkpoint_path = self.save_checkpoint(
                Path(self.config.output_dir),
                model_name=getattr(self, '_model_name', 'unknown')
            )
            print(f"  Checkpoint saved: {checkpoint_path}")

        print(f"\n{'='*60}")
        print("HYPE Evolution Complete")
        print(f"{'='*60}")
        print(f"Best error: {self.best_error:.4f}")
        print(f"Found in generation: {self.best_generation}")
        if self.best_prompt:
            print(f"Instruction ID: {self.best_prompt.instruction_id}")
            print(f"Exemplar ID: {self.best_prompt.exemplar_id}")

        return self.best_prompt, self.best_error

    def get_results(self) -> Dict[str, Any]:
        """Get full results as dictionary"""
        inst_scores = self.scorer.get_all_instruction_scores()
        ex_scores = self.scorer.get_all_exemplar_scores()

        return {
            "method": "HYPE",
            "config": {
                "num_generations": self.config.num_generations,
                "bmin": self.config.bmin,
                "eta": self.config.eta,
                "num_new_instructions": self.config.num_new_instructions,
                "num_new_exemplars": self.config.num_new_exemplars,
            },
            "best_prompt": {
                "instruction_id": self.best_prompt.instruction_id if self.best_prompt else None,
                "exemplar_id": self.best_prompt.exemplar_id if self.best_prompt else None,
                "instruction": self.best_prompt.instruction if self.best_prompt else None,
                "exemplar": self.best_prompt.exemplar if self.best_prompt else None,
            },
            "best_error": self.best_error,
            "best_generation": self.best_generation,
            "generation_history": [
                {
                    "generation": h.generation,
                    "num_instructions": h.num_instructions,
                    "num_exemplars": h.num_exemplars,
                    "best_error": h.best_error,
                    "new_instructions": h.new_instructions_added,
                    "new_exemplars": h.new_exemplars_added,
                    "evaluations": h.hyperband_evaluations,
                    "timestamp": h.timestamp,
                }
                for h in self.history
            ],
            "component_scores": {
                "instructions": {
                    str(k): {"score": v.score, "variance": v.variance, "evaluations": v.num_evaluations}
                    for k, v in inst_scores.items()
                },
                "exemplars": {
                    str(k): {"score": v.score, "variance": v.variance, "evaluations": v.num_evaluations}
                    for k, v in ex_scores.items()
                }
            },
            "final_pools": {
                "num_instructions": len(self.instructions),
                "num_exemplars": len(self.exemplars),
                "instructions": [
                    {"id": i.id, "source": i.source.value, "generation": i.generation}
                    for i in self.instructions
                ],
                "exemplars": [
                    {"id": e.id, "source": e.source.value, "generation": e.generation}
                    for e in self.exemplars
                ]
            }
        }

    def save_results(self, output_dir: Path, model_name: str = "unknown") -> Tuple[Path, Path]:
        """Save results to JSON and TXT files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON with full results
        results = self.get_results()
        results["model"] = model_name
        results["timestamp"] = timestamp

        json_path = output_dir / f"hype_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # TXT with best prompt
        txt_path = output_dir / f"hype_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"HYPE Results - {timestamp}\n")
            f.write(f"{'='*40}\n\n")
            f.write(f"Best Error: {self.best_error:.4f}\n")
            f.write(f"Best Accuracy: {1 - self.best_error:.2%}\n")
            f.write(f"Found in Generation: {self.best_generation}\n\n")

            if self.best_prompt:
                f.write(f"BEST INSTRUCTION (ID: {self.best_prompt.instruction_id}):\n")
                f.write(f"{self.best_prompt.instruction}\n\n")
                f.write(f"BEST EXEMPLAR (ID: {self.best_prompt.exemplar_id}):\n")
                f.write(f"{self.best_prompt.exemplar}\n")

        return json_path, txt_path

    def save_checkpoint(self, output_dir: Path, model_name: str = "unknown") -> Path:
        """
        Save intermediate checkpoint after each generation.

        This prevents data loss if the process crashes mid-run.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use fixed filename for checkpoint (overwritten each generation)
        checkpoint_path = output_dir / "hype_checkpoint.json"

        results = self.get_results()
        results["model"] = model_name
        results["checkpoint_time"] = datetime.now().isoformat()
        results["is_checkpoint"] = True

        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)

        return checkpoint_path
