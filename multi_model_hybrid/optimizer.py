"""
Multi-Model Hybrid Optimizer with per-model GP selection.

This module extends hybrid_opro_hbbops with:
1. Multi-output GP for cross-model predictions (ICM kernel)
2. Per-model independent candidate selection (top 10 each)
3. Union-based candidate pooling (up to 30 unique)
4. Batch per-model evaluation with Hoeffding bounds

Algorithm (5 phases per iteration):
    Phase 1: Initial screening (Hyperband or load from file)
    Phase 2: OPRO generates 8 new instructions
    Phase 3: Per-model GP selection (8×25=200 → top 10 per model → union ≤30)
    Phase 4: Batch per-model evaluation with Hoeffding bounds
    Phase 5: Retrain multi-output GP
"""
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_model_hybrid.config import (
    MultiModelHybridConfig,
    MultiModelHybridCandidate,
    MultiModelHybridDesignPoint,
)
from multi_model_hybrid.per_model_selector import PerModelSelector
from multi_model_hybrid.batch_evaluator import BatchPerModelEvaluator

from multi_model_optimizer.multi_output_gp import MultiOutputGPTrainer
from multi_model_optimizer.aggregation import aggregate_scores, aggregate_accuracies
from multi_model_optimizer.evaluator_pool import SingleGPUModelManager

from hybrid_opro_hbbops.opro_adapter import OPROInstructionGenerator
from hybrid_opro_hbbops.exemplar_sampler import ExemplarSampler
from hybrid_opro_hbbops.config import ScoredInstruction

from multi_model_hybrid.ape_forward import APEForwardGenerator


class SingleGPUMetaLLM:
    """
    Wrapper for SingleGPUModelManager to provide LLMClient-compatible interface.

    Used by OPROInstructionGenerator in single-GPU mode.
    """

    def __init__(self, model_name: str, manager: SingleGPUModelManager):
        """
        Args:
            model_name: Model to use for generation
            manager: SingleGPUModelManager instance
        """
        self.model_name = model_name
        self.manager = manager

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response."""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a batch of prompts."""
        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512))
        temperature = kwargs.get("temperature", 0.0)

        client = self.manager.load_model(self.model_name, max_new_tokens)

        responses = client.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return responses


class MultiModelHybridOptimizer:
    """
    Multi-Model Hybrid OPRO + HbBoPs Optimizer with per-model selection.

    Finds ONE prompt (instruction + exemplar) that maximizes aggregated
    accuracy across all target models using per-model GP-based selection.

    Key difference from multi_model_optimizer:
        - Phase 3 selects candidates INDEPENDENTLY per model using UCB
        - Total candidates = union of per-model top-10 (up to 30 unique)
        - Phase 4 evaluates all union candidates on each model sequentially

    Example:
        >>> config = MultiModelHybridConfig()
        >>> optimizer = MultiModelHybridOptimizer(config, validation_data, train_data)
        >>> best_inst, best_ex, best_acc, per_model = optimizer.run(num_iterations=10)
    """

    def __init__(
        self,
        config: MultiModelHybridConfig,
        validation_data: List[Dict],
        train_data: List[Dict],
    ):
        """
        Initialize optimizer.

        Args:
            config: Multi-model hybrid configuration
            validation_data: List of {"question": str, "answer": str}
            train_data: GSM8K training data for exemplar sampling
        """
        self.config = config
        self.validation_data = validation_data
        self.train_data = train_data
        self.nvalid = len(validation_data)

        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize encoder
        self._init_encoder()

        # Get device
        self.device = self._get_device()

        # Initialize multi-output GP trainer
        self.gp_trainer = MultiOutputGPTrainer(
            model_names=config.target_models,
            latent_dim=config.gp_latent_dim,
            rank=config.gp_rank,
            train_epochs=config.gp_train_epochs,
            lr=config.gp_lr,
            patience=config.gp_patience,
            device=self.device,
        )

        # Single-GPU model manager
        self.model_manager = SingleGPUModelManager(
            gpu_id=config.single_gpu_id,
            gpu_memory_utilization=0.85,
        )

        # Per-model selector (initialized after GP is trained)
        self.per_model_selector: Optional[PerModelSelector] = None

        # Batch evaluator
        self.batch_evaluator = BatchPerModelEvaluator(
            config=config,
            validation_data=validation_data,
            single_gpu_manager=self.model_manager,
        )

        # Exemplar sampler
        self.exemplar_sampler = ExemplarSampler(
            gsm8k_train_data=train_data,
            seed=config.seed,
        )

        # OPRO generator (initialized lazily)
        self.opro_generator: Optional[OPROInstructionGenerator] = None

        # APE Forward generator (initialized lazily)
        self.ape_generator: Optional[APEForwardGenerator] = None

        # Registries
        self.instructions: Dict[int, str] = {}
        self.instruction_embeddings: Dict[int, np.ndarray] = {}
        self.exemplars: Dict[int, str] = {}
        self.exemplar_embeddings: Dict[int, np.ndarray] = {}
        self.next_instruction_id = 0
        self.next_exemplar_id = 0

        # Design data for GP training
        self.design_data: List[MultiModelHybridDesignPoint] = []

        # Best results
        self.best_instruction: Optional[str] = None
        self.best_instruction_id: int = -1
        self.best_exemplar: Optional[str] = None
        self.best_exemplar_id: int = -1
        self.best_aggregated_accuracy: float = 0.0
        self.best_per_model_accuracies: Dict[str, float] = {}

        # Budget tracking
        self.budget_used = 0

        # Iteration tracking
        self.iteration = 0
        self.iteration_results: List[Dict] = []

    def _get_device(self) -> torch.device:
        """Get torch device based on config."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _init_encoder(self):
        """Initialize BERT encoder for embeddings."""
        print(f"Loading encoder: {self.config.encoder_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.config.encoder_name)
        self.encoder.eval()

        # Use CPU for encoding to save GPU memory for models
        self.encoder_device = torch.device("cpu")
        self.encoder.to(self.encoder_device)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding using BERT.

        Args:
            text: Input text

        Returns:
            (768,) embedding array
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.encoder_device)

            outputs = self.encoder(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        return embedding

    def register_instruction(self, instruction: str) -> int:
        """Register instruction and return its ID."""
        # Check for duplicates
        for inst_id, inst in self.instructions.items():
            if inst == instruction:
                return inst_id

        inst_id = self.next_instruction_id
        self.next_instruction_id += 1
        self.instructions[inst_id] = instruction
        self.instruction_embeddings[inst_id] = self.encode_text(instruction)
        return inst_id

    def register_exemplar(self, exemplar: str) -> int:
        """Register exemplar and return its ID."""
        # Check for duplicates
        for ex_id, ex in self.exemplars.items():
            if ex == exemplar:
                return ex_id

        ex_id = self.next_exemplar_id
        self.next_exemplar_id += 1
        self.exemplars[ex_id] = exemplar
        self.exemplar_embeddings[ex_id] = self.encode_text(exemplar)
        return ex_id

    def run(
        self,
        num_iterations: int = 10,
        verbose: bool = True,
    ) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Run multi-model hybrid optimization.

        Args:
            num_iterations: Number of optimization iterations
            verbose: Print progress

        Returns:
            Tuple of:
                - best_instruction: Best instruction text
                - best_exemplar: Best exemplar text
                - best_aggregated_accuracy: Best aggregated accuracy
                - best_per_model_accuracies: Per-model accuracies
        """
        print("=" * 70)
        print("Multi-Model Hybrid Optimizer (Per-Model GP Selection)")
        print("=" * 70)
        print(f"Target models: {self.config.target_models}")
        print(f"Selection: Top {self.config.gp_top_k} per model, union up to "
              f"{self.config.per_model_selection.max_union_candidates}")
        print(f"Aggregation: {self.config.aggregation} (T={self.config.softmin_temperature})")
        print(f"Validation size: {self.nvalid}")
        print(f"Budget: {self.config.total_llm_budget}")
        print("=" * 70)

        # Phase 1: Initial screening
        self._run_phase1(verbose)

        # Main optimization loop
        for iteration in range(num_iterations):
            if self.budget_used >= self.config.total_llm_budget:
                print(f"\nBudget exhausted at iteration {iteration}")
                break

            self.iteration = iteration + 1
            print(f"\n{'='*70}")
            print(f"Iteration {self.iteration}/{num_iterations}")
            print(f"Budget: {self.budget_used:,}/{self.config.total_llm_budget:,}")
            print(f"Best: {self.best_aggregated_accuracy:.2%}")
            print("=" * 70)

            # Phase 2: OPRO instruction generation
            new_instructions = self._run_phase2_opro(verbose)
            if not new_instructions:
                print("No new instructions, ending optimization")
                break

            # Phase 3: Per-model GP selection
            union_candidates, per_model_info = self._run_phase3_selection(
                new_instructions, verbose
            )

            if not union_candidates:
                print("No candidates selected, ending optimization")
                break

            # Phase 4: Batch per-model evaluation
            self._run_phase4_evaluation(union_candidates, verbose)

            # Phase 5: Retrain GP
            self._run_phase5_gp_retrain(verbose)

            # Save iteration results
            self._save_iteration_results()

        # Final results
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")
        print("\nPer-model accuracies:")
        for model, acc in self.best_per_model_accuracies.items():
            model_short = model.split("/")[-1]
            print(f"  {model_short}: {acc:.2%}")
        if self.best_instruction:
            print(f"\nBest instruction:\n{self.best_instruction}")
        if self.best_exemplar:
            print(f"\nBest exemplar:\n{self.best_exemplar[:500]}...")

        return (
            self.best_instruction,
            self.best_exemplar,
            self.best_aggregated_accuracy,
            self.best_per_model_accuracies,
        )

    # =========================================================================
    # Phase Implementations
    # =========================================================================

    def _run_phase1(self, verbose: bool):
        """
        Phase 1: Initial screening.

        Either loads pre-computed results or runs initial Hyperband.
        """
        print("\n--- Phase 1: Initial Screening ---")

        if self.config.skip_phase1_hbbops:
            self._load_phase1_results(verbose)
        else:
            self._run_phase1_hyperband(verbose)

        # Train initial GP if we have data
        if len(self.design_data) > 0:
            self._train_gp_on_design_data(verbose)

    def _load_phase1_results(self, verbose: bool):
        """Load pre-computed Phase 1 results from file."""
        path = self.config.phase1_results_path
        print(f"Loading Phase 1 results from: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Phase 1 results file not found: {path}\n"
                f"Either set skip_phase1_hbbops=False or provide a valid phase1_results_path."
            )

        # Load initial instructions and exemplars
        # Note: When loading Phase 1 results, we always load from file
        # (APE forward is for fresh runs, not loading pre-computed results)
        self._load_initial_instructions()
        self._load_initial_exemplars()

        # Load results
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            raise RuntimeError(f"Failed to read Phase 1 results file: {path}: {e}") from e

        for line in lines:
            record = json.loads(line)
            inst_id = record["instruction_id"]
            ex_id = record["exemplar_id"]

            # Handle both single-model and multi-model formats
            if "model_error_rates" in record:
                model_error_rates = record["model_error_rates"]
            else:
                error_rate = record["error_rate"]
                model_error_rates = {
                    m: error_rate for m in self.config.target_models
                }

            fidelity = record.get("fidelity", self.nvalid)

            # Add to design data
            point = MultiModelHybridDesignPoint(
                instruction_id=inst_id,
                exemplar_id=ex_id,
                instruction_embedding=self.instruction_embeddings[inst_id],
                exemplar_embedding=self.exemplar_embeddings[ex_id],
                actual_model_errors=model_error_rates,
                actual_model_fidelities={m: fidelity for m in model_error_rates},
                aggregated_error=aggregate_scores(
                    model_error_rates,
                    self.config.aggregation,
                    self.config.softmin_temperature,
                ),
                evaluation_complete=True,
            )
            self.design_data.append(point)

            # Update best
            agg_accuracy = 1.0 - point.aggregated_error
            if agg_accuracy > self.best_aggregated_accuracy:
                self.best_aggregated_accuracy = agg_accuracy
                self.best_instruction = self.instructions[inst_id]
                self.best_instruction_id = inst_id
                self.best_exemplar = self.exemplars[ex_id]
                self.best_exemplar_id = ex_id
                self.best_per_model_accuracies = {
                    m: 1.0 - e for m, e in model_error_rates.items()
                }

        print(f"Loaded {len(self.design_data)} design points")
        print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")

    def _run_phase1_hyperband(self, verbose: bool):
        """Run initial screening (simplified version without full Hyperband)."""
        print("Running initial screening on instruction × exemplar grid...")

        # Generate or load initial instructions
        if self.config.use_ape_forward_init:
            self._generate_initial_instructions()
            self._generate_initial_exemplars()
            # Unload meta-model to free GPU memory for evaluation
            print("Unloading meta-model before evaluation...")
            self.model_manager.unload()
        else:
            self._load_initial_instructions()
            self._load_initial_exemplars()

        # Use batch evaluation at minimum fidelity
        min_fidelity = self.config.bmin

        # Use batch evaluation via evaluator pool - create once outside the loop
        from multi_model_optimizer.evaluator_pool import ModelEvaluatorPool
        pool = ModelEvaluatorPool(self.config)

        try:
            for inst_id in list(self.instructions.keys()):
                for ex_id in list(self.exemplars.keys()):
                    if self.budget_used >= self.config.total_llm_budget:
                        pool.cleanup()
                        return

                    # Evaluate on all models
                    instruction = self.instructions[inst_id]
                    exemplar = self.exemplars[ex_id]

                    model_error_rates = pool.evaluate_prompt_all_models(
                        instruction=instruction,
                        exemplar=exemplar,
                        validation_data=self.validation_data,
                        fidelity=min_fidelity,
                    )

                    self.budget_used += min_fidelity * len(self.config.target_models)

                    # Add to design data
                    point = MultiModelHybridDesignPoint(
                        instruction_id=inst_id,
                        exemplar_id=ex_id,
                        instruction_embedding=self.instruction_embeddings[inst_id],
                        exemplar_embedding=self.exemplar_embeddings[ex_id],
                        actual_model_errors=model_error_rates,
                        actual_model_fidelities={m: min_fidelity for m in model_error_rates},
                        aggregated_error=aggregate_scores(
                            model_error_rates,
                            self.config.aggregation,
                            self.config.softmin_temperature,
                        ),
                        evaluation_complete=True,
                    )
                    self.design_data.append(point)

                    # Update best
                    agg_accuracy = 1.0 - point.aggregated_error
                    if agg_accuracy > self.best_aggregated_accuracy:
                        self.best_aggregated_accuracy = agg_accuracy
                        self.best_instruction = instruction
                        self.best_instruction_id = inst_id
                        self.best_exemplar = exemplar
                        self.best_exemplar_id = ex_id
                        self.best_per_model_accuracies = {
                            m: 1.0 - e for m, e in model_error_rates.items()
                        }
        finally:
            pool.cleanup()

        if verbose:
            print(f"Evaluated {len(self.design_data)} initial prompts")
            print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")

    def _generate_initial_instructions(self) -> List[int]:
        """
        Generate initial instructions using APE forward pass.

        Uses the meta-LLM to infer instructions from training data examples,
        then clusters to select diverse set.
        """
        print("Generating initial instructions via APE forward pass...")

        # Initialize APE generator if needed
        if self.ape_generator is None:
            meta_llm = SingleGPUMetaLLM(
                model_name=self.config.meta_model,
                manager=self.model_manager,
            )
            self.ape_generator = APEForwardGenerator(
                meta_llm=meta_llm,
                encode_fn=self.encode_text,
                num_samples=self.config.ape_num_samples,
                num_candidates=self.config.ape_num_candidates,
                temperature=self.config.meta_temperature,
                max_tokens=self.config.meta_max_tokens,
                seed=self.config.seed,
            )

        # Generate instructions
        instructions = self.ape_generator.generate_instructions(
            train_data=self.train_data,
            num_final=self.config.ape_num_final,
            verbose=True,
        )

        # Register instructions
        inst_ids = [self.register_instruction(inst) for inst in instructions]
        print(f"Generated {len(inst_ids)} initial instructions via APE forward")
        return inst_ids

    def _generate_initial_exemplars(self) -> List[int]:
        """
        Generate initial exemplars by sampling Q/A pairs from train data.

        Samples num_initial_exemplars exemplars, each with initial_exemplar_qa_pairs
        Q/A pairs (typically 2 pairs for Stage 1, vs 5 pairs for dynamic exemplars).
        """
        print(f"Sampling {self.config.num_initial_exemplars} initial exemplars "
              f"({self.config.initial_exemplar_qa_pairs} Q/A pairs each)...")

        exemplars = self.exemplar_sampler.sample(
            n=self.config.num_initial_exemplars,
            k=self.config.initial_exemplar_qa_pairs,
        )

        # Register exemplars
        ex_ids = [self.register_exemplar(ex) for ex in exemplars]
        print(f"Sampled {len(ex_ids)} initial exemplars")
        return ex_ids

    def _load_initial_instructions(self) -> List[int]:
        """Load initial instructions from file (fallback)."""
        path = self.config.initial_instructions_path
        print(f"Loading initial instructions from: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Initial instructions file not found: {path}\n"
                f"Set use_ape_forward_init=True to generate instructions, "
                f"or provide a valid initial_instructions_path."
            )

        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read instructions file: {path}: {e}") from e

        instructions = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                idx = line.find(".")
                if idx > 0:
                    line = line[idx + 1:].strip()
                instructions.append(line)

        inst_ids = [self.register_instruction(inst) for inst in instructions]
        print(f"Loaded {len(inst_ids)} initial instructions")
        return inst_ids

    def _load_initial_exemplars(self) -> List[int]:
        """Load initial exemplars from file."""
        path = self.config.initial_exemplars_path
        print(f"Loading initial exemplars from: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Initial exemplars file not found: {path}\n"
                f"Set use_ape_forward_init=True to generate exemplars, "
                f"or provide a valid initial_exemplars_path."
            )

        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read exemplars file: {path}: {e}") from e

        # Parse exemplar blocks (separated by 80 equals signs)
        blocks = content.split("=" * 80)
        exemplars = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Parse Q/A pairs - skip comment lines starting with #
            lines = block.split("\n")
            examples = []
            current_q = None
            current_a_lines = []

            for line in lines:
                stripped = line.strip()

                # Skip empty lines and comment lines
                if not stripped or stripped.startswith("#"):
                    continue

                if stripped.startswith("Q:"):
                    # Save previous Q/A if exists
                    if current_q and current_a_lines:
                        full_answer = " ".join(current_a_lines)
                        examples.append(f"Q: {current_q}\nA: {full_answer}")
                    # Start new question
                    current_q = stripped[2:].strip()
                    current_a_lines = []
                elif stripped.startswith("A:") and current_q:
                    current_a_lines.append(stripped[2:].strip())
                elif current_a_lines:
                    # Continuation of answer
                    current_a_lines.append(stripped)

            # Don't forget last Q/A pair
            if current_q and current_a_lines:
                full_answer = " ".join(current_a_lines)
                examples.append(f"Q: {current_q}\nA: {full_answer}")

            if examples:
                exemplars.append("\n\n".join(examples))

        ex_ids = [self.register_exemplar(ex) for ex in exemplars]
        print(f"Loaded {len(ex_ids)} initial exemplars")
        return ex_ids

    def _run_phase2_opro(self, verbose: bool) -> List[str]:
        """
        Phase 2: Generate new instructions using OPRO.

        Uses aggregated scores in the meta-prompt.
        """
        print("\n--- Phase 2: OPRO Instruction Generation ---")

        self._init_opro_generator()

        # Get top-k instructions by aggregated accuracy
        scored_instructions = self._get_scored_instructions()

        if verbose:
            print(f"Using {len(scored_instructions)} top instructions for OPRO context")

        # Generate new instructions
        new_instructions = self.opro_generator.generate_candidates(
            scored_instructions=scored_instructions,
            existing_instructions=set(self.instructions.values()),
            verbose=verbose,
        )

        # Filter duplicates
        new_instructions = [
            inst for inst in new_instructions
            if inst not in self.instructions.values()
        ]

        # Register new instructions
        for inst in new_instructions:
            self.register_instruction(inst)

        if verbose:
            print(f"Generated {len(new_instructions)} new unique instructions")

        return new_instructions

    def _init_opro_generator(self):
        """Initialize OPRO generator with meta-model."""
        if self.opro_generator is not None:
            return

        print(f"Initializing OPRO generator with model: {self.config.meta_model}")

        # Create meta-LLM wrapper
        meta_llm = SingleGPUMetaLLM(
            model_name=self.config.meta_model,
            manager=self.model_manager,
        )

        self.opro_generator = OPROInstructionGenerator(
            meta_llm=meta_llm,
            num_candidates=self.config.opro_candidates_per_iter,
            temperature=self.config.meta_temperature,
            max_tokens=self.config.meta_max_tokens,
        )

    def _get_scored_instructions(self) -> List[ScoredInstruction]:
        """Get scored instructions for OPRO context."""
        # Collect scores per instruction
        inst_scores: Dict[int, List[float]] = {}

        for point in self.design_data:
            inst_id = point.instruction_id
            if inst_id not in inst_scores:
                inst_scores[inst_id] = []
            inst_scores[inst_id].append(1.0 - point.aggregated_error)

        # Build scored instructions
        scored_instructions = []
        for inst_id, scores in inst_scores.items():
            best_acc = max(scores)
            scored_instructions.append(
                ScoredInstruction(
                    instruction=self.instructions[inst_id],
                    instruction_id=inst_id,
                    best_accuracy=best_acc,
                    best_exemplar_id=0,
                )
            )

        # Sort and take top-k
        scored_instructions.sort(key=lambda x: x.best_accuracy, reverse=True)
        return scored_instructions[:self.config.opro_keep_top_k]

    def _run_phase3_selection(
        self,
        new_instructions: List[str],
        verbose: bool,
    ) -> Tuple[List[MultiModelHybridCandidate], Dict[str, List[int]]]:
        """
        Phase 3: Per-model GP selection.

        Creates instruction × exemplar grid, then selects:
        - Top 10 candidates per model (based on GP predictions)
        - Returns union (up to 30 unique candidates)
        """
        print("\n--- Phase 3: Per-Model GP Selection ---")

        # Sample new exemplars
        new_exemplars = self.exemplar_sampler.sample(
            n=self.config.num_dynamic_exemplars,
            k=self.config.exemplars_per_sample,
        )
        ex_ids = [self.register_exemplar(ex) for ex in new_exemplars]

        if verbose:
            print(f"  Sampled {len(new_exemplars)} new exemplars")

        # Create candidates (instruction × exemplar grid)
        candidates = []
        for inst in new_instructions:
            inst_id = self.register_instruction(inst)
            inst_emb = self.instruction_embeddings[inst_id]

            for ex_id in ex_ids:
                candidates.append(MultiModelHybridCandidate(
                    instruction=inst,
                    instruction_id=inst_id,
                    instruction_embedding=inst_emb,
                    exemplar=self.exemplars[ex_id],
                    exemplar_id=ex_id,
                    exemplar_embedding=self.exemplar_embeddings[ex_id],
                ))

        if verbose:
            print(f"  Created {len(candidates)} candidates "
                  f"({len(new_instructions)} inst × {len(ex_ids)} ex)")

        # Initialize selector if needed
        if self.per_model_selector is None:
            self.per_model_selector = PerModelSelector(
                gp_trainer=self.gp_trainer,
                config=self.config,
            )

        # Select per model
        if self.gp_trainer.gp_params is None:
            # GP not trained yet - use random selection
            random.shuffle(candidates)
            max_n = self.config.per_model_selection.max_union_candidates
            return candidates[:max_n], {}

        union_candidates, per_model_info = self.per_model_selector.select_candidates(
            candidates
        )

        if verbose:
            print(f"  Selected {len(union_candidates)} unique candidates (union)")
            for model, indices in per_model_info.items():
                model_short = model.split("/")[-1]
                print(f"    {model_short}: top {len(indices)}")

        return union_candidates, per_model_info

    def _run_phase4_evaluation(
        self,
        candidates: List[MultiModelHybridCandidate],
        verbose: bool,
    ) -> None:
        """
        Phase 4: Batch per-model evaluation with Hoeffding bounds.

        Loads each model once, evaluates ALL candidates, then switches.
        """
        print("\n--- Phase 4: Batch Per-Model Evaluation ---")
        print(f"  Evaluating {len(candidates)} candidates on "
              f"{len(self.config.target_models)} models")

        budget_remaining = self.config.total_llm_budget - self.budget_used

        evaluated_candidates, budget_consumed = self.batch_evaluator.evaluate_candidates_batch(
            candidates=candidates,
            best_aggregated_accuracy=self.best_aggregated_accuracy,
            budget_remaining=budget_remaining,
            verbose=verbose,
        )

        self.budget_used += budget_consumed

        # Process results and update best
        for candidate in evaluated_candidates:
            if not candidate.actual_errors:
                continue

            # Check if we have results for all models
            if len(candidate.actual_errors) == len(self.config.target_models):
                accuracies = {m: 1 - e for m, e in candidate.actual_errors.items()}
                agg_acc = aggregate_accuracies(
                    accuracies,
                    self.config.aggregation,
                    self.config.softmin_temperature,
                )
                candidate.aggregated_accuracy = agg_acc

                # Update best
                if agg_acc > self.best_aggregated_accuracy:
                    self.best_aggregated_accuracy = agg_acc
                    self.best_instruction = candidate.instruction
                    self.best_instruction_id = candidate.instruction_id
                    self.best_exemplar = candidate.exemplar
                    self.best_exemplar_id = candidate.exemplar_id
                    self.best_per_model_accuracies = accuracies.copy()

                    if verbose:
                        print(f"\n  *** New best: {agg_acc:.2%} ***")

            # Add to design data
            self._add_to_design_data(candidate)

    def _add_to_design_data(self, candidate: MultiModelHybridCandidate) -> None:
        """Add evaluated candidate to GP training data."""
        point = MultiModelHybridDesignPoint(
            instruction_id=candidate.instruction_id,
            exemplar_id=candidate.exemplar_id,
            instruction_embedding=candidate.instruction_embedding,
            exemplar_embedding=candidate.exemplar_embedding,
            actual_model_errors=candidate.actual_errors.copy(),
            actual_model_fidelities=candidate.actual_fidelities.copy(),
            aggregated_error=1 - (candidate.aggregated_accuracy or 0),
            evaluation_complete=len(candidate.actual_errors) == len(self.config.target_models),
        )
        self.design_data.append(point)

    def _run_phase5_gp_retrain(self, verbose: bool):
        """Phase 5: Retrain GP on accumulated high-fidelity data."""
        print("\n--- Phase 5: GP Retraining ---")

        # Filter to high-fidelity points
        min_fidelity = self.nvalid // 4  # At least 25% of validation set
        high_fidelity_points = [
            p for p in self.design_data
            if p.evaluation_complete and
            all(f >= min_fidelity for f in p.actual_model_fidelities.values())
        ]

        if len(high_fidelity_points) < 10:
            print(f"  Not enough high-fidelity data ({len(high_fidelity_points)}), skipping retrain")
            return

        self._train_gp_on_design_data(verbose, min_fidelity=min_fidelity)

    def _train_gp_on_design_data(
        self,
        verbose: bool,
        min_fidelity: int = 0,
    ):
        """Train GP on design data."""
        # Filter points with data for all models
        points = [
            p for p in self.design_data
            if p.evaluation_complete and
            all(f >= min_fidelity for f in p.actual_model_fidelities.values())
        ]

        if len(points) < 5:
            print(f"  Not enough data points ({len(points)}), skipping GP training")
            return

        # Prepare training data
        inst_embs = np.array([p.instruction_embedding for p in points])
        ex_embs = np.array([p.exemplar_embedding for p in points])

        model_error_rates = {
            m: np.array([p.actual_model_errors.get(m, 0.5) for p in points])
            for m in self.config.target_models
        }

        if verbose:
            print(f"  Training GP on {len(points)} points")

        try:
            self.gp_trainer.train(
                instruction_embeddings=inst_embs,
                exemplar_embeddings=ex_embs,
                model_error_rates=model_error_rates,
                verbose=verbose,
            )
        except Exception as e:
            print(f"  GP training failed: {e}")

    def _save_iteration_results(self):
        """Save iteration results to file."""
        result = {
            "iteration": self.iteration,
            "budget_used": self.budget_used,
            "best_aggregated_accuracy": self.best_aggregated_accuracy,
            "best_per_model_accuracies": self.best_per_model_accuracies,
            "best_instruction_id": self.best_instruction_id,
            "best_exemplar_id": self.best_exemplar_id,
            "num_instructions": len(self.instructions),
            "num_exemplars": len(self.exemplars),
            "num_design_points": len(self.design_data),
            "timestamp": datetime.now().isoformat(),
        }
        self.iteration_results.append(result)

        # Save to file
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, "iteration_results.jsonl")

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def save_final_results(self):
        """Save final results to file."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        final_result = {
            "best_instruction": self.best_instruction,
            "best_exemplar": self.best_exemplar,
            "best_aggregated_accuracy": self.best_aggregated_accuracy,
            "best_per_model_accuracies": self.best_per_model_accuracies,
            "total_budget_used": self.budget_used,
            "num_iterations": len(self.iteration_results),
            "config": {
                "target_models": self.config.target_models,
                "aggregation": self.config.aggregation,
                "softmin_temperature": self.config.softmin_temperature,
                "top_k_per_model": self.config.per_model_selection.top_k_per_model,
                "max_union_candidates": self.config.per_model_selection.max_union_candidates,
            },
            "timestamp": datetime.now().isoformat(),
        }

        output_path = os.path.join(self.config.output_dir, "final_results.json")
        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=2)

        print(f"Results saved to: {output_path}")

    def cleanup(self):
        """Clean up resources."""
        if self.model_manager is not None:
            self.model_manager.unload()
