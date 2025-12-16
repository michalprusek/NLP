"""
Multi-Model Hybrid OPRO + HbBoPs Optimizer.

Finds a single universal prompt that works well across multiple frontier LLMs.
Uses multi-output GP with ICM kernel for cross-model correlation learning
and Bonferroni-corrected sequential testing for efficient evaluation.
"""
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Add parent directory for imports
sys.path.insert(0, "/home/prusek/NLP")

from multi_model_optimizer.config import (
    MultiModelConfig,
    MultiModelDesignPoint,
    MultiModelPromptCandidate,
)
from multi_model_optimizer.aggregation import aggregate_scores, aggregate_accuracies
from multi_model_optimizer.evaluator_pool import ModelEvaluatorPool
from multi_model_optimizer.multi_output_gp import MultiOutputGPTrainer
from multi_model_optimizer.sequential_tester import (
    Decision,
    MultiModelSequentialTester,
)
from hybrid_opro_hbbops.opro_adapter import OPROInstructionGenerator
from hybrid_opro_hbbops.exemplar_sampler import ExemplarSampler


class SingleGPUMetaLLM:
    """
    Wrapper for SingleGPUModelManager to provide LLMClient-compatible interface.

    Used by OPROInstructionGenerator in single-GPU mode.
    """

    def __init__(self, model_name: str, manager):
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
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.0)

        # Load model if needed
        client = self.manager.load_model(self.model_name, max_new_tokens)

        # Generate
        responses = client.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return responses


class MultiModelHybridOptimizer:
    """
    Multi-model universal prompt optimizer.

    Finds ONE prompt (instruction + exemplar) that maximizes aggregated
    accuracy across all target models.

    Algorithm (5 phases per iteration):
        Phase 1: Initial Hyperband screening on all models (or load from file)
        Phase 2: OPRO instruction generation using aggregated scores
        Phase 3: Multi-output GP screening (predicts for all models at once)
        Phase 4: Sequential testing with Bonferroni-corrected bounds
        Phase 5: Retrain multi-output GP on accumulated data

    Example:
        >>> config = MultiModelConfig()
        >>> optimizer = MultiModelHybridOptimizer(config, validation_data, train_data)
        >>> best_prompt, best_accuracy = optimizer.run(num_iterations=10)
    """

    def __init__(
        self,
        config: MultiModelConfig,
        validation_data: List[Dict[str, str]],
        train_data: List[Dict[str, str]],
    ):
        """
        Initialize the multi-model optimizer.

        Args:
            config: Multi-model configuration
            validation_data: List of {"question": str, "answer": str} for evaluation
            train_data: GSM8K training data for exemplar sampling
        """
        self.config = config
        self.validation_data = validation_data
        self.train_data = train_data
        self.nvalid = len(validation_data)

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize encoder for embeddings
        self._init_encoder()

        # Initialize components
        self.evaluator_pool = ModelEvaluatorPool(config)

        self.gp_trainer = MultiOutputGPTrainer(
            model_names=config.target_models,
            latent_dim=config.gp_latent_dim,
            rank=config.gp_rank,
            train_epochs=config.gp_train_epochs,
            lr=config.gp_lr,
            patience=config.gp_patience,
            device=self._get_device(),
        )

        self.sequential_tester = MultiModelSequentialTester(
            model_names=config.target_models,
            confidence=config.sequential_confidence,
            min_samples=config.sequential_min_samples,
            min_promote_samples=config.sequential_min_promote_samples,
            aggregation=config.aggregation,
            temperature=config.softmin_temperature,
        )

        self.exemplar_sampler = ExemplarSampler(
            gsm8k_train_data=train_data,
            seed=config.seed,
        )

        # OPRO generator (initialized lazily when meta-model is needed)
        self.opro_generator: Optional[OPROInstructionGenerator] = None

        # Design data storage (for GP training)
        self.design_data: List[MultiModelDesignPoint] = []

        # Registries
        self.instructions: Dict[int, str] = {}
        self.instruction_embeddings: Dict[int, np.ndarray] = {}
        self.exemplars: Dict[int, str] = {}
        self.exemplar_embeddings: Dict[int, np.ndarray] = {}
        self.next_instruction_id = 0
        self.next_exemplar_id = 0

        # Cache for evaluations: (inst_id, ex_id, fidelity) -> Dict[model, error_rate]
        self.eval_cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}

        # Best results tracking
        self.best_instruction: Optional[str] = None
        self.best_instruction_id: int = -1
        self.best_exemplar: Optional[str] = None
        self.best_exemplar_id: int = -1
        self.best_aggregated_accuracy: float = 0.0
        self.best_per_model_accuracies: Dict[str, float] = {}

        # Budget tracking
        self.budget_used = 0

        # Results logging
        self.iteration_results: List[Dict[str, Any]] = []

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

        device = self._get_device()
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
            # Use [CLS] token embedding
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

    def _load_initial_instructions(self) -> List[int]:
        """Load initial instructions from file."""
        path = self.config.initial_instructions_path
        print(f"Loading initial instructions from: {path}")

        with open(path, "r") as f:
            content = f.read()

        # Parse numbered list (1. instruction\n2. instruction\n...)
        instructions = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                idx = line.find(".")
                if idx > 0:
                    line = line[idx + 1 :].strip()
                instructions.append(line)

        inst_ids = [self.register_instruction(inst) for inst in instructions]
        print(f"Loaded {len(inst_ids)} initial instructions")
        return inst_ids

    def _load_initial_exemplars(self) -> List[int]:
        """Load initial exemplars from file."""
        path = self.config.initial_exemplars_path
        print(f"Loading initial exemplars from: {path}")

        with open(path, "r") as f:
            content = f.read()

        # Parse exemplar blocks (separated by double newlines or markers)
        blocks = content.split("\n\n")
        exemplars = [b.strip() for b in blocks if b.strip()]

        ex_ids = [self.register_exemplar(ex) for ex in exemplars]
        print(f"Loaded {len(ex_ids)} initial exemplars")
        return ex_ids

    def _evaluate_prompt(
        self,
        instruction_id: int,
        exemplar_id: int,
        fidelity: int,
    ) -> Dict[str, float]:
        """
        Evaluate a prompt on all models.

        Uses cache to avoid redundant evaluations.

        Args:
            instruction_id: Instruction registry ID
            exemplar_id: Exemplar registry ID
            fidelity: Number of validation samples

        Returns:
            Dict mapping model_name -> error_rate
        """
        cache_key = (instruction_id, exemplar_id, fidelity)

        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        instruction = self.instructions[instruction_id]
        exemplar = self.exemplars[exemplar_id]

        # Evaluate on all models
        error_rates = self.evaluator_pool.evaluate_prompt_all_models(
            instruction=instruction,
            exemplar=exemplar,
            validation_data=self.validation_data,
            fidelity=fidelity,
        )

        # Update budget (count each model separately)
        self.budget_used += fidelity * len(self.config.target_models)

        # Cache result
        self.eval_cache[cache_key] = error_rates

        return error_rates

    def _init_opro_generator(self):
        """Initialize OPRO generator with meta-model."""
        if self.opro_generator is not None:
            return

        print(f"Initializing OPRO generator with model: {self.config.meta_model}")
        print(f"  Backend: {self.config.meta_model_backend}")

        # Create LLM client for meta-model
        from src.llm_client import create_llm_client, GeminiClient

        # Check if using API-based meta-model (Gemini, OpenAI)
        if self.config.meta_model_backend == "gemini":
            # Use Gemini API directly
            meta_llm = GeminiClient(
                model_name=self.config.meta_model,
                api_key=self.config.meta_model_api_key,
            )
        elif self.config.meta_model_backend in ("openai", "deepinfra"):
            # Use OpenAI-compatible API
            meta_llm = create_llm_client(
                self.config.meta_model,
                backend=self.config.meta_model_backend,
            )
        elif self.config.single_gpu and self.evaluator_pool.single_gpu_manager is not None:
            # Single-GPU mode: use local model manager
            meta_llm = SingleGPUMetaLLM(
                model_name=self.config.meta_model,
                manager=self.evaluator_pool.single_gpu_manager,
            )
        else:
            # Multi-GPU mode: create dedicated vLLM client
            meta_llm = create_llm_client(
                self.config.meta_model,
                backend="vllm",
            )

        self.opro_generator = OPROInstructionGenerator(
            meta_llm=meta_llm,
            num_candidates=self.config.opro_candidates_per_iter,
            temperature=self.config.meta_temperature,
            max_tokens=self.config.meta_max_tokens,
        )

    def run(
        self,
        num_iterations: int = 10,
        verbose: bool = True,
    ) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Run multi-model optimization.

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
        print("=" * 60)
        print("Multi-Model Hybrid OPRO + HbBoPs Optimizer")
        print("=" * 60)
        print(f"Target models: {self.config.target_models}")
        print(f"Aggregation: {self.config.aggregation} (T={self.config.softmin_temperature})")
        print(f"Validation size: {self.nvalid}")
        print(f"Budget: {self.config.total_llm_budget}")
        print("=" * 60)

        # Phase 1: Initial screening
        self._run_phase1(verbose)

        # Choose optimization mode
        if self.config.model_sequential_mode and self.config.single_gpu:
            return self._run_model_sequential(num_iterations, verbose)

        # Original multi-model loop
        for iteration in range(num_iterations):
            if self.budget_used >= self.config.total_llm_budget:
                print(f"\nBudget exhausted at iteration {iteration}")
                break

            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"Budget: {self.budget_used:,}/{self.config.total_llm_budget:,}")
            print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")
            print("=" * 60)

            # Phase 2: OPRO instruction generation
            new_instructions = self._run_phase2_opro(verbose)

            if not new_instructions:
                print("No new instructions generated, ending optimization")
                break

            # Phase 3: GP screening
            top_candidates = self._run_phase3_gp_screening(new_instructions, verbose)

            # Phase 4: Sequential evaluation
            self._run_phase4_evaluation(top_candidates, verbose)

            # Phase 5: Retrain GP
            self._run_phase5_gp_retrain(verbose)

            # Save iteration results
            self._save_iteration_results(iteration)

        # Final results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")
        print("\nPer-model accuracies:")
        for model, acc in self.best_per_model_accuracies.items():
            print(f"  {model}: {acc:.2%}")
        print(f"\nBest instruction:\n{self.best_instruction}")
        print(f"\nBest exemplar:\n{self.best_exemplar[:500]}...")

        return (
            self.best_instruction,
            self.best_exemplar,
            self.best_aggregated_accuracy,
            self.best_per_model_accuracies,
        )

    def _run_phase1(self, verbose: bool):
        """
        Phase 1: Initial Hyperband screening.

        Either runs full HbBoPs on all models or loads pre-computed results.
        """
        print("\n--- Phase 1: Initial Screening ---")

        if self.config.skip_phase1_hbbops:
            self._load_phase1_results()
        else:
            self._run_phase1_hyperband(verbose)

        # Train initial GP
        if len(self.design_data) > 0:
            self._train_gp_on_design_data(verbose)

    def _load_phase1_results(self):
        """Load pre-computed Phase 1 results from file."""
        path = self.config.phase1_results_path
        print(f"Loading Phase 1 results from: {path}")

        # Load initial instructions and exemplars
        init_inst_ids = self._load_initial_instructions()
        init_ex_ids = self._load_initial_exemplars()

        # Load results
        with open(path, "r") as f:
            for line in f:
                record = json.loads(line)
                inst_id = record["instruction_id"]
                ex_id = record["exemplar_id"]

                # This is for single-model results, need to adapt for multi-model
                if "model_error_rates" in record:
                    model_error_rates = record["model_error_rates"]
                else:
                    # Single model format: convert to multi-model
                    error_rate = record["error_rate"]
                    model_error_rates = {
                        m: error_rate for m in self.config.target_models
                    }

                fidelity = record.get("fidelity", self.nvalid)

                # Add to design data
                point = MultiModelDesignPoint(
                    instruction_id=inst_id,
                    exemplar_id=ex_id,
                    instruction_embedding=self.instruction_embeddings[inst_id],
                    exemplar_embedding=self.exemplar_embeddings[ex_id],
                    model_error_rates=model_error_rates,
                    aggregated_error=aggregate_scores(
                        model_error_rates,
                        self.config.aggregation,
                        self.config.softmin_temperature,
                    ),
                    fidelity=fidelity,
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
        """Run Hyperband screening on all models."""
        print("Running Hyperband on initial grid...")

        init_inst_ids = self._load_initial_instructions()
        init_ex_ids = self._load_initial_exemplars()

        # Full grid evaluation at minimum fidelity
        min_fidelity = self.config.bmin

        for inst_id in init_inst_ids:
            for ex_id in init_ex_ids:
                if self.budget_used >= self.config.total_llm_budget:
                    return

                model_error_rates = self._evaluate_prompt(inst_id, ex_id, min_fidelity)

                point = MultiModelDesignPoint(
                    instruction_id=inst_id,
                    exemplar_id=ex_id,
                    instruction_embedding=self.instruction_embeddings[inst_id],
                    exemplar_embedding=self.exemplar_embeddings[ex_id],
                    model_error_rates=model_error_rates,
                    aggregated_error=aggregate_scores(
                        model_error_rates,
                        self.config.aggregation,
                        self.config.softmin_temperature,
                    ),
                    fidelity=min_fidelity,
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

        if verbose:
            print(f"Evaluated {len(self.design_data)} initial prompts")
            print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")

    def _run_phase2_opro(self, verbose: bool) -> List[str]:
        """
        Phase 2: Generate new instructions using OPRO.

        Uses aggregated scores in the meta-prompt.
        """
        print("\n--- Phase 2: OPRO Instruction Generation ---")

        self._init_opro_generator()

        # Get top-k instructions by aggregated accuracy
        scored_instructions = []
        for point in self.design_data:
            scored_instructions.append(
                (self.instructions[point.instruction_id], 1.0 - point.aggregated_error)
            )

        # Deduplicate and sort
        unique_instructions = {}
        for inst, acc in scored_instructions:
            if inst not in unique_instructions or acc > unique_instructions[inst]:
                unique_instructions[inst] = acc

        sorted_instructions = sorted(
            unique_instructions.items(), key=lambda x: x[1], reverse=True
        )[:self.config.opro_keep_top_k]

        if verbose:
            print(f"Using {len(sorted_instructions)} top instructions for OPRO context")

        # Convert to ScoredInstruction format for OPRO
        from hybrid_opro_hbbops.config import ScoredInstruction
        scored_objs = [
            ScoredInstruction(
                instruction=inst,
                instruction_id=self.register_instruction(inst),
                best_accuracy=acc,
                best_exemplar_id=0,  # Not used by OPRO
            )
            for inst, acc in sorted_instructions
        ]

        # Generate new instructions
        new_instructions = self.opro_generator.generate_candidates(
            scored_instructions=scored_objs,
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

    def _run_phase3_gp_screening(
        self,
        new_instructions: List[str],
        verbose: bool,
    ) -> List[MultiModelPromptCandidate]:
        """
        Phase 3: GP-based screening of candidates.

        Uses multi-output GP to predict error rates for all models.
        Selects top-k by aggregated predicted accuracy.
        """
        print("\n--- Phase 3: GP Screening ---")

        if self.gp_trainer.gp_params is None:
            # GP not trained yet, skip screening and evaluate all
            print("GP not trained, evaluating all candidates")
            return self._create_all_candidates(new_instructions)

        # Sample new exemplars
        new_exemplars = []
        for _ in range(self.config.num_dynamic_exemplars):
            exemplar = self.exemplar_sampler.sample_single(k=self.config.exemplars_per_sample)
            if exemplar:
                new_exemplars.append(exemplar)

        if verbose:
            print(f"Sampled {len(new_exemplars)} new exemplars")

        # Register exemplars
        ex_ids = [self.register_exemplar(ex) for ex in new_exemplars]

        # Create candidates (instruction x exemplar grid)
        candidates = []
        for inst in new_instructions:
            inst_id = self.register_instruction(inst)
            inst_emb = self.instruction_embeddings[inst_id]

            for ex_id in ex_ids:
                exemplar = self.exemplars[ex_id]
                ex_emb = self.exemplar_embeddings[ex_id]

                candidates.append(
                    MultiModelPromptCandidate(
                        instruction=inst,
                        instruction_id=inst_id,
                        instruction_embedding=inst_emb,
                        exemplar=exemplar,
                        exemplar_id=ex_id,
                        exemplar_embedding=ex_emb,
                    )
                )

        if verbose:
            print(f"Created {len(candidates)} candidates")

        # GP prediction
        inst_embs = np.array([c.instruction_embedding for c in candidates])
        ex_embs = np.array([c.exemplar_embedding for c in candidates])

        aggregated_errors, per_model_errors = self.gp_trainer.predict_aggregated(
            inst_embs,
            ex_embs,
            self.config.aggregation,
            self.config.softmin_temperature,
        )

        for i, candidate in enumerate(candidates):
            candidate.gp_predicted_errors = {
                m: per_model_errors[m][i] for m in self.config.target_models
            }
            candidate.gp_aggregated_prediction = aggregated_errors[i]

        # Sort by aggregated prediction (lower error = better)
        candidates.sort(key=lambda c: c.gp_aggregated_prediction)
        top_candidates = candidates[: self.config.gp_top_k]

        if verbose:
            print(f"Selected top {len(top_candidates)} candidates by GP prediction")
            for i, c in enumerate(top_candidates[:3]):
                print(
                    f"  {i+1}. Predicted error: {c.gp_aggregated_prediction:.4f}"
                )

        return top_candidates

    def _create_all_candidates(
        self,
        new_instructions: List[str],
    ) -> List[MultiModelPromptCandidate]:
        """Create candidates without GP screening."""
        new_exemplars = []
        for _ in range(self.config.num_dynamic_exemplars):
            exemplar = self.exemplar_sampler.sample_single(k=self.config.exemplars_per_sample)
            if exemplar:
                new_exemplars.append(exemplar)

        ex_ids = [self.register_exemplar(ex) for ex in new_exemplars]

        candidates = []
        for inst in new_instructions:
            inst_id = self.register_instruction(inst)
            inst_emb = self.instruction_embeddings[inst_id]

            for ex_id in ex_ids:
                exemplar = self.exemplars[ex_id]
                ex_emb = self.exemplar_embeddings[ex_id]

                candidates.append(
                    MultiModelPromptCandidate(
                        instruction=inst,
                        instruction_id=inst_id,
                        instruction_embedding=inst_emb,
                        exemplar=exemplar,
                        exemplar_id=ex_id,
                        exemplar_embedding=ex_emb,
                    )
                )

        # Limit to gp_top_k (random selection)
        if len(candidates) > self.config.gp_top_k:
            random.shuffle(candidates)
            candidates = candidates[: self.config.gp_top_k]

        return candidates

    def _run_phase4_evaluation(
        self,
        candidates: List[MultiModelPromptCandidate],
        verbose: bool,
    ):
        """
        Phase 4: Batch evaluation with minimal model switching.

        Evaluates ALL candidates on each model before switching to next model.
        This minimizes GPU memory operations in single-GPU mode.
        """
        print("\n--- Phase 4: Batch Evaluation (Per-Model) ---")

        if not candidates:
            print("No candidates to evaluate")
            return

        # Use fixed fidelity for batch evaluation
        fidelity = min(100, self.nvalid)  # Reasonable sample size

        # Prepare candidates for batch evaluation
        cand_dicts = [
            {"instruction": c.instruction, "exemplar": c.exemplar}
            for c in candidates
        ]

        # Batch evaluate: all candidates on each model, then switch
        results = self.evaluator_pool.evaluate_candidates_batch_per_model(
            candidates=cand_dicts,
            validation_data=self.validation_data,
            fidelity=fidelity,
        )

        # Update budget
        self.budget_used += fidelity * len(candidates) * len(self.config.target_models)

        # Process results
        print(f"\n--- Results Summary ---")
        for i, (candidate, error_rates) in enumerate(zip(candidates, results)):
            # Convert error rates to accuracies
            accuracies = {m: 1.0 - err for m, err in error_rates.items()}

            # Compute aggregated accuracy
            agg_acc = aggregate_accuracies(
                accuracies,
                strategy=self.config.aggregation,
                temperature=self.config.softmin_temperature,
            )

            # Store results
            candidate.samples_used = fidelity
            candidate.actual_accuracies = accuracies
            candidate.aggregated_accuracy = agg_acc

            if verbose:
                print(f"\nCandidate {i+1}/{len(candidates)}:")
                print(f"  Aggregated accuracy: {agg_acc:.2%}")
                for m, acc in accuracies.items():
                    model_short = m.split("/")[-1]
                    print(f"    {model_short}: {acc:.2%}")

            # Update best if better
            if agg_acc > self.best_aggregated_accuracy:
                self.best_aggregated_accuracy = agg_acc
                self.best_instruction = candidate.instruction
                self.best_instruction_id = candidate.instruction_id
                self.best_exemplar = candidate.exemplar
                self.best_exemplar_id = candidate.exemplar_id
                self.best_per_model_accuracies = accuracies.copy()

                print(f"  *** New best! ***")

            # Add to design data
            model_error_rates = {m: 1.0 - acc for m, acc in accuracies.items()}
            point = MultiModelDesignPoint(
                instruction_id=candidate.instruction_id,
                exemplar_id=candidate.exemplar_id,
                instruction_embedding=candidate.instruction_embedding,
                exemplar_embedding=candidate.exemplar_embedding,
                model_error_rates=model_error_rates,
                aggregated_error=1.0 - agg_acc,
                fidelity=fidelity,
            )
            self.design_data.append(point)

    def _run_phase5_gp_retrain(self, verbose: bool):
        """Phase 5: Retrain GP on accumulated high-fidelity data."""
        print("\n--- Phase 5: GP Retraining ---")

        # Filter to high-fidelity points
        high_fidelity_points = [
            p for p in self.design_data if p.fidelity >= self.nvalid // 2
        ]

        if len(high_fidelity_points) < 10:
            print(f"Not enough high-fidelity data ({len(high_fidelity_points)}), skipping retrain")
            return

        self._train_gp_on_design_data(verbose, min_fidelity=self.nvalid // 2)

    def _train_gp_on_design_data(
        self,
        verbose: bool,
        min_fidelity: int = 0,
    ):
        """Train GP on design data."""
        points = [p for p in self.design_data if p.fidelity >= min_fidelity]

        if len(points) < 5:
            print(f"Not enough data points ({len(points)}), skipping GP training")
            return

        # Prepare training data
        inst_embs = np.array([p.instruction_embedding for p in points])
        ex_embs = np.array([p.exemplar_embedding for p in points])

        model_error_rates = {
            m: np.array([p.model_error_rates[m] for p in points])
            for m in self.config.target_models
        }

        if verbose:
            print(f"Training GP on {len(points)} points")

        self.gp_trainer.train(
            instruction_embeddings=inst_embs,
            exemplar_embeddings=ex_embs,
            model_error_rates=model_error_rates,
            verbose=verbose,
        )

    def _save_iteration_results(self, iteration: int):
        """Save iteration results to file."""
        result = {
            "iteration": iteration + 1,
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

    # =========================================================================
    # Model-Sequential Mode Methods
    # =========================================================================

    def _run_model_sequential(
        self,
        num_iterations: int,
        verbose: bool,
    ) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Model-sequential optimization: process each model completely before switching.

        This minimizes model switching by:
        1. Loading each model once
        2. Running N/M iterations with that model
        3. Using GP to predict cross-model performance
        4. Final verification on top candidates

        Args:
            num_iterations: Total number of optimization iterations
            verbose: Print progress

        Returns:
            Same as run(): (instruction, exemplar, accuracy, per_model_accuracies)
        """
        print("\n" + "=" * 60)
        print("MODEL-SEQUENTIAL MODE")
        print("=" * 60)
        print(f"Minimizing model switches: 4 main + verification")

        iterations_per_model = max(1, num_iterations // len(self.config.target_models))
        print(f"Iterations per model: {iterations_per_model}")

        # Storage for cross-model GP predictions
        self.gp_cross_predictions: Dict[Tuple[int, int], Dict[str, float]] = {}

        # Process each model sequentially
        for model_idx, current_model in enumerate(self.config.target_models):
            if self.budget_used >= self.config.total_llm_budget:
                print(f"\nBudget exhausted at model {model_idx}")
                break

            print(f"\n{'='*60}")
            print(f"MODEL PHASE {model_idx + 1}/{len(self.config.target_models)}")
            print(f"Model: {current_model}")
            print(f"{'='*60}")

            # Pre-load the model ONCE
            self._preload_model(current_model)

            # Run iterations with this model
            for local_iter in range(iterations_per_model):
                global_iter = model_idx * iterations_per_model + local_iter

                if self.budget_used >= self.config.total_llm_budget:
                    print(f"\nBudget exhausted at iteration {global_iter}")
                    break

                print(f"\n--- Iteration {local_iter + 1}/{iterations_per_model} (global: {global_iter + 1}) ---")
                print(f"Budget: {self.budget_used:,}/{self.config.total_llm_budget:,}")
                print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")

                # Phase 2: OPRO with current model as meta-model
                new_instructions = self._run_phase2_opro_single_model(current_model, verbose)

                if not new_instructions:
                    print("No new instructions generated")
                    continue

                # Phase 3: GP screening (predicts all models)
                top_candidates = self._run_phase3_gp_screening(new_instructions, verbose)

                # Phase 4: Evaluate ONLY on current model
                self._run_phase4_evaluation_single_model(top_candidates, current_model, verbose)

                # Phase 5: Retrain GP with new data
                self._run_phase5_gp_retrain(verbose)

                # Save iteration results
                self._save_iteration_results(global_iter)

        # Final verification phase - evaluate top candidates on ALL models
        if self.config.final_verification_top_k > 0:
            self._run_final_verification(verbose)

        # Final results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE (Model-Sequential)")
        print("=" * 60)
        print(f"Best aggregated accuracy: {self.best_aggregated_accuracy:.2%}")
        print("\nPer-model accuracies:")
        for model, acc in self.best_per_model_accuracies.items():
            print(f"  {model}: {acc:.2%}")
        print(f"\nBest instruction:\n{self.best_instruction}")
        if self.best_exemplar:
            print(f"\nBest exemplar:\n{self.best_exemplar[:500]}...")

        return (
            self.best_instruction,
            self.best_exemplar,
            self.best_aggregated_accuracy,
            self.best_per_model_accuracies,
        )

    def _preload_model(self, model_name: str):
        """Pre-load a model to minimize switching during iteration."""
        if self.evaluator_pool.single_gpu_manager is not None:
            print(f"Pre-loading model: {model_name}")
            self.evaluator_pool.single_gpu_manager.load_model(
                model_name,
                self.config.task_max_tokens,
            )

    def _run_phase2_opro_single_model(
        self,
        current_model: str,
        verbose: bool,
    ) -> List[str]:
        """
        OPRO generation using current model as meta-model.

        Uses aggregated scores (combining actual evaluations + GP predictions).
        """
        print("\n--- Phase 2: OPRO Instruction Generation ---")

        # Initialize OPRO generator for current model if needed
        self._init_opro_generator_for_model(current_model)

        # Get top-k instructions by aggregated accuracy (actual + GP predicted)
        scored_instructions = self._get_scored_instructions_with_gp()

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

    def _init_opro_generator_for_model(self, model_name: str):
        """Initialize or reinitialize OPRO generator for a specific model."""
        # In single-GPU mode, use the current model as meta-model
        if self.config.single_gpu and self.evaluator_pool.single_gpu_manager is not None:
            print(f"Using {model_name} as meta-model")
            meta_llm = SingleGPUMetaLLM(
                model_name=model_name,
                manager=self.evaluator_pool.single_gpu_manager,
            )
            self.opro_generator = OPROInstructionGenerator(
                meta_llm=meta_llm,
                num_candidates=self.config.opro_candidates_per_iter,
                temperature=self.config.meta_temperature,
                max_tokens=self.config.meta_max_tokens,
            )
        elif self.opro_generator is None:
            # Fallback to standard initialization
            self._init_opro_generator()

    def _get_scored_instructions_with_gp(self) -> List:
        """
        Get scored instructions combining actual evaluations and GP predictions.

        For each instruction:
        - Use actual scores where available
        - Use GP predictions for models not yet evaluated
        - Aggregate using configured strategy
        """
        from hybrid_opro_hbbops.config import ScoredInstruction

        # Collect actual evaluations per instruction
        inst_scores: Dict[int, Dict[str, List[float]]] = {}

        for point in self.design_data:
            inst_id = point.instruction_id
            if inst_id not in inst_scores:
                inst_scores[inst_id] = {m: [] for m in self.config.target_models}

            for model, error in point.model_error_rates.items():
                if model in inst_scores[inst_id]:
                    inst_scores[inst_id][model].append(1.0 - error)

        # Compute aggregated scores
        scored_instructions = []
        for inst_id, model_accs in inst_scores.items():
            # Average accuracy per model
            avg_accs = {}
            for model, accs in model_accs.items():
                if accs:
                    avg_accs[model] = np.mean(accs)

            # If we have data for all models, use actual aggregation
            if len(avg_accs) == len(self.config.target_models):
                agg_acc = aggregate_accuracies(
                    avg_accs,
                    self.config.aggregation,
                    self.config.softmin_temperature,
                )
            elif avg_accs:
                # Use average of available models
                agg_acc = np.mean(list(avg_accs.values()))
            else:
                continue

            scored_instructions.append(
                ScoredInstruction(
                    instruction=self.instructions[inst_id],
                    instruction_id=inst_id,
                    best_accuracy=agg_acc,
                    best_exemplar_id=0,
                )
            )

        # Sort by accuracy and take top-k
        scored_instructions.sort(key=lambda x: x.best_accuracy, reverse=True)
        return scored_instructions[:self.config.opro_keep_top_k]

    def _run_phase4_evaluation_single_model(
        self,
        candidates: List[MultiModelPromptCandidate],
        current_model: str,
        verbose: bool,
    ):
        """
        Evaluate candidates on a SINGLE model only.

        Store results and update GP cross-predictions for other models.
        """
        print(f"\n--- Phase 4: Single-Model Evaluation ({current_model.split('/')[-1]}) ---")

        if not candidates:
            print("No candidates to evaluate")
            return

        fidelity = min(100, self.nvalid)

        for i, candidate in enumerate(candidates):
            # Build prompts for current model
            prompts = []
            answers = []
            for d in self.validation_data[:fidelity]:
                prompt = self.evaluator_pool.format_prompt(
                    d["question"], candidate.instruction, candidate.exemplar
                )
                prompts.append(prompt)
                answers.append(d["answer"])

            # Evaluate using single-GPU manager (model already loaded)
            from multi_model_optimizer.evaluator_pool import extract_answer, compare_answers

            client = self.evaluator_pool.single_gpu_manager.load_model(
                current_model, self.config.task_max_tokens
            )

            responses = client.generate_batch(
                prompts,
                max_new_tokens=self.config.task_max_tokens,
                temperature=0.0,
            )

            # Count correct
            num_correct = 0
            for response, expected in zip(responses, answers):
                extracted = extract_answer(response)
                if compare_answers(extracted, expected):
                    num_correct += 1

            actual_accuracy = num_correct / len(prompts) if prompts else 0.0
            actual_error = 1.0 - actual_accuracy

            self.budget_used += fidelity

            # Store cross-prediction
            key = (candidate.instruction_id, candidate.exemplar_id)
            if key not in self.gp_cross_predictions:
                self.gp_cross_predictions[key] = {}
            self.gp_cross_predictions[key][current_model] = actual_accuracy

            # Get GP predictions for other models (if GP is trained)
            predicted_accs = {current_model: actual_accuracy}
            if self.gp_trainer.gp_params is not None:
                inst_emb = candidate.instruction_embedding.reshape(1, -1)
                ex_emb = candidate.exemplar_embedding.reshape(1, -1)

                try:
                    aggregated_errors, per_model_errors = self.gp_trainer.predict_aggregated(
                        inst_emb,
                        ex_emb,
                        self.config.aggregation,
                        self.config.softmin_temperature,
                    )
                    for model in self.config.target_models:
                        if model != current_model:
                            predicted_accs[model] = 1.0 - per_model_errors[model][0]
                            # Store GP prediction
                            if model not in self.gp_cross_predictions[key]:
                                self.gp_cross_predictions[key][model] = predicted_accs[model]
                except Exception as e:
                    print(f"  GP prediction failed: {e}")

            # Compute aggregated accuracy (actual + predicted)
            agg_acc = aggregate_accuracies(
                predicted_accs,
                self.config.aggregation,
                self.config.softmin_temperature,
            )

            if verbose:
                print(f"\nCandidate {i+1}/{len(candidates)}:")
                print(f"  {current_model.split('/')[-1]}: {actual_accuracy:.2%} (actual)")
                print(f"  Aggregated (with GP): {agg_acc:.2%}")

            # Update best if better
            if agg_acc > self.best_aggregated_accuracy:
                self.best_aggregated_accuracy = agg_acc
                self.best_instruction = candidate.instruction
                self.best_instruction_id = candidate.instruction_id
                self.best_exemplar = candidate.exemplar
                self.best_exemplar_id = candidate.exemplar_id
                self.best_per_model_accuracies = predicted_accs.copy()
                print(f"  *** New best (GP-augmented)! ***")

            # Add to design data (only with actual evaluation for current model)
            point = MultiModelDesignPoint(
                instruction_id=candidate.instruction_id,
                exemplar_id=candidate.exemplar_id,
                instruction_embedding=candidate.instruction_embedding,
                exemplar_embedding=candidate.exemplar_embedding,
                model_error_rates={current_model: actual_error},
                aggregated_error=actual_error,  # Single model for now
                fidelity=fidelity,
            )
            self.design_data.append(point)

    def _run_final_verification(self, verbose: bool):
        """
        Verify top-K candidates on ALL models.

        This is the only place where we need to switch between all models,
        but only for a small number of candidates.
        """
        print("\n" + "=" * 60)
        print("FINAL VERIFICATION PHASE")
        print("=" * 60)

        if not self.gp_cross_predictions:
            print("No candidates to verify")
            return

        # Select top-K by aggregated GP-augmented accuracy
        candidates_by_score = []

        for (inst_id, ex_id), model_accs in self.gp_cross_predictions.items():
            if not model_accs:
                continue

            # Fill in missing models with GP predictions if possible
            if len(model_accs) < len(self.config.target_models) and self.gp_trainer.gp_params is not None:
                try:
                    inst_emb = self.instruction_embeddings[inst_id].reshape(1, -1)
                    ex_emb = self.exemplar_embeddings[ex_id].reshape(1, -1)
                    _, per_model_errors = self.gp_trainer.predict_aggregated(
                        inst_emb, ex_emb,
                        self.config.aggregation,
                        self.config.softmin_temperature,
                    )
                    for model in self.config.target_models:
                        if model not in model_accs:
                            model_accs[model] = 1.0 - per_model_errors[model][0]
                except:
                    pass

            # Compute aggregated accuracy
            if model_accs:
                agg_acc = aggregate_accuracies(
                    model_accs,
                    self.config.aggregation,
                    self.config.softmin_temperature,
                )
                candidates_by_score.append((inst_id, ex_id, agg_acc, model_accs))

        if not candidates_by_score:
            print("No candidates with scores")
            return

        candidates_by_score.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates_by_score[:self.config.final_verification_top_k]

        print(f"Verifying top {len(top_candidates)} candidates on all models")

        for inst_id, ex_id, predicted_agg, _ in top_candidates:
            instruction = self.instructions[inst_id]
            exemplar = self.exemplars[ex_id]

            if verbose:
                print(f"\nCandidate (inst={inst_id}, ex={ex_id}):")
                print(f"  Predicted aggregated: {predicted_agg:.2%}")

            # Full evaluation on all models
            fidelity = min(200, self.nvalid)  # Higher fidelity for final verification

            actual_errors = self.evaluator_pool.evaluate_prompt_all_models(
                instruction=instruction,
                exemplar=exemplar,
                validation_data=self.validation_data,
                fidelity=fidelity,
            )

            self.budget_used += fidelity * len(self.config.target_models)

            actual_accs = {m: 1.0 - e for m, e in actual_errors.items()}
            actual_agg = aggregate_accuracies(
                actual_accs,
                self.config.aggregation,
                self.config.softmin_temperature,
            )

            if verbose:
                print(f"  Actual aggregated: {actual_agg:.2%}")
                for m, acc in actual_accs.items():
                    print(f"    {m.split('/')[-1]}: {acc:.2%}")

            # Update cross-predictions with actual values
            self.gp_cross_predictions[(inst_id, ex_id)] = actual_accs

            # Update best if better
            if actual_agg > self.best_aggregated_accuracy:
                self.best_aggregated_accuracy = actual_agg
                self.best_instruction = instruction
                self.best_instruction_id = inst_id
                self.best_exemplar = exemplar
                self.best_exemplar_id = ex_id
                self.best_per_model_accuracies = actual_accs
                print(f"  *** New best! ***")

    def save_final_results(self):
        """Save final results to file."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save best prompt
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
            },
            "timestamp": datetime.now().isoformat(),
        }

        output_path = os.path.join(self.config.output_dir, "final_results.json")
        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=2)

        print(f"Results saved to: {output_path}")
