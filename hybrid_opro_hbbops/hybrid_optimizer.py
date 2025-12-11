"""
Hybrid OPRO + HbBoPs optimizer.

Combines OPRO instruction generation with HbBoPs multi-fidelity GP screening.
"""
import sys
import re
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import HybridConfig, ScoredInstruction, DesignPoint, PromptCandidate
from .exemplar_sampler import ExemplarSampler
from .opro_adapter import OPROInstructionGenerator
from .sequential_tester import SequentialTester, Decision

# Reuse existing implementations
from hbbops_improved_2.hbbops import HbBoPs, Prompt, PromptEncoder
from generative_hbbops.gp_model import GPTrainer
from src.llm_client import create_llm_client


# Simple number pattern for answer extraction
NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'


def extract_answer(text: str) -> Optional[str]:
    """Extract last number from model output."""
    if not text:
        return None
    numbers = re.findall(NUMBER_PATTERN, text)
    return numbers[-1] if numbers else None


def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """Compare two numbers with tolerance."""
    if predicted == ground_truth:
        return True
    try:
        pred_clean = predicted.replace(',', '')
        gt_clean = ground_truth.replace(',', '')
        return abs(float(pred_clean) - float(gt_clean)) <= tolerance
    except (ValueError, TypeError):
        return False


class HybridOPROHbBoPs:
    """
    Hybrid OPRO + HbBoPs optimizer.

    Algorithm:
        Phase 1: Run full Hyperband on initial instruction × exemplar grid
        Phase 2: Extract top 20 prompts → unique instructions → OPRO generates 8 new
        Phase 3: GP screening (8 instructions × 25 dynamic exemplars = 200 candidates)
        Phase 4: Full evaluation of top 10 by GP prediction
        Phase 5: Retrain GP on ALL accumulated high-fidelity data
        Iterate phases 2-5 until budget exhausted
    """

    def __init__(
        self,
        config: HybridConfig,
        validation_data: List[Dict],
        gsm8k_train_data: List[Dict],
    ):
        """
        Args:
            config: HybridConfig with all parameters
            validation_data: List of {'question': str, 'answer': str} for evaluation
            gsm8k_train_data: List of {'question': str, 'answer': str} for exemplar sampling
        """
        self.config = config
        self.validation_data = validation_data.copy()
        random.seed(config.seed)
        random.shuffle(self.validation_data)
        self.nvalid = len(validation_data)
        self.gsm8k_train_data = gsm8k_train_data

        # Device
        self.device = self._get_device(config.device)
        print(f"Using device: {self.device}")

        # Components
        print("Initializing encoder...")
        self.encoder = PromptEncoder(config.encoder_name)

        print("Initializing GP trainer...")
        self.gp_trainer = GPTrainer(
            latent_dim=config.gp_latent_dim,
            train_epochs=config.gp_train_epochs,
            lr=config.gp_lr,
            patience=config.gp_patience,
            device=self.device,
            use_leaky_relu=True,
            leaky_relu_slope=0.01,
        )

        print("Initializing exemplar sampler...")
        self.exemplar_sampler = ExemplarSampler(gsm8k_train_data, config.seed)

        # Initialize LLM clients
        print(f"Initializing task LLM: {config.task_model}...")
        self.task_llm = create_llm_client(config.task_model, config.task_backend)

        # Reuse task_llm if same model (saves GPU memory)
        if config.meta_model == config.task_model:
            print(f"Reusing task LLM as meta LLM (same model)")
            self.meta_llm = self.task_llm
        else:
            print(f"Initializing meta LLM: {config.meta_model}...")
            self.meta_llm = create_llm_client(config.meta_model, "vllm")

        # OPRO adapter
        self.opro_generator = OPROInstructionGenerator(
            meta_llm=self.meta_llm,
            num_candidates=config.opro_candidates_per_iter,
            temperature=config.meta_temperature,
            max_tokens=config.meta_max_tokens,
        )

        # State: instruction and exemplar registries
        self.all_instructions: Dict[int, str] = {}
        self.all_exemplars: Dict[int, str] = {}
        self.instruction_embeddings: Dict[int, np.ndarray] = {}
        self.exemplar_embeddings: Dict[int, np.ndarray] = {}

        # Accumulated design data (ALL evaluations)
        self.design_data: List[DesignPoint] = []

        # Evaluation cache: (instruction_id, exemplar_id, fidelity) -> error_rate
        self.evaluation_cache: Dict[Tuple[int, int, int], float] = {}

        # Top prompts tracking for OPRO
        self.top_prompts: List[ScoredInstruction] = []

        # Counters
        self.next_instruction_id = 0
        self.next_exemplar_id = 0
        self.iteration = 0

        # Budget tracking
        self.budget_used = 0

        # Best result
        self.best_prompt: Optional[Prompt] = None
        self.best_accuracy: float = 0.0

        # GP state
        self._gp_trained = False

    def _get_device(self, device: str) -> torch.device:
        """Determine device to use."""
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def run(self, verbose: bool = True) -> Tuple[Prompt, float]:
        """
        Run the hybrid optimization algorithm.

        Returns:
            (best_prompt, best_accuracy)
        """
        if verbose:
            print("=" * 70)
            print("Starting Hybrid OPRO + HbBoPs Optimization")
            print(f"Budget: {self.config.total_llm_budget} evaluations")
            print("=" * 70)

        # ========== PHASE 1: Initial Hyperband or Load from File ==========
        if self.config.skip_phase1_hbbops:
            if verbose:
                print("\n[PHASE 1] Loading pre-computed results from file...")
            self._run_phase1_from_file(verbose)
        else:
            if verbose:
                print("\n[PHASE 1] Running initial Hyperband...")
            self._run_phase1_hyperband(verbose)

        if verbose:
            print(f"\nPhase 1 complete. Budget used: {self.budget_used}/{self.config.total_llm_budget}")
            print(f"Top prompts collected: {len(self.top_prompts)}")

        # ========== ITERATIVE PHASES 2-5 ==========
        while self.budget_used < self.config.total_llm_budget:
            # Check if we have enough budget for at least one full evaluation
            remaining_budget = self.config.total_llm_budget - self.budget_used
            if remaining_budget < len(self.validation_data):
                if verbose:
                    print(f"\n{'=' * 70}")
                    print(f"Budget nearly exhausted: {self.budget_used}/{self.config.total_llm_budget}")
                    print(f"Remaining ({remaining_budget}) < validation set size ({len(self.validation_data)})")
                    print("Stopping optimization and returning best prompt.")
                    print("=" * 70)
                break

            self.iteration += 1

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"ITERATION {self.iteration}")
                print(f"Budget: {self.budget_used}/{self.config.total_llm_budget}")
                print("=" * 70)

            # ========== PHASE 2: OPRO instruction generation ==========
            if verbose:
                print("\n[PHASE 2] OPRO instruction generation...")

            new_instructions = self._run_phase2_opro(verbose)

            if not new_instructions:
                if verbose:
                    print("No new instructions generated. Stopping.")
                break

            # ========== PHASE 3: GP screening ==========
            if verbose:
                print(f"\n[PHASE 3] GP screening {len(new_instructions)} × {self.config.num_dynamic_exemplars} candidates...")

            top_candidates = self._run_phase3_gp_screening(new_instructions, verbose)

            # ========== PHASE 4: Full evaluation ==========
            if verbose:
                print(f"\n[PHASE 4] Full evaluation of top {len(top_candidates)} candidates...")

            self._run_phase4_evaluation(top_candidates, verbose)

            # ========== PHASE 5: Retrain GP ==========
            if verbose:
                print("\n[PHASE 5] Retraining GP on accumulated data...")

            self._run_phase5_gp_retrain(verbose)

            # Update best
            self._update_best_prompt()

            if verbose:
                print(f"\nCurrent best: accuracy={self.best_accuracy:.4f}")

        if verbose:
            print("\n" + "=" * 70)
            print("Optimization complete!")
            print(f"Best accuracy: {self.best_accuracy:.4f}")
            print(f"Total evaluations: {self.budget_used}")
            print(f"Iterations: {self.iteration}")
            print("=" * 70)

        return self.best_prompt, self.best_accuracy

    # ==================== PHASE IMPLEMENTATIONS ====================

    def _run_phase1_hyperband(self, verbose: bool) -> None:
        """Phase 1: Run full Hyperband on initial grid."""
        # Load initial instructions and exemplars
        instructions = self._load_instructions(self.config.initial_instructions_path)
        exemplars = self._load_exemplars(self.config.initial_exemplars_path)

        if verbose:
            print(f"  Loaded {len(instructions)} instructions, {len(exemplars)} exemplars")

        # Register them
        for inst in instructions:
            self._register_instruction(inst)
        for ex in exemplars:
            self._register_exemplar(ex)

        # Create HbBoPs evaluator callback
        evaluator = self._create_evaluator()

        # Create HbBoPs instance
        hbbops = HbBoPs(
            instructions=instructions,
            exemplars=exemplars,
            validation_data=self.validation_data,
            llm_evaluator=evaluator,
            encoder_name=self.config.encoder_name,
            bmin=self.config.bmin,
            eta=self.config.eta,
            device=str(self.device),
            seed=self.config.seed,
        )

        # Run Hyperband
        best_prompt, best_error = hbbops.run_hyperband(verbose=verbose)

        if verbose:
            print(f"\n  HbBoPs best: accuracy={1-best_error:.4f}")

        # Copy design data
        for p_idx, inst_emb, ex_emb, val_error, fidelity in hbbops.design_data:
            prompt = hbbops.prompts[p_idx]
            self.design_data.append(
                DesignPoint(
                    instruction_id=prompt.instruction_id,
                    exemplar_id=prompt.exemplar_id,
                    instruction_embedding=inst_emb,
                    exemplar_embedding=ex_emb,
                    error_rate=val_error,
                    fidelity=fidelity,
                )
            )

        # Copy evaluation cache
        for (inst_id, ex_id, fid), error in hbbops.evaluation_cache.items():
            self.evaluation_cache[(inst_id, ex_id, fid)] = error

        # Copy embeddings
        self.instruction_embeddings.update(hbbops.instruction_embeddings)
        self.exemplar_embeddings.update(hbbops.exemplar_embeddings)

        # Calculate budget used
        budget = sum(fid for (_, _, fid) in hbbops.evaluation_cache.keys())
        self.budget_used += budget

        # Extract top prompts for OPRO
        self._extract_top_prompts_from_hbbops(hbbops)

    def _run_phase1_from_file(self, verbose: bool) -> None:
        """Phase 1 alternative: Load pre-computed results from JSONL file."""
        # Load initial instructions and exemplars (for text lookup)
        instructions = self._load_instructions(self.config.initial_instructions_path)
        exemplars = self._load_exemplars(self.config.initial_exemplars_path)

        if verbose:
            print(f"  Loaded {len(instructions)} instructions, {len(exemplars)} exemplars")

        # Register them and compute embeddings
        for inst in instructions:
            self._register_instruction(inst)
        for ex in exemplars:
            self._register_exemplar(ex)

        # Load pre-computed results from JSONL
        results = []
        with open(self.config.phase1_results_path, "r") as f:
            for line in f:
                results.append(json.loads(line))

        if verbose:
            print(f"  Loaded {len(results)} pre-computed evaluations from {self.config.phase1_results_path}")

        # Sort by error rate (ascending = best first)
        results.sort(key=lambda x: x["error_rate"])

        # Add ALL results to design_data for GP training
        for r in results:
            inst_id = r["instruction_id"]
            ex_id = r["exemplar_id"]
            error_rate = r["error_rate"]

            self.design_data.append(
                DesignPoint(
                    instruction_id=inst_id,
                    exemplar_id=ex_id,
                    instruction_embedding=self.instruction_embeddings[inst_id],
                    exemplar_embedding=self.exemplar_embeddings[ex_id],
                    error_rate=error_rate,
                    fidelity=self.nvalid,  # Full fidelity
                )
            )

            # Add to evaluation cache
            self.evaluation_cache[(inst_id, ex_id, self.nvalid)] = error_rate

        if verbose:
            print(f"  Added {len(results)} points to GP design data")

        # Extract top k prompts for OPRO context
        top_k = self.config.opro_keep_top_k
        for r in results[:top_k]:
            inst_id = r["instruction_id"]
            ex_id = r["exemplar_id"]
            accuracy = 1 - r["error_rate"]

            self.top_prompts.append(
                ScoredInstruction(
                    instruction=self.all_instructions[inst_id],
                    instruction_id=inst_id,
                    best_accuracy=accuracy,
                    best_exemplar_id=ex_id,
                    embedding=self.instruction_embeddings[inst_id],
                )
            )

        if verbose:
            print(f"  Top {len(self.top_prompts)} prompts extracted for OPRO")
            print(f"  Best accuracy from file: {1 - results[0]['error_rate']:.4f}")

        # Train GP on all loaded data
        if verbose:
            print("  Training GP on pre-computed data...")

        inst_embs = np.array([dp.instruction_embedding for dp in self.design_data])
        ex_embs = np.array([dp.exemplar_embedding for dp in self.design_data])
        errors = np.array([dp.error_rate for dp in self.design_data])

        try:
            self.gp_trainer.train(inst_embs, ex_embs, errors, verbose=False)
            self._gp_trained = True
            if verbose:
                print(f"  GP trained on {len(self.design_data)} observations")
        except Exception as e:
            if verbose:
                print(f"  GP training failed: {e}")

        # Update best prompt
        best_r = results[0]
        self.best_accuracy = 1 - best_r["error_rate"]
        self.best_prompt = Prompt(
            instruction=self.all_instructions[best_r["instruction_id"]],
            exemplar=self.all_exemplars[best_r["exemplar_id"]],
            instruction_id=best_r["instruction_id"],
            exemplar_id=best_r["exemplar_id"],
        )

        # No budget used (pre-computed)
        if verbose:
            print("  No LLM budget used (pre-computed results)")

    def _run_phase2_opro(self, verbose: bool) -> List[str]:
        """Phase 2: Use OPRO to generate new instructions."""
        # Get unique instructions from top prompts
        unique_instructions = self._get_unique_top_instructions()

        if verbose:
            print(f"  Top {len(unique_instructions)} unique instructions for OPRO context")

        # Existing instruction texts
        existing = set(self.all_instructions.values())

        # Generate new instructions
        new_instructions = self.opro_generator.generate_candidates(
            scored_instructions=unique_instructions,
            existing_instructions=existing,
            verbose=verbose,
        )

        if verbose:
            print(f"  Generated {len(new_instructions)} new unique instructions")

        # Register new instructions
        for inst in new_instructions:
            self._register_instruction(inst)

        return new_instructions

    def _run_phase3_gp_screening(
        self,
        new_instructions: List[str],
        verbose: bool,
    ) -> List[PromptCandidate]:
        """Phase 3: GP-based screening of instruction × exemplar candidates."""
        # Sample new exemplars dynamically from GSM8K train
        new_exemplars = self.exemplar_sampler.sample(
            n=self.config.num_dynamic_exemplars,
            k=self.config.exemplars_per_sample,
        )

        if verbose:
            print(f"  Sampled {len(new_exemplars)} new exemplars from GSM8K train")

        # Register new exemplars
        new_ex_ids = []
        for ex in new_exemplars:
            ex_id = self._register_exemplar(ex)
            new_ex_ids.append(ex_id)

        # Get instruction IDs for new instructions
        new_inst_ids = []
        for inst in new_instructions:
            for inst_id, text in self.all_instructions.items():
                if text == inst:
                    new_inst_ids.append(inst_id)
                    break

        # Create all candidates (8 instructions × 25 exemplars = 200)
        candidates: List[PromptCandidate] = []
        for inst_id in new_inst_ids:
            inst_emb = self.instruction_embeddings[inst_id]
            for ex_id in new_ex_ids:
                ex_emb = self.exemplar_embeddings[ex_id]
                candidates.append(
                    PromptCandidate(
                        instruction=self.all_instructions[inst_id],
                        instruction_id=inst_id,
                        instruction_embedding=inst_emb,
                        exemplar=self.all_exemplars[ex_id],
                        exemplar_id=ex_id,
                        exemplar_embedding=ex_emb,
                    )
                )

        if verbose:
            print(f"  Created {len(candidates)} candidates for GP screening")

        # Use GP to predict accuracy (no LLM calls!)
        if self._gp_trained:
            # Prepare batch for GP prediction
            inst_embs = np.array([c.instruction_embedding for c in candidates])
            ex_embs = np.array([c.exemplar_embedding for c in candidates])

            # Get predictions
            inst_tensor = torch.tensor(inst_embs, dtype=torch.float32, device=self.device)
            ex_tensor = torch.tensor(ex_embs, dtype=torch.float32, device=self.device)

            means, stds = self.gp_trainer.predict(inst_tensor, ex_tensor)
            means = means.cpu().numpy() if hasattr(means, 'cpu') else np.array([means])
            stds = stds.cpu().numpy() if hasattr(stds, 'cpu') else np.array([stds])

            for i, candidate in enumerate(candidates):
                pred_error = float(means[i]) if len(means.shape) > 0 else float(means)
                candidate.gp_predicted_accuracy = 1 - pred_error

            # Sort by predicted accuracy (descending)
            candidates.sort(key=lambda c: c.gp_predicted_accuracy, reverse=True)

            if verbose:
                print("  Top 5 by GP prediction:")
                for c in candidates[:5]:
                    print(
                        f"    inst={c.instruction_id}, ex={c.exemplar_id}, "
                        f"pred_acc={c.gp_predicted_accuracy:.4f}"
                    )
        else:
            # No GP yet - random selection
            random.shuffle(candidates)
            if verbose:
                print("  GP not trained yet, using random selection")

        # Return top k
        return candidates[: self.config.gp_top_k]

    def _run_phase4_evaluation(
        self,
        candidates: List[PromptCandidate],
        verbose: bool,
    ) -> None:
        """Phase 4: Sequential evaluation with Hoeffding early stopping.

        Uses exponential sample sizes (10 * 2^k) with dynamic stopping:
        - DROP: Upper bound < best_accuracy (can't beat champion)
        - PROMOTE: Lower bound > best_accuracy (definitely better)
        - FULL: Complete evaluation when sequential test is inconclusive
        """
        if not self.config.sequential_testing:
            # Fallback to original full-fidelity evaluation
            return self._run_phase4_full_evaluation(candidates, verbose)

        tester = SequentialTester(
            confidence=self.config.sequential_confidence,
            min_samples=self.config.sequential_min_samples,
            min_promote_samples=self.config.sequential_min_promote_samples,
        )

        # Get current best accuracy (champion)
        best_accuracy = self.top_prompts[0].best_accuracy if self.top_prompts else 0.0

        for i, candidate in enumerate(candidates):
            if verbose:
                gp_pred = candidate.gp_predicted_accuracy
                gp_str = f"gp={gp_pred:.3f}" if gp_pred is not None else ""
                print(f"  Candidate {i+1}/{len(candidates)}: "
                      f"inst={candidate.instruction_id}, ex={candidate.exemplar_id} {gp_str}")

            # Create Prompt object
            prompt = Prompt(
                instruction=candidate.instruction,
                exemplar=candidate.exemplar,
                instruction_id=candidate.instruction_id,
                exemplar_id=candidate.exemplar_id,
            )

            # Sequential testing loop
            n_samples = 0
            successes = 0
            decision = Decision.CONTINUE

            while decision == Decision.CONTINUE:
                # Get next step (10 * 2^k)
                next_n = tester.get_next_step(n_samples, self.nvalid)

                # Check budget
                additional_samples = next_n - n_samples
                if self.budget_used + additional_samples > self.config.total_llm_budget:
                    if verbose:
                        print(f"    Budget exhausted at n={n_samples}")
                    break

                # Evaluate additional samples
                new_successes = self._evaluate_samples(
                    prompt, self.validation_data[n_samples:next_n]
                )
                successes += new_successes
                n_samples = next_n
                self.budget_used += additional_samples

                # Make decision
                decision, accuracy, lb, ub = tester.decide(
                    successes, n_samples, best_accuracy
                )

                if verbose:
                    print(f"    n={n_samples}: acc={accuracy:.3f}, "
                          f"bounds=[{lb:.3f}, {ub:.3f}] → {decision.value}")

                # If full dataset reached, we're done
                if n_samples >= self.nvalid:
                    decision = Decision.CONTINUE  # Mark as full eval
                    break

            # Record final result
            final_accuracy = successes / n_samples if n_samples > 0 else 0.0
            final_error = 1 - final_accuracy

            candidate.actual_accuracy = final_accuracy
            candidate.samples_used = n_samples
            candidate.decision = decision.value if decision != Decision.CONTINUE else "full"

            # Add to design data (with actual fidelity used)
            self.design_data.append(
                DesignPoint(
                    instruction_id=candidate.instruction_id,
                    exemplar_id=candidate.exemplar_id,
                    instruction_embedding=candidate.instruction_embedding,
                    exemplar_embedding=candidate.exemplar_embedding,
                    error_rate=final_error,
                    fidelity=n_samples,
                )
            )

            # Update top prompts only for PROMOTE or full evaluation
            if decision == Decision.PROMOTE or n_samples >= self.nvalid:
                self._update_top_prompts_from_candidate(candidate)
                # Update champion for next candidates
                if self.top_prompts:
                    best_accuracy = self.top_prompts[0].best_accuracy

    def _run_phase4_full_evaluation(
        self,
        candidates: List[PromptCandidate],
        verbose: bool,
    ) -> None:
        """Original Phase 4: Full-fidelity evaluation (fallback)."""
        for i, candidate in enumerate(candidates):
            # Check budget
            remaining = self.config.total_llm_budget - self.budget_used
            if remaining < self.nvalid:
                if verbose:
                    print(f"  Budget exhausted after {i} evaluations")
                break

            # Create Prompt object
            prompt = Prompt(
                instruction=candidate.instruction,
                exemplar=candidate.exemplar,
                instruction_id=candidate.instruction_id,
                exemplar_id=candidate.exemplar_id,
            )

            # Full-fidelity evaluation
            error_rate = self._evaluate_prompt(prompt, self.nvalid)
            candidate.actual_accuracy = 1 - error_rate
            candidate.samples_used = self.nvalid
            candidate.decision = "full"

            # Add to design data
            self.design_data.append(
                DesignPoint(
                    instruction_id=candidate.instruction_id,
                    exemplar_id=candidate.exemplar_id,
                    instruction_embedding=candidate.instruction_embedding,
                    exemplar_embedding=candidate.exemplar_embedding,
                    error_rate=error_rate,
                    fidelity=self.nvalid,
                )
            )

            if verbose:
                gp_pred = candidate.gp_predicted_accuracy
                gp_str = f"gp_pred={gp_pred:.4f}, " if gp_pred is not None else ""
                print(
                    f"  Evaluated {i+1}/{len(candidates)}: "
                    f"inst={candidate.instruction_id}, ex={candidate.exemplar_id}, "
                    f"{gp_str}actual_acc={candidate.actual_accuracy:.4f}"
                )

            # Update top prompts
            self._update_top_prompts_from_candidate(candidate)

    def _evaluate_samples(self, prompt: Prompt, data: List[Dict]) -> int:
        """Evaluate prompt on specific samples, return number of successes.

        Args:
            prompt: The prompt to evaluate
            data: List of data items to evaluate on

        Returns:
            Number of correct answers (successes)
        """
        if not data:
            return 0

        prompts = [
            f"Question: {ex['question']}\n\n{str(prompt)}\n\nAnswer:"
            for ex in data
        ]

        try:
            responses = self.task_llm.generate_batch(
                prompts, max_tokens=self.config.task_max_tokens
            )
        except Exception as e:
            print(f"LLM error: {e}")
            return 0

        successes = 0
        for i, ex in enumerate(data):
            gold = re.findall(NUMBER_PATTERN, ex["answer"])
            gold = gold[-1] if gold else None

            pred = extract_answer(responses[i])

            if gold and pred and compare_numbers(pred, gold):
                successes += 1

        return successes

    def _run_phase5_gp_retrain(self, verbose: bool) -> None:
        """Phase 5: Retrain GP on ALL accumulated high-fidelity data."""
        # Filter to high-fidelity observations (at least 50% of full)
        min_fidelity = self.nvalid // 2
        high_fidelity_data = [
            dp for dp in self.design_data if dp.fidelity >= min_fidelity
        ]

        if len(high_fidelity_data) < 4:
            if verbose:
                print(f"  Not enough data for GP training ({len(high_fidelity_data)} points)")
            return

        # Prepare arrays
        inst_embs = np.array([dp.instruction_embedding for dp in high_fidelity_data])
        ex_embs = np.array([dp.exemplar_embedding for dp in high_fidelity_data])
        errors = np.array([dp.error_rate for dp in high_fidelity_data])

        # Check variance
        if np.std(errors) < 1e-6:
            if verbose:
                print("  No variance in error rates, skipping GP training")
            return

        # Train GP
        try:
            self.gp_trainer.train(inst_embs, ex_embs, errors, verbose=False)
            self._gp_trained = True
            if verbose:
                print(f"  GP trained on {len(high_fidelity_data)} high-fidelity observations")
        except Exception as e:
            if verbose:
                print(f"  GP training failed: {e}")

    # ==================== HELPER METHODS ====================

    def _register_instruction(self, text: str) -> int:
        """Register an instruction and compute its embedding."""
        inst_id = self.next_instruction_id
        self.next_instruction_id += 1
        self.all_instructions[inst_id] = text
        self.instruction_embeddings[inst_id] = self.encoder.encode(text)
        return inst_id

    def _register_exemplar(self, text: str) -> int:
        """Register an exemplar and compute its embedding."""
        ex_id = self.next_exemplar_id
        self.next_exemplar_id += 1
        self.all_exemplars[ex_id] = text
        self.exemplar_embeddings[ex_id] = self.encoder.encode(text)
        return ex_id

    def _create_evaluator(self):
        """Create evaluator callback for HbBoPs."""
        def evaluator(prompt: Prompt, validation_data: List[Dict]) -> float:
            return self._llm_evaluate(prompt, validation_data)
        return evaluator

    def _evaluate_prompt(self, prompt: Prompt, fidelity: int) -> float:
        """Evaluate a prompt with caching and budget tracking."""
        cache_key = (prompt.instruction_id, prompt.exemplar_id, fidelity)

        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # Actual LLM evaluation
        error_rate = self._llm_evaluate(prompt, self.validation_data[:fidelity])

        self.evaluation_cache[cache_key] = error_rate
        self.budget_used += fidelity

        return error_rate

    def _llm_evaluate(self, prompt: Prompt, data: List[Dict]) -> float:
        """Perform actual LLM evaluation."""
        prompts = [
            f"Question: {ex['question']}\n\n{str(prompt)}\n\nAnswer:"
            for ex in data
        ]

        try:
            responses = self.task_llm.generate_batch(
                prompts, max_tokens=self.config.task_max_tokens
            )
        except Exception as e:
            print(f"LLM error: {e}")
            return 1.0

        errors = 0
        for i, ex in enumerate(data):
            gold = re.findall(NUMBER_PATTERN, ex["answer"])
            gold = gold[-1] if gold else None

            pred = extract_answer(responses[i])

            if gold is None or pred is None or not compare_numbers(pred, gold):
                errors += 1

        return errors / len(data)

    def _extract_top_prompts_from_hbbops(self, hbbops: HbBoPs) -> None:
        """Extract top prompts from HbBoPs results."""
        # Get all full-fidelity evaluations
        full_fidelity_evals = []
        for (inst_id, ex_id, fid), error in hbbops.evaluation_cache.items():
            if fid == self.nvalid:
                full_fidelity_evals.append((inst_id, ex_id, 1 - error))

        # Sort by accuracy
        full_fidelity_evals.sort(key=lambda x: x[2], reverse=True)

        # Keep top k
        for inst_id, ex_id, accuracy in full_fidelity_evals[: self.config.opro_keep_top_k]:
            self.top_prompts.append(
                ScoredInstruction(
                    instruction=self.all_instructions[inst_id],
                    instruction_id=inst_id,
                    best_accuracy=accuracy,
                    best_exemplar_id=ex_id,
                    embedding=self.instruction_embeddings[inst_id],
                )
            )

    def _get_unique_top_instructions(self) -> List[ScoredInstruction]:
        """Get unique instructions from top prompts (max accuracy per instruction)."""
        inst_to_best: Dict[int, ScoredInstruction] = {}

        for sp in self.top_prompts:
            if (
                sp.instruction_id not in inst_to_best
                or sp.best_accuracy > inst_to_best[sp.instruction_id].best_accuracy
            ):
                inst_to_best[sp.instruction_id] = sp

        return list(inst_to_best.values())

    def _update_top_prompts_from_candidate(self, candidate: PromptCandidate) -> None:
        """Update top prompts with new evaluation result."""
        scored = ScoredInstruction(
            instruction=candidate.instruction,
            instruction_id=candidate.instruction_id,
            best_accuracy=candidate.actual_accuracy,
            best_exemplar_id=candidate.exemplar_id,
            embedding=candidate.instruction_embedding,
        )

        self.top_prompts.append(scored)
        self.top_prompts.sort(key=lambda x: x.best_accuracy, reverse=True)
        self.top_prompts = self.top_prompts[: self.config.opro_keep_top_k]

    def _update_best_prompt(self) -> None:
        """Update best prompt from top prompts."""
        if not self.top_prompts:
            return

        best = self.top_prompts[0]
        if best.best_accuracy > self.best_accuracy:
            self.best_accuracy = best.best_accuracy
            self.best_prompt = Prompt(
                instruction=best.instruction,
                exemplar=self.all_exemplars[best.best_exemplar_id],
                instruction_id=best.instruction_id,
                exemplar_id=best.best_exemplar_id,
            )

    def _load_instructions(self, path: str) -> List[str]:
        """Load instructions from file."""
        with open(path, "r") as f:
            return [
                re.sub(r"^\d+\.\s*", "", line.strip())
                for line in f
                if line.strip() and not line.startswith("#") and line[0].isdigit()
            ]

    def _load_exemplars(self, path: str) -> List[str]:
        """Load exemplars from file."""
        with open(path, "r") as f:
            content = f.read()

        exemplars = []
        for block in content.split("=" * 80):
            if not block.strip():
                continue
            lines = [l for l in block.split("\n") if not l.startswith("#")]
            examples, current_q = [], None
            for line in lines:
                line = line.strip()
                if line.startswith("Q:"):
                    current_q = line[2:].strip()
                elif line.startswith("A:") and current_q:
                    examples.append(f"Q: {current_q}\nA: {line[2:].strip()}")
                    current_q = None
            if examples:
                exemplars.append("\n\n".join(examples))
        return exemplars

    # ==================== PROPERTIES ====================

    @property
    def budget_remaining(self) -> int:
        """Remaining budget."""
        return max(0, self.config.total_llm_budget - self.budget_used)

    @property
    def num_instructions(self) -> int:
        """Number of unique instructions."""
        return len(self.all_instructions)

    @property
    def num_exemplars(self) -> int:
        """Number of unique exemplars."""
        return len(self.all_exemplars)

    @property
    def num_evaluations(self) -> int:
        """Number of cached evaluations."""
        return len(self.evaluation_cache)
