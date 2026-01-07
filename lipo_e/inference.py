"""Inference pipeline for LIPO-E.

Optimization loop:
1. Optimize joint latent (32D) using UCB/LogEI
2. Decode instruction → Vec2Text → text
3. Decode exemplars → hard select from pool
4. Build prompt, evaluate, add to GP, repeat
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import json
import os
from datetime import datetime
from botorch.optim import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound, qLogExpectedImprovement

from lipo_e.config import LIPOEConfig
from lipo_e.encoder import GTREncoder, StructureAwareVAE
from lipo_e.gp import GPWithEI
from lipo_e.training import QAPair, HbBoPsEvaluator

# Reuse working Vec2Text inverter from LIPO
from lipo.inference import Vec2TextInverter


@dataclass
class IterationRecord:
    """Record for single optimization iteration."""
    iteration: int
    instruction: str
    exemplar_ids: List[int]
    exemplar_texts: List[str]
    num_exemplars: int
    predicted_error: float
    actual_error: float
    improved: bool
    best_error_so_far: float
    cosine_similarity: float
    rejection_attempts: int


class LIPOEInference:
    """InvBO inference for joint instruction-exemplar optimization."""

    def __init__(
        self,
        vae: StructureAwareVAE,
        gp: GPWithEI,
        gtr_encoder: GTREncoder,
        qa_pool: List[QAPair],
        pool_embeddings: torch.Tensor,
        instructions: List[str],
        evaluator: HbBoPsEvaluator,
        config: LIPOEConfig,
    ):
        self.vae = vae
        self.gp = gp
        self.gtr_encoder = gtr_encoder
        self.qa_pool = qa_pool
        self.pool_embeddings = pool_embeddings
        self.instructions = instructions
        self.evaluator = evaluator
        self.config = config
        self.device = config.device

        # Vec2Text inverter (from lipo.inference)
        self.inverter = Vec2TextInverter(
            model_type=config.vec2text_model,
            beam_width=config.vec2text_beam,
            max_length=config.vec2text_max_length,
            device=config.device,
            num_steps=50,  # Number of correction steps
        )

        # Best results
        self.best_error = float("inf")
        self.best_instruction: Optional[str] = None
        self.best_exemplars: List[int] = []
        self.history: List[IterationRecord] = []

    def optimize_latent(
        self,
        iteration: int,
    ) -> torch.Tensor:
        """Optimize joint latent using acquisition function.

        Args:
            iteration: Current iteration (for adaptive beta)

        Returns:
            z_opt: Optimal latent (total_dim,)
        """
        bounds = self.gp.get_bounds(margin=self.config.latent_margin)

        # Adaptive UCB beta
        if self.config.ucb_beta_adaptive:
            progress = iteration / max(self.config.iterations, 1)
            ucb_beta = self.config.ucb_beta - progress * (
                self.config.ucb_beta - self.config.ucb_beta_final
            )
        else:
            ucb_beta = self.config.ucb_beta

        # Create acquisition function
        if self.config.acquisition_type == "ucb":
            acq_func = qUpperConfidenceBound(
                model=self.gp.gp_model,
                beta=ucb_beta,
            )
        else:
            acq_func = qLogExpectedImprovement(
                model=self.gp.gp_model,
                best_f=self.gp.best_f,
            )

        # Optimize
        z_opt, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
        )

        # Add noise for diversity
        if self.config.latent_noise_scale > 0:
            noise = torch.randn_like(z_opt) * self.config.latent_noise_scale
            z_opt = z_opt + noise
            # Clip to bounds
            z_opt = torch.clamp(z_opt, bounds[0], bounds[1])

        return z_opt.squeeze()

    def decode_and_generate(
        self,
        z_opt: torch.Tensor,
    ) -> Tuple[str, List[int], float]:
        """Decode optimal latent to instruction and exemplars.

        Args:
            z_opt: Optimal latent (total_dim,)

        Returns:
            instruction: Generated instruction text
            exemplar_ids: Selected pool indices
            cosine_sim: Round-trip cosine similarity
        """
        self.vae.eval()
        instruction_dim = self.config.instruction_latent_dim

        z_inst = z_opt[:instruction_dim].unsqueeze(0)
        z_ex = z_opt[instruction_dim:].unsqueeze(0)

        with torch.no_grad():
            # Decode instruction embedding
            inst_emb_recon = self.vae.instruction_decoder(z_inst)

            # Decode exemplar selection
            selection_probs, num_ex_logits, selected_indices = self.vae.exemplar_decoder(
                z_ex,
                self.pool_embeddings,
                hard=True,
            )

            # Get number of exemplars
            num_ex = num_ex_logits.argmax(dim=-1).item()
            num_ex = min(num_ex, self.config.num_slots)

        # Vec2Text inversion
        instruction = self.inverter.invert(inst_emb_recon)

        # Compute round-trip cosine similarity
        inst_emb_new = self.gtr_encoder.encode_single(instruction)
        cosine_sim = F.cosine_similarity(
            inst_emb_recon.squeeze(),
            inst_emb_new,
            dim=0,
        ).item()

        # Get exemplar indices
        exemplar_ids = selected_indices[0, :num_ex].tolist()

        return instruction, exemplar_ids, cosine_sim

    def run_iteration(
        self,
        iteration: int,
    ) -> IterationRecord:
        """Run single optimization iteration.

        Args:
            iteration: Iteration number

        Returns:
            record: Iteration results
        """
        # Optimize latent
        z_opt = self.optimize_latent(iteration)

        # Predict error at optimum
        z_opt_norm = self.gp._normalize_X(z_opt.unsqueeze(0))
        with torch.no_grad():
            pred = self.gp.likelihood(self.gp.gp_model(z_opt_norm))
            predicted_error = -(pred.mean.item() * self.gp.y_std + self.gp.y_mean)

        # Decode with rejection sampling
        best_instruction = None
        best_exemplars = []
        best_cosine = 0.0
        rejection_attempts = 0

        for attempt in range(self.config.max_rejection_attempts):
            instruction, exemplar_ids, cosine_sim = self.decode_and_generate(z_opt)

            if cosine_sim >= self.config.cosine_sim_threshold:
                best_instruction = instruction
                best_exemplars = exemplar_ids
                best_cosine = cosine_sim
                break

            if cosine_sim > best_cosine:
                best_instruction = instruction
                best_exemplars = exemplar_ids
                best_cosine = cosine_sim

            rejection_attempts += 1

            # Add more noise for next attempt
            noise = torch.randn_like(z_opt) * self.config.latent_noise_scale * 2
            z_opt = z_opt + noise

        if best_instruction is None:
            best_instruction = instruction
            best_exemplars = exemplar_ids
            best_cosine = cosine_sim

        # Evaluate
        qa_pairs = [self.qa_pool[i] for i in best_exemplars]
        actual_error = self.evaluator.evaluate(
            instruction=best_instruction,
            qa_pairs=qa_pairs,
            fidelity=len(self.evaluator.validation_data),
        )

        # Update best
        improved = actual_error < self.best_error
        if improved:
            self.best_error = actual_error
            self.best_instruction = best_instruction
            self.best_exemplars = best_exemplars

        # Add observation to GP
        # Note: clone().detach() to avoid inference tensor issues
        inst_emb = self.gtr_encoder.encode_single(best_instruction).clone().detach()
        ex_embs = torch.stack([
            self.pool_embeddings[i] for i in best_exemplars
        ]).clone().detach() if best_exemplars else torch.zeros(1, 768, device=self.device)
        ex_mask = torch.ones(len(best_exemplars), dtype=torch.bool, device=self.device) if best_exemplars else None

        with torch.no_grad():
            z_real = self.vae.encode_joint(
                inst_emb.unsqueeze(0),
                ex_embs.unsqueeze(0),
                ex_mask.unsqueeze(0) if ex_mask is not None else None,
            ).squeeze()

        self.gp.add_observation(
            z_real,
            torch.tensor(actual_error, device=self.device),
        )

        # Record
        record = IterationRecord(
            iteration=iteration,
            instruction=best_instruction,
            exemplar_ids=best_exemplars,
            exemplar_texts=[self.qa_pool[i].format() for i in best_exemplars],
            num_exemplars=len(best_exemplars),
            predicted_error=predicted_error,
            actual_error=actual_error,
            improved=improved,
            best_error_so_far=self.best_error,
            cosine_similarity=best_cosine,
            rejection_attempts=rejection_attempts,
        )
        self.history.append(record)

        return record

    def run(
        self,
        num_iterations: int,
        output_dir: str,
    ) -> List[IterationRecord]:
        """Run full optimization loop.

        Args:
            num_iterations: Number of iterations
            output_dir: Directory to save results

        Returns:
            history: All iteration records
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("LIPO-E Inference")
        print(f"{'=' * 60}")
        print(f"Iterations: {num_iterations}")
        print(f"Acquisition: {self.config.acquisition_type}")
        print(f"UCB beta: {self.config.ucb_beta} → {self.config.ucb_beta_final}")
        print(f"Pool size: {len(self.qa_pool)} Q/A pairs")
        print(f"Max exemplars: {self.config.num_slots}")
        print(f"{'=' * 60}\n")

        for iteration in range(num_iterations):
            record = self.run_iteration(iteration)

            # Log
            status = "IMPROVED" if record.improved else ""
            print(
                f"[{iteration + 1:3d}/{num_iterations}] "
                f"error={record.actual_error:.4f} "
                f"(pred={record.predicted_error:.4f}) "
                f"best={record.best_error_so_far:.4f} "
                f"K={record.num_exemplars} "
                f"cos={record.cosine_similarity:.3f} "
                f"{status}"
            )

            # Log full instruction
            print(f"  Instruction: {record.instruction}")
            if record.exemplar_ids:
                print(f"  Exemplars: {record.exemplar_ids}")

            # Save periodically
            if (iteration + 1) % 10 == 0:
                self.save_results(output_dir)

        # Final save
        self.save_results(output_dir)

        # Summary
        print(f"\n{'=' * 60}")
        print("FINAL RESULTS")
        print(f"{'=' * 60}")
        print(f"Best error rate: {self.best_error:.4f}")
        print(f"Best accuracy: {1 - self.best_error:.4f}")
        print(f"Best instruction: {self.best_instruction}")
        print(f"Best exemplars ({len(self.best_exemplars)}): {self.best_exemplars}")
        print(f"{'=' * 60}\n")

        return self.history

    def save_results(self, output_dir: str):
        """Save current results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save history
        history_path = os.path.join(output_dir, f"history_{timestamp}.json")
        with open(history_path, "w") as f:
            json.dump([asdict(r) for r in self.history], f, indent=2)

        # Save best result
        best_path = os.path.join(output_dir, "best_result.json")
        with open(best_path, "w") as f:
            json.dump({
                "best_error": self.best_error,
                "best_accuracy": 1 - self.best_error,
                "best_instruction": self.best_instruction,
                "best_exemplar_ids": self.best_exemplars,
                "best_exemplar_texts": [
                    self.qa_pool[i].format() for i in self.best_exemplars
                ],
                "num_iterations": len(self.history),
            }, f, indent=2)
