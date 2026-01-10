"""Inference pipeline for BOLT.

Optimization loop:
1. Optimize joint latent (32D) using UCB/LogEI
2. Decode instruction via Vec2Text inversion
3. Decode exemplars via hard selection from pool
4. Build prompt, evaluate, add to GP, repeat
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models.model import Model

from bolt.config import BOLTConfig


class AcquisitionWithPriorPenalty(AnalyticAcquisitionFunction):
    """Wraps acquisition function with Gaussian prior penalty.

    Penalizes latent points far from origin to keep optimization
    in the region where Vec2Text produces coherent text.

    Score(z) = base_acq(z) - weight * 0.5 * ||z||²

    The penalty term is log P(z) where P(z) = N(0, I), so:
    log P(z) = -0.5 * ||z||² (ignoring constant)
    """

    def __init__(
        self,
        base_acq: AnalyticAcquisitionFunction,
        distance_weight: float = 2.0,
    ):
        # Initialize with the base acquisition function's model
        super().__init__(model=base_acq.model)
        self.base_acq = base_acq
        self.distance_weight = distance_weight

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute acquisition value with prior penalty.

        Args:
            X: Input tensor of shape (batch, q, d) or (batch, d)

        Returns:
            Penalized acquisition values
        """
        # Get base acquisition value
        base_value = self.base_acq(X)

        # Compute squared L2 norm of latent points
        # X shape: (batch, q, d) for q-batch or (batch, d)
        if X.dim() == 3:
            # q-batch case: (batch, q, d)
            squared_norm = (X ** 2).sum(dim=-1).mean(dim=-1)  # (batch,)
        else:
            # Standard case: (batch, d)
            squared_norm = (X ** 2).sum(dim=-1)  # (batch,)

        # Prior penalty: -0.5 * ||z||² (higher norm = more penalty)
        # We subtract because we want to penalize far-from-origin points
        prior_penalty = self.distance_weight * 0.5 * squared_norm

        return base_value - prior_penalty


from bolt.encoder import GTREncoder, StructureAwareVAE
from bolt.gp import GPWithEI
from bolt.training import HbBoPsEvaluator, QAPair
from lipo.inference import Vec2TextInverter

warnings.filterwarnings("ignore", message="Very small noise values detected")
warnings.filterwarnings("ignore", message="Optimization failed")
warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")


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


class BOLTInference:
    """BO inference for joint instruction-exemplar optimization."""

    def __init__(
        self,
        vae: StructureAwareVAE,
        gp: GPWithEI,
        gtr_encoder: GTREncoder,
        qa_pool: List[QAPair],
        pool_embeddings: torch.Tensor,
        instructions: List[str],
        evaluator: HbBoPsEvaluator,
        config: BOLTConfig,
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

        self.inverter = Vec2TextInverter(
            model_type=config.vec2text_model,
            beam_width=config.vec2text_beam,
            max_length=config.vec2text_max_length,
            device=config.device,
            num_steps=50,
        )

        # Best results
        self.best_error = float("inf")
        self.best_instruction: Optional[str] = None
        self.best_exemplars: List[int] = []
        self.history: List[IterationRecord] = []

    def optimize_latent(self, iteration: int) -> torch.Tensor:
        """Optimize joint latent using acquisition function.

        Args:
            iteration: Current iteration (for adaptive beta)

        Returns:
            Optimal latent tensor of shape (total_dim,)
        """
        bounds = self.gp.get_bounds(margin=self.config.latent_margin)

        if self.config.ucb_beta_adaptive:
            progress = iteration / max(self.config.iterations, 1)
            ucb_beta = self.config.ucb_beta - progress * (
                self.config.ucb_beta - self.config.ucb_beta_final
            )
        else:
            ucb_beta = self.config.ucb_beta

        if self.config.acquisition_type == "ucb":
            base_acq = qUpperConfidenceBound(
                model=self.gp.gp_model,
                beta=ucb_beta,
            )
        else:
            base_acq = qLogExpectedImprovement(
                model=self.gp.gp_model,
                best_f=self.gp.best_f,
            )

        # Wrap with prior penalty to keep latent near origin (where Vec2Text works)
        if self.config.distance_penalty_enabled:
            acq_func = AcquisitionWithPriorPenalty(
                base_acq=base_acq,
                distance_weight=self.config.distance_weight,
            )
        else:
            acq_func = base_acq

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            z_opt, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=self.config.num_restarts,
                raw_samples=self.config.raw_samples,
                options={"maxiter": 200, "batch_limit": 5},
            )

        if self.config.latent_noise_scale > 0:
            noise = torch.randn_like(z_opt) * self.config.latent_noise_scale
            z_opt = torch.clamp(z_opt + noise, bounds[0], bounds[1])

        return z_opt.squeeze()

    def decode_and_generate(self, z_opt: torch.Tensor) -> Tuple[str, List[int], float]:
        """Decode optimal latent to instruction and exemplars.

        Args:
            z_opt: Optimal latent of shape (total_dim,)

        Returns:
            Tuple of (instruction text, exemplar pool indices, round-trip cosine similarity)
        """
        self.vae.eval()
        instruction_dim = self.config.instruction_latent_dim
        num_exemplars = self.config.num_exemplars

        z_inst = z_opt[:instruction_dim].unsqueeze(0)
        z_ex = z_opt[instruction_dim:].unsqueeze(0)

        with torch.no_grad():
            inst_emb_recon = self.vae.instruction_decoder(z_inst)
            selected_indices, _ = self.vae.scorer.select_top_k(
                z_inst, z_ex, self.pool_embeddings,
                k=num_exemplars,
                use_mmr=self.config.use_mmr,
                mmr_lambda=self.config.mmr_lambda,
            )

        instruction = self.inverter.invert(inst_emb_recon)
        if not instruction or len(instruction.strip()) < 3:
            raise ValueError(
                f"Vec2Text generated invalid instruction: '{instruction}'. "
                f"Embedding may be out of distribution."
            )

        inst_emb_new = self.gtr_encoder.encode_single(instruction)
        cosine_sim = F.cosine_similarity(
            inst_emb_recon.squeeze(), inst_emb_new, dim=0
        ).item()

        exemplar_ids = selected_indices.squeeze(0).tolist()
        pool_size = len(self.pool_embeddings)
        for eid in exemplar_ids:
            if eid < 0 or eid >= pool_size:
                raise ValueError(
                    f"Invalid exemplar ID {eid} (pool size: {pool_size}). "
                    f"This may indicate a scorer bug."
                )

        return instruction, exemplar_ids, cosine_sim

    def run_iteration(self, iteration: int) -> IterationRecord:
        """Run single optimization iteration.

        Args:
            iteration: Iteration number

        Returns:
            Iteration results record
        """
        z_opt = self.optimize_latent(iteration)

        z_opt_norm = self.gp._normalize_X(z_opt.unsqueeze(0))
        with torch.no_grad():
            pred = self.gp.likelihood(self.gp.gp_model(z_opt_norm))
            predicted_error = -(pred.mean.item() * self.gp.y_std + self.gp.y_mean)

        # Rejection sampling for high-quality instruction generation
        best_instruction, best_exemplars, best_cosine = self.decode_and_generate(z_opt)
        rejection_attempts = 0

        for _ in range(self.config.max_rejection_attempts - 1):
            if best_cosine >= self.config.cosine_sim_threshold:
                break

            rejection_attempts += 1
            noise = torch.randn_like(z_opt) * self.config.latent_noise_scale * 2
            z_opt = z_opt + noise

            instruction, exemplar_ids, cosine_sim = self.decode_and_generate(z_opt)
            if cosine_sim > best_cosine:
                best_instruction = instruction
                best_exemplars = exemplar_ids
                best_cosine = cosine_sim

        qa_pairs = [self.qa_pool[i] for i in best_exemplars]
        actual_error = self.evaluator.evaluate(
            instruction=best_instruction,
            qa_pairs=qa_pairs,
            fidelity=len(self.evaluator.validation_data),
        )

        improved = actual_error < self.best_error
        if improved:
            self.best_error = actual_error
            self.best_instruction = best_instruction
            self.best_exemplars = best_exemplars

        # Encode actual instruction for GP observation
        inst_emb = self.gtr_encoder.encode_single(best_instruction).detach()
        ex_embs = torch.stack([self.pool_embeddings[i] for i in best_exemplars]).detach()
        ex_mask = torch.ones(len(best_exemplars), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            z_real = self.vae.encode_joint(
                inst_emb.unsqueeze(0),
                ex_embs.unsqueeze(0),
                ex_mask.unsqueeze(0),
            ).squeeze()

        self.gp.add_observation(z_real, torch.tensor(actual_error, device=self.device))

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

    def run(self, num_iterations: int, output_dir: str) -> List[IterationRecord]:
        """Run full optimization loop.

        Args:
            num_iterations: Number of iterations
            output_dir: Directory to save results

        Returns:
            All iteration records
        """
        os.makedirs(output_dir, exist_ok=True)

        separator = "=" * 60
        print(f"\n{separator}")
        print("BOLT Inference")
        print(separator)
        print(f"Iterations: {num_iterations}")
        print(f"Acquisition: {self.config.acquisition_type}")
        print(f"UCB beta: {self.config.ucb_beta} -> {self.config.ucb_beta_final}")
        print(f"Pool size: {len(self.qa_pool)} Q/A pairs")
        print(f"Num exemplars: {self.config.num_exemplars}")
        print(f"{separator}\n")

        for iteration in range(num_iterations):
            record = self.run_iteration(iteration)

            status = " IMPROVED" if record.improved else ""
            print(
                f"[{iteration + 1:3d}/{num_iterations}] "
                f"error={record.actual_error:.4f} "
                f"(pred={record.predicted_error:.4f}) "
                f"best={record.best_error_so_far:.4f} "
                f"K={record.num_exemplars} "
                f"cos={record.cosine_similarity:.3f}{status}"
            )
            print(f"  Instruction: {record.instruction}")
            print(f"  Exemplars: {record.exemplar_ids}")

            if (iteration + 1) % 10 == 0:
                self.save_results(output_dir)

        self.save_results(output_dir)

        print(f"\n{separator}")
        print("FINAL RESULTS")
        print(separator)
        print(f"Best error rate: {self.best_error:.4f}")
        print(f"Best accuracy: {1 - self.best_error:.4f}")
        print(f"Best instruction: {self.best_instruction}")
        print(f"Best exemplars ({len(self.best_exemplars)}): {self.best_exemplars}")
        print(f"{separator}\n")

        return self.history

    def save_results(self, output_dir: str) -> None:
        """Save current results to disk."""
        import tempfile

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        history_data = [asdict(r) for r in self.history]
        best_result = {
            "best_error": self.best_error,
            "best_accuracy": 1 - self.best_error,
            "best_instruction": self.best_instruction,
            "best_exemplar_ids": self.best_exemplars,
            "best_exemplar_texts": [self.qa_pool[i].format() for i in self.best_exemplars],
            "num_iterations": len(self.history),
        }

        try:
            os.makedirs(output_dir, exist_ok=True)

            history_path = os.path.join(output_dir, f"history_{timestamp}.json")
            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2)

            best_path = os.path.join(output_dir, "best_result.json")
            with open(best_path, "w") as f:
                json.dump(best_result, f, indent=2)

            print(f"  Results saved to {output_dir}")
        except (OSError, IOError) as e:
            print(f"[WARNING] Failed to save results to {output_dir}: {e}")
            # Fallback to temp directory
            fallback_dir = tempfile.gettempdir()
            try:
                history_path = os.path.join(fallback_dir, f"bolt_history_{timestamp}.json")
                with open(history_path, "w") as f:
                    json.dump(history_data, f, indent=2)

                best_path = os.path.join(fallback_dir, f"bolt_best_{timestamp}.json")
                with open(best_path, "w") as f:
                    json.dump(best_result, f, indent=2)

                print(f"  Results saved to fallback: {fallback_dir}")
            except Exception as fallback_e:
                print(f"[ERROR] Fallback save also failed: {fallback_e}")
