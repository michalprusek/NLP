"""InvBO inference pipeline for Inverse HbBoPs.

Provides:
- Vec2TextInverter: Embedding-to-text inverter (512_tokens model)
- InverseHbBoPsInference: Complete inference with LogEI optimization

Self-contained - no imports from other modules outside inverse_hbbops/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

from inverse_hbbops.encoder import GTRInstructionEncoder, InstructionVAE
from inverse_hbbops.gp import GPWithEI
from inverse_hbbops.instruction import InstructionOnlyPrompt


@dataclass
class InversionResult:
    """Result of latent-to-text inversion."""
    instruction_text: str
    latent: torch.Tensor
    embedding: torch.Tensor
    cosine_similarity: float
    predicted_error: float
    ei_value: float


@dataclass
class IterationRecord:
    """Record of a single optimization iteration."""
    iteration: int
    instruction: str
    cosine_similarity: float
    predicted_error: float
    actual_error: Optional[float]
    improved: bool
    best_error_so_far: float
    gp_samples: int
    log_ei: Optional[float] = None
    gap: float = 0.0
    inversion_iters: int = 1


@dataclass
class InversionStepResult:
    """Result of InvBO inversion step."""
    z_inv: torch.Tensor  # Inverted latent
    z_original: torch.Tensor  # Original latent before inversion
    gap: float  # Cosine distance (1 - cosine_similarity) between embeddings
    final_loss: float  # Final cosine loss
    converged: bool  # Whether optimization converged


class Vec2TextInverter:
    """Vec2Text embedding-to-text inverter.

    Uses 512_tokens model (vec2text/gtr-512-noise-0.00001) by default.
    This model supports longer sequences (up to 512 tokens vs 32 tokens).
    """

    def __init__(
        self,
        num_steps: int = 50,
        beam_width: int = 8,
        max_length: int = 128,
        device: str = "auto",
        model_type: str = "512_tokens",
    ):
        """Initialize inverter.

        Args:
            num_steps: Max new tokens for generation
            beam_width: Beam search width
            max_length: Maximum output length
            device: Device to use
            model_type: "32_tokens" or "512_tokens" (default)
        """
        if model_type not in ("32_tokens", "512_tokens"):
            raise ValueError(f"model_type must be '32_tokens' or '512_tokens'")

        self.num_steps = num_steps
        self.beam_width = beam_width
        self.max_length = max_length
        self.device = self._get_device(device)
        self.model_type = model_type
        self._corrector = None
        self._inversion_model = None

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self):
        """Lazy load Vec2Text model."""
        if self.model_type == "32_tokens":
            self._load_32_tokens()
        else:
            self._load_512_tokens()

    def _load_32_tokens(self):
        """Load ielabgroup Vec2Text with corrector (32 token limit)."""
        if self._corrector is not None:
            return

        import vec2text
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel
        from vec2text.models.corrector_encoder import CorrectorEncoderModel

        print("Loading Vec2Text (32_tokens with corrector)...")

        inv_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
        )
        inv_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )
        inversion_model = InversionModel(inv_config)
        inversion_model.load_state_dict(load_file(inv_weights), strict=False)
        inversion_model = inversion_model.to(self.device).eval()

        corr_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_corrector", "model.safetensors"
        )
        corr_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_corrector"
        )
        corrector_model = CorrectorEncoderModel(corr_config)
        corrector_model.load_state_dict(load_file(corr_weights), strict=False)
        corrector_model = corrector_model.to(self.device).eval()

        self._corrector = vec2text.load_corrector(inversion_model, corrector_model)
        print(f"  Vec2Text (32_tokens) loaded on {self.device}")

    def _load_512_tokens(self):
        """Load Vec2Text InversionModel (512 token limit)."""
        if self._inversion_model is not None:
            return

        import os
        import json
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        print("Loading Vec2Text (512_tokens InversionModel)...")

        model_dir = snapshot_download("vec2text/gtr-512-noise-0.00001")
        config = InversionConfig.from_pretrained(model_dir)
        print(f"  Config: max_seq_length={config.max_seq_length}")

        self._inversion_model = InversionModel(config)

        # Load sharded weights
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        self._inversion_model.load_state_dict(state_dict, strict=False)
        self._inversion_model = self._inversion_model.to(self.device).eval()

        print(f"  Vec2Text (512_tokens) loaded on {self.device}")

    def invert(self, embedding: torch.Tensor) -> str:
        """Invert embedding to text.

        Args:
            embedding: 768D GTR embedding

        Returns:
            Reconstructed text
        """
        self._load_model()

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        if self.model_type == "32_tokens":
            return self._invert_32_tokens(embedding)
        else:
            return self._invert_512_tokens(embedding)

    def _invert_32_tokens(self, embedding: torch.Tensor) -> str:
        """Invert using ielabgroup corrector."""
        import vec2text

        result = vec2text.invert_embeddings(
            embeddings=embedding,
            corrector=self._corrector,
            num_steps=self.num_steps,
            sequence_beam_width=self.beam_width,
        )
        return result[0] if isinstance(result, list) else result

    def _invert_512_tokens(self, embedding: torch.Tensor) -> str:
        """Invert using InversionModel direct generation."""
        gen_kwargs = {
            "num_beams": self.beam_width,
            "max_length": self.max_length,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
        }

        with torch.no_grad():
            output_ids = self._inversion_model.generate(
                inputs={"frozen_embeddings": embedding},
                generation_kwargs=gen_kwargs,
            )

        tokenizer = self._inversion_model.tokenizer
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return result.strip()


class InverseHbBoPsInference:
    """InvBO inference pipeline for Inverse HbBoPs.

    Pipeline:
        1. Optimize in 10D latent space using LogEI acquisition
        2. Decode optimal latent to 768D embedding via VAE decoder
        3. Invert embedding to text via Vec2Text (512_tokens)
        4. Evaluate and add to GP

    Uses 512_tokens Vec2Text model which is more exploratory.
    """

    def __init__(
        self,
        gp: GPWithEI,
        vae: InstructionVAE,
        gtr: Optional[GTRInstructionEncoder] = None,
        evaluator: Optional[Callable[[str, List[dict]], float]] = None,
        validation_data: Optional[List[dict]] = None,
        vec2text_model: str = "512_tokens",
        vec2text_beam: int = 8,
        device: str = "cuda",
        initial_best_instruction: Optional[str] = None,
        initial_best_error: Optional[float] = None,
    ):
        """Initialize inference pipeline.

        Args:
            gp: Trained GPWithEI
            vae: Trained InstructionVAE
            gtr: GTR encoder (for validation)
            evaluator: Function (instruction, data) -> error_rate
            validation_data: Validation Q/A pairs for evaluation
            vec2text_model: "32_tokens" or "512_tokens" (default)
            vec2text_beam: Beam width for Vec2Text
            device: Device to use
            initial_best_instruction: Best instruction from Hyperband (to avoid null)
            initial_best_error: Best error from Hyperband
        """
        self.gp = gp
        self.vae = vae
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.evaluator = evaluator
        self.validation_data = validation_data

        self.gtr = gtr if gtr is not None else GTRInstructionEncoder(device=str(self.device))
        self.inverter = Vec2TextInverter(
            beam_width=vec2text_beam,
            device=str(self.device),
            model_type=vec2text_model,
        )

        # History
        self.iteration_history: List[IterationRecord] = []
        self.total_llm_calls: int = 0
        self.best_error: float = float("inf")
        self.best_instruction: Optional[str] = None

        # Initialize best from Hyperband results or GP
        if initial_best_instruction is not None and initial_best_error is not None:
            self.best_instruction = initial_best_instruction
            self.best_error = initial_best_error
        elif gp.y_best is not None:
            self.best_error = gp.y_best
            # Note: instruction not available from GP alone

    def _get_latent_bounds(self, margin: float = 0.2) -> torch.Tensor:
        """Compute latent space bounds from training data."""
        encoder = self.gp.feature_extractor
        encoder.eval()

        with torch.no_grad():
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            X_norm = (self.gp.X_train - self.gp.X_min) / denom
            all_latents = encoder(X_norm)

        z_min = all_latents.min(dim=0)[0]
        z_max = all_latents.max(dim=0)[0]

        # Expand bounds for exploration
        z_range = z_max - z_min
        z_min = z_min - margin * z_range
        z_max = z_max + margin * z_range

        return torch.stack([z_min, z_max]).to(self.device)

    def optimize_latent_botorch(
        self,
        num_restarts: int = 20,
        raw_samples: int = 512,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize latent using BoTorch qLogExpectedImprovement.

        Uses multi-start L-BFGS-B with proper gradient flow through:
            latent z -> VAE decoder -> embedding -> GP posterior -> qLogEI

        This is the recommended optimization method as it:
        1. Uses numerically stable LogEI formulation
        2. Provides proper gradient-based optimization
        3. Uses sophisticated multi-start initialization

        Args:
            num_restarts: Number of L-BFGS-B restarts (default: 20)
            raw_samples: Raw samples for initialization seeding (default: 512)
            verbose: Print progress

        Returns:
            (optimal_latent, log_ei) tuple where:
            - optimal_latent: Best latent tensor, shape (10,)
            - log_ei: Log expected improvement value at optimal point
        """
        from inverse_hbbops.botorch_acq import LatentSpaceAcquisition, get_latent_bounds

        if verbose:
            print(f"Optimizing with BoTorch qLogEI ({num_restarts} restarts, {raw_samples} raw samples)...")

        # Get latent bounds from training data
        bounds = get_latent_bounds(
            encoder=self.gp.feature_extractor,
            X_train=self.gp.X_train,
            X_min=self.gp.X_min,
            X_max=self.gp.X_max,
            margin=0.2,
        )

        # Create acquisition optimizer
        acq_optimizer = LatentSpaceAcquisition(
            gp_model=self.gp.gp_model,
            vae=self.vae,
            bounds=bounds,
            device=self.device,
        )

        # Optimize - best_f is best observed error rate (we minimize error)
        best_f = self.gp.y_best
        z_opt, log_ei = acq_optimizer.optimize(
            best_f=best_f,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        if verbose:
            print(f"  BoTorch LogEI: {log_ei.item():.4f}")

        # Return as 1D tensor
        return z_opt.squeeze(), log_ei.item()

    def inversion_step(
        self,
        text: str,
        n_steps: int = 100,
        lr: float = 0.1,
        convergence_threshold: float = 0.01,
        verbose: bool = False,
    ) -> InversionStepResult:
        """InvBO-style inversion: find latent that reconstructs given text.

        Given text from Vec2Text, find z_inv such that:
            z_inv = argmin ||decoder(z) - GTR(text)||²

        This closes the loop and eliminates misalignment between decoder
        output and what Vec2Text can actually reconstruct.

        Args:
            text: Text to invert (from Vec2Text)
            n_steps: Maximum optimization steps
            lr: Learning rate for Adam
            convergence_threshold: Stop if loss < threshold
            verbose: Print progress

        Returns:
            InversionStepResult with inverted latent and metrics
        """
        # Get target embedding from text
        target_emb = self.gtr.encode_tensor(text)

        # Warm start: encode target embedding to get initial z
        encoder = self.gp.feature_extractor
        encoder.eval()

        with torch.no_grad():
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            target_norm = (target_emb - self.gp.X_min) / denom
            z_init = encoder(target_norm.unsqueeze(0)).squeeze(0)

        # Clone for optimization
        z = z_init.clone().requires_grad_(True)
        z_original = z_init.clone()

        optimizer = torch.optim.Adam([z], lr=lr)

        final_loss = float("inf")
        converged = False

        for step in range(n_steps):
            optimizer.zero_grad()

            # Decode latent to embedding (gradients flow through VAE decoder)
            decoded = self.vae.decode(z)

            # Cosine loss to target
            loss = 1 - F.cosine_similarity(
                decoded.unsqueeze(0), target_emb.unsqueeze(0)
            )

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            if verbose and (step + 1) % 20 == 0:
                print(f"    Inversion step {step + 1}: loss = {final_loss:.4f}")

            if final_loss < convergence_threshold:
                converged = True
                break

        z_inv = z.detach()

        # Gap as cosine distance in embedding space (more stable than L2 in latent)
        with torch.no_grad():
            self.vae.eval()
            emb_original = self.vae.decode(z_original)
            emb_inv = self.vae.decode(z_inv)
        cosine_gap = 1 - F.cosine_similarity(
            emb_original.unsqueeze(0), emb_inv.unsqueeze(0)
        ).item()
        gap = cosine_gap

        if verbose:
            print(f"  Inversion: gap = {gap:.4f}, loss = {final_loss:.4f}, converged = {converged}")

        return InversionStepResult(
            z_inv=z_inv,
            z_original=z_original,
            gap=gap,
            final_loss=final_loss,
            converged=converged,
        )

    def optimize_latent(
        self,
        n_restarts: int = 20,
        max_iter: int = 100,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize latent using LogEI with L-BFGS-B.

        Args:
            n_restarts: Number of random restarts
            max_iter: Max iterations per restart
            verbose: Print progress

        Returns:
            (optimal_latent, log_ei) tuple
        """
        if verbose:
            print("Optimizing latent with LogEI...")

        bounds_tensor = self._get_latent_bounds()
        z_min = bounds_tensor[0].cpu().numpy()
        z_max = bounds_tensor[1].cpu().numpy()
        bounds = [(z_min[i], z_max[i]) for i in range(len(z_min))]

        def neg_log_ei(z_np):
            """Negative LogEI for minimization."""
            z = torch.tensor(z_np, dtype=torch.float32, device=self.device)

            # Decode latent to embedding
            self.vae.eval()
            with torch.no_grad():
                embedding = self.vae.decode(z)

            # Compute LogEI
            log_ei = self.gp.log_expected_improvement(embedding)
            return -log_ei

        best_z = None
        best_log_ei = float("-inf")

        for restart in range(n_restarts):
            z0 = np.random.uniform(low=z_min, high=z_max)

            try:
                result = minimize(
                    neg_log_ei,
                    z0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iter},
                )

                log_ei = -result.fun
                if log_ei > best_log_ei:
                    best_log_ei = log_ei
                    best_z = result.x

            except RuntimeError as e:
                if verbose:
                    print(f"  L-BFGS-B restart {restart + 1} failed: {e}")
                continue

        if best_z is None:
            # Fallback to random sampling
            if verbose:
                print("  All restarts failed, using random sampling")
            best_z = np.random.uniform(low=z_min, high=z_max)
            z = torch.tensor(best_z, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                embedding = self.vae.decode(z)
            best_log_ei = self.gp.log_expected_improvement(embedding)

        if verbose:
            print(f"  Best LogEI: {best_log_ei:.4f}")

        return torch.tensor(best_z, dtype=torch.float32, device=self.device), best_log_ei

    def run_iteration(
        self,
        iteration: int,
        num_restarts: int = 20,
        raw_samples: int = 512,
        use_inversion: bool = True,
        max_inversion_iters: int = 3,
        gap_threshold: float = 0.1,
        skip_eval: bool = False,
        verbose: bool = True,
    ) -> IterationRecord:
        """Run a single optimization iteration using BoTorch qLogEI.

        Uses BoTorch's gradient-based LogEI optimization with multi-start
        L-BFGS-B for finding optimal latent points. Includes InvBO inversion
        loop to close the gap between decoder output and Vec2Text reconstruction.

        Args:
            iteration: Iteration number
            num_restarts: Number of L-BFGS-B restarts for BoTorch optimization
            raw_samples: Raw samples for initialization seeding
            use_inversion: Use InvBO inversion loop to improve alignment
            max_inversion_iters: Maximum inversion iterations
            gap_threshold: Gap threshold for re-inversion (cosine distance)
            skip_eval: Skip LLM evaluation (use GP prediction)
            verbose: Print progress

        Returns:
            IterationRecord with results
        """
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Optimize latent using BoTorch qLogEI
        z_opt, log_ei = self.optimize_latent_botorch(
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            verbose=verbose,
        )

        # Decode to embedding
        self.vae.eval()
        with torch.no_grad():
            embedding = self.vae.decode(z_opt)

        # Invert to text
        instruction = self.inverter.invert(embedding.clone())

        if verbose:
            print(f"  Generated:\n{instruction}")

        # Inversion loop if enabled - closes the decoder→Vec2Text→GTR gap
        gap = 0.0
        inv_iters = 1
        if use_inversion:
            for inv_iter in range(max_inversion_iters):
                inv_result = self.inversion_step(instruction, n_steps=100, lr=0.1, verbose=False)
                gap = inv_result.gap

                if gap <= gap_threshold:
                    if verbose:
                        print(f"  Gap {gap:.4f} <= {gap_threshold}, accepting")
                    break

                if verbose:
                    print(f"  Gap {gap:.4f} > {gap_threshold}, re-inverting")

                # Re-decode from inverted latent
                z_opt = inv_result.z_inv
                with torch.no_grad():
                    embedding = self.vae.decode(z_opt)
                instruction = self.inverter.invert(embedding.clone())
                inv_iters += 1

                if verbose:
                    print(f"  Re-generated:\n{instruction}")

        # Re-encode for GP prediction
        # IMPORTANT: Predict on GTR(text), not decoder(z), for alignment with training data
        reencoded = self.gtr.encode_tensor(instruction)
        cosine_sim = F.cosine_similarity(
            embedding.unsqueeze(0), reencoded.unsqueeze(0)
        ).item()

        if verbose:
            print(f"  Cosine similarity: {cosine_sim:.4f}")

        # GP prediction on re-encoded embedding (matches training data distribution)
        pred_error, pred_std = self.gp.predict(reencoded)

        if verbose:
            print(f"  Predicted error: {pred_error:.4f} +/- {pred_std:.4f}")

        # Evaluate with LLM (or use GP prediction)
        actual_error = None
        if not skip_eval and self.evaluator is not None and self.validation_data is not None:
            prompt = InstructionOnlyPrompt(instruction=instruction, instruction_id=-1)
            actual_error = self.evaluator(prompt, self.validation_data)
            self.total_llm_calls += len(self.validation_data)

            if verbose:
                print(f"  Actual error: {actual_error:.4f}")

        # Update best
        error_to_use = actual_error if actual_error is not None else pred_error
        improved = error_to_use < self.best_error
        if improved:
            self.best_error = error_to_use
            self.best_instruction = instruction
            if verbose:
                print(f"  NEW BEST!")

        # Add to GP and retrain
        retrain_success = self.gp.add_observation_and_retrain(
            reencoded,
            error_to_use,
            epochs=500,
            patience=10,
            verbose=False,
        )
        if not retrain_success and verbose:
            print(f"  Warning: GP retraining failed (Cholesky error), continuing with previous model")

        record = IterationRecord(
            iteration=iteration,
            instruction=instruction,
            cosine_similarity=cosine_sim,
            predicted_error=pred_error,
            actual_error=actual_error,
            improved=improved,
            best_error_so_far=self.best_error,
            gp_samples=self.gp.get_training_size(),
            log_ei=log_ei,
            gap=gap,
            inversion_iters=inv_iters,
        )

        self.iteration_history.append(record)

        return record

    def run(
        self,
        iterations: int = 10,
        num_restarts: int = 20,
        raw_samples: int = 512,
        use_inversion: bool = True,
        max_inversion_iters: int = 3,
        gap_threshold: float = 0.1,
        skip_eval: bool = False,
        verbose: bool = True,
    ) -> List[IterationRecord]:
        """Run multiple optimization iterations.

        Args:
            iterations: Number of iterations
            num_restarts: Number of L-BFGS-B restarts for BoTorch optimization
            raw_samples: Raw samples for initialization seeding
            use_inversion: Use InvBO inversion loop to improve alignment
            max_inversion_iters: Maximum inversion iterations per step
            gap_threshold: Gap threshold for re-inversion
            skip_eval: Skip LLM evaluation
            verbose: Print progress

        Returns:
            List of IterationRecords
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting InvBO Inference (BoTorch + Inversion)")
            print("=" * 60)
            print(f"  Initial best error: {self.best_error:.4f}")
            print(f"  GP samples: {self.gp.get_training_size()}")
            print(f"  Use inversion: {use_inversion}")
            print(f"  BoTorch: {num_restarts} restarts, {raw_samples} raw samples")

        for i in range(iterations):
            self.run_iteration(
                iteration=i + 1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                use_inversion=use_inversion,
                max_inversion_iters=max_inversion_iters,
                gap_threshold=gap_threshold,
                skip_eval=skip_eval,
                verbose=verbose,
            )

        if verbose:
            print("\n" + "=" * 60)
            print("InvBO Inference Complete")
            print("=" * 60)
            print(f"  Best error: {self.best_error:.4f}")
            print(f"  Best instruction:\n{self.best_instruction}")
            print(f"  Total LLM calls: {self.total_llm_calls}")

        return self.iteration_history
