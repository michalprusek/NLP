"""InvBO inference pipeline for LIPO.

Provides:
- Vec2TextInverter: Embedding-to-text inverter (512_tokens model)
- LIPOHyperbandInference: Complete inference with LogEI optimization

Self-contained within lipo package. Uses external libs: vec2text, safetensors, huggingface_hub.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

from lipo.config import Config
from lipo.encoder import GTRInstructionEncoder, InstructionVAE
from lipo.gp import GPWithEI
from lipo.instruction import InstructionOnlyPrompt


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
    rejection_attempts: int = 0  # How many candidates were rejected before acceptance
    low_quality_accepted: bool = False  # Whether this was forced acceptance below threshold
    # Optimization Gap Test metrics (z_opt vs z_real after Vec2Text inversion)
    z_opt_z_real_cosine: float = 0.0     # Cosine sim in VAE latent space (64D)
    z_opt_z_real_euclidean: float = 0.0  # Euclidean distance in VAE latent space
    z_opt_z_real_gp_cosine: float = 0.0  # Cosine sim in GP adapter space (10D)
    predicted_error_at_z_real: float = 0.0  # GP prediction at actual z_real point


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

        # Clear CUDA cache before loading - Vec2Text loads its own GTR model internally
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

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


class LIPOHyperbandInference:
    """InvBO inference pipeline for LIPO.

    Pipeline:
        1. Optimize in 64D VAE latent space using LogEI acquisition
           (GP uses adapter: 64D → 10D for kernel computation)
        2. Decode optimal latent to 768D embedding via VAE decoder
        3. Invert embedding to text via Vec2Text (512_tokens)
        4. Evaluate and add to GP

    Uses 512_tokens Vec2Text model for longer instruction generation.
    """

    def __init__(
        self,
        gp: GPWithEI,
        vae: InstructionVAE,
        config: Config,
        gtr: Optional[GTRInstructionEncoder] = None,
        evaluator: Optional[Callable[[str, List[dict]], float]] = None,
        validation_data: Optional[List[dict]] = None,
        initial_best_instruction: Optional[str] = None,
        initial_best_error: Optional[float] = None,
    ):
        """Initialize inference pipeline.

        Args:
            gp: Trained GPWithEI
            vae: Trained InstructionVAE
            config: Unified pipeline configuration
            gtr: GTR encoder (for validation)
            evaluator: Function (instruction, data) -> error_rate
            validation_data: Validation Q/A pairs for evaluation
            initial_best_instruction: Best instruction from Hyperband (to avoid null)
            initial_best_error: Best error from Hyperband
        """
        self.gp = gp
        self.vae = vae
        self.config = config
        # VAEWithAdapter for decoding (64D -> 768D)
        self.vae_with_adapter = gp.vae_with_adapter
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.evaluator = evaluator
        self.validation_data = validation_data

        self.gtr = gtr if gtr is not None else GTRInstructionEncoder(device=str(self.device))
        self.inverter = Vec2TextInverter(
            beam_width=config.vec2text_beam,
            device=str(self.device),
            model_type=config.vec2text_model,
            max_length=getattr(config, 'vec2text_max_length', 128),
        )

        # History
        self.iteration_history: List[IterationRecord] = []
        self.total_llm_calls: int = 0
        self.best_error: float = float("inf")
        self.best_instruction: Optional[str] = None
        self._consecutive_retrain_failures: int = 0  # Track consecutive GP retrain failures

        # Initialize best from Hyperband results or GP
        if initial_best_instruction is not None and initial_best_error is not None:
            self.best_instruction = initial_best_instruction
            self.best_error = initial_best_error
        elif gp.best_error_rate is not None:
            self.best_error = gp.best_error_rate  # Use property that returns positive error rate
            # Note: instruction not available from GP alone

    def optimize_latent_botorch(
        self,
        num_restarts: int = 64,
        raw_samples: int = 1024,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize VAE latent using BoTorch qLogExpectedImprovement.

        Uses multi-start L-BFGS-B with proper gradient flow through:
            z (64D VAE latent) -> adapter -> z_gp (10D) -> GP posterior -> qLogEI

        This is the recommended optimization method as it:
        1. Optimizes in rich 64D VAE latent space
        2. Uses adapter to compress to 10D for efficient GP
        3. Gradients flow through adapter for latent optimization

        Args:
            num_restarts: Number of L-BFGS-B restarts (default: 64)
            raw_samples: Raw samples for initialization seeding (default: 1024)
            verbose: Print progress
            seed: Optional random seed for reproducibility

        Returns:
            (optimal_latent, log_ei) tuple where:
            - optimal_latent: Best VAE latent tensor, shape (64,)
            - log_ei: Log expected improvement value at optimal point
        """
        from lipo.botorch_acq import LatentSpaceAcquisition, get_latent_bounds

        if verbose:
            print(f"Optimizing with BoTorch qLogEI ({num_restarts} restarts, {raw_samples} raw samples)...")

        # Get latent bounds from training data (64D VAE latent space)
        bounds = get_latent_bounds(
            encoder=self.gp.vae_with_adapter,
            X_train=self.gp.X_train,
            X_min=self.gp.X_min,
            X_max=self.gp.X_max,
            margin=self.config.latent_margin,
        )

        # Create acquisition optimizer
        # Optimization path: z (64D) → GP (with adapter) → z_gp (10D) → kernel → qLogEI
        # After optimization: z_opt (64D) → VAE decoder → embedding (768D)
        acq_optimizer = LatentSpaceAcquisition(
            gp_model=self.gp.gp_model,
            bounds=bounds,
            device=self.device,
        )

        # best_f is the best observed GP target value (-min_error).
        # Since GP predicts -error_rate, qLogEI maximizes (finds lower error).
        best_f = self.gp.y_best
        z_opt, log_ei = acq_optimizer.optimize(
            best_f=best_f,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            seed=seed,
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

        # Warm start: encode target embedding to VAE latent (64D)
        encoder = self.gp.vae_with_adapter
        encoder.eval()

        with torch.no_grad():
            # Encode 768D embedding to 64D VAE latent (unnormalized)
            z_vae = encoder.encode_vae(target_emb.unsqueeze(0)).squeeze(0)

            # Normalize to GP input space
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            z_init = (z_vae - self.gp.X_min) / denom

        # Clone for optimization
        z = z_init.clone().requires_grad_(True)
        z_original = z_init.clone()

        optimizer = torch.optim.Adam([z], lr=lr)

        final_loss = float("inf")
        converged = False

        # Precompute denormalization constants for gradient flow
        x_range = self.gp.X_max - self.gp.X_min

        for step in range(n_steps):
            optimizer.zero_grad()

            # Denormalize z to VAE latent space (64D), then decode to 768D embedding
            z_vae = z * x_range + self.gp.X_min
            decoded = self.vae_with_adapter.decode(z_vae)

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
            self.vae_with_adapter.eval()
            z_vae_original = z_original * x_range + self.gp.X_min
            z_vae_inv = z_inv * x_range + self.gp.X_min
            emb_original = self.vae_with_adapter.decode(z_vae_original)
            emb_inv = self.vae_with_adapter.decode(z_vae_inv)
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

    def run_iteration(
        self,
        iteration: int,
        num_restarts: int = 64,
        raw_samples: int = 1024,
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

        Also enforces cosine similarity threshold between decoder output and
        re-encoded text to reject misaligned candidates.

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

        # Get thresholds from config
        cosine_sim_threshold = getattr(self.config, 'cosine_sim_threshold', 0.90)
        max_rejection_attempts = getattr(self.config, 'max_rejection_attempts', 5)

        # Initialize rejection tracking (in case loop doesn't execute)
        rejection_attempts = 0
        low_quality_accepted = False

        # Main loop with rejection for low cosine similarity
        for attempt in range(max_rejection_attempts):
            # Optimize latent using BoTorch qLogEI
            # Use different seed for each attempt to get different candidates
            attempt_seed = iteration * max_rejection_attempts + attempt
            z_opt, log_ei = self.optimize_latent_botorch(
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                verbose=verbose and (attempt == 0),  # Only verbose on first attempt
                seed=attempt_seed,
            )

            # Denormalize and decode to embedding (64D latent -> 768D embedding)
            # z_opt is normalized [0,1], need to convert back to VAE latent space
            x_range = self.gp.X_max - self.gp.X_min
            z_unnorm = z_opt * x_range + self.gp.X_min

            # Save original z_opt for optimization gap measurement
            z_opt_original = z_opt.clone()

            self.vae_with_adapter.eval()
            with torch.no_grad():
                embedding = self.vae_with_adapter.decode(z_unnorm)

            # Invert to text
            instruction = self.inverter.invert(embedding.clone())

            if verbose:
                if attempt > 0:
                    print(f"  [Attempt {attempt + 1}] Generated:\n{instruction}")
                else:
                    print(f"  Generated:\n{instruction}")

            # Inversion loop if enabled - closes the decoder→Vec2Text→GTR gap
            gap = 0.0
            inv_iters = 1
            if use_inversion:
                for inv_iter in range(max_inversion_iters):
                    inv_result = self.inversion_step(
                        instruction,
                        n_steps=self.config.inversion_n_steps,
                        lr=self.config.inversion_lr,
                        convergence_threshold=self.config.inversion_convergence_threshold,
                        verbose=False,
                    )
                    gap = inv_result.gap

                    if gap <= gap_threshold:
                        if verbose:
                            print(f"  Gap {gap:.4f} <= {gap_threshold}, accepting")
                        break

                    if verbose:
                        print(f"  Gap {gap:.4f} > {gap_threshold}, re-inverting")

                    # Re-decode from inverted latent (z_inv is normalized, need denormalization)
                    z_opt = inv_result.z_inv
                    z_unnorm = z_opt * x_range + self.gp.X_min
                    with torch.no_grad():
                        embedding = self.vae_with_adapter.decode(z_unnorm)
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

            # Check cosine similarity threshold
            if cosine_sim >= cosine_sim_threshold:
                # Good alignment, proceed with this candidate
                rejection_attempts = attempt
                low_quality_accepted = False
                break
            else:
                if attempt < max_rejection_attempts - 1:
                    if verbose:
                        print(f"  REJECTED: Cosine sim {cosine_sim:.4f} < {cosine_sim_threshold:.2f}, "
                              f"retrying ({attempt + 1}/{max_rejection_attempts})")
                else:
                    # Always warn about low-quality acceptance - this affects optimization quality
                    print(f"WARNING: Accepting low-quality candidate after {max_rejection_attempts} attempts")
                    print(f"  Cosine similarity: {cosine_sim:.4f} < threshold {cosine_sim_threshold:.2f}")
                    print(f"  This may indicate: Vec2Text inversion issues or threshold too strict")
                    rejection_attempts = attempt
                    low_quality_accepted = True

        # GP prediction on re-encoded embedding (matches training data distribution)
        pred_error, pred_std = self.gp.predict(reencoded)

        if verbose:
            print(f"  Predicted error: {pred_error:.4f} +/- {pred_std:.4f}")

        # === Optimization Gap Test ===
        # Measure gap between z_opt (BoTorch proposal) and z_real (actual text embedding)
        # If gap is large, GP is optimizing in "empty space" where no real text maps
        with torch.no_grad():
            # Encode re-encoded text back to VAE latent space
            z_real = self.vae_with_adapter.encode_vae(reencoded.unsqueeze(0)).squeeze(0)
            # Normalize to GP input space [0,1] for fair comparison
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            z_real_norm = (z_real - self.gp.X_min) / denom

            # Gap metrics in VAE latent space (64D)
            z_opt_z_real_cosine = F.cosine_similarity(
                z_opt_original.unsqueeze(0), z_real_norm.unsqueeze(0)
            ).item()
            z_opt_z_real_euclidean = torch.dist(z_opt_original, z_real_norm).item()

            # Gap metrics in GP adapter space (10D)
            z_opt_unnorm = z_opt_original * x_range + self.gp.X_min
            z_opt_gp = self.vae_with_adapter.adapter(z_opt_unnorm.unsqueeze(0))
            z_real_gp = self.vae_with_adapter.adapter(z_real.unsqueeze(0))
            z_opt_z_real_gp_cosine = F.cosine_similarity(z_opt_gp, z_real_gp).item()

            # GP prediction at z_real (actual text) vs at z_opt (dream)
            # This shows if GP was "fooled" by holes in latent space
            pred_error_at_z_real, _ = self.gp.predict(reencoded)

        if verbose:
            print(f"  Optimization Gap: VAE cosine={z_opt_z_real_cosine:.4f}, "
                  f"GP cosine={z_opt_z_real_gp_cosine:.4f}, euclidean={z_opt_z_real_euclidean:.4f}")
            if z_opt_z_real_cosine < 0.85:
                print(f"  WARNING: Large optimization gap! GP may be optimizing in empty space.")

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

        # Add observation and retrain GP from scratch
        # Full retraining ensures normalization parameters and noise values are recomputed
        # Determine fidelity for new observation
        if actual_error is not None and self.validation_data is not None:
            # Actual LLM evaluation - use full fidelity
            fidelity = len(self.validation_data)
        elif self.validation_data is not None:
            # GP prediction only - use lower fidelity to reflect model uncertainty
            fidelity = 100  # Conservative fidelity for model predictions
        else:
            fidelity = 100  # Conservative default when no validation data

        self.gp.add_observation(
            reencoded,
            error_to_use,
            fidelity=fidelity,
        )
        retrain_success = self.gp.train(
            epochs=self.config.gp_retrain_epochs,
            patience=self.config.gp_retrain_patience,
            verbose=verbose,
        )
        if not retrain_success:
            self._consecutive_retrain_failures += 1
            print(f"ERROR: GP retraining failed at iteration {iteration}")
            print(f"  Consecutive failures: {self._consecutive_retrain_failures}")
            print(f"  Training samples: {self.gp.get_training_size()}")

            if self._consecutive_retrain_failures >= 3:
                raise RuntimeError(
                    f"GP retraining failed {self._consecutive_retrain_failures} consecutive times. "
                    f"This indicates a systematic problem with the training data. "
                    f"Possible causes: duplicate observations, numerical overflow, or ill-conditioned kernel."
                )
            print(f"  WARNING: Continuing with previous model (attempt {self._consecutive_retrain_failures}/3)")
        else:
            self._consecutive_retrain_failures = 0

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
            rejection_attempts=rejection_attempts,
            low_quality_accepted=low_quality_accepted,
            # Optimization Gap Test metrics
            z_opt_z_real_cosine=z_opt_z_real_cosine,
            z_opt_z_real_euclidean=z_opt_z_real_euclidean,
            z_opt_z_real_gp_cosine=z_opt_z_real_gp_cosine,
            predicted_error_at_z_real=pred_error_at_z_real,
        )

        self.iteration_history.append(record)

        return record

    def run(
        self,
        iterations: int = 10,
        num_restarts: int = 64,
        raw_samples: int = 1024,
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

    def round_trip_test(self, instruction: str) -> dict:
        """Test reconstruction fidelity: Text → GTR → VAE → Vec2Text → Text'

        This measures how well the VAE + Vec2Text pipeline can reconstruct
        a known instruction. High similarity (>0.95) means good reconstruction.

        Args:
            instruction: Original instruction text to test

        Returns:
            dict with:
                - semantic_similarity: cosine similarity between GTR(original) and GTR(reconstructed)
                - reconstructed_text: the reconstructed instruction
                - original_text: the input instruction
        """
        # Encode original text
        emb_original = self.gtr.encode_tensor(instruction)

        # VAE encode -> decode (full round-trip through latent space)
        self.vae_with_adapter.eval()
        with torch.no_grad():
            z = self.vae_with_adapter.encode_vae(emb_original.unsqueeze(0))
            emb_decoded = self.vae_with_adapter.decode(z.squeeze(0))

        # Vec2Text inversion (embedding -> text)
        reconstructed = self.inverter.invert(emb_decoded.clone())

        # Re-encode reconstructed text
        emb_reconstructed = self.gtr.encode_tensor(reconstructed)

        # Compute semantic similarity in GTR embedding space
        semantic_sim = F.cosine_similarity(
            emb_original.unsqueeze(0),
            emb_reconstructed.unsqueeze(0)
        ).item()

        return {
            "semantic_similarity": semantic_sim,
            "reconstructed_text": reconstructed,
            "original_text": instruction,
        }

    def run_round_trip_diagnostic(
        self, instructions: List[str], verbose: bool = True
    ) -> dict:
        """Run round-trip test on multiple instructions.

        Recommended: Run on top-K training instructions before inference
        to establish baseline reconstruction quality.

        Args:
            instructions: List of instructions to test
            verbose: Print summary

        Returns:
            dict with:
                - mean_similarity: average cosine similarity
                - min_similarity: worst case
                - max_similarity: best case
                - below_90: count of instructions with sim < 0.90 (poor)
                - below_95: count of instructions with sim < 0.95 (acceptable)
                - results: list of individual test results
        """
        import numpy as np

        results = []
        for inst in instructions:
            result = self.round_trip_test(inst)
            results.append(result)

        sims = [r["semantic_similarity"] for r in results]
        summary = {
            "mean_similarity": float(np.mean(sims)),
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
            "below_90": sum(1 for s in sims if s < 0.90),
            "below_95": sum(1 for s in sims if s < 0.95),
            "results": results,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print("ROUND-TRIP DIAGNOSTIC")
            print("=" * 60)
            print(f"Tested: {len(instructions)} instructions")
            print(f"Mean similarity: {summary['mean_similarity']:.4f}")
            print(f"Min: {summary['min_similarity']:.4f}, Max: {summary['max_similarity']:.4f}")
            print(f"Below 0.90 (poor): {summary['below_90']}")
            print(f"Below 0.95 (acceptable): {summary['below_95']}")

            # Interpretation
            if summary['mean_similarity'] >= 0.95:
                print("Interpretation: EXCELLENT - VAE+Vec2Text preserve meaning well")
            elif summary['mean_similarity'] >= 0.90:
                print("Interpretation: ACCEPTABLE - minor semantic drift expected")
            else:
                print("Interpretation: POOR - significant meaning loss in reconstruction")
                print("  This may explain lack of optimization improvement.")

            # Show worst case if it's particularly bad
            if summary['min_similarity'] < 0.85:
                worst_idx = sims.index(summary['min_similarity'])
                worst = results[worst_idx]
                print(f"\nWorst reconstruction (sim={worst['semantic_similarity']:.4f}):")
                print(f"  Original: {worst['original_text'][:100]}...")
                print(f"  Reconstructed: {worst['reconstructed_text'][:100]}...")

        return summary
