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

from lipo.config import Config, get_device
from lipo.encoder import GTRInstructionEncoder, InstructionVAE
from lipo.gp import GPWithEI
from lipo.instruction import InstructionOnlyPrompt
from lipo.turbo import (
    TrustRegionManager,
    PotentialAwareAnchorSelector,
    create_turbo_manager,
    create_pas_selector,
)
from lipo.quality_kpi import compute_gp_spearman, compute_system_gap, format_kpi_report


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
    z_opt_z_real_cosine: float = 0.0     # Cosine sim in VAE latent space (32D)
    z_opt_z_real_euclidean: float = 0.0  # Euclidean distance in VAE latent space
    z_opt_z_real_gp_cosine: float = 0.0  # Same as z_opt_z_real_cosine (no adapter, GP on 32D)
    predicted_error_at_z_real: float = 0.0  # GP prediction at actual z_real point
    # TuRBO trust region state
    trust_region_length: float = 0.0  # Current trust region side length
    trust_region_action: str = ""  # Action taken: "none", "expand", "shrink", "restart"
    anchor_idx: int = -1  # Index of selected anchor in training data


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
        self.device = get_device(device)
        self.model_type = model_type
        self._corrector = None
        self._inversion_model = None

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
        # Note: gc.collect() must come before empty_cache() to free Python objects first
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

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


def validate_roundtrip_quality(
    vae: InstructionVAE,
    gtr: GTRInstructionEncoder,
    inverter: Vec2TextInverter,
    instructions: List[str],
    n_samples: int = 20,
    verbose: bool = True,
) -> dict:
    """Validate round-trip quality of VAE + Vec2Text pipeline.

    Tests how well the full pipeline can reconstruct known instructions:
        instruction → GTR → VAE(encode→decode) → Vec2Text → GTR → cosine_sim

    Poor round-trip quality indicates GP may optimize in "empty space" -
    regions that don't correspond to valid instructions.

    Args:
        vae: Trained InstructionVAE
        gtr: GTR encoder
        inverter: Vec2Text inverter
        instructions: List of instructions to test
        n_samples: Number of random samples to test
        verbose: Print progress and results

    Returns:
        Dict with metrics:
        - mean_sim: Mean cosine similarity
        - std_sim: Std of similarities
        - min_sim: Worst reconstruction
        - poor_count: Number with sim < 0.90
        - samples: List of (original, reconstructed, sim) tuples
    """
    import random
    import numpy as np

    # Sample instructions
    n = min(n_samples, len(instructions))
    samples = random.sample(instructions, n)

    vae.eval()
    vae_dev = next(vae.parameters()).device
    sims = []
    sample_results = []

    for i, instruction in enumerate(samples):
        # Full pipeline - ensure device consistency
        emb_original = gtr.encode_tensor(instruction)
        with torch.no_grad():
            emb_for_vae = emb_original.to(vae_dev)
            mu, _ = vae.encode(emb_for_vae)
            decoded = vae.decode(mu)

        reconstructed = inverter.invert(decoded)
        emb_recon = gtr.encode_tensor(reconstructed)

        sim = F.cosine_similarity(
            emb_original.unsqueeze(0),
            emb_recon.unsqueeze(0)
        ).item()
        sims.append(sim)
        sample_results.append((instruction, reconstructed, sim))

        if verbose and (i + 1) % 5 == 0:
            print(f"  Validated {i + 1}/{n} samples...")

    sims_arr = np.array(sims)
    results = {
        "mean_sim": float(np.mean(sims_arr)),
        "std_sim": float(np.std(sims_arr)),
        "min_sim": float(np.min(sims_arr)),
        "max_sim": float(np.max(sims_arr)),
        "poor_count": int(np.sum(sims_arr < 0.90)),
        "acceptable_count": int(np.sum(sims_arr >= 0.90)),
        "n_samples": n,
        "samples": sample_results,
    }

    if verbose:
        quality = "GOOD" if results["mean_sim"] >= 0.90 else "POOR"
        print(f"""
============================================================
ROUND-TRIP DIAGNOSTIC
============================================================
Tested: {n} instructions
Mean similarity: {results['mean_sim']:.4f}
Min: {results['min_sim']:.4f}, Max: {results['max_sim']:.4f}
Below 0.90 (poor): {results['poor_count']}
Below 0.95 (acceptable): {n - results['acceptable_count'] + results['poor_count']}
Interpretation: {quality} - {"good reconstruction" if quality == "GOOD" else "significant meaning loss in reconstruction"}
  {"" if quality == "GOOD" else "This may explain lack of optimization improvement."}
""")
        # Show worst example
        worst_idx = np.argmin(sims_arr)
        orig, recon, wsim = sample_results[worst_idx]
        print(f"Worst reconstruction (sim={wsim:.4f}):")
        print(f"  Original: {orig[:80]}...")
        print(f"  Reconstructed: {recon[:100]}...")

    return results


class ZSInvertRefiner:
    """ZSInvert-style iterative refinement for Vec2Text output.

    After Vec2Text inverts embedding→text, there's often semantic drift
    (similarity ~0.85). ZSInvert uses gradient-based optimization to refine
    the latent z so decode(z) better matches GTR(text), then re-inverts.

    Algorithm (per iteration):
        1. Encode current text with GTR → target_emb
        2. Gradient descent: minimize ||decode(z) - target_emb||
        3. Decode optimized z and re-invert with Vec2Text
        4. Check improvement, continue if significant

    This "tightens" the loop between latent optimization and text generation.
    """

    def __init__(
        self,
        vae_with_adapter,
        gtr_encoder: GTRInstructionEncoder,
        inverter: Vec2TextInverter,
        n_iterations: int = 3,
        lr: float = 0.1,
        n_steps_per_iter: int = 50,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        device: str = "cuda",
    ):
        """Initialize ZSInvert refiner.

        Args:
            vae_with_adapter: VAEWithAdapter for decoding latents to embeddings
            gtr_encoder: GTR encoder for text→embedding
            inverter: Vec2Text inverter for embedding→text
            n_iterations: Maximum refinement iterations
            lr: Learning rate for gradient descent
            n_steps_per_iter: Optimization steps per iteration
            improvement_threshold: Minimum improvement to continue
            patience: Number of iterations without improvement before stopping
            device: Torch device
        """
        self.vae = vae_with_adapter
        self.gtr = gtr_encoder
        self.inverter = inverter
        self.n_iterations = n_iterations
        self.lr = lr
        self.n_steps = n_steps_per_iter
        self.threshold = improvement_threshold
        self.patience = patience
        self.device = torch.device(device)

    def refine(
        self,
        initial_text: str,
        initial_z: torch.Tensor,
        X_min: torch.Tensor,
        X_max: torch.Tensor,
        verbose: bool = False,
    ) -> Tuple[str, torch.Tensor, dict]:
        """Iteratively refine text using gradient-based latent optimization.

        The goal is to find text where GTR(text) ≈ original_decoder_output.
        We use a FIXED target (the original decoder output) to measure progress
        consistently across iterations.

        Algorithm:
            1. Compute fixed target = decode(initial_z) - this is what we're trying to match
            2. Each iteration:
               a. Optimize z to minimize ||decode(z) - GTR(current_best_text)||
               b. Decode optimized z and invert via Vec2Text
               c. Measure sim = cosine(GTR(new_text), fixed_target)
               d. Keep new_text if it improves similarity to fixed target

        Args:
            initial_text: Vec2Text output to refine
            initial_z: Normalized latent [0,1]^d from optimization
            X_min, X_max: Denormalization parameters
            verbose: Print progress

        Returns:
            (refined_text, refined_z_normalized, metrics) tuple where:
            - refined_text: Best refined instruction text
            - refined_z_normalized: Corresponding latent in [0,1]^d
            - metrics: Dict with iterations, initial_sim, final_sim, improvement
        """
        best_text = initial_text
        z = initial_z.clone().to(self.device)
        x_range = (X_max - X_min).to(self.device)
        X_min_dev = X_min.to(self.device)
        patience_counter = 0
        iteration = 0

        metrics = {
            "iterations": 0,
            "initial_sim": 0.0,
            "final_sim": 0.0,
            "improvement": 0.0,
        }

        self.vae.eval()

        try:
            # Compute FIXED target = original decoder output
            # This is what we're trying to match with GTR(text)
            vae_dev = self.vae.device
            with torch.no_grad():
                z_unnorm = initial_z.to(self.device) * x_range + X_min_dev
                fixed_target = self.vae.decode(z_unnorm.to(vae_dev))

            # Compute initial similarity: cosine(GTR(initial_text), fixed_target)
            initial_gtr = self.gtr.encode_tensor(initial_text).to(vae_dev)
            initial_sim = F.cosine_similarity(
                initial_gtr.unsqueeze(0),
                fixed_target.unsqueeze(0)
            ).item()
            metrics["initial_sim"] = initial_sim
            best_sim = initial_sim

            if verbose:
                print(f"    ZSInvert: fixed target from decode(z), initial sim = {initial_sim:.4f}")

            for iteration in range(self.n_iterations):
                # 1. Get current best text embedding as optimization target
                # (we optimize z so decode(z) → GTR(best_text))
                current_target = self.gtr.encode_tensor(best_text).to(vae_dev)

                # 2. Gradient-based refinement of z toward current_target
                z_opt = z.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([z_opt], lr=self.lr)

                for step in range(self.n_steps):
                    optimizer.zero_grad()
                    z_unnorm = z_opt * x_range + X_min_dev
                    decoded = self.vae.decode(z_unnorm.to(vae_dev))
                    loss = 1 - F.cosine_similarity(
                        decoded.unsqueeze(0),
                        current_target.unsqueeze(0)
                    )
                    loss.backward()
                    optimizer.step()

                    # Clamp to valid normalized range [0, 1]
                    with torch.no_grad():
                        z_opt.data.clamp_(0, 1)

                # 3. Decode refined latent and invert to text
                z = z_opt.detach()
                with torch.no_grad():
                    z_unnorm = z * x_range + X_min_dev
                    new_emb = self.vae.decode(z_unnorm.to(vae_dev))

                new_text = self.inverter.invert(new_emb.clone())

                # 4. Evaluate improvement against FIXED target
                # Measure: cosine(GTR(new_text), fixed_target)
                # This is consistent - we always compare to the same goal
                new_gtr = self.gtr.encode_tensor(new_text).to(vae_dev)
                new_sim = F.cosine_similarity(
                    new_gtr.unsqueeze(0),
                    fixed_target.unsqueeze(0)
                ).item()

                if verbose:
                    print(f"    ZSInvert iter {iteration + 1}: sim {best_sim:.4f} → {new_sim:.4f}")

                if new_sim > best_sim + self.threshold:
                    best_sim = new_sim
                    best_text = new_text
                    metrics["iterations"] = iteration + 1
                    patience_counter = 0  # Reset patience on improvement
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        # Patience exhausted, stop
                        if verbose:
                            print(f"    ZSInvert: patience exhausted ({self.patience}), stopping")
                        break

        except Exception as e:
            # Log error and return best result so far
            import warnings
            warnings.warn(
                f"ZSInvert refinement failed at iteration {iteration}: {e}. "
                f"Returning best result found before failure."
            )
            metrics["error"] = str(e)
            metrics["iterations"] = iteration

        metrics["final_sim"] = best_sim
        metrics["improvement"] = best_sim - metrics["initial_sim"]

        return best_text, z, metrics


class LIPOHyperbandInference:
    """InvBO inference pipeline for LIPO.

    Pipeline:
        1. Optimize in 32D VAE latent space using LogEI acquisition
           (GP operates directly on 32D latent with ARD kernel)
        2. Decode optimal latent to 768D embedding via VAE decoder
        3. Invert embedding to text via Vec2Text (512_tokens)
        4. Evaluate and add to GP

    Uses 512_tokens Vec2Text model for longer instruction generation.
    Includes KPI tracking for GP quality and optimization gap monitoring.
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
        # VAEWithAdapter for decoding (32D -> 768D)
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
        self._total_retrain_failures: int = 0  # Track total GP retrain failures (never resets)

        # KPI tracking lists for periodic quality reporting
        self._predicted_errors: List[float] = []
        self._actual_errors: List[float] = []
        self._z_gaps: List[float] = []

        # Initialize best from Hyperband results or GP
        if initial_best_instruction is not None and initial_best_error is not None:
            self.best_instruction = initial_best_instruction
            self.best_error = initial_best_error
        elif gp.best_error_rate is not None:
            self.best_error = gp.best_error_rate  # Use property that returns positive error rate
            # Note: instruction not available from GP alone

        # Initialize TuRBO trust region manager
        self.use_turbo = getattr(config, 'turbo_enabled', True)
        self.use_pas = getattr(config, 'pas_enabled', True)

        if self.use_turbo:
            self.trust_region = create_turbo_manager(config, self.device)
        else:
            self.trust_region = None

        if self.use_pas:
            self.anchor_selector = create_pas_selector(config, self.device)
        else:
            self.anchor_selector = None

        # Initialize ZSInvert refiner (only for 512_tokens Vec2Text model)
        # ZSInvert is disabled for 32_tokens model as it's not beneficial
        zsinvert_enabled = getattr(config, 'zsinvert_enabled', True)
        vec2text_model = getattr(config, 'vec2text_model', '512_tokens')
        self.use_zsinvert = zsinvert_enabled and vec2text_model == '512_tokens'

        if zsinvert_enabled and not self.use_zsinvert:
            print("  ZSInvert disabled: only supported with 512_tokens Vec2Text model")

        if self.use_zsinvert:
            self.zsinvert_refiner = ZSInvertRefiner(
                vae_with_adapter=self.vae_with_adapter,
                gtr_encoder=self.gtr,
                inverter=self.inverter,
                n_iterations=getattr(config, 'zsinvert_iterations', 3),
                lr=getattr(config, 'zsinvert_lr', 0.1),
                n_steps_per_iter=getattr(config, 'zsinvert_steps_per_iter', 50),
                improvement_threshold=getattr(config, 'zsinvert_improvement_threshold', 0.01),
                patience=getattr(config, 'zsinvert_patience', 5),
                device=str(self.device),
            )
        else:
            self.zsinvert_refiner = None

        # Cache global bounds (computed on first iteration)
        self._global_bounds: Optional[torch.Tensor] = None

    def _get_global_bounds(self) -> torch.Tensor:
        """Get global latent bounds from training data.

        Computes bounds on first call and caches them.

        Returns:
            Global bounds tensor, shape (2, latent_dim)
        """
        if self._global_bounds is None:
            from lipo.botorch_acq import get_latent_bounds
            self._global_bounds = get_latent_bounds(
                encoder=self.gp.vae_with_adapter,
                X_train=self.gp.X_train,
                X_min=self.gp.X_min,
                X_max=self.gp.X_max,
                margin=self.config.latent_margin,
            )
        return self._global_bounds

    def optimize_latent_botorch(
        self,
        num_restarts: int = 64,
        raw_samples: int = 1024,
        verbose: bool = True,
        seed: Optional[int] = None,
        bounds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize VAE latent using BoTorch qLogExpectedImprovement.

        Uses multi-start L-BFGS-B with proper gradient flow:
            z (32D VAE latent) -> GP posterior -> qLogEI

        GP operates directly on 32D VAE latent (no adapter compression).
        ARD lengthscales allow the kernel to learn which dimensions matter.

        Args:
            num_restarts: Number of L-BFGS-B restarts (default: 64)
            raw_samples: Raw samples for initialization seeding (default: 1024)
            verbose: Print progress
            seed: Optional random seed for reproducibility
            bounds: Optional custom bounds (e.g., trust region bounds). If None, uses global bounds.

        Returns:
            (optimal_latent, log_ei) tuple where:
            - optimal_latent: Best VAE latent tensor, shape (latent_dim,)
            - log_ei: Log expected improvement value at optimal point
        """
        from lipo.botorch_acq import (
            LatentSpaceAcquisition,
            get_latent_bounds,
        )

        if verbose:
            print(f"Optimizing with BoTorch qLogEI ({num_restarts} restarts, {raw_samples} raw samples)...")

        # Get latent bounds (use provided bounds or compute global bounds)
        if bounds is None:
            bounds = get_latent_bounds(
                encoder=self.gp.vae_with_adapter,
                X_train=self.gp.X_train,
                X_min=self.gp.X_min,
                X_max=self.gp.X_max,
                margin=self.config.latent_margin,
            )

        # Create acquisition optimizer
        # Optimization path: z (32D) → GP → kernel (32D ARD) → qLogEI
        # TuRBO trust region constrains search (bounds parameter)
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

        # Warm start: encode target embedding to VAE latent (32D)
        encoder = self.gp.vae_with_adapter
        encoder.eval()

        # Ensure target_emb is on VAE device for gradient computation
        vae_device = self.vae_with_adapter.device
        target_emb = target_emb.to(vae_device)

        with torch.no_grad():
            # Encode 768D embedding to 32D VAE latent (unnormalized)
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

            # Denormalize z to VAE latent space (32D), then decode to 768D embedding
            z_vae = z * x_range + self.gp.X_min
            decoded = self.vae_with_adapter.decode(z_vae.to(self.vae_with_adapter.device))

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
            vae_dev = self.vae_with_adapter.device
            emb_original = self.vae_with_adapter.decode(z_vae_original.to(vae_dev))
            emb_inv = self.vae_with_adapter.decode(z_vae_inv.to(vae_dev))
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

    def _log_iteration_summary(
        self,
        record: IterationRecord,
        pred_std: float = 0.0,
    ) -> None:
        """Print consolidated iteration summary with all metrics.

        Centralizes all iteration metrics in one formatted output block
        for easy monitoring and debugging.

        Args:
            record: IterationRecord with all iteration data
            pred_std: Prediction standard deviation from GP
        """
        # IMPORTANT: Never truncate prompts in log output (per CLAUDE.md)
        instr_display = record.instruction

        actual_str = f"{record.actual_error:.4f}" if record.actual_error is not None else "N/A (skipped)"
        improved_str = "YES" if record.improved else "no"
        log_ei_str = f"{record.log_ei:.4f}" if record.log_ei is not None else "N/A"

        print(f"""
═══════════════════════════════════════════════════════════════
ITERATION {record.iteration} SUMMARY
═══════════════════════════════════════════════════════════════
Instruction: {instr_display}
─────────────────────────────────────────────────────────────────
PERFORMANCE METRICS:
  Predicted Error:    {record.predicted_error:.4f} ± {pred_std:.4f}
  Actual Error:       {actual_str}
  Best Error So Far:  {record.best_error_so_far:.4f}
  Improved:           {improved_str}
─────────────────────────────────────────────────────────────────
OPTIMIZATION GAP METRICS:
  VAE Latent Cosine:  {record.z_opt_z_real_cosine:.4f}
  VAE Latent L2:      {record.z_opt_z_real_euclidean:.4f}
  GP Space Cosine:    {record.z_opt_z_real_gp_cosine:.4f}
  Pred @ z_real:      {record.predicted_error_at_z_real:.4f}
─────────────────────────────────────────────────────────────────
GENERATION QUALITY:
  Cosine Similarity:  {record.cosine_similarity:.4f}
  LogEI:              {log_ei_str}
  Gap (inversion):    {record.gap:.4f}
  Inversion Iters:    {record.inversion_iters}
  Rejection Attempts: {record.rejection_attempts}
  Low Quality Accept: {record.low_quality_accepted}
─────────────────────────────────────────────────────────────────
GP Status:
  Training Samples:   {record.gp_samples}
═══════════════════════════════════════════════════════════════""")

    def compute_inference_kpis(self) -> dict:
        """Compute GP Spearman and System Gap KPIs from tracked data.

        Returns:
            Dictionary with:
            - gp_quality: GP prediction quality metrics
            - system_gap: Optimization gap metrics
        """
        gp_kpi = compute_gp_spearman(self._predicted_errors, self._actual_errors)
        gap_kpi = compute_system_gap(self._z_gaps)
        return {"gp_quality": gp_kpi, "system_gap": gap_kpi}

    def _log_kpi_report(self, iteration: int) -> None:
        """Log periodic KPI report.

        Called every 10 iterations to monitor optimization quality.
        """
        kpis = self.compute_inference_kpis()
        print(f"\n{format_kpi_report(kpis, iteration)}\n")

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

        Candidate Rejection:
            Enforces cosine similarity threshold between decoder(z) and GTR(text)
            to reject misaligned candidates that Vec2Text failed to reconstruct properly.
            - Candidates with cosine_sim < config.cosine_sim_threshold (default 0.90) are rejected
            - Up to config.max_rejection_attempts (default 5) attempts with different seeds
            - If all attempts fail, accepts the best candidate with a WARNING

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
            IterationRecord with results including rejection_attempts and low_quality_accepted
        """
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Get thresholds from config
        cosine_sim_threshold = getattr(self.config, 'cosine_sim_threshold', 0.90)
        max_rejection_attempts = getattr(self.config, 'max_rejection_attempts', 5)

        # Initialize rejection tracking (in case loop doesn't execute)
        rejection_attempts = 0
        low_quality_accepted = False

        # === TuRBO + PAS: Select anchor and compute trust region bounds ===
        anchor_idx = -1
        turbo_action = ""
        trust_region_length = 0.0

        global_bounds = self._get_global_bounds()

        if self.use_turbo and self.trust_region is not None:
            trust_region_length = self.trust_region.state.length

            # Select anchor using PAS (Potential-Aware Selection)
            if self.use_pas and self.anchor_selector is not None:
                try:
                    anchor, anchor_idx = self.anchor_selector.select_anchor(
                        gp_model=self.gp.gp_model,
                        X_train=self.gp.X_train,
                        y_train=self.gp.y_train,
                        X_min=self.gp.X_min,
                        X_max=self.gp.X_max,
                        trust_length=trust_region_length,
                        global_bounds=global_bounds,
                        verbose=verbose,
                    )
                    self.trust_region.set_anchor(anchor)
                except RuntimeError as e:
                    # PAS failed (e.g., GP numerical issues) - fall back to best-y
                    print(f"WARNING: PAS anchor selection failed: {e}")
                    print("  Falling back to best-y anchor selection")
                    best_idx = self.gp.y_train.argmax().item()
                    denom = self.gp.X_max - self.gp.X_min
                    denom[denom == 0] = 1.0
                    anchor = (self.gp.X_train[best_idx] - self.gp.X_min) / denom
                    self.trust_region.set_anchor(anchor)
                    anchor_idx = best_idx
            else:
                # Without PAS, use best observed point as anchor
                best_idx = self.gp.y_train.argmax().item()
                denom = self.gp.X_max - self.gp.X_min
                denom[denom == 0] = 1.0
                anchor = (self.gp.X_train[best_idx] - self.gp.X_min) / denom
                self.trust_region.set_anchor(anchor)
                anchor_idx = best_idx

            # Get ARD lengthscales from GP kernel for LOL-BO style scaling
            # Lengthscales are in 32D VAE space (no adapter) - each dimension gets its own lengthscale
            lengthscales = None
            try:
                if hasattr(self.gp.gp_model, 'covar_module'):
                    base_kernel = self.gp.gp_model.covar_module.base_kernel
                    if hasattr(base_kernel, 'lengthscale'):
                        lengthscales = base_kernel.lengthscale.detach().squeeze()
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not extract ARD lengthscales from GP kernel: {e}. "
                    f"Falling back to uniform trust region scaling."
                )

            # Get ARD-aware trust region bounds (LOL-BO style)
            bounds = self.trust_region.get_ard_bounds(global_bounds, lengthscales)

            if verbose:
                print(f"  TuRBO: {self.trust_region.get_state_summary()}")
                if lengthscales is not None:
                    ls_str = ", ".join([f"{ls:.3f}" for ls in lengthscales[:5].tolist()])
                    print(f"  ARD lengthscales (first 5): [{ls_str}, ...]")
                print(f"  Anchor: idx={anchor_idx}, using {'PAS' if self.use_pas else 'best-y'} selection")
        else:
            # Global optimization (no trust region)
            bounds = global_bounds

        # Main loop with rejection for low cosine similarity
        for attempt in range(max_rejection_attempts):
            # Optimize latent using BoTorch qLogEI (within trust region if enabled)
            # Use different seed for each attempt to get different candidates
            attempt_seed = iteration * max_rejection_attempts + attempt
            z_opt, log_ei = self.optimize_latent_botorch(
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                verbose=verbose and (attempt == 0),  # Only verbose on first attempt
                seed=attempt_seed,
                bounds=bounds,
            )

            # Denormalize and decode to embedding (16D latent -> 768D embedding)
            # z_opt is normalized [0,1], need to convert back to VAE latent space
            x_range = self.gp.X_max - self.gp.X_min
            z_unnorm = z_opt * x_range + self.gp.X_min

            # Save original z_opt for optimization gap measurement
            z_opt_original = z_opt.clone()

            # Decode to embedding
            self.vae_with_adapter.eval()
            with torch.no_grad():
                z_decode = z_unnorm.to(self.vae_with_adapter.device)
                embedding = self.vae_with_adapter.decode(z_decode)

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
                        z_decode = z_unnorm.to(self.vae_with_adapter.device)
                        embedding = self.vae_with_adapter.decode(z_decode)
                    instruction = self.inverter.invert(embedding.clone())
                    inv_iters += 1

                    if verbose:
                        print(f"  Re-generated:\n{instruction}")

            # ZSInvert refinement - runs EVERY iteration after Vec2Text
            # Improves text-embedding alignment via gradient-based latent optimization
            if self.use_zsinvert and self.zsinvert_refiner is not None:
                instruction, z_refined, zsinvert_metrics = self.zsinvert_refiner.refine(
                    initial_text=instruction,
                    initial_z=z_opt,
                    X_min=self.gp.X_min,
                    X_max=self.gp.X_max,
                    verbose=verbose,
                )
                # Update z_opt if refinement occurred
                if zsinvert_metrics["iterations"] > 0:
                    z_opt = z_refined
                    # Re-decode to get updated embedding
                    z_unnorm = z_opt * x_range + self.gp.X_min
                    with torch.no_grad():
                        z_decode = z_unnorm.to(self.vae_with_adapter.device)
                        embedding = self.vae_with_adapter.decode(z_decode)

                if verbose:
                    print(f"  ZSInvert: {zsinvert_metrics['iterations']} iters, "
                          f"sim: {zsinvert_metrics['initial_sim']:.4f} → {zsinvert_metrics['final_sim']:.4f} "
                          f"(Δ={zsinvert_metrics['improvement']:+.4f})")

            # Re-encode for GP prediction
            # IMPORTANT: Predict on GTR(text), not decoder(z), for alignment with training data
            reencoded = self.gtr.encode_tensor(instruction)
            cosine_sim = F.cosine_similarity(
                embedding.unsqueeze(0), reencoded.unsqueeze(0)
            ).item()

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

            # Gap metrics in VAE latent space (32D) - this is also GP space (no adapter)
            z_opt_z_real_cosine = F.cosine_similarity(
                z_opt_original.unsqueeze(0), z_real_norm.unsqueeze(0)
            ).item()
            z_opt_z_real_euclidean = torch.dist(z_opt_original, z_real_norm).item()

            # GP space is now same as VAE latent space (no adapter compression)
            z_opt_z_real_gp_cosine = z_opt_z_real_cosine  # Same metric, no adapter

            # GP prediction at z_real (actual text) vs at z_opt (dream)
            # This shows if GP was "fooled" by holes in latent space
            pred_error_at_z_real, _ = self.gp.predict(reencoded)

        # Critical warning for large optimization gap (not in summary)
        if z_opt_z_real_cosine < 0.85:
            print(f"  WARNING: Large optimization gap (VAE cosine={z_opt_z_real_cosine:.4f})! "
                  f"GP may be optimizing in empty space.")

        # Evaluate with LLM (or use GP prediction)
        actual_error = None
        if not skip_eval and self.evaluator is not None and self.validation_data is not None:
            prompt = InstructionOnlyPrompt(instruction=instruction, instruction_id=-1)
            actual_error = self.evaluator(prompt, self.validation_data)
            self.total_llm_calls += len(self.validation_data)

        # Update best
        error_to_use = actual_error if actual_error is not None else pred_error
        improved = error_to_use < self.best_error
        if improved:
            self.best_error = error_to_use
            self.best_instruction = instruction

        # === TuRBO: Update trust region state based on iteration result ===
        if self.use_turbo and self.trust_region is not None:
            turbo_info = self.trust_region.update(improved=improved)
            turbo_action = turbo_info["action"]
            trust_region_length = turbo_info["length_after"]

            if verbose and turbo_action != "none":
                if turbo_action == "expand":
                    print(f"  TuRBO: EXPANDED trust region to L={trust_region_length:.4f}")
                elif turbo_action == "shrink":
                    print(f"  TuRBO: SHRUNK trust region to L={trust_region_length:.4f}")
                elif turbo_action == "restart":
                    print(f"  TuRBO: RESTARTED trust region (restart #{turbo_info['restart_count']})")

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
            self._total_retrain_failures += 1
            print(f"ERROR: GP retraining failed at iteration {iteration}")
            print(f"  Consecutive failures: {self._consecutive_retrain_failures}")
            print(f"  Total failures: {self._total_retrain_failures}")
            print(f"  Training samples: {self.gp.get_training_size()}")

            if self._consecutive_retrain_failures >= 3:
                raise RuntimeError(
                    f"GP retraining failed {self._consecutive_retrain_failures} consecutive times. "
                    f"This indicates a systematic problem with the training data. "
                    f"Possible causes: duplicate observations, numerical overflow, or ill-conditioned kernel."
                )
            if self._total_retrain_failures >= 5:
                print(f"  WARNING: {self._total_retrain_failures} total GP retrain failures - results may be unreliable")
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
            # TuRBO trust region state
            trust_region_length=trust_region_length,
            trust_region_action=turbo_action,
            anchor_idx=anchor_idx,
        )

        self.iteration_history.append(record)

        # Track KPI data
        self._predicted_errors.append(pred_error)
        self._actual_errors.append(actual_error)
        self._z_gaps.append(z_opt_z_real_euclidean)

        # Print consolidated iteration summary
        if verbose:
            self._log_iteration_summary(record, pred_std=pred_std)

        # Periodic KPI report every 10 iterations
        if iteration % 10 == 0 and iteration > 0:
            self._log_kpi_report(iteration)

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
        # Ensure device consistency for cross-device scenarios
        self.vae_with_adapter.eval()
        vae_dev = self.vae_with_adapter.device
        with torch.no_grad():
            emb_for_vae = emb_original.to(vae_dev)
            z = self.vae_with_adapter.encode_vae(emb_for_vae.unsqueeze(0))
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
