"""Inference pipeline for InvBO decoder inversion.

Provides:
- EI optimization in 10D latent space via L-BFGS-B
- Vec2Text inversion from 768D embedding to text
- Complete pipeline from latent to novel instruction
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, List, Tuple

from generation.invbo_decoder.encoder import GTRInstructionEncoder, InstructionVAE, InstructionFeatureExtractor
from generation.invbo_decoder.gp import GPWithEI
from generation.invbo_decoder.decoder import LatentDecoder
from generation.invbo_decoder.trust_region import TrustRegionManager, TRConfig


@dataclass
class IterationRecord:
    """Record of a single optimization iteration."""

    iteration: int
    instruction: str
    cosine_similarity: float
    predicted_error: float
    actual_error: Optional[float]  # None if skip_eval mode
    gap: float
    improved: bool
    best_error_so_far: float
    trust_region_radius: Optional[float]
    gp_samples: int
    log_ei: Optional[float] = None
    inversion_iters: int = 1


@dataclass
class InversionStepResult:
    """Result of InvBO inversion step."""

    z_inv: torch.Tensor  # Inverted latent
    z_original: torch.Tensor  # Original latent before inversion
    gap: float  # ||z_original - z_inv||
    final_loss: float  # Final cosine loss
    converged: bool  # Whether optimization converged


@dataclass
class InversionResult:
    """Result of latent-to-text inversion."""

    instruction_text: str
    latent: torch.Tensor
    embedding: torch.Tensor
    cosine_similarity: float  # Between decoded and re-encoded embedding
    predicted_error: float  # GP prediction for this latent
    ei_value: float  # Expected improvement at this latent


class Vec2TextInverter:
    """Vec2Text embedding-to-text inverter.

    Supports two model types:
    - "32_tokens": ielabgroup/vec2text_gtr-base-st_* with corrector (default, 32 token limit)
    - "512_tokens": vec2text/gtr-512-noise-0.00001 without corrector (512 token limit)

    Lazily loads models on first use.
    """

    def __init__(
        self,
        num_steps: int = 50,
        beam_width: int = 4,
        max_length: int = 128,
        device: str = "auto",
        model_type: str = "32_tokens",
    ):
        """Initialize inverter.

        Args:
            num_steps: Correction iterations (for 32_tokens) or max_new_tokens (for 512_tokens)
            beam_width: Beam search width
            max_length: Maximum output length
            device: Device to use
            model_type: "32_tokens" (with corrector) or "512_tokens" (InversionModel only)
        """
        if model_type not in ("32_tokens", "512_tokens"):
            raise ValueError(f"model_type must be '32_tokens' or '512_tokens', got '{model_type}'")

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
        """Lazy load Vec2Text model based on model_type."""
        if self.model_type == "32_tokens":
            self._load_32_tokens()
        else:
            self._load_512_tokens()

    def _load_32_tokens(self):
        """Load ielabgroup Vec2Text corrector (with InversionModel + CorrectorEncoderModel, 32 token limit)."""
        if self._corrector is not None:
            return

        import vec2text
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel
        from vec2text.models.corrector_encoder import CorrectorEncoderModel

        print("Loading Vec2Text models (32_tokens: ielabgroup with corrector)...")

        # Load InversionModel
        inv_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
        )
        inv_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )
        inversion_model = InversionModel(inv_config)
        inversion_model.load_state_dict(load_file(inv_weights), strict=False)
        inversion_model = inversion_model.to(self.device).eval()

        # Load CorrectorEncoderModel
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
        """Load Vec2Text InversionModel (without corrector, 512 token limit)."""
        if self._inversion_model is not None:
            return

        import os
        import json
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        print("Loading Vec2Text InversionModel (512_tokens: gtr-512-noise-0.00001)...")

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
        """Invert using ielabgroup corrector (32 token limit)."""
        import vec2text

        result = vec2text.invert_embeddings(
            embeddings=embedding,
            corrector=self._corrector,
            num_steps=self.num_steps,
            sequence_beam_width=self.beam_width,
        )

        return result[0] if isinstance(result, list) else result

    def _invert_512_tokens(self, embedding: torch.Tensor) -> str:
        """Invert using InversionModel direct generation (512 token limit)."""
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


class InvBOInference:
    """Complete inference pipeline for InvBO decoder inversion.

    Pipeline:
        1. Optimize in 10D latent space using EI acquisition
        2. Decode optimal latent to 768D embedding
        3. Invert embedding to text via Vec2Text
        4. Validate by re-encoding and checking cosine similarity
    """

    def __init__(
        self,
        gp: GPWithEI,
        decoder: LatentDecoder,
        gtr: Optional[GTRInstructionEncoder] = None,
        vec2text_steps: int = 50,
        vec2text_beam: int = 4,
        vec2text_model: str = "ielabgroup",
        trust_region_config: Optional[TRConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize inference pipeline.

        Args:
            gp: Trained GP with feature extractor
            decoder: Trained latent decoder
            gtr: GTR encoder (for validation)
            vec2text_steps: Vec2Text correction steps
            vec2text_beam: Vec2Text beam width
            vec2text_model: "ielabgroup" (with corrector) or "cowboys" (simpler)
            trust_region_config: Optional trust region configuration
            seed: Random seed for reproducible optimization
        """
        self.gp = gp
        self.decoder = decoder
        self.device = gp.device
        self.seed = seed

        self.gtr = gtr if gtr is not None else GTRInstructionEncoder(device=str(self.device))
        self.inverter = Vec2TextInverter(
            num_steps=vec2text_steps,
            beam_width=vec2text_beam,
            device=str(self.device),
            model_type=vec2text_model,
        )

        # Trust region (initialized lazily)
        self.trust_region: Optional[TrustRegionManager] = None
        self.trust_region_config = trust_region_config

    def init_trust_region(self, anchor_latent: torch.Tensor) -> TrustRegionManager:
        """Initialize trust region around anchor latent.

        Args:
            anchor_latent: Best known latent (10,)

        Returns:
            Initialized TrustRegionManager
        """
        config = self.trust_region_config or TRConfig()
        self.trust_region = TrustRegionManager(
            anchor=anchor_latent,
            config=config,
            device=str(self.device),
        )
        return self.trust_region

    def get_best_training_latent(self) -> Tuple[torch.Tensor, int, float]:
        """Get latent of best training sample (lowest error).

        Returns:
            (latent, index, error_rate) tuple
        """
        best_idx = self.gp.y_train.argmin().item()
        best_error = self.gp.y_train[best_idx].item()

        # Get latent for best embedding
        best_embedding = self.gp.X_train[best_idx]
        best_latent = self.gp.get_latent(best_embedding)

        return best_latent, best_idx, best_error

    def _get_latent_bounds(self, margin: float = 0.2) -> torch.Tensor:
        """Compute latent space bounds from training data.

        Args:
            margin: Fraction to expand bounds beyond training data range

        Returns:
            Bounds tensor, shape (2, latent_dim)
        """
        from generation.invbo_decoder.botorch_acq import get_latent_bounds

        return get_latent_bounds(
            encoder=self.gp.feature_extractor,
            X_train=self.gp.X_train,
            X_min=self.gp.X_min,
            X_max=self.gp.X_max,
            margin=margin,
        )

    def optimize_latent_botorch(
        self,
        num_restarts: int = 20,
        raw_samples: int = 512,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize latent using BoTorch qLogExpectedImprovement.

        Uses multi-start L-BFGS-B with proper gradient flow through:
            latent z -> decoder -> embedding -> GP posterior -> qLogEI

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
        from generation.invbo_decoder.botorch_acq import LatentSpaceAcquisition

        if verbose:
            print(f"Optimizing with BoTorch qLogEI ({num_restarts} restarts, {raw_samples} raw samples)...")

        # Get latent bounds from training data
        bounds = self._get_latent_bounds(margin=0.2)

        # Create acquisition optimizer
        acq_optimizer = LatentSpaceAcquisition(
            gp_model=self.gp.gp_model,
            decoder=self.decoder,
            bounds=bounds,
            device=self.device,
        )

        # Optimize - best_f is best observed error rate (we minimize error)
        best_f = self.gp.y_best
        z_opt, log_ei = acq_optimizer.optimize(
            best_f=best_f,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            seed=self.seed,
        )

        if verbose:
            print(f"  BoTorch LogEI: {log_ei.item():.4f}")

        # Return as 1D tensor
        return z_opt.squeeze(), log_ei.item()

    def optimize_latent_adaptive(
        self,
        n_candidates: int = 500,
        xi: float = 0.01,
        use_trust_region: bool = True,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize latent using adaptive trust region sampling.

        Combines trust region constraints with EI optimization.
        Best for iterative optimization as it respects trust region bounds.

        Args:
            n_candidates: Number of candidates to sample
            xi: EI exploration parameter
            use_trust_region: Whether to use trust region constraints
            verbose: Print progress

        Returns:
            (optimal_latent, log_ei) tuple
        """
        if use_trust_region and self.trust_region is None:
            # Initialize trust region from best training sample
            best_latent, _, _ = self.get_best_training_latent()
            self.init_trust_region(best_latent)

        if verbose:
            if use_trust_region:
                print(f"Sampling {n_candidates} candidates in trust region (radius={self.trust_region.radius:.4f})...")
            else:
                print(f"Sampling {n_candidates} candidates globally...")

        best_z = None
        best_log_ei = -float("inf")

        self.decoder.eval()

        for i in range(n_candidates):
            if use_trust_region:
                z = self.trust_region.sample_in_region(n_samples=1)
            else:
                z = self.sample_in_trust_region(n_samples=1, radius=0.3)

            with torch.no_grad():
                embedding = self.decoder(z)

            log_ei = self.gp.log_expected_improvement(embedding, xi=xi)

            if log_ei > best_log_ei:
                best_log_ei = log_ei
                best_z = z

        if verbose:
            print(f"  Best LogEI: {best_log_ei:.4f}")

        return best_z, best_log_ei

    def run_single_iteration(
        self,
        num_restarts: int = 20,
        raw_samples: int = 512,
        use_inversion: bool = True,
        max_inversion_iters: int = 3,
        gap_threshold: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[InversionResult, float, float]:
        """Run a single optimization iteration using BoTorch qLogEI.

        Uses BoTorch's gradient-based LogEI optimization with multi-start
        L-BFGS-B for finding optimal latent points.

        Args:
            num_restarts: Number of L-BFGS-B restarts for BoTorch optimization
            raw_samples: Raw samples for initialization seeding
            use_inversion: Use InvBO inversion loop
            max_inversion_iters: Maximum inversion iterations
            gap_threshold: Threshold for re-inversion
            verbose: Print progress

        Returns:
            (result, gap, log_ei) tuple where:
            - result: InversionResult with instruction and metrics
            - gap: Cosine gap between z_opt and z_inv embeddings
            - log_ei: Log expected improvement at z_opt
        """
        # Optimize latent using BoTorch qLogEI
        z_opt, log_ei = self.optimize_latent_botorch(
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            verbose=verbose,
        )

        # Decode to embedding
        self.decoder.eval()
        with torch.no_grad():
            embedding = self.decoder(z_opt)

        # Invert to text
        text = self.inverter.invert(embedding.clone())

        if verbose:
            print(f"  Generated:\n{text}")

        # Inversion loop if enabled
        gap = 0.0
        inv_iters = 1
        if use_inversion:
            for inv_iter in range(max_inversion_iters):
                inv_result = self.inversion_step(text, n_steps=100, lr=0.1, verbose=False)
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
                    embedding = self.decoder(z_opt)
                text = self.inverter.invert(embedding.clone())
                inv_iters += 1

                if verbose:
                    print(f"  Re-generated:\n{text}")

        # Re-encode for validation
        reencoded = self.gtr.encode_tensor(text)
        cosine_sim = F.cosine_similarity(
            embedding.unsqueeze(0), reencoded.unsqueeze(0)
        ).item()

        if verbose:
            print(f"  Cosine similarity: {cosine_sim:.4f}")

        # GP prediction
        pred_mean, pred_std = self.gp.predict(embedding)
        ei = self.gp.expected_improvement(embedding)

        result = InversionResult(
            instruction_text=text,
            latent=z_opt,
            embedding=embedding,
            cosine_similarity=cosine_sim,
            predicted_error=pred_mean,
            ei_value=ei,
        )

        return result, gap, log_ei

    def optimize_latent_lbfgs(
        self,
        n_restarts: int = 10,
        max_iter: int = 100,
        xi: float = 0.01,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Find optimal latent using L-BFGS-B with EI acquisition.

        Optimizes Expected Improvement in the 10D latent space.

        Args:
            n_restarts: Number of random restarts
            max_iter: Max iterations per restart
            xi: EI exploration parameter
            verbose: Print progress

        Returns:
            Optimal 10D latent tensor
        """
        if verbose:
            print("Optimizing latent space with L-BFGS-B...")

        # Get bounds from training data latents
        encoder = self.gp.feature_extractor
        encoder.eval()

        # Compute latent bounds from training embeddings
        all_latents = []
        with torch.no_grad():
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            X_norm = (self.gp.X_train - self.gp.X_min) / denom
            all_latents = encoder(X_norm)

        z_min = all_latents.min(dim=0)[0].cpu().numpy()
        z_max = all_latents.max(dim=0)[0].cpu().numpy()
        bounds = [(z_min[i], z_max[i]) for i in range(len(z_min))]

        # Expand bounds slightly for exploration
        for i in range(len(bounds)):
            margin = 0.1 * (bounds[i][1] - bounds[i][0])
            bounds[i] = (bounds[i][0] - margin, bounds[i][1] + margin)

        def neg_ei(z_np):
            """Negative EI for minimization."""
            z = torch.tensor(z_np, dtype=torch.float32, device=self.device)

            # Decode latent to embedding
            self.decoder.eval()
            with torch.no_grad():
                embedding = self.decoder(z)

            # Compute EI
            ei = self.gp.expected_improvement(embedding, xi=xi)
            return -ei

        best_z = None
        best_ei = -float("inf")

        for restart in range(n_restarts):
            # Random starting point within bounds
            z0 = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
            )

            try:
                result = minimize(
                    neg_ei,
                    z0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iter},
                )

                ei = -result.fun
                if ei > best_ei:
                    best_ei = ei
                    best_z = result.x

            except Exception as e:
                if verbose:
                    print(f"  Restart {restart + 1} failed: {e}")
                continue

        if best_z is None:
            raise RuntimeError("All optimization restarts failed")

        if verbose:
            print(f"  Best EI: {best_ei:.6f}")

        return torch.tensor(best_z, dtype=torch.float32, device=self.device)

    def optimize_latent_random(
        self,
        n_candidates: int = 1000,
        xi: float = 0.01,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Find optimal latent using random sampling.

        Faster alternative to L-BFGS-B for quick exploration.

        Args:
            n_candidates: Number of random candidates
            xi: EI exploration parameter
            verbose: Print progress

        Returns:
            Best 10D latent tensor
        """
        if verbose:
            print(f"Sampling {n_candidates} latent candidates...")

        encoder = self.gp.feature_extractor
        encoder.eval()

        # Get latent distribution from training data
        with torch.no_grad():
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            X_norm = (self.gp.X_train - self.gp.X_min) / denom
            all_latents = encoder(X_norm)

        z_mean = all_latents.mean(dim=0)
        z_std = all_latents.std(dim=0) + 1e-6

        best_z = None
        best_ei = -float("inf")

        self.decoder.eval()

        for i in range(n_candidates):
            # Sample from Gaussian around observed latents
            z = z_mean + z_std * torch.randn_like(z_mean) * 1.5

            with torch.no_grad():
                embedding = self.decoder(z)

            ei = self.gp.expected_improvement(embedding, xi=xi)

            if ei > best_ei:
                best_ei = ei
                best_z = z

        if verbose:
            print(f"  Best EI: {best_ei:.6f}")

        return best_z

    def invert_latent(
        self,
        latent: torch.Tensor,
        validate: bool = True,
        verbose: bool = True,
    ) -> InversionResult:
        """Invert latent to instruction text.

        Pipeline:
            latent (10D) -> decoder -> embedding (768D) -> Vec2Text -> text

        Args:
            latent: 10D latent tensor
            validate: Compute cosine similarity with re-encoded text
            verbose: Print progress

        Returns:
            InversionResult with text and metrics
        """
        if verbose:
            print("Inverting latent to text...")

        # Decode latent to embedding
        self.decoder.eval()
        with torch.no_grad():
            embedding = self.decoder(latent)

        # Invert to text
        instruction_text = self.inverter.invert(embedding)

        if verbose:
            print(f"  Generated: {instruction_text}")

        # Validate by re-encoding
        cosine_sim = 0.0
        if validate:
            reencoded = self.gtr.encode_tensor(instruction_text)
            cosine_sim = F.cosine_similarity(
                embedding.unsqueeze(0), reencoded.unsqueeze(0)
            ).item()

            if verbose:
                print(f"  Cosine similarity: {cosine_sim:.4f}")

        # Get GP prediction
        pred_mean, pred_std = self.gp.predict(embedding)
        ei = self.gp.expected_improvement(embedding)

        return InversionResult(
            instruction_text=instruction_text,
            latent=latent,
            embedding=embedding,
            cosine_similarity=cosine_sim,
            predicted_error=pred_mean,
            ei_value=ei,
        )

    def run_optimization(
        self,
        method: str = "lbfgs",
        n_restarts: int = 10,
        n_candidates: int = 1000,
        trust_radius: float = 0.3,
        xi: float = 0.01,
        verbose: bool = True,
    ) -> InversionResult:
        """Run complete optimization pipeline.

        Args:
            method: "lbfgs", "random", or "trust_region"
            n_restarts: L-BFGS restarts
            n_candidates: Random/trust-region sampling candidates
            trust_radius: Perturbation radius for trust_region method
            xi: EI exploration parameter
            verbose: Print progress

        Returns:
            InversionResult with novel instruction
        """
        if verbose:
            print("\n" + "=" * 60)
            print("InvBO Optimization Pipeline")
            print("=" * 60)
            print(f"Best observed error: {self.gp.y_best:.4f}")

        # Optimize latent
        if method == "lbfgs":
            optimal_latent = self.optimize_latent_lbfgs(
                n_restarts=n_restarts, xi=xi, verbose=verbose
            )
        elif method == "trust_region":
            optimal_latent = self.optimize_latent_trust_region(
                n_candidates=n_candidates, radius=trust_radius, xi=xi, verbose=verbose
            )
        else:
            optimal_latent = self.optimize_latent_random(
                n_candidates=n_candidates, xi=xi, verbose=verbose
            )

        # Invert to text
        result = self.invert_latent(optimal_latent, validate=True, verbose=verbose)

        if verbose:
            print("\n" + "-" * 40)
            print("Optimization Results:")
            print(f"  Novel instruction: {result.instruction_text}")
            print(f"  Predicted error: {result.predicted_error:.4f}")
            print(f"  EI value: {result.ei_value:.6f}")
            print(f"  Cosine similarity: {result.cosine_similarity:.4f}")
            print("-" * 40)

        return result

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

            # Decode latent to embedding (gradients flow through decoder in any mode)
            decoded = self.decoder(z)

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

        # Gap jako cosine distance v embedding space (stabilnější než L2 v latentu)
        # L2 v latentu může být vysoká i když embeddingy jsou podobné
        with torch.no_grad():
            self.decoder.eval()
            emb_original = self.decoder(z_original)
            emb_inv = self.decoder(z_inv)
        cosine_gap = 1 - F.cosine_similarity(
            emb_original.unsqueeze(0), emb_inv.unsqueeze(0)
        ).item()
        gap = cosine_gap  # Nyní v rozsahu [0, 2] místo [0, ∞)

        if verbose:
            print(f"  Inversion: gap = {gap:.4f}, loss = {final_loss:.4f}, converged = {converged}")

        return InversionStepResult(
            z_inv=z_inv,
            z_original=z_original,
            gap=gap,
            final_loss=final_loss,
            converged=converged,
        )

    def optimize_with_inversion(
        self,
        method: str = "lbfgs",
        n_restarts: int = 10,
        n_candidates: int = 1000,
        trust_radius: float = 0.3,
        xi: float = 0.01,
        max_inversion_iters: int = 3,
        gap_threshold: float = 0.1,  # Cosine distance threshold (bylo 0.5 pro L2)
        verbose: bool = True,
    ) -> InversionResult:
        """Run optimization with InvBO-style inversion loop.

        Pipeline:
            1. z* = argmax EI(z)                    # Find optimal latent
            2. embedding* = decoder(z*)             # Decode to 768D
            3. text* = Vec2Text(embedding*)         # Invert to text
            4. z_inv = argmin ||decoder(z) - GTR(text*)||²  # INVERSION STEP
            5. If ||z* - z_inv|| > threshold:
                 z* = z_inv                         # Use inverted latent
                 goto 2

        Args:
            method: "lbfgs", "random", or "trust_region"
            n_restarts: L-BFGS restarts
            n_candidates: Random/trust-region sampling candidates
            trust_radius: Perturbation radius for trust_region method
            xi: EI exploration parameter
            max_inversion_iters: Maximum inversion iterations
            gap_threshold: Threshold for re-inversion
            verbose: Print progress

        Returns:
            InversionResult with novel instruction
        """
        if verbose:
            print("\n" + "=" * 60)
            print("InvBO Optimization with Inversion")
            print("=" * 60)
            print(f"Best observed error: {self.gp.y_best:.4f}")

        # Initial EI optimization
        if method == "lbfgs":
            z_star = self.optimize_latent_lbfgs(
                n_restarts=n_restarts, xi=xi, verbose=verbose
            )
        elif method == "trust_region":
            z_star = self.optimize_latent_trust_region(
                n_candidates=n_candidates, radius=trust_radius, xi=xi, verbose=verbose
            )
        else:
            z_star = self.optimize_latent_random(
                n_candidates=n_candidates, xi=xi, verbose=verbose
            )

        # Inversion loop
        for inv_iter in range(max_inversion_iters):
            if verbose:
                print(f"\nInversion iteration {inv_iter + 1}/{max_inversion_iters}")

            # Decode to embedding
            self.decoder.eval()
            with torch.no_grad():
                embedding = self.decoder(z_star)

            # Invert to text
            text = self.inverter.invert(embedding.clone())

            if verbose:
                print(f"  Generated: {text}")

            # Inversion step
            inv_result = self.inversion_step(
                text,
                n_steps=100,
                lr=0.1,
                verbose=verbose,
            )

            if inv_result.gap <= gap_threshold:
                if verbose:
                    print(f"  Gap {inv_result.gap:.4f} <= {gap_threshold}, accepting")
                break

            if verbose:
                print(f"  Gap {inv_result.gap:.4f} > {gap_threshold}, using z_inv")

            # Use inverted latent for next iteration
            z_star = inv_result.z_inv

        # Final result
        result = self.invert_latent(z_star, validate=True, verbose=verbose)

        if verbose:
            print("\n" + "-" * 40)
            print("Optimization with Inversion Results:")
            print(f"  Novel instruction: {result.instruction_text}")
            print(f"  Predicted error: {result.predicted_error:.4f}")
            print(f"  EI value: {result.ei_value:.6f}")
            print(f"  Cosine similarity: {result.cosine_similarity:.4f}")
            print(f"  Inversion iterations: {inv_iter + 1}")
            print("-" * 40)

        return result

    def sample_in_trust_region(
        self,
        n_samples: int = 1,
        radius: float = 0.3,
    ) -> torch.Tensor:
        """Sample latents within trust region of known training latents.

        Uses convex combinations of known latents plus small perturbations.
        This ensures samples stay near the decoder's training distribution.

        Args:
            n_samples: Number of samples to generate
            radius: Perturbation radius (as fraction of latent std)

        Returns:
            Tensor of shape (n_samples, latent_dim) with trust-region samples
        """
        encoder = self.gp.feature_extractor
        encoder.eval()

        # Get training latents
        with torch.no_grad():
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            X_norm = (self.gp.X_train - self.gp.X_min) / denom
            anchor_latents = encoder(X_norm)

        n_anchors = len(anchor_latents)
        z_std = anchor_latents.std(dim=0)

        samples = []
        for _ in range(n_samples):
            # Random convex combination weights
            weights = torch.rand(n_anchors, device=self.device)
            weights = weights / weights.sum()

            # Convex combination of anchors
            center = (weights.unsqueeze(-1) * anchor_latents).sum(0)

            # Small perturbation
            noise = torch.randn_like(center) * z_std * radius
            sample = center + noise

            samples.append(sample)

        return torch.stack(samples) if n_samples > 1 else samples[0]

    def optimize_latent_trust_region(
        self,
        n_candidates: int = 500,
        radius: float = 0.3,
        xi: float = 0.01,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Find optimal latent using trust-region constrained sampling.

        Faster and safer alternative to random sampling - stays near known latents.

        Args:
            n_candidates: Number of trust-region candidates
            radius: Perturbation radius for trust region
            xi: EI exploration parameter
            verbose: Print progress

        Returns:
            Best 10D latent tensor
        """
        if verbose:
            print(f"Sampling {n_candidates} trust-region candidates (radius={radius})...")

        best_z = None
        best_ei = -float("inf")

        self.decoder.eval()

        for i in range(n_candidates):
            z = self.sample_in_trust_region(n_samples=1, radius=radius)

            with torch.no_grad():
                embedding = self.decoder(z)

            ei = self.gp.expected_improvement(embedding, xi=xi)

            if ei > best_ei:
                best_ei = ei
                best_z = z

        if verbose:
            print(f"  Best EI: {best_ei:.6f}")

        return best_z

    def validate_inversion_gap(
        self,
        n_samples: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Measure inversion gap on random latent samples.

        The inversion gap is ||z - encoder(GTR(Vec2Text(decoder(z))))||.
        This measures the full cycle: latent -> text -> latent.

        Args:
            n_samples: Number of samples to test
            verbose: Print progress

        Returns:
            Dictionary with gap statistics
        """
        if verbose:
            print("\nMeasuring inversion gap...")

        encoder = self.gp.feature_extractor
        encoder.eval()
        self.decoder.eval()

        # Get latent distribution
        with torch.no_grad():
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            X_norm = (self.gp.X_train - self.gp.X_min) / denom
            all_latents = encoder(X_norm)

        z_mean = all_latents.mean(dim=0)
        z_std = all_latents.std(dim=0) + 1e-6

        gaps = []
        cosines = []

        for i in range(n_samples):
            # Sample latent
            z = z_mean + z_std * torch.randn_like(z_mean)

            # Full cycle: z -> decode -> Vec2Text -> GTR -> encode
            with torch.no_grad():
                embedding = self.decoder(z)

            text = self.inverter.invert(embedding)
            reencoded = self.gtr.encode_tensor(text)

            # Re-encode to latent
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            reencoded_norm = (reencoded - self.gp.X_min) / denom

            with torch.no_grad():
                z_recon = encoder(reencoded_norm.unsqueeze(0)).squeeze(0)

            gap = torch.norm(z - z_recon).item()
            cosine = F.cosine_similarity(
                embedding.unsqueeze(0), reencoded.unsqueeze(0)
            ).item()

            gaps.append(gap)
            cosines.append(cosine)

            if verbose:
                print(f"  Sample {i + 1}: gap={gap:.4f}, cosine={cosine:.4f}")
                print(f"    Text: {text}")

        stats = {
            "mean_gap": np.mean(gaps),
            "std_gap": np.std(gaps),
            "min_gap": np.min(gaps),
            "max_gap": np.max(gaps),
            "mean_cosine": np.mean(cosines),
            "std_cosine": np.std(cosines),
        }

        if verbose:
            print("\nInversion Gap Statistics:")
            print(f"  Mean gap: {stats['mean_gap']:.4f} +/- {stats['std_gap']:.4f}")
            print(f"  Mean cosine: {stats['mean_cosine']:.4f} +/- {stats['std_cosine']:.4f}")

        return stats
