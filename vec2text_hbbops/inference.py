"""Inference pipeline for Vec2Text-integrated HbBoPs.

Provides complete pipeline from EI optimization in latent space
to text generation via Vec2Text inversion.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from vec2text_hbbops.hbbops_vec2text import HbBoPsVec2Text, Prompt
from vec2text_hbbops.encoder import GTRPromptEncoder


@dataclass
class InversionResult:
    """Result of latent-to-text inversion."""

    instruction_text: str
    exemplar_text: str
    instruction_cosine: float
    exemplar_cosine: float
    latent: torch.Tensor
    evaluated_error: float = None  # Error rate on validation set (if evaluated)


@dataclass
class OptimizationResult:
    """Complete optimization result."""

    # Best from discrete grid
    best_from_grid: Optional[Prompt]
    best_grid_error: float

    # Reconstructed version of best
    best_reconstructed: Optional[InversionResult]

    # Novel prompt from latent optimization
    novel_from_latent: Optional[InversionResult]

    # Metadata
    num_evaluations: int
    design_data_size: int


class Vec2TextInverter:
    """Vec2Text embedding-to-text inverter.

    Uses ielabgroup/vec2text_gtr-base-st_* models for inversion.
    """

    def __init__(
        self,
        num_steps: int = 50,
        beam_width: int = 4,
        device: str = "auto",
    ):
        """Initialize Vec2Text inverter.

        Args:
            num_steps: Number of correction iterations
            beam_width: Beam search width
            device: Device to use
        """
        self.num_steps = num_steps
        self.beam_width = beam_width
        self.device = self._get_device(device)

        self._corrector = None

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return device if device != "auto" else "cpu"

    def _load_corrector(self):
        """Lazy load Vec2Text corrector.

        Uses manual loading with safetensors to avoid meta tensor issues.
        Models:
            - ielabgroup/vec2text_gtr-base-st_inversion
            - ielabgroup/vec2text_gtr-base-st_corrector
        """
        if self._corrector is not None:
            return

        import vec2text
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel
        from vec2text.models.corrector_encoder import CorrectorEncoderModel

        print("Loading Vec2Text models...")

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

        # Create corrector pipeline
        self._corrector = vec2text.load_corrector(inversion_model, corrector_model)
        print(f"  Vec2Text loaded on {self.device}")

    def invert(self, embedding: torch.Tensor) -> str:
        """Invert embedding to text.

        Args:
            embedding: 768D GTR embedding tensor

        Returns:
            Reconstructed text string
        """
        self._load_corrector()

        import vec2text

        # Ensure correct shape and device
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Invert
        result = vec2text.invert_embeddings(
            embeddings=embedding,
            corrector=self._corrector,
            num_steps=self.num_steps,
            sequence_beam_width=self.beam_width,
        )

        return result[0] if isinstance(result, list) else result

    def invert_batch(self, embeddings: torch.Tensor) -> List[str]:
        """Invert batch of embeddings.

        Args:
            embeddings: (N, 768) tensor

        Returns:
            List of reconstructed texts
        """
        self._load_corrector()

        import vec2text

        embeddings = embeddings.to(self.device)

        results = vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=self._corrector,
            num_steps=self.num_steps,
            sequence_beam_width=self.beam_width,
        )

        return results if isinstance(results, list) else [results]


class Vec2TextHbBoPsInference:
    """Complete inference pipeline for Vec2Text-integrated HbBoPs.

    Pipeline:
        1. Run HbBoPs optimization (discrete grid search with BO)
        2. Get best prompt and its latent representation
        3. Optionally optimize in latent space for novel prompts
        4. Decode latents to 768D embeddings via AE decoder
        5. Invert embeddings to text via Vec2Text
        6. Evaluate novel prompts on validation set

    This enables:
        - Finding best prompts from predefined grid
        - Generating novel prompts by optimizing in latent space
        - Converting any 10D latent to readable instruction+exemplar text
        - Evaluating novel prompts against the full validation dataset
    """

    def __init__(
        self,
        hbbops: HbBoPsVec2Text,
        vec2text_steps: int = 50,
        vec2text_beam: int = 4,
        llm_evaluator=None,
    ):
        """Initialize inference pipeline.

        Args:
            hbbops: Trained HbBoPsVec2Text instance (with AE trained)
            vec2text_steps: Vec2Text correction iterations
            vec2text_beam: Vec2Text beam width
            llm_evaluator: Optional evaluator for novel prompts (callable)
        """
        self.hbbops = hbbops
        self.inverter = Vec2TextInverter(
            num_steps=vec2text_steps,
            beam_width=vec2text_beam,
            device=str(hbbops.device),
        )
        self.encoder = hbbops.encoder
        self.llm_evaluator = llm_evaluator

    def invert_latent_to_text(
        self,
        latent: torch.Tensor,
        verify: bool = True,
    ) -> InversionResult:
        """Invert latent to instruction and exemplar text.

        Pipeline:
            10D latent -> AE decode -> 1536D -> split -> 2x 768D -> Vec2Text

        Args:
            latent: 10D latent tensor
            verify: Whether to verify reconstruction quality

        Returns:
            InversionResult with texts and metrics
        """
        # Decode latent to embeddings
        inst_emb, ex_emb = self.hbbops.decode_latent(latent)

        # Invert to text
        inst_text = self.inverter.invert(inst_emb)
        ex_text = self.inverter.invert(ex_emb)

        # Verify reconstruction quality
        inst_cosine = 0.0
        ex_cosine = 0.0

        if verify:
            # Re-encode and compare
            inst_reenc = self.encoder.encode_tensor(inst_text)
            ex_reenc = self.encoder.encode_tensor(ex_text)

            inst_cosine = torch.nn.functional.cosine_similarity(
                inst_emb.unsqueeze(0).cpu(),
                inst_reenc.unsqueeze(0).cpu(),
            ).item()

            ex_cosine = torch.nn.functional.cosine_similarity(
                ex_emb.unsqueeze(0).cpu(),
                ex_reenc.unsqueeze(0).cpu(),
            ).item()

        return InversionResult(
            instruction_text=inst_text,
            exemplar_text=ex_text,
            instruction_cosine=inst_cosine,
            exemplar_cosine=ex_cosine,
            latent=latent,
        )

    def evaluate_novel_prompt(
        self,
        instruction: str,
        exemplar: str,
        verbose: bool = True,
    ) -> float:
        """Evaluate a novel prompt on the full validation dataset.

        Args:
            instruction: Instruction text from Vec2Text
            exemplar: Exemplar text from Vec2Text
            verbose: Print progress

        Returns:
            Error rate on validation set (0.0 to 1.0)
        """
        if self.llm_evaluator is None:
            if verbose:
                print("  No LLM evaluator provided, skipping evaluation")
            return None

        # Create a temporary Prompt object
        novel_prompt = Prompt(
            instruction_id=-1,  # Novel, not from grid
            exemplar_id=-1,
            instruction=instruction,
            exemplar=exemplar,
        )

        if verbose:
            print(f"  Evaluating novel prompt on {len(self.hbbops.validation_data)} samples...")

        # Evaluate on full validation set
        error_rate = self.llm_evaluator(novel_prompt, self.hbbops.validation_data)

        if verbose:
            accuracy = (1.0 - error_rate) * 100
            print(f"  Novel prompt error rate: {error_rate:.4f} (accuracy: {accuracy:.2f}%)")

        return error_rate

    def sample_latent_candidates(
        self,
        n_candidates: int = 100,
        perturbation_std: float = 0.5,
    ) -> List[torch.Tensor]:
        """Sample candidate latents from observed distribution.

        Samples around the distribution of observed latents from the grid.

        Args:
            n_candidates: Number of candidates to sample
            perturbation_std: Standard deviation for perturbations

        Returns:
            List of candidate latent tensors
        """
        # Get all observed latents from prompts
        observed_latents = []
        for prompt in self.hbbops.prompts:
            latent = self.hbbops.get_prompt_latent(prompt)
            observed_latents.append(latent)

        observed = torch.stack(observed_latents)

        # Compute statistics
        mean = observed.mean(dim=0)
        std = observed.std(dim=0) + 1e-6

        # Sample around observed distribution
        candidates = []
        for _ in range(n_candidates):
            # Sample from Gaussian fitted to observed
            z = mean + std * perturbation_std * torch.randn_like(mean)
            candidates.append(z)

        return candidates

    def optimize_latent(
        self,
        n_candidates: int = 100,
        perturbation_std: float = 0.5,
    ) -> torch.Tensor:
        """Find optimal point in latent space via acquisition function.

        Uses random sampling around observed distribution.

        Args:
            n_candidates: Number of candidates to evaluate
            perturbation_std: Perturbation scale

        Returns:
            Optimal 10D latent tensor
        """
        if self.hbbops.gp_model is None:
            raise RuntimeError("GP model not trained. Run run_hyperband() first.")

        candidates = self.sample_latent_candidates(n_candidates, perturbation_std)

        # Find incumbent (best observed value)
        vmin_b = self.hbbops.best_validation_error

        best_latent = None
        best_ei = -float("inf")

        for latent in candidates:
            # Decode to embedding and compute EI
            inst_emb, ex_emb = self.hbbops.decode_latent(latent)

            # Create dummy prompt with decoded embeddings
            X = torch.cat([inst_emb.unsqueeze(0), ex_emb.unsqueeze(0)], dim=1)

            # Normalize
            denominator = self.hbbops.X_max - self.hbbops.X_min
            denominator[denominator == 0] = 1.0
            X_norm = (X - self.hbbops.X_min) / denominator

            # Predict
            self.hbbops.gp_model.eval()
            self.hbbops.likelihood.eval()

            try:
                with torch.no_grad():
                    pred = self.hbbops.likelihood(self.hbbops.gp_model(X_norm))
                    mean = (
                        pred.mean.item() * self.hbbops.y_std.item()
                        + self.hbbops.y_mean.item()
                    )
                    std = pred.stddev.item() * self.hbbops.y_std.item()
            except Exception:
                continue

            # Compute EI
            if std <= 0:
                ei = max(vmin_b - mean, 0)
            else:
                from scipy.stats import norm

                z = (vmin_b - mean) / std
                ei = (vmin_b - mean) * norm.cdf(z) + std * norm.pdf(z)

            if ei > best_ei:
                best_ei = ei
                best_latent = latent

        return best_latent

    def run_full_pipeline(
        self,
        run_hyperband: bool = True,
        optimize_latent: bool = True,
        n_latent_candidates: int = 100,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run complete optimization and inversion pipeline.

        Steps:
            1. Train autoencoder (if not already trained)
            2. Run HbBoPs optimization (optional)
            3. Invert best prompt latent to text
            4. Optimize in latent space for novel prompt (optional)
            5. Invert novel latent to text

        Args:
            run_hyperband: Whether to run HbBoPs optimization
            optimize_latent: Whether to optimize in latent space
            n_latent_candidates: Candidates for latent optimization
            verbose: Print progress

        Returns:
            OptimizationResult with all results
        """
        # Ensure autoencoder is trained
        if not self.hbbops.ae_trained:
            self.hbbops.train_autoencoder(verbose=verbose)

        # Run HbBoPs or use already-loaded best prompt
        best_prompt = None
        best_error = float("inf")

        if run_hyperband:
            best_prompt, best_error = self.hbbops.run_hyperband(verbose=verbose)
        else:
            # Use best prompt from grid that was already loaded
            best_error = self.hbbops.best_validation_error
            best_prompt = self.hbbops.best_prompt

        # Get latent for best prompt
        best_reconstructed = None
        if best_prompt is not None:
            if verbose:
                print("\n" + "=" * 60)
                print("Inverting Best Prompt via Vec2Text")
                print("=" * 60)

            best_latent = self.hbbops.get_prompt_latent(best_prompt)
            best_reconstructed = self.invert_latent_to_text(best_latent, verify=True)

            if verbose:
                print(f"Original instruction: {best_prompt.instruction[:80]}...")
                print(f"Reconstructed: {best_reconstructed.instruction_text[:80]}...")
                print(f"Instruction cosine: {best_reconstructed.instruction_cosine:.4f}")
                print(f"Exemplar cosine: {best_reconstructed.exemplar_cosine:.4f}")

        # Optimize in latent space for novel prompt
        novel_result = None
        if optimize_latent and self.hbbops.gp_model is not None:
            if verbose:
                print("\n" + "=" * 60)
                print("Optimizing in Latent Space for Novel Prompt")
                print("=" * 60)

            novel_latent = self.optimize_latent(n_candidates=n_latent_candidates)
            if novel_latent is not None:
                novel_result = self.invert_latent_to_text(novel_latent, verify=True)

                if verbose:
                    print(f"Novel instruction: {novel_result.instruction_text[:80]}...")
                    print(f"Novel exemplar: {novel_result.exemplar_text[:80]}...")
                    print(f"Instruction cosine: {novel_result.instruction_cosine:.4f}")
                    print(f"Exemplar cosine: {novel_result.exemplar_cosine:.4f}")

                # Evaluate novel prompt on full validation set
                if self.llm_evaluator is not None:
                    if verbose:
                        print("\n" + "=" * 60)
                        print("Evaluating Novel Prompt on Validation Set")
                        print("=" * 60)

                    novel_error = self.evaluate_novel_prompt(
                        instruction=novel_result.instruction_text,
                        exemplar=novel_result.exemplar_text,
                        verbose=verbose,
                    )
                    novel_result.evaluated_error = novel_error

        return OptimizationResult(
            best_from_grid=best_prompt,
            best_grid_error=best_error,
            best_reconstructed=best_reconstructed,
            novel_from_latent=novel_result,
            num_evaluations=len(self.hbbops.evaluation_cache),
            design_data_size=len(self.hbbops.design_data),
        )

    # ========================================================================
    # Instruction-Only Pipeline (Vec2Text works better on short text)
    # ========================================================================

    def invert_instruction_latent_to_text(
        self,
        latent: torch.Tensor,
        verify: bool = True,
    ) -> str:
        """Invert instruction latent to text only.

        Pipeline:
            10D latent -> Instruction AE decode -> 768D -> Vec2Text -> text

        Args:
            latent: 10D instruction latent tensor
            verify: Whether to return cosine similarity

        Returns:
            Instruction text (and cosine if verify=True)
        """
        # Decode latent to instruction embedding
        inst_emb = self.hbbops.decode_instruction_latent(latent)

        # Invert to text
        inst_text = self.inverter.invert(inst_emb)

        if verify:
            # Re-encode and compute cosine similarity
            inst_reenc = self.encoder.encode_tensor(inst_text)
            cosine = torch.nn.functional.cosine_similarity(
                inst_emb.unsqueeze(0).cpu(),
                inst_reenc.unsqueeze(0).cpu(),
            ).item()
            return inst_text, cosine

        return inst_text

    def optimize_instruction_latent(
        self,
        n_candidates: int = 100,
        perturbation_std: float = 0.5,
        best_exemplar_id: int = None,
    ) -> torch.Tensor:
        """Optimize in instruction latent space only.

        Uses EI acquisition function on instruction latent + fixed exemplar.

        Args:
            n_candidates: Number of candidates to evaluate
            perturbation_std: Perturbation scale
            best_exemplar_id: Fixed exemplar ID from grid

        Returns:
            Optimal 10D instruction latent tensor
        """
        if self.hbbops.gp_model is None:
            raise RuntimeError("GP model not trained. Run load_from_grid() first.")

        if not hasattr(self.hbbops, "instruction_autoencoder"):
            raise RuntimeError(
                "Instruction autoencoder not trained. "
                "Call train_instruction_autoencoder_from_grid() first."
            )

        # Sample instruction latents from observed distribution
        observed_latents = []
        for inst_id in self.hbbops.instruction_embeddings.keys():
            try:
                latent = self.hbbops.encode_instruction(inst_id)
                observed_latents.append(latent)
            except Exception:
                pass

        if not observed_latents:
            raise RuntimeError("No instruction embeddings available for sampling")

        observed = torch.stack(observed_latents)
        mean = observed.mean(dim=0)
        std = observed.std(dim=0) + 1e-6

        # Get fixed exemplar embedding
        ex_emb = torch.tensor(
            self.hbbops.exemplar_embeddings[best_exemplar_id],
            dtype=torch.float32,
            device=self.hbbops.device,
        )

        # Find incumbent (best observed value)
        vmin_b = self.hbbops.best_validation_error

        best_latent = None
        best_ei = -float("inf")

        for _ in range(n_candidates):
            # Sample instruction latent
            z_inst = mean + std * perturbation_std * torch.randn_like(mean)

            # Decode to instruction embedding
            inst_emb = self.hbbops.decode_instruction_latent(z_inst)

            # Combine with fixed exemplar for GP prediction
            X = torch.cat([inst_emb.unsqueeze(0), ex_emb.unsqueeze(0)], dim=1)

            # Normalize
            denominator = self.hbbops.X_max - self.hbbops.X_min
            denominator[denominator == 0] = 1.0
            X_norm = (X - self.hbbops.X_min) / denominator

            # Predict
            self.hbbops.gp_model.eval()
            self.hbbops.likelihood.eval()

            try:
                with torch.no_grad():
                    pred = self.hbbops.likelihood(self.hbbops.gp_model(X_norm))
                    mean_pred = (
                        pred.mean.item() * self.hbbops.y_std.item()
                        + self.hbbops.y_mean.item()
                    )
                    std_pred = pred.stddev.item() * self.hbbops.y_std.item()
            except Exception:
                continue

            # Compute EI
            if std_pred <= 0:
                ei = max(vmin_b - mean_pred, 0)
            else:
                from scipy.stats import norm

                z = (vmin_b - mean_pred) / std_pred
                ei = (vmin_b - mean_pred) * norm.cdf(z) + std_pred * norm.pdf(z)

            if ei > best_ei:
                best_ei = ei
                best_latent = z_inst

        return best_latent

    def run_instruction_only_pipeline(
        self,
        n_latent_candidates: int = 100,
        perturbation_std: float = 0.5,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run instruction-only optimization pipeline.

        This pipeline:
            1. Uses best exemplar from grid (fixed)
            2. Optimizes only instruction in latent space
            3. Inverts instruction latent to text via Vec2Text
            4. Combines novel instruction + fixed exemplar
            5. Evaluates on full validation set

        Vec2Text works well on short instructions (~30 tokens) but fails
        on long exemplars (~200+ tokens). This approach fixes the exemplar
        from the pre-evaluated grid.

        Args:
            n_latent_candidates: Number of candidates for latent optimization
            perturbation_std: Perturbation scale for sampling
            verbose: Print progress

        Returns:
            OptimizationResult with novel instruction + fixed exemplar
        """
        # Ensure instruction autoencoder is trained
        if not hasattr(self.hbbops, "instruction_ae_trained") or not self.hbbops.instruction_ae_trained:
            raise RuntimeError(
                "Instruction autoencoder not trained. "
                "Call hbbops.train_instruction_autoencoder_from_grid() first."
            )

        # Get best exemplar from grid
        best_ex_id, best_ex_text = self.hbbops.get_best_exemplar_from_grid()

        if verbose:
            print("\n" + "=" * 60)
            print("Instruction-Only Optimization Pipeline")
            print("=" * 60)
            print(f"Fixed exemplar ID: {best_ex_id}")
            print(f"Fixed exemplar: {best_ex_text[:80]}...")

        # Get best prompt from grid for reference
        best_prompt = self.hbbops.best_prompt
        best_error = self.hbbops.best_validation_error

        # Reconstruct best instruction (for reference)
        best_reconstructed = None
        if best_prompt is not None:
            if verbose:
                print("\n" + "-" * 40)
                print("Inverting Best Instruction via Vec2Text")
                print("-" * 40)

            best_inst_latent = self.hbbops.encode_instruction(best_prompt.instruction_id)
            inst_text, inst_cosine = self.invert_instruction_latent_to_text(
                best_inst_latent, verify=True
            )

            if verbose:
                print(f"Original: {best_prompt.instruction[:80]}...")
                print(f"Reconstructed: {inst_text[:80]}...")
                print(f"Instruction cosine: {inst_cosine:.4f}")

            best_reconstructed = InversionResult(
                instruction_text=inst_text,
                exemplar_text=best_ex_text,  # Fixed from grid
                instruction_cosine=inst_cosine,
                exemplar_cosine=1.0,  # Fixed, so perfect
                latent=best_inst_latent,
            )

        # Optimize in instruction latent space
        novel_result = None
        if self.hbbops.gp_model is not None:
            if verbose:
                print("\n" + "-" * 40)
                print("Optimizing Instruction Latent Space")
                print("-" * 40)

            novel_latent = self.optimize_instruction_latent(
                n_candidates=n_latent_candidates,
                perturbation_std=perturbation_std,
                best_exemplar_id=best_ex_id,
            )

            if novel_latent is not None:
                novel_inst_text, novel_inst_cosine = self.invert_instruction_latent_to_text(
                    novel_latent, verify=True
                )

                if verbose:
                    print(f"Novel instruction: {novel_inst_text[:80]}...")
                    print(f"Instruction cosine: {novel_inst_cosine:.4f}")
                    print(f"Using fixed exemplar (cosine: 1.0)")

                novel_result = InversionResult(
                    instruction_text=novel_inst_text,
                    exemplar_text=best_ex_text,  # Fixed from grid
                    instruction_cosine=novel_inst_cosine,
                    exemplar_cosine=1.0,  # Fixed, so perfect
                    latent=novel_latent,
                )

                # Evaluate novel prompt on full validation set
                if self.llm_evaluator is not None:
                    if verbose:
                        print("\n" + "-" * 40)
                        print("Evaluating Novel Prompt on Validation Set")
                        print("-" * 40)

                    novel_error = self.evaluate_novel_prompt(
                        instruction=novel_inst_text,
                        exemplar=best_ex_text,
                        verbose=verbose,
                    )
                    novel_result.evaluated_error = novel_error

        return OptimizationResult(
            best_from_grid=best_prompt,
            best_grid_error=best_error,
            best_reconstructed=best_reconstructed,
            novel_from_latent=novel_result,
            num_evaluations=len(self.hbbops.evaluation_cache),
            design_data_size=len(self.hbbops.design_data),
        )
