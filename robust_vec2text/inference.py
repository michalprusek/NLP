"""Robust Inference Pipeline.

Gradient-based latent optimization with cycle consistency checks.
No fallback - returns results with cosine score for filtering.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from robust_vec2text.vae import InstructionVAE
from robust_vec2text.gp import GPTrainer
from robust_vec2text.encoder import GTRPromptEncoder


@dataclass
class InversionResult:
    """Result of Vec2Text inversion with cycle consistency."""

    text: str
    target_embedding: torch.Tensor
    realized_embedding: torch.Tensor
    cosine_similarity: float
    passed_threshold: bool


class RobustInference:
    """Inference pipeline with gradient-based optimization.

    Pipeline:
        1. Start from best known latent
        2. Gradient-based EI optimization
        3. Decode latent to target embedding
        4. Vec2Text invert to candidate text
        5. Cycle consistency check

    No fallback strategy - returns result with cosine score.
    Caller can filter based on threshold.
    """

    def __init__(
        self,
        vae: InstructionVAE,
        gp_trainer: GPTrainer,
        gtr: GTRPromptEncoder,
        device: str = "cuda",
        cosine_threshold: float = 0.95,
    ):
        """Initialize inference pipeline.

        Args:
            vae: Trained VAE model
            gp_trainer: Trained GP trainer
            gtr: GTR encoder
            device: Device to use
            cosine_threshold: Threshold for cycle consistency
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        self.gp_trainer = gp_trainer
        self.gtr = gtr
        self.cosine_threshold = cosine_threshold

        # Vec2Text will be loaded lazily
        self._vec2text_corrector = None
        self._vec2text_model = None

    def _load_vec2text(self):
        """Lazy load Vec2Text models.

        Uses manual loading with safetensors to avoid meta tensor issues.
        Models:
            - ielabgroup/vec2text_gtr-base-st_inversion
            - ielabgroup/vec2text_gtr-base-st_corrector
        """
        if self._vec2text_corrector is not None:
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
        self._vec2text_corrector = vec2text.load_corrector(inversion_model, corrector_model)
        print(f"  Vec2Text loaded on {self.device}")

    def optimize_latent_gradient(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        n_steps: int = 50,
        lr: float = 0.1,
        xi: float = 0.01,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Gradient-based EI optimization in latent space.

        Uses autodiff through GP to maximize Expected Improvement.

        Args:
            initial_latent: Starting latent vector (32,)
            best_y: Best observed error rate (for EI)
            n_steps: Optimization steps
            lr: Learning rate
            xi: EI exploration parameter
            verbose: Print progress

        Returns:
            Optimized latent vector (32,)
        """
        # Clone and require grad
        z = initial_latent.clone().detach().to(self.device)
        z.requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=lr)

        best_ei = float("-inf")
        best_z = z.clone().detach()

        for step in range(n_steps):
            optimizer.zero_grad()

            # Compute EI (we want to maximize it)
            ei = self._compute_ei_differentiable(z.unsqueeze(0), best_y, xi)

            # Negative for minimization
            loss = -ei

            loss.backward()
            optimizer.step()

            # Track best
            if ei.item() > best_ei:
                best_ei = ei.item()
                best_z = z.clone().detach()

            if verbose and (step + 1) % 10 == 0:
                print(f"  Step {step+1}: EI = {ei.item():.6f}")

        if verbose:
            print(f"  Best EI: {best_ei:.6f}")

        return best_z

    def _compute_ei_differentiable(
        self,
        z: torch.Tensor,
        best_y: float,
        xi: float = 0.01,
    ) -> torch.Tensor:
        """Compute differentiable EI for gradient optimization.

        Uses PyTorch distributions for differentiability.
        Directly calls GP model in eval mode but with gradients enabled.

        Args:
            z: Latent points (batch, 32)
            best_y: Best observed value
            xi: Exploration parameter

        Returns:
            EI values (batch,)
        """
        import gpytorch

        # Ensure z is on the right device and normalized like training data
        z = z.to(self.device)

        # Normalize z using GP trainer's stored normalization params
        z_norm = (z - self.gp_trainer.X_mean) / self.gp_trainer.X_std

        # Get GP predictions with gradients (don't use no_grad)
        self.gp_trainer.gp_model.eval()
        self.gp_trainer.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.gp_trainer.likelihood(self.gp_trainer.gp_model(z_norm))
            mean_norm = pred.mean
            var_norm = pred.variance

        # Denormalize predictions
        mean = mean_norm * self.gp_trainer.y_std + self.gp_trainer.y_mean
        std = torch.sqrt(var_norm) * self.gp_trainer.y_std

        # For minimization: EI = E[max(best_y - f(x) - xi, 0)]
        improvement = best_y - mean - xi
        std_safe = std + 1e-8

        # Normal distribution for differentiable CDF/PDF
        normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        Z = improvement / std_safe

        # EI = improvement * Phi(Z) + std * phi(Z)
        cdf = normal.cdf(Z)
        pdf = torch.exp(normal.log_prob(Z))

        ei = improvement * cdf + std_safe * pdf
        ei = torch.clamp(ei, min=0)  # EI is non-negative

        return ei.squeeze()

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to target embedding.

        Args:
            z: Latent vector (32,) or (1, 32)

        Returns:
            Target embedding (768,), L2-normalized
        """
        self.vae.eval()

        if z.dim() == 1:
            z = z.unsqueeze(0)

        with torch.no_grad():
            target_emb = self.vae.decode(z.to(self.device))

        return target_emb.squeeze(0)

    def invert_embedding(
        self,
        target_embedding: torch.Tensor,
        num_steps: int = 50,
        beam_width: int = 8,
    ) -> str:
        """Invert embedding to text via Vec2Text.

        Args:
            target_embedding: Target embedding (768,)
            num_steps: Vec2Text correction steps
            beam_width: Beam search width

        Returns:
            Reconstructed text
        """
        self._load_vec2text()

        import vec2text

        # Ensure correct shape
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)

        # Move to correct device
        target_embedding = target_embedding.to(self.device)

        # Vec2Text inversion
        with torch.no_grad():
            result = vec2text.invert_embeddings(
                embeddings=target_embedding,
                corrector=self._vec2text_corrector,
                num_steps=num_steps,
                sequence_beam_width=beam_width,
            )

        return result[0] if result else ""

    def invert_with_safety(
        self,
        latent: torch.Tensor,
        num_steps: int = 50,
        beam_width: int = 8,
    ) -> InversionResult:
        """Invert latent with cycle consistency check.

        Pipeline:
            1. Decode latent -> target_embedding
            2. Vec2Text invert -> candidate_text
            3. GTR encode candidate_text -> realized_embedding
            4. Check cosine(target, realized) > threshold

        Args:
            latent: Latent vector (32,)
            num_steps: Vec2Text steps
            beam_width: Beam width

        Returns:
            InversionResult with text, embeddings, and cosine score
        """
        # 1. Decode latent to target embedding
        target_emb = self.decode_latent(latent)

        # 2. Invert via Vec2Text
        candidate_text = self.invert_embedding(target_emb, num_steps, beam_width)

        # 3. Re-encode with GTR
        realized_emb = self.gtr.encode_tensor(candidate_text).to(self.device)

        # 4. Compute cosine similarity
        cosine = F.cosine_similarity(
            target_emb.unsqueeze(0),
            realized_emb.unsqueeze(0),
        ).item()

        return InversionResult(
            text=candidate_text,
            target_embedding=target_emb,
            realized_embedding=realized_emb,
            cosine_similarity=cosine,
            passed_threshold=cosine >= self.cosine_threshold,
        )

    def full_pipeline(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        n_opt_steps: int = 50,
        opt_lr: float = 0.1,
        v2t_steps: int = 50,
        v2t_beam: int = 8,
        verbose: bool = True,
    ) -> InversionResult:
        """Run full optimization and inversion pipeline.

        Args:
            initial_latent: Starting latent (from best grid instruction)
            best_y: Best observed error rate
            n_opt_steps: Gradient optimization steps
            opt_lr: Optimization learning rate
            v2t_steps: Vec2Text steps
            v2t_beam: Vec2Text beam width
            verbose: Print progress

        Returns:
            InversionResult with optimized instruction
        """
        if verbose:
            print("\n" + "-" * 40)
            print("Gradient-Based Latent Optimization")
            print("-" * 40)

        # Optimize latent
        optimized_latent = self.optimize_latent_gradient(
            initial_latent=initial_latent,
            best_y=best_y,
            n_steps=n_opt_steps,
            lr=opt_lr,
            verbose=verbose,
        )

        if verbose:
            print("\n" + "-" * 40)
            print("Vec2Text Inversion with Safety Check")
            print("-" * 40)

        # Invert with cycle consistency
        result = self.invert_with_safety(
            latent=optimized_latent,
            num_steps=v2t_steps,
            beam_width=v2t_beam,
        )

        if verbose:
            print(f"  Candidate: {result.text[:80]}...")
            print(f"  Cosine similarity: {result.cosine_similarity:.4f}")
            print(f"  Passed threshold ({self.cosine_threshold}): {result.passed_threshold}")

        return result
