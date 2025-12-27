"""Robust Inference Pipeline.

Gradient-based latent optimization with cycle consistency checks.
No fallback - returns results with cosine score for filtering.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from robust_vec2text.vae import InstructionVAE
from robust_vec2text.encoder import GTRPromptEncoder
from robust_vec2text.exemplar_selector import ExemplarSelector


@dataclass
class InversionResult:
    """Result of Vec2Text inversion."""

    text: str
    target_embedding: torch.Tensor
    realized_embedding: torch.Tensor
    cosine_similarity: float


class RobustInference:
    """Inference pipeline with gradient-based optimization.

    Pipeline:
        1. Start from best known latent
        2. Gradient-based EI optimization (using HbBoPs GP)
        3. Decode latent to target embedding
        4. Vec2Text invert to candidate text
        5. Return result with cosine score

    Uses HbBoPs-style GP for EI computation (same as hbbops_improved_2).
    """

    def __init__(
        self,
        vae: InstructionVAE,
        exemplar_selector: ExemplarSelector,
        exemplar_emb: torch.Tensor,
        gtr: GTRPromptEncoder,
        device: str = "cuda",
    ):
        """Initialize inference pipeline.

        Args:
            vae: Trained VAE model
            exemplar_selector: HbBoPs-style GP for error prediction
            exemplar_emb: Fixed exemplar embedding for EI optimization (768,)
            gtr: GTR encoder
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        self.exemplar_selector = exemplar_selector
        self.exemplar_emb = exemplar_emb.to(self.device)
        self.gtr = gtr

        # Vec2Text will be loaded lazily
        self._vec2text_corrector = None
        self._vec2text_model = None

    def _load_vec2text(self):
        """Lazy load Vec2Text InversionModel.

        Uses vec2text/gtr-512-noise-0.00001 for longer sequence support (max_seq_length=512).
        Single-step generation without corrector.

        Uses manual loading with safetensors to avoid meta tensor issues.
        """
        if self._vec2text_model is not None:
            return

        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download, snapshot_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        print("Loading Vec2Text InversionModel (gtr-512-noise-0.00001)...")

        # Download model files
        model_dir = snapshot_download("vec2text/gtr-512-noise-0.00001")

        # Load config
        config = InversionConfig.from_pretrained(model_dir)
        print(f"  Config: max_seq_length={config.max_seq_length}")

        # Create model and load weights manually
        self._vec2text_model = InversionModel(config)

        # Load sharded safetensors weights
        import os
        import json

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index["weight_map"].values())

        # Load all shards and merge
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        # Load state dict
        self._vec2text_model.load_state_dict(state_dict, strict=False)
        self._vec2text_model = self._vec2text_model.to(self.device).eval()

        print(f"  Vec2Text loaded on {self.device}")

    def optimize_latent_gradient(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        n_steps: int = 500,
        lr: float = 0.1,
        patience: int = 10,
        xi: float = 0.01,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Gradient-based EI optimization in latent space.

        Uses autodiff through GP to maximize Expected Improvement.

        Args:
            initial_latent: Starting latent vector (32,)
            best_y: Best observed error rate (for EI)
            n_steps: Maximum optimization steps
            lr: Learning rate
            patience: Early stopping patience (stop if no improvement for N steps)
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
        patience_counter = 0

        for step in range(n_steps):
            optimizer.zero_grad()

            # Compute EI (we want to maximize it)
            ei = self._compute_ei_differentiable(z.unsqueeze(0), best_y, xi)

            # Negative for minimization
            loss = -ei

            loss.backward()
            optimizer.step()

            # Track best with early stopping
            if ei.item() > best_ei:
                best_ei = ei.item()
                best_z = z.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (step + 1) % 50 == 0:
                print(f"  Step {step+1}: EI = {ei.item():.6f}")

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at step {step+1} (no improvement for {patience} steps)")
                break

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

        Pipeline:
            1. VAE decode: z (32D) → inst_emb (768D)
            2. HbBoPs GP: (inst_emb, exemplar_emb) → predicted error
            3. Compute EI from GP predictions

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

        # Ensure z is on the right device
        z = z.to(self.device)

        # 1. Decode VAE latent to instruction embedding (differentiable)
        inst_emb = self.vae.decode(z)  # (batch, 768)

        # 2. Prepare GP input: concatenate [inst_emb || exemplar_emb]
        batch_size = inst_emb.shape[0]
        exemplar_emb = self.exemplar_emb.unsqueeze(0).expand(batch_size, -1)  # (batch, 768)
        X = torch.cat([inst_emb, exemplar_emb], dim=1)  # (batch, 1536)

        # Normalize using ExemplarSelector's stored params
        selector = self.exemplar_selector
        denominator = selector.X_max - selector.X_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        X_norm = (X - selector.X_min) / denominator

        # 3. Get GP predictions with gradients (don't use no_grad)
        selector.gp_model.eval()
        selector.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = selector.likelihood(selector.gp_model(X_norm))
            mean_norm = pred.mean
            var_norm = pred.variance

        # Denormalize predictions
        mean = mean_norm * selector.y_std + selector.y_mean
        std = torch.sqrt(var_norm) * selector.y_std

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
        # NOTE: Removed clamp to allow gradients to flow even when EI < 0
        # This gives the optimizer signal to move away from bad regions

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
        num_beams: int = 8,
        max_length: int = 512,
    ) -> str:
        """Invert embedding to text via Vec2Text InversionModel.

        Uses single-step generation (no corrector refinement).

        Args:
            target_embedding: Target embedding (768,)
            num_beams: Beam search width
            max_length: Maximum output length

        Returns:
            Reconstructed text
        """
        self._load_vec2text()

        # Ensure correct shape (batch, 768)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        target_embedding = target_embedding.to(self.device)

        # Generate text using InversionModel
        with torch.no_grad():
            output_ids = self._vec2text_model.generate(
                inputs={
                    "frozen_embeddings": target_embedding,
                },
                generation_kwargs={
                    "num_beams": num_beams,
                    "max_length": max_length,
                },
            )

        # Decode tokens to text
        tokenizer = self._vec2text_model.tokenizer
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return result

    def invert_with_safety(
        self,
        latent: torch.Tensor,
        num_beams: int = 8,
        max_length: int = 512,
    ) -> InversionResult:
        """Invert latent via Vec2Text.

        Pipeline:
            1. Decode latent -> target_embedding
            2. Vec2Text invert -> candidate_text
            3. GTR encode candidate_text -> realized_embedding
            4. Compute cosine similarity (no threshold filtering)

        Args:
            latent: Latent vector (32,)
            num_beams: Beam search width
            max_length: Maximum output length

        Returns:
            InversionResult with text, embeddings, and cosine score
        """
        # 1. Decode latent to target embedding
        target_emb = self.decode_latent(latent)

        # 2. Invert via Vec2Text
        candidate_text = self.invert_embedding(target_emb, num_beams, max_length)

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
        )

    def full_pipeline(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        n_opt_steps: int = 500,
        opt_lr: float = 0.1,
        opt_patience: int = 10,
        v2t_beams: int = 8,
        v2t_max_length: int = 512,
        verbose: bool = True,
    ) -> InversionResult:
        """Run full optimization and inversion pipeline.

        Args:
            initial_latent: Starting latent (from best grid instruction)
            best_y: Best observed error rate
            n_opt_steps: Maximum gradient optimization steps
            opt_lr: Optimization learning rate
            opt_patience: Early stopping patience
            v2t_beams: Vec2Text beam search width
            v2t_max_length: Vec2Text maximum output length
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
            patience=opt_patience,
            verbose=verbose,
        )

        if verbose:
            print("\n" + "-" * 40)
            print("Vec2Text Inversion with Safety Check")
            print("-" * 40)

        # Invert with cycle consistency
        result = self.invert_with_safety(
            latent=optimized_latent,
            num_beams=v2t_beams,
            max_length=v2t_max_length,
        )

        if verbose:
            print(f"  Candidate: {result.text[:80]}...")
            print(f"  Cosine similarity: {result.cosine_similarity:.4f}")

        return result
