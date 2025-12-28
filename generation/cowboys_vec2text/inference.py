"""COWBOYS Inference Pipeline.

pCN MCMC optimization with Vec2Text inversion.
Replaces gradient-based optimization with probabilistic sampling.

This is the instruction-only version (no exemplars).
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .vae import InstructionVAE
from .encoder import GTRPromptEncoder
from .mcmc import pCNSampler, MCMCConfig, MCMCResult
from .trust_region import TrustRegionManager, TRConfig

# Import instruction-only GP class
from robust_vec2text.exemplar_selector import InstructionSelector


@dataclass
class InversionResult:
    """Result of Vec2Text inversion."""

    text: str
    target_embedding: torch.Tensor
    realized_embedding: torch.Tensor
    cosine_similarity: float
    log_ei: float  # Log Expected Improvement
    optimized_latent: torch.Tensor = None
    reembedded_latent: torch.Tensor = None


@dataclass
class CowboysResult:
    """Result of full COWBOYS optimization pipeline."""

    best_result: InversionResult
    mcmc_result: MCMCResult
    n_candidates_decoded: int


class CowboysInference:
    """COWBOYS inference pipeline with pCN MCMC optimization.

    This is the instruction-only version (no exemplars).

    Pipeline:
        1. Start from best known latent (anchor for trust region)
        2. Run pCN MCMC sampling within trust region
        3. Decode top candidates via VAE
        4. Invert to text via Vec2Text
        5. Return best candidate by LogEI score

    Attributes:
        vae: Trained VAE model
        instruction_selector: InstructionSelector GP for error prediction
        mcmc_sampler: pCN MCMC sampler
    """

    def __init__(
        self,
        vae: InstructionVAE,
        instruction_selector: InstructionSelector,
        gtr: GTRPromptEncoder,
        device: str = "cuda",
    ):
        """Initialize inference pipeline.

        Args:
            vae: Trained VAE model
            instruction_selector: InstructionSelector GP for error prediction
            gtr: GTR encoder
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        self.instruction_selector = instruction_selector
        self.gtr = gtr

        # Initialize pCN MCMC sampler (instruction-only, no exemplar_emb)
        self.mcmc_sampler = pCNSampler(
            vae=vae,
            instruction_selector=instruction_selector,
            device=str(device),
        )

        # Vec2Text loaded lazily
        self._vec2text_model = None

    def _load_vec2text(self):
        """Lazy load Vec2Text InversionModel."""
        if self._vec2text_model is not None:
            return

        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        print("Loading Vec2Text InversionModel (gtr-512-noise-0.00001)...")

        model_dir = snapshot_download("vec2text/gtr-512-noise-0.00001")
        config = InversionConfig.from_pretrained(model_dir)
        print(f"  Config: max_seq_length={config.max_seq_length}")

        self._vec2text_model = InversionModel(config)

        import os
        import json

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        shard_files = set(index["weight_map"].values())

        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        self._vec2text_model.load_state_dict(state_dict, strict=False)
        self._vec2text_model = self._vec2text_model.to(self.device).eval()

        print(f"  Vec2Text loaded on {self.device}")

    def optimize_latent_mcmc(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        mcmc_config: MCMCConfig,
        trust_region: Optional[TrustRegionManager] = None,
        verbose: bool = True,
    ) -> MCMCResult:
        """Run pCN MCMC optimization in latent space.

        Replaces optimize_latent_gradient() from robust_vec2text.

        Args:
            initial_latent: Starting latent (32,)
            best_y: Best observed error rate
            mcmc_config: MCMC configuration
            trust_region: Optional trust region manager
            verbose: Print progress

        Returns:
            MCMCResult with candidates and best latent
        """
        if verbose:
            print("  Running pCN MCMC sampling...")

        # Generate multiple starting points
        initial_latents = [initial_latent.clone()]
        if trust_region is not None:
            for _ in range(mcmc_config.n_chains - 1):
                initial_latents.append(trust_region.get_random_point_in_region())
        else:
            for _ in range(mcmc_config.n_chains - 1):
                noise = torch.randn_like(initial_latent) * 0.5
                initial_latents.append(initial_latent + noise)

        # Run MCMC chains
        result = self.mcmc_sampler.sample_multiple_chains(
            initial_latents, best_y, mcmc_config, trust_region, verbose
        )

        return result

    def decode_and_invert_single(
        self,
        z: torch.Tensor,
        num_beams: int = 8,
        max_length: int = 512,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> Tuple[str, float, torch.Tensor]:
        """Decode and invert a single latent.

        Args:
            z: Latent vector (32,)
            num_beams: Vec2Text beam width
            max_length: Maximum output length
            no_repeat_ngram_size: Block repeating n-grams
            repetition_penalty: Penalize repeated tokens

        Returns:
            Tuple of (text, cosine_similarity, realized_embedding)
        """
        self._load_vec2text()

        # Decode latent to target embedding
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device)

        with torch.no_grad():
            target_emb = self.vae.decode(z).squeeze(0)

        # Invert via Vec2Text
        target_emb_batch = target_emb.unsqueeze(0)

        gen_kwargs = {
            "num_beams": num_beams,
            "max_length": max_length,
        }
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.no_grad():
            output_ids = self._vec2text_model.generate(
                inputs={"frozen_embeddings": target_emb_batch},
                generation_kwargs=gen_kwargs,
            )

        tokenizer = self._vec2text_model.tokenizer
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Re-encode and compute cosine
        realized_emb = self.gtr.encode_tensor(text).to(self.device)
        cosine = F.cosine_similarity(
            target_emb.unsqueeze(0), realized_emb.unsqueeze(0)
        ).item()

        return text, cosine, realized_emb

    def decode_and_invert_batch(
        self,
        latents: List[torch.Tensor],
        num_beams: int = 8,
        max_length: int = 512,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
        verbose: bool = True,
    ) -> List[Tuple[str, float, torch.Tensor, torch.Tensor]]:
        """Decode and invert batch of latents.

        Args:
            latents: List of latent vectors (32,)
            num_beams: Vec2Text beam width
            max_length: Maximum output length
            no_repeat_ngram_size: Block repeating n-grams
            repetition_penalty: Penalize repeated tokens
            verbose: Print progress

        Returns:
            List of (text, cosine, realized_emb, target_emb) tuples
        """
        results = []

        for i, z in enumerate(latents):
            text, cosine, realized_emb = self.decode_and_invert_single(
                z, num_beams, max_length, no_repeat_ngram_size, repetition_penalty
            )

            # Also get target embedding for later use
            with torch.no_grad():
                target_emb = self.vae.decode(z.unsqueeze(0)).squeeze(0)

            results.append((text, cosine, realized_emb, target_emb))

            if verbose and (i + 1) % 5 == 0:
                print(f"    Decoded {i+1}/{len(latents)}")

        return results

    def rank_candidates_by_ei(
        self,
        candidates: List[Tuple[str, float, torch.Tensor, torch.Tensor]],
        latents: List[torch.Tensor],
        best_y: float,
        verbose: bool = True,
    ) -> List[Tuple[str, float, torch.Tensor, torch.Tensor, float, torch.Tensor]]:
        """Rank candidates by LogEI score.

        Args:
            candidates: List of (text, cosine, realized_emb, target_emb) tuples
            latents: Corresponding latent vectors
            best_y: Best observed error rate
            verbose: Print progress

        Returns:
            Sorted list of (text, cosine, realized_emb, target_emb, log_ei, latent)
        """
        if verbose:
            print("  Ranking candidates by LogEI...")

        ranked = []
        for (text, cosine, realized_emb, target_emb), z in zip(candidates, latents):
            log_ei = self.mcmc_sampler.compute_log_ei(z, best_y)
            if isinstance(log_ei, torch.Tensor):
                log_ei = log_ei.item()

            ranked.append((text, cosine, realized_emb, target_emb, log_ei, z))

        # Sort by log EI (higher = better)
        ranked.sort(key=lambda x: x[4], reverse=True)

        return ranked

    def full_pipeline(
        self,
        initial_latent: torch.Tensor,
        best_y: float,
        mcmc_config: MCMCConfig,
        trust_region: Optional[TrustRegionManager] = None,
        v2t_beams: int = 8,
        v2t_max_length: int = 512,
        v2t_no_repeat_ngram_size: int = 3,
        v2t_repetition_penalty: float = 1.2,
        max_decode: int = 20,
        verbose: bool = True,
    ) -> CowboysResult:
        """Run full COWBOYS optimization and inversion pipeline.

        Steps:
            1. pCN MCMC sampling within trust region
            2. Select top candidates by log EI
            3. Decode via VAE -> Vec2Text
            4. Rank by LogEI score
            5. Return best candidate

        Args:
            initial_latent: Starting latent (from best grid instruction)
            best_y: Best observed error rate
            mcmc_config: MCMC configuration
            trust_region: Optional trust region manager
            v2t_beams: Vec2Text beam width
            v2t_max_length: Vec2Text max output length
            v2t_no_repeat_ngram_size: Block repeating n-grams
            v2t_repetition_penalty: Penalize repeated tokens
            max_decode: Maximum candidates to decode (for efficiency)
            verbose: Print progress

        Returns:
            CowboysResult with best candidate and statistics
        """
        if verbose:
            print("\n" + "-" * 40)
            print("COWBOYS pCN MCMC Optimization")
            print("-" * 40)

        # 1. MCMC sampling
        mcmc_result = self.optimize_latent_mcmc(
            initial_latent, best_y, mcmc_config, trust_region, verbose
        )

        if not mcmc_result.candidates:
            # Fallback to best MCMC point if no samples collected
            mcmc_result.candidates = [mcmc_result.best_latent]

        # 2. Select top candidates by log EI
        n_to_decode = min(len(mcmc_result.candidates), max_decode)

        # Sort by log EI
        candidates_with_ei = []
        for z in mcmc_result.candidates:
            log_ei = self.mcmc_sampler.compute_log_ei(z, best_y)
            if isinstance(log_ei, torch.Tensor):
                log_ei = log_ei.item()
            candidates_with_ei.append((z, log_ei))

        candidates_with_ei.sort(key=lambda x: x[1], reverse=True)
        top_latents = [z for z, _ in candidates_with_ei[:n_to_decode]]

        if verbose:
            print(f"  Decoding {n_to_decode} top candidates via Vec2Text...")

        # 3. Decode and invert
        decoded = self.decode_and_invert_batch(
            top_latents,
            v2t_beams,
            v2t_max_length,
            v2t_no_repeat_ngram_size,
            v2t_repetition_penalty,
            verbose,
        )

        # 4. Rank by LogEI
        ranked = self.rank_candidates_by_ei(decoded, top_latents, best_y, verbose)

        # Get best result (now includes the corresponding latent)
        best_text, best_cosine, best_realized, best_target, best_log_ei, optimized_latent = ranked[0]

        # Compute reembedded latent
        with torch.no_grad():
            reembedded_latent = self.vae.get_latent(best_realized.unsqueeze(0)).squeeze(0)

        if verbose:
            print(f"\n  Best candidate:")
            print(f"    Text: {best_text}")
            print(f"    Cosine: {best_cosine:.4f}")
            print(f"    LogEI: {best_log_ei:.4f}")

        best_result = InversionResult(
            text=best_text,
            target_embedding=best_target,
            realized_embedding=best_realized,
            cosine_similarity=best_cosine,
            log_ei=best_log_ei,
            optimized_latent=optimized_latent,
            reembedded_latent=reembedded_latent,
        )

        return CowboysResult(
            best_result=best_result,
            mcmc_result=mcmc_result,
            n_candidates_decoded=len(decoded),
        )

    def invert_with_safety(
        self,
        latent: torch.Tensor,
        num_beams: int = 8,
        max_length: int = 512,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> InversionResult:
        """Invert latent via Vec2Text.

        Compatibility method for simpler use cases.

        Args:
            latent: Latent vector (32,)
            num_beams: Beam search width
            max_length: Maximum output length
            no_repeat_ngram_size: Block repeating n-grams
            repetition_penalty: Penalize repeated tokens

        Returns:
            InversionResult with text, embeddings, and scores
        """
        text, cosine, realized_emb = self.decode_and_invert_single(
            latent, num_beams, max_length, no_repeat_ngram_size, repetition_penalty
        )

        with torch.no_grad():
            target_emb = self.vae.decode(latent.unsqueeze(0)).squeeze(0)
            reembedded_latent = self.vae.get_latent(realized_emb.unsqueeze(0)).squeeze(0)

        # Compute log EI (using current best as reference)
        log_ei = self.mcmc_sampler.compute_log_ei(latent, best_y=0.5)  # Default best
        if isinstance(log_ei, torch.Tensor):
            log_ei = log_ei.item()

        return InversionResult(
            text=text,
            target_embedding=target_emb,
            realized_embedding=realized_emb,
            cosine_similarity=cosine,
            log_ei=log_ei,
            optimized_latent=latent.clone().detach(),
            reembedded_latent=reembedded_latent,
        )
