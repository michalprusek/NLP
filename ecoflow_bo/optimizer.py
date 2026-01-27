"""
EcoFlowBO: Main optimizer class for Bayesian Optimization in latent space.

Combines all components:
- MatryoshkaEncoder: 768D → 8D hierarchical latent
- RectifiedFlowDecoder: 1-step deterministic decoding
- CoarseToFineGP: Progressive dimension unlocking
- DensityAwareAcquisition: Manifold-respecting exploration
- CycleConsistencyChecker: Hallucination detection
"""

import torch
from typing import Optional, Callable, List, Tuple, Dict, Any
from pathlib import Path

from .config import EcoFlowConfig
from .encoder import MatryoshkaEncoder
from .velocity_network import VelocityNetwork
from .cfm_decoder import RectifiedFlowDecoder
from .latent_gp import CoarseToFineGP
from .density_acquisition import DensityAwareAcquisition
from .cycle_consistency import AdaptiveCycleChecker


class EcoFlowBO:
    """
    Embedding-Conditioned Flow for Bayesian Optimization.

    Usage:
        # Load trained models
        optimizer = EcoFlowBO.from_checkpoint("results/ecoflow_checkpoints/best.pt")

        # Define objective (e.g., prompt quality score)
        def objective(embedding: torch.Tensor) -> float:
            text = decode_to_text(embedding)
            return evaluate_prompt_quality(text)

        # Run optimization
        best_z, best_embedding, best_score = optimizer.optimize(
            initial_embeddings=initial_prompts,
            initial_scores=initial_scores,
            objective=objective,
            n_iterations=50,
        )
    """

    def __init__(
        self,
        encoder: MatryoshkaEncoder,
        decoder: RectifiedFlowDecoder,
        config: Optional[EcoFlowConfig] = None,
    ):
        if config is None:
            config = EcoFlowConfig()

        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        # Initialize components
        self.gp = CoarseToFineGP(config.gp)
        self.acquisition = DensityAwareAcquisition(config.acquisition)
        self.cycle_checker = AdaptiveCycleChecker(
            encoder, decoder, config.cycle
        )

        self.device = config.device
        self.best_z = None
        self.best_embedding = None
        self.best_score = float("-inf")
        self._initialized = False

        # History for analysis
        self.history: List[Dict[str, Any]] = []

    def to(self, device: str) -> "EcoFlowBO":
        """Move models to device."""
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return self

    def eval(self) -> "EcoFlowBO":
        """Set models to eval mode."""
        self.encoder.eval()
        self.decoder.velocity_net.eval()
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[EcoFlowConfig] = None,
        device: str = "cuda",
    ) -> "EcoFlowBO":
        """
        Load optimizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pt file
            config: Optional config override
            device: Device to load to

        Returns:
            EcoFlowBO instance
        """
        # weights_only=False required for EcoFlowConfig dataclass in checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if config is None:
            config = checkpoint.get("config", EcoFlowConfig())

        # Create models
        encoder = MatryoshkaEncoder(config.encoder)
        velocity_net = VelocityNetwork(config.velocity_net)
        decoder = RectifiedFlowDecoder(velocity_net, config.decoder)

        # Load weights
        encoder.load_state_dict(checkpoint["encoder"])
        velocity_net.load_state_dict(checkpoint["velocity_net"])

        if checkpoint.get("is_reflowed", False):
            decoder.mark_as_reflowed()

        optimizer = cls(encoder, decoder, config)
        optimizer = optimizer.to(device)
        optimizer.eval()

        print(f"Loaded EcoFlowBO from {checkpoint_path}")
        print(f"  Reflowed: {decoder.is_reflowed}")
        print(f"  Euler steps: {decoder.config.euler_steps}")

        return optimizer

    def initialize(
        self,
        initial_embeddings: torch.Tensor,
        initial_scores: torch.Tensor,
    ):
        """
        Initialize GP with initial observations.

        Args:
            initial_embeddings: Known embeddings [N, 768]
            initial_scores: Corresponding objective values [N]
        """
        initial_embeddings = initial_embeddings.to(self.device)
        initial_scores = initial_scores.to(self.device)

        # Encode to latent space
        with torch.no_grad():
            z = self.encoder.encode_deterministic(initial_embeddings)

        # Initialize GP
        self.gp.fit(z, initial_scores)

        # Track best
        best_idx = initial_scores.argmax()
        self.best_z = z[best_idx]
        self.best_embedding = initial_embeddings[best_idx]
        self.best_score = initial_scores[best_idx].item()

        # Calibrate cycle checker on valid samples
        self.cycle_checker.calibrate(z)

        self._initialized = True

        print(f"Initialized with {len(initial_scores)} points")
        print(f"  Best initial score: {self.best_score:.4f}")
        print(f"  GP stage: {self.gp.current_stage} ({self.gp.n_active_dims}D)")

    def step(
        self,
        objective: Callable[[torch.Tensor], float],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform one optimization step.

        Args:
            objective: Function that takes embedding [768] and returns score
            batch_size: Number of candidates to evaluate

        Returns:
            Dict with step results

        Raises:
            RuntimeError: If initialize() was not called first
        """
        if not self._initialized:
            raise RuntimeError(
                "EcoFlowBO.step() called before initialize(). "
                "Call initialize(initial_embeddings, initial_scores) first."
            )

        # Generate and score candidates
        candidates = self.acquisition.generate_candidates(
            self.gp, self.best_z
        )
        _, acq_values = self.acquisition.select_best_candidates(
            self.gp, candidates, n_select=len(candidates)
        )

        # Sort by acquisition value
        sorted_indices = acq_values.argsort(descending=True)
        candidates_sorted = candidates[sorted_indices]
        acq_sorted = acq_values[sorted_indices]

        # Select valid candidate using cycle consistency
        z_selected, x_decoded, cycle_error, n_tried = (
            self.cycle_checker.select_valid_from_ranked(
                candidates_sorted,
                acq_sorted,
                active_dims=self.gp.active_dims,  # Pass list of indices, not length
            )
        )

        # Evaluate objective
        with torch.no_grad():
            score = objective(x_decoded)

        # Update GP
        self.gp.update(
            z_selected.unsqueeze(0),
            torch.tensor([score], device=self.device),
        )

        # Update best if improved
        if score > self.best_score:
            self.best_z = z_selected
            self.best_embedding = x_decoded
            self.best_score = score
            improved = True
        else:
            improved = False

        # Track history
        result = {
            "score": score,
            "best_score": self.best_score,
            "improved": improved,
            "cycle_error": cycle_error,
            "n_tried": n_tried,
            "acq_value": acq_sorted[0].item(),
            "gp_stage": self.gp.current_stage,
            "n_points": self.gp.n_points,
        }
        self.history.append(result)

        return result

    def optimize(
        self,
        initial_embeddings: torch.Tensor,
        initial_scores: torch.Tensor,
        objective: Callable[[torch.Tensor], float],
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run full optimization loop.

        Args:
            initial_embeddings: Starting embeddings [N, 768]
            initial_scores: Initial objective values [N]
            objective: Objective function
            n_iterations: Number of BO iterations
            verbose: Print progress

        Returns:
            best_z: Best latent [latent_dim]
            best_embedding: Best embedding [768]
            best_score: Best objective value
        """
        self.initialize(initial_embeddings, initial_scores)

        for i in range(n_iterations):
            result = self.step(objective)

            if verbose:
                status = "★" if result["improved"] else " "
                print(
                    f"[{i+1:3d}/{n_iterations}] {status} "
                    f"score={result['score']:.4f} "
                    f"best={result['best_score']:.4f} "
                    f"stage={result['gp_stage']} "
                    f"cycle_err={result['cycle_error']:.3f} "
                    f"tried={result['n_tried']}"
                )

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Best score: {self.best_score:.4f}")
            print(f"  Total evaluations: {self.gp.n_points}")
            print(f"  Final GP stage: {self.gp.current_stage}")

        return self.best_z, self.best_embedding, self.best_score

    def get_top_k(self, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k observations from GP.

        Returns:
            z: Top-k latents [k, latent_dim]
            scores: Top-k scores [k]
        """
        scores = self.gp.train_y
        top_indices = scores.argsort(descending=True)[:k]

        return self.gp.train_z[top_indices], scores[top_indices]

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to embedding (deterministic).

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            embeddings: [B, 768]
        """
        with torch.no_grad():
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return self.decoder.decode_deterministic(z)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the optimization run."""
        if not self.history:
            return {}

        scores = [h["score"] for h in self.history]
        improvements = [h["improved"] for h in self.history]
        cycle_errors = [h["cycle_error"] for h in self.history]

        return {
            "n_iterations": len(self.history),
            "best_score": self.best_score,
            "n_improvements": sum(improvements),
            "improvement_rate": sum(improvements) / len(improvements),
            "mean_cycle_error": sum(cycle_errors) / len(cycle_errors),
            "final_gp_stage": self.gp.current_stage,
            "total_gp_points": self.gp.n_points,
            "scores": scores,
        }


class EcoFlowBOWithVec2Text(EcoFlowBO):
    """
    EcoFlowBO with integrated vec2text decoder for text generation.

    Provides end-to-end: latent z → embedding → text
    """

    def __init__(
        self,
        encoder: MatryoshkaEncoder,
        decoder: RectifiedFlowDecoder,
        config: Optional[EcoFlowConfig] = None,
        vec2text_model: str = "gtr-base",
    ):
        super().__init__(encoder, decoder, config)
        self.vec2text_model = vec2text_model
        self._inverter = None

    @property
    def inverter(self):
        """Lazy load vec2text inverter."""
        if self._inverter is None:
            from vec2text import Inverter
            self._inverter = Inverter.from_pretrained(self.vec2text_model)
        return self._inverter

    def decode_to_text(self, z: torch.Tensor) -> List[str]:
        """
        Decode latent to text via embedding → vec2text.

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            texts: List of decoded texts
        """
        embeddings = self.decode_latent(z)
        texts = self.inverter.invert(embeddings.cpu().numpy())
        return texts

    def optimize_with_text_objective(
        self,
        initial_texts: List[str],
        text_objective: Callable[[str], float],
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Optimize with text-based objective.

        Args:
            initial_texts: Starting prompts
            text_objective: Function that scores a text prompt
            n_iterations: Number of iterations
            verbose: Print progress

        Returns:
            best_z: Best latent
            best_text: Best text prompt
            best_score: Best score
        """
        # Encode initial texts to embeddings
        from sentence_transformers import SentenceTransformer
        encoder_model = SentenceTransformer("sentence-transformers/gtr-t5-base")
        initial_embeddings = torch.tensor(
            encoder_model.encode(initial_texts, convert_to_numpy=True),
            device=self.device,
        )

        # Evaluate initial texts
        initial_scores = torch.tensor(
            [text_objective(t) for t in initial_texts],
            device=self.device,
        )

        # Wrapper objective: embedding → text → score
        def embedding_objective(embedding: torch.Tensor) -> float:
            text = self.inverter.invert(embedding.cpu().numpy().reshape(1, -1))[0]
            return text_objective(text)

        # Run optimization
        best_z, best_embedding, best_score = self.optimize(
            initial_embeddings,
            initial_scores,
            embedding_objective,
            n_iterations,
            verbose,
        )

        # Get best text
        best_text = self.decode_to_text(best_z.unsqueeze(0))[0]

        return best_z, best_text, best_score
