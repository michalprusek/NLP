"""
EcoFlowBO: Main optimizer class for Bayesian Optimization in latent space.

Combines all components:
- MatryoshkaEncoder: 768D → z_core (16D) + z_detail (32D) residual latent
- RectifiedFlowDecoder: 1-step deterministic decoding from 48D latent
- CoarseToFineGP: Progressive dimension unlocking on z_core only (16D)
- DensityAwareAcquisition: Manifold-respecting exploration
- CycleConsistencyChecker: Hallucination detection on z_core
- DetailRetriever: Get z_detail from training set via nearest neighbor

Key Innovation: Residual Latent Decomposition
- GP operates on z_core (16D) - tractable optimization
- z_detail (32D) is retrieved from training set - high-fidelity decoding
- Total 48D capacity for decoder without GP curse of dimensionality
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
from .cycle_consistency import CycleConsistencyChecker
from .detail_retriever import create_detail_retriever, SimpleDetailRetriever


class EcoFlowBO:
    """
    Embedding-Conditioned Flow for Bayesian Optimization.

    Key Innovation: Residual Latent Architecture
    - z_full = [z_core (16D), z_detail (32D)] = 48D
    - GP optimizes only z_core (tractable 16D)
    - z_detail retrieved from training set (nearest neighbor)
    - Decoder uses full 48D for high-fidelity reconstruction

    Usage:
        # Load trained models
        optimizer = EcoFlowBO.from_checkpoint("results/ecoflow_checkpoints/best.pt")

        # Define objective (e.g., prompt quality score)
        def objective(embedding: torch.Tensor) -> float:
            text = decode_to_text(embedding)
            return evaluate_prompt_quality(text)

        # Run optimization
        best_z_core, best_embedding, best_score = optimizer.optimize(
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

        # Residual latent dimensions
        self.core_dim = config.encoder.latent_dim  # 16D
        self.detail_dim = config.encoder.detail_dim  # 32D
        self.full_dim = self.core_dim + self.detail_dim  # 48D

        # Initialize components
        self.gp = CoarseToFineGP(config.gp)
        self.acquisition = DensityAwareAcquisition(config.acquisition)
        self.cycle_checker = CycleConsistencyChecker(
            encoder, decoder, config.cycle, adaptive=True
        )

        # Detail retriever (initialized in initialize() with training data)
        self.detail_retriever: Optional[SimpleDetailRetriever] = None
        self.detail_mode = config.residual_latent.detail_mode

        self.device = config.device
        self.best_z_core = None
        self.best_z_detail = None
        self.best_embedding = None
        self.best_score = float("-inf")
        self._initialized = False

        # History for analysis
        self.history: List[Dict[str, Any]] = []

    @property
    def best_z(self) -> Optional[torch.Tensor]:
        """Backwards compatibility: returns best_z_core."""
        return self.best_z_core

    @property
    def best_z_full(self) -> Optional[torch.Tensor]:
        """Get best full latent [z_core, z_detail] (48D)."""
        if self.best_z_core is None or self.best_z_detail is None:
            return None
        return torch.cat([self.best_z_core, self.best_z_detail], dim=-1)

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
        training_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Initialize GP with initial observations and set up detail retriever.

        Args:
            initial_embeddings: Known embeddings [N, 768]
            initial_scores: Corresponding objective values [N]
            training_embeddings: Full training set for detail retriever [M, 768]
                                If None, uses initial_embeddings
        """
        initial_embeddings = initial_embeddings.to(self.device)
        initial_scores = initial_scores.to(self.device)

        # Encode to residual latent space
        with torch.no_grad():
            z_core, z_detail = self.encoder.encode_deterministic_full(initial_embeddings)

        # Set up detail retriever from training data
        if training_embeddings is not None:
            training_embeddings = training_embeddings.to(self.device)
            with torch.no_grad():
                train_z_core, train_z_detail = self.encoder.encode_deterministic_full(
                    training_embeddings
                )
        else:
            # Use initial embeddings as training set
            train_z_core, train_z_detail = z_core, z_detail

        self.detail_retriever = create_detail_retriever(
            z_cores=train_z_core,
            z_details=train_z_detail,
            mode=self.detail_mode,
            device=self.device,
            use_faiss=len(train_z_core) > 50000,
        )

        # Enable residual latent mode on cycle checker
        self.cycle_checker.set_detail_retriever(
            self.detail_retriever,
            core_dim=self.core_dim,
        )

        print(f"[DetailRetriever] Mode={self.detail_mode}, "
              f"training set size={len(train_z_core)}")

        # Initialize GP with z_core only (GP operates on 16D)
        self.gp.fit(z_core, initial_scores)

        # Track best
        best_idx = initial_scores.argmax()
        self.best_z_core = z_core[best_idx]
        self.best_z_detail = z_detail[best_idx]
        self.best_embedding = initial_embeddings[best_idx]
        self.best_score = initial_scores[best_idx].item()

        # Calibrate cycle checker on z_core (checking core consistency)
        self.cycle_checker.calibrate(z_core)

        self._initialized = True

        print(f"Initialized with {len(initial_scores)} points")
        print(f"  Best initial score: {self.best_score:.4f}")
        print(f"  GP stage: {self.gp.current_stage} ({self.gp.n_active_dims}D)")
        print(f"  Latent: z_core={self.core_dim}D, z_detail={self.detail_dim}D")

    def step(
        self,
        objective: Callable[[torch.Tensor], float],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform one optimization step.

        Residual Latent Workflow:
        1. GP generates z_core candidates (16D)
        2. Cycle checker retrieves z_detail and validates decode
        3. GP is updated with z_core only
        4. Best tracking maintains z_core, z_detail, and embedding

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

        # Generate z_core candidates (GP operates on 16D z_core)
        candidates = self.acquisition.generate_candidates(
            self.gp, self.best_z_core
        )
        _, acq_values = self.acquisition.select_best_candidates(
            self.gp, candidates, n_select=len(candidates)
        )

        # Sort by acquisition value
        sorted_indices = acq_values.argsort(descending=True)
        candidates_sorted = candidates[sorted_indices]
        acq_sorted = acq_values[sorted_indices]

        # Select valid candidate using cycle consistency
        # Cycle checker handles z_detail retrieval internally for residual mode
        z_core_selected, x_decoded, cycle_error, n_tried = (
            self.cycle_checker.select_valid_from_ranked(
                candidates_sorted,
                acq_sorted,
                active_dims=self.gp.active_dims,  # Pass list of indices, not length
            )
        )

        # Evaluate objective
        with torch.no_grad():
            score = objective(x_decoded)

        # Update GP with z_core only (GP operates on 16D)
        self.gp.update(
            z_core_selected.unsqueeze(0),
            torch.tensor([score], device=self.device),
        )

        # Update best if improved
        if score > self.best_score:
            self.best_z_core = z_core_selected
            # Get corresponding z_detail for best tracking
            if self.detail_retriever is not None:
                self.best_z_detail = self.detail_retriever.get_detail(
                    z_core_selected.unsqueeze(0)
                ).squeeze(0)
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

        Residual Latent Architecture:
        - GP optimizes over z_core (16D) only
        - z_detail (32D) is retrieved from training set via nearest neighbor
        - Returns z_core for continued optimization; use best_z_full for full latent

        Args:
            initial_embeddings: Starting embeddings [N, 768]
            initial_scores: Initial objective values [N]
            objective: Objective function taking embedding [768] → score
            n_iterations: Number of BO iterations
            verbose: Print progress

        Returns:
            best_z_core: Best z_core latent [core_dim] (16D)
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

        return self.best_z_core, self.best_embedding, self.best_score

    def get_top_k(self, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k observations from GP.

        Returns:
            z_core: Top-k z_core latents [k, core_dim] (16D)
            scores: Top-k scores [k]
        """
        scores = self.gp.train_y
        top_indices = scores.argsort(descending=True)[:k]

        return self.gp.train_z[top_indices], scores[top_indices]

    def decode_latent(self, z: torch.Tensor, is_z_core: bool = True) -> torch.Tensor:
        """
        Decode latent to embedding (deterministic).

        For residual latent mode:
        - If is_z_core=True (default), retrieves z_detail and decodes z_full
        - If is_z_core=False or z is already 48D, decodes directly

        Args:
            z: Latent codes [B, latent_dim] - z_core (16D) or z_full (48D)
            is_z_core: Whether z is z_core only (requires detail retrieval)

        Returns:
            embeddings: [B, 768]
        """
        with torch.no_grad():
            if z.dim() == 1:
                z = z.unsqueeze(0)

            # Handle residual latent mode
            if is_z_core and self.detail_retriever is not None and z.shape[-1] == self.core_dim:
                z_full = self.detail_retriever.get_full_latent(z)
                return self.decoder.decode_deterministic(z_full)
            else:
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

    def decode_to_text(self, z: torch.Tensor, is_z_core: bool = True) -> List[str]:
        """
        Decode latent to text via embedding → vec2text.

        For residual latent mode:
        - If is_z_core=True (default), retrieves z_detail automatically

        Args:
            z: Latent codes [B, latent_dim] - z_core (16D) or z_full (48D)
            is_z_core: Whether z is z_core only

        Returns:
            texts: List of decoded texts
        """
        embeddings = self.decode_latent(z, is_z_core=is_z_core)
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
