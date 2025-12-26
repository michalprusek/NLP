"""
Latent space optimization for HyLO2.

Key innovation: Optimize directly in the 10D latent space instead of 768D embedding space.

Algorithm:
    1. Initialize latent point (from best observed prompt or training latents)
    2. Gradient ascent on latent to maximize EI (with bounds)
    3. After optimization, project latent to 768D using LatentProjector
    4. Select best exemplar by scanning all exemplars with projected instruction
    5. Pass projected embedding to Vec2Text for text inversion

Advantages over 768D optimization:
    - Lower dimensionality (10 vs 768) = faster, fewer local minima
    - No backprop through FeatureExtractor = simpler gradient computation
    - On-manifold optimization = latent points stay in learned distribution
    - Better Vec2Text alignment = LatentProjector learned training distribution
"""
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field

from .gp_model import GPTrainer2


@dataclass
class LatentOptimizationResult:
    """Results from latent space optimization."""
    optimized_latent: torch.Tensor              # (10,) optimized latent point
    projected_instruction_emb: torch.Tensor     # (768,) projected to embedding space
    selected_exemplar_idx: int
    selected_exemplar_emb: torch.Tensor         # (768,)
    final_ei: float
    trajectory: List[Dict] = field(default_factory=list)
    n_iterations: int = 0
    strategy: str = "latent_space"


class LatentSpaceOptimizer:
    """Optimize directly in the 10D latent space.

    Key difference from CoordinateDescentOptimizer:
    - Original: Optimize 768D instruction_emb -> FeatureExtractor -> latent -> GP -> EI
    - New: Optimize 10D latent directly -> GP -> EI

    The optimization is bounded to stay within the training latent distribution,
    which ensures the projected embeddings are in a valid region for Vec2Text.
    """

    def __init__(
        self,
        gp_trainer: GPTrainer2,
        exemplar_embeddings: np.ndarray,
        n_steps: int = 500,
        lr: float = 0.1,                        # Higher LR for 10D space
        convergence_threshold: float = 1e-6,
        max_iterations: int = 10,
        n_restarts: int = 5,
        perturbation_scale: float = 0.1,
        device: torch.device = None,
        use_log_ei: bool = False,
        ei_epsilon: float = 1e-8,
        latent_bounds_sigma: float = 3.0,       # Bound search to +/- 3 sigma
        use_latent_bounds: bool = True          # Whether to apply bounds
    ):
        """
        Args:
            gp_trainer: Trained GPTrainer2 instance
            exemplar_embeddings: (M, 768) all exemplar embeddings
            n_steps: Gradient steps per optimization run
            lr: Learning rate for gradient ascent (higher for 10D vs 768D)
            convergence_threshold: EI improvement threshold for convergence
            max_iterations: Maximum optimization iterations (for multi-start)
            n_restarts: Number of random restarts
            perturbation_scale: Scale of noise for perturbations
            device: Torch device
            use_log_ei: Use log(EI) for better gradient flow
            ei_epsilon: Epsilon for numerical stability
            latent_bounds_sigma: Constrain optimization to +/- N sigma
        """
        self.gp_trainer = gp_trainer
        self.exemplar_embeddings = torch.tensor(
            exemplar_embeddings, dtype=torch.float32, device=device
        )
        self.n_steps = n_steps
        self.lr = lr
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.n_restarts = n_restarts
        self.perturbation_scale = perturbation_scale
        self.device = device or torch.device("cpu")
        self.use_log_ei = use_log_ei
        self.ei_epsilon = ei_epsilon
        self.latent_bounds_sigma = latent_bounds_sigma
        self.use_latent_bounds = use_latent_bounds
        self.num_exemplars = len(exemplar_embeddings)

    def _get_latent_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bounds for latent space optimization."""
        return self.gp_trainer.get_latent_bounds(sigma=self.latent_bounds_sigma)

    def _project_to_bounds(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent point to valid bounds (if enabled)."""
        if not self.use_latent_bounds:
            return latent  # No bounds - allow extrapolation
        lower, upper = self._get_latent_bounds()
        return torch.clamp(latent, lower, upper)

    def _optimize_latent_single(
        self,
        init_latent: torch.Tensor,
        vmin_b: float,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """Single gradient ascent optimization run on latent point.

        Args:
            init_latent: (10,) starting latent point
            vmin_b: Best observed error rate
            verbose: Print progress

        Returns:
            (optimized_latent, best_ei, ei_history)
        """
        # Clone and enable gradients
        latent = init_latent.clone().to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=self.lr)

        best_ei = -float('inf')
        best_latent = None
        history = []
        steps_without_improvement = 0
        perturbation_interval = self.n_steps // 5  # Perturb 5 times

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Compute EI directly in latent space
            ei = self.gp_trainer.compute_ei_in_latent_space(
                latent.squeeze(),
                vmin_b,
                use_log_ei=self.use_log_ei,
                ei_epsilon=self.ei_epsilon
            )

            # Gradient ASCENT: minimize -EI
            loss = -ei
            loss.backward()
            optimizer.step()

            # Project to bounds after gradient step
            with torch.no_grad():
                latent.data = self._project_to_bounds(latent.data)

            ei_val = ei.item()
            history.append(ei_val)

            if ei_val > best_ei:
                best_ei = ei_val
                best_latent = latent.detach().clone()
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # Basin hopping: add noise if stuck
            if steps_without_improvement > perturbation_interval:
                with torch.no_grad():
                    noise = torch.randn_like(latent) * self.perturbation_scale
                    latent.data = self._project_to_bounds(best_latent.clone() + noise)
                steps_without_improvement = 0
                if verbose:
                    print(f"    Basin hop at step {step}")

            if verbose and step % 100 == 0:
                print(f"  Step {step}: EI = {ei_val:.6f}")

        return best_latent.squeeze(), best_ei, history

    def _scan_exemplars(
        self,
        projected_instruction_emb: torch.Tensor,
        vmin_b: float
    ) -> Tuple[int, torch.Tensor, float]:
        """Find best exemplar for given projected instruction embedding.

        After projecting the optimized latent to 768D, we scan all exemplars
        to find the one that maximizes EI when paired with the instruction.

        Args:
            projected_instruction_emb: (768,) projected instruction embedding
            vmin_b: Best observed error rate

        Returns:
            (best_idx, best_emb, best_ei)
        """
        p = self.gp_trainer.gp_params

        best_ei = -float('inf')
        best_idx = 0

        # Move instruction to device
        inst = projected_instruction_emb.to(self.device)
        if inst.dim() == 1:
            inst = inst.unsqueeze(0)

        for i in range(self.num_exemplars):
            exemplar_emb = self.exemplar_embeddings[i].to(self.device)
            if exemplar_emb.dim() == 1:
                exemplar_emb = exemplar_emb.unsqueeze(0)

            # Concatenate and normalize
            X = torch.cat([inst, exemplar_emb], dim=1)
            denom = p.X_max - p.X_min
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            X_norm = (X - p.X_min) / denom

            # Get latent and compute EI
            with torch.no_grad():
                inst_n = X_norm[:, :768]
                ex_n = X_norm[:, 768:]
                lat = p.feature_extractor(inst_n, ex_n)
                ei = self.gp_trainer.compute_ei_in_latent_space(
                    lat.squeeze(), vmin_b
                ).item()

            if ei > best_ei:
                best_ei = ei
                best_idx = i

        return best_idx, self.exemplar_embeddings[best_idx], best_ei

    def optimize(
        self,
        init_instruction_emb: torch.Tensor,
        init_exemplar_idx: int,
        vmin_b: float,
        top_k_samples: Optional[List[Dict]] = None,
        instruction_embeddings: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> LatentOptimizationResult:
        """Run multi-start latent space optimization.

        Args:
            init_instruction_emb: (768,) starting instruction embedding
            init_exemplar_idx: Starting exemplar index
            vmin_b: Best observed error rate
            top_k_samples: Optional list of top K samples to start from
                           Each dict has 'instruction_id' and 'exemplar_id'
            instruction_embeddings: (N, 768) instruction embeddings (needed for top_k)
            verbose: Print progress

        Returns:
            LatentOptimizationResult with optimized latent and projected embedding
        """
        p = self.gp_trainer.gp_params

        if verbose:
            print(f"Starting Latent Space Optimization")
            print(f"Latent dim: {self.gp_trainer.latent_dim}, Steps: {self.n_steps}")
            bounds_str = f"+/- {self.latent_bounds_sigma} sigma" if self.use_latent_bounds else "DISABLED"
            print(f"Bounds: {bounds_str}")

        # Get initial latent from instruction + exemplar
        init_exemplar_emb = self.exemplar_embeddings[init_exemplar_idx].to(self.device)
        init_instruction_emb = init_instruction_emb.to(self.device)

        with torch.no_grad():
            init_latent = self.gp_trainer.get_latent_features(
                init_instruction_emb,
                init_exemplar_emb
            )

        trajectory = []

        # Record initial state
        init_ei = self.gp_trainer.compute_ei_in_latent_space(init_latent, vmin_b).item()
        trajectory.append({
            'iteration': 0,
            'phase': 'initial',
            'latent': init_latent.cpu().numpy().copy(),
            'ei': init_ei,
            'exemplar_idx': init_exemplar_idx
        })

        if verbose:
            print(f"Initial EI (best sample): {init_ei:.6f}")

        # Collect starting points from TOP K evaluated prompts
        start_latents = []

        if top_k_samples is not None and instruction_embeddings is not None:
            # Start from top K evaluated prompts
            if verbose:
                print(f"Multi-start from top {len(top_k_samples)} evaluated prompts:")

            for i, sample in enumerate(top_k_samples):
                inst_id = sample['instruction_id']
                ex_id = sample['exemplar_id']
                error = sample.get('error_rate', 0)

                inst_emb = torch.tensor(
                    instruction_embeddings[inst_id],
                    dtype=torch.float32, device=self.device
                )
                ex_emb = self.exemplar_embeddings[ex_id].to(self.device)

                with torch.no_grad():
                    latent = self.gp_trainer.get_latent_features(inst_emb, ex_emb)
                    ei = self.gp_trainer.compute_ei_in_latent_space(latent, vmin_b).item()

                start_latents.append((f'top{i+1}_i{inst_id}_e{ex_id}', latent.clone()))

                if verbose:
                    print(f"  {i+1}. inst={inst_id}, ex={ex_id}, error={error:.4f}, EI={ei:.6f}")
        else:
            # Fallback: just use initial + random training latents
            start_latents.append(('initial', init_latent.clone()))
            for i in range(min(self.n_restarts, len(p.train_latents))):
                idx = np.random.randint(len(p.train_latents))
                start_latents.append((f'train_{idx}', p.train_latents[idx].clone()))

        if verbose:
            print(f"Total starting points: {len(start_latents)}")

        # Run optimization from each start
        best_latent = None
        best_ei = -float('inf')
        best_start_name = ""

        for start_name, start_latent in start_latents:
            if verbose:
                print(f"\n  Running from {start_name}...")

            opt_latent, opt_ei, history = self._optimize_latent_single(
                start_latent, vmin_b, verbose=False
            )

            if verbose:
                print(f"    -> EI = {opt_ei:.6f}")

            if opt_ei > best_ei:
                best_ei = opt_ei
                best_latent = opt_latent
                best_start_name = start_name

        trajectory.append({
            'iteration': -1,
            'phase': 'latent_optimized',
            'latent': best_latent.cpu().numpy().copy(),
            'ei': best_ei,
            'start_name': best_start_name
        })

        # Project optimized latent to instruction embedding
        projected_inst = self.gp_trainer.project_latent_to_embedding(
            best_latent, denormalize=True
        )

        if verbose:
            print(f"\nProjecting latent to 768D embedding...")

        # Select best exemplar
        best_ex_idx, best_ex_emb, final_ei = self._scan_exemplars(
            projected_inst, vmin_b
        )

        trajectory.append({
            'iteration': -1,
            'phase': 'final',
            'latent': best_latent.cpu().numpy().copy(),
            'ei': final_ei,
            'exemplar_idx': best_ex_idx
        })

        if verbose:
            print(f"\n{'='*50}")
            print(f"Latent Space Optimization Complete!")
            print(f"Best start: {best_start_name}")
            print(f"Latent EI: {best_ei:.6f}")
            print(f"Final EI (with exemplar {best_ex_idx}): {final_ei:.6f}")

        return LatentOptimizationResult(
            optimized_latent=best_latent,
            projected_instruction_emb=projected_inst,
            selected_exemplar_idx=int(best_ex_idx),
            selected_exemplar_emb=best_ex_emb.squeeze(),
            final_ei=float(final_ei),
            trajectory=trajectory,
            n_iterations=self.n_steps * len(start_latents),
            strategy="latent_space"
        )
