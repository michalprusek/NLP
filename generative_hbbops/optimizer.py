"""
Embedding optimization strategies for HyLO.

Strategy A (Coordinate Descent):
    1. Fix exemplar embedding
    2. Optimize instruction embedding via gradient ascent on EI
    3. Scan all exemplars to find best match
    4. Repeat until convergence

Strategy B (Gumbel-Softmax):
    1. Jointly optimize instruction embedding + soft exemplar weights
    2. Use Gumbel-Softmax with temperature annealing
    3. Temperature decreases to force hard selection
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field

from .gp_model import GPTrainer


@dataclass
class OptimizationResult:
    """Results from embedding optimization."""
    optimized_instruction_emb: torch.Tensor  # (768,)
    selected_exemplar_idx: int
    selected_exemplar_emb: torch.Tensor  # (768,)
    final_ei: float
    trajectory: List[Dict] = field(default_factory=list)
    n_iterations: int = 0
    strategy: str = ""


class CoordinateDescentOptimizer:
    """Strategy A: Coordinate Descent optimization with multi-start.

    Algorithm:
        1. Multi-start: Try multiple starting points (all instructions + random)
        2. For each start:
            a. Fix current exemplar
            b. Gradient ascent on instruction embedding to maximize EI
            c. Scan all exemplars, select one that maximizes EI
            d. Basin hopping: periodically add noise to escape local maxima
            e. Check convergence
        3. Return best result across all starts

    This strategy guarantees valid exemplars since we always select
    from the discrete set of available exemplars.
    """

    def __init__(
        self,
        gp_trainer: GPTrainer,
        exemplar_embeddings: np.ndarray,
        instruction_embeddings: np.ndarray = None,
        n_steps: int = 500,
        lr: float = 0.01,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 10,
        n_restarts: int = 5,
        perturbation_scale: float = 0.1,
        device: torch.device = None,
        gradient_clip_norm: Optional[float] = None,
        use_log_ei: bool = False,
        ei_epsilon: float = 1e-8
    ):
        """
        Args:
            gp_trainer: Trained GPTrainer instance
            exemplar_embeddings: (M, 768) all exemplar embeddings
            instruction_embeddings: (K, 768) all instruction embeddings for multi-start
            n_steps: Gradient steps per coordinate descent iteration
            lr: Learning rate for gradient ascent
            convergence_threshold: EI improvement threshold for convergence
            max_iterations: Maximum coordinate descent iterations
            n_restarts: Number of random restarts for global search
            perturbation_scale: Scale of noise for basin hopping
            device: Torch device
            gradient_clip_norm: Max gradient norm for clipping (None = disabled)
            use_log_ei: Use log(EI) instead of EI for better gradient flow
            ei_epsilon: Epsilon for numerical stability
        """
        self.gp_trainer = gp_trainer
        self.exemplar_embeddings = torch.tensor(
            exemplar_embeddings, dtype=torch.float32, device=device
        )
        self.instruction_embeddings = None
        if instruction_embeddings is not None:
            self.instruction_embeddings = torch.tensor(
                instruction_embeddings, dtype=torch.float32, device=device
            )
        self.n_steps = n_steps
        self.lr = lr
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.n_restarts = n_restarts
        self.perturbation_scale = perturbation_scale
        self.device = device or torch.device("cpu")
        self.num_exemplars = len(exemplar_embeddings)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_log_ei = use_log_ei
        self.ei_epsilon = ei_epsilon

    def _optimize_instruction(
        self,
        init_instruction_emb: torch.Tensor,
        fixed_exemplar_emb: torch.Tensor,
        vmin_b: float,
        verbose: bool = False,
        use_basin_hopping: bool = True
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """Gradient ascent on instruction embedding with fixed exemplar.

        Uses basin hopping: periodically adds noise to escape local maxima.

        Args:
            init_instruction_emb: Starting instruction embedding
            fixed_exemplar_emb: Fixed exemplar embedding (not optimized)
            vmin_b: Best observed error rate
            verbose: Print progress
            use_basin_hopping: Add periodic perturbation

        Returns:
            (optimized_emb, best_ei, ei_history)
        """
        # Clone and enable gradients
        instruction_emb = init_instruction_emb.clone().to(self.device).requires_grad_(True)
        exemplar_emb = fixed_exemplar_emb.clone().to(self.device).detach()

        optimizer = torch.optim.Adam([instruction_emb], lr=self.lr)

        best_ei = -float('inf')
        best_embedding = None
        history = []
        steps_without_improvement = 0
        perturbation_interval = self.n_steps // 5  # Perturb 5 times during optimization

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Compute EI (differentiable)
            ei = self.gp_trainer.compute_ei_differentiable(
                instruction_emb.squeeze(),
                exemplar_emb.squeeze(),
                vmin_b,
                use_log_ei=self.use_log_ei,
                ei_epsilon=self.ei_epsilon
            )

            # Gradient ASCENT: minimize -EI
            loss = -ei
            loss.backward()

            # Gradient clipping for stability
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([instruction_emb], self.gradient_clip_norm)

            optimizer.step()

            ei_val = ei.item()
            history.append(ei_val)

            if ei_val > best_ei:
                best_ei = ei_val
                best_embedding = instruction_emb.detach().clone()
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # Basin hopping: add noise if stuck
            if use_basin_hopping and steps_without_improvement > perturbation_interval:
                with torch.no_grad():
                    # Add Gaussian noise scaled by perturbation_scale
                    noise = torch.randn_like(instruction_emb) * self.perturbation_scale
                    instruction_emb.data = best_embedding.clone() + noise
                steps_without_improvement = 0
                if verbose:
                    print(f"    Basin hop at step {step}")

            if verbose and step % 100 == 0:
                print(f"  Step {step}: EI = {ei_val:.6f}")

        return best_embedding.squeeze(), best_ei, history

    def _scan_exemplars(
        self,
        instruction_emb: torch.Tensor,
        vmin_b: float
    ) -> Tuple[int, torch.Tensor, float]:
        """Find best exemplar for given instruction embedding.

        Scans all exemplars and returns the one with highest EI.

        Args:
            instruction_emb: Optimized instruction embedding
            vmin_b: Best observed error rate

        Returns:
            (best_idx, best_emb, best_ei)
        """
        # Expand instruction to batch
        inst_batch = instruction_emb.unsqueeze(0).expand(self.num_exemplars, -1)

        # Compute EI for all exemplars
        ei_values = self.gp_trainer.compute_ei_batch(
            inst_batch,
            self.exemplar_embeddings,
            vmin_b
        )

        # Ensure we get Python int, not tensor
        if isinstance(ei_values, torch.Tensor):
            ei_values = ei_values.detach().cpu().numpy()
        best_idx = int(np.argmax(ei_values))
        best_ei = float(ei_values[best_idx])
        best_emb = self.exemplar_embeddings[best_idx]

        return best_idx, best_emb, best_ei

    def _run_single_optimization(
        self,
        init_instruction_emb: torch.Tensor,
        init_exemplar_idx: int,
        vmin_b: float,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, int, torch.Tensor, float, List[Dict], int]:
        """Run single coordinate descent from given start point."""
        current_inst_emb = init_instruction_emb.clone().to(self.device)
        # Ensure index is Python int
        current_ex_idx = int(init_exemplar_idx)
        current_ex_emb = self.exemplar_embeddings[current_ex_idx]

        prev_ei = -float('inf')
        trajectory = []

        # Record initial state
        with torch.no_grad():
            init_ei = self.gp_trainer.compute_ei_differentiable(
                current_inst_emb.squeeze(),
                current_ex_emb.squeeze(),
                vmin_b
            ).item()

        trajectory.append({
            'iteration': 0,
            'phase': 'initial',
            'exemplar_idx': current_ex_idx,
            'ei': init_ei,
            'instruction_emb': current_inst_emb.cpu().numpy().copy(),
            'exemplar_emb': current_ex_emb.cpu().numpy().copy()
        })

        for iteration in range(1, self.max_iterations + 1):
            # Optimize instruction with basin hopping
            current_inst_emb, inst_ei, ei_history = self._optimize_instruction(
                current_inst_emb,
                current_ex_emb,
                vmin_b,
                verbose=False,
                use_basin_hopping=True
            )

            trajectory.append({
                'iteration': iteration,
                'phase': 'instruction_optimization',
                'exemplar_idx': current_ex_idx,
                'ei': inst_ei,
                'instruction_emb': current_inst_emb.cpu().numpy().copy(),
                'exemplar_emb': current_ex_emb.cpu().numpy().copy()
            })

            # Scan all exemplars
            new_ex_idx, new_ex_emb, scan_ei = self._scan_exemplars(current_inst_emb, vmin_b)
            new_ex_idx = int(new_ex_idx)  # Ensure Python int

            trajectory.append({
                'iteration': iteration,
                'phase': 'exemplar_scan',
                'exemplar_idx': new_ex_idx,
                'ei': scan_ei,
                'instruction_emb': current_inst_emb.cpu().numpy().copy(),
                'exemplar_emb': new_ex_emb.cpu().numpy().copy()
            })

            current_ex_idx = new_ex_idx
            current_ex_emb = new_ex_emb

            # Check convergence
            ei_improvement = scan_ei - prev_ei
            if ei_improvement < self.convergence_threshold and iteration > 1:
                break
            prev_ei = scan_ei

        with torch.no_grad():
            final_ei = self.gp_trainer.compute_ei_differentiable(
                current_inst_emb.squeeze(),
                current_ex_emb.squeeze(),
                vmin_b
            ).item()

        return current_inst_emb, current_ex_idx, current_ex_emb, final_ei, trajectory, iteration

    def optimize(
        self,
        init_instruction_emb: torch.Tensor,
        init_exemplar_idx: int,
        vmin_b: float,
        verbose: bool = True
    ) -> OptimizationResult:
        """Run multi-start coordinate descent optimization.

        Tries multiple starting points:
        1. The provided initial embedding
        2. All available instruction embeddings (if provided)
        3. Random perturbations of best points

        Args:
            init_instruction_emb: Starting instruction embedding
            init_exemplar_idx: Starting exemplar index
            vmin_b: Best observed error rate
            verbose: Print progress

        Returns:
            OptimizationResult with best embeddings across all starts
        """
        if verbose:
            print(f"Starting Multi-Start Coordinate Descent optimization")
            print(f"Restarts: {self.n_restarts}, Basin hopping scale: {self.perturbation_scale}")

        # Collect all starting points
        start_points = []

        # 1. Original starting point
        start_points.append((init_instruction_emb.clone(), init_exemplar_idx, "initial"))

        # 2. All instruction embeddings (try each with all exemplars)
        if self.instruction_embeddings is not None:
            for inst_idx in range(len(self.instruction_embeddings)):
                inst_emb = self.instruction_embeddings[inst_idx]
                # Find best exemplar for this instruction
                best_ex_idx, _, _ = self._scan_exemplars(inst_emb, vmin_b)
                start_points.append((inst_emb.clone(), int(best_ex_idx), f"inst_{inst_idx}"))

        # 3. Random perturbations
        for r in range(self.n_restarts):
            # Random instruction from available ones
            if self.instruction_embeddings is not None:
                rand_idx = int(np.random.randint(len(self.instruction_embeddings)))
                base_emb = self.instruction_embeddings[rand_idx].clone()
            else:
                base_emb = init_instruction_emb.clone()

            # Add noise
            noise = torch.randn_like(base_emb) * self.perturbation_scale * 2
            perturbed = base_emb + noise

            # Find best exemplar
            best_ex_idx, _, _ = self._scan_exemplars(perturbed, vmin_b)
            start_points.append((perturbed, int(best_ex_idx), f"random_{r}"))

        if verbose:
            print(f"Total starting points: {len(start_points)}")

        # Run optimization from each start
        best_result = None
        best_ei = -float('inf')
        all_trajectories = []

        for i, (start_emb, start_ex_idx, start_name) in enumerate(start_points):
            if verbose and i % 10 == 0:
                print(f"\n  Start {i+1}/{len(start_points)} ({start_name})...")

            inst_emb, ex_idx, ex_emb, final_ei, trajectory, n_iter = self._run_single_optimization(
                start_emb, start_ex_idx, vmin_b, verbose=False
            )

            if verbose and i % 10 == 0:
                print(f"    -> EI = {final_ei:.6f}, exemplar = {ex_idx}")

            if final_ei > best_ei:
                best_ei = final_ei
                best_result = (inst_emb, ex_idx, ex_emb, trajectory, n_iter, start_name)

            all_trajectories.append({
                'start_name': start_name,
                'final_ei': final_ei,
                'exemplar_idx': ex_idx
            })

        inst_emb, ex_idx, ex_emb, trajectory, n_iter, start_name = best_result

        if verbose:
            print(f"\n{'='*50}")
            print(f"Multi-Start Complete!")
            print(f"Best start: {start_name}")
            print(f"Final exemplar: {ex_idx}, Final EI: {best_ei:.6f}")

            # Show top 5 results
            sorted_results = sorted(all_trajectories, key=lambda x: x['final_ei'], reverse=True)
            print(f"\nTop 5 results:")
            for j, r in enumerate(sorted_results[:5]):
                print(f"  {j+1}. {r['start_name']}: EI={r['final_ei']:.6f}, ex={r['exemplar_idx']}")

        return OptimizationResult(
            optimized_instruction_emb=inst_emb.squeeze(),
            selected_exemplar_idx=ex_idx,
            selected_exemplar_emb=ex_emb.squeeze(),
            final_ei=best_ei,
            trajectory=trajectory,
            n_iterations=n_iter,
            strategy="coordinate_descent_multistart"
        )


class GumbelSoftmaxOptimizer:
    """Strategy B: Gumbel-Softmax joint optimization.

    Algorithm:
        1. Initialize instruction embedding + uniform exemplar weights
        2. While temperature > final_temperature:
            a. Compute soft exemplar embedding as weighted average
            b. Compute EI for (instruction, soft_exemplar)
            c. Gradient ascent on both instruction_emb and logits
            d. Anneal temperature
        3. Final selection: argmax of logits

    This strategy can find non-obvious combinations by allowing smooth
    optimization over the exemplar space.
    """

    def __init__(
        self,
        gp_trainer: GPTrainer,
        exemplar_embeddings: np.ndarray,
        n_steps: int = 1000,
        lr: float = 0.01,
        initial_temperature: float = 5.0,
        final_temperature: float = 0.1,
        anneal_rate: float = 0.99,
        device: torch.device = None,
        gradient_clip_norm: Optional[float] = None,
        use_log_ei: bool = False,
        ei_epsilon: float = 1e-8
    ):
        """
        Args:
            gp_trainer: Trained GPTrainer instance
            exemplar_embeddings: (M, 768) all exemplar embeddings
            n_steps: Total optimization steps
            lr: Learning rate
            initial_temperature: Starting softmax temperature
            final_temperature: Final temperature after annealing
            anneal_rate: Temperature decay rate per step
            device: Torch device
            gradient_clip_norm: Max gradient norm for clipping (None = disabled)
            use_log_ei: Use log(EI) instead of EI for better gradient flow
            ei_epsilon: Epsilon for numerical stability
        """
        self.gp_trainer = gp_trainer
        self.exemplar_embeddings = torch.tensor(
            exemplar_embeddings, dtype=torch.float32, device=device
        )
        self.n_steps = n_steps
        self.lr = lr
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_rate = anneal_rate
        self.device = device or torch.device("cpu")
        self.num_exemplars = len(exemplar_embeddings)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_log_ei = use_log_ei
        self.ei_epsilon = ei_epsilon

    def _gumbel_softmax(
        self,
        logits: torch.Tensor,
        temperature: float,
        hard: bool = False
    ) -> torch.Tensor:
        """Sample from Gumbel-Softmax distribution.

        Gumbel-Softmax provides a differentiable approximation to argmax.
        As temperature -> 0, the output approaches one-hot.

        Args:
            logits: (M,) unnormalized log-probabilities
            temperature: Softmax temperature (lower = sharper)
            hard: If True, return straight-through one-hot

        Returns:
            (M,) soft or hard selection weights
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logits) + 1e-20
        ) + 1e-20)

        # Gumbel-Softmax sample
        y = F.softmax((logits + gumbel_noise) / temperature, dim=0)

        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y)
            y_hard[y.argmax()] = 1.0
            y = (y_hard - y).detach() + y

        return y

    def _compute_soft_exemplar(
        self,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted average of exemplar embeddings.

        Args:
            weights: (M,) Gumbel-Softmax weights (sum to 1)

        Returns:
            (768,) weighted exemplar embedding
        """
        # weights: (M,), exemplar_embeddings: (M, 768)
        return torch.sum(weights.unsqueeze(1) * self.exemplar_embeddings, dim=0)

    def optimize(
        self,
        init_instruction_emb: torch.Tensor,
        vmin_b: float,
        init_exemplar_idx: int = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """Run Gumbel-Softmax joint optimization.

        Args:
            init_instruction_emb: Starting instruction embedding
            vmin_b: Best observed error rate
            init_exemplar_idx: Starting exemplar index (for trajectory visualization)
            verbose: Print progress

        Returns:
            OptimizationResult with final embeddings
        """
        # Initialize instruction embedding
        instruction_emb = init_instruction_emb.clone().to(self.device).requires_grad_(True)

        # Initialize uniform logits (log probabilities)
        logits = torch.zeros(self.num_exemplars, device=self.device, requires_grad=True)

        # If initial exemplar provided, bias logits towards it
        if init_exemplar_idx is not None:
            with torch.no_grad():
                logits[init_exemplar_idx] = 2.0  # Slight bias towards initial exemplar

        # Optimizer for both
        optimizer = torch.optim.Adam([instruction_emb, logits], lr=self.lr)

        temperature = self.initial_temperature
        trajectory = []
        best_ei = -float('inf')
        best_instruction_emb = None
        best_exemplar_idx = None

        if verbose:
            print(f"Starting Gumbel-Softmax optimization")
            print(f"Initial temperature: {temperature:.2f}")

        # Record TRUE initial state (with actual best exemplar, not weighted)
        if init_exemplar_idx is not None:
            with torch.no_grad():
                init_exemplar_emb = self.exemplar_embeddings[init_exemplar_idx]
                init_ei = self.gp_trainer.compute_ei_differentiable(
                    instruction_emb.squeeze().detach(),
                    init_exemplar_emb,
                    vmin_b
                ).item()
            trajectory.append({
                'iteration': -1,  # Before optimization
                'phase': 'initial',
                'exemplar_idx': init_exemplar_idx,
                'ei': init_ei,
                'instruction_emb': instruction_emb.detach().cpu().numpy().copy(),
                'exemplar_emb': init_exemplar_emb.cpu().numpy().copy(),
                'temperature': temperature,
                'soft_ei': init_ei,
                'top_exemplar_prob': 1.0
            })

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Sample weights using Gumbel-Softmax
            weights = self._gumbel_softmax(logits, temperature, hard=False)

            # Compute soft exemplar embedding
            soft_exemplar_emb = self._compute_soft_exemplar(weights)

            # Compute EI
            ei = self.gp_trainer.compute_ei_differentiable(
                instruction_emb.squeeze(),
                soft_exemplar_emb,
                vmin_b,
                use_log_ei=self.use_log_ei,
                ei_epsilon=self.ei_epsilon
            )

            # Gradient ascent
            loss = -ei
            loss.backward()

            # Gradient clipping for stability
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([instruction_emb, logits], self.gradient_clip_norm)

            optimizer.step()

            ei_val = ei.item()

            # Track best (using hard selection for evaluation)
            with torch.no_grad():
                hard_idx = logits.argmax().item()
                hard_exemplar_emb = self.exemplar_embeddings[hard_idx]
                hard_ei = self.gp_trainer.compute_ei_differentiable(
                    instruction_emb.squeeze().detach(),
                    hard_exemplar_emb,
                    vmin_b
                ).item()

                if hard_ei > best_ei:
                    best_ei = hard_ei
                    best_instruction_emb = instruction_emb.detach().clone()
                    best_exemplar_idx = hard_idx

            # Record trajectory every step for smooth visualization
            if step % 1 == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=0)
                    top_idx = int(logits.argmax().item())
                    top_prob = probs[top_idx].item()
                    current_exemplar_emb = self.exemplar_embeddings[top_idx]

                # Store in format compatible with visualizer
                trajectory.append({
                    'iteration': step,
                    'phase': 'gumbel_softmax',
                    'exemplar_idx': top_idx,
                    'ei': hard_ei,
                    'instruction_emb': instruction_emb.detach().cpu().numpy().copy(),
                    'exemplar_emb': current_exemplar_emb.cpu().numpy().copy(),
                    'temperature': temperature,
                    'soft_ei': ei_val,
                    'top_exemplar_prob': top_prob
                })

                if verbose and step % 100 == 0:
                    print(f"Step {step}: temp={temperature:.3f}, "
                          f"soft_EI={ei_val:.6f}, hard_EI={hard_ei:.6f}, "
                          f"top_ex={top_idx} (p={top_prob:.3f})")

            # Anneal temperature
            temperature = max(
                self.final_temperature,
                temperature * self.anneal_rate
            )

        # Final selection
        with torch.no_grad():
            final_idx = logits.argmax().item()
            final_exemplar_emb = self.exemplar_embeddings[final_idx]
            final_ei = self.gp_trainer.compute_ei_differentiable(
                best_instruction_emb.squeeze(),
                final_exemplar_emb,
                vmin_b
            ).item()

        if verbose:
            print(f"\nGumbel-Softmax complete!")
            print(f"Final exemplar: {final_idx}, Final EI: {final_ei:.6f}")
            print(f"Best EI achieved: {best_ei:.6f} with exemplar {best_exemplar_idx}")

        return OptimizationResult(
            optimized_instruction_emb=best_instruction_emb.squeeze(),
            selected_exemplar_idx=best_exemplar_idx,
            selected_exemplar_emb=self.exemplar_embeddings[best_exemplar_idx].squeeze(),
            final_ei=best_ei,
            trajectory=trajectory,
            n_iterations=self.n_steps,
            strategy="gumbel_softmax"
        )
