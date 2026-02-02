"""
Latent Space Bayesian Optimization for prompt optimization.

Key insight: Do BO in flow's noise space z ~ N(0, I) instead of embedding space x.
The GP works better in Gaussian z-space, and the flow handles manifold structure.

Two modes:
1. Pure Latent BO: GP in z-space, unguided flow z → x
2. Hybrid Latent BO: GP in z-space + guided flow refinement in x-space

Usage:
    >>> latent_bo = LatentSpaceBO(flow_model, decoder, evaluator, llm_client)
    >>> latent_bo.initialize_from_embeddings(X_init, Y_init, instructions)
    >>> for _ in range(100):
    ...     result = latent_bo.step()
"""

import logging
from typing import Optional

import gpytorch
import torch
import torch.nn.functional as F
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from ecoflow.decoder import SonarDecoder
from ecoflow.flow_model import FlowMatchingModel
from ecoflow.guided_flow import GuidedFlowSampler
from ecoflow.gp_surrogate import SonarGPSurrogate
from shared.gsm8k_evaluator import GSM8KEvaluator
from shared.llm_client import LLMClient

logger = logging.getLogger(__name__)


class LatentSpaceGP:
    """
    GP surrogate operating in flow's latent (noise) space.

    Since z ~ N(0, I), standard Matern kernel works naturally.
    No need for ArcCosine or geodesic kernels.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        device: str = "cuda",
        lengthscale_init: float = 1.0,
    ):
        self.input_dim = input_dim
        self.device = device
        self.lengthscale_init = lengthscale_init
        self.model: Optional[SingleTaskGP] = None
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        # Normalization stats (set during fit)
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.y_min: float = 0.0
        self.y_max: float = 1.0

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Fit GP on (z, y) pairs with robust numerical handling."""
        self.train_X = X.to(self.device).double()  # Use double precision for stability
        self.train_Y = Y.to(self.device).double().unsqueeze(-1)  # [N, 1] for BoTorch

        # Check for NaN/Inf in inputs
        if torch.isnan(self.train_X).any() or torch.isinf(self.train_X).any():
            logger.warning("Training X contains NaN/Inf! Replacing with randn.")
            nan_mask = torch.isnan(self.train_X) | torch.isinf(self.train_X)
            self.train_X[nan_mask] = torch.randn_like(self.train_X)[nan_mask]

        if torch.isnan(self.train_Y).any() or torch.isinf(self.train_Y).any():
            logger.warning("Training Y contains NaN/Inf! Replacing with mean.")
            valid_y = self.train_Y[~torch.isnan(self.train_Y) & ~torch.isinf(self.train_Y)]
            fill_val = valid_y.mean() if len(valid_y) > 0 else 0.5
            nan_mask = torch.isnan(self.train_Y) | torch.isinf(self.train_Y)
            self.train_Y[nan_mask] = fill_val

        # Normalize X to prevent scale issues (z-scores are already ~N(0,1) but enforce)
        self.x_mean = self.train_X.mean(dim=0, keepdim=True)
        self.x_std = self.train_X.std(dim=0, keepdim=True).clamp(min=1e-6)
        train_X_norm = (self.train_X - self.x_mean) / self.x_std

        # Standardize Y to [0, 1] range for better GP conditioning
        self.y_min = self.train_Y.min()
        self.y_max = self.train_Y.max()
        y_range = (self.y_max - self.y_min).clamp(min=1e-6)
        train_Y_norm = (self.train_Y - self.y_min) / y_range

        try:
            # Standard GP with Matern kernel - works well in Gaussian z-space
            self.model = SingleTaskGP(
                train_X_norm,
                train_Y_norm,
            ).to(self.device)

            # Set reasonable lengthscale for normalized z-space
            if hasattr(self.model.covar_module, 'base_kernel'):
                self.model.covar_module.base_kernel.lengthscale = self.lengthscale_init
            else:
                self.model.covar_module.lengthscale = self.lengthscale_init

            # Use BoTorch's robust fitting (handles numerical issues)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)

            self.model.eval()
            logger.debug(f"GP fitted on {len(X)} points (BoTorch fit)")

        except Exception as e:
            logger.warning(f"GP fitting failed: {e}. Using fallback with high noise.")
            # Fallback: create simple GP with high observation noise
            self.model = SingleTaskGP(
                train_X_norm,
                train_Y_norm,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.model.likelihood.noise = 0.1  # High noise for stability
            self.model.eval()
            logger.debug("GP fallback: using fixed high-noise likelihood")

    def update(self, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Update GP with new observations."""
        X_new = X_new.to(self.device).double()
        Y_new = Y_new.to(self.device).double()

        if self.train_X is None:
            self.fit(X_new, Y_new)
        else:
            # Store unnormalized for refitting
            train_X_unnorm = self.train_X  # Already stored unnormalized
            train_Y_unnorm = self.train_Y.squeeze(-1)

            new_X = torch.cat([train_X_unnorm, X_new], dim=0)
            new_Y = torch.cat([train_Y_unnorm, Y_new], dim=0)
            self.fit(new_X, new_Y)

    def optimize_acquisition(
        self,
        n_candidates: int = 512,
        n_restarts: int = 10,
        alpha: float = 1.96,
    ) -> torch.Tensor:
        """
        Find z* = argmax UCB(z) in latent space.

        Returns:
            Optimal z* [1, D] in original (unnormalized) space
        """
        if self.model is None:
            raise RuntimeError("GP not fitted yet")

        acq = UpperConfidenceBound(self.model, beta=alpha**2)

        # Bounds for normalized z-space: use ~3 sigma in normalized space
        bounds = torch.tensor(
            [[-3.0] * self.input_dim, [3.0] * self.input_dim],
            device=self.device,
            dtype=torch.double,
        )

        try:
            # Optimize acquisition in normalized space
            z_opt_norm, _ = optimize_acqf(
                acq,
                bounds=bounds,
                q=1,
                num_restarts=n_restarts,
                raw_samples=n_candidates,
            )

            # Convert back to original space
            z_opt = z_opt_norm * self.x_std + self.x_mean

        except Exception as e:
            logger.warning(f"Acquisition optimization failed: {e}. Using random sample.")
            # Fallback: random sample in original z-space
            z_opt = torch.randn(1, self.input_dim, device=self.device, dtype=torch.double)

        return z_opt.float()  # Return float32 for flow model compatibility

    def sample_candidates(
        self,
        n_samples: int,
        alpha: float = 1.96,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample multiple z candidates weighted by UCB.

        Args:
            n_samples: Number of candidates
            alpha: UCB exploration weight
            temperature: Sampling temperature (higher = more exploration)

        Returns:
            z candidates [n_samples, D] in original space
        """
        if self.model is None:
            # No GP yet, sample from prior
            return torch.randn(n_samples, self.input_dim, device=self.device)

        # Generate random candidates in normalized space
        z_candidates_norm = torch.randn(
            n_samples * 10, self.input_dim,
            device=self.device, dtype=torch.double
        )

        # Score with UCB
        self.model.eval()
        with torch.no_grad():
            try:
                posterior = self.model.posterior(z_candidates_norm)
                mean = posterior.mean.squeeze(-1)
                std = posterior.variance.sqrt().squeeze(-1)
                ucb = mean + alpha * std

                # Handle NaN in UCB scores
                if torch.isnan(ucb).any():
                    logger.warning("UCB contains NaN, using random selection")
                    indices = torch.randperm(len(z_candidates_norm))[:n_samples]
                else:
                    # Sample proportional to UCB (softmax sampling)
                    probs = F.softmax(ucb / temperature, dim=0)
                    indices = torch.multinomial(probs, n_samples, replacement=False)

                z_selected_norm = z_candidates_norm[indices]

            except Exception as e:
                logger.warning(f"UCB scoring failed: {e}. Using random samples.")
                indices = torch.randperm(len(z_candidates_norm))[:n_samples]
                z_selected_norm = z_candidates_norm[indices]

        # Convert back to original space
        z_selected = z_selected_norm * self.x_std + self.x_mean

        return z_selected.float()  # Return float32 for flow model compatibility


class LatentSpaceBO:
    """
    Bayesian Optimization in flow's latent (noise) space.

    Key advantages:
    1. z ~ N(0, I) is well-suited for standard GP kernels
    2. No complex manifold geometry to handle
    3. Flow handles the transformation to valid embeddings

    Two modes:
    - use_guided_refinement=False: Pure latent BO (simple, fast)
    - use_guided_refinement=True: Hybrid with guided flow refinement
    """

    def __init__(
        self,
        flow_model: FlowMatchingModel,
        decoder: SonarDecoder,
        evaluator: GSM8KEvaluator,
        llm_client: LLMClient,
        gp_x_space: Optional[SonarGPSurrogate] = None,
        guided_sampler: Optional[GuidedFlowSampler] = None,
        eval_subset_size: int = 150,
        device: str = "cuda",
        use_guided_refinement: bool = False,
    ):
        """
        Initialize Latent Space BO.

        Args:
            flow_model: Trained FlowMatchingModel
            decoder: SonarDecoder for embedding-to-text
            evaluator: GSM8KEvaluator for scoring
            llm_client: LLM for evaluation
            gp_x_space: Optional GP in x-space for guided refinement
            guided_sampler: Optional GuidedFlowSampler for refinement
            eval_subset_size: GSM8K subset size
            device: Computation device
            use_guided_refinement: Use guided flow after z-space optimization
        """
        self.flow_model = flow_model
        self.decoder = decoder
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.device = device
        self.use_guided_refinement = use_guided_refinement

        # GP in z-space (primary)
        self.gp_z = LatentSpaceGP(input_dim=1024, device=device)

        # Optional: GP in x-space for guided refinement
        self.gp_x = gp_x_space
        self.guided_sampler = guided_sampler

        # State
        self.train_Z: Optional[torch.Tensor] = None  # Latent codes
        self.train_X: Optional[torch.Tensor] = None  # Embeddings (for reference)
        self.train_Y: Optional[torch.Tensor] = None  # Scores
        self.best_score: float = 0.0
        self.best_prompt: str = ""
        self.best_z: Optional[torch.Tensor] = None
        self.iteration: int = 0

        # Fixed evaluation indices
        import random
        dataset_size = len(evaluator)
        self.eval_subset_size = min(eval_subset_size, dataset_size)
        self.eval_indices = random.sample(range(dataset_size), self.eval_subset_size)

        self._prompts: dict[int, str] = {}

    def invert_embeddings(
        self,
        embeddings: torch.Tensor,
        num_steps: int = 100,
        verify: bool = True,
        max_error: float = 0.5,  # Relaxed threshold
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Invert embeddings x → z via reverse ODE.

        Args:
            embeddings: SONAR embeddings [N, 1024]
            num_steps: ODE integration steps (more = more accurate)
            verify: Verify inversion quality via forward pass
            max_error: Maximum allowed reconstruction error

        Returns:
            Tuple of (z_codes, reconstruction_errors)
        """
        logger.info(f"Inverting {len(embeddings)} embeddings to latent space...")

        z_codes_list = []
        errors_list = []

        # Process one embedding at a time for better error handling
        for i, emb in enumerate(embeddings):
            try:
                with torch.no_grad():
                    # Use flow's encode method (reverse ODE: t=1 → t=0)
                    z = self.flow_model.encode(
                        emb.unsqueeze(0).to(self.device),
                        method="heun",
                        num_steps=num_steps,
                        normalize=True,  # Normalize to training scale first
                    )

                    # Check for NaN/Inf
                    if torch.isnan(z).any() or torch.isinf(z).any():
                        logger.warning(f"  Sample {i}: inversion produced NaN/Inf, using random z")
                        z = torch.randn(1, 1024, device=self.device)

                    # Verify this single sample
                    if verify:
                        x_rec = self._flow_forward(z)
                        error = (emb.to(self.device) - x_rec.squeeze(0)).norm().item()
                        if error > max_error:
                            logger.debug(f"  Sample {i}: high error={error:.4f}, using random z")
                            z = torch.randn(1, 1024, device=self.device)
                            error = float('nan')  # Mark as invalid
                    else:
                        error = 0.0

                    z_codes_list.append(z.squeeze(0))
                    errors_list.append(error)

            except Exception as e:
                logger.warning(f"  Sample {i}: inversion failed ({e}), using random z")
                z_codes_list.append(torch.randn(1024, device=self.device))
                errors_list.append(float('nan'))

        z_codes = torch.stack(z_codes_list)
        errors = torch.tensor(errors_list, device=self.device)

        # Report statistics
        valid_errors = errors[~torch.isnan(errors)]
        if len(valid_errors) > 0:
            mean_error = valid_errors.mean().item()
            max_observed = valid_errors.max().item()
            logger.info(f"  Inversion quality: mean_error={mean_error:.4f}, max={max_observed:.4f}")
            logger.info(f"  Valid inversions: {len(valid_errors)}/{len(embeddings)}")
        else:
            logger.warning("  All inversions failed! Using random initialization.")

        return z_codes, errors

    def _flow_forward(self, z: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """Transform z → x via flow (forward ODE)."""
        # We need to integrate from t=0 to t=1 starting from z
        # The flow_model.sample() generates new z from randn, but we want to use our z

        # Manually integrate
        with torch.no_grad():
            x = z.clone()

            # For spherical flows, z should already be on unit sphere
            if self.flow_model.is_spherical:
                x = F.normalize(x, p=2, dim=-1)

            x = self.flow_model._integrate(x, method="heun", num_steps=num_steps, forward=True)
            x = self.flow_model.denormalize(x)

        return x

    def _flow_forward_guided(
        self,
        z: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Transform z → x via guided flow (uses x-space GP for guidance)."""
        if self.guided_sampler is None or self.gp_x is None:
            return self._flow_forward(z, num_steps)

        # Update x-space GP in guided sampler
        self.guided_sampler.update_gp(self.gp_x)

        # Use guided sampling starting from z
        with torch.no_grad():
            x = z.clone()
            if self.flow_model.is_spherical:
                x = F.normalize(x, p=2, dim=-1)

            # Guided integration
            x = self.guided_sampler._integrate_guided(
                x,
                num_steps=num_steps,
                method="heun",
            )
            x = self.flow_model.denormalize(x)

        return x

    def initialize_from_embeddings(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
        instructions: list[str],
        inversion_steps: int = 100,
    ) -> dict:
        """
        Initialize from pre-evaluated embeddings (warm start).

        This is the ONLY place where inversion is needed!
        During optimization, we propose z directly.

        Args:
            embeddings: SONAR embeddings [N, 1024]
            scores: Accuracy scores [N]
            instructions: Text instructions [N]
            inversion_steps: ODE steps for inversion
        """
        logger.info(f"Initializing Latent BO from {len(embeddings)} pre-evaluated points...")

        # Invert embeddings to z-space (ONE-TIME COST)
        z_codes, errors = self.invert_embeddings(
            embeddings,
            num_steps=inversion_steps,
            verify=True,
        )

        # Store state
        self.train_Z = z_codes.to(self.device)
        self.train_X = embeddings.to(self.device)
        self.train_Y = scores.to(self.device)

        # Track best
        best_idx = scores.argmax().item()
        self.best_score = scores[best_idx].item()
        self.best_prompt = instructions[best_idx]
        self.best_z = z_codes[best_idx:best_idx+1]

        for i, instr in enumerate(instructions):
            self._prompts[i] = instr

        # Fit GP in z-space
        self.gp_z.fit(self.train_Z, self.train_Y)

        # Optionally fit x-space GP for guided refinement
        if self.use_guided_refinement and self.gp_x is not None:
            self.gp_x.fit(self.train_X, self.train_Y)
            if self.guided_sampler is not None:
                self.guided_sampler.update_gp(self.gp_x)

        logger.info(f"Initialization complete: {len(scores)} points, best={self.best_score:.4f}")
        logger.info(f"Best prompt: {self.best_prompt}")

        return {
            "n_samples": len(scores),
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
            "mean_inversion_error": errors.mean().item(),
        }

    def initialize_from_random(self, n_initial: int = 1) -> dict:
        """
        Initialize from random z samples (cold start).

        Generates random z ~ N(0, I), transforms to x via flow,
        decodes to text, and evaluates.

        Args:
            n_initial: Number of initial random samples to evaluate
        """
        logger.info(f"Cold start: generating {n_initial} random samples...")

        z_list = []
        x_list = []
        y_list = []
        prompt_list = []

        for i in range(n_initial):
            # Sample random z
            z = torch.randn(1, 1024, device=self.device)

            # For spherical flows, normalize to unit sphere
            if self.flow_model.is_spherical:
                z = F.normalize(z, p=2, dim=-1)

            # Transform z → x via flow
            x = self._flow_forward(z)

            # Decode to text
            try:
                prompts = self.decoder.decode(x)
                prompt = prompts[0]
            except Exception as e:
                logger.warning(f"Decode failed for sample {i}: {e}")
                prompt = "Solve the problem step by step."

            # Evaluate
            score = self._evaluate_prompt(prompt)

            z_list.append(z.squeeze(0))
            x_list.append(x.squeeze(0))
            y_list.append(score)
            prompt_list.append(prompt)

            logger.info(f"  Sample {i}: score={score:.4f}, z_norm={z.norm().item():.2f}")
            logger.info(f"    Prompt: {prompt[:100]}...")

        # Store state
        self.train_Z = torch.stack(z_list).to(self.device)
        self.train_X = torch.stack(x_list).to(self.device)
        self.train_Y = torch.tensor(y_list, device=self.device)

        # Track best
        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_prompt = prompt_list[best_idx]
        self.best_z = self.train_Z[best_idx:best_idx+1]

        for i, prompt in enumerate(prompt_list):
            self._prompts[i] = prompt

        # Fit GP in z-space (need at least 2 points for GP)
        if n_initial >= 2:
            self.gp_z.fit(self.train_Z, self.train_Y)
        else:
            logger.info("Only 1 initial sample - GP will be fitted after first iteration")

        logger.info(f"Cold start complete: {n_initial} samples, best={self.best_score:.4f}")

        return {
            "n_samples": n_initial,
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
        }

    def _is_valid_prompt(self, text: str) -> bool:
        """Check if prompt is valid for evaluation."""
        if not text or len(text) < 10:
            return False
        words = text.split()
        if len(words) < 3:
            return False
        # Check for excessive repetition
        if words:
            word_counts: dict[str, int] = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5 and len(words) > 5:
                return False
        return True

    def _evaluate_prompt(self, prompt: str) -> float:
        """Evaluate prompt on GSM8K subset."""
        if not self._is_valid_prompt(prompt):
            return 0.0

        formatted_questions = []
        for idx in self.eval_indices:
            example = self.evaluator.dataset[idx]
            question = example["question"]
            formatted = f"Q: {question}\n{prompt}\nA:"
            formatted_questions.append(formatted)

        try:
            outputs = self.llm_client.generate_batch(
                formatted_questions,
                max_new_tokens=512,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return 0.0

        result = self.evaluator.evaluate_batch(outputs, self.eval_indices)
        return result["accuracy"]

    def step(
        self,
        alpha: float = 1.96,
        n_candidates: int = 512,
        n_restarts: int = 10,
    ) -> dict:
        """
        Execute one Latent BO iteration.

        Algorithm:
        1. Optimize UCB in z-space to get z*
        2. Transform z* → x via flow (optionally guided)
        3. Decode x → text prompt
        4. Evaluate prompt on GSM8K
        5. Update GP with (z*, score)

        Args:
            alpha: UCB exploration weight
            n_candidates: Candidates for acquisition optimization
            n_restarts: Optimization restarts

        Returns:
            Dict with iteration results
        """
        self.iteration += 1

        # 1. Optimize acquisition in z-space (or sample randomly if GP not fitted)
        if self.gp_z.model is None:
            # GP not fitted yet - sample random z
            logger.info("GP not fitted, sampling random z...")
            z_new = torch.randn(1, 1024, device=self.device, dtype=torch.double)
            if self.flow_model.is_spherical:
                z_new = F.normalize(z_new, p=2, dim=-1)
            z_new = z_new.float()
        else:
            z_new = self.gp_z.optimize_acquisition(
                n_candidates=n_candidates,
                n_restarts=n_restarts,
                alpha=alpha,
            )

        # 2. Transform z → x via flow
        if self.use_guided_refinement:
            x_new = self._flow_forward_guided(z_new)
        else:
            x_new = self._flow_forward(z_new)

        # 3. Decode to text
        try:
            prompts = self.decoder.decode(x_new)
            prompt = prompts[0]
        except Exception as e:
            logger.warning(f"Decode failed: {e}")
            prompt = ""

        # 4. Evaluate
        score = self._evaluate_prompt(prompt)

        # 5. Update state first
        self.train_Z = torch.cat([self.train_Z, z_new], dim=0)
        self.train_X = torch.cat([self.train_X, x_new], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device)], dim=0)

        # 6. Refit GP with ALL data (more robust than incremental update)
        self.gp_z.fit(self.train_Z, self.train_Y)

        # Optionally update x-space GP for guided refinement
        if self.use_guided_refinement and self.gp_x is not None:
            self.gp_x.update(x_new, torch.tensor([score], device=self.device))

        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            self.best_z = z_new
            logger.info(f"NEW BEST: {self.best_score:.4f}")

        self._prompts[len(self._prompts)] = prompt

        logger.info(
            f"Iter {self.iteration}: score={score:.4f}, best={self.best_score:.4f}, "
            f"n_obs={len(self.train_Z)}"
        )
        logger.info(f"Prompt: {prompt}")

        return {
            "iteration": self.iteration,
            "score": score,
            "best_so_far": self.best_score,
            "best_prompt": self.best_prompt,
            "n_observations": len(self.train_Z),
            "prompt": prompt,
            "z_norm": z_new.norm().item(),
            "x_norm": x_new.norm().item(),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save state to checkpoint."""
        state = {
            "train_Z": self.train_Z.cpu(),
            "train_X": self.train_X.cpu(),
            "train_Y": self.train_Y.cpu(),
            "best_score": self.best_score,
            "best_prompt": self.best_prompt,
            "best_z": self.best_z.cpu() if self.best_z is not None else None,
            "iteration": self.iteration,
            "eval_indices": self.eval_indices,
            "prompts": self._prompts,
        }
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load state from checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.train_Z = state["train_Z"].to(self.device)
        self.train_X = state["train_X"].to(self.device)
        self.train_Y = state["train_Y"].to(self.device)
        self.best_score = state["best_score"]
        self.best_prompt = state["best_prompt"]
        self.best_z = state["best_z"].to(self.device) if state["best_z"] is not None else None
        self.iteration = state["iteration"]
        self.eval_indices = state["eval_indices"]
        self._prompts = state.get("prompts", {})

        # Re-fit GPs
        self.gp_z.fit(self.train_Z, self.train_Y)
        if self.use_guided_refinement and self.gp_x is not None:
            self.gp_x.fit(self.train_X, self.train_Y)

        logger.info(f"Loaded checkpoint: iteration={self.iteration}, best={self.best_score:.4f}")
