#!/usr/bin/env python3
"""
FlowPO-HD: Manifold-Guided High-Dimensional Prompt Optimization.

Main optimization loop that combines:
1. GP-based acquisition in 1024D SONAR space
2. ManifoldKeeper velocity field to stay on instruction manifold
3. TuRBO trust regions for high-dimensional optimization

Usage:
    # Full run with LLM evaluation
    uv run python -m flowpo_hd.scripts.run_flowpo_hd --iterations 50

    # Skip LLM evaluation (test optimization mechanics)
    uv run python -m flowpo_hd.scripts.run_flowpo_hd --iterations 10 --skip-llm-eval

    # With custom ManifoldKeeper checkpoint
    uv run python -m flowpo_hd.scripts.run_flowpo_hd --manifold-keeper-path path/to/checkpoint.pt
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from flowpo_hd.config import FlowPOHDConfig, get_device
from flowpo_hd.flow_guided_acquisition import (
    FlowGuidedAcquisition,
    SaasFlowGuidedAcquisition,
    create_flow_guided_acquisition,
    create_saas_flow_guided_acquisition,
)
from flowpo_hd.improved_gp import (
    VanillaFlowGuidedAcquisition,
    VanillaGPWithAcquisition,
    create_vanilla_flow_guided_acquisition,
)
from flowpo_hd.manifold_keeper import ManifoldKeeperMLP, create_manifold_keeper
from flowpo_hd.turbo_1024 import TuRBO1024, create_turbo_1024
from flowpo_hd.utils import (
    SONARHelper,
    save_results,
    set_seed,
    setup_logging,
)
from flowpo_hd.warm_start import load_warm_start, WarmStartData
from flowpo_hd.saas_gp import SaasGPWithAcquisition, SaasConfig, create_saas_gp

# Import GP from existing infrastructure (fallback for non-SAAS)
from lido_pp.gp.high_dim_gp import IsotropicHighDimGP, AdaptiveHighDimGP

logger = logging.getLogger(__name__)


class FlowPOHDOptimizer:
    """
    Main optimizer for FlowPO-HD.

    Combines:
    - IsotropicHighDimGP / AdaptiveHighDimGP for acquisition
    - ManifoldKeeper for manifold guidance
    - TuRBO-1024 for trust region management
    - FlowGuidedAcquisition for optimization
    """

    def __init__(
        self,
        config: FlowPOHDConfig,
        manifold_keeper: ManifoldKeeperMLP,
        sonar_helper: SONARHelper,
    ):
        """
        Initialize optimizer.

        Args:
            config: FlowPOHDConfig
            manifold_keeper: Trained ManifoldKeeper model
            sonar_helper: SONAR encoder/decoder helper
        """
        self.config = config
        self.device = torch.device(config.device)
        self.manifold_keeper = manifold_keeper.to(self.device)
        self.manifold_keeper.eval()
        self.sonar_helper = sonar_helper

        # Create GP
        logger.info(f"Creating GP (type={config.gp_type})...")
        if config.gp_type == "vanilla":
            # VanillaGP with dimension-scaled prior (benchmark winner: 96% coverage)
            # See FINDINGS.md - best calibration, Hvarfner 2024
            self.gp = VanillaGPWithAcquisition(device=config.device)
            self.use_saas = False  # Not SAAS but uses similar interface
            self.use_vanilla = True
        elif config.gp_type == "saas":
            # SAAS GP with qLogEI (good Spearman 0.82, but 73% coverage)
            saas_config = SaasConfig(
                warmup_steps=config.saas_warmup_steps,
                num_samples=config.saas_num_samples,
                thinning=config.saas_thinning,
                raw_samples=config.saas_raw_samples,
            )
            self.gp = create_saas_gp(config=saas_config, device=config.device)
            self.use_saas = True
            self.use_vanilla = False
        elif config.gp_type == "isotropic":
            self.gp = IsotropicHighDimGP(
                latent_dim=config.sonar_dim,
                device=config.device,
                ucb_beta=config.gp_ucb_beta_start,
                trust_region_scale=config.gp_trust_region_scale,
            )
            self.use_saas = False
            self.use_vanilla = False
        else:  # adaptive
            self.gp = AdaptiveHighDimGP(
                latent_dim=config.sonar_dim,
                device=config.device,
                switch_threshold=config.gp_switch_threshold,
                ucb_beta=config.gp_ucb_beta_start,
                trust_region_scale=config.gp_trust_region_scale,
            )
            self.use_saas = False
            self.use_vanilla = False

        # Create TuRBO manager
        logger.info("Creating TuRBO-1024...")
        self.turbo = create_turbo_1024(config)

        # Create flow-guided acquisition
        if config.gp_type == "vanilla":
            logger.info("Creating VanillaFlowGuidedAcquisition (qLogNEI + dimension-scaled prior)...")
            self.fga = create_vanilla_flow_guided_acquisition(config, manifold_keeper)
        elif config.gp_type == "saas":
            logger.info("Creating SaasFlowGuidedAcquisition (qLogNEI)...")
            self.fga = create_saas_flow_guided_acquisition(config, manifold_keeper)
        else:
            logger.info("Creating FlowGuidedAcquisition (UCB)...")
            self.fga = create_flow_guided_acquisition(config, manifold_keeper)

        # Training data
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.instructions: List[str] = []
        self.best_instruction: Optional[str] = None
        self.best_error_rate: float = 1.0

        # Global bounds (unit hypercube for normalized SONAR)
        self.global_bounds = torch.stack([
            torch.zeros(config.sonar_dim, device=self.device),
            torch.ones(config.sonar_dim, device=self.device),
        ])

        # Normalization stats (computed from initial data)
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None

    def _normalize_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize embedding to [0, 1]^d."""
        if self.X_min is None or self.X_max is None:
            return x

        # Ensure device consistency
        x = x.to(self.X_min.device)
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1.0
        return (x - self.X_min) / denom

    def _denormalize_embedding(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize embedding from [0, 1]^d."""
        if self.X_min is None or self.X_max is None:
            return x_norm

        # Ensure device consistency (SAAS GP on CPU, normalization stats may be on GPU)
        x_norm = x_norm.to(self.X_min.device)
        return x_norm * (self.X_max - self.X_min) + self.X_min

    def _update_normalization(self, X: torch.Tensor):
        """Update normalization stats."""
        if self.X_min is None:
            # Store on CPU to avoid GPU memory fragmentation
            self.X_min = X.min(dim=0).values.cpu()
            self.X_max = X.max(dim=0).values.cpu()
        else:
            X_cpu = X.cpu()
            self.X_min = torch.minimum(self.X_min, X_cpu.min(dim=0).values)
            self.X_max = torch.maximum(self.X_max, X_cpu.max(dim=0).values)

    def initialize_with_warm_start(self, warm_start_data: WarmStartData):
        """
        Initialize optimizer with warm-start data from HbBoPs.

        Uses ALL high-fidelity HbBoPs evaluations (not just top 10).

        Args:
            warm_start_data: WarmStartData with embeddings and error rates
        """
        logger.info(f"Initializing with {len(warm_start_data)} warm-start points...")
        logger.info(f"  Beta prior: α={warm_start_data.beta_alpha:.2f}, β={warm_start_data.beta_beta:.2f}")

        # Move data to device
        warm_start_data = warm_start_data.to(self.device)

        # Update normalization from warm-start embeddings
        self._update_normalization(warm_start_data.X)

        # Normalize
        X_norm = self._normalize_embedding(warm_start_data.X)

        # Store (use smoothed error rates from Beta posterior)
        self.X_train = X_norm
        self.y_train = warm_start_data.y
        self.instructions = warm_start_data.instructions
        self.warm_start_variances = warm_start_data.variances

        # Update best
        best_idx = self.y_train.argmin()
        self.best_instruction = self.instructions[best_idx]
        self.best_error_rate = self.y_train[best_idx].item()

        # Fit GP (use FGA's internal GP for SAAS/Vanilla)
        if self.use_saas or getattr(self, 'use_vanilla', False):
            # SaasFlowGuidedAcquisition and VanillaFlowGuidedAcquisition have their own internal GP
            self.fga.fit(self.X_train, self.y_train, variances=warm_start_data.variances)
        else:
            self.gp.fit(self.X_train, self.y_train)

        # Set TuRBO anchor
        self.turbo.set_anchor(self.X_train[best_idx], self.best_error_rate)

        logger.info(f"Initialized. Best error rate: {self.best_error_rate:.4f}")
        logger.info(f"Best instruction: {self.best_instruction[:100]}...")

    def initialize_with_seed_instructions(
        self,
        seed_instructions: List[str],
        error_rates: List[float],
    ):
        """
        Initialize optimizer with seed instructions and their error rates.

        Args:
            seed_instructions: List of seed instruction texts
            error_rates: Corresponding error rates
        """
        logger.info(f"Initializing with {len(seed_instructions)} seed instructions...")

        # Encode instructions
        embeddings = self.sonar_helper.encode(seed_instructions)
        embeddings = embeddings.to(self.device)

        # Update normalization
        self._update_normalization(embeddings)

        # Normalize
        X_norm = self._normalize_embedding(embeddings)

        # Store
        self.X_train = X_norm
        self.y_train = torch.tensor(error_rates, device=self.device)
        self.instructions = list(seed_instructions)

        # Update best
        best_idx = self.y_train.argmin()
        self.best_instruction = self.instructions[best_idx]
        self.best_error_rate = self.y_train[best_idx].item()

        # Fit GP
        self.gp.fit(self.X_train, self.y_train)

        # Set TuRBO anchor
        self.turbo.set_anchor(self.X_train[best_idx], self.best_error_rate)

        logger.info(f"Initialized. Best error rate: {self.best_error_rate:.4f}")
        logger.info(f"Best instruction: {self.best_instruction[:100]}...")

    def optimize_candidate(
        self,
        iteration: int,
        total_iterations: int,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate an optimized candidate instruction.

        Args:
            iteration: Current iteration
            total_iterations: Total iterations

        Returns:
            (embedding, decoded_instruction)
        """
        # Get adaptive parameters
        ucb_beta = self.config.get_ucb_beta(iteration, total_iterations)
        lambda_penalty = self.config.fga_lambda_penalty

        logger.info(f"Iteration {iteration}: ucb_beta={ucb_beta:.3f}, lambda_penalty={lambda_penalty:.5f}")

        # Get trust region bounds
        if self.config.turbo_enabled:
            lengthscales = None
            if hasattr(self.gp, 'gp_model') and self.gp.gp_model is not None:
                try:
                    lengthscales = self.gp.gp_model.covar_module.base_kernel.lengthscale.detach()
                    if lengthscales.dim() > 1:
                        lengthscales = lengthscales.squeeze()
                except Exception:
                    pass

            bounds = self.turbo.get_ard_scaled_bounds(self.global_bounds, lengthscales)
        else:
            bounds = self.global_bounds

        # Optimize with flow guidance
        if self.use_saas or getattr(self, 'use_vanilla', False):
            # SaasFlowGuidedAcquisition/VanillaFlowGuidedAcquisition use qLogNEI with internal GP
            z_opt, acq_value = self.fga.optimize(
                bounds=bounds,
                X_train=self.X_train,
                y_train=self.y_train,
            )
        else:
            # FlowGuidedAcquisition uses UCB gradients
            z_opt, acq_value = self.fga.optimize(
                gp=self.gp,
                bounds=bounds,
                lambda_penalty=lambda_penalty,
                ucb_beta=ucb_beta,
                X_train=self.X_train,
                y_train=self.y_train,
            )

        # === DIAGNOSTIC: GP Prediction for candidate ===
        self._log_candidate_diagnostics(z_opt, acq_value)

        # Denormalize
        z_opt_denorm = self._denormalize_embedding(z_opt)

        # Decode to instruction (SONAR expects float32 on CPU)
        z_opt_float32 = z_opt_denorm.to(device="cpu", dtype=torch.float32)
        # Ensure correct shape: (1, 1024)
        if z_opt_float32.dim() == 1:
            z_opt_float32 = z_opt_float32.unsqueeze(0)
        elif z_opt_float32.dim() > 2:
            z_opt_float32 = z_opt_float32.view(1, -1)

        logger.debug(f"  SONAR input: shape={z_opt_float32.shape}, device={z_opt_float32.device}")
        instruction = self.sonar_helper.decode(z_opt_float32)[0]

        logger.info(f"  Generated instruction: {instruction[:100]}...")

        return z_opt, instruction

    def _log_candidate_diagnostics(self, z_opt: torch.Tensor, acq_value: float):
        """Log comprehensive diagnostics for the optimized candidate."""
        z_opt_2d = z_opt.unsqueeze(0) if z_opt.dim() == 1 else z_opt

        # 1. GP Prediction for candidate
        if (self.use_saas or getattr(self, 'use_vanilla', False)) and hasattr(self.fga, '_gp') and self.fga._gp is not None:
            try:
                pred = self.fga._gp.predict(z_opt_2d)
                self._last_predicted_mean = pred.mean.item()
                self._last_predicted_std = pred.std.item()
                logger.info(f"  GP Prediction: mean={self._last_predicted_mean:.4f}, std={self._last_predicted_std:.4f}")
            except Exception as e:
                logger.warning(f"  GP prediction failed: {e}")
                self._last_predicted_mean = None
                self._last_predicted_std = None

        # 2. Distance to best known point
        if self.X_train is not None and len(self.X_train) > 0:
            best_idx = self.y_train.argmin()
            best_z = self.X_train[best_idx].to(z_opt.device)
            dist_to_best = torch.norm(z_opt.flatten() - best_z.flatten()).item()

            # Distance to nearest training point
            X_train_dev = self.X_train.to(z_opt.device)
            dists = torch.norm(X_train_dev - z_opt_2d, dim=1)
            min_dist = dists.min().item()
            nearest_idx = dists.argmin().item()
            nearest_error = self.y_train[nearest_idx].item()

            logger.info(f"  Distance to best (err={self.best_error_rate:.4f}): {dist_to_best:.4f}")
            logger.info(f"  Distance to nearest (err={nearest_error:.4f}): {min_dist:.4f}")

            # 3. How much candidate moved from best
            if dist_to_best < 0.01:
                logger.warning(f"  WARNING: Candidate very close to best! May be stuck.")

        # 4. Acquisition value analysis
        logger.info(f"  Acquisition value: {acq_value:.4f}")

    def add_observation(
        self,
        z_opt: torch.Tensor,
        instruction: str,
        error_rate: float,
    ):
        """
        Add a new observation and update models.

        Args:
            z_opt: Optimized embedding (normalized)
            instruction: Decoded instruction
            error_rate: Evaluated error rate
        """
        # === DIAGNOSTIC: Compare predicted vs actual ===
        self._log_prediction_error(error_rate)

        # Add to training data (ensure device consistency)
        z_opt = z_opt.unsqueeze(0) if z_opt.dim() == 1 else z_opt
        y_new = torch.tensor([error_rate], device=self.device)

        if self.X_train is None:
            self.X_train = z_opt.to(self.device)
            self.y_train = y_new
        else:
            z_opt = z_opt.to(self.X_train.device)
            self.X_train = torch.cat([self.X_train, z_opt], dim=0)
            self.y_train = torch.cat([self.y_train, y_new], dim=0)

        self.instructions.append(instruction)

        # Update best
        improved = error_rate < self.best_error_rate
        if improved:
            self.best_instruction = instruction
            self.best_error_rate = error_rate
            logger.info(f"  *** NEW BEST! Error rate: {error_rate:.4f} ***")

        # Refit GP
        if self.use_saas or getattr(self, 'use_vanilla', False):
            self.fga.fit(self.X_train, self.y_train)
        else:
            self.gp.fit(self.X_train, self.y_train)

    def _log_prediction_error(self, actual_error_rate: float):
        """Log the difference between GP prediction and actual error rate."""
        logger.info("=" * 60)
        logger.info("  POST-EVALUATION ANALYSIS")
        logger.info("=" * 60)

        if hasattr(self, '_last_predicted_mean') and self._last_predicted_mean is not None:
            pred_mean = self._last_predicted_mean
            pred_std = self._last_predicted_std

            # Prediction error
            pred_error = actual_error_rate - pred_mean
            pred_error_sigmas = pred_error / pred_std if pred_std > 0 else float('inf')

            logger.info(f"  GP predicted:  {pred_mean:.4f} ± {pred_std:.4f}")
            logger.info(f"  Actual result: {actual_error_rate:.4f}")
            logger.info(f"  Prediction error: {pred_error:+.4f} ({pred_error_sigmas:+.2f}σ)")

            # Interpretation
            if abs(pred_error_sigmas) > 2:
                logger.warning(f"  WARNING: GP was very wrong (>{2}σ off)!")
                if pred_error > 0:
                    logger.warning(f"  GP was OVERCONFIDENT - predicted better than reality")
                else:
                    logger.warning(f"  GP was UNDERCONFIDENT - reality was better than predicted")
            elif abs(pred_error_sigmas) < 0.5:
                logger.info(f"  GP prediction was accurate (<0.5σ)")

            # Did GP expect improvement?
            expected_improvement = self.best_error_rate - pred_mean
            actual_improvement = self.best_error_rate - actual_error_rate

            logger.info(f"  Expected improvement over best: {expected_improvement:+.4f}")
            logger.info(f"  Actual improvement over best:   {actual_improvement:+.4f}")

            if expected_improvement > 0 and actual_improvement <= 0:
                logger.warning(f"  PROBLEM: GP expected improvement but got worse!")
            elif expected_improvement <= 0:
                logger.info(f"  Note: GP did NOT expect improvement (exploration mode)")

        else:
            logger.info(f"  Actual result: {actual_error_rate:.4f}")
            logger.info(f"  (No GP prediction available)")

        logger.info("=" * 60)


def evaluate_instruction_with_llm(
    instruction: str,
    config: FlowPOHDConfig,
    llm_client=None,
    gsm8k_evaluator=None,
) -> float:
    """
    Evaluate an instruction on GSM8K using LLM.

    Args:
        instruction: Instruction to evaluate
        config: Configuration
        llm_client: Pre-created LLM client (created if None)
        gsm8k_evaluator: Pre-created GSM8K evaluator (created if None)

    Returns:
        Error rate (1 - accuracy)
    """
    from src.llm_client import create_llm_client
    from src.gsm8k_evaluator import GSM8KEvaluator
    import random

    # Create client if not provided
    if llm_client is None:
        llm_client = create_llm_client(config.eval_model, backend=config.eval_backend)

    # Create evaluator if not provided
    if gsm8k_evaluator is None:
        gsm8k_evaluator = GSM8KEvaluator(
            dataset_path="datasets/gsm8k",
            split="test",
        )

    # Sample random minibatch
    total_examples = len(gsm8k_evaluator)
    indices = random.sample(range(total_examples), min(config.eval_minibatch_size, total_examples))

    # Get questions
    questions = [gsm8k_evaluator.dataset[i]['question'] for i in indices]

    # Format prompts (Q_end style from OPRO)
    formatted_prompts = [f"Q: {q}\n{instruction}\nA:" for q in questions]

    # Generate answers
    outputs = llm_client.generate_batch(
        formatted_prompts,
        temperature=0.0,
        max_new_tokens=config.eval_max_tokens,
    )

    # Evaluate
    results = gsm8k_evaluator.evaluate_batch(outputs, indices)
    accuracy = results['accuracy']
    error_rate = 1.0 - accuracy

    logger.info(f"  Evaluation: accuracy={accuracy:.4f}, error_rate={error_rate:.4f}")

    return error_rate


def run_optimization(
    config: FlowPOHDConfig,
    iterations: int,
    hbbops_path: Optional[str] = None,
    min_fidelity: int = 600,
    skip_llm_eval: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run FlowPO-HD optimization loop.

    Args:
        config: FlowPOHDConfig
        iterations: Number of optimization iterations
        hbbops_path: Path to HbBoPs results for warm-start
        min_fidelity: Minimum fidelity for warm-start data (default 600)
        skip_llm_eval: Skip LLM evaluation (use random error rates)
        verbose: Verbose logging

    Returns:
        Results dict
    """
    device = torch.device(config.device)
    set_seed(config.seed)

    # Initialize SONAR helper on CPU to avoid GPU memory conflicts with vLLM
    # SONAR encode/decode is fast enough on CPU for occasional use
    logger.info("Initializing SONAR (on CPU to save GPU memory for vLLM)...")
    sonar_helper = SONARHelper(
        device="cpu",
        normalize=config.sonar_normalize,
    )

    # Load or create ManifoldKeeper
    manifold_keeper_path = Path(config.manifold_keeper_path)
    if manifold_keeper_path.exists():
        logger.info(f"Loading ManifoldKeeper from {manifold_keeper_path}...")
        checkpoint = torch.load(manifold_keeper_path, map_location=device)

        # Detect number of blocks from checkpoint
        block_keys = [k for k in checkpoint["model_state_dict"].keys() if k.startswith("blocks.")]
        num_blocks = max(int(k.split(".")[1]) for k in block_keys) + 1 if block_keys else config.mk_num_blocks

        # Create ManifoldKeeper with correct architecture
        from flowpo_hd.manifold_keeper import ManifoldKeeperMLP
        manifold_keeper = ManifoldKeeperMLP(
            dim=config.sonar_dim,
            hidden_dim=config.mk_hidden_dim,
            num_blocks=num_blocks,
            time_dim=config.mk_time_dim,
            dropout=config.mk_dropout,
        )
        manifold_keeper.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  Loaded ManifoldKeeper with {num_blocks} blocks")
    else:
        logger.warning(f"ManifoldKeeper not found at {manifold_keeper_path}")
        logger.warning("Creating untrained ManifoldKeeper (results may be poor)")
        manifold_keeper = create_manifold_keeper(config)

    # Create optimizer
    optimizer = FlowPOHDOptimizer(config, manifold_keeper, sonar_helper)

    # Load warm-start data from HbBoPs (use ALL high-fidelity evaluations)
    if hbbops_path and Path(hbbops_path).exists() and config.use_warm_start:
        logger.info(f"Loading warm-start data from {hbbops_path}...")
        logger.info(f"  Min fidelity: {min_fidelity}")

        warm_start_data = load_warm_start(
            hbbops_path=hbbops_path,
            min_fidelity=min_fidelity,
            device="cpu",  # Load on CPU, will move to GPU in optimizer
            cache_path=config.warm_start_cache_path,
        )

        logger.info(f"  Loaded {len(warm_start_data)} warm-start points")
        logger.info(f"  Error rate range: [{warm_start_data.y.min():.3f}, {warm_start_data.y.max():.3f}]")

        # Initialize with warm-start data
        optimizer.initialize_with_warm_start(warm_start_data)
    else:
        # Fallback to manual seed instructions
        logger.warning("No warm-start data, using default seed instructions")
        seed_instructions = [
            "Let's think step by step.",
            "Solve this problem carefully.",
            "Show your work and reasoning.",
            "Break down the problem into steps.",
            "Think through this logically.",
        ]
        seed_error_rates = [0.5, 0.52, 0.48, 0.51, 0.49]
        optimizer.initialize_with_seed_instructions(seed_instructions, seed_error_rates)

    # Results tracking
    results = {
        "config": vars(config),
        "iterations": [],
        "best_instruction": None,
        "best_error_rate": 1.0,
        "start_time": datetime.now().isoformat(),
    }

    # Create LLM client and evaluator once (if not skipping LLM eval)
    llm_client = None
    gsm8k_evaluator = None
    if not skip_llm_eval:
        from src.llm_client import create_llm_client
        from src.gsm8k_evaluator import GSM8KEvaluator

        logger.info("Initializing LLM client...")
        llm_client = create_llm_client(config.eval_model, backend=config.eval_backend)

        logger.info("Loading GSM8K test set...")
        gsm8k_evaluator = GSM8KEvaluator(
            dataset_path="datasets/gsm8k",
            split="test",
        )
        logger.info(f"GSM8K test set: {len(gsm8k_evaluator)} examples")

    # Main optimization loop
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting FlowPO-HD optimization for {iterations} iterations")
    logger.info(f"{'='*60}\n")

    for i in range(iterations):
        logger.info(f"\n--- Iteration {i+1}/{iterations} ---")

        # Generate candidate
        z_opt, instruction = optimizer.optimize_candidate(i, iterations)

        # Evaluate
        if skip_llm_eval:
            # Random error rate for testing
            error_rate = 0.3 + 0.4 * torch.rand(1).item()
            logger.info(f"  [SKIP] Random error rate: {error_rate:.4f}")
        else:
            error_rate = evaluate_instruction_with_llm(
                instruction, config, llm_client, gsm8k_evaluator
            )

        # Add observation
        optimizer.add_observation(z_opt, instruction, error_rate)

        # Record
        results["iterations"].append({
            "iteration": i + 1,
            "instruction": instruction,
            "error_rate": error_rate,
            "best_so_far": optimizer.best_error_rate,
        })

        # Update best
        if optimizer.best_error_rate < results["best_error_rate"]:
            results["best_instruction"] = optimizer.best_instruction
            results["best_error_rate"] = optimizer.best_error_rate

        # Log progress
        logger.info(f"  Error rate: {error_rate:.4f}")
        logger.info(f"  Best so far: {optimizer.best_error_rate:.4f}")

    # Final results
    results["end_time"] = datetime.now().isoformat()
    results["turbo_stats"] = optimizer.turbo.get_statistics()

    logger.info(f"\n{'='*60}")
    logger.info("FlowPO-HD Optimization Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Best error rate: {results['best_error_rate']:.4f}")
    logger.info(f"Best instruction: {results['best_instruction'][:200]}...")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FlowPO-HD: High-Dimensional Prompt Optimization")

    # Main arguments
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of optimization iterations")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Paths
    parser.add_argument("--manifold-keeper-path", type=str,
                       default="flowpo_hd/checkpoints/manifold_keeper.pt",
                       help="Path to trained ManifoldKeeper")
    parser.add_argument("--hbbops-path", type=str,
                       default="lipo/data/hbbops_results_20260102.json",
                       help="Path to HbBoPs results for warm-start")
    parser.add_argument("--min-fidelity", type=int, default=600,
                       help="Minimum fidelity for warm-start data (default 600)")
    parser.add_argument("--output-dir", type=str,
                       default="flowpo_hd/results",
                       help="Output directory for results")
    parser.add_argument("--gp-type", type=str, default="vanilla",
                       choices=["vanilla", "saas", "isotropic", "adaptive"],
                       help="GP type (vanilla=best calibration 96%%, saas=good ranking)")

    # Evaluation
    parser.add_argument("--skip-llm-eval", action="store_true",
                       help="Skip LLM evaluation (use random error rates)")
    parser.add_argument("--eval-model", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model for evaluation")
    parser.add_argument("--eval-minibatch-size", type=int, default=100,
                       help="Minibatch size for evaluation")

    # Optimization parameters
    parser.add_argument("--lambda-penalty", type=float, default=0.001,
                       help="Velocity penalty weight")
    parser.add_argument("--ucb-beta-start", type=float, default=4.0,
                       help="Initial UCB beta")
    parser.add_argument("--ucb-beta-end", type=float, default=2.0,
                       help="Final UCB beta")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"run_{timestamp}.log"
    setup_logging(log_file=str(log_file))

    # Create config
    config = FlowPOHDConfig(
        seed=args.seed,
        device=get_device(args.device),
        manifold_keeper_path=args.manifold_keeper_path,
        results_dir=args.output_dir,
        eval_model=args.eval_model,
        eval_minibatch_size=args.eval_minibatch_size,
        fga_lambda_penalty=args.lambda_penalty,
        gp_ucb_beta_start=args.ucb_beta_start,
        gp_ucb_beta_end=args.ucb_beta_end,
        gp_type=args.gp_type,
        hbbops_results_path=args.hbbops_path,
        warm_start_min_fidelity=args.min_fidelity,
    )

    # Run optimization
    results = run_optimization(
        config=config,
        iterations=args.iterations,
        hbbops_path=args.hbbops_path,
        min_fidelity=args.min_fidelity,
        skip_llm_eval=args.skip_llm_eval,
    )

    # Save results
    results_path = Path(args.output_dir) / f"results_{timestamp}.json"
    save_results(results_path, results)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
