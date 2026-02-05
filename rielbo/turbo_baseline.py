"""TuRBO baseline for comparison with Spherical Subspace BO.

TuRBO (Trust Region Bayesian Optimization) is a standard BO baseline that:
- Uses GP in the full latent space
- Maintains a local trust region around the best point
- Adapts trust region size based on success/failure

This provides a fair comparison to Subspace BO since both use:
- Same SELFIES VAE codec (256D latent space)
- Same GuacaMol oracle
- Same cold start data
- Same number of iterations

The key difference:
- TuRBO: GP operates in full 256D (struggles with limited data)
- Subspace BO: Projects to S^15 for tractable GP

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m rielbo.turbo_baseline \
        --task-id pdop --n-cold-start 100 --iterations 500 --seed 42
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from rielbo.gp_diagnostics import GPDiagnostics, diagnose_gp_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TurboState:
    """Trust region state for TuRBO."""

    def __init__(
        self,
        dim: int,
        batch_size: int = 1,
        length: float = 0.8,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        failure_counter: int = 0,
        failure_tolerance: int = 5,
        success_counter: int = 0,
        success_tolerance: int = 3,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.length = length
        self.length_min = length_min
        self.length_max = length_max
        self.failure_counter = failure_counter
        self.failure_tolerance = failure_tolerance
        self.success_counter = success_counter
        self.success_tolerance = success_tolerance
        self.best_value = -float("inf")
        self.restart_triggered = False

    def update(self, y_next: float):
        """Update state based on new observation."""
        if y_next > self.best_value:
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = y_next
        else:
            self.success_counter = 0
            self.failure_counter += 1

        # Expand trust region on success
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0

        # Shrink trust region on failure
        if self.failure_counter >= self.failure_tolerance:
            self.length = self.length / 2.0
            self.failure_counter = 0

        # Restart if trust region too small
        if self.length < self.length_min:
            self.restart_triggered = True


class TuRBOBaseline:
    """TuRBO baseline operating in full 256D latent space."""

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        device: str = "cuda",
        n_candidates: int = 2000,
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.device = device
        self.n_candidates = n_candidates
        self.trust_region = trust_region
        self.seed = seed
        self.verbose = verbose

        # Data storage
        self.train_Z = None  # Latent vectors
        self.train_Y = None  # Scores
        self.smiles_observed = set()

        # Best tracking
        self.best_score = -float("inf")
        self.best_smiles = None
        self.best_z = None
        self.mean_norm = None

        # History
        self.history = []

        # GP
        self.gp = None

        # Trust region state
        self.turbo_state = None

        # GP diagnostics
        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []

    def cold_start(self, smiles_list: list, scores: list):
        """Initialize with cold start data."""
        logger.info(f"Cold start with {len(smiles_list)} molecules")

        # Encode SMILES to latent
        z_list = []
        valid_scores = []
        valid_smiles = []

        for smi, score in zip(smiles_list, scores):
            try:
                z = self.codec.encode([smi])
                if z is not None and not torch.isnan(z).any():
                    z_list.append(z)
                    valid_scores.append(score)
                    valid_smiles.append(smi)
                    self.smiles_observed.add(smi)
            except (ValueError, RuntimeError) as e:
                # Expected encoding failures (invalid SMILES, tensor issues)
                logger.info(f"Failed to encode {smi[:50]}...: {e}")
                continue
            except (torch.cuda.OutOfMemoryError, MemoryError):
                # Critical memory errors - re-raise
                raise

        if not z_list:
            raise ValueError("No valid molecules in cold start")

        self.train_Z = torch.cat(z_list, dim=0).to(self.device)
        self.train_Y = torch.tensor(valid_scores, dtype=torch.float32, device=self.device)

        # Compute mean norm for reconstruction
        norms = self.train_Z.norm(dim=-1)
        self.mean_norm = norms.mean().item()

        # Initialize best
        best_idx = self.train_Y.argmax()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = valid_smiles[best_idx]
        self.best_z = self.train_Z[best_idx].clone()

        # Initialize trust region
        self.turbo_state = TurboState(
            dim=self.input_dim,
            length=self.trust_region,
            failure_tolerance=max(5, self.input_dim // 20),
        )
        self.turbo_state.best_value = self.best_score

        logger.info(f"Cold start: {len(valid_smiles)} molecules, best={self.best_score:.4f}")
        logger.info(f"Mean norm: {self.mean_norm:.2f}")

        self.history.append({
            "iteration": 0,
            "best_score": self.best_score,
            "n_evaluated": len(self.smiles_observed),
        })

    def _normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize latent vectors to zero mean, unit variance for GP."""
        self._z_mean = self.train_Z.mean(dim=0)
        self._z_std = self.train_Z.std(dim=0).clamp(min=1e-6)
        return (z - self._z_mean) / self._z_std

    def _denormalize_z(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize latent vectors."""
        return z_norm * self._z_std + self._z_mean

    def _fit_gp(self):
        """Fit GP in normalized latent space."""
        # Normalize training data
        train_Z_norm = self._normalize_z(self.train_Z)

        # Standardize Y
        y_mean = self.train_Y.mean()
        y_std = self.train_Y.std().clamp(min=1e-6)
        train_Y_norm = (self.train_Y - y_mean) / y_std

        # Fit GP with Matern kernel (standard for BO)
        self.gp = SingleTaskGP(
            train_Z_norm.double(),
            train_Y_norm.double().unsqueeze(-1),
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.input_dim)),
        )

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

        self._y_mean = y_mean
        self._y_std = y_std
        self._train_Z_norm = train_Z_norm  # Store for diagnostics

    def _generate_candidates(self) -> torch.Tensor:
        """Generate Sobol candidates in trust region around best point."""
        # Normalize best point
        z_center_norm = self._normalize_z(self.best_z.unsqueeze(0))

        # Generate Sobol candidates
        sobol = SobolEngine(self.input_dim, scramble=True, seed=self.seed + len(self.history))
        pert = sobol.draw(self.n_candidates).to(dtype=torch.float32, device=self.device)

        # Scale to trust region
        half_length = self.turbo_state.length / 2
        z_cand_norm = z_center_norm - half_length + self.turbo_state.length * pert

        return z_cand_norm

    def _select_candidate(self, z_cand_norm: torch.Tensor) -> torch.Tensor:
        """Select best candidate using Expected Improvement."""
        # EI acquisition
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        ei = qExpectedImprovement(
            model=self.gp,
            best_f=(self.best_score - self._y_mean) / self._y_std,
            sampler=sampler,
        )

        # Evaluate all candidates
        with torch.no_grad():
            ei_values = ei(z_cand_norm.double().unsqueeze(1))

        # Select best
        best_idx = ei_values.argmax()
        z_opt_norm = z_cand_norm[best_idx]

        return z_opt_norm

    def _step(self, iteration: int = 0) -> tuple[float | None, str]:
        """Single optimization step.

        Returns:
            Tuple of (score, smiles). Score is None if duplicate/decode failed.
            SMILES is empty string if decode failed.
        """
        # Fit GP
        self._fit_gp()

        # Generate and select candidate
        z_cand_norm = self._generate_candidates()

        # Run GP diagnostics (every 10 iterations to reduce overhead)
        if iteration % 10 == 0:
            metrics = self.gp_diagnostics.analyze(
                self.gp,
                self._train_Z_norm,
                (self.train_Y - self._y_mean) / self._y_std,
                z_cand_norm[:100],  # Sample of candidates for extrapolation check
            )
            self.gp_diagnostics.log_summary(metrics, prefix=f"[Iter {iteration}]")
            self.diagnostic_history.append(self.gp_diagnostics.get_summary_dict(metrics))

        z_opt_norm = self._select_candidate(z_cand_norm)

        # Denormalize
        z_opt = self._denormalize_z(z_opt_norm.unsqueeze(0)).float()

        # Decode to SMILES
        try:
            smiles = self.codec.decode(z_opt)[0]
        except (ValueError, RuntimeError, IndexError) as e:
            # Expected decoding failures
            logger.info(f"Decode failed: {e}")
            return None, ""
        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Critical memory errors - re-raise
            raise

        # Skip if already evaluated
        if smiles in self.smiles_observed:
            return None, smiles  # Return smiles even for duplicates

        # Evaluate
        score = self.oracle.score(smiles)
        self.smiles_observed.add(smiles)

        # Update data
        self.train_Z = torch.cat([self.train_Z, z_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])

        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            self.best_z = z_opt.squeeze()

        # Update trust region
        self.turbo_state.update(score)

        return score, smiles

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        logger.info(f"Starting TuRBO optimization for {n_iterations} iterations")

        for i in range(1, n_iterations + 1):
            score, _smiles = self._step(iteration=i)

            # Handle restart
            if self.turbo_state.restart_triggered:
                logger.info(f"Iter {i}: Trust region restart triggered")
                self.turbo_state = TurboState(
                    dim=self.input_dim,
                    length=self.trust_region,
                    failure_tolerance=max(5, self.input_dim // 20),
                )
                self.turbo_state.best_value = self.best_score

            # Log progress
            if i % log_interval == 0 or i == n_iterations:
                logger.info(
                    f"Iter {i}/{n_iterations}: best={self.best_score:.4f}, "
                    f"n_eval={len(self.smiles_observed)}, "
                    f"tr_len={self.turbo_state.length:.4f}"
                )

            self.history.append({
                "iteration": i,
                "best_score": self.best_score,
                "n_evaluated": len(self.smiles_observed),
            })

        logger.info(f"Optimization complete: best={self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")


def main():
    parser = argparse.ArgumentParser(description="TuRBO Baseline on GuacaMol")

    parser.add_argument("--task-id", type=str, default="pdop")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--trust-region", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load components
    logger.info("Loading codec and oracle...")
    from shared.guacamol.codec import SELFIESVAECodec
    from shared.guacamol.data import load_guacamol_data
    from shared.guacamol.oracle import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    oracle = GuacaMolOracle(task_id=args.task_id)

    # Load cold start
    logger.info("Loading cold start data...")
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=args.n_cold_start,
        task_id=args.task_id,
    )

    # Create optimizer
    optimizer = TuRBOBaseline(
        codec=codec,
        oracle=oracle,
        input_dim=256,
        device=args.device,
        n_candidates=args.n_candidates,
        trust_region=args.trust_region,
        seed=args.seed,
    )

    # Run
    optimizer.cold_start(smiles_list, scores)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    # Save
    results_dir = "rielbo/results/guacamol"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"turbo_{args.task_id}_seed{args.seed}_{timestamp}.json")

    results = {
        "method": "turbo",
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "gp_diagnostics": optimizer.diagnostic_history,
        "args": vars(args),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
