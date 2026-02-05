"""Vanilla BO with Hvarfner's dimension-scaled lengthscale priors.

Implements the approach from:
  Hvarfner et al. (ICML 2024): "Vanilla Bayesian Optimization Performs Great
  in High Dimensions"

Key idea: Standard GP priors push lengthscales too short in high D, causing
overfitting. BoTorch 0.16+ already uses RBF + LogNormal(√2 + log(D)/2, √3)
with ARD as default. This module simply uses those defaults with:
- Proper [0,1]^D min-max input normalization
- BoTorch default GP (RBF + Hvarfner priors + Standardize outcome)
- TuRBO-style trust region for local exploration
- Thompson Sampling acquisition

Differences from TuRBO baseline:
- Hvarfner's LogNormal lengthscale prior (vs old Gamma default)
- RBF kernel (vs Matern-5/2)
- [0,1]^D normalization (vs z-score)

Differences from Subspace BO:
- No random projection — GP operates in full 256D
- Relies on the lengthscale prior to handle dimensionality

Usage:
    from rielbo.vanilla_bo import VanillaBO

    optimizer = VanillaBO(
        codec=codec,
        oracle=oracle,
        input_dim=256,
    )
"""

import logging
import math

import gpytorch
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from rielbo.gp_diagnostics import GPDiagnostics

logger = logging.getLogger(__name__)


class VanillaBO:
    """Vanilla BO in full latent space with BoTorch's Hvarfner defaults.

    Uses [0,1]^D min-max normalization and BoTorch's default SingleTaskGP
    which includes RBF + LogNormal(√2+log(D)/2, √3) + ARD + Standardize.
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        device: str = "cuda",
        n_candidates: int = 2000,
        ucb_beta: float = 2.0,
        acqf: str = "ts",
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
        # Trust region parameters
        failure_tolerance: int | None = None,
        success_tolerance: int = 3,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
    ):
        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta
        self.acqf = acqf
        self.trust_region = trust_region
        self.seed = seed
        self.verbose = verbose

        # Trust region
        self.failure_tolerance = failure_tolerance or max(5, input_dim // 20)
        self.success_tolerance = success_tolerance
        self.length_min = length_min
        self.length_max = length_max

        # Trust region state
        self._tr_length = trust_region
        self._tr_success_count = 0
        self._tr_failure_count = 0
        self._tr_restart_count = 0

        # Data
        self.train_Z = None  # Raw latent vectors [N, D]
        self.train_Y = None  # Scores [N]
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.best_z = None
        self.mean_norm = None
        self.iteration = 0
        self.fallback_count = 0

        # [0,1]^D normalization bounds (set from cold start data)
        self._z_min = None  # [D]
        self._z_max = None  # [D]

        # GP
        self.gp = None

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "gp_mean": [],
            "gp_std": [],
            "tr_length": [],
        }

        # GP diagnostics
        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []

        # Compute Hvarfner prior params for logging
        ls_mu = math.sqrt(2) + math.log(input_dim) / 2
        ls_sigma = math.sqrt(3)
        logger.info(
            f"VanillaBO: D={input_dim}, acqf={acqf}, "
            f"BoTorch defaults: RBF+ARD+LogNormal({ls_mu:.2f}, {ls_sigma:.2f}), "
            f"median_ℓ={math.exp(ls_mu):.1f}, mode_ℓ={math.exp(ls_mu - ls_sigma**2):.2f}"
        )

    def _to_unit_cube(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize raw latent vectors to [0, 1]^D."""
        return (z - self._z_min) / (self._z_max - self._z_min)

    def _from_unit_cube(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1]^D to raw latent space."""
        return z_norm * (self._z_max - self._z_min) + self._z_min

    def _fit_gp(self):
        """Fit GP using BoTorch defaults on [0,1]^D normalized data.

        BoTorch 0.16+ default SingleTaskGP uses:
        - RBF kernel with ARD
        - LogNormal(√2 + log(D)/2, √3) lengthscale prior
        - Standardize() outcome transform (Y normalization)
        - No ScaleKernel wrapper
        """
        Z_norm = self._to_unit_cube(self.train_Z).double()
        Y = self.train_Y.double().unsqueeze(-1)

        try:
            # Use BoTorch defaults — already has Hvarfner priors
            self.gp = SingleTaskGP(Z_norm, Y).to(self.device)

            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            # Diagnostics every 10 iterations
            if self.verbose and self.iteration % 10 == 0:
                metrics = self.gp_diagnostics.analyze(
                    self.gp, Z_norm.float(),
                    self.gp.outcome_transform(Y)[0].squeeze(-1).float(),
                )
                self.gp_diagnostics.log_summary(
                    metrics, prefix=f"[Iter {self.iteration}]"
                )
                self.diagnostic_history.append(
                    self.gp_diagnostics.get_summary_dict(metrics)
                )

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            self.fallback_count += 1
            logger.warning(f"GP fit failed (fallback #{self.fallback_count}): {e}")
            self.gp = SingleTaskGP(
                Z_norm, Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.gp.likelihood.noise = 0.1
            self.gp.eval()

    def _generate_candidates(self) -> torch.Tensor:
        """Generate Sobol candidates in trust region around best point in [0,1]^D."""
        z_best_norm = self._to_unit_cube(self.best_z.unsqueeze(0))

        half_length = self._tr_length / 2
        tr_lb = (z_best_norm - half_length).clamp(0.0, 1.0)
        tr_ub = (z_best_norm + half_length).clamp(0.0, 1.0)

        sobol = SobolEngine(
            self.input_dim, scramble=True,
            seed=self.seed + self.iteration,
        )
        pert = sobol.draw(self.n_candidates).to(
            dtype=torch.float32, device=self.device
        )

        z_cand = tr_lb + (tr_ub - tr_lb) * pert
        return z_cand

    def _optimize_acquisition(self) -> tuple[torch.Tensor, dict]:
        """Find optimal z* using acquisition function."""
        diag = {}

        try:
            z_cand_norm = self._generate_candidates()

            if self.acqf == "ts":
                thompson = MaxPosteriorSampling(
                    model=self.gp, replacement=False
                )
                z_opt = thompson(
                    z_cand_norm.double().unsqueeze(0), num_samples=1
                )
                z_opt = z_opt.squeeze(0).float()

            elif self.acqf == "ei":
                # BoTorch Standardize handles Y transform internally
                best_f = self.best_score
                ei = qExpectedImprovement(self.gp, best_f=best_f)
                with torch.no_grad():
                    ei_vals = ei(z_cand_norm.double().unsqueeze(-2))
                best_idx = ei_vals.argmax()
                z_opt = z_cand_norm[best_idx:best_idx + 1]

            elif self.acqf == "ucb":
                with torch.no_grad():
                    post = self.gp.posterior(z_cand_norm.double())
                    ucb_vals = (
                        post.mean.squeeze()
                        + self.ucb_beta * post.variance.sqrt().squeeze()
                    )
                best_idx = ucb_vals.argmax()
                z_opt = z_cand_norm[best_idx:best_idx + 1]

            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf}")

            # Diagnostics (in original Y scale since Standardize is applied)
            with torch.no_grad():
                post = self.gp.posterior(z_opt.double())
                diag["gp_mean"] = post.mean.item()
                diag["gp_std"] = post.variance.sqrt().item()

            # Denormalize back to original latent space
            z_opt_raw = self._from_unit_cube(z_opt.float())
            return z_opt_raw, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            logger.warning(f"Acquisition failed: {e}")
            z_opt = self.best_z.unsqueeze(0) + 0.01 * torch.randn(
                1, self.input_dim, device=self.device
            )
            return z_opt, {"gp_mean": 0, "gp_std": 1, "is_fallback": True}

    def _update_trust_region(self, score: float):
        """TuRBO-style trust region adaptation."""
        if score > self.best_score:
            self._tr_success_count += 1
            self._tr_failure_count = 0
        else:
            self._tr_failure_count += 1
            self._tr_success_count = 0

        if self._tr_success_count >= self.success_tolerance:
            self._tr_length = min(2.0 * self._tr_length, self.length_max)
            self._tr_success_count = 0

        if self._tr_failure_count >= self.failure_tolerance:
            self._tr_length = self._tr_length / 2.0
            self._tr_failure_count = 0

        if self._tr_length < self.length_min:
            self._tr_restart_count += 1
            logger.info(
                f"Trust region restart #{self._tr_restart_count} "
                f"(was {self._tr_length:.6f})"
            )
            self._tr_length = self.trust_region
            self._tr_success_count = 0
            self._tr_failure_count = 0

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i:i + 64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        self.mean_norm = embeddings.norm(dim=-1).mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        self.train_Z = embeddings
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Compute [0,1] normalization bounds from cold start data
        # Add small margin to avoid exact 0/1 at boundaries
        self._z_min = embeddings.min(dim=0).values
        self._z_max = embeddings.max(dim=0).values
        margin = (self._z_max - self._z_min) * 0.05
        self._z_min = self._z_min - margin
        self._z_max = self._z_max + margin
        # Prevent zero-range dimensions
        zero_range = (self._z_max - self._z_min).abs() < 1e-8
        self._z_max[zero_range] = self._z_min[zero_range] + 1.0

        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]
        self.best_z = self.train_Z[best_idx].clone()

        # Initial GP fit
        self._fit_gp()

        logger.info(
            f"Cold start done. Best: {self.best_score:.4f} "
            f"(n={len(self.train_Y)})"
        )

    def step(self) -> dict:
        """One BO iteration."""
        self.iteration += 1

        # Refit GP
        self._fit_gp()

        # Optimize acquisition
        z_opt, diag = self._optimize_acquisition()
        diag["tr_length"] = self._tr_length

        # Decode
        smiles_list = self.codec.decode(z_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            return {
                "score": 0.0, "best_score": self.best_score,
                "smiles": "", "is_duplicate": True, **diag,
            }

        if smiles in self.smiles_observed:
            return {
                "score": 0.0, "best_score": self.best_score,
                "smiles": smiles, "is_duplicate": True, **diag,
            }

        # Evaluate
        score = self.oracle.score(smiles)

        # Update trust region BEFORE updating best
        self._update_trust_region(score)

        # Update training data
        self.train_Z = torch.cat([self.train_Z, z_opt], dim=0)
        self.train_Y = torch.cat([
            self.train_Y,
            torch.tensor([score], device=self.device, dtype=torch.float32),
        ])
        self.smiles_observed.append(smiles)

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            self.best_z = z_opt.squeeze().clone()
            logger.info(f"New best! {score:.4f}: {smiles}")

        return {
            "score": score, "best_score": self.best_score,
            "smiles": smiles, "is_duplicate": False, **diag,
        }

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(
            f"VanillaBO: {n_iterations} iterations, D={self.input_dim}, "
            f"acqf={self.acqf}, tr={self.trust_region}"
        )

        pbar = tqdm(range(n_iterations), desc="Optimizing")
        n_dup = 0

        for i in pbar:
            result = self.step()

            self.history["iteration"].append(i)
            self.history["best_score"].append(self.best_score)
            self.history["current_score"].append(result["score"])
            self.history["n_evaluated"].append(len(self.smiles_observed))
            self.history["gp_mean"].append(result.get("gp_mean", 0))
            self.history["gp_std"].append(result.get("gp_std", 0))
            self.history["tr_length"].append(result.get("tr_length", 0))

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "gp_s": f"{result.get('gp_std', 0):.4f}",
                "tr": f"{self._tr_length:.3f}",
                "dup": n_dup,
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}+/-{result.get('gp_std', 0):.4f} | "
                    f"TR: {self._tr_length:.3f} | dup: {n_dup}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f} | Duplicates: {n_dup}/{n_iterations} ({100*n_dup/n_iterations:.1f}%)")
        logger.info(f"Best SMILES: {self.best_smiles}")
