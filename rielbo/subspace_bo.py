"""Spherical Subspace BO for high-dimensional optimization.

Projects S^(D-1) → S^(d-1) to make GP tractable with limited data.

Pipeline:
    x [N, D] → normalize → u [N, D] on S^(D-1) (directions)
    u [N, D] → project → v [N, d] on S^(d-1)  (d=16)
    GP operates on S^15 with ArcCosine kernel
    v* → lift → u* on S^(D-1) → x* = u* * mean_norm → decode

Mathematical Foundation:
    Original embedding: x ∈ R^D (D=256)
    Direction sphere: u = x/||x|| ∈ S^(D-1) (unit sphere)
    Subspace sphere: v ∈ S^(d-1) (d=16)

    Orthonormal projection: A ∈ R^{D×d}, A.T @ A = I_d
    Project: u → v = normalize(u @ A)
    Lift:    v → u = normalize(v @ A.T)

Key insight: Use mean norm from training data for magnitude reconstruction.
Simpler and more robust than stereographic for subspace methods.
"""

import logging

import gpytorch
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

logger = logging.getLogger(__name__)


class ArcCosineKernel(gpytorch.kernels.Kernel):
    """ArcCosine kernel for unit sphere data.

    k(x, y) = 1 - arccos(x·y) / π

    Measures geodesic distance on the sphere.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        x1_norm = x1 / (x1.norm(dim=-1, keepdim=True) + 1e-8)
        x2_norm = x2 / (x2.norm(dim=-1, keepdim=True) + 1e-8)

        if diag:
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        return 1.0 - torch.arccos(cos_sim) / torch.pi


class SphericalSubspaceBO:
    """Spherical Subspace BO - projects S^(D-1) → S^(d-1) for tractable GP.

    Key innovations:
    1. Orthonormal subspace projection preserves angular structure
    2. ArcCosine kernel for Riemannian geometry
    3. Mean norm reconstruction (simple, robust)

    Why this works:
    - 256D with 100 points: 0.39 points/dim → GP overfits
    - 16D with 100 points: 6.25 points/dim → GP generalizes
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        subspace_dim: int = 16,
        device: str = "cuda",
        n_candidates: int = 2000,
        ucb_beta: float = 2.0,
        acqf: str = "ts",  # "ts", "ei", "ucb"
        trust_region: float = 0.8,  # TuRBO-style trust region length
        seed: int = 42,
        verbose: bool = True,
        kernel: str = "arccosine",  # "arccosine" or "matern"
    ):
        if subspace_dim >= input_dim:
            raise ValueError(
                f"subspace_dim ({subspace_dim}) must be < input_dim ({input_dim})"
            )
        if subspace_dim < 2:
            raise ValueError(f"subspace_dim must be >= 2, got {subspace_dim}")
        if not (0 < trust_region <= 2.0):
            raise ValueError(f"trust_region must be in (0, 2], got {trust_region}")

        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim  # D (256)
        self.subspace_dim = subspace_dim  # d (16)
        self.n_candidates = n_candidates
        self.ucb_beta = ucb_beta
        self.acqf = acqf
        self.trust_region = trust_region
        self.verbose = verbose
        self.seed = seed
        self.kernel = kernel
        self.fallback_count = 0

        # Orthonormal projection A ∈ R^{D×d}
        torch.manual_seed(seed)
        A_raw = torch.randn(input_dim, subspace_dim, device=device)
        self.A, _ = torch.linalg.qr(A_raw)
        logger.info(f"Subspace: S^{input_dim-1} → S^{subspace_dim-1}, kernel={kernel}")

        # GP
        self.gp = None
        self.likelihood = None

        # Training data
        self.train_X = None  # Original embeddings [N, D]
        self.train_U = None  # Directions [N, D] on S^(D-1)
        self.train_V = None  # Subspace [N, d] on S^(d-1)
        self.train_Y = None  # Scores [N]
        self.mean_norm = None  # Mean embedding norm for reconstruction
        self.smiles_observed = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        self.history = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "gp_mean": [],
            "gp_std": [],
            "nearest_train_cos": [],
            "embedding_norm": [],
        }

    def project_to_subspace(self, u: torch.Tensor) -> torch.Tensor:
        """Project from S^(D-1) to S^(d-1)."""
        v = u @ self.A
        return F.normalize(v, p=2, dim=-1)

    def lift_to_original(self, v: torch.Tensor) -> torch.Tensor:
        """Lift from S^(d-1) to S^(D-1)."""
        u = v @ self.A.T
        return F.normalize(u, p=2, dim=-1)

    def _create_kernel(self):
        """Create covariance kernel."""
        if self.kernel == "arccosine":
            return gpytorch.kernels.ScaleKernel(ArcCosineKernel())
        elif self.kernel == "matern":
            return gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
        else:
            logger.warning(f"Unknown kernel '{self.kernel}', using arccosine")
            return gpytorch.kernels.ScaleKernel(ArcCosineKernel())

    def _fit_gp(self):
        """Fit GP on subspace sphere."""
        self.train_V = self.project_to_subspace(self.train_U)

        X = self.train_V.double()
        Y = self.train_Y.double().unsqueeze(-1)

        try:
            covar_module = self._create_kernel()
            self.gp = SingleTaskGP(X, Y, covar_module=covar_module).to(self.device)
            self.likelihood = self.gp.likelihood
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()

            # Diagnostics
            if self.verbose:
                with torch.no_grad():
                    pred = self.gp.posterior(X).mean.squeeze(-1)
                    corr = torch.corrcoef(torch.stack([pred, Y.squeeze(-1)]))[0, 1]
                    logger.info(
                        f"GP: corr={corr.item():.3f}, n={len(X)}, "
                        f"noise={self.likelihood.noise.item():.4f}"
                    )
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            # Numerical issues - use fallback
            self.fallback_count += 1
            logger.warning(f"GP fit failed (fallback #{self.fallback_count}): {e}")
            self.gp = SingleTaskGP(
                X, Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.likelihood = self.gp.likelihood
            self.likelihood.noise = 0.1
            self.gp.eval()

    def _generate_sobol_candidates(self, n_candidates: int) -> torch.Tensor:
        """Generate candidates in trust region around best training point."""
        # Find best training point in subspace
        best_idx = self.train_Y.argmax()
        v_best = self.train_V[best_idx:best_idx+1]  # [1, d]

        # Trust region bounds
        half_length = self.trust_region / 2
        tr_lb = v_best - half_length
        tr_ub = v_best + half_length

        # Generate Sobol sequence
        sobol = SobolEngine(self.subspace_dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=torch.float32, device=self.device)

        # Scale to trust region
        v_cand = tr_lb + (tr_ub - tr_lb) * pert  # [n_candidates, d]

        # Project to sphere
        v_cand = F.normalize(v_cand, p=2, dim=-1)

        return v_cand

    def _optimize_acquisition(self) -> tuple[torch.Tensor, dict]:
        """Find optimal v* using Thompson Sampling, EI, or UCB."""
        diag = {}

        try:
            if self.acqf == "ts":
                # Thompson Sampling with Sobol candidates
                v_cand = self._generate_sobol_candidates(self.n_candidates)

                thompson = MaxPosteriorSampling(model=self.gp, replacement=False)
                v_opt = thompson(v_cand.double().unsqueeze(0), num_samples=1)
                v_opt = v_opt.squeeze(0).float()
                v_opt = F.normalize(v_opt, p=2, dim=-1)

            elif self.acqf == "ei":
                # Expected Improvement with Sobol candidates
                v_cand = self._generate_sobol_candidates(self.n_candidates)

                ei = qExpectedImprovement(self.gp, best_f=self.train_Y.max().double())
                with torch.no_grad():
                    ei_vals = ei(v_cand.double().unsqueeze(-2))
                best_idx = ei_vals.argmax()
                v_opt = v_cand[best_idx:best_idx+1]

            elif self.acqf == "ucb":
                # UCB with Sobol candidates
                v_cand = self._generate_sobol_candidates(self.n_candidates)

                with torch.no_grad():
                    post = self.gp.posterior(v_cand.double())
                    ucb_vals = post.mean.squeeze() + self.ucb_beta * post.variance.sqrt().squeeze()
                best_idx = ucb_vals.argmax()
                v_opt = v_cand[best_idx:best_idx+1]

            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf}")

            # Diagnostics
            with torch.no_grad():
                post = self.gp.posterior(v_opt.double())
                diag["gp_mean"] = post.mean.item()
                diag["gp_std"] = post.variance.sqrt().item()
                cos_sims = (v_opt @ self.train_V.T).squeeze()
                diag["nearest_train_cos"] = cos_sims.max().item()

            u_opt = self.lift_to_original(v_opt)
            return u_opt, diag

        except (RuntimeError, torch.linalg.LinAlgError) as e:
            # Numerical issues - use random fallback
            logger.warning(f"Acquisition failed: {e}")
            u_opt = F.normalize(torch.randn(1, self.input_dim, device=self.device), dim=-1)
            return u_opt, {"gp_mean": 0, "gp_std": 1, "nearest_train_cos": 0, "is_fallback": True}

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm
        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i:i+64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        # Compute mean norm from ALL cold start molecules
        self.mean_norm = embeddings.norm(dim=-1).mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        # Store embeddings
        self.train_X = embeddings

        # Extract directions
        self.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        # Track best
        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        # Fit GP
        self._fit_gp()

        logger.info(f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})")
        logger.info(f"Best SMILES: {self.best_smiles}")

    def step(self) -> dict:
        """One BO iteration."""
        self.iteration += 1

        # Optimize on subspace, lift to original sphere
        u_opt, diag = self._optimize_acquisition()

        # Reconstruct embedding with mean norm
        x_opt = u_opt * self.mean_norm
        diag["embedding_norm"] = self.mean_norm

        # Decode
        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            return {"score": 0.0, "best_score": self.best_score, "smiles": "",
                    "is_duplicate": True, **diag}

        if smiles in self.smiles_observed:
            return {"score": 0.0, "best_score": self.best_score, "smiles": smiles,
                    "is_duplicate": True, **diag}

        # Evaluate
        score = self.oracle.score(smiles)

        # Update
        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device)])
        self.smiles_observed.append(smiles)

        if self.iteration % 10 == 0:
            self._fit_gp()

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            logger.info(f"New best! {score:.4f}: {smiles}")

        return {"score": score, "best_score": self.best_score, "smiles": smiles,
                "is_duplicate": False, **diag}

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(f"Spherical Subspace BO: {n_iterations} iterations")
        logger.info(f"S^{self.input_dim-1} → S^{self.subspace_dim-1}")

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
            self.history["nearest_train_cos"].append(result.get("nearest_train_cos", 0))
            self.history["embedding_norm"].append(result.get("embedding_norm", 0))

            if result["is_duplicate"]:
                n_dup += 1

            pbar.set_postfix({
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "gp_μ": f"{result.get('gp_mean', 0):.2f}",
                "dup": n_dup,
            })

            if (i + 1) % log_interval == 0 and self.verbose:
                logger.info(
                    f"Iter {i+1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Curr: {result['score']:.4f} | "
                    f"GP: {result.get('gp_mean', 0):.2f}±{result.get('gp_std', 0):.2f} | "
                    f"cos_near: {result.get('nearest_train_cos', 0):.3f}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")
