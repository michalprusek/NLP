"""Spherical Ensemble BO: Multi-scale subspaces with max-std selection.

Maintains K independent random subspace projections S^(D-1) -> S^(d_k-1),
each at a different dimensionality (e.g. d=[4,8,12,16,20,24]). At each
iteration, generates a candidate from each subspace via Thompson Sampling,
then selects the candidate where the GP posterior is most uncertain (max std).
"""

import logging
from dataclasses import dataclass

import gpytorch
import torch
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from rielbo.kernels import create_kernel
from rielbo.spherical_transforms import GeodesicTrustRegion

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for Spherical Ensemble BO."""

    member_dims: list[int] | None = None
    retirement_interval: int = 100

    kernel_type: str = "arccosine"
    kernel_order: int = 0

    geodesic_tr: bool = True
    geodesic_max_angle: float = 0.5
    geodesic_global_fraction: float = 0.2

    adaptive_tr: bool = True
    tr_init: float = 0.4
    tr_min: float = 0.02
    tr_max: float = 0.8
    tr_success_tol: int = 3
    tr_fail_tol: int = 10
    tr_grow_factor: float = 1.5
    tr_shrink_factor: float = 0.5
    max_restarts: int = 5

    n_candidates: int = 2000

    def __post_init__(self):
        if self.member_dims is None:
            self.member_dims = [4, 8, 12, 16, 20, 24]

    @property
    def n_subspaces(self) -> int:
        return len(self.member_dims)

    @classmethod
    def from_preset(cls, preset: str) -> "EnsembleConfig":
        """Create config from preset name."""
        presets = {
            "default": cls(),  # [4,8,12,16,20,24]
            "small": cls(member_dims=[4, 8, 12]),
            "medium": cls(member_dims=[8, 12, 16, 20]),
            "large": cls(member_dims=[4, 8, 12, 16, 20, 24, 32]),
            "conservative": cls(retirement_interval=150),
            "aggressive": cls(retirement_interval=75),
        }
        if preset not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. Available: {list(presets.keys())}"
            )
        return presets[preset]


class SubspaceMember:
    """One subspace in the ensemble: projection matrix + GP + adaptive TR state."""

    def __init__(
        self,
        member_id: int,
        input_dim: int,
        subspace_dim: int,
        config: EnsembleConfig,
        device: str,
        seed: int,
    ):
        self.member_id = member_id
        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.config = config
        self.device = device
        self.seed = seed

        torch.manual_seed(seed)
        A_raw = torch.randn(input_dim, subspace_dim, device=device)
        self.A, _ = torch.linalg.qr(A_raw)

        self.gp = None
        self.geodesic_tr = None
        if config.geodesic_tr:
            self.geodesic_tr = GeodesicTrustRegion(
                max_angle=config.geodesic_max_angle,
                global_fraction=config.geodesic_global_fraction,
                device=device,
            )

        self.tr_length = config.tr_init if config.adaptive_tr else None
        self._tr_success_count = 0
        self._tr_fail_count = 0
        self.n_restarts = 0

        self.n_selected = 0
        self.n_improved = 0
        self.last_std = 0.0

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Project S^(D-1) → S^(d-1)."""
        v = u @ self.A
        return F.normalize(v, p=2, dim=-1)

    def lift(self, v: torch.Tensor) -> torch.Tensor:
        """Lift S^(d-1) → S^(D-1)."""
        u = v @ self.A.T
        return F.normalize(u, p=2, dim=-1)

    def fit_gp(self, train_U: torch.Tensor, train_Y: torch.Tensor):
        """Fit GP in this member's subspace on shared training data."""
        train_V = self.project(train_U)
        X = train_V.double()
        Y = train_Y.double().unsqueeze(-1)

        try:
            covar_module = create_kernel(
                kernel_type=self.config.kernel_type,
                kernel_order=self.config.kernel_order,
                use_scale=True,
            )
            self.gp = SingleTaskGP(X, Y, covar_module=covar_module).to(self.device)
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            self.gp.eval()
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            logger.error(f"Member {self.member_id} GP fit failed: {e}")
            self.gp = SingleTaskGP(
                X,
                Y,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-2)
                ),
            ).to(self.device)
            self.gp.likelihood.noise = 0.1
            self.gp.eval()

    def generate_candidate(
        self, train_U: torch.Tensor, train_Y: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Generate one TS candidate and return (u_opt, posterior_std)."""
        if self.gp is None:
            u_opt = F.normalize(
                torch.randn(1, self.input_dim, device=self.device), dim=-1
            )
            self.last_std = 1.0
            return u_opt, 1.0

        best_idx = train_Y.argmax()
        v_best = self.project(train_U[best_idx : best_idx + 1])

        if self.geodesic_tr is not None:
            if self.config.adaptive_tr and self.tr_length is not None:
                radius = self.config.geodesic_max_angle * self.tr_length
            else:
                radius = self.config.geodesic_max_angle * 0.8
            v_cand = self.geodesic_tr.sample(
                center=v_best,
                n_samples=self.config.n_candidates,
                adaptive_radius=radius,
            )
        else:
            v_cand = torch.randn(
                self.config.n_candidates, self.subspace_dim, device=self.device
            )
            v_cand = F.normalize(v_cand, dim=-1)

        try:
            thompson = MaxPosteriorSampling(model=self.gp, replacement=False)
            v_opt = thompson(v_cand.double().unsqueeze(0), num_samples=1)
            v_opt = v_opt.squeeze(0).float()
            v_opt = F.normalize(v_opt, p=2, dim=-1)
        except Exception as e:
            logger.error(f"Member {self.member_id} TS failed: {e}")
            v_opt = v_cand[0:1]

        with torch.no_grad():
            post = self.gp.posterior(v_opt.double())
            std = post.variance.sqrt().item()

        self.last_std = std
        u_opt = self.lift(v_opt)
        return u_opt, std

    def update_tr(self, improved: bool):
        """Update adaptive trust region (TuRBO-style grow/shrink + restart)."""
        if not self.config.adaptive_tr or self.tr_length is None:
            return

        cfg = self.config
        if improved:
            self._tr_success_count += 1
            self._tr_fail_count = 0
        else:
            self._tr_success_count = 0
            self._tr_fail_count += 1

        if self._tr_success_count >= cfg.tr_success_tol:
            self.tr_length = min(self.tr_length * cfg.tr_grow_factor, cfg.tr_max)
            self._tr_success_count = 0
            logger.debug(
                f"Member {self.member_id} TR grow → {self.tr_length:.4f}"
            )
        elif self._tr_fail_count >= cfg.tr_fail_tol:
            self.tr_length *= cfg.tr_shrink_factor
            self._tr_fail_count = 0
            if self.tr_length < cfg.tr_min:
                self._restart()
            else:
                logger.debug(
                    f"Member {self.member_id} TR shrink → {self.tr_length:.4f}"
                )

    def _restart(self):
        """Restart with fresh random QR projection."""
        cfg = self.config
        if self.n_restarts >= cfg.max_restarts:
            self.tr_length = cfg.tr_init
            logger.info(
                f"Member {self.member_id}: max restarts ({cfg.max_restarts}) reached, "
                f"resetting TR to {cfg.tr_init}"
            )
            return

        self.n_restarts += 1
        self.tr_length = cfg.tr_init
        self._tr_success_count = 0
        self._tr_fail_count = 0

        torch.manual_seed(self.seed + self.n_restarts * 1000)
        A_raw = torch.randn(self.input_dim, self.subspace_dim, device=self.device)
        self.A, _ = torch.linalg.qr(A_raw)

        logger.info(
            f"Member {self.member_id} restart #{self.n_restarts}, TR → {cfg.tr_init}"
        )

    def retire(
        self, new_seed: int, train_U: torch.Tensor, train_Y: torch.Tensor
    ):
        """Replace this member with a fresh random projection and refit GP."""
        old_std = self.last_std
        self.seed = new_seed
        torch.manual_seed(new_seed)
        A_raw = torch.randn(self.input_dim, self.subspace_dim, device=self.device)
        self.A, _ = torch.linalg.qr(A_raw)

        self.tr_length = self.config.tr_init if self.config.adaptive_tr else None
        self._tr_success_count = 0
        self._tr_fail_count = 0
        self.n_restarts = 0
        self.n_selected = 0
        self.n_improved = 0
        self.last_std = 0.0

        self.fit_gp(train_U, train_Y)

        logger.info(
            f"Member {self.member_id} retired (old std={old_std:.4f}) → "
            f"fresh projection (seed={new_seed})"
        )


class SphericalEnsembleBO:
    """Spherical Ensemble BO with multiple independent subspaces.

    Maintains K independent (projection, GP) pairs on S^(D-1).
    At each iteration, generates a TS candidate from each subspace,
    then selects the candidate from the most uncertain GP (max posterior std).

    Periodically retires the member with the most collapsed GP posterior
    and replaces it with a fresh random subspace.
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        config: EnsembleConfig | None = None,
        device: str = "cuda",
        seed: int = 42,
        verbose: bool = True,
    ):
        if config is None:
            config = EnsembleConfig()

        self.config = config
        self.device = device
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.seed = seed
        self.verbose = verbose

        self.members: list[SubspaceMember] = []
        for k, dim_k in enumerate(config.member_dims):
            member = SubspaceMember(
                member_id=k,
                input_dim=input_dim,
                subspace_dim=dim_k,
                config=config,
                device=device,
                seed=seed + k * 100,
            )
            self.members.append(member)

        self.train_X = None
        self.train_U = None
        self.train_Y = None
        self.mean_norm = None
        self.smiles_observed: list[str] = []
        self.best_score = float("-inf")
        self.best_smiles = ""
        self.iteration = 0

        self._next_retirement_seed = seed + config.n_subspaces * 100
        self.n_retirements = 0
        self.fallback_count = 0

        self.history: dict[str, list] = {
            "iteration": [],
            "best_score": [],
            "current_score": [],
            "n_evaluated": [],
            "selected_member": [],
            "selected_std": [],
            "member_stds": [],
            "n_retirements": [],
        }

        self._log_config()

    def _log_config(self):
        """Log configuration summary."""
        cfg = self.config
        dims_str = ", ".join(str(d) for d in cfg.member_dims)
        logger.info(
            f"SphericalEnsembleBO: {cfg.n_subspaces} subspaces, "
            f"dims=[{dims_str}]"
        )
        logger.info(
            f"Selection: max posterior std | "
            f"Retirement every {cfg.retirement_interval} iter"
        )
        logger.info(
            f"Per-member: geodesic_tr={cfg.geodesic_tr}, "
            f"adaptive_tr={cfg.adaptive_tr}"
        )

    def cold_start(self, smiles_list: list[str], scores: torch.Tensor):
        """Initialize with pre-scored molecules."""
        logger.info(f"Cold start: {len(smiles_list)} molecules")

        from tqdm import tqdm

        embeddings = []
        for i in tqdm(range(0, len(smiles_list), 64), desc="Encoding"):
            batch = smiles_list[i : i + 64]
            with torch.no_grad():
                emb = self.codec.encode(batch)
            embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0).to(self.device)

        norms = embeddings.norm(dim=-1)
        self.mean_norm = norms.mean().item()
        logger.info(f"Mean embedding norm: {self.mean_norm:.2f}")

        self.train_X = embeddings
        self.train_U = F.normalize(embeddings, p=2, dim=-1)
        self.train_Y = scores.to(self.device).float()
        self.smiles_observed = smiles_list.copy()

        best_idx = self.train_Y.argmax().item()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = smiles_list[best_idx]

        for member in self.members:
            member.fit_gp(self.train_U, self.train_Y)

        logger.info(
            f"Cold start done. Best: {self.best_score:.4f} (n={len(self.train_Y)})"
        )
        logger.info(f"Best SMILES: {self.best_smiles}")

    def _select_candidate(self) -> tuple[torch.Tensor, int, dict]:
        """Generate TS candidates from all members, select most uncertain."""
        candidates = []
        for member in self.members:
            u_opt, std = member.generate_candidate(self.train_U, self.train_Y)
            candidates.append((u_opt, std, member.member_id))

        best_idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
        u_opt, std, member_id = candidates[best_idx]

        member_stds = [c[1] for c in candidates]

        diag = {
            "selected_member": member_id,
            "selected_std": std,
            "member_stds": member_stds,
        }
        return u_opt, member_id, diag

    def _maybe_retire(self):
        """Retire the member with the most collapsed GP posterior."""
        if self.iteration % self.config.retirement_interval != 0:
            return
        if self.iteration == 0:
            return

        worst_idx = min(
            range(len(self.members)),
            key=lambda i: self.members[i].last_std,
        )

        self._next_retirement_seed += 1
        worst = self.members[worst_idx]
        logger.info(
            f"Retiring member {worst.member_id} "
            f"(std={worst.last_std:.4f}, selected={worst.n_selected}x, "
            f"improved={worst.n_improved}x)"
        )
        worst.retire(self._next_retirement_seed, self.train_U, self.train_Y)
        self.n_retirements += 1

    def step(self) -> dict:
        """One BO iteration."""
        self.iteration += 1
        self._maybe_retire()

        u_opt, selected_member_id, sel_diag = self._select_candidate()
        x_opt = u_opt * self.mean_norm

        smiles_list = self.codec.decode(x_opt)
        smiles = smiles_list[0] if smiles_list else ""

        if not smiles:
            self.members[selected_member_id].update_tr(improved=False)
            return {
                "score": 0.0,
                "best_score": self.best_score,
                "smiles": "",
                "is_duplicate": True,
                "is_decode_failure": True,
                **sel_diag,
            }

        if smiles in self.smiles_observed:
            self.members[selected_member_id].update_tr(improved=False)
            return {
                "score": 0.0,
                "best_score": self.best_score,
                "smiles": smiles,
                "is_duplicate": True,
                **sel_diag,
            }

        score = self.oracle.score(smiles)

        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat(
            [
                self.train_Y,
                torch.tensor([score], device=self.device, dtype=torch.float32),
            ]
        )
        self.smiles_observed.append(smiles)

        selected_member = self.members[selected_member_id]
        selected_member.n_selected += 1

        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_smiles = smiles
            selected_member.n_improved += 1
            logger.info(
                f"New best! {score:.4f} (member {selected_member_id}): {smiles}"
            )

        selected_member.update_tr(improved=improved)

        if self.iteration % 10 == 0:
            for member in self.members:
                member.fit_gp(self.train_U, self.train_Y)

        return {
            "score": score,
            "best_score": self.best_score,
            "smiles": smiles,
            "is_duplicate": False,
            **sel_diag,
        }

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        from tqdm import tqdm

        logger.info(
            f"SphericalEnsembleBO: {n_iterations} iterations, "
            f"{self.config.n_subspaces} subspaces"
        )

        pbar = tqdm(range(n_iterations), desc="Ensemble BO")
        n_dup = 0

        for i in pbar:
            result = self.step()

            self.history["iteration"].append(i)
            self.history["best_score"].append(self.best_score)
            self.history["current_score"].append(result["score"])
            self.history["n_evaluated"].append(len(self.smiles_observed))
            self.history["selected_member"].append(
                result.get("selected_member", -1)
            )
            self.history["selected_std"].append(result.get("selected_std", 0))
            self.history["member_stds"].append(result.get("member_stds", []))
            self.history["n_retirements"].append(self.n_retirements)

            if result.get("is_duplicate", False):
                n_dup += 1

            postfix = {
                "best": f"{self.best_score:.4f}",
                "curr": f"{result['score']:.4f}",
                "mem": result.get("selected_member", "?"),
                "std": f"{result.get('selected_std', 0):.3f}",
                "dup": n_dup,
            }
            pbar.set_postfix(postfix)

            if (i + 1) % log_interval == 0 and self.verbose:
                stds = result.get("member_stds", [])
                std_str = ", ".join(f"{s:.3f}" for s in stds)
                logger.info(
                    f"Iter {i + 1}/{n_iterations} | Best: {self.best_score:.4f} | "
                    f"Member {result.get('selected_member', '?')} | "
                    f"Stds: [{std_str}] | Retirements: {self.n_retirements}"
                )

        logger.info(f"Done. Best: {self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")

        for m in self.members:
            rate = m.n_improved / max(m.n_selected, 1) * 100
            logger.info(
                f"  Member {m.member_id} (d={m.subspace_dim}): "
                f"selected {m.n_selected}x, "
                f"improved {m.n_improved}x ({rate:.1f}%), "
                f"restarts: {m.n_restarts}, last_std: {m.last_std:.4f}"
            )
