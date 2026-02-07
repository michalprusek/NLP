"""Core data types for RieLBO optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StepResult:
    """Result from a single BO iteration."""

    score: float
    best_score: float
    smiles: str
    is_duplicate: bool = False
    is_decode_failure: bool = False

    # Diagnostics
    gp_mean: float = 0.0
    gp_std: float = 0.0
    nearest_train_cos: float = 0.0
    embedding_norm: float = 0.0
    subspace_dim: int = 0
    tr_length: float = 0.0
    n_restarts: int = 0
    ur_radius: float = 0.0
    ur_rotations: int = 0
    acqf_used: str = ""
    active_subspace: int = 0
    mll: float | None = None

    # Raw embedding for downstream use
    x_opt: torch.Tensor | None = None


@dataclass
class TrainingData:
    """Shared training data container for all BO components."""

    train_X: torch.Tensor | None = None   # Original embeddings [N, D]
    train_U: torch.Tensor | None = None   # Directions [N, D] on S^(D-1)
    train_V: torch.Tensor | None = None   # Subspace [N, d] on S^(d-1)
    train_Y: torch.Tensor | None = None   # Scores [N]
    mean_norm: float = 0.0
    smiles_observed: list[str] = field(default_factory=list)
    best_score: float = float("-inf")
    best_smiles: str = ""
    best_idx: int = -1

    @property
    def n_observed(self) -> int:
        return len(self.smiles_observed)

    def update_best(self) -> None:
        """Recompute best score and index from train_Y."""
        if self.train_Y is not None and len(self.train_Y) > 0:
            self.best_idx = self.train_Y.argmax().item()
            self.best_score = self.train_Y[self.best_idx].item()
            if self.best_idx < len(self.smiles_observed):
                self.best_smiles = self.smiles_observed[self.best_idx]

    def add_observation(
        self,
        x_opt: torch.Tensor,
        u_opt: torch.Tensor,
        score: float,
        smiles: str,
    ) -> bool:
        """Add a new observation. Returns True if it improved best_score."""
        self.train_X = torch.cat([self.train_X, x_opt], dim=0)
        self.train_U = torch.cat([self.train_U, u_opt], dim=0)
        self.train_Y = torch.cat([
            self.train_Y,
            torch.tensor([score], device=self.train_Y.device, dtype=torch.float32),
        ])
        self.smiles_observed.append(smiles)

        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_smiles = smiles
            self.best_idx = len(self.train_Y) - 1
        return improved


@dataclass
class BOHistory:
    """Tracks optimization history for logging and serialization."""

    iteration: list[int] = field(default_factory=list)
    best_score: list[float] = field(default_factory=list)
    current_score: list[float] = field(default_factory=list)
    n_evaluated: list[int] = field(default_factory=list)
    gp_mean: list[float] = field(default_factory=list)
    gp_std: list[float] = field(default_factory=list)
    nearest_train_cos: list[float] = field(default_factory=list)
    embedding_norm: list[float] = field(default_factory=list)
    subspace_dim: list[int] = field(default_factory=list)
    tr_length: list[float] = field(default_factory=list)
    n_restarts: list[int] = field(default_factory=list)
    ur_radius: list[float] = field(default_factory=list)
    ur_rotations: list[int] = field(default_factory=list)
    acqf_used: list[str] = field(default_factory=list)
    active_subspace: list[int] = field(default_factory=list)

    def append_from_result(self, result: StepResult, n_eval: int) -> None:
        """Append a step result to history."""
        self.iteration.append(len(self.iteration))
        self.best_score.append(result.best_score)
        self.current_score.append(result.score)
        self.n_evaluated.append(n_eval)
        self.gp_mean.append(result.gp_mean)
        self.gp_std.append(result.gp_std)
        self.nearest_train_cos.append(result.nearest_train_cos)
        self.embedding_norm.append(result.embedding_norm)
        self.subspace_dim.append(result.subspace_dim)
        self.tr_length.append(result.tr_length)
        self.n_restarts.append(result.n_restarts)
        self.ur_radius.append(result.ur_radius)
        self.ur_rotations.append(result.ur_rotations)
        self.acqf_used.append(result.acqf_used)
        self.active_subspace.append(result.active_subspace)

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return {
            "iteration": self.iteration,
            "best_score": self.best_score,
            "current_score": self.current_score,
            "n_evaluated": self.n_evaluated,
            "gp_mean": self.gp_mean,
            "gp_std": self.gp_std,
            "nearest_train_cos": self.nearest_train_cos,
            "embedding_norm": self.embedding_norm,
            "subspace_dim": self.subspace_dim,
            "tr_length": self.tr_length,
            "n_restarts": self.n_restarts,
            "ur_radius": self.ur_radius,
            "ur_rotations": self.ur_rotations,
            "acqf_used": self.acqf_used,
            "active_subspace": self.active_subspace,
        }
