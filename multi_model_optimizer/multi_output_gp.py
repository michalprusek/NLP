"""
Multi-Output Gaussian Process for cross-model prompt optimization.

Uses Intrinsic Coregionalization Model (ICM) kernel to capture correlations
between different LLM performances. This enables more sample-efficient
optimization by sharing information across models.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch
import torch.nn as nn

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class FeatureExtractor(nn.Module):
    """
    Structural-aware feature extractor for deep kernel GP.

    Same architecture as single-output GP:
    - Instruction encoder: Linear(768, 64) -> ReLU -> Linear(64, 32)
    - Exemplar encoder:   Linear(768, 64) -> ReLU -> Linear(64, 32)
    - Joint encoder:      Linear(64, 32) -> ReLU -> Linear(32, latent_dim)
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract latent features from instruction and exemplar embeddings.

        Args:
            instruction_emb: (batch, 768) instruction embeddings
            exemplar_emb: (batch, 768) exemplar embeddings

        Returns:
            (batch, latent_dim) latent features
        """
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)
        combined = torch.cat([inst_features, ex_features], dim=1)
        return self.joint_encoder(combined)


class MultiOutputGP(gpytorch.models.ExactGP):
    """
    Multi-output Gaussian Process with ICM (Intrinsic Coregionalization Model) kernel.

    The ICM kernel decomposes as:
        K((x, i), (x', j)) = K_data(x, x') * K_task(i, j)

    where:
        - K_data: Matérn 5/2 kernel on latent features
        - K_task: Low-rank covariance matrix between tasks (models)

    This captures that if a prompt works well on one model, it's likely
    to work well on correlated models.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        feature_extractor: FeatureExtractor,
        num_tasks: int,
        rank: int = 2,
        input_dim: int = 768,
    ):
        """
        Args:
            train_x: (N, 2*input_dim) concatenated instruction+exemplar embeddings
            train_y: (N, num_tasks) error rates for each model
            likelihood: MultitaskGaussianLikelihood
            feature_extractor: Trained FeatureExtractor module
            num_tasks: Number of models (output dimensions)
            rank: Rank of ICM task covariance (default 2)
            input_dim: Dimension of instruction/exemplar embeddings
        """
        super().__init__(train_x, train_y, likelihood)

        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.feature_extractor = feature_extractor

        # Mean module for each task
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(),
            num_tasks=num_tasks,
        )

        # Data kernel (operates on latent features)
        data_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=feature_extractor.latent_dim,
            )
        )

        # ICM kernel = data_kernel x IndexKernel (task correlations)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_kernel,
            num_tasks=num_tasks,
            rank=rank,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """
        Forward pass through multi-output GP.

        Args:
            x: (N, 2*input_dim) concatenated embeddings

        Returns:
            MultitaskMultivariateNormal distribution over all tasks
        """
        # Split concatenated embeddings
        instruction_emb = x[:, : self.input_dim]
        exemplar_emb = x[:, self.input_dim :]

        # Extract latent features
        latent = self.feature_extractor(instruction_emb, exemplar_emb)

        # GP prior
        mean = self.mean_module(latent)
        covar = self.covar_module(latent)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)


@dataclass
class MultiOutputGPParams:
    """Container for trained multi-output GP parameters."""

    model: MultiOutputGP
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood
    feature_extractor: FeatureExtractor
    X_min: torch.Tensor
    X_max: torch.Tensor
    y_mean: torch.Tensor  # (num_tasks,)
    y_std: torch.Tensor  # (num_tasks,)
    model_names: List[str]
    device: torch.device


class MultiOutputGPTrainer:
    """
    Trainer for multi-output GP with ICM kernel.

    Manages training and inference for predicting error rates
    across multiple models simultaneously.
    """

    def __init__(
        self,
        model_names: List[str],
        latent_dim: int = 10,
        rank: int = 2,
        train_epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_names: List of model names (defines output order)
            latent_dim: Dimension of latent space
            rank: Rank of ICM task covariance
            train_epochs: Maximum training epochs
            lr: Learning rate for Adam optimizer
            patience: Early stopping patience
            device: Torch device
        """
        self.model_names = model_names
        self.num_tasks = len(model_names)
        self.latent_dim = latent_dim
        self.rank = rank
        self.train_epochs = train_epochs
        self.lr = lr
        self.patience = patience
        self.device = device or torch.device("cpu")

        self.gp_params: Optional[MultiOutputGPParams] = None

    def train(
        self,
        instruction_embeddings: np.ndarray,
        exemplar_embeddings: np.ndarray,
        model_error_rates: Dict[str, np.ndarray],
        verbose: bool = True,
    ) -> MultiOutputGPParams:
        """
        Train multi-output GP on provided data.

        Args:
            instruction_embeddings: (N, 768) instruction embeddings
            exemplar_embeddings: (N, 768) exemplar embeddings
            model_error_rates: Dict mapping model_name -> (N,) error rates
            verbose: Print training progress

        Returns:
            MultiOutputGPParams containing trained model
        """
        N = len(instruction_embeddings)

        # Validate input
        for name in self.model_names:
            if name not in model_error_rates:
                raise ValueError(f"Missing error rates for model: {name}")
            if len(model_error_rates[name]) != N:
                raise ValueError(
                    f"Error rates length mismatch for {name}: "
                    f"{len(model_error_rates[name])} vs {N}"
                )

        # Concatenate embeddings
        X = np.concatenate([instruction_embeddings, exemplar_embeddings], axis=1)

        # Stack error rates in model order: (N, num_tasks)
        y = np.stack([model_error_rates[name] for name in self.model_names], axis=1)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Input normalization to unit cube
        X_min = X_tensor.min(dim=0).values
        X_max = X_tensor.max(dim=0).values
        denom = X_max - X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X_tensor - X_min) / denom

        # Output standardization per task
        y_mean = y_tensor.mean(dim=0)  # (num_tasks,)
        y_std = y_tensor.std(dim=0)  # (num_tasks,)
        y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
        y_norm = (y_tensor - y_mean) / y_std

        # Initialize model
        feature_extractor = FeatureExtractor(
            input_dim=768,
            latent_dim=self.latent_dim,
        ).to(self.device)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)

        model = MultiOutputGP(
            train_x=X_norm,
            train_y=y_norm,
            likelihood=likelihood,
            feature_extractor=feature_extractor,
            num_tasks=self.num_tasks,
            rank=self.rank,
            input_dim=768,
        ).to(self.device)

        # Training mode
        model.train()
        likelihood.train()

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # Loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.train_epochs):
            optimizer.zero_grad()

            try:
                output = model(X_norm)
                loss = -mll(output, y_norm)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()

                if loss_val < best_loss - 1e-4:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and epoch % 500 == 0:
                    print(f"Epoch {epoch}: loss = {loss_val:.4f}")

            except Exception as e:
                if verbose:
                    print(f"Training error at epoch {epoch}: {e}")
                continue

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        if verbose:
            print(f"Multi-output GP training complete. Final loss: {best_loss:.4f}")

            # Print task correlations from ICM kernel
            task_covar = model.covar_module.task_covar_module
            if hasattr(task_covar, "covar_factor"):
                B = task_covar.covar_factor.detach()  # (num_tasks, rank)
                v = task_covar.var.detach()  # (num_tasks,)
                # Task covariance = B @ B^T + diag(v)
                task_cov = B @ B.T + torch.diag(v)
                # Normalize to correlation
                std = torch.sqrt(torch.diag(task_cov))
                task_corr = task_cov / (std.unsqueeze(0) * std.unsqueeze(1))
                print("Task correlations:")
                for i, name_i in enumerate(self.model_names):
                    for j, name_j in enumerate(self.model_names):
                        if j > i:
                            print(f"  {name_i} <-> {name_j}: {task_corr[i, j].item():.3f}")

        # Store parameters
        self.gp_params = MultiOutputGPParams(
            model=model,
            likelihood=likelihood,
            feature_extractor=feature_extractor,
            X_min=X_min,
            X_max=X_max,
            y_mean=y_mean,
            y_std=y_std,
            model_names=self.model_names,
            device=self.device,
        )

        return self.gp_params

    def predict(
        self,
        instruction_embeddings: np.ndarray,
        exemplar_embeddings: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Predict error rates for all models.

        Args:
            instruction_embeddings: (N, 768) instruction embeddings
            exemplar_embeddings: (N, 768) exemplar embeddings

        Returns:
            Tuple of:
                - means: Dict mapping model_name -> (N,) predicted error rates
                - stds: Dict mapping model_name -> (N,) prediction uncertainties
        """
        if self.gp_params is None:
            raise RuntimeError("GP not trained. Call train() first.")

        p = self.gp_params

        # Concatenate and normalize
        X = np.concatenate([instruction_embeddings, exemplar_embeddings], axis=1)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=p.device)

        denom = p.X_max - p.X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        X_norm = (X_tensor - p.X_min) / denom

        # Predict
        with torch.no_grad():
            p.model.eval()
            p.likelihood.eval()

            pred = p.likelihood(p.model(X_norm))

            # De-standardize: (N, num_tasks)
            means_tensor = pred.mean * p.y_std + p.y_mean
            # Variance for multitask: need to get it from covariance
            # For diagonal variance per task
            var_tensor = pred.variance * (p.y_std ** 2)
            stds_tensor = torch.sqrt(var_tensor)

        # Convert to dict
        means_np = means_tensor.cpu().numpy()
        stds_np = stds_tensor.cpu().numpy()

        means = {name: means_np[:, i] for i, name in enumerate(self.model_names)}
        stds = {name: stds_np[:, i] for i, name in enumerate(self.model_names)}

        return means, stds

    def predict_aggregated(
        self,
        instruction_embeddings: np.ndarray,
        exemplar_embeddings: np.ndarray,
        aggregation: str = "weighted_softmin",
        temperature: float = 0.1,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict and aggregate error rates across models.

        Args:
            instruction_embeddings: (N, 768) instruction embeddings
            exemplar_embeddings: (N, 768) exemplar embeddings
            aggregation: Aggregation strategy
            temperature: For weighted_softmin

        Returns:
            Tuple of:
                - aggregated: (N,) aggregated error rates
                - per_model: Dict mapping model_name -> (N,) error rates
        """
        from multi_model_optimizer.aggregation import aggregate_scores

        means, _ = self.predict(instruction_embeddings, exemplar_embeddings)

        N = len(instruction_embeddings)
        aggregated = np.zeros(N)

        for i in range(N):
            error_rates = {name: means[name][i] for name in self.model_names}
            aggregated[i] = aggregate_scores(error_rates, aggregation, temperature)

        return aggregated, means
