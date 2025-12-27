"""GP-based Exemplar Selection for Optimized Instructions.

Uses HbBoPs-style deep kernel GP trained on full grid data
to predict optimal exemplars for novel instructions.

Key difference from original HbBoPs: Uses GTR encoder (768D)
instead of BERT for consistency with Vec2Text pipeline.
"""

import json
import torch
import torch.nn as nn
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from robust_vec2text.encoder import GTRPromptEncoder


class FeatureExtractor(nn.Module):
    """Structural-aware feature extractor for deep kernel GP.

    Architecture (from HbBoPs paper):
    - Separate encoders: Lin(768, 64) → ReLU → Lin(64, 32) → ReLU
    - Joint encoder: Lin(64, 32) → ReLU → Lin(32, 10)

    Uses GTR embeddings (768D) instead of BERT.
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()

        # Separate encoders for instruction and exemplar
        self.instruction_encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.exemplar_encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        # Joint encoder to 10-dim latent space
        self.joint_encoder = nn.Sequential(
            nn.Linear(2 * 32, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(
        self, instruction_emb: torch.Tensor, exemplar_emb: torch.Tensor
    ) -> torch.Tensor:
        # Handle both 2D and 3D inputs (BoTorch passes 3D during posterior)
        inst_features = self.instruction_encoder(instruction_emb)
        ex_features = self.exemplar_encoder(exemplar_emb)
        combined = torch.cat([inst_features, ex_features], dim=-1)  # concat on last dim
        return self.joint_encoder(combined)


class DeepKernelGP(gpytorch.models.ExactGP, GPyTorchModel):
    """Gaussian Process with structural-aware deep kernel.

    Uses ARD Matérn 5/2 kernel on 10-dim latent features.
    Input: [instruction_768 || exemplar_768] = 1536D

    Inherits from GPyTorchModel for BoTorch compatibility
    (enables LogExpectedImprovement and other acquisition functions).
    """

    _num_outputs = 1  # Required for BoTorch

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        feature_extractor: FeatureExtractor,
        input_dim: int = 768,
    ):
        # train_y must be (N,) not (N, 1) for ExactGP
        train_y_squeezed = train_y.squeeze(-1) if train_y.dim() > 1 else train_y
        super().__init__(train_x, train_y_squeezed, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=10,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        self.feature_extractor = feature_extractor
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        # Split concatenated embeddings [inst_768 || ex_768]
        # Handle both 2D (batch, 1536) and 3D (batch, n, 1536) inputs
        # BoTorch passes 3D tensors during posterior computation
        if x.dim() == 3:
            # Shape: (batch, n, 1536) -> split on last dim
            instruction_emb = x[..., :self.input_dim]  # (..., 768)
            exemplar_emb = x[..., self.input_dim:]     # (..., 768)
        else:
            # Shape: (batch, 1536) -> split on last dim
            instruction_emb = x[:, :self.input_dim]
            exemplar_emb = x[:, self.input_dim:]
        latent = self.feature_extractor(instruction_emb, exemplar_emb)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(latent), self.covar_module(latent)
        )


class ExemplarSelector:
    """Select optimal exemplars using HbBoPs-style GP prediction.

    Trains a deep kernel GP on full grid data (625 pairs) and uses it
    to predict error rates for novel (instruction, exemplar) combinations.
    """

    def __init__(
        self,
        instructions: List[str],
        exemplars: List[str],
        gtr: Optional[GTRPromptEncoder] = None,
        device: str = "cuda",
    ):
        """Initialize selector.

        Args:
            instructions: List of all instructions (for embedding cache)
            exemplars: List of all exemplars
            gtr: GTR encoder (will create one if not provided)
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.instructions = instructions
        self.exemplars = exemplars

        # GTR encoder (same as Vec2Text)
        self.gtr = gtr if gtr is not None else GTRPromptEncoder()

        # Pre-compute embeddings
        print("Pre-computing GTR embeddings for instructions and exemplars...")
        self.instruction_embeddings = self._encode_texts(instructions)
        self.exemplar_embeddings = self._encode_texts(exemplars)
        print(f"  Cached {len(self.instruction_embeddings)} instruction embeddings")
        print(f"  Cached {len(self.exemplar_embeddings)} exemplar embeddings")

        # GP components (initialized during training)
        self.gp_model: Optional[DeepKernelGP] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self.feature_extractor: Optional[FeatureExtractor] = None

        # Training data (stored for incremental updates)
        self.X_train: Optional[torch.Tensor] = None  # (N, 1536)
        self.y_train: Optional[torch.Tensor] = None  # (N,)

        # Normalization parameters
        self.X_min: Optional[torch.Tensor] = None
        self.X_max: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

    def _encode_texts(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        """Encode texts to GTR embeddings."""
        embeddings = {}
        for idx, text in enumerate(texts):
            emb = self.gtr.encode_tensor(text)
            embeddings[idx] = emb.squeeze()
        return embeddings

    def train_from_grid(
        self,
        grid_path: str,
        top_k: int = 25,
        epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
    ) -> bool:
        """Train GP on top-k prompts from grid data.

        Args:
            grid_path: Path to grid JSONL file
            top_k: Number of top prompts to use (sorted by error rate)
            epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if verbose:
            print(f"Loading grid data from {grid_path}...")

        # Load grid data
        grid_data = []
        with open(grid_path, "r") as f:
            for line in f:
                grid_data.append(json.loads(line))

        # Sort by error rate and take top-k
        grid_data.sort(key=lambda x: x["error_rate"])
        grid_data = grid_data[:top_k]

        if verbose:
            print(f"  Using top {top_k} prompts (error range: {grid_data[0]['error_rate']:.4f} - {grid_data[-1]['error_rate']:.4f})")

        if verbose:
            print(f"  Loaded {len(grid_data)} grid entries")

        # Prepare training tensors
        X_inst_list = []
        X_ex_list = []
        y_list = []

        for entry in grid_data:
            inst_id = entry["instruction_id"]
            ex_id = entry["exemplar_id"]
            error_rate = entry["error_rate"]

            X_inst_list.append(self.instruction_embeddings[inst_id])
            X_ex_list.append(self.exemplar_embeddings[ex_id])
            y_list.append(error_rate)

        X_inst = torch.stack(X_inst_list).to(self.device)
        X_ex = torch.stack(X_ex_list).to(self.device)
        y = torch.tensor(y_list, dtype=torch.float32, device=self.device)

        # Concatenate for GP input: [inst_768 || ex_768]
        X = torch.cat([X_inst, X_ex], dim=1)  # (N, 1536)

        if verbose:
            print(f"  Training data shape: X={X.shape}, y={y.shape}")
            print(f"  Error rate range: [{y.min():.4f}, {y.max():.4f}]")

        # Unit cube normalization for inputs
        self.X_min = X.min(dim=0)[0]
        self.X_max = X.max(dim=0)[0]
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0  # Avoid division by zero
        X_norm = (X - self.X_min) / denominator

        # Z-score standardization for outputs
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Initialize GP
        self.feature_extractor = FeatureExtractor(input_dim=768).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = DeepKernelGP(
            X_norm, y_norm, self.likelihood, self.feature_extractor
        ).to(self.device)

        # Training
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0

        if verbose:
            print("Training GP...")

        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_norm)
                    loss = -mll(output, y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

                    if verbose and (epoch + 1) % 100 == 0:
                        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

                except RuntimeError as e:
                    if "cholesky" in str(e).lower():
                        if verbose:
                            print(f"  Cholesky error at epoch {epoch + 1}")
                        return False
                    raise

        if verbose:
            print(f"  GP training complete (epochs={epoch + 1}, loss={best_loss:.4f})")

        return True

    def predict_error(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor,
    ) -> Tuple[float, float]:
        """Predict error rate for (instruction, exemplar) pair.

        Args:
            instruction_emb: Instruction embedding (768,)
            exemplar_emb: Exemplar embedding (768,)

        Returns:
            Tuple of (mean, std) predictions
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained. Call train_from_grid() first.")

        # Ensure correct shape
        if instruction_emb.dim() == 1:
            instruction_emb = instruction_emb.unsqueeze(0)
        if exemplar_emb.dim() == 1:
            exemplar_emb = exemplar_emb.unsqueeze(0)

        # Move to device
        instruction_emb = instruction_emb.to(self.device)
        exemplar_emb = exemplar_emb.to(self.device)

        # Concatenate
        X = torch.cat([instruction_emb, exemplar_emb], dim=1)

        # Normalize
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0
        X_norm = (X - self.X_min) / denominator

        # Predict
        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(X_norm))
            mean_norm = pred.mean.item()
            std_norm = pred.stddev.item()

        # Denormalize
        mean = mean_norm * self.y_std.item() + self.y_mean.item()
        std = std_norm * self.y_std.item()

        return mean, std

    def select_best_exemplar(
        self,
        instruction: str,
        top_k: int = 5,
        verbose: bool = True,
    ) -> List[Tuple[int, str, float, float]]:
        """Find best exemplars for given instruction.

        Args:
            instruction: Novel instruction text
            top_k: Return top K exemplars
            verbose: Print progress

        Returns:
            List of (exemplar_id, exemplar_text, predicted_error, uncertainty)
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained. Call train_from_grid() first.")

        # Encode instruction with GTR
        inst_emb = self.gtr.encode_tensor(instruction).to(self.device)

        if verbose:
            print(f"Predicting error rates for {len(self.exemplars)} exemplars...")

        # Predict for all exemplars
        predictions = []

        for ex_id, ex_emb in self.exemplar_embeddings.items():
            mean, std = self.predict_error(inst_emb, ex_emb.to(self.device))
            predictions.append((ex_id, self.exemplars[ex_id], mean, std))

        # Sort by predicted error (ascending - lower is better)
        predictions.sort(key=lambda x: x[2])

        if verbose:
            print(f"  Best predicted error: {predictions[0][2]:.4f}")
            print(f"  Worst predicted error: {predictions[-1][2]:.4f}")

        return predictions[:top_k]

    def select_for_instruction_embedding(
        self,
        instruction_emb: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[int, str, float, float]]:
        """Find best exemplars for given instruction embedding.

        Useful when instruction embedding is already computed.

        Args:
            instruction_emb: Instruction embedding (768,)
            top_k: Return top K exemplars

        Returns:
            List of (exemplar_id, exemplar_text, predicted_error, uncertainty)
        """
        if self.gp_model is None:
            raise RuntimeError("GP not trained. Call train_from_grid() first.")

        instruction_emb = instruction_emb.to(self.device)

        predictions = []
        for ex_id, ex_emb in self.exemplar_embeddings.items():
            mean, std = self.predict_error(instruction_emb, ex_emb.to(self.device))
            predictions.append((ex_id, self.exemplars[ex_id], mean, std))

        predictions.sort(key=lambda x: x[2])
        return predictions[:top_k]

    def train_with_decoded_embeddings(
        self,
        decoded_instruction_embeddings: Dict[int, torch.Tensor],
        exemplar_embeddings: Dict[int, torch.Tensor],
        grid_path: str,
        top_k: int = 25,
        epochs: int = 3000,
        lr: float = 0.01,
        patience: int = 10,
        verbose: bool = True,
    ) -> bool:
        """Train GP using VAE-decoded instruction embeddings.

        This is the key fix for EI=0 problem. By training on decoded
        embeddings, the GP operates in the same distribution as the
        gradient-based EI optimization.

        Reference: COWBOYS paper (Return of the Latent Space COWBOYS)

        Args:
            decoded_instruction_embeddings: Dict mapping inst_id → VAE-decoded embedding
            exemplar_embeddings: Dict mapping ex_id → original GTR embedding
            grid_path: Path to grid JSONL file
            top_k: Number of top prompts to use (sorted by error rate)
            epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if verbose:
            print(f"Loading grid data from {grid_path}...")

        # Load grid data
        grid_data = []
        with open(grid_path, "r") as f:
            for line in f:
                grid_data.append(json.loads(line))

        # Sort by error rate and take top-k
        grid_data.sort(key=lambda x: x["error_rate"])
        grid_data = grid_data[:top_k]

        if verbose:
            print(f"  Using top {top_k} prompts (error range: {grid_data[0]['error_rate']:.4f} - {grid_data[-1]['error_rate']:.4f})")
            print(f"  Training on VAE-DECODED instruction embeddings")

        # Prepare training tensors using DECODED embeddings
        X_inst_list = []
        X_ex_list = []
        y_list = []

        for entry in grid_data:
            inst_id = entry["instruction_id"]
            ex_id = entry["exemplar_id"]
            error_rate = entry["error_rate"]

            # Use DECODED instruction embedding (key difference!)
            X_inst_list.append(decoded_instruction_embeddings[inst_id].to(self.device))
            X_ex_list.append(exemplar_embeddings[ex_id].to(self.device))
            y_list.append(error_rate)

        X_inst = torch.stack(X_inst_list).to(self.device)
        X_ex = torch.stack(X_ex_list).to(self.device)
        y = torch.tensor(y_list, dtype=torch.float32, device=self.device)

        # Concatenate for GP input: [inst_768 || ex_768]
        X = torch.cat([X_inst, X_ex], dim=1)  # (N, 1536)

        # Store raw training data for incremental updates
        self.X_train = X.clone()
        self.y_train = y.clone()

        if verbose:
            print(f"  Training data shape: X={X.shape}, y={y.shape}")
            print(f"  Error rate range: [{y.min():.4f}, {y.max():.4f}]")

        # Train GP on stored data
        return self._train_gp(epochs=epochs, lr=lr, patience=patience, verbose=verbose)

    def _train_gp(
        self,
        epochs: int = 500,
        lr: float = 0.01,
        patience: int = 10,
        verbose: bool = False,
    ) -> bool:
        """Internal method to train GP on stored X_train and y_train.

        Recomputes normalization and reinitializes GP model.

        Args:
            epochs: Maximum training epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            True if training succeeded
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data. Call train_with_decoded_embeddings first.")

        X = self.X_train
        y = self.y_train

        # Unit cube normalization for inputs
        self.X_min = X.min(dim=0)[0]
        self.X_max = X.max(dim=0)[0]
        denominator = self.X_max - self.X_min
        denominator[denominator == 0] = 1.0  # Avoid division by zero
        X_norm = (X - self.X_min) / denominator

        # Z-score standardization for outputs
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        # Initialize GP
        self.feature_extractor = FeatureExtractor(input_dim=768).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = DeepKernelGP(
            X_norm, y_norm, self.likelihood, self.feature_extractor
        ).to(self.device)

        # Training
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.AdamW(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        best_loss = float("inf")
        patience_counter = 0

        if verbose:
            print("Training GP on decoded embeddings...")

        with gpytorch.settings.cholesky_jitter(1e-4):
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    output = self.gp_model(X_norm)
                    loss = -mll(output, y_norm)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

                    if verbose and (epoch + 1) % 100 == 0:
                        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

                except RuntimeError as e:
                    if "cholesky" in str(e).lower():
                        if verbose:
                            print(f"  Cholesky error at epoch {epoch + 1}")
                        return False
                    raise

        if verbose:
            print(f"  GP training complete (epochs={epoch + 1}, loss={best_loss:.4f})")

        return True

    def add_observation_and_retrain(
        self,
        decoded_inst_emb: torch.Tensor,
        exemplar_emb: torch.Tensor,
        error_rate: float,
        epochs: int = 100,
        verbose: bool = False,
    ) -> bool:
        """Add new observation and retrain GP.

        Appends new data point to existing training data,
        recomputes normalization, and retrains GP.

        Args:
            decoded_inst_emb: VAE-decoded instruction embedding (768,)
            exemplar_emb: Exemplar embedding (768,)
            error_rate: Evaluated error rate
            epochs: GP training epochs
            verbose: Print progress

        Returns:
            True if retraining succeeded
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data. Call train_with_decoded_embeddings first.")

        # Ensure correct shapes and device
        if decoded_inst_emb.dim() == 2:
            decoded_inst_emb = decoded_inst_emb.squeeze(0)
        if exemplar_emb.dim() == 2:
            exemplar_emb = exemplar_emb.squeeze(0)

        decoded_inst_emb = decoded_inst_emb.to(self.device)
        exemplar_emb = exemplar_emb.to(self.device)

        # Create new data point: [inst_768 || ex_768]
        new_X = torch.cat([decoded_inst_emb, exemplar_emb]).unsqueeze(0)  # (1, 1536)
        new_y = torch.tensor([error_rate], device=self.device)  # (1,)

        # Append to training data
        self.X_train = torch.cat([self.X_train, new_X], dim=0)
        self.y_train = torch.cat([self.y_train, new_y], dim=0)

        if verbose:
            print(f"  Added observation: error={error_rate:.4f}, total samples={len(self.y_train)}")

        # Retrain GP with updated data
        return self._train_gp(epochs=epochs, patience=10, verbose=verbose)
