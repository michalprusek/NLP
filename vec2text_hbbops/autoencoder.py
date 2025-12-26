"""Regularized Autoencoder for prompt embedding compression.

Maps concatenated instruction+exemplar embeddings (1536D) to/from
10D latent space compatible with DeepKernelGP.

Regularization strategies for low-data regime (~200 samples):
- Denoising: Gaussian noise injection during training
- Dropout: In encoder and decoder layers
- Weight decay: L2 regularization via optimizer
- Sparse penalty: L1 on latent activations
- Cosine loss: Preserve embedding direction for Vec2Text
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional


class PromptAutoencoder(nn.Module):
    """Regularized Autoencoder for prompt embeddings.

    Architecture:
        Encoder: 1536D -> 512 -> 128 -> 10D (latent)
        Decoder: 10D -> 128 -> 512 -> 1536D

    The 1536D input is concatenation of:
        - instruction embedding (768D from GTR)
        - exemplar embedding (768D from GTR)

    Attributes:
        input_dim: Input dimension (1536 = 768 + 768)
        latent_dim: Latent space dimension (10 for GP)
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout probability
        noise_std: Gaussian noise std for denoising
    """

    def __init__(
        self,
        input_dim: int = 1536,
        latent_dim: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        noise_std: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """Initialize autoencoder.

        Args:
            input_dim: Input embedding dimension (1536 for concat inst+ex)
            latent_dim: Latent space dimension (10 for GP compatibility)
            hidden_dims: Hidden layer sizes (default: [512, 128])
            dropout_rate: Dropout probability in hidden layers
            noise_std: Gaussian noise std for denoising (0 to disable)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.use_batch_norm = use_batch_norm

        # Build encoder: input_dim -> hidden_dims -> latent_dim
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]

        for i in range(len(dims) - 1):
            # Linear layer
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Only add activation/regularization for non-final layers
            if i < len(dims) - 2:
                encoder_layers.append(nn.LeakyReLU(0.1))
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
                encoder_layers.append(nn.Dropout(dropout_rate))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: latent_dim -> hidden_dims (reversed) -> input_dim
        decoder_layers = []
        dims_rev = [latent_dim] + hidden_dims[::-1] + [input_dim]

        for i in range(len(dims_rev) - 1):
            decoder_layers.append(nn.Linear(dims_rev[i], dims_rev[i + 1]))

            # Only add activation/regularization for non-final layers
            if i < len(dims_rev) - 2:
                decoder_layers.append(nn.LeakyReLU(0.1))
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(dims_rev[i + 1]))
                decoder_layers.append(nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for denoising autoencoder.

        Only adds noise during training.

        Args:
            x: Input tensor

        Returns:
            Noisy tensor (during training) or original (during eval)
        """
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (batch, 1536)

        Returns:
            Latent tensor of shape (batch, 10)
        """
        x_noisy = self.add_noise(x)
        return self.encoder(x_noisy)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.

        Args:
            z: Latent tensor of shape (batch, 10)

        Returns:
            Reconstruction of shape (batch, 1536)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch, 1536)

        Returns:
            Tuple of (reconstruction, latent)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def split_reconstruction(
        self, x_recon: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split 1536D reconstruction into instruction and exemplar.

        Args:
            x_recon: Reconstruction of shape (batch, 1536)

        Returns:
            Tuple of (instruction_emb, exemplar_emb), each (batch, 768)
        """
        inst_emb = x_recon[:, :768]
        ex_emb = x_recon[:, 768:]
        return inst_emb, ex_emb


class AutoencoderLoss(nn.Module):
    """Combined loss for regularized autoencoder.

    Loss = L_recon + lambda_cosine * L_cosine + lambda_sparse * L_sparse

    Where:
        L_recon: MSE reconstruction loss
        L_cosine: Cosine embedding loss (preserves direction)
        L_sparse: L1 penalty on latent activations
    """

    def __init__(
        self,
        lambda_cosine: float = 0.5,
        lambda_sparse: float = 0.001,
    ):
        """Initialize loss function.

        Args:
            lambda_cosine: Weight for cosine similarity loss
            lambda_sparse: Weight for L1 sparsity penalty on latent
        """
        super().__init__()

        self.lambda_cosine = lambda_cosine
        self.lambda_sparse = lambda_sparse

        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            x: Original input (batch, 1536)
            x_recon: Reconstruction (batch, 1536)
            z: Latent representation (batch, 10)

        Returns:
            Tuple of (total_loss, components_dict)
        """
        batch_size = x.size(0)

        # Reconstruction loss (MSE)
        loss_recon = self.mse(x_recon, x)

        # Cosine similarity loss - computed separately for instruction and exemplar
        # This preserves the direction of embeddings which is critical for Vec2Text
        inst_orig, ex_orig = x[:, :768], x[:, 768:]
        inst_recon, ex_recon = x_recon[:, :768], x_recon[:, 768:]

        targets = torch.ones(batch_size, device=x.device)
        loss_cosine_inst = self.cosine(inst_orig, inst_recon, targets)
        loss_cosine_ex = self.cosine(ex_orig, ex_recon, targets)
        loss_cosine = (loss_cosine_inst + loss_cosine_ex) / 2

        # Sparse penalty on latent (L1 regularization)
        loss_sparse = z.abs().mean()

        # Total loss
        total = (
            loss_recon
            + self.lambda_cosine * loss_cosine
            + self.lambda_sparse * loss_sparse
        )

        components = {
            "recon": loss_recon.item(),
            "cosine": loss_cosine.item(),
            "sparse": loss_sparse.item(),
            "total": total.item(),
        }

        return total, components


class InstructionAutoencoder(nn.Module):
    """Autoencoder for instruction embeddings only (768D → 10D → 768D).

    Smaller architecture for instruction-only optimization.
    Exemplars are kept fixed from grid.

    Architecture:
        Encoder: 768D -> 256 -> 64 -> 10D (latent)
        Decoder: 10D -> 64 -> 256 -> 768D
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        noise_std: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """Initialize instruction autoencoder.

        Args:
            input_dim: Input embedding dimension (768 for instruction)
            latent_dim: Latent space dimension (10 for GP compatibility)
            hidden_dims: Hidden layer sizes (default: [256, 64])
            dropout_rate: Dropout probability in hidden layers
            noise_std: Gaussian noise std for denoising (0 to disable)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 64]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.use_batch_norm = use_batch_norm

        # Build encoder: input_dim -> hidden_dims -> latent_dim
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]

        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.LeakyReLU(0.1))
                if use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
                encoder_layers.append(nn.Dropout(dropout_rate))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: latent_dim -> hidden_dims (reversed) -> input_dim
        decoder_layers = []
        dims_rev = [latent_dim] + hidden_dims[::-1] + [input_dim]

        for i in range(len(dims_rev) - 1):
            decoder_layers.append(nn.Linear(dims_rev[i], dims_rev[i + 1]))
            if i < len(dims_rev) - 2:
                decoder_layers.append(nn.LeakyReLU(0.1))
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(dims_rev[i + 1]))
                decoder_layers.append(nn.Dropout(dropout_rate))

        self.decoder = nn.Sequential(*decoder_layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for denoising autoencoder."""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (batch, 768)

        Returns:
            Latent tensor of shape (batch, 10)
        """
        x_noisy = self.add_noise(x)
        return self.encoder(x_noisy)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.

        Args:
            z: Latent tensor of shape (batch, 10)

        Returns:
            Reconstruction of shape (batch, 768)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch, 768)

        Returns:
            Tuple of (reconstruction, latent)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class InstructionAutoencoderLoss(nn.Module):
    """Loss for instruction-only autoencoder.

    Loss = L_recon + lambda_cosine * L_cosine + lambda_sparse * L_sparse
    """

    def __init__(
        self,
        lambda_cosine: float = 0.5,
        lambda_sparse: float = 0.001,
    ):
        super().__init__()
        self.lambda_cosine = lambda_cosine
        self.lambda_sparse = lambda_sparse
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            x: Original input (batch, 768)
            x_recon: Reconstruction (batch, 768)
            z: Latent representation (batch, 10)

        Returns:
            Tuple of (total_loss, components_dict)
        """
        batch_size = x.size(0)

        # Reconstruction loss (MSE)
        loss_recon = self.mse(x_recon, x)

        # Cosine similarity loss
        targets = torch.ones(batch_size, device=x.device)
        loss_cosine = self.cosine(x, x_recon, targets)

        # Sparse penalty on latent (L1 regularization)
        loss_sparse = z.abs().mean()

        # Total loss
        total = (
            loss_recon
            + self.lambda_cosine * loss_cosine
            + self.lambda_sparse * loss_sparse
        )

        components = {
            "recon": loss_recon.item(),
            "cosine": loss_cosine.item(),
            "sparse": loss_sparse.item(),
            "total": total.item(),
        }

        return total, components
