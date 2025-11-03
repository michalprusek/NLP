"""Deep residual MLP classifier for binary classification."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim: int, dropout: float = 0.3):
        """Initialize residual block.

        Args:
            dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.relu(x + self.net(x))


class DeepResidualMLP(nn.Module):
    """Deep residual MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_residual_blocks: int = 3,
        dropout: float = 0.3
    ):
        """Initialize deep residual MLP.

        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden dimensions (e.g., [512, 256, 128])
            num_residual_blocks: Number of residual blocks per hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        layers = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Residual blocks at first hidden dimension
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_dims[0], dropout))

        # Progressive dimensionality reduction with residual connections
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]

            # Projection to next dimension
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Residual blocks at current dimension
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(out_dim, dropout))

        self.network = nn.Sequential(*layers)

        # Output layer (binary classification)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, 1)
        """
        x = self.network(x)
        return self.output(x)
