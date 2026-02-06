"""Pytest configuration for RieLBO tests."""

import sys
from pathlib import Path

import torch

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class MockCodec:
    """Mock codec for testing."""

    def __init__(self, dim=256, device="cpu"):
        self.dim = dim
        self.device = device

    def encode(self, smiles_list):
        n = len(smiles_list)
        return torch.randn(n, self.dim, device=self.device)

    def decode(self, embeddings):
        n = embeddings.shape[0]
        return [f"SMILES_{i}_{torch.randint(0, 10000, (1,)).item()}" for i in range(n)]


class MockOracle:
    """Mock oracle that returns random scores."""

    def score(self, smiles):
        return torch.rand(1).item()
