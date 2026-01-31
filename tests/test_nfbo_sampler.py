import torch
import pytest
from nfbo.sampler import NFBoSampler

def test_sampler_fit_and_sample():
    dim = 2
    sampler = NFBoSampler(dim=dim, n_flow_layers=2, hidden_dim=8, flow_epochs=2, device="cpu")

    # Create synthetic data: skewed towards [1, 1]
    n_data = 100
    train_X = torch.randn(n_data, dim)
    # Higher score for points closer to [1, 1]
    train_Y = -((train_X - 1.0)**2).sum(dim=-1)

    # Fit the flow on training data
    sampler.fit_flow(train_X)

    # Generate candidates using the trained flow and GP
    n_candidates = 1
    candidates = sampler.generate_candidates(train_X, train_Y, n_candidates=n_candidates)

    assert candidates.shape == (n_candidates, dim)
    assert not torch.isnan(candidates).any()
