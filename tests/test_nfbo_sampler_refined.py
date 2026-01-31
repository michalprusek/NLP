import torch
import pytest
from nfbo.sampler import NFBoSampler

def test_sampler_latent_bo_logic():
    dim = 2
    # Use CPU for fast testing
    sampler = NFBoSampler(dim=dim, n_flow_layers=2, hidden_dim=8, flow_epochs=1, device="cpu")
    
    # Create synthetic data
    n_data = 20
    train_X = torch.randn(n_data, dim)
    train_Y = torch.randn(n_data)
    
    # 1. Test fitting flow
    sampler.fit_flow(train_X)
    
    # 2. Test candidate generation (Latent BO)
    candidates = sampler.generate_candidates(train_X, train_Y, n_candidates=1)
    
    assert candidates.shape == (1, dim)
    assert not torch.isnan(candidates).any()
    
    # Test batch generation
    candidates_batch = sampler.generate_candidates(train_X, train_Y, n_candidates=5)
    assert candidates_batch.shape == (5, dim)
