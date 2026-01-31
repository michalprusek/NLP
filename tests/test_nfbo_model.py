import torch
import pytest
from nfbo.model import RealNVP

def test_realnvp_shapes():
    dim = 4
    model = RealNVP(dim=dim, n_layers=2, hidden_dim=8)
    
    # Test forward (inference/density estimation)
    x = torch.randn(10, dim)
    z, log_det = model(x)
    
    assert z.shape == x.shape
    assert log_det.shape == (10,)
    
    # Test inverse (sampling)
    z_in = torch.randn(10, dim)
    x_out = model.inverse(z_in)
    assert x_out.shape == (10, dim)
    
    # Test reconstruction
    x_recon = model.inverse(z)
    assert torch.allclose(x, x_recon, atol=1e-5)

def test_log_prob():
    dim = 2
    model = RealNVP(dim=dim, n_layers=2, hidden_dim=8)
    x = torch.randn(5, dim)
    log_p = model.log_prob(x)
    assert log_p.shape == (5,)
    assert not torch.isnan(log_p).any()

def test_sampling():
    dim = 2
    model = RealNVP(dim=dim, n_layers=2, hidden_dim=8)
    samples = model.sample(20, device="cpu")
    assert samples.shape == (20, dim)
