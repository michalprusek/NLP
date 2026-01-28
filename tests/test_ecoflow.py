"""
Unit tests for EcoFlow-BO components.

Run with: pytest tests/test_ecoflow.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecoflow_bo.config import EcoFlowConfig, EncoderConfig, DiTVelocityNetConfig, DecoderConfig
from ecoflow_bo.encoder import MatryoshkaEncoder
from ecoflow_bo.velocity_network import VelocityNetwork
from ecoflow_bo.cfm_decoder import RectifiedFlowDecoder
from ecoflow_bo.losses import KLDivergenceLoss, InfoNCELoss, MatryoshkaCFMLoss
from ecoflow_bo.latent_gp import CoarseToFineGP
from ecoflow_bo.density_acquisition import DensityAwareAcquisition
from ecoflow_bo.cycle_consistency import CycleConsistencyChecker
from ecoflow_bo.perceiver_decoder import PerceiverDecoder, PerceiverDecoderConfig


# Fixtures
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def encoder_config():
    return EncoderConfig(
        input_dim=768,
        latent_dim=8,
        hidden_dims=[256, 128],
        dropout=0.1,
        matryoshka_dims=[2, 4, 8],
    )


@pytest.fixture
def velocity_config():
    """DiT-based velocity network config for testing."""
    return DiTVelocityNetConfig(
        data_dim=768,
        condition_dim=8,
        hidden_dim=128,  # Small for fast tests
        n_layers=2,
        n_heads=4,
    )


@pytest.fixture
def encoder(encoder_config, device):
    return MatryoshkaEncoder(encoder_config).to(device)


@pytest.fixture
def velocity_net(velocity_config, device):
    return VelocityNetwork(velocity_config).to(device)


@pytest.fixture
def decoder(velocity_net):
    config = DecoderConfig(euler_steps=10)
    return RectifiedFlowDecoder(velocity_net, config)


# Encoder tests
class TestMatryoshkaEncoder:
    def test_forward_shape(self, encoder, device):
        x = torch.randn(32, 768, device=device)
        z, mu, log_sigma = encoder(x)

        assert z.shape == (32, 8)
        assert mu.shape == (32, 8)
        assert log_sigma.shape == (32, 8)

    def test_deterministic_encode(self, encoder, device):
        x = torch.randn(32, 768, device=device)

        encoder.eval()
        z1 = encoder.encode_deterministic(x)
        z2 = encoder.encode_deterministic(x)

        assert torch.allclose(z1, z2)

    def test_matryoshka_embeddings(self, encoder, device):
        x = torch.randn(32, 768, device=device)
        z_list, mu, log_sigma = encoder.get_matryoshka_embeddings(x)

        assert len(z_list) == 3
        assert z_list[0].shape == (32, 2)
        assert z_list[1].shape == (32, 4)
        assert z_list[2].shape == (32, 8)

    def test_simcse_augmentation(self, encoder, device):
        """Two forward passes with dropout should give different z (SimCSE-style)."""
        x = torch.randn(32, 768, device=device)

        # SimCSE augmentation: two forward passes with different dropout masks
        encoder.train()
        z1, _, _ = encoder(x)
        z2, _, _ = encoder(x)

        # Different due to dropout
        assert not torch.allclose(z1, z2)
        # But correlated (same input)
        cosine_sim = torch.nn.functional.cosine_similarity(z1, z2, dim=-1).mean()
        assert cosine_sim > 0.5


# Velocity network tests
class TestVelocityNetwork:
    def test_forward_shape(self, velocity_net, device):
        B = 32
        x_t = torch.randn(B, 768, device=device)
        t = torch.rand(B, device=device)
        z = torch.randn(B, 8, device=device)

        v = velocity_net(x_t, t, z)
        assert v.shape == (B, 768)

    def test_time_embedding(self, velocity_net, device):
        """Different times should produce different internal activations."""
        B = 32
        x_t = torch.randn(B, 768, device=device)
        z = torch.randn(B, 8, device=device)

        t1 = torch.zeros(B, device=device)
        t2 = torch.ones(B, device=device)

        # Get time embeddings directly
        t_emb1 = velocity_net.time_embed(t1)
        t_emb2 = velocity_net.time_embed(t2)

        # Time embeddings should differ for different times
        assert not torch.allclose(t_emb1, t_emb2)


# Decoder tests
class TestRectifiedFlowDecoder:
    def test_cfm_loss(self, decoder, device):
        x_target = torch.randn(32, 768, device=device)
        z = torch.randn(32, 8, device=device)

        loss = decoder.compute_cfm_loss(x_target, z)

        assert loss.ndim == 0  # Scalar
        assert loss > 0

    def test_decode_shape(self, decoder, device):
        z = torch.randn(32, 8, device=device)

        x = decoder.decode(z, n_steps=5)

        assert x.shape == (32, 768)

    def test_decode_deterministic(self, decoder, device):
        z = torch.randn(8, 8, device=device)

        x1 = decoder.decode_deterministic(z, seed=42)
        x2 = decoder.decode_deterministic(z, seed=42)

        assert torch.allclose(x1, x2, atol=1e-5)


# Loss tests
class TestLosses:
    def test_kl_loss(self, device):
        kl_loss = KLDivergenceLoss()

        mu = torch.zeros(32, 8, device=device)
        log_sigma = torch.zeros(32, 8, device=device)

        # KL(N(0,1) || N(0,1)) = 0
        loss = kl_loss(mu, log_sigma)
        assert loss < 0.01

        # Non-zero mu should increase KL
        mu = torch.ones(32, 8, device=device)
        loss = kl_loss(mu, log_sigma)
        assert loss > 0.1

    def test_infonce_loss(self, device):
        """Test InfoNCE loss for contrastive learning."""
        loss_fn = InfoNCELoss(temperature=0.05)

        z1 = torch.randn(64, 8, device=device)
        z2 = z1 + 0.1 * torch.randn_like(z1)  # Small perturbation (positive pairs)

        loss = loss_fn(z1, z2)

        assert loss > 0
        assert loss.ndim == 0  # Scalar

    def test_matryoshka_cfm_loss(self, encoder, decoder, device):
        """Test Matryoshka CFM loss at multiple dimensions."""
        loss_fn = MatryoshkaCFMLoss(
            matryoshka_dims=[2, 4, 8],
            matryoshka_weights=[0.4, 0.35, 0.25],
        )

        x = torch.randn(16, 768, device=device)
        z = torch.randn(16, 8, device=device)

        loss, details = loss_fn(decoder, x, z)

        assert loss > 0
        assert "cfm_dim2" in details
        assert "cfm_dim4" in details
        assert "cfm_dim8" in details


# GP tests
class TestCoarseToFineGP:
    def test_stage_progression(self, device):
        gp = CoarseToFineGP()

        assert gp.current_stage == 0
        # Default config: active_dims_schedule=[[0,1,2,3], [0-7], [0-15]]
        assert gp.active_dims == [0, 1, 2, 3]

        # Default config: points_per_stage=[10, 15, 30], so need 10 to advance to stage 1
        # Add points incrementally to trigger stage advance
        z = torch.randn(10, 16, device=device)  # latent_dim=16
        y = torch.randn(10, device=device)
        gp.fit(z, y)

        # Now update with more points to trigger advancement
        z_new = torch.randn(5, 16, device=device)
        y_new = torch.randn(5, device=device)
        gp.update(z_new, y_new)

        # With 15 points (>= 10), should advance to stage 1
        assert gp.current_stage >= 1

    def test_predict(self, device):
        gp = CoarseToFineGP()

        z = torch.randn(10, 8, device=device)
        y = torch.randn(10, device=device)

        gp.fit(z, y)

        z_test = torch.randn(5, 8, device=device)
        mean, var = gp.predict(z_test)

        assert mean.shape == (5,)
        assert var.shape == (5,)
        assert (var >= 0).all()


# Acquisition tests
class TestDensityAwareAcquisition:
    def test_log_prior_density(self, device):
        acq = DensityAwareAcquisition()

        # Origin has highest density under N(0,I)
        z_origin = torch.zeros(1, 8, device=device)
        z_far = torch.ones(1, 8, device=device) * 3

        density_origin = acq.log_prior_density(z_origin)
        density_far = acq.log_prior_density(z_far)

        assert density_origin > density_far

    def test_candidate_generation(self, device):
        acq = DensityAwareAcquisition()
        gp = CoarseToFineGP()

        # Initialize GP with some data
        z = torch.randn(10, 8, device=device)
        y = torch.randn(10, device=device)
        gp.fit(z, y)

        z_best = gp.train_z[y.argmax()]

        candidates = acq.generate_candidates(gp, z_best, n_candidates=100)

        assert candidates.shape == (100, 8)


# Cycle consistency tests
class TestCycleConsistency:
    def test_cycle_error(self, encoder, decoder, device):
        checker = CycleConsistencyChecker(encoder, decoder)

        z = torch.randn(8, 8, device=device)

        encoder.eval()
        decoder.velocity_net.eval()

        x_decoded, z_reencoded, error = checker.compute_cycle_error(z)

        assert x_decoded.shape == (8, 768)
        assert z_reencoded.shape == (8, 8)
        assert error.shape == (8,)

    def test_validity_check(self, encoder, decoder, device):
        checker = CycleConsistencyChecker(encoder, decoder)
        checker.error_threshold = 10.0  # Lenient for test

        z = torch.randn(8, 8, device=device) * 0.5  # Small z for better reconstruction

        encoder.eval()
        decoder.velocity_net.eval()

        valid_mask, errors = checker.is_valid(z)

        assert valid_mask.shape == (8,)
        assert errors.shape == (8,)


# Integration test
class TestIntegration:
    def test_encode_decode_cycle(self, encoder, decoder, device):
        """Test full encode-decode cycle."""
        x = torch.randn(16, 768, device=device)

        encoder.eval()
        decoder.velocity_net.eval()

        with torch.no_grad():
            z = encoder.encode_deterministic(x)
            x_recon = decoder.decode_deterministic(z)

        assert x_recon.shape == x.shape

        # Cosine similarity should be positive (learned mapping)
        # Note: Without training, this is just random, so we just check it runs
        cosine_sim = torch.nn.functional.cosine_similarity(x, x_recon, dim=-1).mean()
        assert not torch.isnan(cosine_sim)


# Perceiver Decoder tests
class TestPerceiverDecoder:
    @pytest.fixture
    def perceiver_config(self):
        return PerceiverDecoderConfig(
            latent_dim=16,
            output_dim=768,
            hidden_size=256,  # Small for fast tests
            depth=2,
            num_heads=8,
            readout_heads=8,
        )

    @pytest.fixture
    def perceiver(self, perceiver_config, device):
        return PerceiverDecoder(perceiver_config).to(device)

    def test_forward_shape(self, perceiver, device):
        """Test basic forward pass shape."""
        z = torch.randn(32, 16, device=device)
        out = perceiver(z)
        assert out.shape == (32, 768)

    def test_forward_with_matryoshka(self, perceiver, device):
        """Test forward with Matryoshka masking."""
        z = torch.randn(8, 16, device=device)

        # Full dims
        out_full = perceiver(z)

        # Masked to 8 dims
        out_8d = perceiver.forward_with_matryoshka(z, active_dims=8)

        # Masked to 4 dims
        out_4d = perceiver.forward_with_matryoshka(z, active_dims=4)

        assert out_full.shape == (8, 768)
        assert out_8d.shape == (8, 768)
        assert out_4d.shape == (8, 768)

        # Outputs should differ for different active dims
        assert not torch.allclose(out_full, out_8d, atol=1e-5)
        assert not torch.allclose(out_8d, out_4d, atol=1e-5)

    def test_matryoshka_masking_zeros_tokens(self, perceiver, device):
        """Verify that tokens for z=0 dims are properly zeroed."""
        z = torch.randn(4, 16, device=device)
        z[:, 8:] = 0.0  # Mask last 8 dims

        # Access expander directly to check token zeroing
        tokens = perceiver.expander(z)

        # Tokens for masked dims should be zero (after pos_embed and masking)
        assert torch.allclose(tokens[:, 8:, :], torch.zeros_like(tokens[:, 8:, :]))
        # Tokens for active dims should be non-zero
        assert not torch.allclose(tokens[:, :8, :], torch.zeros_like(tokens[:, :8, :]))

    def test_single_sample(self, perceiver, device):
        """Test with batch size 1."""
        z = torch.randn(1, 16, device=device)
        out = perceiver(z)
        assert out.shape == (1, 768)

    def test_config_validation(self):
        """Test that invalid configs raise errors."""
        # hidden_size not divisible by num_heads
        with pytest.raises(ValueError, match="divisible by num_heads"):
            PerceiverDecoderConfig(hidden_size=100, num_heads=16)

        # Invalid dropout
        with pytest.raises(ValueError, match="dropout"):
            PerceiverDecoderConfig(dropout=1.5)

        # Invalid depth
        with pytest.raises(ValueError, match="depth"):
            PerceiverDecoderConfig(depth=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
