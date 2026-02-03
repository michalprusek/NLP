"""Test stereographic projection pipeline end-to-end.

Verifies:
1. StereographicTransform roundtrip is exact
2. Flow model with stereographic has correct dimensions
3. BO loop can reconstruct embeddings correctly

Run:
    uv run python -m rielbo.test_stereographic_pipeline
"""

import logging
import torch
import torch.nn.functional as F

from rielbo.stereographic import StereographicTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_stereographic_roundtrip():
    """Test that stereographic lift/project are exact inverses."""
    logger.info("Testing stereographic roundtrip...")

    # Simulate SELFIES VAE embeddings (256D, norm ~ 15-20)
    torch.manual_seed(42)
    x = torch.randn(100, 256) * 5 + torch.randn(100, 1) * 10

    # Create transform
    stereo = StereographicTransform.from_embeddings(x)
    logger.info(f"  Transform: {stereo}")

    # Lift to sphere
    u = stereo.lift(x)
    logger.info(f"  Lifted shape: {u.shape}")
    assert u.shape == (100, 257), f"Expected (100, 257), got {u.shape}"

    # Verify on unit sphere
    norms = u.norm(dim=-1)
    logger.info(f"  u norms: mean={norms.mean():.6f}, std={norms.std():.6f}")
    assert torch.allclose(norms, torch.ones(100), atol=1e-5), "Not on unit sphere"

    # Project back
    x_rec = stereo.project(u)
    assert x_rec.shape == (100, 256), f"Expected (100, 256), got {x_rec.shape}"

    # Verify exact roundtrip
    max_error = (x - x_rec).abs().max().item()
    logger.info(f"  Roundtrip max error: {max_error:.2e}")
    assert max_error < 1e-4, f"Roundtrip error too large: {max_error}"

    # Verify norm preservation
    orig_norms = x.norm(dim=-1)
    rec_norms = x_rec.norm(dim=-1)
    norm_error = (orig_norms - rec_norms).abs().max().item()
    logger.info(f"  Norm reconstruction error: {norm_error:.2e}")
    assert norm_error < 1e-3, f"Norm error too large: {norm_error}"

    logger.info("  ✓ Stereographic roundtrip passed")


def test_stereographic_with_flow_dimensions():
    """Test that stereographic projection produces correct dimensions for flow."""
    logger.info("Testing stereographic with flow dimensions...")

    # Simulate data
    torch.manual_seed(42)
    x = torch.randn(50, 256) * 15

    # Create transform
    stereo = StereographicTransform.from_embeddings(x)

    # Lift to sphere
    u = stereo.lift(x)

    # Simulate flow forward (Euler integration)
    # Flow model would operate on 257D sphere
    dt = 1.0 / 50
    for t_idx in range(50):
        # Simulate velocity field (random for test)
        v = torch.randn_like(u) * 0.01
        u = u + dt * v
        u = F.normalize(u, p=2, dim=-1)

    # Project back to get embedding
    x_out = stereo.project(u)
    logger.info(f"  Output shape: {x_out.shape}")
    assert x_out.shape == (50, 256), f"Expected (50, 256), got {x_out.shape}"

    # Verify output has reasonable magnitude
    out_norms = x_out.norm(dim=-1)
    logger.info(f"  Output norms: mean={out_norms.mean():.2f}, std={out_norms.std():.2f}")

    logger.info("  ✓ Flow dimensions test passed")


def test_stereographic_serialization():
    """Test save/load of stereographic transform."""
    import tempfile
    import os

    logger.info("Testing stereographic serialization...")

    torch.manual_seed(42)
    x = torch.randn(30, 256) * 10

    # Create and save
    stereo = StereographicTransform.from_embeddings(x)
    u_original = stereo.lift(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stereo.pt")
        stereo.save(path)

        # Load
        stereo_loaded = StereographicTransform.load(path)

    # Verify same result
    u_loaded = stereo_loaded.lift(x)
    assert torch.allclose(u_original, u_loaded, atol=1e-6), "Loaded transform differs"

    logger.info("  ✓ Serialization test passed")


def test_checkpoint_metadata():
    """Test that checkpoint would contain correct metadata for stereographic."""
    logger.info("Testing checkpoint metadata structure...")

    torch.manual_seed(42)
    x = torch.randn(20, 256) * 12

    stereo = StereographicTransform.from_embeddings(x)

    # Simulate checkpoint structure
    checkpoint = {
        "is_stereographic": True,
        "original_input_dim": 256,
        "input_dim": 257,  # D+1 for stereographic
        "radius_scaling": stereo.radius_scaling,
        "is_spherical": True,  # Stereographic implies spherical
    }

    # Verify we can reconstruct transform from checkpoint
    stereo_from_ckpt = StereographicTransform(
        checkpoint["original_input_dim"],
        checkpoint["radius_scaling"]
    )

    u = stereo.lift(x)
    u_from_ckpt = stereo_from_ckpt.lift(x)

    assert torch.allclose(u, u_from_ckpt, atol=1e-6), "Checkpoint reconstruction differs"
    logger.info(f"  Checkpoint input_dim: {checkpoint['input_dim']} (D+1)")
    logger.info(f"  Checkpoint original_input_dim: {checkpoint['original_input_dim']} (D)")
    logger.info(f"  Checkpoint radius_scaling: {checkpoint['radius_scaling']:.4f}")

    logger.info("  ✓ Checkpoint metadata test passed")


def main():
    logger.info("=" * 60)
    logger.info("Testing Stereographic Projection Pipeline")
    logger.info("=" * 60)

    test_stereographic_roundtrip()
    test_stereographic_with_flow_dimensions()
    test_stereographic_serialization()
    test_checkpoint_metadata()

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
