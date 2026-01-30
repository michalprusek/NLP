# Testing Patterns

**Analysis Date:** 2026-01-28

## Test Framework

**Runner:**
- pytest 9.0.2+ (listed in `pyproject.toml` dev dependencies)
- Config: `pyproject.toml` (no pytest.ini or pytest.toml found)

**Assertion Library:**
- PyTest's built-in assertions (assert statements)
- torch testing utilities: `torch.allclose()`, `torch.isnan()`

**Run Commands:**
```bash
pytest tests/test_ecoflow.py -v              # Run all tests verbose
pytest tests/test_ecoflow.py::TestClassName  # Run specific test class
pytest tests/test_ecoflow.py -k test_name    # Run by name pattern
pytest tests/test_ecoflow.py --tb=short      # Short traceback format
```

## Test File Organization

**Location:**
- Tests co-located with source: `tests/` directory at project root
- Test file: `tests/test_ecoflow.py`
- One test file per major module (EcoFlow-BO has comprehensive test suite)

**Naming:**
- Test file: `test_*.py` pattern
- Test classes: `Test<ComponentName>` (PascalCase)
- Test methods: `test_<functionality>` (snake_case)

**Structure:**
```
tests/
└── test_ecoflow.py          # EcoFlow-BO component tests
    ├── Fixtures             # Reusable test data
    ├── TestMatryoshkaEncoder
    ├── TestVelocityNetwork
    ├── TestRectifiedFlowDecoder
    ├── TestLosses
    ├── TestCoarseToFineGP
    ├── TestDensityAwareAcquisition
    ├── TestCycleConsistency
    ├── TestIntegration
    └── TestPerceiverDecoder
```

## Test Structure

**Suite Organization:**
```python
# Fixtures first (reusable test data)
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

# Then test classes, grouped by component
class TestMatryoshkaEncoder:
    def test_forward_shape(self, encoder, device):
        # Arrange
        x = torch.randn(32, 768, device=device)

        # Act
        z, mu, log_sigma = encoder(x)

        # Assert
        assert z.shape == (32, 8)
        assert mu.shape == (32, 8)
        assert log_sigma.shape == (32, 8)
```

**Patterns:**

1. **Arrange-Act-Assert:** Most tests follow AAA pattern (implicit in structure)
   ```python
   def test_deterministic_encode(self, encoder, device):
       x = torch.randn(32, 768, device=device)               # Arrange
       encoder.eval()
       z1 = encoder.encode_deterministic(x)
       z2 = encoder.encode_deterministic(x)                 # Act
       assert torch.allclose(z1, z2)                        # Assert
   ```

2. **Fixtures:** Heavy use of pytest fixtures for shared setup
   - Device fixture handles CUDA vs CPU: `device = "cuda" if torch.cuda.is_available() else "cpu"`
   - Config fixtures create fresh objects for each test
   - Nested fixture dependencies: `encoder(encoder_config, device)`

3. **Tear-down:** Implicit through Python garbage collection; no explicit cleanup in tests

## Mocking

**Framework:** No external mocking library detected (no `unittest.mock` imports)

**Patterns:**
- Tests use real objects with small dimensions for speed
- Configuration sizes reduced for test efficiency:
  ```python
  @pytest.fixture
  def velocity_config():
      return DiTVelocityNetConfig(
          data_dim=768,
          condition_dim=8,
          hidden_dim=128,  # Small for fast tests
          n_layers=2,
          n_heads=4,
      )
  ```

**What to Mock:**
- (Not practiced in this codebase - tests use real components)

**What NOT to Mock:**
- Model components (`encoder`, `decoder`, `velocity_net`) - test with real implementations
- Losses and metrics - compute real values for numerical correctness
- Device selection - test on available hardware

## Fixtures and Factories

**Test Data:**
```python
# Fixtures are the primary factory pattern
@pytest.fixture
def encoder(encoder_config, device):
    return MatryoshkaEncoder(encoder_config).to(device)

@pytest.fixture
def decoder(velocity_net):
    config = DecoderConfig(euler_steps=10)
    return RectifiedFlowDecoder(velocity_net, config)

# Usage in tests
def test_something(self, encoder, decoder, device):
    x = torch.randn(16, 768, device=device)
    # encoder and decoder are ready to use
```

**Location:**
- Fixtures defined at top of `tests/test_ecoflow.py`
- Module-level fixtures (before test classes)
- Class-level fixtures (defined within test class with `@pytest.fixture`)

**Factory Example:**
```python
@pytest.fixture
def perceiver_config(self):
    """Create Perceiver config for testing."""
    return PerceiverDecoderConfig(
        latent_dim=16,
        output_dim=768,
        hidden_size=256,  # Small for fast tests
        depth=2,
        num_heads=8,
        readout_heads=8,
    )
```

## Coverage

**Requirements:** Not enforced (no coverage.rc or pytest coverage settings)

**View Coverage:**
```bash
pytest tests/test_ecoflow.py --cov=ecoflow_bo --cov-report=html
# Then open htmlcov/index.html
```

## Test Types

**Unit Tests:**
- Scope: Individual component functions and methods
- Approach: Test shapes, determinism, and basic correctness
- Examples: `test_forward_shape()`, `test_deterministic_encode()`, `test_kl_loss()`
- Assertion style: `assert tensor.shape == expected_shape`, `assert loss > 0`, `assert torch.allclose(...)`

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Test encode-decode cycles, cycle consistency, loss computation with multiple modules
- Example from test suite:
  ```python
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
  ```

**E2E Tests:**
- Not detected in this codebase
- Run scripts (`run_opro.py`, `run_protegi.py`) serve as manual E2E validation

## Common Patterns

**Async Testing:**
- Not applicable (codebase is synchronous PyTorch)

**Error Testing:**
```python
def test_config_validation(self):
    """Test that invalid configs raise errors."""
    # hidden_size not divisible by num_heads
    with pytest.raises(ValueError, match="divisible by num_heads"):
        PerceiverDecoderConfig(hidden_size=100, num_heads=16)

    # Invalid dropout
    with pytest.raises(ValueError, match="dropout"):
        PerceiverDecoderConfig(dropout=1.5)
```

**Device-Agnostic Testing:**
```python
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# All tests accept device fixture and use it
def test_forward_shape(self, encoder, device):
    x = torch.randn(32, 768, device=device)  # data on correct device
    z, mu, log_sigma = encoder(x)
    assert z.device.type == device or (device == "cpu" and z.device.type in ["cpu"])
```

**Determinism Testing:**
```python
def test_decode_deterministic(self, decoder, device):
    z = torch.randn(8, 8, device=device)

    x1 = decoder.decode_deterministic(z, seed=42)
    x2 = decoder.decode_deterministic(z, seed=42)

    assert torch.allclose(x1, x2, atol=1e-5)  # Tight tolerance for determinism
```

**Numerical Accuracy Testing:**
```python
def test_kl_loss(self, device):
    kl_loss = KLDivergenceLoss()

    mu = torch.zeros(32, 8, device=device)
    log_sigma = torch.zeros(32, 8, device=device)

    # KL(N(0,1) || N(0,1)) = 0
    loss = kl_loss(mu, log_sigma)
    assert loss < 0.01  # Should be very close to 0

    # Non-zero mu should increase KL
    mu = torch.ones(32, 8, device=device)
    loss = kl_loss(mu, log_sigma)
    assert loss > 0.1
```

**Matrix Shape Testing:**
```python
def test_matryoshka_embeddings(self, encoder, device):
    x = torch.randn(32, 768, device=device)
    z_list, mu, log_sigma = encoder.get_matryoshka_embeddings(x)

    assert len(z_list) == 3
    assert z_list[0].shape == (32, 2)
    assert z_list[1].shape == (32, 4)
    assert z_list[2].shape == (32, 8)
```

## Test Execution

**Run single test file:**
```bash
pytest tests/test_ecoflow.py -v
```

**Run specific test class:**
```bash
pytest tests/test_ecoflow.py::TestMatryoshkaEncoder -v
```

**Run specific test method:**
```bash
pytest tests/test_ecoflow.py::TestMatryoshkaEncoder::test_forward_shape -v
```

**Run with short output:**
```bash
pytest tests/test_ecoflow.py --tb=short
```

**Run directly from test file:**
```bash
python tests/test_ecoflow.py
```
(Script includes `if __name__ == "__main__": pytest.main([__file__, "-v"])`)

---

*Testing analysis: 2026-01-28*
