# Testing Patterns

**Analysis Date:** 2026-01-31

## Test Framework

**Runner:**
- pytest 9.0.2+
- Config: `pyproject.toml` with `[dependency-groups] dev = ["pytest>=9.0.2"]`
- Located in: `/home/prusek/NLP/tests/`

**Assertion Library:**
- Built-in pytest assertions
- Custom torch assertions: `torch.isclose()`, `torch.allclose(atol=)`, `torch.isnan()`, `torch.isinf()`

**Run Commands:**
```bash
uv run pytest tests/ -x -q              # Run all tests, stop on first failure
uv run pytest tests/test_gp_surrogate.py # Run specific test file
uv run pytest tests/ -v                  # Verbose output with test names
uv run pytest tests/ --tb=short          # Short traceback format
```

## Test File Organization

**Location:**
- Separate from implementation: `/home/prusek/NLP/tests/` directory
- Test discovery: pytest convention `test_*.py` files

**Naming:**
- Test files: `test_<module_name>.py` (e.g., `test_gp_surrogate.py`, `test_batch_selection.py`)
- Test classes: `Test<Feature>` (e.g., `TestBinomialVariance`, `TestGPFitting`, `TestBatchDiversity`)
- Test methods: `test_<specific_behavior>` (e.g., `test_variance_at_p_half`, `test_fit_basic`)

**Structure:**
```
tests/
├── conftest.py                  # Pytest configuration and shared fixtures
├── test_batch_selection.py      # Tests for batch selection (BO candidate selection)
├── test_gp_surrogate.py         # Tests for GP surrogates (heteroscedastic & MSR)
├── test_nfbo_model.py           # Tests for RealNVP flow model
├── test_nfbo_sampler.py         # Tests for NFBoSampler
└── test_nfbo_sampler_refined.py # Additional sampler tests
```

## Test Structure

**Setup and Configuration (`conftest.py`):**
```python
"""Pytest configuration for ecoflow tests."""

import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

**Fixture patterns (`test_batch_selection.py` example):**
```python
@pytest.fixture
def device():
    """Get CUDA device."""
    return torch.device("cuda")

@pytest.fixture
def fitted_gp(device):
    """Create and fit a GP surrogate for testing."""
    from ecoflow.gp_surrogate import create_surrogate

    gp = create_surrogate("msr", D=1024, device=device)
    X = torch.randn(20, 1024, device=device)
    Y = torch.rand(20, device=device)
    gp.fit(X, Y)
    return gp
```

**Conditional test skipping:**
```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
```

**Test class organization:**
```python
class TestBinomialVariance:
    """Test variance formula: Var(p) = p(1-p)/n"""

    def test_variance_at_p_half(self):
        """Variance should be maximal at p=0.5."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.5], device=DEVICE)
        var = gp._compute_variance(Y)
        # Var(0.5) = 0.5 * 0.5 / 100 = 0.0025
        assert torch.isclose(var[0], torch.tensor(0.0025, device=DEVICE), rtol=1e-4)

    def test_variance_at_p_nine(self):
        """Variance should be lower at p=0.9."""
        gp = HeteroscedasticSonarGP(n_eval=100, device=DEVICE)
        Y = torch.tensor([0.9], device=DEVICE)
        var = gp._compute_variance(Y)
        # Var(0.9) = 0.9 * 0.1 / 100 = 0.0009
        assert torch.isclose(var[0], torch.tensor(0.0009, device=DEVICE), rtol=1e-4)
```

## Mocking

**Framework:** None explicitly; tests use fixtures for data/models instead

**No external mocking libraries detected** - tests create real objects:
- Real GP surrogates: `create_surrogate("msr", D=1024, device=device)`
- Real tensors: `torch.randn(20, 1024, device=device)`
- Real models: `RealNVP(dim=dim, n_layers=2, hidden_dim=8)`

**Patterns:**
- Fixtures provide pre-configured objects instead of mocking
- Device selection handled dynamically: `device = torch.device("cuda") if torch.cuda.is_available() else "cpu"`

**What to Mock:**
- External API calls (not tested in current suite)
- Database queries (not present in codebase)

**What NOT to Mock:**
- Core model computations (should be real)
- Tensor operations (should use real torch)
- Data flow through optimization loops

## Fixtures and Factories

**Test Data:**
```python
# Synthetic tensors created on-demand
X = torch.randn(20, 1024, device=device)  # 20 random 1024D points
Y = torch.rand(20, device=device)          # 20 random scores in [0, 1]

# Fitted surrogates as fixtures
@pytest.fixture
def fitted_gp(device):
    gp = create_surrogate("msr", D=1024, device=device)
    X = torch.randn(20, 1024, device=device)
    Y = torch.rand(20, device=device)
    gp.fit(X, Y)
    return gp
```

**Location:**
- Local fixtures in `conftest.py` (project-wide)
- Test-specific fixtures defined in test file (class-local)
- Device fixture (`test_batch_selection.py`, `test_gp_surrogate.py`):
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**Factory patterns:**
```python
def create_surrogate(method: str, D: int, device: str) -> BaseGPSurrogate:
    """Factory function returning appropriate GP surrogate."""
    # Creates either SonarGPSurrogate, BAxUSGPSurrogate, or HeteroscedasticSonarGP
```

## Coverage

**Requirements:** Not enforced (no coverage configuration in pyproject.toml)

**View Coverage:**
```bash
# Install coverage if needed
pip install pytest-cov

# Run with coverage
uv run pytest tests/ --cov=opro --cov=protegi --cov=gepa --cov=ecoflow --cov=nfbo --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Scope: Single functions/methods in isolation
- Approach: Mathematical correctness, edge cases, invariants
- Examples: `test_variance_at_p_half()`, `test_variance_monotonic_from_half()`, `test_log_prob()`
- Location: `test_gp_surrogate.py` (variance formulas), `test_nfbo_model.py` (RealNVP operations)

**Integration Tests:**
- Scope: Multiple components working together
- Approach: End-to-end workflow testing
- Examples: `test_fit_basic()` + `test_predict_after_fit()`, `test_gradient_computable()` with fitting
- Location: `TestGPFitting` class in `test_gp_surrogate.py`

**E2E Tests:**
- Framework: Not found (would be in CI/CD if present)
- Full optimization runs tested manually before production

## Common Patterns

**Assertion patterns:**
```python
# Tensor shape verification
assert z.shape == x.shape
assert log_det.shape == (10,)

# Numerical correctness with tolerance
assert torch.isclose(var[0], torch.tensor(0.0025, device=DEVICE), rtol=1e-4)
assert torch.allclose(x, x_recon, atol=1e-5)

# NaN/Inf checking
assert not torch.isnan(var[0])
assert not torch.isinf(var[0])
assert not torch.any(torch.isnan(mean))

# Logical assertions
assert lp_min_dist > 0, "LP should not select duplicate points"
assert len(torch.unique(indices)) == batch_size, "Indices should be unique"
```

**Async Testing:**
Not used - all operations are synchronous

**Error Testing:**
```python
# From test_gp_surrogate.py - error cases with try/except
def test_fit_basic(self):
    """GP should fit without error."""
    gp = HeteroscedasticSonarGP(D=64, n_eval=100, device=DEVICE)
    X = torch.randn(10, 64, device=DEVICE)
    Y = torch.rand(10, device=DEVICE)
    gp.fit(X, Y)  # Should not raise
    assert gp.model is not None
```

**Data validation in tests:**
```python
def test_no_nan_in_batch(self):
    """No NaN values for a batch with extreme values."""
    gp = HeteroscedasticSonarGP(n_eval=150, device=DEVICE)
    Y = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=DEVICE)
    var = gp._compute_variance(Y)
    assert not torch.any(torch.isnan(var))  # Critical for numerical stability
    assert not torch.any(torch.isinf(var))
```

## Test Classes and Organization

**TestBinomialVariance** (`test_gp_surrogate.py`):
- Tests variance formula: Var(p) = p(1-p)/n
- Methods: `test_variance_at_p_half()`, `test_variance_monotonic_from_half()`, `test_variance_scales_with_n_eval()`

**TestNumericalStability** (`test_gp_surrogate.py`):
- Tests extreme values don't cause NaN/Inf
- Methods: `test_variance_at_zero()`, `test_variance_at_one()`, `test_variance_floor_applied()`, `test_no_nan_in_batch()`

**TestGPFitting** (`test_gp_surrogate.py`):
- Tests GP fitting workflow and predictions
- Methods: `test_fit_basic()`, `test_predict_after_fit()`, `test_update_incremental()`, `test_gradient_computable()`, `test_uncertainty_reasonable()`

**TestCreateSurrogateFactory** (`test_gp_surrogate.py`):
- Tests factory function for creating different surrogates
- Methods: `test_heteroscedastic_method()`, `test_msr_method()`, `test_case_insensitive()`, `test_invalid_method_raises()`

**TestBatchDiversity** (`test_batch_selection.py`):
- Tests Local Penalization produces diverse batch selections
- Methods: `test_lp_more_diverse_than_greedy()`, `test_batch_points_are_distinct()`

**TestSelectBatchCandidates** (`test_batch_selection.py`):
- Tests batch candidate selection with different methods
- Methods: vary by configuration and selection method

**TestEdgeCases** (`test_batch_selection.py`):
- Tests boundary conditions and error handling
- Single candidate, empty batches, etc.

## Running Tests Locally

**Full suite:**
```bash
cd /home/prusek/NLP
uv run pytest tests/ -x -q
```

**Specific test class:**
```bash
uv run pytest tests/test_gp_surrogate.py::TestBinomialVariance -v
```

**Specific test method:**
```bash
uv run pytest tests/test_gp_surrogate.py::TestBinomialVariance::test_variance_at_p_half -v
```

**With output:**
```bash
uv run pytest tests/ -s  # Show print statements
```

## Known Test Dependencies

- PyTorch with CUDA (tests skip if CUDA unavailable)
- `torch.cuda.is_available()` checked in conftest and individual tests
- GPyTorch and BoTorch for surrogate models
- No external data files required (synthetic data generated in tests)

---

*Testing analysis: 2026-01-31*
