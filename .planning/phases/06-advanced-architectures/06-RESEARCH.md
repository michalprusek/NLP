# Phase 6: Advanced Architectures - Research

**Researched:** 2026-02-01
**Domain:** U-Net MLP with FiLM conditioning, Mamba/SSM velocity networks, and architecture scaling variants for flow matching
**Confidence:** MEDIUM

## Summary

This phase implements three advanced architecture variants for the flow matching study: (1) U-Net MLP with FiLM (Feature-wise Linear Modulation) conditioning for skip-connection-based velocity prediction, (2) Mamba/SSM velocity network as an experimental alternative to transformers, and (3) Tiny/Small/Base scaling variants for dataset size ablation studies.

The key challenge is that standard U-Net and Mamba architectures are designed for sequential or image data, not 1D embeddings. For U-Net MLP, the "encoder-decoder" structure becomes a series of MLPs that downsample/upsample the embedding dimension with skip connections, using FiLM layers to modulate based on timestep. For Mamba, the primary challenge is adapting the selective state-space mechanism to non-sequential embedding-to-embedding mapping - research indicates treating the embedding dimensions as a "virtual sequence" is one approach, though this is experimental.

Architecture scaling follows the DiT pattern: Tiny (~500K params), Small (~2.5M params), and Base (~10M params) variants for each architecture type. This allows measuring how model capacity interacts with dataset size.

**Primary recommendation:** Implement U-Net MLP with FiLM as the main new architecture (~2.5M params), use bidirectional Mamba similar to Point Mamba papers for experimental SSM velocity network, and create Tiny/Small/Base variants by scaling hidden_dim and num_layers following DiT-S/B/L ratios.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.nn | 2.8+ | Neural network modules | Core PyTorch, existing infrastructure |
| mamba-ssm | 2.3.0+ | Mamba SSM blocks | Official state-spaces/mamba library |
| causal-conv1d | (optional) | Fast 1D convolution for Mamba | Performance optimization for Mamba |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| einops | 0.7+ | Tensor reshape operations | Optional, clearer dimension handling |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| mamba-ssm | Mamba2 (from same lib) | Mamba2 has MIMO but more complex; start with Mamba1 |
| Custom FiLM | timm FiLM | timm adds dependency; FiLM is simple enough to implement |
| Bidirectional Mamba | Unidirectional | Unidirectional ignores half the context; bidirectional better for non-causal task |

**Installation:**
```bash
# Add mamba-ssm for experimental Mamba velocity network
pip install mamba-ssm
# Optional: for faster convolutions
pip install causal-conv1d
```

**Note:** mamba-ssm requires Linux, NVIDIA GPU, PyTorch 1.12+, and CUDA 11.6+. Installation may fail on other platforms. If installation fails, the Mamba architecture should be marked as blocked.

## Architecture Patterns

### Recommended Project Structure
```
study/flow_matching/models/
├── __init__.py          # Extended create_model factory
├── mlp.py               # SimpleMLP (existing)
├── dit.py               # DiTVelocityNetwork (existing)
├── unet_mlp.py          # NEW: U-Net MLP with FiLM
├── mamba_velocity.py    # NEW: Mamba SSM velocity network (experimental)
└── scaling.py           # NEW: Architecture scaling utilities
```

### Pattern 1: FiLM Layer for Time Conditioning
**What:** Feature-wise affine transformation: out = gamma * x + beta, where gamma/beta are predicted from timestep
**When to use:** Alternative to AdaLN for conditioning; good for encoder-decoder architectures
**Example:**
```python
# Source: https://arxiv.org/abs/1709.07871 (FiLM paper)
# Source: https://github.com/ethanjperez/film
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation: out = gamma * x + beta
    where gamma and beta are predicted from conditioning (timestep).
    """

    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        # Predict gamma (scale) and beta (shift) from conditioning
        self.film_params = nn.Linear(conditioning_dim, 2 * feature_dim)
        # Initialize to identity transformation (gamma=1, beta=0)
        nn.init.zeros_(self.film_params.weight)
        nn.init.zeros_(self.film_params.bias)
        self.film_params.bias.data[:feature_dim] = 1.0  # gamma = 1

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Feature tensor [B, D]
            cond: Conditioning tensor [B, cond_dim] (e.g., time embedding)

        Returns:
            Modulated features [B, D]
        """
        params = self.film_params(cond)  # [B, 2*D]
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * x + beta
```

### Pattern 2: U-Net MLP with FiLM (1D Encoder-Decoder)
**What:** MLP-based encoder-decoder with skip connections and FiLM time conditioning
**When to use:** When skip connections might help preserve embedding structure across depth
**Example:**
```python
# Source: Adapted from U-Net architecture + FiLM paper
class UNetMLPBlock(nn.Module):
    """Single U-Net MLP block with FiLM conditioning."""

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.film = FiLMLayer(out_dim, cond_dim)
        self.act = nn.SiLU()

        # Residual if dimensions match
        self.residual = in_dim == out_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear1(x))
        h = self.film(h, cond)
        h = self.act(self.linear2(h))
        if self.residual:
            h = h + x
        return h

class UNetMLP(nn.Module):
    """U-Net MLP velocity network with FiLM time conditioning.

    Architecture (~2.5M params with default config):
    - Encoder: 1024 -> 512 -> 256 (bottleneck)
    - Decoder: 256 -> 512 -> 1024 with skip connections
    - FiLM modulation at each block
    - Output layer zero-initialized
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: tuple = (512, 256),  # Encoder dimensions
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Encoder blocks
        encoder_dims = [input_dim] + list(hidden_dims)
        self.encoder = nn.ModuleList([
            UNetMLPBlock(encoder_dims[i], encoder_dims[i+1], time_embed_dim)
            for i in range(len(hidden_dims))
        ])

        # Decoder blocks (reverse, with skip connection dimensions)
        decoder_dims = list(reversed(hidden_dims)) + [input_dim]
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims)):
            # Skip connection doubles input dimension (except bottleneck)
            in_dim = decoder_dims[i] if i == 0 else decoder_dims[i] * 2
            out_dim = decoder_dims[i+1]
            self.decoder.append(UNetMLPBlock(in_dim, out_dim, time_embed_dim))

        # Output projection (zero-init)
        self.output_proj = nn.Linear(input_dim * 2, input_dim)  # +skip from input
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity.

        Args:
            x: Input [B, 1024]
            t: Time [B] in [0, 1]

        Returns:
            Velocity [B, 1024]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # Time embedding
        t_emb = timestep_embedding(t, self.time_embed_dim)
        cond = self.time_mlp(t_emb)

        # Encoder with skip storage
        skips = [x]  # Store input as first skip
        h = x
        for block in self.encoder:
            h = block(h, cond)
            skips.append(h)

        # Decoder with skip connections
        skips = skips[:-1]  # Don't skip bottleneck to itself
        for i, block in enumerate(self.decoder):
            if i > 0:  # Skip bottleneck
                skip = skips.pop()
                h = torch.cat([h, skip], dim=-1)
            h = block(h, cond)

        # Final output with input skip
        h = torch.cat([h, skips.pop()], dim=-1)  # Concat with original input
        return self.output_proj(h)
```

### Pattern 3: Mamba Velocity Network (Experimental)
**What:** Selective SSM for velocity prediction, treating embedding dimensions as virtual sequence
**When to use:** Experimental architecture exploration; may or may not work for non-sequential embeddings
**Example:**
```python
# Source: https://github.com/state-spaces/mamba + Mamba3D/PointMamba papers
# EXPERIMENTAL: Mamba is designed for sequences, adapting to embeddings is non-trivial
from mamba_ssm import Mamba

class MambaVelocityNetwork(nn.Module):
    """Experimental Mamba-based velocity network.

    Key adaptations for embedding-to-embedding (non-sequential):
    1. Treat 1024-dim embedding as "sequence" of chunks
    2. Use bidirectional processing (not causal)
    3. Aggregate with linear projection

    WARNING: This is experimental. Mamba is designed for sequences,
    not for unordered embedding transformations.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        time_embed_dim: int = 256,
        chunk_size: int = 64,  # Treat embedding as 16 chunks of 64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.n_chunks = input_dim // chunk_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project chunks to hidden dim
        self.input_proj = nn.Linear(chunk_size, hidden_dim)

        # Mamba layers (bidirectional: forward + backward)
        self.mamba_fwd = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.mamba_bwd = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        # Time conditioning (additive after each layer)
        self.time_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, chunk_size)  # *2 for bidirectional
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity.

        Args:
            x: Input [B, 1024]
            t: Time [B] in [0, 1]

        Returns:
            Velocity [B, 1024]
        """
        B = x.shape[0]
        if t.dim() == 2:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)

        # Time embedding
        t_emb = timestep_embedding(t, 256)  # TODO: make configurable
        cond = self.time_mlp(t_emb)  # [B, hidden_dim]

        # Reshape embedding to sequence of chunks
        # [B, 1024] -> [B, 16, 64]
        x = x.view(B, self.n_chunks, self.chunk_size)

        # Project to hidden dim
        h = self.input_proj(x)  # [B, 16, hidden_dim]

        # Bidirectional Mamba
        h_fwd = h
        h_bwd = h.flip(dims=[1])  # Reverse sequence

        for i, (mamba_f, mamba_b, time_proj) in enumerate(
            zip(self.mamba_fwd, self.mamba_bwd, self.time_projs)
        ):
            # Forward direction
            h_fwd = mamba_f(h_fwd)
            # Backward direction (on reversed sequence)
            h_bwd = mamba_b(h_bwd)

            # Add time conditioning
            t_cond = time_proj(cond).unsqueeze(1)  # [B, 1, hidden_dim]
            h_fwd = h_fwd + t_cond
            h_bwd = h_bwd + t_cond

        # Reverse backward output and concatenate
        h_bwd = h_bwd.flip(dims=[1])
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 16, hidden_dim*2]

        # Project back to chunk size
        v = self.output_proj(h)  # [B, 16, 64]

        # Reshape to original dimension
        v = v.view(B, self.input_dim)  # [B, 1024]

        return v
```

### Pattern 4: Architecture Scaling Variants
**What:** Define Tiny/Small/Base configs following DiT scaling patterns
**When to use:** For ablation studies with different dataset sizes
**Example:**
```python
# Source: DiT paper scaling patterns (S/B/L/XL)
# Adapted for velocity networks targeting ~500K, ~2.5M, ~10M params

SCALING_CONFIGS = {
    # MLP scaling: adjust hidden_dim and num_layers
    "mlp_tiny": {"hidden_dim": 128, "num_layers": 4},    # ~250K params
    "mlp_small": {"hidden_dim": 256, "num_layers": 5},   # ~1M params (current default)
    "mlp_base": {"hidden_dim": 384, "num_layers": 6},    # ~2.2M params

    # DiT scaling: adjust hidden_dim and num_layers
    "dit_tiny": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4},   # ~3M params
    "dit_small": {"hidden_dim": 384, "num_layers": 3, "num_heads": 6},  # ~9M params (current)
    "dit_base": {"hidden_dim": 512, "num_layers": 4, "num_heads": 8},   # ~20M params

    # U-Net MLP scaling: adjust hidden_dims tuple
    "unet_tiny": {"hidden_dims": (256, 128)},            # ~600K params
    "unet_small": {"hidden_dims": (512, 256)},           # ~2.5M params (target)
    "unet_base": {"hidden_dims": (768, 384)},            # ~5.5M params

    # Mamba scaling (experimental): adjust hidden_dim and num_layers
    "mamba_tiny": {"hidden_dim": 128, "num_layers": 2},  # ~500K params
    "mamba_small": {"hidden_dim": 256, "num_layers": 4}, # ~2M params
    "mamba_base": {"hidden_dim": 384, "num_layers": 6},  # ~5M params
}

def get_scaled_config(arch: str, scale: str) -> dict:
    """Get architecture config for given scale.

    Args:
        arch: Architecture type ('mlp', 'dit', 'unet', 'mamba')
        scale: Scale variant ('tiny', 'small', 'base')

    Returns:
        Config dict for create_model()
    """
    key = f"{arch}_{scale}"
    if key not in SCALING_CONFIGS:
        raise ValueError(f"Unknown config: {key}. Available: {list(SCALING_CONFIGS.keys())}")
    return SCALING_CONFIGS[key]
```

### Anti-Patterns to Avoid

- **Treating Mamba as drop-in replacement for Transformer:** Mamba is designed for sequential data with causal structure; naively applying to embeddings may produce poor results
- **Using unidirectional Mamba for non-causal tasks:** Flow matching is not causal; use bidirectional processing
- **Forgetting FiLM initialization:** FiLM should start as identity (gamma=1, beta=0) to avoid training instability
- **Complex U-Net topologies:** Keep U-Net simple for 1D; 2-3 encoder/decoder levels are sufficient
- **Ignoring Mamba CUDA requirements:** mamba-ssm requires NVIDIA GPU with CUDA; have fallback for CPU/non-NVIDIA systems

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Selective SSM | Custom recurrence | mamba-ssm.Mamba | Hardware-optimized CUDA kernels |
| FiLM layer | Complex conditioning | Simple gamma*x + beta | FiLM is intentionally simple; complexity is in conditioning network |
| Bidirectional Mamba | Complex fusion | Process forward + backward, concatenate | Simple and effective; used in Point Mamba |
| Parameter counting for scaling | Manual calculation | `sum(p.numel() for p in model.parameters())` | Always verify actual counts |

**Key insight:** U-Net MLP with FiLM is the most likely to work well since it builds on proven patterns (skip connections, FiLM modulation). Mamba is truly experimental for this use case.

## Common Pitfalls

### Pitfall 1: Mamba Installation Failures
**What goes wrong:** `pip install mamba-ssm` fails with CUDA version mismatch or build errors
**Why it happens:** mamba-ssm requires specific CUDA toolkit version, Linux, and NVIDIA GPU
**How to avoid:**
- Ensure CUDA version matches PyTorch CUDA version
- Use `--no-build-isolation` if pip complains
- Have fallback: if mamba-ssm unavailable, skip Mamba experiments
**Warning signs:** ImportError, nvcc compilation errors, CUDA version mismatch warnings

### Pitfall 2: U-Net Skip Connection Dimension Mismatch
**What goes wrong:** RuntimeError about tensor dimension mismatch during skip connection concat
**Why it happens:** Encoder and decoder block dimensions don't align for concatenation
**How to avoid:** Track encoder output dimensions carefully; decoder input = decoder_output + skip_output
**Warning signs:** Shape mismatch in torch.cat()

### Pitfall 3: FiLM Starting with Random Modulation
**What goes wrong:** Training is unstable, loss oscillates wildly in first epochs
**Why it happens:** FiLM gamma/beta start with random values, disrupting feature magnitudes
**How to avoid:** Initialize FiLM projection to output gamma=1, beta=0 (identity transform)
**Warning signs:** Large loss variance, NaN in first few batches

### Pitfall 4: Mamba Chunk Size Not Dividing Input Dimension
**What goes wrong:** ValueError when reshaping 1024 -> (n_chunks, chunk_size)
**Why it happens:** 1024 must be divisible by chunk_size for reshaping
**How to avoid:** Use chunk_size in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
**Warning signs:** Reshape errors in forward pass

### Pitfall 5: Mamba Not Learning (Experimental Architecture)
**What goes wrong:** Mamba velocity network loss plateaus at high value, doesn't converge
**Why it happens:** Mamba is designed for sequences; embedding-to-embedding may not be suitable
**How to avoid:**
- This is expected for experimental architecture
- Document failure as valid research result
- Fall back to U-Net MLP if Mamba doesn't work
**Warning signs:** Loss not decreasing after many epochs, validation loss higher than I-CFM

### Pitfall 6: Scaling Variants with Wrong Parameter Counts
**What goes wrong:** "Base" variant has fewer params than "Small", or counts are way off target
**Why it happens:** Parameter count depends on hidden_dim^2, not linear
**How to avoid:** Always verify actual parameter count after creating model; adjust config if needed
**Warning signs:** Model summary shows unexpected parameter count

## Code Examples

### Complete U-Net MLP Implementation
```python
# Source: Combining U-Net architecture + FiLM paper patterns
# See Pattern 2 above for full implementation
```

### Extended Model Factory
```python
# Source: Extension of existing study/flow_matching/models/__init__.py
def create_model(arch: str, scale: str = "small", **kwargs) -> nn.Module:
    """Create velocity network by architecture name and scale.

    Args:
        arch: Architecture name ('mlp', 'dit', 'unet', 'mamba')
        scale: Scale variant ('tiny', 'small', 'base')
        **kwargs: Override any config parameter

    Returns:
        Velocity network module
    """
    # Get base config for scale
    config = get_scaled_config(arch, scale)
    config.update(kwargs)  # Allow overrides

    if arch == "mlp":
        return SimpleMLP(input_dim=1024, **config)
    elif arch == "dit":
        return DiTVelocityNetwork(input_dim=1024, **config)
    elif arch == "unet":
        return UNetMLP(input_dim=1024, **config)
    elif arch == "mamba":
        try:
            return MambaVelocityNetwork(input_dim=1024, **config)
        except ImportError:
            raise ImportError(
                "mamba-ssm not installed. Install with: pip install mamba-ssm"
            )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

# Log parameter counts after creation
model = create_model("unet", "small")
params = sum(p.numel() for p in model.parameters())
print(f"UNet-MLP (small): {params:,} params")  # Should be ~2.5M
```

### Verification Tests
```python
# Source: Standard testing patterns
def test_unet_mlp():
    """Verify U-Net MLP architecture."""
    from study.flow_matching.models import UNetMLP

    model = UNetMLP(input_dim=1024, hidden_dims=(512, 256))
    params = sum(p.numel() for p in model.parameters())
    print(f"UNetMLP: {params:,} params")

    # Verify forward pass
    x = torch.randn(4, 1024)
    t = torch.rand(4)
    v = model(x, t)

    assert v.shape == (4, 1024), f"Output shape wrong: {v.shape}"
    assert not torch.isnan(v).any(), "NaN in output"
    print("U-Net MLP test passed!")

def test_mamba_velocity():
    """Test Mamba velocity network if available."""
    try:
        from study.flow_matching.models import MambaVelocityNetwork
    except ImportError:
        print("Mamba not available, skipping test")
        return

    model = MambaVelocityNetwork(input_dim=1024, hidden_dim=256, num_layers=4).cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"MambaVelocityNetwork: {params:,} params")

    # Verify forward pass
    x = torch.randn(4, 1024, device="cuda")
    t = torch.rand(4, device="cuda")
    v = model(x, t)

    assert v.shape == (4, 1024), f"Output shape wrong: {v.shape}"
    assert not torch.isnan(v).any(), "NaN in output"
    print("Mamba velocity test passed!")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Concatenation conditioning | FiLM/AdaLN conditioning | 2018+ | Better feature modulation, less parameter overhead |
| Unidirectional Mamba | Bidirectional Mamba | Mamba3D/PointMamba 2024 | Essential for non-causal tasks |
| Fixed architecture | Scale variants (S/B/L/XL) | DiT 2022 | Enables dataset size ablations |
| 2D U-Net only | 1D MLP U-Net | Various | Applies U-Net principles to embeddings |

**Deprecated/outdated:**
- Concatenation-based time conditioning: FiLM/AdaLN is more effective
- Pure Transformer for everything: Mamba/SSM offers linear complexity alternative (experimental)

## Open Questions

1. **Will U-Net skip connections help for 1024D embeddings?**
   - What we know: U-Net skips help preserve spatial/local structure in images
   - What's unclear: Whether "local structure" exists in SONAR embedding dimensions
   - Recommendation: Implement and ablate; compare to MLP baseline

2. **Is Mamba suitable for non-sequential embedding transformation?**
   - What we know: Mamba excels at sequences; Point Mamba adapts for point clouds
   - What's unclear: Whether chunked embedding is a valid "sequence" for SSM
   - Recommendation: Mark as experimental; expect possible failure; document results either way

3. **Optimal scaling ratios for different dataset sizes**
   - What we know: DiT uses S/B/L/XL with 2-3x param jumps
   - What's unclear: How dataset size correlates with optimal model size for flow matching
   - Recommendation: Start with 3 variants (Tiny/Small/Base); extend if needed

4. **FiLM vs AdaLN for U-Net conditioning**
   - What we know: Both work for time conditioning; AdaLN is standard for DiT
   - What's unclear: Which is better for encoder-decoder architecture
   - Recommendation: Use FiLM for U-Net (simpler, matches literature); could ablate later

## Sources

### Primary (HIGH confidence)
- [FiLM Paper (arXiv:1709.07871)](https://arxiv.org/abs/1709.07871) - Feature-wise Linear Modulation
- [FiLM Implementation (GitHub)](https://github.com/ethanjperez/film) - Official PyTorch code
- [DiT Paper](https://arxiv.org/abs/2212.09748) - Architecture scaling patterns (S/B/L/XL)
- [mamba-ssm (GitHub)](https://github.com/state-spaces/mamba) - Official Mamba implementation
- [mamba-ssm (PyPI)](https://pypi.org/project/mamba-ssm/) - Latest version 2.3.0
- Existing codebase: `study/flow_matching/models/` - Interface patterns to follow

### Secondary (MEDIUM confidence)
- [Mamba3D (arXiv:2404.14966)](https://arxiv.org/abs/2404.14966) - Bidirectional SSM for point clouds
- [StruMamba3D (arXiv:2506.21541)](https://arxiv.org/abs/2506.21541) - Structural Mamba for 3D
- [PointMamba patterns](https://github.com/Yangzhangcst/Mamba-in-CV) - Mamba adaptations for non-sequence data
- [Residual MLP patterns](https://www.shadecoder.com/topics/residual-mlp-a-comprehensive-guide-for-2025) - Skip connection best practices

### Tertiary (LOW confidence)
- Mamba for embedding-to-embedding: No direct precedent found; experimental approach derived from point cloud papers
- U-Net MLP for 1D: Adapted from 2D U-Net principles; limited direct literature

## Metadata

**Confidence breakdown:**
- U-Net MLP with FiLM: MEDIUM-HIGH - Well-understood components, novel combination for 1D
- Mamba velocity network: LOW - Experimental adaptation, may not work
- Architecture scaling: HIGH - Follows established DiT patterns
- Overall: MEDIUM - Two solid contributions, one experimental

**Research date:** 2026-02-01
**Valid until:** 2026-03-01 (30 days - experimental domain, fast-moving field)

---

## Appendix: Target Parameter Counts

### U-Net MLP Scaling
| Variant | hidden_dims | Estimated Params |
|---------|-------------|------------------|
| tiny | (256, 128) | ~600K |
| small | (512, 256) | ~2.5M |
| base | (768, 384) | ~5.5M |

### Mamba Velocity Network Scaling (Experimental)
| Variant | hidden_dim | num_layers | d_state | Estimated Params |
|---------|------------|------------|---------|------------------|
| tiny | 128 | 2 | 16 | ~500K |
| small | 256 | 4 | 16 | ~2M |
| base | 384 | 6 | 16 | ~5M |

Note: Mamba parameter counts depend heavily on d_state and expand factor. Verify after implementation.

## Appendix: Mamba Installation Troubleshooting

```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Install with matching versions
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm

# If build fails:
pip install mamba-ssm --no-build-isolation

# If still fails, mark Mamba as blocked and proceed without it
```
