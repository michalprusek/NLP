"""Architecture scaling configurations for velocity networks.

This module provides scaling configurations for ablation studies measuring
how model capacity interacts with dataset size.

Scaling follows DiT-style with approximately 4x parameter jumps between levels:
- Tiny: ~0.3-3M params (fast iteration, debugging)
- Small: ~1-10M params (current defaults, baseline)
- Base: ~2-20M params (production scale)

Usage:
    from study.flow_matching.models.scaling import get_scaled_config

    config = get_scaled_config('mlp', 'tiny')  # {'hidden_dim': 128, 'num_layers': 4}
    config = get_scaled_config('dit', 'base')  # {'hidden_dim': 512, ...}
"""

from typing import Dict, List, Any

# Architecture scaling configurations
# Each config provides kwargs to pass to the respective model constructor
SCALING_CONFIGS: Dict[str, Dict[str, Any]] = {
    # MLP scaling: ~300K (tiny), ~920K (small), ~2.2M (base) params
    "mlp_tiny": {"hidden_dim": 128, "num_layers": 4},
    "mlp_small": {"hidden_dim": 256, "num_layers": 5},  # Current default
    "mlp_base": {"hidden_dim": 384, "num_layers": 6},

    # DiT scaling: ~3M (tiny), ~9.3M (small), ~20M (base) params
    "dit_tiny": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4},
    "dit_small": {"hidden_dim": 384, "num_layers": 3, "num_heads": 6},  # Current default
    "dit_base": {"hidden_dim": 512, "num_layers": 4, "num_heads": 8},

    # U-Net MLP scaling: ~2M (tiny), ~6.9M (small), ~15M (base) params
    # Note: Concatenative skip connections make param counts higher than simple MLPs
    "unet_tiny": {"hidden_dims": (256, 128)},
    "unet_small": {"hidden_dims": (512, 256)},  # Current default
    "unet_base": {"hidden_dims": (768, 384)},

    # Mamba scaling (experimental): ~500K (tiny), ~2M (small), ~5M (base) params
    # Requires mamba-ssm package (currently blocked by CUDA version mismatch)
    "mamba_tiny": {"hidden_dim": 128, "num_layers": 2},
    "mamba_small": {"hidden_dim": 256, "num_layers": 4},  # Current default
    "mamba_base": {"hidden_dim": 384, "num_layers": 6},
}


def get_scaled_config(arch: str, scale: str) -> Dict[str, Any]:
    """Get architecture configuration for a given scale.

    Args:
        arch: Architecture name ('mlp', 'dit', 'unet', 'mamba').
        scale: Scale level ('tiny', 'small', 'base').

    Returns:
        Configuration dict to pass as kwargs to create_model().

    Raises:
        ValueError: If arch/scale combination is not available.

    Examples:
        >>> get_scaled_config('mlp', 'tiny')
        {'hidden_dim': 128, 'num_layers': 4}
        >>> get_scaled_config('unet', 'base')
        {'hidden_dims': (768, 384)}
    """
    key = f"{arch}_{scale}"
    if key not in SCALING_CONFIGS:
        available = list_available_scales()
        raise ValueError(
            f"Unknown configuration: '{key}'. "
            f"Available architectures: {list(available.keys())}. "
            f"Available scales: {list(available.get(arch, ['tiny', 'small', 'base']))}. "
            f"All configs: {list(SCALING_CONFIGS.keys())}"
        )
    # Return a copy to prevent mutation
    return dict(SCALING_CONFIGS[key])


def list_available_scales() -> Dict[str, List[str]]:
    """List available scales for each architecture.

    Returns:
        Dict mapping architecture names to list of available scale levels.

    Examples:
        >>> list_available_scales()
        {'mlp': ['tiny', 'small', 'base'], 'dit': ['tiny', 'small', 'base'], ...}
    """
    result: Dict[str, List[str]] = {}
    for key in SCALING_CONFIGS:
        arch, scale = key.rsplit("_", 1)
        if arch not in result:
            result[arch] = []
        result[arch].append(scale)
    # Sort scales for consistent ordering
    for arch in result:
        result[arch].sort(key=lambda s: ["tiny", "small", "base"].index(s))
    return result
