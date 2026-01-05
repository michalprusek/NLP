"""Quality KPIs for LIPO pipeline monitoring.

Three key metrics for diagnosing optimization quality:
1. VAE Quality (Q_VAE): Reconstruction quality with percentile analysis
2. GP Predictive Power (Q_GP): Spearman correlation between predictions and actuals
3. System Consistency (Q_Sys): Optimization gap between z_opt and z_real
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F


def compute_vae_quality(
    vae,
    embeddings: torch.Tensor,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute VAE reconstruction quality with percentile analysis.

    This metric measures how well the VAE reconstructs input embeddings.
    The 10th percentile (Q10) is the critical metric - it should be >0.90
    to ensure consistent round-trip quality for all instructions.

    Args:
        vae: InstructionVAE model
        embeddings: Input embeddings tensor (N, 768)
        device: Device for computation (defaults to embeddings.device)

    Returns:
        Dictionary with:
        - cosine_mean: Mean cosine similarity
        - cosine_std: Standard deviation
        - percentile_10: 10th percentile similarity (critical: should be >0.90)
        - percentile_25: 25th percentile similarity
        - below_90_count: Number of samples with similarity < 0.90
        - below_90_pct: Percentage of samples below threshold
        - quality_tier: "good" (Q10>0.92), "acceptable" (Q10>0.90), "poor"
    """
    if device is None:
        device = embeddings.device

    vae.eval()
    embeddings = embeddings.to(device)

    with torch.no_grad():
        z = vae.encode_mu(embeddings)
        reconstructed = vae.decode(z)
        similarities = F.cosine_similarity(embeddings, reconstructed, dim=-1)

    similarities_np = similarities.cpu().numpy()

    percentile_10 = float(np.percentile(similarities_np, 10))
    quality_tier = (
        "good" if percentile_10 > 0.92 else
        "acceptable" if percentile_10 > 0.90 else
        "poor"
    )

    return {
        "cosine_mean": float(similarities.mean().item()),
        "cosine_std": float(similarities.std().item()),
        "cosine_min": float(similarities.min().item()),
        "cosine_max": float(similarities.max().item()),
        "percentile_10": percentile_10,
        "percentile_25": float(np.percentile(similarities_np, 25)),
        "below_90_count": int((similarities < 0.90).sum().item()),
        "below_90_pct": float((similarities < 0.90).sum().item() / len(similarities) * 100),
        "quality_tier": quality_tier,
    }


def compute_gp_spearman(
    predicted_errors: List[float],
    actual_errors: List[float],
) -> Dict[str, Any]:
    """Compute Spearman rank correlation between GP predictions and actual errors.

    This metric measures how well the GP can rank instructions by quality.
    A high correlation means the GP can reliably guide optimization.

    Current baseline: ~0 (random, GP falls to prior mean)
    Target: >0.4 for meaningful optimization guidance

    Args:
        predicted_errors: List of GP predicted error rates
        actual_errors: List of actual evaluated error rates

    Returns:
        Dictionary with:
        - spearman_correlation: Rank correlation coefficient [-1, 1]
        - p_value: Statistical significance
        - is_significant: Whether p < 0.05
        - quality_tier: "good" (>0.4), "poor" (>0.2), "random"
        - n_samples: Number of samples used
    """
    from scipy.stats import spearmanr

    if len(predicted_errors) < 3:
        return {
            "spearman_correlation": None,
            "p_value": None,
            "is_significant": False,
            "quality_tier": "insufficient_data",
            "n_samples": len(predicted_errors),
        }

    # Filter out NaN values
    valid_pairs = [
        (p, a) for p, a in zip(predicted_errors, actual_errors)
        if not (np.isnan(p) or np.isnan(a))
    ]

    if len(valid_pairs) < 3:
        return {
            "spearman_correlation": None,
            "p_value": None,
            "is_significant": False,
            "quality_tier": "insufficient_data",
            "n_samples": len(valid_pairs),
        }

    pred, actual = zip(*valid_pairs)
    correlation, p_value = spearmanr(pred, actual)

    # Handle NaN correlation (constant values)
    if np.isnan(correlation):
        correlation = 0.0
        p_value = 1.0

    quality_tier = (
        "good" if correlation > 0.4 else
        "poor" if correlation > 0.2 else
        "random"
    )

    return {
        "spearman_correlation": float(correlation),
        "p_value": float(p_value),
        "is_significant": p_value < 0.05,
        "quality_tier": quality_tier,
        "n_samples": len(valid_pairs),
    }


def compute_system_gap(
    z_opt_z_real_euclidean: List[float],
) -> Dict[str, Any]:
    """Analyze optimization gap between proposed z_opt and realized z_real.

    This metric measures how far the actual latent (after Vec2Text inversion)
    is from the GP's proposed optimal point. A large gap means the GP is
    optimizing in regions that don't map well to valid text.

    Target: mean < 0.5, max < 1.0
    Current baseline: 1.0-1.8 (very high, GP optimizes "fiction")

    Args:
        z_opt_z_real_euclidean: List of Euclidean distances between
            z_opt (GP proposal) and z_real (GTR encoding of generated text)

    Returns:
        Dictionary with:
        - gap_mean: Mean optimization gap
        - gap_std: Standard deviation
        - gap_min: Minimum gap
        - gap_max: Maximum gap
        - within_threshold_pct: Percentage of samples with gap < 0.5
        - quality_tier: "good" (mean<0.5), "acceptable" (mean<1.0), "poor"
    """
    if len(z_opt_z_real_euclidean) == 0:
        return {
            "gap_mean": None,
            "gap_std": None,
            "gap_min": None,
            "gap_max": None,
            "within_threshold_pct": None,
            "quality_tier": "no_data",
        }

    gaps = np.array(z_opt_z_real_euclidean)
    gap_mean = float(gaps.mean())

    quality_tier = (
        "good" if gap_mean < 0.5 else
        "acceptable" if gap_mean < 1.0 else
        "poor"
    )

    return {
        "gap_mean": gap_mean,
        "gap_std": float(gaps.std()),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "within_threshold_pct": float((gaps < 0.5).sum() / len(gaps) * 100),
        "quality_tier": quality_tier,
    }


def compute_all_kpis(
    vae=None,
    embeddings: Optional[torch.Tensor] = None,
    predicted_errors: Optional[List[float]] = None,
    actual_errors: Optional[List[float]] = None,
    z_gaps: Optional[List[float]] = None,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute all available KPIs.

    Convenience function to compute all metrics at once.
    Only computes metrics for which data is provided.

    Returns:
        Dictionary with keys:
        - vae_quality: VAE reconstruction metrics (if vae and embeddings provided)
        - gp_spearman: GP prediction quality (if errors provided)
        - system_gap: Optimization gap metrics (if z_gaps provided)
    """
    result = {}

    if vae is not None and embeddings is not None:
        result["vae_quality"] = compute_vae_quality(vae, embeddings, device)

    if predicted_errors is not None and actual_errors is not None:
        result["gp_spearman"] = compute_gp_spearman(predicted_errors, actual_errors)

    if z_gaps is not None:
        result["system_gap"] = compute_system_gap(z_gaps)

    return result


def format_kpi_report(kpis: Dict[str, Dict[str, Any]], iteration: int = 0) -> str:
    """Format KPIs as a human-readable report string.

    Args:
        kpis: Dictionary from compute_all_kpis()
        iteration: Current iteration number for header

    Returns:
        Formatted string report
    """
    lines = [
        f"=== KPI Report (iter {iteration}) ===",
    ]

    if "vae_quality" in kpis:
        vq = kpis["vae_quality"]
        lines.extend([
            f"VAE Quality:",
            f"  Cosine Mean: {vq['cosine_mean']:.4f} (std: {vq['cosine_std']:.4f})",
            f"  Percentile 10: {vq['percentile_10']:.4f} (target: >0.90)",
            f"  Below 90%: {vq['below_90_count']} samples ({vq['below_90_pct']:.1f}%)",
            f"  Tier: {vq['quality_tier']}",
        ])

    if "gp_spearman" in kpis:
        gp = kpis["gp_spearman"]
        if gp["spearman_correlation"] is not None:
            lines.extend([
                f"GP Predictive Power:",
                f"  Spearman Correlation: {gp['spearman_correlation']:.3f}",
                f"  p-value: {gp['p_value']:.4f} ({'significant' if gp['is_significant'] else 'not significant'})",
                f"  Tier: {gp['quality_tier']}",
            ])
        else:
            lines.append(f"GP Predictive Power: {gp['quality_tier']}")

    if "system_gap" in kpis:
        sg = kpis["system_gap"]
        if sg["gap_mean"] is not None:
            lines.extend([
                f"System Gap (z_opt vs z_real):",
                f"  Mean: {sg['gap_mean']:.3f} (std: {sg['gap_std']:.3f})",
                f"  Range: [{sg['gap_min']:.3f}, {sg['gap_max']:.3f}]",
                f"  Within threshold (<0.5): {sg['within_threshold_pct']:.1f}%",
                f"  Tier: {sg['quality_tier']}",
            ])

    return "\n".join(lines)
