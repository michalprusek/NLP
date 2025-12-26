#!/usr/bin/env python3
"""Visualize latent space with UMAP: EI landscape, optimum, projections.

Creates 2D UMAP visualization showing:
1. EI (Expected Improvement) landscape as heatmap
2. Maximum EI point (latent optimum)
3. Projection to GTR space (decoded embedding)
4. Vec2Text reconstructed text re-embedded

Usage:
    uv run python -m vec2text_hbbops.visualize_latent
    uv run python -m vec2text_hbbops.visualize_latent --output plots/latent_viz.png
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text_hbbops.hbbops_vec2text import HbBoPsVec2Text
from vec2text_hbbops.inference import Vec2TextHbBoPsInference
from vec2text_hbbops.training import TrainingConfig


def compute_ei(mean: float, std: float, best_val: float) -> float:
    """Compute Expected Improvement."""
    if std <= 0:
        return max(best_val - mean, 0)
    z = (best_val - mean) / std
    return (best_val - mean) * norm.cdf(z) + std * norm.pdf(z)


def predict_with_gp(
    hbbops: HbBoPsVec2Text,
    latent: torch.Tensor,
) -> Tuple[float, float]:
    """Get GP prediction (mean, std) for a latent point."""
    # Decode latent to embedding
    inst_emb, ex_emb = hbbops.decode_latent(latent)

    # Create input for GP
    X = torch.cat([inst_emb.unsqueeze(0), ex_emb.unsqueeze(0)], dim=1)

    # Normalize
    denominator = hbbops.X_max - hbbops.X_min
    denominator[denominator == 0] = 1.0
    X_norm = (X - hbbops.X_min) / denominator

    # Predict
    hbbops.gp_model.eval()
    hbbops.likelihood.eval()

    with torch.no_grad():
        pred = hbbops.likelihood(hbbops.gp_model(X_norm))
        mean = pred.mean.item() * hbbops.y_std.item() + hbbops.y_mean.item()
        std = pred.stddev.item() * hbbops.y_std.item()

    return mean, std


def collect_latent_points(
    hbbops: HbBoPsVec2Text,
    inference: Vec2TextHbBoPsInference,
    n_grid: int = 500,
) -> dict:
    """Collect all latent points for visualization.

    Returns dict with:
        - observed_latents: Latents from design data (training points)
        - observed_errors: Error rates for observed points
        - grid_latents: Random grid for EI landscape
        - grid_ei: EI values for grid points
        - best_latent: Latent with max EI
        - best_decoded_emb: GTR embedding after AE decode
        - vec2text_reembed: Re-embedding of Vec2Text output
    """
    print("Collecting latent points...")

    # 1. Get observed latents from design data
    observed_latents = []
    observed_errors = []

    for entry in hbbops.design_data:
        prompt_idx, inst_emb, ex_emb, error_rate, fidelity = entry
        prompt = hbbops.prompts[prompt_idx]
        latent = hbbops.get_prompt_latent(prompt)
        observed_latents.append(latent.cpu().numpy())
        observed_errors.append(error_rate)

    observed_latents = np.array(observed_latents)
    observed_errors = np.array(observed_errors)
    print(f"  Observed points: {len(observed_latents)}")

    # 2. Sample grid for EI landscape
    # Sample around observed distribution
    mean = observed_latents.mean(axis=0)
    std = observed_latents.std(axis=0) + 1e-6

    grid_latents = []
    grid_ei = []
    best_val = hbbops.best_validation_error

    print(f"  Sampling {n_grid} grid points for EI landscape...")
    for _ in range(n_grid):
        # Sample from expanded distribution
        z = mean + std * 1.5 * np.random.randn(10)
        z_tensor = torch.tensor(z, dtype=torch.float32, device=hbbops.device)

        try:
            m, s = predict_with_gp(hbbops, z_tensor)
            ei = compute_ei(m, s, best_val)
            grid_latents.append(z)
            grid_ei.append(ei)
        except Exception:
            continue

    grid_latents = np.array(grid_latents)
    grid_ei = np.array(grid_ei)
    print(f"  Grid points computed: {len(grid_latents)}")

    # 3. Find max EI point
    print("  Finding max EI point...")
    best_ei_idx = np.argmax(grid_ei)
    best_latent = grid_latents[best_ei_idx]
    best_latent_tensor = torch.tensor(best_latent, dtype=torch.float32, device=hbbops.device)

    # 4. Decode best latent to GTR embedding
    inst_emb_decoded, ex_emb_decoded = hbbops.decode_latent(best_latent_tensor)
    # Concatenate for visualization (we'll use instruction part)
    best_decoded_emb = inst_emb_decoded.cpu().numpy()

    # 5. Vec2Text inversion and re-embedding
    print("  Running Vec2Text inversion...")
    inv_result = inference.invert_latent_to_text(best_latent_tensor, verify=True)

    # Re-encode the Vec2Text output
    vec2text_inst_emb = inference.encoder.encode(inv_result.instruction_text)

    print(f"  Vec2Text instruction: {inv_result.instruction_text[:60]}...")
    print(f"  Instruction cosine: {inv_result.instruction_cosine:.4f}")

    return {
        "observed_latents": observed_latents,
        "observed_errors": observed_errors,
        "grid_latents": grid_latents,
        "grid_ei": grid_ei,
        "best_latent": best_latent,
        "best_decoded_emb": best_decoded_emb,
        "vec2text_reembed": vec2text_inst_emb,
        "vec2text_text": inv_result.instruction_text,
        "instruction_cosine": inv_result.instruction_cosine,
    }


def create_visualization(
    data: dict,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Create UMAP visualization with EI heatmap."""
    from scipy.interpolate import griddata

    print("\nCreating UMAP visualization...")

    # Combine all latent points for UMAP fitting
    all_latents = np.vstack([
        data["observed_latents"],
        data["grid_latents"],
        data["best_latent"].reshape(1, -1),
    ])

    # Fit UMAP
    print("  Fitting UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    all_2d = reducer.fit_transform(all_latents)

    # Split back
    n_obs = len(data["observed_latents"])
    n_grid = len(data["grid_latents"])

    observed_2d = all_2d[:n_obs]
    grid_2d = all_2d[n_obs:n_obs + n_grid]
    best_latent_2d = all_2d[n_obs + n_grid]

    # Find max EI in 2D space
    max_ei_idx = np.argmax(data["grid_ei"])
    max_ei_2d = grid_2d[max_ei_idx]
    max_ei_value = data["grid_ei"][max_ei_idx]

    print(f"  Max EI value: {max_ei_value:.6f}")
    print(f"  Max EI location (UMAP): ({max_ei_2d[0]:.3f}, {max_ei_2d[1]:.3f})")

    # Create regular grid for heatmap interpolation
    print("  Interpolating EI heatmap...")
    margin = 0.5
    x_min, x_max = grid_2d[:, 0].min() - margin, grid_2d[:, 0].max() + margin
    y_min, y_max = grid_2d[:, 1].min() - margin, grid_2d[:, 1].max() + margin

    grid_resolution = 200
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate EI values onto regular grid
    Zi = griddata(
        grid_2d,
        data["grid_ei"],
        (Xi, Yi),
        method="cubic",
        fill_value=0,
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))

    # 1. Plot EI as continuous heatmap
    heatmap = ax.imshow(
        Zi,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        alpha=0.9,
    )
    cbar = plt.colorbar(heatmap, ax=ax, label="Expected Improvement (EI)", pad=0.02)

    # Add contour lines
    contours = ax.contour(
        Xi, Yi, Zi,
        levels=15,
        colors="white",
        linewidths=0.5,
        alpha=0.7,
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.4f")

    # 2. Plot observed points (training data for GP)
    obs_scatter = ax.scatter(
        observed_2d[:, 0],
        observed_2d[:, 1],
        c=data["observed_errors"],
        cmap="RdYlGn_r",
        s=120,
        edgecolors="white",
        linewidths=2,
        marker="o",
        label="GP training points",
        zorder=5,
    )

    # 3. Mark best observed point
    best_obs_idx = np.argmin(data["observed_errors"])
    ax.scatter(
        observed_2d[best_obs_idx, 0],
        observed_2d[best_obs_idx, 1],
        c="none",
        s=250,
        marker="o",
        edgecolors="cyan",
        linewidths=3,
        label=f"Best observed (err={data['observed_errors'][best_obs_idx]:.3f})",
        zorder=6,
    )

    # 4. Plot max EI point (latent optimum) - TRUE MAXIMUM
    ax.scatter(
        max_ei_2d[0],
        max_ei_2d[1],
        c="red",
        s=400,
        marker="*",
        edgecolors="white",
        linewidths=2,
        label=f"Max EI = {max_ei_value:.5f}",
        zorder=10,
    )

    # 5. Add annotation for Vec2Text result
    ax.annotate(
        f"Max EI → Vec2Text\ncosine: {data['instruction_cosine']:.3f}",
        xy=(max_ei_2d[0], max_ei_2d[1]),
        xytext=(max_ei_2d[0] + 1.0, max_ei_2d[1] + 1.0),
        fontsize=11,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="red", alpha=0.9),
        zorder=11,
    )

    # Labels and title
    ax.set_xlabel("UMAP Dimension 1", fontsize=13)
    ax.set_ylabel("UMAP Dimension 2", fontsize=13)
    ax.set_title(
        "Vec2Text-HbBoPs: EI Landscape in Latent Space\n"
        f"10D Latent → UMAP 2D | Max EI = {max_ei_value:.5f}",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=11)

    # Add text box with Vec2Text output
    textstr = (
        f"Novel prompt from Max EI (Vec2Text):\n"
        f"{data['vec2text_text'][:100]}..."
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.95, edgecolor="orange")
    ax.text(
        0.02, 0.02, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {output_path}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize latent space with UMAP")
    parser.add_argument(
        "--output",
        type=str,
        default="vec2text_hbbops/results/latent_visualization.png",
        help="Output path for plot",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=500,
        help="Number of grid points for EI landscape",
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=500,
        help="AE training epochs",
    )
    parser.add_argument(
        "--v2t-steps",
        type=int,
        default=50,
        help="Vec2Text correction steps",
    )
    parser.add_argument(
        "--v2t-beam",
        type=int,
        default=8,
        help="Vec2Text beam width",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot (just save)",
    )

    args = parser.parse_args()

    # Load data
    root = Path(__file__).parent.parent

    with open(root / "datasets/hbbops/instructions_25.txt") as f:
        instructions = [line.strip() for line in f if line.strip()]
    with open(root / "datasets/hbbops/examples_25.txt") as f:
        exemplars = [ex.strip() for ex in f.read().split("\n\n") if ex.strip()]
    with open(root / "hbbops_improved_2/data/validation.json") as f:
        validation_data = json.load(f)

    print("=" * 60)
    print("Vec2Text-HbBoPs Latent Space Visualization")
    print("=" * 60)

    # Create HbBoPs
    hbbops = HbBoPsVec2Text(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=validation_data,
        llm_evaluator=None,
        seed=42,
    )

    # Train AE and load GP
    config = TrainingConfig(max_epochs=args.ae_epochs, patience=20)

    print("\nTraining autoencoder on full grid...")
    hbbops.train_autoencoder_from_grid(
        "datasets/hbbops/full_grid_combined.jsonl",
        config,
        verbose=True,
    )

    print("\nLoading top-25 for GP...")
    hbbops.load_from_grid(
        "datasets/hbbops/full_grid_combined.jsonl",
        top_k=25,
        verbose=True,
    )

    # Create inference pipeline
    inference = Vec2TextHbBoPsInference(
        hbbops,
        vec2text_steps=args.v2t_steps,
        vec2text_beam=args.v2t_beam,
    )

    # Collect data for visualization
    data = collect_latent_points(hbbops, inference, n_grid=args.n_grid)

    # Create visualization
    create_visualization(
        data,
        output_path=Path(args.output),
        show=not args.no_show,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
