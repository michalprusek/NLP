"""EI Landscape Visualization for COWBOYS Vec2Text.

Creates 2D UMAP projection of the 32D latent space to visualize:
- Expected Improvement (EI) surface
- MCMC trajectory samples
- z_opt (optimized latent target)
- z_realized (actual latent after Vec2Text inversion)
- Trust region boundary

This is a "reality check" for diagnosing inversion gap between
where MCMC finds high EI and where Vec2Text actually lands.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, TYPE_CHECKING

try:
    import umap
except ImportError:
    umap = None
    print("Warning: umap-learn not installed. Run: pip install umap-learn")

from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from scipy.stats.qmc import LatinHypercube

if TYPE_CHECKING:
    from .inference import CowboysInference
    from .trust_region import TrustRegionManager


def sample_grid_around_center(
    center: np.ndarray,
    span: float,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample points in 32D hypercube around center using Latin Hypercube Sampling.

    LHS provides better coverage of high-dimensional space than random sampling.

    Args:
        center: Center point (32,)
        span: Half-width of sampling region in each dimension
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 32) with sampled points
    """
    sampler = LatinHypercube(d=32, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Scale from [0, 1] to [-span, span] around center
    scaled = (unit_samples - 0.5) * 2 * span
    grid_32d = center + scaled

    return grid_32d


def visualize_ei_landscape(
    inference: "CowboysInference",
    center_latent: torch.Tensor,
    realized_text: str,
    best_y: float,
    trajectory_latents: Optional[List[torch.Tensor]] = None,
    trust_region: Optional["TrustRegionManager"] = None,
    span: float = 2.0,
    n_grid_samples: int = 500,
    umap_neighbors: int = 15,
    save_path: str = "ei_landscape.png",
) -> Dict[str, Any]:
    """Visualize EI landscape with UMAP projection.

    Creates a 2D visualization showing:
    - EI surface (contour plot)
    - MCMC trajectory samples
    - z_opt (red star) - where MCMC found high EI
    - z_realized (white X) - where Vec2Text actually landed
    - Trust region boundary (yellow dashed line)

    Args:
        inference: CowboysInference instance with VAE, GP, and MCMC sampler
        center_latent: z_opt - the optimized latent (32D)
        realized_text: Generated instruction text from Vec2Text
        best_y: Best observed error rate for EI computation
        trajectory_latents: List of MCMC samples for trajectory visualization
        trust_region: Optional trust region manager for boundary overlay
        span: Sampling distance from center in 32D space
        n_grid_samples: Number of 32D grid samples for EI evaluation
        umap_neighbors: UMAP n_neighbors parameter
        save_path: Path to save the visualization

    Returns:
        Dictionary with diagnostic metrics:
        - inversion_gap_32d: Distance between z_opt and z_realized in 32D
        - inversion_gap_2d: Distance in UMAP 2D space
        - log_ei_at_opt: LogEI value at z_opt
        - log_ei_at_realized: LogEI value at z_realized
        - n_trajectory_points: Number of MCMC samples used
        - trust_region_radius: Trust region radius if provided
    """
    if umap is None:
        raise ImportError("umap-learn is required for visualization. Install with: pip install umap-learn")

    device = inference.device
    vae = inference.vae
    gtr = inference.gtr

    # Ensure center_latent is on the right device
    center_latent = center_latent.to(device)

    # Step 1: Extract z_realized from generated text
    with torch.no_grad():
        text_emb = gtr.encode_tensor(realized_text).to(device)
        z_realized = vae.get_latent(text_emb.unsqueeze(0)).squeeze(0)

    # Step 2: Sample grid points in 32D space
    center_np = center_latent.cpu().numpy()
    grid_32d = sample_grid_around_center(center_np, span, n_grid_samples)

    # Step 3: Compute EI for grid points
    grid_tensor = torch.tensor(grid_32d, dtype=torch.float32).to(device)
    ei_values = []
    batch_size = 100

    for i in range(0, len(grid_tensor), batch_size):
        batch = grid_tensor[i:i + batch_size]
        log_ei = inference.mcmc_sampler.compute_log_ei(batch, best_y)
        # Convert log EI to EI for visualization
        ei = torch.exp(log_ei).cpu().numpy()
        if ei.ndim == 0:
            ei_values.append(ei.item())
        else:
            ei_values.extend(ei.tolist())

    ei_values = np.array(ei_values)

    # Step 4: Combine all points for joint UMAP fitting
    trajectory_list = []
    if trajectory_latents and len(trajectory_latents) > 0:
        for z in trajectory_latents:
            trajectory_list.append(z.cpu().numpy())
        trajectory_np = np.array(trajectory_list)
    else:
        trajectory_np = np.empty((0, 32))

    n_traj = len(trajectory_list)
    idx_z_opt = n_traj
    idx_z_realized = n_traj + 1
    idx_grid_start = n_traj + 2

    all_32d_points = np.vstack([
        trajectory_np if n_traj > 0 else np.empty((0, 32)),
        center_np.reshape(1, -1),
        z_realized.cpu().numpy().reshape(1, -1),
        grid_32d,
    ])

    # Step 5: Fit UMAP
    n_neighbors = min(umap_neighbors, len(all_32d_points) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    all_2d = reducer.fit_transform(all_32d_points)

    # Extract projected positions
    trajectory_2d = all_2d[:n_traj] if n_traj > 0 else None
    z_opt_2d = all_2d[idx_z_opt]
    z_realized_2d = all_2d[idx_z_realized]
    grid_2d = all_2d[idx_grid_start:]

    # Step 6: Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    # EI surface using RBF interpolation
    rbf = RBFInterpolator(grid_2d, ei_values, kernel="thin_plate_spline", smoothing=0.001)

    # Create dense grid for smooth contours
    x_margin = (grid_2d[:, 0].max() - grid_2d[:, 0].min()) * 0.1
    y_margin = (grid_2d[:, 1].max() - grid_2d[:, 1].min()) * 0.1
    xx, yy = np.meshgrid(
        np.linspace(grid_2d[:, 0].min() - x_margin, grid_2d[:, 0].max() + x_margin, 200),
        np.linspace(grid_2d[:, 1].min() - y_margin, grid_2d[:, 1].max() + y_margin, 200),
    )
    ei_surface = rbf(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Clip negative EI values (can occur from interpolation)
    ei_surface = np.clip(ei_surface, 0, None)

    # EI contour plot
    contour = ax.contourf(xx, yy, ei_surface, levels=20, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="Expected Improvement")
    ax.contour(xx, yy, ei_surface, levels=10, colors="white", alpha=0.3, linewidths=0.5)

    # MCMC Trajectory points
    if trajectory_2d is not None and len(trajectory_2d) > 0:
        ax.scatter(
            trajectory_2d[:, 0],
            trajectory_2d[:, 1],
            c="lightblue",
            s=20,
            alpha=0.5,
            label=f"MCMC samples (n={n_traj})",
        )

    # z_opt marker (red star)
    ax.scatter(
        z_opt_2d[0],
        z_opt_2d[1],
        c="red",
        marker="*",
        s=300,
        label=r"$z_{opt}$ (MCMC target)",
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
    )

    # z_realized marker (white X)
    ax.scatter(
        z_realized_2d[0],
        z_realized_2d[1],
        c="white",
        marker="X",
        s=200,
        label=r"$z_{realized}$ (Generated text)",
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )

    # Connect with dashed line (inversion gap)
    ax.plot(
        [z_opt_2d[0], z_realized_2d[0]],
        [z_opt_2d[1], z_realized_2d[1]],
        "r--",
        linewidth=2,
        alpha=0.7,
    )

    # Trust Region Boundary Overlay
    tr_radius = None
    if trust_region is not None:
        try:
            anchor = trust_region.anchor.cpu().numpy()
            tr_radius = trust_region.state.radius

            # Sample points on L-infinity boundary
            n_boundary_points = 100
            boundary_points_32d = []
            np.random.seed(42)

            for _ in range(n_boundary_points):
                random_dir = np.random.randn(32)
                # Normalize to L-inf unit ball (max abs value = 1)
                random_dir = random_dir / np.abs(random_dir).max()
                point = anchor + tr_radius * random_dir
                boundary_points_32d.append(point)

            # Project boundary to 2D using same UMAP transform
            boundary_2d = reducer.transform(np.array(boundary_points_32d))

            # Draw convex hull of boundary points
            hull = ConvexHull(boundary_2d)
            for simplex in hull.simplices:
                ax.plot(
                    boundary_2d[simplex, 0],
                    boundary_2d[simplex, 1],
                    "y--",
                    linewidth=2,
                    alpha=0.8,
                )
            ax.plot([], [], "y--", linewidth=2, label=f"Trust Region (r={tr_radius:.2f})")
        except Exception as e:
            print(f"Warning: Could not draw trust region boundary: {e}")

    # Compute diagnostic metrics
    gap_2d = float(np.linalg.norm(z_opt_2d - z_realized_2d))
    gap_32d = float(torch.norm(center_latent - z_realized).item())

    # Annotation with metrics
    ax.annotate(
        f"Gap (32D): {gap_32d:.3f}\nGap (2D): {gap_2d:.3f}",
        xy=(z_realized_2d[0], z_realized_2d[1]),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    # Truncate text for title
    text_display = realized_text[:60] + "..." if len(realized_text) > 60 else realized_text

    ax.set_title(
        f"EI Landscape & Inversion Drift (UMAP projection)\n"
        f'Text: "{text_display}"',
        fontsize=12,
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Compute EI at key points
    log_ei_at_opt = inference.mcmc_sampler.compute_log_ei(center_latent, best_y)
    log_ei_at_realized = inference.mcmc_sampler.compute_log_ei(z_realized, best_y)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")

    return {
        "inversion_gap_32d": gap_32d,
        "inversion_gap_2d": gap_2d,
        "log_ei_at_opt": log_ei_at_opt.item() if isinstance(log_ei_at_opt, torch.Tensor) else log_ei_at_opt,
        "log_ei_at_realized": log_ei_at_realized.item() if isinstance(log_ei_at_realized, torch.Tensor) else log_ei_at_realized,
        "n_trajectory_points": n_traj,
        "trust_region_radius": tr_radius,
    }
