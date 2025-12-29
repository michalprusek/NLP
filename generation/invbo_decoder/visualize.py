"""EI Landscape Visualization for InvBO Decoder.

Creates 2D UMAP projection of the 10D latent space to visualize:
- Expected Improvement (EI) surface
- z_opt (optimized latent target)
- z_realized (actual latent after Vec2Text inversion)
- Trust region boundary

Adapted from COWBOYS visualization for InvBO's 10D latent space.
"""

import torch
import torch.nn.functional as F
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
    from .inference import InvBOInference
    from .trust_region import TrustRegionManager


def sample_grid_around_center(
    center: np.ndarray,
    span: float,
    n_samples: int,
    latent_dim: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Sample points in latent hypercube around center using Latin Hypercube Sampling.

    LHS provides better coverage of high-dimensional space than random sampling.

    Args:
        center: Center point (latent_dim,)
        span: Half-width of sampling region in each dimension
        n_samples: Number of samples to generate
        latent_dim: Dimension of latent space (10 for InvBO)
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, latent_dim) with sampled points
    """
    sampler = LatinHypercube(d=latent_dim, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Scale from [0, 1] to [-span, span] around center
    scaled = (unit_samples - 0.5) * 2 * span
    grid = center + scaled

    return grid


def visualize_ei_landscape(
    inference: "InvBOInference",
    center_latent: torch.Tensor,
    realized_text: str,
    best_y: float,
    trust_region: Optional["TrustRegionManager"] = None,
    span: float = 1.0,
    n_grid_samples: int = 300,
    umap_neighbors: int = 15,
    save_path: str = "ei_landscape.png",
) -> Dict[str, Any]:
    """Visualize EI landscape with UMAP projection.

    Creates a 2D visualization showing:
    - EI surface (contour plot)
    - z_opt (red star) - where optimization found high EI
    - z_realized (white X) - where Vec2Text actually landed
    - Trust region boundary (yellow dashed line)

    Args:
        inference: InvBOInference instance with GP and decoder
        center_latent: z_opt - the optimized latent (10D)
        realized_text: Generated instruction text from Vec2Text
        best_y: Best observed error rate for EI computation
        trust_region: Optional trust region manager for boundary overlay
        span: Sampling distance from center in latent space
        n_grid_samples: Number of grid samples for EI evaluation
        umap_neighbors: UMAP n_neighbors parameter
        save_path: Path to save the visualization

    Returns:
        Dictionary with diagnostic metrics:
        - cosine_gap: Cosine distance between z_opt and z_realized embeddings
        - inversion_gap_2d: Distance in UMAP 2D space
        - log_ei_at_opt: LogEI value at z_opt
        - log_ei_at_realized: LogEI value at z_realized
        - trust_region_radius: Trust region radius if provided
    """
    if umap is None:
        raise ImportError("umap-learn is required for visualization. Install with: pip install umap-learn")

    device = inference.device
    decoder = inference.decoder
    gp = inference.gp
    gtr = inference.gtr

    # Determine latent dimension
    latent_dim = center_latent.shape[0]

    # Ensure center_latent is on the right device
    center_latent = center_latent.to(device)

    # Step 1: Extract z_realized from generated text
    with torch.no_grad():
        text_emb = gtr.encode_tensor(realized_text).to(device)
        z_realized = gp.get_latent(text_emb)

    # Step 2: Sample grid points in latent space
    center_np = center_latent.cpu().numpy()
    grid_latent = sample_grid_around_center(center_np, span, n_grid_samples, latent_dim)

    # Step 3: Compute EI for grid points
    grid_tensor = torch.tensor(grid_latent, dtype=torch.float32).to(device)
    ei_values = []
    batch_size = 50

    decoder.eval()
    with torch.no_grad():
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i + batch_size]
            # Decode to embeddings
            embeddings = decoder(batch)
            # Compute EI for each embedding
            for emb in embeddings:
                ei = gp.expected_improvement(emb)
                ei_values.append(max(0, ei))  # Clip to positive for visualization

    ei_values = np.array(ei_values)

    # Step 4: Combine all points for joint UMAP fitting
    idx_z_opt = 0
    idx_z_realized = 1
    idx_grid_start = 2

    all_latent_points = np.vstack([
        center_np.reshape(1, -1),
        z_realized.cpu().numpy().reshape(1, -1),
        grid_latent,
    ])

    # Step 5: Fit UMAP
    n_neighbors = min(umap_neighbors, len(all_latent_points) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    all_2d = reducer.fit_transform(all_latent_points)

    # Extract projected positions
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

    # z_opt marker (red star)
    ax.scatter(
        z_opt_2d[0],
        z_opt_2d[1],
        c="red",
        marker="*",
        s=300,
        label=r"$z_{opt}$ (EI optimum)",
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
            tr_radius = trust_region.radius

            # Sample points on L-infinity boundary
            n_boundary_points = 50
            boundary_points_latent = []
            np.random.seed(42)

            for _ in range(n_boundary_points):
                random_dir = np.random.randn(latent_dim)
                # Normalize to L-inf unit ball (max abs value = 1)
                random_dir = random_dir / np.abs(random_dir).max()
                point = anchor + tr_radius * random_dir
                boundary_points_latent.append(point)

            # Project boundary to 2D using same UMAP transform
            boundary_2d = reducer.transform(np.array(boundary_points_latent))

            # Draw convex hull of boundary points
            try:
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
            except Exception:
                pass  # ConvexHull can fail with too few points
        except Exception as e:
            print(f"Warning: Could not draw trust region boundary: {e}")

    # Compute diagnostic metrics
    # Use cosine distance in embedding space (more stable than L2 in latent space)
    with torch.no_grad():
        emb_opt = decoder(center_latent)
        emb_realized = decoder(z_realized)
    cosine_gap = 1 - F.cosine_similarity(
        emb_opt.unsqueeze(0), emb_realized.unsqueeze(0)
    ).item()
    gap_2d = float(np.linalg.norm(z_opt_2d - z_realized_2d))

    # Annotation with metrics
    ax.annotate(
        f"Cosine Gap: {cosine_gap:.4f}\nGap (2D): {gap_2d:.3f}",
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
        f"InvBO EI Landscape (UMAP projection, {latent_dim}D latent)\n"
        f'Text: "{text_display}"',
        fontsize=12,
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Compute EI at key points
    with torch.no_grad():
        emb_opt_for_ei = decoder(center_latent)
        emb_realized_for_ei = decoder(z_realized)
    log_ei_at_opt = gp.log_expected_improvement(emb_opt_for_ei)
    log_ei_at_realized = gp.log_expected_improvement(emb_realized_for_ei)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {save_path}")

    return {
        "cosine_gap": cosine_gap,
        "inversion_gap_2d": gap_2d,
        "log_ei_at_opt": log_ei_at_opt,
        "log_ei_at_realized": log_ei_at_realized,
        "trust_region_radius": tr_radius,
    }
