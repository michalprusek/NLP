"""EI Landscape Visualization with UMAP projection.

Visualizes Expected Improvement landscape in 2D using UMAP projection
of the 32D VAE latent space. Shows:
- EI heatmap (raw values, no clipping)
- Training points (colored by error rate)
- Optimized latent (star)
- Re-embedded latent (cross) with arrow for cycle consistency
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from pathlib import Path
from typing import Optional
import warnings

try:
    import umap
except ImportError:
    umap = None


def compute_ei_batch(
    latents: torch.Tensor,
    vae,
    exemplar_selector,
    exemplar_emb: torch.Tensor,
    best_y: float,
    xi: float = 0.01,
    device: str = "cuda",
) -> np.ndarray:
    """Compute EI for a batch of latent points.

    Args:
        latents: Latent points (N, 32)
        vae: Trained VAE model
        exemplar_selector: ExemplarSelector with trained GP
        exemplar_emb: Fixed exemplar embedding (768,)
        best_y: Best observed error rate
        xi: EI exploration parameter
        device: Device to use

    Returns:
        EI values (N,)
    """
    import gpytorch

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    latents = latents.to(device)
    exemplar_emb = exemplar_emb.to(device)

    # Move VAE to device
    vae = vae.to(device)
    vae.eval()

    # Decode to instruction embeddings
    with torch.no_grad():
        inst_emb = vae.decode(latents)  # (N, 768)

    # Prepare GP input
    batch_size = inst_emb.shape[0]
    exemplar_expanded = exemplar_emb.unsqueeze(0).expand(batch_size, -1)
    X = torch.cat([inst_emb, exemplar_expanded], dim=1)  # (N, 1536)

    # Normalize using ExemplarSelector's stored params (move to device)
    selector = exemplar_selector
    X_min = selector.X_min.to(device)
    X_max = selector.X_max.to(device)
    denominator = X_max - X_min
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    X_norm = (X - X_min) / denominator

    # Move GP model and likelihood to device
    selector.gp_model = selector.gp_model.to(device)
    selector.likelihood = selector.likelihood.to(device)
    selector.gp_model.eval()
    selector.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = selector.likelihood(selector.gp_model(X_norm))
        mean_norm = pred.mean
        var_norm = pred.variance

    # Denormalize (move y_std and y_mean to device)
    y_std = selector.y_std.to(device) if isinstance(selector.y_std, torch.Tensor) else torch.tensor(selector.y_std, device=device)
    y_mean = selector.y_mean.to(device) if isinstance(selector.y_mean, torch.Tensor) else torch.tensor(selector.y_mean, device=device)
    mean = mean_norm * y_std + y_mean
    std = torch.sqrt(var_norm) * y_std

    # Compute EI (for minimization)
    improvement = best_y - mean - xi
    std_safe = std + 1e-8

    normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
    Z = improvement / std_safe
    cdf = normal.cdf(Z)
    pdf = torch.exp(normal.log_prob(Z))

    ei = improvement * cdf + std_safe * pdf

    return ei.cpu().numpy()


def sample_around_latents(
    training_latents: torch.Tensor,
    n_samples: int = 1000,
    std: float = 0.5,
) -> torch.Tensor:
    """Sample points around training latents in 32D space.

    Args:
        training_latents: Training latent points (N, 32)
        n_samples: Number of samples to generate
        std: Standard deviation of Gaussian noise

    Returns:
        Sampled points (n_samples, 32)
    """
    n_train = training_latents.shape[0]
    latent_dim = training_latents.shape[1]

    # Randomly select base points
    indices = torch.randint(0, n_train, (n_samples,))
    base_points = training_latents[indices]

    # Add Gaussian noise
    noise = torch.randn_like(base_points) * std
    samples = base_points + noise

    return samples


def visualize_ei_landscape(
    vae,
    exemplar_selector,
    exemplar_emb: torch.Tensor,
    training_latents: torch.Tensor,
    training_errors: torch.Tensor,
    optimized_latent: torch.Tensor,
    reembedded_latent: torch.Tensor,
    best_y: float,
    iteration: int,
    output_dir: str,
    n_samples: int = 1000,
    grid_resolution: int = 200,
    sample_std: float = 0.5,
    device: str = "cuda",
    figsize: tuple = (12, 10),
):
    """Visualize EI landscape with UMAP projection.

    Args:
        vae: Trained VAE model
        exemplar_selector: ExemplarSelector with trained GP
        exemplar_emb: Fixed exemplar embedding (768,)
        training_latents: Training latent points (N, 32)
        training_errors: Error rates for training points (N,)
        optimized_latent: Optimized latent from EI optimization (32,)
        reembedded_latent: Re-embedded latent from cycle consistency (32,)
        best_y: Best observed error rate
        iteration: Current iteration number
        output_dir: Directory to save visualization
        n_samples: Number of samples for EI landscape
        grid_resolution: Resolution of interpolation grid
        sample_std: Standard deviation for sampling around training points
        device: Device to use
        figsize: Figure size
    """
    if umap is None:
        warnings.warn("umap-learn not installed. Skipping EI landscape visualization.")
        return

    print(f"  Generating EI landscape visualization (iteration {iteration})...")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to tensors if needed
    if isinstance(training_latents, np.ndarray):
        training_latents = torch.from_numpy(training_latents).float()
    if isinstance(training_errors, np.ndarray):
        training_errors = torch.from_numpy(training_errors).float()
    if isinstance(optimized_latent, np.ndarray):
        optimized_latent = torch.from_numpy(optimized_latent).float()
    if isinstance(reembedded_latent, np.ndarray):
        reembedded_latent = torch.from_numpy(reembedded_latent).float()

    # Ensure proper shapes and move to CPU
    if optimized_latent.dim() == 2:
        optimized_latent = optimized_latent.squeeze(0)
    if reembedded_latent.dim() == 2:
        reembedded_latent = reembedded_latent.squeeze(0)

    # Move all latents to CPU for UMAP
    training_latents = training_latents.cpu()
    optimized_latent = optimized_latent.cpu()
    reembedded_latent = reembedded_latent.cpu()

    # 1. Sample points around training latents (on CPU)
    samples = sample_around_latents(training_latents, n_samples, sample_std)

    # 2. Compute EI for all sampled points
    ei_values = compute_ei_batch(
        samples, vae, exemplar_selector, exemplar_emb.cpu(), best_y, device=device
    )

    # 3. Combine all points for UMAP (all on CPU now)
    all_latents = torch.cat([
        training_latents,
        samples,
        optimized_latent.unsqueeze(0),
        reembedded_latent.unsqueeze(0),
    ], dim=0)

    # 4. Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    all_2d = reducer.fit_transform(all_latents.cpu().numpy())

    # Split back
    n_train = training_latents.shape[0]
    train_2d = all_2d[:n_train]
    samples_2d = all_2d[n_train:n_train + n_samples]
    optimized_2d = all_2d[-2]
    reembedded_2d = all_2d[-1]

    # 5. Create interpolation grid
    x_min, x_max = samples_2d[:, 0].min(), samples_2d[:, 0].max()
    y_min, y_max = samples_2d[:, 1].min(), samples_2d[:, 1].max()

    # Add padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    grid_x, grid_y = np.mgrid[
        x_min:x_max:complex(grid_resolution),
        y_min:y_max:complex(grid_resolution)
    ]

    # 6. Interpolate EI values using RBF (smooth thin plate spline)
    rbf = RBFInterpolator(samples_2d, ei_values, kernel='thin_plate_spline', smoothing=0.001)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_ei = rbf(grid_points).reshape(grid_x.shape)

    # 7. Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # EI heatmap (HyLO style: YlOrRd with alpha)
    im = ax.imshow(
        grid_ei.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        cmap="YlOrRd",
        alpha=0.7,
    )

    # EI contour lines with labels
    contour_levels = np.linspace(np.nanmin(grid_ei), np.nanmax(grid_ei), 15)
    cs = ax.contour(grid_x, grid_y, grid_ei, levels=contour_levels,
                    colors='darkred', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.4f')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Expected Improvement (EI)")

    # Training points (colored by error rate)
    scatter = ax.scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=training_errors.cpu().numpy(),
        cmap="RdYlGn_r",
        s=80,
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label="Training points",
    )
    cbar2 = plt.colorbar(scatter, ax=ax, label="Error Rate", location="left", pad=0.1)

    # Optimized latent (star)
    ax.scatter(
        optimized_2d[0],
        optimized_2d[1],
        marker="*",
        s=400,
        c="red",
        edgecolors="white",
        linewidths=2,
        zorder=10,
        label="Optimized latent",
    )

    # Re-embedded latent (cross)
    ax.scatter(
        reembedded_2d[0],
        reembedded_2d[1],
        marker="X",
        s=300,
        c="blue",
        edgecolors="white",
        linewidths=2,
        zorder=10,
        label="Re-embedded latent",
    )

    # Arrow for cycle consistency
    ax.annotate(
        "",
        xy=(reembedded_2d[0], reembedded_2d[1]),
        xytext=(optimized_2d[0], optimized_2d[1]),
        arrowprops=dict(
            arrowstyle="->",
            color="magenta",
            lw=2,
            connectionstyle="arc3,rad=0.1",
        ),
        zorder=9,
    )

    # Calculate cycle consistency using cosine similarity (L2 doesn't make sense in embedding space)
    cosine_sim = torch.nn.functional.cosine_similarity(
        optimized_latent.unsqueeze(0),
        reembedded_latent.unsqueeze(0)
    ).item()

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title(
        f"EI Landscape (Iteration {iteration})\n"
        f"Best error: {best_y:.4f} | Cycle consistency: cosine = {cosine_sim:.4f}"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Save
    save_path = output_path / f"ei_landscape_iter_{iteration:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved to {save_path}")


def get_training_latents(
    vae,
    exemplar_selector,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract training latents from ExemplarSelector.

    The ExemplarSelector stores X_train as (N, 1536) where first 768 dims
    are instruction embeddings. We need to encode these through VAE to get
    the 32D latents.

    Args:
        vae: Trained VAE model
        exemplar_selector: ExemplarSelector with X_train and y_train
        device: Device to use

    Returns:
        Tuple of (training_latents, training_errors)
        - training_latents: (N, 32) VAE latents
        - training_errors: (N,) error rates
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Get instruction embeddings (first 768 dims of X_train)
    X_train = exemplar_selector.X_train  # (N, 1536)
    inst_emb = X_train[:, :768]  # (N, 768)

    # Move VAE to device and encode
    vae = vae.to(device)
    vae.eval()
    with torch.no_grad():
        latents = vae.get_latent(inst_emb.to(device))

    return latents.cpu(), exemplar_selector.y_train.cpu()
