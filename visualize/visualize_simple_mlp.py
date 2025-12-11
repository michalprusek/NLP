"""
Visualize GP with Simple MLP kernel (768d → 10d reduction).

Contrast with structural-aware version that uses separate instruction/exemplar encoders.
This version uses a single MLP: concat(inst, ex) [1536d] → 10d latent space.
"""
import json
import sys
import torch
import torch.nn as nn
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs, PromptEncoder, Prompt
from hbbops.run_hbbops import load_instructions, load_exemplars


class SimpleFeatureExtractor(nn.Module):
    """Simple MLP: 1536d (concat) → 10d latent space.

    No structural awareness - just dimensionality reduction.
    """
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 128, latent_dim: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1536) concatenated embeddings."""
        return self.mlp(x)


class SimpleDeepKernelGP(gpytorch.models.ExactGP):
    """GP with simple MLP kernel (no structural awareness)."""

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 feature_extractor: SimpleFeatureExtractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=10,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
        )
        self.feature_extractor = feature_extractor

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        latent = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(latent), self.covar_module(latent)
        )


def train_simple_gp(X: np.ndarray, y: np.ndarray, device: torch.device,
                    n_epochs: int = 200, lr: float = 0.01):
    """Train GP with simple MLP feature extractor."""
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Normalize
    X_min, X_max = X_tensor.min(dim=0).values, X_tensor.max(dim=0).values
    denom = X_max - X_min
    denom[denom == 0] = 1.0
    X_norm = (X_tensor - X_min) / denom

    y_mean, y_std = y_tensor.mean(), y_tensor.std()
    if y_std == 0:
        y_std = torch.tensor(1.0, device=device)
    y_norm = (y_tensor - y_mean) / y_std

    # Initialize
    feature_extractor = SimpleFeatureExtractor().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SimpleDeepKernelGP(X_norm, y_norm, likelihood, feature_extractor).to(device)

    # Train
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_norm)
        loss = -mll(output, y_norm)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    return model, likelihood, X_min, X_max, y_mean, y_std


def compute_ei_batch(model, likelihood, X_norm, y_mean, y_std, vmin_b):
    """Compute Expected Improvement for normalized inputs."""
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred = likelihood(model(X_norm))
        means = pred.mean.cpu().numpy() * y_std.item() + y_mean.item()
        stds = pred.stddev.cpu().numpy() * y_std.item()

    ei_values = np.zeros(len(means))
    for i, (mean, std) in enumerate(zip(means, stds)):
        if std <= 0:
            ei_values[i] = max(vmin_b - mean, 0)
        else:
            z = (vmin_b - mean) / std
            ei_values[i] = (vmin_b - mean) * norm.cdf(z) + std * norm.pdf(z)

    return ei_values, means, stds


def main():
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets" / "hbbops"
    data_dir = base_dir / "hbbops" / "data"
    results_dir = Path(__file__).parent
    full_grid_path = datasets_dir / "full_grid_combined.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    instructions = load_instructions(str(datasets_dir / "instructions_25.txt"))
    exemplars = load_exemplars(str(datasets_dir / "examples_25.txt"))

    with open(data_dir / "validation.json", 'r') as f:
        validation_data = json.load(f)

    # Create prompts
    prompts = []
    for i, inst in enumerate(instructions):
        for j, ex in enumerate(exemplars):
            prompts.append(Prompt(instruction=inst, exemplar=ex, instruction_id=i, exemplar_id=j))

    # Load ground truth results
    results = []
    with open(full_grid_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    id_to_idx = {(p.instruction_id, p.exemplar_id): idx for idx, p in enumerate(prompts)}
    valid_results = [r for r in results if (r['instruction_id'], r['exemplar_id']) in id_to_idx]
    print(f"Valid results: {len(valid_results)}")

    # Encode prompts
    print("Encoding prompts with GTR...")
    encoder = PromptEncoder()

    embeddings = []
    accuracies = []
    for res in valid_results:
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = prompts[p_idx]

        inst_emb = encoder.encode(prompt.instruction)
        ex_emb = encoder.encode(prompt.exemplar)
        # Concatenate: [inst_emb, ex_emb] = 1536d
        combined_emb = np.concatenate([inst_emb, ex_emb])
        embeddings.append(combined_emb)
        accuracies.append(1.0 - res['error_rate'])

    X = np.array(embeddings)  # (625, 1536)
    y = np.array([res['error_rate'] for res in valid_results])  # error rates for GP
    accuracies = np.array(accuracies)

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Train simple GP
    print("Training Simple MLP Deep Kernel GP...")
    model, likelihood, X_min, X_max, y_mean, y_std = train_simple_gp(X, y, device)

    # Extract 10D latent features
    print("Extracting latent features...")
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    denom = X_max - X_min
    denom[denom == 0] = 1.0
    X_norm = (X_tensor - X_min) / denom

    model.eval()
    with torch.no_grad():
        latent_features = model.feature_extractor(X_norm).cpu().numpy()

    print(f"Latent features shape: {latent_features.shape}")

    # Compute EI
    print("Computing Expected Improvement...")
    vmin_b = y.min()  # best error rate
    ei_values, gp_means, gp_stds = compute_ei_batch(model, likelihood, X_norm, y_mean, y_std, vmin_b)

    # UMAP projection
    print("Computing UMAP...")
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(latent_features)

    # Interpolate EI to grid
    print("Interpolating EI surface...")
    rbf = RBFInterpolator(embedding_2d, ei_values, kernel='thin_plate_spline', smoothing=0.001)

    margin = 0.5
    x_min, x_max = embedding_2d[:, 0].min() - margin, embedding_2d[:, 0].max() + margin
    y_min, y_max = embedding_2d[:, 1].min() - margin, embedding_2d[:, 1].max() + margin

    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    ei_grid = rbf(grid_points).reshape(xx.shape)

    # Find global maximum of EI
    print("Finding global EI maximum...")

    def neg_ei(xy):
        return -rbf(xy.reshape(1, -1))[0]

    best_ei = -np.inf
    best_xy = None

    top_ei_idx = np.argsort(ei_values)[-10:]
    start_points = list(embedding_2d[top_ei_idx])
    start_points += [np.array([np.random.uniform(x_min, x_max),
                               np.random.uniform(y_min, y_max)]) for _ in range(20)]

    for start in start_points:
        result = minimize(neg_ei, start, method='L-BFGS-B',
                          bounds=[(x_min, x_max), (y_min, y_max)])
        if -result.fun > best_ei:
            best_ei = -result.fun
            best_xy = result.x

    print(f"Global EI maximum: {best_ei:.6f} at UMAP ({best_xy[0]:.3f}, {best_xy[1]:.3f})")

    # Best actual prompt
    best_actual_idx = np.argmax(accuracies)

    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # EI heatmap
    im = ax.imshow(ei_grid, extent=[x_min, x_max, y_min, y_max],
                   origin='lower', cmap='YlOrRd', alpha=0.7, aspect='auto')

    # EI contours
    contour_levels = np.linspace(ei_grid.min(), ei_grid.max(), 15)
    cs = ax.contour(xx, yy, ei_grid, levels=contour_levels, colors='darkred',
                    linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.3f')

    # Scatter prompts colored by accuracy
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                         c=accuracies, cmap='viridis', s=60, alpha=0.9,
                         edgecolors='white', linewidths=0.5,
                         vmin=0, vmax=1)

    # Mark global EI maximum
    ax.scatter(best_xy[0], best_xy[1], c='cyan', s=400, marker='*',
               edgecolors='black', linewidths=2, zorder=10,
               label=f'Max EI = {best_ei:.4f}')

    # Mark best actual accuracy
    ax.scatter(embedding_2d[best_actual_idx, 0], embedding_2d[best_actual_idx, 1],
               c='lime', s=300, marker='*', edgecolors='black', linewidths=2, zorder=10,
               label=f'Best acc = {accuracies[best_actual_idx]:.2%}')

    # Colorbars
    plt.colorbar(scatter, ax=ax, label='Accuracy', shrink=0.6, pad=0.02)
    plt.colorbar(im, ax=ax, label='Expected Improvement', shrink=0.6, pad=0.08)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'Simple MLP Deep Kernel GP: Accuracy (points) + EI (heatmap)\n'
                 f'N={len(valid_results)} | Best acc: {accuracies[best_actual_idx]:.2%} | '
                 f'Max EI: {best_ei:.4f} at ({best_xy[0]:.2f}, {best_xy[1]:.2f})\n'
                 f'Architecture: 1536d → 128d → 10d (no structural awareness)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = results_dir / "simple_mlp_gp_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")

    # Stats
    print(f"\n{'='*60}")
    print(f"Simple MLP Architecture: 1536 → 128 → 10")
    print(f"Max EI (interpolated): {best_ei:.6f}")
    print(f"  Location: UMAP ({best_xy[0]:.3f}, {best_xy[1]:.3f})")
    print(f"Best actual: ({valid_results[best_actual_idx]['instruction_id']}, "
          f"{valid_results[best_actual_idx]['exemplar_id']}) acc={accuracies[best_actual_idx]:.4f}")


if __name__ == "__main__":
    main()
