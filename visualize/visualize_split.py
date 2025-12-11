"""
Visualize GP learned embeddings with Expected Improvement heatmap.

Single plot showing:
- Scatter of prompts colored by accuracy
- EI as contour/heatmap (interpolated)
- Global EI maximum in latent space

Uses structural-aware feature extractor (separate instruction/exemplar encoders).
"""
import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs
from hbbops.run_hbbops import load_instructions, load_exemplars


def compute_ei_batch(hbbops, X_norm, vmin_b):
    """Compute Expected Improvement for normalized inputs."""
    hbbops.gp_model.eval()
    hbbops.likelihood.eval()

    with torch.no_grad():
        pred = hbbops.likelihood(hbbops.gp_model(X_norm))
        means = pred.mean.cpu().numpy() * hbbops.y_std.item() + hbbops.y_mean.item()
        stds = pred.stddev.cpu().numpy() * hbbops.y_std.item()

    ei_values = np.zeros(len(means))
    for i, (mean, std) in enumerate(zip(means, stds)):
        if std <= 0:
            ei_values[i] = max(vmin_b - mean, 0)
        else:
            z = (vmin_b - mean) / std
            ei_values[i] = (vmin_b - mean) * norm.cdf(z) + std * norm.pdf(z)

    return ei_values, means, stds


def main():
    base_dir = Path(__file__).parent.parent  # /home/prusek/NLP
    data_dir = base_dir / "hbbops" / "data"
    datasets_dir = base_dir / "datasets" / "hbbops"
    results_dir = Path(__file__).parent  # visualize/
    full_grid_path = datasets_dir / "full_grid_combined.jsonl"

    print("Loading data...")
    instructions = load_instructions(str(datasets_dir / "instructions_25.txt"))
    exemplars = load_exemplars(str(datasets_dir / "examples_25.txt"))

    with open(data_dir / "validation.json", 'r') as f:
        validation_data = json.load(f)
    n_valid = len(validation_data)

    # Initialize HbBoPs
    hbbops = HbBoPs(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=validation_data,
        llm_evaluator=lambda p, d: 0.0,
        device="auto"
    )

    # Load results
    results = []
    with open(full_grid_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    id_to_idx = {(p.instruction_id, p.exemplar_id): idx for idx, p in enumerate(hbbops.prompts)}
    valid_results = [r for r in results if (r['instruction_id'], r['exemplar_id']) in id_to_idx]
    print(f"Valid results: {len(valid_results)}")

    # Train GP on ALL data
    print("Training Deep Kernel GP (structural-aware)...")
    hbbops.design_data = []
    for res in valid_results:
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = hbbops.prompts[p_idx]
        inst_emb, ex_emb = hbbops.embed_prompt(prompt)
        hbbops.design_data.append((p_idx, inst_emb, ex_emb, res['error_rate'], n_valid))

    hbbops.train_gp(fidelity=n_valid, min_observations=10)
    hbbops.gp_model.eval()

    # Extract 10D latent features
    print("Extracting latent features...")
    dk_features = []
    bert_embeddings = []
    accuracies = []

    for res in valid_results:
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = hbbops.prompts[p_idx]

        inst_emb, ex_emb = hbbops.embed_prompt(prompt)
        bert_embeddings.append(np.concatenate([inst_emb, ex_emb]))

        inst_tensor = torch.tensor(inst_emb, dtype=torch.float32, device=hbbops.device).unsqueeze(0)
        ex_tensor = torch.tensor(ex_emb, dtype=torch.float32, device=hbbops.device).unsqueeze(0)
        X_input = torch.cat([inst_tensor, ex_tensor], dim=1)
        denominator = hbbops.X_max - hbbops.X_min
        denominator[denominator == 0] = 1.0
        X_norm = (X_input - hbbops.X_min) / denominator

        with torch.no_grad():
            inst_norm = X_norm[:, :768]
            ex_norm = X_norm[:, 768:]
            latent = hbbops.gp_model.feature_extractor(inst_norm, ex_norm)
            dk_features.append(latent.cpu().numpy().squeeze())

        accuracies.append(1.0 - res['error_rate'])

    X_dk = np.array(dk_features)  # (625, 10)
    X_bert = np.array(bert_embeddings)
    accuracies = np.array(accuracies)

    # Compute EI
    print("Computing Expected Improvement...")
    vmin_b = min(res['error_rate'] for res in valid_results)
    X_all = torch.tensor(X_bert, dtype=torch.float32, device=hbbops.device)
    denominator = hbbops.X_max - hbbops.X_min
    denominator[denominator == 0] = 1.0
    X_all_norm = (X_all - hbbops.X_min) / denominator
    ei_values, gp_means, gp_stds = compute_ei_batch(hbbops, X_all_norm, vmin_b)

    # UMAP projection
    print("Computing UMAP...")
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(X_dk)

    # Interpolate EI to grid using RBF
    print("Interpolating EI surface...")
    rbf = RBFInterpolator(embedding_2d, ei_values, kernel='thin_plate_spline', smoothing=0.001)

    # Create dense grid
    margin = 0.5
    x_min, x_max = embedding_2d[:, 0].min() - margin, embedding_2d[:, 0].max() + margin
    y_min, y_max = embedding_2d[:, 1].min() - margin, embedding_2d[:, 1].max() + margin

    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    ei_grid = rbf(grid_points).reshape(xx.shape)

    # Find global maximum of EI in the interpolated surface
    print("Finding global EI maximum...")

    def neg_ei(xy):
        return -rbf(xy.reshape(1, -1))[0]

    # Multi-start optimization
    best_ei = -np.inf
    best_xy = None

    # Start from top EI points + random starts
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

    # Find nearest prompt to the EI maximum
    distances = np.linalg.norm(embedding_2d - best_xy, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_prompt = hbbops.prompts[id_to_idx[(valid_results[nearest_idx]['instruction_id'],
                                                valid_results[nearest_idx]['exemplar_id'])]]

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
    cbar_acc = plt.colorbar(scatter, ax=ax, label='Accuracy', shrink=0.6, pad=0.02)
    cbar_ei = plt.colorbar(im, ax=ax, label='Expected Improvement', shrink=0.6, pad=0.08)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'Structural Deep Kernel GP: Accuracy (points) + EI (heatmap)\n'
                 f'N={len(valid_results)} | Best acc: {accuracies[best_actual_idx]:.2%} | '
                 f'Max EI: {best_ei:.4f} at ({best_xy[0]:.2f}, {best_xy[1]:.2f})',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = results_dir / "full_gp_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")

    # Stats
    print(f"\n{'='*60}")
    print(f"Max EI (interpolated): {best_ei:.6f}")
    print(f"  Location: UMAP ({best_xy[0]:.3f}, {best_xy[1]:.3f})")
    print(f"  Nearest prompt: ({nearest_prompt.instruction_id}, {nearest_prompt.exemplar_id})")
    print(f"  Nearest prompt accuracy: {accuracies[nearest_idx]:.4f}")
    print(f"  Distance to nearest: {distances[nearest_idx]:.4f}")
    print(f"\nBest actual: ({valid_results[best_actual_idx]['instruction_id']}, "
          f"{valid_results[best_actual_idx]['exemplar_id']}) acc={accuracies[best_actual_idx]:.4f}")


if __name__ == "__main__":
    main()
