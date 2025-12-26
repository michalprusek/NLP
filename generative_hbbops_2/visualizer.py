"""
Visualization functions for HyLO.

Creates 3 separate PNG files:
1. After initial GP training (latent space + EI heatmap)
2. During optimization (trajectory in latent space)
3. After inversion verification (re-embedded point)
"""
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import norm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings


class HyLOVisualizer:
    """Creates visualizations for HyLO optimization stages.

    All plots show:
    - 2D UMAP projection of 10D latent space
    - Scatter of prompts colored by accuracy
    - EI heatmap (interpolated)
    - Key points marked with stars
    """

    def __init__(
        self,
        output_dir: str,
        dpi: int = 300
    ):
        """
        Args:
            output_dir: Directory for saving visualizations
            dpi: DPI for saved images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Cached UMAP reducer (fitted once, reused for consistency)
        self._reducer: Optional[umap.UMAP] = None
        self._embedding_2d: Optional[np.ndarray] = None
        self._latent_features: Optional[np.ndarray] = None

    def _fit_umap(
        self,
        latent_features: np.ndarray,
        random_state: int = 42
    ) -> Tuple[umap.UMAP, np.ndarray]:
        """Fit UMAP on latent features.

        UMAP is fitted once and reused for all visualizations
        to ensure consistent coordinates.

        Args:
            latent_features: (N, 10) latent features from GP
            random_state: Random seed for reproducibility

        Returns:
            (fitted_reducer, 2D_embeddings)
        """
        if self._reducer is not None and np.array_equal(latent_features, self._latent_features):
            return self._reducer, self._embedding_2d

        self._reducer = umap.UMAP(random_state=random_state)
        self._embedding_2d = self._reducer.fit_transform(latent_features)
        self._latent_features = latent_features.copy()

        return self._reducer, self._embedding_2d

    def _interpolate_ei(
        self,
        embedding_2d: np.ndarray,
        ei_values: np.ndarray,
        grid_res: int = 200,
        margin: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate EI values to dense grid using RBF.

        Args:
            embedding_2d: (N, 2) UMAP coordinates
            ei_values: (N,) EI values at each point
            grid_res: Resolution of interpolation grid
            margin: Extra margin around data bounds

        Returns:
            (xx_grid, yy_grid, ei_grid)
        """
        # Bounds with margin
        x_min = embedding_2d[:, 0].min() - margin
        x_max = embedding_2d[:, 0].max() + margin
        y_min = embedding_2d[:, 1].min() - margin
        y_max = embedding_2d[:, 1].max() + margin

        # Create grid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # RBF interpolation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rbf = RBFInterpolator(
                embedding_2d, ei_values,
                kernel='thin_plate_spline',
                smoothing=0.001
            )
            ei_grid = rbf(grid_points).reshape(xx.shape)

        return xx, yy, ei_grid

    def _find_ei_maximum(
        self,
        embedding_2d: np.ndarray,
        ei_values: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Find global maximum of EI in the interpolated surface.

        Uses multi-start L-BFGS-B optimization.

        Args:
            embedding_2d: (N, 2) UMAP coordinates
            ei_values: (N,) EI values
            xx, yy: Grid meshes

        Returns:
            (best_xy, best_ei)
        """
        x_min, x_max = xx.min(), xx.max()
        y_min, y_max = yy.min(), yy.max()

        # RBF for optimization
        rbf = RBFInterpolator(
            embedding_2d, ei_values,
            kernel='thin_plate_spline',
            smoothing=0.001
        )

        def neg_ei(xy):
            return -rbf(xy.reshape(1, -1))[0]

        # Multi-start optimization
        best_ei = -np.inf
        best_xy = None

        # Start from top EI points + random starts
        top_idx = np.argsort(ei_values)[-10:]
        start_points = list(embedding_2d[top_idx])
        start_points += [
            np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
            for _ in range(20)
        ]

        for start in start_points:
            try:
                result = minimize(
                    neg_ei, start, method='L-BFGS-B',
                    bounds=[(x_min, x_max), (y_min, y_max)]
                )
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_xy = result.x
            except:
                continue

        return best_xy, best_ei

    def _create_base_plot(
        self,
        embedding_2d: np.ndarray,
        accuracies: np.ndarray,
        ei_values: np.ndarray,
        vmin_b: float,
        title: str,
        training_indices: np.ndarray = None
    ) -> Tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray]:
        """Create base plot with EI heatmap and accuracy scatter.

        Args:
            embedding_2d: (N, 2) UMAP coordinates
            accuracies: (N,) accuracy values (1 - error_rate)
            ei_values: (N,) EI values
            vmin_b: Best observed error rate
            title: Plot title
            training_indices: If provided, only plot these points

        Returns:
            (figure, axes, xx_grid, yy_grid)
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Interpolate EI (using all points for smooth heatmap)
        xx, yy, ei_grid = self._interpolate_ei(embedding_2d, ei_values)

        # EI heatmap
        im = ax.imshow(
            ei_grid,
            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            origin='lower',
            cmap='YlOrRd',
            alpha=0.7,
            aspect='auto'
        )

        # EI contours
        contour_levels = np.linspace(ei_grid.min(), ei_grid.max(), 15)
        cs = ax.contour(
            xx, yy, ei_grid,
            levels=contour_levels,
            colors='darkred',
            linewidths=0.5,
            alpha=0.6
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.3f')

        # Select points to plot
        if training_indices is not None:
            plot_emb = embedding_2d[training_indices]
            plot_acc = accuracies[training_indices]
            point_size = 200  # Larger for fewer points
        else:
            plot_emb = embedding_2d
            plot_acc = accuracies
            point_size = 60

        # Scatter prompts colored by accuracy
        scatter = ax.scatter(
            plot_emb[:, 0], plot_emb[:, 1],
            c=plot_acc, cmap='viridis',
            s=point_size, alpha=0.9,
            edgecolors='black', linewidths=1.5,
            vmin=0, vmax=1
        )

        # Colorbars
        plt.colorbar(scatter, ax=ax, label='Accuracy', shrink=0.6, pad=0.02)
        plt.colorbar(im, ax=ax, label='Expected Improvement', shrink=0.6, pad=0.08)

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(title, fontsize=13)

        return fig, ax, xx, yy

    def plot_initial_gp(
        self,
        latent_features: np.ndarray,
        error_rates: np.ndarray,
        ei_values: np.ndarray,
        vmin_b: float,
        training_indices: np.ndarray = None,
        filename: str = "hylo_1_initial_gp.png"
    ) -> Path:
        """Graph 1: After initial GP training.

        Shows:
        - Scatter of training prompts colored by accuracy
        - EI heatmap (interpolated)
        - Cyan star at maximum EI location
        - Lime star at best actual prompt

        Args:
            latent_features: (N, 10) latent features from training data
            error_rates: (N,) error rates
            ei_values: (N,) EI values
            vmin_b: Best observed error rate
            training_indices: If provided, only plot these points
            filename: Output filename

        Returns:
            Path to saved file
        """
        accuracies = 1.0 - error_rates

        # Fit UMAP
        reducer, embedding_2d = self._fit_umap(latent_features)

        # Find EI maximum
        xx, yy, ei_grid = self._interpolate_ei(embedding_2d, ei_values)
        max_xy, max_ei = self._find_ei_maximum(embedding_2d, ei_values, xx, yy)

        # Determine which indices to use for best accuracy
        if training_indices is not None:
            plot_acc = accuracies[training_indices]
            best_local_idx = np.argmax(plot_acc)
            best_idx = training_indices[best_local_idx]
            n_points = len(training_indices)
        else:
            best_idx = np.argmax(accuracies)
            n_points = len(latent_features)

        # Create plot
        fig, ax, _, _ = self._create_base_plot(
            embedding_2d, accuracies, ei_values, vmin_b,
            f"HyLO Phase 1: Initial GP Training\n"
            f"N={n_points} training samples | Best acc: {accuracies[best_idx]:.2%} | "
            f"Max EI: {max_ei:.4f}",
            training_indices=training_indices
        )

        # Mark EI maximum
        ax.scatter(
            max_xy[0], max_xy[1],
            c='cyan', s=400, marker='*',
            edgecolors='black', linewidths=2, zorder=10,
            label=f'Max EI = {max_ei:.4f}'
        )

        # Mark best actual accuracy (from training samples)
        ax.scatter(
            embedding_2d[best_idx, 0], embedding_2d[best_idx, 1],
            c='lime', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=10,
            label=f'Best acc = {accuracies[best_idx]:.2%}'
        )

        ax.legend(loc='upper right', fontsize=10)

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization 1 to {output_path}")
        return output_path

    def plot_optimization_trajectory(
        self,
        latent_features: np.ndarray,
        error_rates: np.ndarray,
        ei_values: np.ndarray,
        trajectory: List[Dict],
        gp_trainer: 'GPTrainer',
        vmin_b: float,
        training_indices: np.ndarray = None,
        filename: str = "hylo_2_optimization.png"
    ) -> Path:
        """Graph 2: During embedding optimization.

        Shows:
        - Same base plot as Graph 1
        - Trajectory of optimization steps (line with arrows)
        - Green circle at start point
        - Red star at end point
        - Markers where exemplar changes (coordinate descent only)

        Args:
            latent_features: (N, 10) training latent features
            error_rates: (N,) error rates
            ei_values: (N,) EI values
            trajectory: Optimization trajectory from optimizer
            gp_trainer: GPTrainer instance for computing latent features
            vmin_b: Best observed error rate
            training_indices: If provided, only plot these points
            filename: Output filename

        Returns:
            Path to saved file
        """
        import torch

        accuracies = 1.0 - error_rates

        # Use cached UMAP
        if self._reducer is None:
            self._fit_umap(latent_features)

        embedding_2d = self._embedding_2d

        # Extract trajectory points in latent space
        traj_latents = []
        traj_ei = []
        exemplar_changes = []

        prev_exemplar = None
        for i, step in enumerate(trajectory):
            if 'instruction_emb' in step and 'exemplar_emb' in step:
                inst_emb = torch.tensor(step['instruction_emb'], dtype=torch.float32)
                ex_emb = torch.tensor(step['exemplar_emb'], dtype=torch.float32)

                with torch.no_grad():
                    latent = gp_trainer.get_latent_features(inst_emb, ex_emb)
                    traj_latents.append(latent.cpu().numpy())

                traj_ei.append(step.get('ei', 0))

                # Track exemplar changes
                curr_exemplar = step.get('exemplar_idx')
                if prev_exemplar is not None and curr_exemplar != prev_exemplar:
                    exemplar_changes.append(len(traj_latents) - 1)
                prev_exemplar = curr_exemplar

        if len(traj_latents) == 0:
            print("Warning: No trajectory data available for visualization")
            return self.plot_initial_gp(
                latent_features, error_rates, ei_values, vmin_b,
                training_indices=training_indices,
                filename=filename
            )

        traj_latents = np.array(traj_latents)

        # Project trajectory to 2D using fitted UMAP
        traj_2d = self._reducer.transform(traj_latents)

        # Create plot
        fig, ax, _, _ = self._create_base_plot(
            embedding_2d, accuracies, ei_values, vmin_b,
            f"HyLO Phase 2: Embedding Optimization\n"
            f"Trajectory: {len(traj_2d)} steps | Final EI: {traj_ei[-1]:.4f}",
            training_indices=training_indices
        )

        # Plot trajectory line
        ax.plot(
            traj_2d[:, 0], traj_2d[:, 1],
            'b-', linewidth=2, alpha=0.7, zorder=5
        )

        # Add arrows showing direction
        for i in range(0, len(traj_2d) - 1, max(1, len(traj_2d) // 10)):
            dx = traj_2d[i + 1, 0] - traj_2d[i, 0]
            dy = traj_2d[i + 1, 1] - traj_2d[i, 1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax.annotate(
                    '', xy=(traj_2d[i + 1, 0], traj_2d[i + 1, 1]),
                    xytext=(traj_2d[i, 0], traj_2d[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    zorder=6
                )

        # Mark start point
        ax.scatter(
            traj_2d[0, 0], traj_2d[0, 1],
            c='green', s=200, marker='o',
            edgecolors='black', linewidths=2, zorder=10,
            label=f'Start (EI={traj_ei[0]:.4f})'
        )

        # Mark end point
        ax.scatter(
            traj_2d[-1, 0], traj_2d[-1, 1],
            c='red', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=10,
            label=f'End (EI={traj_ei[-1]:.4f})'
        )

        # Mark exemplar changes
        for idx in exemplar_changes:
            ax.scatter(
                traj_2d[idx, 0], traj_2d[idx, 1],
                c='orange', s=100, marker='D',
                edgecolors='black', linewidths=1, zorder=9
            )

        if exemplar_changes:
            ax.scatter([], [], c='orange', s=100, marker='D',
                       edgecolors='black', label='Exemplar switch')

        ax.legend(loc='upper right', fontsize=10)

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization 2 to {output_path}")
        return output_path

    def plot_inversion_verification(
        self,
        latent_features: np.ndarray,
        error_rates: np.ndarray,
        ei_values: np.ndarray,
        optimized_latent: np.ndarray,
        reembedded_latent: np.ndarray,
        vmin_b: float,
        cosine_similarity: float,
        training_indices: np.ndarray = None,
        filename: str = "hylo_3_verification.png"
    ) -> Path:
        """Graph 3: After re-embedding inverted text.

        Shows:
        - Same base plot as Graph 1
        - Red star at optimized embedding location
        - Blue diamond at re-embedded location (after Vec2Text)
        - Arrow showing displacement
        - Annotation with cosine similarity

        Args:
            latent_features: (N, 10) training latent features
            error_rates: (N,) error rates
            ei_values: (N,) EI values
            optimized_latent: (10,) latent features of optimized embedding
            reembedded_latent: (10,) latent features after Vec2Text + re-encode
            vmin_b: Best observed error rate
            cosine_similarity: Cosine sim between optimized and re-embedded
            training_indices: If provided, only plot these points
            filename: Output filename

        Returns:
            Path to saved file
        """
        accuracies = 1.0 - error_rates

        # Use cached UMAP
        if self._reducer is None:
            self._fit_umap(latent_features)

        embedding_2d = self._embedding_2d

        # Project optimized and re-embedded points
        opt_2d = self._reducer.transform(optimized_latent.reshape(1, -1))[0]
        reemb_2d = self._reducer.transform(reembedded_latent.reshape(1, -1))[0]

        # Find EI maximum
        xx, yy, ei_grid = self._interpolate_ei(embedding_2d, ei_values)
        max_xy, max_ei = self._find_ei_maximum(embedding_2d, ei_values, xx, yy)

        # Create plot
        fig, ax, _, _ = self._create_base_plot(
            embedding_2d, accuracies, ei_values, vmin_b,
            f"HyLO Phase 3: Inversion Verification\n"
            f"Cosine Similarity: {cosine_similarity:.4f} | Max EI: {max_ei:.4f}",
            training_indices=training_indices
        )

        # Mark EI maximum
        ax.scatter(
            max_xy[0], max_xy[1],
            c='cyan', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=8,
            label=f'Max EI = {max_ei:.4f}'
        )

        # Mark optimized point
        ax.scatter(
            opt_2d[0], opt_2d[1],
            c='red', s=400, marker='*',
            edgecolors='black', linewidths=2, zorder=10,
            label='Optimized embedding'
        )

        # Mark re-embedded point
        ax.scatter(
            reemb_2d[0], reemb_2d[1],
            c='blue', s=250, marker='D',
            edgecolors='black', linewidths=2, zorder=10,
            label='Re-embedded (Vec2Text)'
        )

        # Arrow showing displacement
        displacement = np.linalg.norm(opt_2d - reemb_2d)
        ax.annotate(
            '', xy=(reemb_2d[0], reemb_2d[1]),
            xytext=(opt_2d[0], opt_2d[1]),
            arrowprops=dict(
                arrowstyle='->',
                color='purple',
                lw=2,
                connectionstyle='arc3,rad=0.1'
            ),
            zorder=9
        )

        # Add annotation with similarity
        mid_x = (opt_2d[0] + reemb_2d[0]) / 2
        mid_y = (opt_2d[1] + reemb_2d[1]) / 2
        ax.annotate(
            f'cos={cosine_similarity:.3f}\nd={displacement:.2f}',
            (mid_x, mid_y),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.legend(loc='upper right', fontsize=10)

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization 3 to {output_path}")
        return output_path
