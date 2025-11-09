#!/usr/bin/env python3
"""
Visualization of prompt optimization trajectories using embeddings and UMAP.

This script loads optimization results (ProTeGi or OPRO), embeds prompts using
a local sentence transformer model, reduces dimensionality with UMAP, and creates
interactive 2D/3D visualizations.

Generates 4 HTML files (when dimensions='both'):
- 2D scatter plot colored by accuracy (RdYlGn heatmap)
- 2D scatter plot colored by iteration (Viridis colorscale)
- 3D scatter plot colored by accuracy (RdYlGn heatmap)
- 3D scatter plot colored by iteration (Viridis colorscale)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm


def load_optimization_results(json_path: str) -> Tuple[List[str], List[float], List[int]]:
    """
    Load prompts, scores, and iterations from optimization JSON.

    Args:
        json_path: Path to optimization results JSON

    Returns:
        Tuple of (prompts, scores, iterations)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts = []
    scores = []
    iterations = []

    for entry in data['history']:
        prompts.append(entry['prompt'])
        scores.append(entry['score'])
        iterations.append(entry['iteration'])

    print(f"Loaded {len(prompts)} prompts from {data['method'].upper()} optimization")

    # Handle different JSON formats
    if 'model' in data:
        print(f"Model: {data['model']}")
    elif 'task_model' in data:
        print(f"Task model: {data['task_model']}")
        if 'meta_model' in data and data['meta_model'] != data['task_model']:
            print(f"Meta model: {data['meta_model']}")

    print(f"Iterations: 0-{max(iterations)}")
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")

    return prompts, scores, iterations


def embed_prompts(
    prompts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Create embeddings for prompts using sentence transformers.

    Args:
        prompts: List of prompt strings
        model_name: HuggingFace model name for embeddings
            - 'all-MiniLM-L6-v2': Fast, 384-dim (default)
            - 'all-mpnet-base-v2': Better quality, 768-dim
            - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Embeddings array of shape (n_prompts, embedding_dim)
    """
    print(f"\nGenerating embeddings with {model_name}...")
    model = SentenceTransformer(model_name, device=device)

    # Encode with progress bar
    embeddings = model.encode(
        prompts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def reduce_dimensionality(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP.

    Args:
        embeddings: Input embeddings
        n_components: Output dimensions (2 or 3)
        n_neighbors: UMAP n_neighbors (controls local vs global structure)
            - Smaller (5-10): Focus on local structure
            - Larger (15-50): Preserve more global structure
        min_dist: Minimum distance between points in low-D space
            - Smaller (0.0-0.1): Tighter clusters
            - Larger (0.1-0.5): More spread out
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        random_state: Random seed for reproducibility

    Returns:
        Reduced embeddings of shape (n_prompts, n_components)
    """
    print(f"\nReducing to {n_components}D using UMAP...")
    print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )

    reduced = reducer.fit_transform(embeddings)
    print(f"Reduced shape: {reduced.shape}")

    return reduced


def create_2d_visualization(
    coords_2d: np.ndarray,
    scores: List[float],
    iterations: List[int],
    prompts: List[str],
    output_path: str
):
    """
    Create interactive 2D scatter plot with accuracy heatmap.

    Args:
        coords_2d: 2D coordinates from UMAP
        scores: Accuracy scores for each prompt
        iterations: Iteration number for each prompt
        prompts: Original prompt texts (for hover info)
        output_path: Path to save HTML file
    """
    df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'accuracy': scores,
        'iteration': iterations,
        'prompt': prompts
    })

    # Create figure with plotly express
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='accuracy',
        color_continuous_scale='RdYlGn',  # Red (low) -> Yellow -> Green (high)
        hover_data={
            'prompt': True,
            'accuracy': ':.4f',
            'iteration': True,
            'x': False,
            'y': False
        },
        title='Prompt Optimization Trajectory (2D)',
        labels={'accuracy': 'Accuracy'}
    )

    # Update layout
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(
        width=1000,
        height=800,
        font=dict(size=12),
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        hovermode='closest'
    )

    # Add iteration annotations for selected points (first and last of each iteration)
    iteration_groups = df.groupby('iteration')
    for iter_num in sorted(df['iteration'].unique()):
        iter_data = iteration_groups.get_group(iter_num)
        # Mark first point of each iteration
        first_point = iter_data.iloc[0]
        fig.add_annotation(
            x=first_point['x'],
            y=first_point['y'],
            text=f"I{iter_num}",
            showarrow=False,
            font=dict(size=8, color='black'),
            bgcolor='white',
            opacity=0.7
        )

    # Save
    fig.write_html(output_path)
    print(f"\n2D visualization saved to: {output_path}")


def create_3d_visualization(
    coords_3d: np.ndarray,
    scores: List[float],
    iterations: List[int],
    prompts: List[str],
    output_path: str
):
    """
    Create interactive 3D scatter plot with accuracy heatmap.

    Args:
        coords_3d: 3D coordinates from UMAP
        scores: Accuracy scores for each prompt
        iterations: Iteration number for each prompt
        prompts: Original prompt texts (for hover info)
        output_path: Path to save HTML file
    """
    df = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'accuracy': scores,
        'iteration': iterations,
        'prompt': prompts
    })

    # Create 3D scatter
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='accuracy',
        color_continuous_scale='RdYlGn',
        hover_data={
            'prompt': True,
            'accuracy': ':.4f',
            'iteration': True,
            'x': False,
            'y': False,
            'z': False
        },
        title='Prompt Optimization Trajectory (3D)',
        labels={'accuracy': 'Accuracy'}
    )

    # Update layout
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
    fig.update_layout(
        width=1000,
        height=800,
        font=dict(size=12),
        scene=dict(
            xaxis_title='UMAP Component 1',
            yaxis_title='UMAP Component 2',
            zaxis_title='UMAP Component 3'
        ),
        hovermode='closest'
    )

    # Save
    fig.write_html(output_path)
    print(f"3D visualization saved to: {output_path}")


def create_2d_visualization_by_iteration(
    coords_2d: np.ndarray,
    scores: List[float],
    iterations: List[int],
    prompts: List[str],
    output_path: str
):
    """
    Create interactive 2D scatter plot colored by iteration.

    Args:
        coords_2d: 2D coordinates from UMAP
        scores: Accuracy scores for each prompt
        iterations: Iteration number for each prompt
        prompts: Original prompt texts (for hover info)
        output_path: Path to save HTML file
    """
    df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'accuracy': scores,
        'iteration': iterations,
        'prompt': prompts
    })

    # Create figure with discrete color scale for iterations
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='iteration',
        color_continuous_scale='Viridis',  # Purple (early) -> Green -> Yellow (late)
        hover_data={
            'prompt': True,
            'accuracy': ':.4f',
            'iteration': True,
            'x': False,
            'y': False
        },
        title='Prompt Optimization Trajectory by Iteration (2D)',
        labels={'iteration': 'Iteration'}
    )

    # Update layout
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(
        width=1000,
        height=800,
        font=dict(size=12),
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        hovermode='closest'
    )

    # Add iteration annotations for selected points
    iteration_groups = df.groupby('iteration')
    for iter_num in sorted(df['iteration'].unique()):
        iter_data = iteration_groups.get_group(iter_num)
        # Mark first point of each iteration
        first_point = iter_data.iloc[0]
        fig.add_annotation(
            x=first_point['x'],
            y=first_point['y'],
            text=f"I{iter_num}",
            showarrow=False,
            font=dict(size=8, color='black'),
            bgcolor='white',
            opacity=0.7
        )

    # Save
    fig.write_html(output_path)
    print(f"2D visualization (by iteration) saved to: {output_path}")


def create_3d_visualization_by_iteration(
    coords_3d: np.ndarray,
    scores: List[float],
    iterations: List[int],
    prompts: List[str],
    output_path: str
):
    """
    Create interactive 3D scatter plot colored by iteration.

    Args:
        coords_3d: 3D coordinates from UMAP
        scores: Accuracy scores for each prompt
        iterations: Iteration number for each prompt
        prompts: Original prompt texts (for hover info)
        output_path: Path to save HTML file
    """
    df = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'accuracy': scores,
        'iteration': iterations,
        'prompt': prompts
    })

    # Create 3D scatter with discrete color scale
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='iteration',
        color_continuous_scale='Viridis',
        hover_data={
            'prompt': True,
            'accuracy': ':.4f',
            'iteration': True,
            'x': False,
            'y': False,
            'z': False
        },
        title='Prompt Optimization Trajectory by Iteration (3D)',
        labels={'iteration': 'Iteration'}
    )

    # Update layout
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
    fig.update_layout(
        width=1000,
        height=800,
        font=dict(size=12),
        scene=dict(
            xaxis_title='UMAP Component 1',
            yaxis_title='UMAP Component 2',
            zaxis_title='UMAP Component 3'
        ),
        hovermode='closest'
    )

    # Save
    fig.write_html(output_path)
    print(f"3D visualization (by iteration) saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize prompt optimization using embeddings and UMAP'
    )
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to optimization results JSON'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for embedding model'
    )
    parser.add_argument(
        '--dimensions',
        type=str,
        default='both',
        choices=['2d', '3d', 'both'],
        help='Dimensionality for visualization (default: both)'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter (default: 15)'
    )
    parser.add_argument(
        '--min-dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter (default: 0.1)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean', 'manhattan'],
        help='Distance metric for UMAP (default: cosine)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/output',
        help='Output directory for HTML visualizations'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename for outputs
    base_name = Path(args.json_path).stem

    # Load data
    prompts, scores, iterations = load_optimization_results(args.json_path)

    # Generate embeddings
    embeddings = embed_prompts(
        prompts,
        model_name=args.embedding_model,
        device=args.device
    )

    # Create visualizations
    if args.dimensions in ['2d', 'both']:
        coords_2d = reduce_dimensionality(
            embeddings,
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric
        )
        # Create both accuracy and iteration-colored versions
        create_2d_visualization(
            coords_2d,
            scores,
            iterations,
            prompts,
            output_path=str(output_dir / f"{base_name}_2d_accuracy.html")
        )
        create_2d_visualization_by_iteration(
            coords_2d,
            scores,
            iterations,
            prompts,
            output_path=str(output_dir / f"{base_name}_2d_iteration.html")
        )

    if args.dimensions in ['3d', 'both']:
        coords_3d = reduce_dimensionality(
            embeddings,
            n_components=3,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric
        )
        # Create both accuracy and iteration-colored versions
        create_3d_visualization(
            coords_3d,
            scores,
            iterations,
            prompts,
            output_path=str(output_dir / f"{base_name}_3d_accuracy.html")
        )
        create_3d_visualization_by_iteration(
            coords_3d,
            scores,
            iterations,
            prompts,
            output_path=str(output_dir / f"{base_name}_3d_iteration.html")
        )

    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
