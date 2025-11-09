#!/usr/bin/env python3
"""
Backtest surrogate models for prompt optimization.

Tests ensemble regressor (CatBoost + RandomForest + NGBoost) by simulating
the actual OPRO optimization process: train on iterations 0..i, predict i+1.
"""

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities
from utils import load_optimization_results, embed_prompts


class EnsembleRegressor:
    """
    Ensemble of CatBoost, Random Forest, and NGBoost.

    Provides both point predictions and uncertainty estimates.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.models = {
            'catboost': CatBoostRegressor(
                iterations=300,
                depth=4,
                learning_rate=0.05,
                loss_function='RMSE',
                random_seed=42,
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'ngboost': NGBRegressor(
                n_estimators=300,
                learning_rate=0.01,
                minibatch_frac=0.5,
                Dist=Normal,
                verbose=False,
                random_state=42
            )
        }
        self.weights = None
        self.is_fitted = False

    def fit(self, X, y):
        """Fit all models in the ensemble."""
        if self.verbose:
            print(f"  Training ensemble on {len(X)} samples...")

        # Train each model
        for name, model in self.models.items():
            if self.verbose:
                print(f"    Training {name}...")
            model.fit(X, y)

        # Compute weights based on training performance (simple equal weights for now)
        self.weights = {'catboost': 0.4, 'random_forest': 0.3, 'ngboost': 0.3}
        self.is_fitted = True

    def predict(self, X):
        """Predict using weighted average of all models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get predictions from all models
        predictions = np.array([
            model.predict(X) for model in self.models.values()
        ])

        # Weighted average
        weights = np.array([self.weights[name] for name in self.models.keys()])
        return np.average(predictions, axis=0, weights=weights)

    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimate from ensemble variance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get predictions from all models
        predictions = np.array([
            model.predict(X) for model in self.models.values()
        ])

        # Return ensemble mean and std
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


def backtest_sequential(
    embeddings: np.ndarray,
    scores: np.ndarray,
    iterations: np.ndarray,
    prompts: List[str],
    model_class=EnsembleRegressor,
    verbose=True
) -> Dict:
    """
    Sequential backtest: train on iterations 0..i, predict i+1.

    Args:
        embeddings: Prompt embeddings
        scores: Actual accuracy scores
        iterations: Iteration numbers
        prompts: Original prompts (for logging)
        model_class: Model class to use for prediction
        verbose: Print progress

    Returns:
        Dictionary with backtest results
    """
    max_iteration = iterations.max()
    results = {
        'iteration': [],
        'train_size': [],
        'test_size': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'predictions': [],
        'actuals': [],
        'test_indices': []
    }

    print("\n" + "="*60)
    print("SEQUENTIAL BACKTEST")
    print("="*60)
    print(f"Strategy: Train on iter 0..i → Predict iter i+1")
    print(f"Iterations to test: 0 → {max_iteration-1}")
    print("="*60 + "\n")

    for i in range(max_iteration):
        # Training data: all iterations up to and including i
        train_mask = iterations <= i
        train_X = embeddings[train_mask]
        train_y = scores[train_mask]

        # Test data: iteration i+1
        test_mask = iterations == (i + 1)
        test_X = embeddings[test_mask]
        test_y = scores[test_mask]

        if len(test_y) == 0:
            # No test data for this iteration
            continue

        if verbose:
            print(f"Iteration {i} → {i+1}:")
            print(f"  Train: {len(train_y)} samples (iters 0-{i})")
            print(f"  Test:  {len(test_y)} samples (iter {i+1})")

        # Train model
        model = model_class(verbose=False)
        model.fit(train_X, train_y)

        # Predict
        predictions = model.predict(test_X)

        # Metrics
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        if verbose:
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            print()

        # Store results
        results['iteration'].append(i + 1)
        results['train_size'].append(len(train_y))
        results['test_size'].append(len(test_y))
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['r2'].append(r2)
        results['predictions'].append(predictions)
        results['actuals'].append(test_y)
        results['test_indices'].append(np.where(test_mask)[0])

    return results


def plot_backtest_results(results: Dict, output_path: str):
    """Create comprehensive visualization of backtest results."""

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'RMSE per Iteration',
            'MAE per Iteration',
            'R² Score per Iteration',
            'Training Set Size Growth',
            'Actual vs Predicted (All Iterations)',
            'Prediction Errors Distribution'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'histogram'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    iterations = results['iteration']

    # Define metrics and their configurations
    metric_configs = [
        ('rmse', 'RMSE', 'red', 1, 1),
        ('mae', 'MAE', 'orange', 1, 2),
        ('r2', 'R²', 'green', 2, 1),
    ]

    # Add metric traces
    for metric_key, name, color, row, col in metric_configs:
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=results[metric_key],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ),
            row=row, col=col
        )

    # Add R² baseline
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # 4. Training size growth
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=results['train_size'],
            mode='lines+markers',
            name='Train Size',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0,100,200,0.2)'
        ),
        row=2, col=2
    )

    # 5. Actual vs Predicted scatter
    all_actuals = np.concatenate(results['actuals'])
    all_predictions = np.concatenate(results['predictions'])

    fig.add_trace(
        go.Scatter(
            x=all_actuals,
            y=all_predictions,
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=results['iteration'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.46, len=0.25, y=0.17, title='Iteration')
            )
        ),
        row=3, col=1
    )
    # Perfect prediction line
    min_val = min(all_actuals.min(), all_predictions.min())
    max_val = max(all_actuals.max(), all_predictions.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )

    # 6. Error distribution
    errors = all_predictions - all_actuals
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=30,
            name='Errors',
            marker=dict(color='purple', opacity=0.7)
        ),
        row=3, col=2
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=3, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)
    fig.update_xaxes(title_text="Actual Accuracy", row=3, col=1)
    fig.update_xaxes(title_text="Prediction Error", row=3, col=2)

    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=2, col=1)
    fig.update_yaxes(title_text="Training Samples", row=2, col=2)
    fig.update_yaxes(title_text="Predicted Accuracy", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)

    # Update layout
    fig.update_layout(
        height=1200,
        width=1400,
        title_text="Sequential Backtest Results: Ensemble Regressor",
        showlegend=False,
        font=dict(size=11)
    )

    fig.write_html(output_path)
    print(f"\nBacktest visualization saved to: {output_path}")


def print_summary_statistics(results: Dict):
    """Print summary statistics of backtest."""
    print("\n" + "="*60)
    print("BACKTEST SUMMARY STATISTICS")
    print("="*60)

    rmse_values = np.array(results['rmse'])
    mae_values = np.array(results['mae'])
    r2_values = np.array(results['r2'])

    print(f"\nRMSE across iterations:")
    print(f"  Mean:   {rmse_values.mean():.4f}")
    print(f"  Median: {np.median(rmse_values):.4f}")
    print(f"  Std:    {rmse_values.std():.4f}")
    print(f"  Min:    {rmse_values.min():.4f} (iteration {results['iteration'][rmse_values.argmin()]})")
    print(f"  Max:    {rmse_values.max():.4f} (iteration {results['iteration'][rmse_values.argmax()]})")

    print(f"\nMAE across iterations:")
    print(f"  Mean:   {mae_values.mean():.4f}")
    print(f"  Median: {np.median(mae_values):.4f}")
    print(f"  Min:    {mae_values.min():.4f}")
    print(f"  Max:    {mae_values.max():.4f}")

    print(f"\nR² across iterations:")
    print(f"  Mean:   {r2_values.mean():.4f}")
    print(f"  Median: {np.median(r2_values):.4f}")
    print(f"  Min:    {r2_values.min():.4f}")
    print(f"  Max:    {r2_values.max():.4f}")

    # Check for improvement over time
    first_half = rmse_values[:len(rmse_values)//2]
    second_half = rmse_values[len(rmse_values)//2:]

    if len(first_half) > 0 and len(second_half) > 0:
        print(f"\nLearning curve:")
        print(f"  Early RMSE (first half):  {first_half.mean():.4f}")
        print(f"  Late RMSE (second half):  {second_half.mean():.4f}")
        improvement = (first_half.mean() - second_half.mean()) / first_half.mean() * 100
        if improvement > 0:
            print(f"  Improvement: {improvement:.1f}% (better with more data)")
        else:
            print(f"  Change: {improvement:.1f}% (degradation with more data)")


def save_detailed_results(results: Dict, prompts: List[str], output_path: str):
    """Save detailed per-prompt results to CSV."""
    rows = []

    for i, iter_num in enumerate(results['iteration']):
        test_indices = results['test_indices'][i]
        predictions = results['predictions'][i]
        actuals = results['actuals'][i]

        for j, idx in enumerate(test_indices):
            rows.append({
                'iteration': iter_num,
                'prompt': prompts[idx],
                'actual_accuracy': actuals[j],
                'predicted_accuracy': predictions[j],
                'error': predictions[j] - actuals[j],
                'abs_error': abs(predictions[j] - actuals[j]),
                'train_size': results['train_size'][i]
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Backtest surrogate models for prompt optimization'
    )
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to optimization results JSON'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-mpnet-base-v2',
        help='Sentence transformer model (default: all-mpnet-base-v2)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for embedding model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/output',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename
    base_name = Path(args.json_path).stem

    # Load data
    prompts, scores, iterations = load_optimization_results(args.json_path)
    scores_array = np.array(scores)
    iterations_array = np.array(iterations)

    # Generate embeddings
    embeddings = embed_prompts(
        prompts,
        model_name=args.embedding_model,
        device=args.device
    )

    # Run backtest
    results = backtest_sequential(
        embeddings,
        scores_array,
        iterations_array,
        prompts,
        model_class=EnsembleRegressor,
        verbose=True
    )

    # Print summary
    print_summary_statistics(results)

    # Save visualizations
    plot_backtest_results(
        results,
        output_path=str(output_dir / f"{base_name}_backtest_results.html")
    )

    # Save detailed CSV
    save_detailed_results(
        results,
        prompts,
        output_path=str(output_dir / f"{base_name}_backtest_details.csv")
    )

    print("\n✓ Backtest complete!")


if __name__ == '__main__':
    main()
