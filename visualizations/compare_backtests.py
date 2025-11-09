#!/usr/bin/env python3
"""
Compare backtest results across different OPRO runs.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path


def load_backtest_results(csv_path):
    """Load backtest results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def create_comparison_plot(results_dict, output_path):
    """Create comparison visualization for multiple backtests."""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'RMSE Comparison',
            'MAE Comparison',
            'R² Comparison',
            'Prediction Quality Over Time'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, (name, df) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]

        # Aggregate by iteration
        grouped = df.groupby('iteration').agg({
            'abs_error': ['mean', 'std'],
            'error': 'mean',
            'actual_accuracy': 'mean'
        }).reset_index()

        rmse = grouped[('abs_error', 'mean')].apply(lambda x: x)
        mae = grouped[('abs_error', 'mean')]
        iterations = grouped['iteration']

        # 1. RMSE
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=rmse,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                legendgroup=name
            ),
            row=1, col=1
        )

        # 2. MAE
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=mae,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=False,
                legendgroup=name
            ),
            row=1, col=2
        )

        # 3. Plot error variance instead of R² computation
        # (Original R² loop wasn't using its results)
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=grouped[('abs_error', 'std')],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=False,
                legendgroup=name
            ),
            row=2, col=1
        )

        # 4. Cumulative performance
        cumulative_rmse = [
            df[df['iteration'] <= iterations.iloc[i-1]]['abs_error'].mean()
            for i in range(1, len(iterations) + 1)
        ]

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=cumulative_rmse,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=False,
                legendgroup=name
            ),
            row=2, col=2
        )

    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=2)

    fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="Std of Errors", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative MAE", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=900,
        width=1400,
        title_text="Backtest Comparison: Ensemble Regressor Across Models",
        font=dict(size=12)
    )

    fig.write_html(output_path)
    print(f"Comparison plot saved to: {output_path}")


def print_comparison_table(results_dict):
    """Print comparison table of key metrics."""
    print("\n" + "="*80)
    print("BACKTEST COMPARISON TABLE")
    print("="*80)

    headers = ["Metric"] + list(results_dict.keys())
    print(f"{'Metric':<30} " + " ".join([f"{name:>15}" for name in results_dict.keys()]))
    print("-" * 80)

    # Compute metrics for each dataset
    metrics = {}
    for name, df in results_dict.items():
        metrics[name] = {
            'Mean MAE': df['abs_error'].mean(),
            'Median MAE': df['abs_error'].median(),
            'Std MAE': df['abs_error'].std(),
            'Mean Error': df['error'].mean(),
            '% within ±0.05': (df['abs_error'] <= 0.05).mean() * 100,
            '% within ±0.10': (df['abs_error'] <= 0.10).mean() * 100,
            'Worst Error': df['abs_error'].max(),
            'Best Error': df['abs_error'].min()
        }

    # Print table
    for metric_name in metrics[list(results_dict.keys())[0]].keys():
        row = f"{metric_name:<30}"
        for name in results_dict.keys():
            value = metrics[name][metric_name]
            if '%' in metric_name:
                row += f" {value:>14.1f}%"
            else:
                row += f" {value:>15.4f}"
        print(row)

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare backtest results'
    )
    parser.add_argument(
        'csv_paths',
        nargs='+',
        help='Paths to backtest CSV files'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Labels for each backtest (default: use filenames)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations/output/backtest_comparison.html',
        help='Output path for comparison plot'
    )

    args = parser.parse_args()

    # Load results
    results_dict = {}
    for i, csv_path in enumerate(args.csv_paths):
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            # Extract meaningful name from path
            path = Path(csv_path)
            if 'gpt' in path.stem.lower():
                label = 'GPT-3.5'
            elif 'qwen' in path.stem.lower():
                label = 'Qwen2.5-7B'
            else:
                label = path.stem.replace('_backtest_details', '')

        results_dict[label] = load_backtest_results(csv_path)
        print(f"Loaded {label}: {len(results_dict[label])} predictions")

    # Create comparison plot
    create_comparison_plot(results_dict, args.output)

    # Print comparison table
    print_comparison_table(results_dict)


if __name__ == '__main__':
    main()
