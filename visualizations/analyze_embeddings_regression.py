#!/usr/bin/env python3
"""
Ridge regression analysis of prompt embeddings vs accuracy.

Fits a regularized linear model (Ridge regression) predicting accuracy from
embedding features. Uses cross-validation to select optimal regularization parameter.
Analyzes studentized residuals to identify prompts that performed unexpectedly
well/poorly given their semantic embedding.

Note: Ridge regularization is necessary because we have p >> n (768 features, ~80 samples).
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# Import shared utilities
from utils import load_optimization_results, embed_prompts


def compute_studentized_residuals_ridge(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute studentized residuals for Ridge regression.

    For Ridge regression, we use approximate studentization based on
    the effective degrees of freedom and leverage approximation.

    Args:
        y_true: True values (n,)
        y_pred: Predicted values (n,)
        X: Design matrix (n, p) without intercept, centered
        alpha: Ridge regularization parameter

    Returns:
        Tuple of (raw_residuals, standardized_residuals, studentized_residuals)
    """
    n = len(y_true)
    p = X.shape[1]

    # Raw residuals
    residuals = y_true - y_pred

    # Compute hat matrix H = X(X'X + alpha*I)^{-1}X' for Ridge
    # and extract leverage values
    XtX = X.T @ X
    XtX_ridge = XtX + alpha * np.eye(p)

    try:
        H = X @ np.linalg.solve(XtX_ridge, X.T)
        h = np.diag(H)  # Leverage values
        df_eff = np.trace(H)  # Effective degrees of freedom
    except np.linalg.LinAlgError as e:
        # Fallback if matrix is singular
        print(f"WARNING: Hat matrix computation failed ({e}). Using approximate leverage values.", file=sys.stderr)
        print(f"WARNING: Studentized residuals may be less accurate with this approximation.", file=sys.stderr)
        h = np.ones(n) * (p / n)
        df_eff = p

    # Compute mean squared error with effective degrees of freedom
    rss = np.sum(residuals ** 2)
    mse = rss / max(n - df_eff, 1) if n > df_eff else rss / max(n - p, 1)

    # Standardized residuals
    standardized_residuals = residuals / np.sqrt(mse) if mse > 1e-10 else residuals

    # Studentized residuals (externally studentized)
    # Account for leverage: Var(e_i) = sigma^2 * (1 - h_i)
    h_safe = np.clip(h, 0, 0.99)  # Prevent division by zero
    studentized_residuals = residuals / np.sqrt(mse * (1 - h_safe)) if mse > 1e-10 else standardized_residuals

    return residuals, standardized_residuals, studentized_residuals


def plot_studentized_residuals(
    studentized_residuals: np.ndarray,
    fitted_values: np.ndarray,
    prompts: List[str],
    scores: List[float],
    iterations: List[int],
    output_path: str
):
    """
    Create diagnostic plots for studentized residuals.

    Args:
        studentized_residuals: Studentized residuals
        fitted_values: Fitted accuracy values
        prompts: Original prompts (for hover)
        scores: True accuracy scores
        iterations: Iteration numbers
        output_path: Path to save HTML file
    """
    df = pd.DataFrame({
        'fitted': fitted_values,
        'studentized_residual': studentized_residuals,
        'actual_accuracy': scores,
        'iteration': iterations,
        'prompt': prompts,
        'index': range(len(prompts))
    })

    # Identify outliers (|studentized residual| > 2)
    df['outlier'] = np.abs(df['studentized_residual']) > 2
    df['outlier_severe'] = np.abs(df['studentized_residual']) > 3

    # Create residual plot
    fig = px.scatter(
        df,
        x='fitted',
        y='studentized_residual',
        color='iteration',
        color_continuous_scale='Viridis',
        hover_data={
            'prompt': True,
            'actual_accuracy': ':.4f',
            'fitted': ':.4f',
            'studentized_residual': ':.3f',
            'iteration': True,
            'index': True
        },
        title='Studentized Residuals vs Fitted Values',
        labels={
            'fitted': 'Fitted Accuracy',
            'studentized_residual': 'Studentized Residual',
            'iteration': 'Iteration'
        }
    )

    # Add reference lines at 0, ±2, ±3
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    fig.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.3,
                  annotation_text="±2σ threshold", annotation_position="right")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", opacity=0.3)
    fig.add_hline(y=3, line_dash="dash", line_color="darkred", opacity=0.5,
                  annotation_text="±3σ severe", annotation_position="right")
    fig.add_hline(y=-3, line_dash="dash", line_color="darkred", opacity=0.5)

    # Update layout
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
    fig.update_layout(
        width=1200,
        height=700,
        font=dict(size=12),
        hovermode='closest',
        showlegend=True
    )

    # Save
    try:
        fig.write_html(output_path)
        print(f"\nStudentized residuals plot saved to: {output_path}")
    except (IOError, PermissionError) as e:
        print(f"Error writing studentized residuals plot to {output_path}: {e}", file=sys.stderr)
        print("Warning: Could not save studentized residuals plot", file=sys.stderr)

    # Print outlier statistics
    n_outliers = df['outlier'].sum()
    n_severe = df['outlier_severe'].sum()
    print(f"\nOutlier analysis:")
    print(f"  Moderate outliers (|r*| > 2): {n_outliers} ({100*n_outliers/len(df):.1f}%)")
    print(f"  Severe outliers (|r*| > 3): {n_severe} ({100*n_severe/len(df):.1f}%)")

    if n_severe > 0:
        print(f"\nSevere outliers:")
        severe_df = df[df['outlier_severe']].sort_values('studentized_residual',
                                                          key=lambda x: abs(x),
                                                          ascending=False)
        for _, row in severe_df.iterrows():
            direction = "overperformed" if row['studentized_residual'] > 0 else "underperformed"
            print(f"  Index {row['index']}: {direction} (r*={row['studentized_residual']:.2f})")
            print(f"    Actual: {row['actual_accuracy']:.4f}, Fitted: {row['fitted']:.4f}")
            print(f"    Prompt: {row['prompt'][:80]}...")


def plot_qq_plot(
    studentized_residuals: np.ndarray,
    output_path: str
):
    """
    Create Q-Q plot to check normality of studentized residuals.

    Args:
        studentized_residuals: Studentized residuals
        output_path: Path to save HTML file
    """
    # Theoretical quantiles (normal distribution)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(studentized_residuals)))

    # Sort residuals
    sorted_residuals = np.sort(studentized_residuals)

    # Create Q-Q plot
    fig = go.Figure()

    # Points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sorted_residuals,
        mode='markers',
        name='Residuals',
        marker=dict(size=6, color='blue', opacity=0.6)
    ))

    # Reference line (y = x)
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Normal',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Q-Q Plot: Studentized Residuals vs Normal Distribution',
        xaxis_title='Theoretical Quantiles (Normal)',
        yaxis_title='Sample Quantiles (Studentized Residuals)',
        width=800,
        height=700,
        font=dict(size=12),
        showlegend=True
    )

    try:
        fig.write_html(output_path)
        print(f"Q-Q plot saved to: {output_path}")
    except (IOError, PermissionError) as e:
        print(f"Error writing Q-Q plot to {output_path}: {e}", file=sys.stderr)
        print("Warning: Could not save Q-Q plot", file=sys.stderr)

    # Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(studentized_residuals)
    print(f"\nShapiro-Wilk normality test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: Residuals are NOT normally distributed (p < 0.05)")
    else:
        print(f"  Result: Residuals appear normally distributed (p >= 0.05)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze prompt embeddings with linear regression'
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
        help='Output directory for plots'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename
    base_name = Path(args.json_path).stem

    # Load data
    try:
        prompts, scores, iterations = load_optimization_results(args.json_path)
        scores_array = np.array(scores)
    except FileNotFoundError:
        print(f"Error: File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when reading: {args.json_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading optimization results: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate embeddings
    try:
        embeddings = embed_prompts(
            prompts,
            model_name=args.embedding_model,
            device=args.device
        )
    except Exception as e:
        print(f"Error generating embeddings with model '{args.embedding_model}': {e}", file=sys.stderr)
        print("Possible issues: model not found, network error during download, or device incompatibility", file=sys.stderr)
        sys.exit(1)

    # Split data into 80% train, 20% test
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT (80/20)")
    print("="*60)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embeddings,
        scores_array,
        np.arange(len(scores_array)),
        test_size=0.2,
        random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Total samples: {len(scores_array)}")
    print(f"  Training samples: {len(X_train)} (80%)")
    print(f"  Test samples: {len(X_test)} (20%)")
    print(f"  Features: {embeddings.shape[1]}")
    print(f"  Train ratio p/n: {embeddings.shape[1]/len(X_train):.1f}")

    # Fit Ridge regression with cross-validation for alpha selection ON TRAINING DATA ONLY
    print(f"\n{'='*60}")
    print("RIDGE REGRESSION WITH CV (on training data)")
    print("="*60)

    # Try range of alpha values
    alphas = np.logspace(-2, 4, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train, y_train)

    print(f"  Best alpha (selected by CV): {ridge_cv.alpha_:.4f}")

    # Use best model
    model = ridge_cv

    # Predictions on BOTH train and test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training performance
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    # Test performance
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Cross-validated R² on training data
    cv_scores = cross_val_score(
        Ridge(alpha=ridge_cv.alpha_),
        X_train,
        y_train,
        cv=5,
        scoring='r2'
    )

    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nTraining set (n={len(X_train)}):")
    print(f"  R² score: {r2_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")
    print(f"  MAE: {mae_train:.4f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\nTest set (n={len(X_test)}):")
    print(f"  R² score: {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  MSE: {mse_test:.4f}")
    print(f"  MAE: {mae_test:.4f}")

    # Overfitting check
    overfit_r2 = r2_train - r2_test
    overfit_rmse = rmse_test - rmse_train
    print(f"\nOverfitting analysis:")
    print(f"  R² gap (train - test): {overfit_r2:+.4f}")
    print(f"  RMSE gap (test - train): {overfit_rmse:+.4f}")

    if overfit_r2 > 0.2:
        print(f"  ⚠️  Severe overfitting detected (R² gap > 0.2)")
    elif overfit_r2 > 0.1:
        print(f"  ⚠️  Moderate overfitting detected (R² gap > 0.1)")
    else:
        print(f"  ✓ No significant overfitting")

    # Print test set predictions
    print(f"\n{'='*60}")
    print("TEST SET PREDICTIONS (sample)")
    print("="*60)
    test_results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_test_pred,
        'error': y_test - y_test_pred,
        'abs_error': np.abs(y_test - y_test_pred)
    }).sort_values('abs_error', ascending=False)

    print("\nWorst 5 predictions on test set:")
    for i, (idx, row) in enumerate(test_results.head(5).iterrows()):
        print(f"  {i+1}. Actual: {row['actual']:.4f}, Predicted: {row['predicted']:.4f}, Error: {row['error']:+.4f}")

    print("\nBest 5 predictions on test set:")
    for i, (idx, row) in enumerate(test_results.tail(5).iterrows()):
        print(f"  {i+1}. Actual: {row['actual']:.4f}, Predicted: {row['predicted']:.4f}, Error: {row['error']:+.4f}")

    # For residual analysis, use FULL dataset predictions
    y_pred_full = model.predict(embeddings)

    # Compute studentized residuals for Ridge (on full dataset)
    print("\n" + "="*60)
    print("STUDENTIZED RESIDUALS ANALYSIS (full dataset)")
    print("="*60)
    raw_residuals, standardized_residuals, studentized_residuals = compute_studentized_residuals_ridge(
        scores_array, y_pred_full, embeddings, ridge_cv.alpha_
    )

    # Create residual plot
    plot_studentized_residuals(
        studentized_residuals,
        y_pred_full,
        prompts,
        scores,
        iterations,
        output_path=str(output_dir / f"{base_name}_studentized_residuals.html")
    )

    # Create Q-Q plot
    plot_qq_plot(
        studentized_residuals,
        output_path=str(output_dir / f"{base_name}_qq_plot.html")
    )

    # Save detailed results with train/test indicator
    dataset_split = np.array(['train'] * len(scores_array))
    dataset_split[idx_test] = 'test'

    results_df = pd.DataFrame({
        'prompt': prompts,
        'actual_accuracy': scores,
        'predicted_accuracy': y_pred_full,
        'raw_residual': raw_residuals,
        'standardized_residual': standardized_residuals,
        'studentized_residual': studentized_residuals,
        'iteration': iterations,
        'dataset': dataset_split
    })

    csv_path = output_dir / f"{base_name}_regression_results.csv"
    try:
        results_df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
    except (IOError, PermissionError) as e:
        print(f"Error writing CSV to {csv_path}: {e}", file=sys.stderr)
        print("Warning: Could not save detailed results to CSV", file=sys.stderr)

    print("\n✓ Regression analysis complete!")


if __name__ == '__main__':
    main()
