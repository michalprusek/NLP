#!/usr/bin/env python3
"""Incremental training analysis: Train regressor on iterations 0-i, predict iteration i+1."""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading datasets...")
df_data = pd.read_csv('prompt_dataset.csv')  # iteration, prompt, accuracy
df_embeddings = pd.read_csv('prompt_embeddings_all-mpnet-base-v2.csv')  # prompt, embedding

# Parse embeddings from JSON
print("Parsing embeddings...")
df_embeddings['embedding'] = df_embeddings['embedding'].apply(json.loads)

# Merge on prompt
print("Merging datasets...")
df = pd.merge(df_data, df_embeddings, on='prompt', how='inner')
print(f"Merged dataset: {len(df)} prompts")

# Convert embeddings to numpy array
X = np.array(df['embedding'].tolist())
y = df['accuracy'].values
iterations = df['iteration'].values

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Iteration range: {iterations.min()} - {iterations.max()}")

# Find unique iterations
unique_iterations = sorted(df['iteration'].unique())
print(f"Unique iterations: {len(unique_iterations)} ({unique_iterations[0]} to {unique_iterations[-1]})")

# Incremental training loop
mae_scores = []
train_sizes = []
test_sizes = []
iteration_numbers = []

print("\n" + "="*80)
print("INCREMENTAL TRAINING ANALYSIS")
print("="*80)

for i, train_iter in enumerate(unique_iterations[:-1]):  # Exclude last iteration (no next iter to test)
    test_iter = unique_iterations[i + 1]

    # Training data: all prompts from iteration 0 to train_iter (inclusive)
    train_mask = iterations <= train_iter
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Test data: all prompts from iteration test_iter
    test_mask = iterations == test_iter
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Skip if no test data
    if len(X_test) == 0:
        continue

    # Progress update
    print(f"\n[{i+1}/{len(unique_iterations)-1}] Processing iteration {train_iter} → {test_iter}...", flush=True)
    print(f"  Training on {len(X_train)} prompts...", end=' ', flush=True)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train regressor
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8
    )
    regressor.fit(X_train_scaled, y_train)
    print("✓", flush=True)

    # Predict on test set
    print(f"  Predicting on {len(X_test)} test prompts...", end=' ', flush=True)
    y_pred = regressor.predict(X_test_scaled)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print("✓", flush=True)

    # Store results
    mae_scores.append(mae)
    train_sizes.append(len(X_train))
    test_sizes.append(len(X_test))
    iteration_numbers.append(train_iter)

    print(f"  ➜ MAE: {mae:.6f} ({mae*100:.3f}% accuracy points)", flush=True)

print("="*80)

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: MAE vs Training Iteration
ax1.plot(iteration_numbers, mae_scores, 'o-', linewidth=2, markersize=6, color='#2E86AB')
ax1.set_xlabel('Training Iteration (predicting next iteration)', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
ax1.set_title('Incremental Learning: MAE of Accuracy Predictions', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(left=min(iteration_numbers)-1)

# Add trend line
z = np.polyfit(iteration_numbers, mae_scores, 1)
p = np.poly1d(z)
ax1.plot(iteration_numbers, p(iteration_numbers), "--", alpha=0.5, color='red',
         label=f'Trend: {z[0]:.2e}x + {z[1]:.4f}')
ax1.legend()

# Add statistics text
stats_text = (f"Min MAE: {min(mae_scores):.6f} (iter {iteration_numbers[np.argmin(mae_scores)]})\n"
              f"Max MAE: {max(mae_scores):.6f} (iter {iteration_numbers[np.argmax(mae_scores)]})\n"
              f"Mean MAE: {np.mean(mae_scores):.6f}\n"
              f"Std MAE: {np.std(mae_scores):.6f}")
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Training size vs MAE (to see if more data helps)
ax2.plot(train_sizes, mae_scores, 'o-', linewidth=2, markersize=6, color='#A23B72')
ax2.set_xlabel('Training Set Size (number of prompts)', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
ax2.set_title('MAE vs Training Set Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('incremental_prediction_mae.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Graph saved to: incremental_prediction_mae.png")

# Save results to CSV
results_df = pd.DataFrame({
    'train_iteration': iteration_numbers,
    'test_iteration': [unique_iterations[i+1] for i in range(len(iteration_numbers))],
    'train_size': train_sizes,
    'test_size': test_sizes,
    'mae': mae_scores
})
results_df.to_csv('incremental_prediction_results.csv', index=False)
print(f"✓ Results saved to: incremental_prediction_results.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total iterations analyzed: {len(mae_scores)}")
print(f"MAE range: {min(mae_scores):.6f} - {max(mae_scores):.6f}")
print(f"MAE trend slope: {z[0]:.2e} (negative = improving over time)")
print(f"Training size range: {min(train_sizes)} - {max(train_sizes)} prompts")
