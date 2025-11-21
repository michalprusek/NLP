import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv('prompt_dataset.csv')

# Calculate mean accuracy
mean_accuracy = df['accuracy'].mean()
print(f"Mean accuracy across all prompts: {mean_accuracy:.6f}")

# Calculate baseline MAE (always predict mean)
baseline_predictions = np.full(len(df), mean_accuracy)
baseline_mae = mean_absolute_error(df['accuracy'], baseline_predictions)

print(f"\nBaseline MAE (always predict mean): {baseline_mae:.6f}")
print(f"Baseline MAE in percentage points: {baseline_mae*100:.3f}%")

# Calculate std for context
std_accuracy = df['accuracy'].std()
print(f"\nStandard deviation of accuracy: {std_accuracy:.6f}")
print(f"Coefficient of variation: {(std_accuracy/mean_accuracy)*100:.2f}%")
