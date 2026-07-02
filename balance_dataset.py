import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the encoded dataset
df = pd.read_csv("india_encoded.csv")

# Handle any lingering NaNs that median couldn't fill (e.g. all-NaN columns)
df.fillna(0, inplace=True)

# Create risk_category binning
def get_risk_category(score):
    if score < 50:
        return 0  # High Risk
    elif score <= 65:
        return 1  # Medium Risk
    else:
        return 2  # Low Risk

df['risk_category'] = df['final_feasibility_score'].apply(get_risk_category)

# Show counts before SMOTE
counts_before = df['risk_category'].value_counts().to_dict()
print("Counts before SMOTE:")
for cat, count in counts_before.items():
    label = "High Risk (0)" if cat == 0 else "Medium Risk (1)" if cat == 1 else "Low Risk (2)"
    print(f"  {label}: {count}")

# Define SMOTE sampling strategy
# User requested "at least 200 samples each"
strategy = {}
for cat in [0, 1, 2]:
    current_count = counts_before.get(cat, 0)
    strategy[cat] = max(200, current_count)

# It's possible SMOTE fails if a class has fewer than k_neighbors (default 5).
# If High Risk has only 2 rows, k_neighbors must be set to 1.
k_neighbors = 5
min_samples = df['risk_category'].value_counts().min()
if min_samples <= 5:
    k_neighbors = max(1, min_samples - 1)

smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=42)

# Separate features and target
X = df.drop(columns=['risk_category'])
y = df['risk_category']

# Apply SMOTE
X_res, y_res = smote.fit_resample(X, y)

# Re-attach risk_category
X_res['risk_category'] = y_res

# Show counts after SMOTE
counts_after = X_res['risk_category'].value_counts().to_dict()
print("\nCounts after SMOTE:")
for cat, count in counts_after.items():
    label = "High Risk (0)" if cat == 0 else "Medium Risk (1)" if cat == 1 else "Low Risk (2)"
    print(f"  {label}: {count}")

# Drop risk_category helper column
df_balanced = X_res.drop(columns=['risk_category'])

# Print final shape
print(f"\nFinal shape: {df_balanced.shape}")

# Save the balanced dataset
df_balanced.to_csv("india_balanced.csv", index=False)
print("Saved balanced dataset to india_balanced.csv")
