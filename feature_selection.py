import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("india_balanced.csv")
target_col = "final_feasibility_score"

X = df.drop(columns=[target_col])
y = df[target_col]

# Train XGBoost
print("Training XGBoost model...")
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Calculate SHAP feature importance
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'SHAP': shap_importance
}).sort_values(by='SHAP', ascending=False)

print("\n--- SHAP Feature Importance (Ranked) ---")
for idx, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['SHAP']:.6f}")

# Remove features < 0.005
kept_by_shap = importance_df[importance_df['SHAP'] >= 0.005]['Feature'].tolist()
print(f"\nFeatures kept after SHAP threshold (>= 0.005): {len(kept_by_shap)}")

X_reduced = X[kept_by_shap]

# Check correlation > 0.90
corr_matrix = X_reduced.corr().abs()
shap_dict = dict(zip(importance_df['Feature'], importance_df['SHAP']))

to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.90:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            # Ignore if already marked for drop
            if col1 in to_drop or col2 in to_drop:
                continue
            if shap_dict[col1] < shap_dict[col2]:
                to_drop.add(col1)
            else:
                to_drop.add(col2)

print(f"\nFeatures dropped due to high correlation (>0.90): {list(to_drop)}")
final_features = [col for col in kept_by_shap if col not in to_drop]

print(f"\nRemaining columns count (features only): {len(final_features)}")
print(f"Final list of kept columns:\n{final_features}")

# Save reduced dataset and column names
final_columns_to_keep = final_features + [target_col]
df_final = df[final_columns_to_keep]
df_final.to_csv("india_reduced.csv", index=False)

with open("selected_features.txt", "w") as f:
    for col in final_features:
        f.write(col + "\n")
        
print("\nSaved india_reduced.csv and selected_features.txt")
