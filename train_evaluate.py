import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# 1. Load dataset
df = pd.read_csv("india_reduced.csv")
target_col = "final_feasibility_score"

# 2. Separate Target from Features
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Original feature columns count: {X.shape[1]}")

# 3. Scale all feature columns using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA (95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA components created: {X_pca.shape[1]}")

# Helper function to train and evaluate XGBoost
def evaluate_xgboost(X_data, y_data, description):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # Using defaults for quick comparison, or could use same params as before
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n--- {description} ---")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")
    
    return r2, rmse, mae

# 5. Evaluate PCA
r2_pca, rmse_pca, mae_pca = evaluate_xgboost(X_pca, y, "PCA Components (95% Variance)")

# 6. Evaluate Non-PCA (Original Reduced Features)
# Note: Should use scaled or unscaled? XGBoost doesn't strictly need scaling, but we'll use unscaled as typical, or scaled. Let's use the standard unscaled X.
r2_orig, rmse_orig, mae_orig = evaluate_xgboost(X, y, "Non-PCA Original Reduced Features")

# 7. Compare and save the better version
print("\n--- Comparison ---")
if r2_orig > r2_pca:
    print("Non-PCA (Original Reduced Features) performed better.")
    df.to_csv("india_final.csv", index=False)
    print("Saved Non-PCA dataset as india_final.csv")
else:
    print("PCA performed better.")
    # Create a DataFrame for PCA components
    pca_cols = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca[target_col] = y.values
    df_pca.to_csv("india_final.csv", index=False)
    print("Saved PCA dataset as india_final.csv")
