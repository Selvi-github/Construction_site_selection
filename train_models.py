import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import warnings
warnings.filterwarnings('ignore')

# 1. Load dataset
df = pd.read_csv("india_final.csv")
target_col = "final_feasibility_score"

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store metrics
metrics = {}

def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    metrics[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
    print(f"\n--- {name} ---")
    print(f"R²  : {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    return model

# 3. Base Models
rf = RandomForestRegressor(n_estimators=200, random_state=42)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
et = ExtraTreesRegressor(n_estimators=200, random_state=42)

print("Training Base Models...")
rf_model = evaluate(rf, "Random Forest")
xgb_model = evaluate(xgb, "XGBoost")
et_model = evaluate(et, "Extra Trees")

# 4 & 5. Stacking Ensemble
print("\nTraining Stacking Ensemble (5-fold CV)...")
estimators = [
    ('rf', rf),
    ('xgb', xgb),
    ('et', et)
]
stacked = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5
)
stacked_model = evaluate(stacked, "Stacked Ensemble")

# 6. Comparison Table
print("\n" + "="*50)
print(f"{'Model':<20} | {'R²':<8} | {'RMSE':<8} | {'MAE':<8}")
print("-" * 50)
for name, m in metrics.items():
    print(f"{name:<20} | {m['R2']:<8.4f} | {m['RMSE']:<8.4f} | {m['MAE']:<8.4f}")
print("="*50)

# 7. Identify best single model for Optuna tuning
single_models = {k: v for k, v in metrics.items() if k != "Stacked Ensemble"}
best_model_name = max(single_models, key=lambda k: single_models[k]['R2'])

print(f"\nBest performing single model is: {best_model_name}")
print("Starting Optuna tuning with 50 trials...")

def objective(trial):
    if best_model_name == "Random Forest":
        n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42
        )
    elif best_model_name == "Extra Trees":
        n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42
        )
    else: # XGBoost
        n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42
        )
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\nBest parameters found for {best_model_name}:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"Best tuned R²: {study.best_value:.4f}")

# 8. Save final stacked model
with open("model_stacked.pkl", "wb") as f:
    pickle.dump(stacked_model, f)
print("\nSaved final stacked model as model_stacked.pkl")
