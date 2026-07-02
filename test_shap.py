import pandas as pd
import numpy as np
import pickle
import shap
import time

# Load stacked model
with open("model_stacked.pkl", "rb") as f:
    model_stacked = pickle.load(f)

with open("selected_features.txt", "r") as f:
    stacked_features = [line.strip() for line in f if line.strip()]

# Create a single dummy row
X_dummy = pd.DataFrame([np.zeros(len(stacked_features))], columns=stacked_features)

print("Starting SHAP KernelExplainer test...")
t0 = time.time()
# KernelExplainer needs a background dataset. We can use 10 zero rows or random rows.
X_background = shap.sample(pd.DataFrame(np.random.randn(10, len(stacked_features)), columns=stacked_features), 10)
explainer = shap.KernelExplainer(model_stacked.predict, X_background)

t1 = time.time()
shap_values = explainer.shap_values(X_dummy)
t2 = time.time()

print(f"Explainer init: {t1-t0:.2f}s")
print(f"SHAP values calc: {t2-t1:.2f}s")
print("SHAP values shape:", shap_values.shape)

