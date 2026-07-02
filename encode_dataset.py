import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Load dataset
df = pd.read_csv("india_master_clean.csv")

# 2. Map risk columns
risk_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4, "Unknown": 0}
risk_cols = [
    "shrink_swell_risk", "liquefaction_risk", "corrosion_risk", "cyclone_risk",
    "monsoon_intensity", "lightning_risk", "earthquake_risk", "flood_risk",
    "landslide_risk", "tsunami_risk", "coastal_erosion_risk",
    "mining_subsidence_risk", "marine_protected_area_risk", "burrowing_animal_risk"
]
for col in risk_cols:
    if col in df.columns:
        df[col] = df[col].map(risk_map)

# 3. Map foundation
foundation_map = {
    "Simple Strip Footing": 1, "Isolated Footing": 2, 
    "Raft Foundation": 3, "Pile Foundation (Deep)": 4
}
if "recommended_foundation" in df.columns:
    df["recommended_foundation"] = df["recommended_foundation"].map(foundation_map)

# 4. Map biodiversity hotspot
hotspot_map = {"Yes": 1, "No": 0}
if "biodiversity_hotspot" in df.columns:
    df["biodiversity_hotspot"] = df["biodiversity_hotspot"].map(hotspot_map)

# 5. Map protected area type
protected_map = {"National Park": 1, "Tiger Reserve": 2, "Marine Park": 3}
if "protected_area_type" in df.columns:
    df["protected_area_type"] = df["protected_area_type"].map(protected_map)

# 6. Apply label encoding for remaining object/string columns
le = LabelEncoder()
object_cols = df.select_dtypes(include=['object', 'string']).columns
for col in object_cols:
    # Convert to object type first to allow integer assignment
    df[col] = df[col].astype(object)
    # Re-apply label encoding, missing values will become a category but we will fill them
    df[col] = df[col].replace('nan', np.nan)
    mask = df[col].notna()
    if mask.any():
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
    
# Convert all object columns to numeric (they should be after encoding/mapping)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 7. Fill missing values with median
for col in df.columns:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# 8. Print shape
print(f"Final shape: {df.shape}")

# 9. Print first 3 rows
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# 10. Save as india_encoded.csv
df.to_csv("india_encoded.csv", index=False)
print("\nSaved encoded dataset to india_encoded.csv")
