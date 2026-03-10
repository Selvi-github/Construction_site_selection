# ============================================================
# INDIA MASTER DATASET — MERGE ALL 4 CSV FILES
# ============================================================

import pandas as pd
import numpy as np
import os

# ── STEP 1: Load All 4 Datasets ──
BASE_DIR = os.path.dirname(__file__)

print("📂 Loading datasets...")

df_soil    = pd.read_csv(os.path.join(BASE_DIR, "india_soil_clean.csv"))
df_climate = pd.read_csv(os.path.join(BASE_DIR, "india_climate_data.csv"))
df_env     = pd.read_csv(os.path.join(BASE_DIR, "india_env_data.csv"))
df_animal  = pd.read_csv(os.path.join(BASE_DIR, "india_animal_data.csv"))

print(f"✅ Soil    : {df_soil.shape}")
print(f"✅ Climate : {df_climate.shape}")
print(f"✅ Env     : {df_env.shape}")
print(f"✅ Animal  : {df_animal.shape}")

# ── STEP 2: Clean location_id (ensure same format) ──
for df in [df_soil, df_climate, df_env, df_animal]:
    df["location_id"] = df["location_id"].astype(str).str.strip()

# ── STEP 3: Merge All 4 Datasets ──
print("\n🔗 Merging datasets...")

df = df_soil.merge(df_climate, on="location_id", suffixes=("","_clim"))
df = df.merge(df_env,          on="location_id", suffixes=("","_env"))
df = df.merge(df_animal,       on="location_id", suffixes=("","_anim"))

# Remove duplicate lat/lon columns from merges
dup_cols = [c for c in df.columns if c.endswith(("_clim","_env","_anim"))]
df = df.drop(columns=dup_cols)

print(f"✅ Merged shape: {df.shape}")

# ── STEP 4: Final Feasibility Score ──
print("\n⚡ Calculating final scores...")

def calculate_final_score(row):
    soil    = row.get("soil_construction_score")    or 50
    climate = row.get("climate_construction_score") or 50
    env     = row.get("env_construction_score")     or 50
    animal  = row.get("animal_construction_score")  or 50

    # NaN check
    if pd.isna(soil):    soil    = 50
    if pd.isna(climate): climate = 50
    if pd.isna(env):     env     = 50
    if pd.isna(animal):  animal  = 50

    # Weighted ensemble
    final = (soil    * 0.35 +
             climate * 0.25 +
             env     * 0.25 +
             animal  * 0.15)

    return round(final, 2)

df["final_feasibility_score"] = df.apply(calculate_final_score, axis=1)

# ── STEP 5: Risk Classification ──
def risk_classification(score):
    if pd.isna(score):  return "Unknown"
    if score >= 70:     return "Low Risk"
    elif score >= 45:   return "Medium Risk"
    else:               return "High Risk"

df["risk_level"] = df["final_feasibility_score"].apply(risk_classification)

# ── STEP 6: Lifespan Prediction ──
def predict_lifespan(row):
    score  = row.get("final_feasibility_score")    or 50
    bc     = row.get("bearing_capacity_kNm2")      or 100
    env    = row.get("env_construction_score")     or 50

    # NaN check
    if pd.isna(score): score = 50
    if pd.isna(bc):    bc    = 100
    if pd.isna(env):   env   = 50

    # Formula
    base  = 25
    base += (float(score) / 100) * 50
    base += (float(bc)    / 500) * 20
    base += (float(env)   / 100) * 15

    if pd.isna(base): base = 50

    base  = round(float(base), 0)
    low   = max(10, int(base - 10))
    high  = int(base + 10)
    conf  = round(50 + (float(score) / 100) * 40, 1)

    return f"{low}-{high} years", conf

lifespan_data              = df.apply(predict_lifespan, axis=1)
df["predicted_lifespan"]   = lifespan_data.apply(lambda x: x[0])
df["confidence_percent"]   = lifespan_data.apply(lambda x: x[1])

# ── STEP 7: Foundation Recommendation ──
def final_foundation(row):
    bc = row.get("bearing_capacity_kNm2") or 100
    lq = row.get("liquefaction_risk")     or "Low"
    ss = row.get("shrink_swell_risk")     or "Low"

    if pd.isna(bc): bc = 100

    if lq == "High":            return "Pile Foundation (Deep)"
    elif float(bc) < 60:        return "Raft Foundation"
    elif ss == "High":          return "Raft Foundation"
    elif float(bc) < 100:       return "Isolated Footing"
    elif float(bc) >= 150:      return "Simple Strip Footing"
    else:                       return "Isolated Footing"

df["recommended_foundation"] = df.apply(final_foundation, axis=1)

# ── STEP 8: Building Success Label (from animal dataset) ──
# Already in animal dataset — just rename for clarity
if "construction_success_label" in df.columns:
    print("✅ Building success labels already present!")
else:
    df["construction_success_label"]    = 0.5
    df["construction_success_category"] = "Moderate"

# ── STEP 9: Drop Null Rows ──
before = len(df)
df = df.dropna(subset=["final_feasibility_score",
                        "soil_construction_score",
                        "climate_construction_score"])
after = len(df)
print(f"🧹 Null rows removed: {before - after}")
print(f"✅ Final clean rows : {after}")

def main():
    # ── STEP 10: Save ──
    out_path = os.path.join(BASE_DIR, "india_master_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"\n🎉 MASTER DATASET READY!")
    print(f"📊 Shape    : {df.shape}")
    print(f"📋 Columns  : {len(df.columns)}")
    print(f"\n🏗️ Score Summary:")
    print(df["final_feasibility_score"].describe())
    print(f"\n⚠️ Risk Levels:")
    print(df["risk_level"].value_counts())
    print(f"\n🏛️ Foundation Types:")
    print(df["recommended_foundation"].value_counts())
    print(f"\n✅ Success Labels:")
    print(df["construction_success_category"].value_counts())
    print(f"\n📅 Lifespan Sample:")
    print(df["predicted_lifespan"].value_counts().head(5))


if __name__ == "__main__":
    main()
