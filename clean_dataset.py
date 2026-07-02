import pandas as pd
import os

# Load the dataset
input_path = "india_master_dataset.csv"
if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    exit(1)

df = pd.read_csv(input_path)

# Drop 6 - same value in every row
drops_1 = [
    "drought_risk", "heat_index_category", "industrial_hazard_risk", 
    "air_quality_zone", "air_quality_area", "bird_zone_risk"
]

# Drop 14 - place name text
drops_2 = [
    "location_id", "nearest_protected_area", "nearest_tiger_corridor", 
    "nearest_elephant_corridor", "nearest_bird_zone", "nearest_endangered_species", 
    "endangered_habitat_name", "nearest_conflict_zone", "nearest_mining_zone", 
    "nearest_industrial_zone", "nearest_marine_area", "hotspot_name", 
    "nearest_cyclone_zone_km", "predicted_lifespan"
]

# Drop 13 - redundant columns
drops_3 = [
    "wind_severity", "seismic_zone", "fog_risk", "slope_risk", "groundwater_risk", 
    "extreme_heat_category", "max_temp_C", "avg_wind_speed_ms", "tiger_corridor_risk", 
    "elephant_corridor_risk", "endangered_habitat_risk", "protected_area_risk", 
    "human_animal_conflict_risk"
]

# Drop 9 - output columns
drops_4 = [
    "soil_construction_score", "climate_construction_score", "env_construction_score", 
    "animal_construction_score", "construction_success_label", "construction_success_category", 
    "construction_viability_score", "risk_level", "confidence_percent"
]

# Combine all drops
all_drops = drops_1 + drops_2 + drops_3 + drops_4

# Find columns to drop that actually exist in the dataframe
cols_to_drop = [col for col in all_drops if col in df.columns]

# Rename blank column
blank_cols = [c for c in df.columns if c.strip() == ""]
if blank_cols:
    df.rename(columns={blank_cols[0]: "clay_percent"}, inplace=True)
elif "Unnamed: 0" in df.columns: # sometimes empty columns are parsed as Unnamed: 0
    df.rename(columns={"Unnamed: 0": "clay_percent"}, inplace=True)
else:
    print("Could not find a blank column to rename. Current columns:")
    print(df.columns)

# Drop columns
df.drop(columns=cols_to_drop, inplace=True)

# Ensure final_feasibility_score is kept
if "final_feasibility_score" not in df.columns:
    print("Warning: final_feasibility_score is missing from the dataset.")

# Print remaining columns count
print(f"Remaining columns count: {len(df.columns)}")

# Print full column list
print("Remaining columns:")
print(list(df.columns))

# Save the file
output_path = "india_master_clean.csv"
df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")
