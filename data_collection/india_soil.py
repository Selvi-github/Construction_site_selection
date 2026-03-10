# Fast Parallel Soil Collection

import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

# Thread-safe results list
results_lock = threading.Lock()
results = []
failed  = []
counter = [0]
TOTAL_LOCATIONS = 0

BASE_DIR = os.path.dirname(__file__)

def get_soilgrids_data(lat, lon):
    soil_props = ["clay","sand","silt","phh2o","bdod","cec","soc","nitrogen"]
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon"     : lon,
        "lat"     : lat,
        "property": soil_props,
        "depth"   : "0-5cm",
        "value"   : "mean"
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    row = {}
    for layer in data["properties"]["layers"]:
        try:
            row[layer["name"]] = layer["depths"][0]["values"]["mean"]
        except:
            row[layer["name"]] = None
    return row

def calculate_bearing_capacity(bdod, clay, sand):
    if bdod is None or clay is None: return None
    bd = bdod / 100
    if sand and sand > 60: return round(bd * 150, 2)
    elif clay and clay > 40: return round(bd * 60, 2)
    else: return round(bd * 100, 2)

def calculate_shrink_swell(clay):
    if clay is None: return "Unknown"
    c = clay / 10
    if c > 40: return "High"
    elif c > 25: return "Medium"
    else: return "Low"

def calculate_liquefaction(sand, bdod):
    if sand is None or bdod is None: return "Unknown"
    s = sand / 10
    bd = bdod / 100
    if s > 60 and bd < 1.4: return "High"
    elif s > 40 and bd < 1.6: return "Medium"
    else: return "Low"

def calculate_permeability(sand, clay):
    if sand is None or clay is None: return None
    return round(max((sand/10 * 2.5) - (clay/10 * 1.2), 0.1), 2)

def calculate_corrosion(ph):
    if ph is None: return "Unknown"
    p = ph / 10
    if p < 5.5: return "High"
    elif p < 6.5: return "Medium"
    else: return "Low"

def estimate_water_table(lat, lon):
    if lon > 79.5 or lon < 72.5: return "Shallow (1-3m) — High Risk"
    elif lat > 28.0: return "Medium (3-8m) — Medium Risk"
    elif 15.0 < lat < 25.0 and 74.0 < lon < 82.0: return "Deep (10-20m) — Low Risk"
    else: return "Medium-Deep (5-12m) — Low-Medium Risk"

def calculate_soil_score(row):
    score = 50
    bc = row.get("bearing_capacity_kNm2")
    if bc:
        if bc > 150: score += 20
        elif bc > 100: score += 10
        elif bc < 60: score -= 20
    ss = row.get("shrink_swell_risk")
    if ss == "Low": score += 15
    elif ss == "High": score -= 20
    lq = row.get("liquefaction_risk")
    if lq == "Low": score += 10
    elif lq == "Medium": score -= 5
    elif lq == "High": score -= 25
    cr = row.get("corrosion_risk")
    if cr == "Low": score += 5
    elif cr == "High": score -= 10
    return max(0, min(100, round(score, 1)))

def recommend_foundation(row):
    bc = row.get("bearing_capacity_kNm2") or 0
    ss = row.get("shrink_swell_risk","")
    lq = row.get("liquefaction_risk","")
    if lq == "High": return "Pile Foundation (Deep)"
    elif bc < 60 or ss == "High": return "Raft Foundation"
    elif bc < 100: return "Isolated Footing with RCC"
    elif bc >= 150: return "Simple Strip Footing"
    else: return "Isolated Footing"

def process_location(loc):
    try:
        soil = get_soilgrids_data(loc["lat"], loc["lon"])

        clay = soil.get("clay")
        sand = soil.get("sand")
        silt = soil.get("silt")
        ph   = soil.get("phh2o")
        bdod = soil.get("bdod")
        cec  = soil.get("cec")
        soc  = soil.get("soc")
        nit  = soil.get("nitrogen")

        bc   = calculate_bearing_capacity(bdod, clay, sand)
        ss   = calculate_shrink_swell(clay)
        lq   = calculate_liquefaction(sand, bdod)
        perm = calculate_permeability(sand, clay)
        wt   = estimate_water_table(loc["lat"], loc["lon"])
        corr = calculate_corrosion(ph)

        row = {
            "location_id"           : loc["name"],
            "latitude"              : loc["lat"],
            "longitude"             : loc["lon"],
            "clay_percent"          : round(clay/10,1)  if clay  else None,
            "sand_percent"          : round(sand/10,1)  if sand  else None,
            "silt_percent"          : round(silt/10,1)  if silt  else None,
            "ph_value"              : round(ph/10,1)    if ph    else None,
            "bulk_density_gcm3"     : round(bdod/100,2) if bdod  else None,
            "cec_cmolkg"            : round(cec/10,1)   if cec   else None,
            "organic_carbon_percent": round(soc/10,2)   if soc   else None,
            "nitrogen_mgkg"         : nit,
            "bearing_capacity_kNm2" : bc,
            "shrink_swell_risk"     : ss,
            "liquefaction_risk"     : lq,
            "permeability_mmhr"     : perm,
            "estimated_water_table" : wt,
            "corrosion_risk"        : corr,
        }
        row["soil_construction_score"] = calculate_soil_score(row)
        row["recommended_foundation"]  = recommend_foundation(row)

        # Thread-safe append
        with results_lock:
            results.append(row)
            counter[0] += 1
            c = counter[0]
            if c % 50 == 0 and TOTAL_LOCATIONS:
                print(f"✅ {c}/{TOTAL_LOCATIONS} done ({round(c/TOTAL_LOCATIONS*100,1)}%)")
            if c % 200 == 0:
                pd.DataFrame(results).to_csv("india_soil_checkpoint.csv", index=False)
                print(f"💾 Checkpoint saved — {c} rows")

        return True

    except Exception as e:
        with results_lock:
            failed.append(loc["name"])
        return False

def main():
    # ── GENERATE LOCATIONS ──
    lat_range = np.arange(8.0, 37.0, 0.5)
    lon_range = np.arange(68.0, 97.5, 0.5)

    locations = []
    for lat in lat_range:
        for lon in lon_range:
            locations.append({
                "name": f"IND_{round(lat,1)}_{round(lon,1)}",
                "lat" : round(lat, 4),
                "lon" : round(lon, 4)
            })

    global TOTAL_LOCATIONS
    TOTAL_LOCATIONS = len(locations)
    print(f"✅ Total locations: {TOTAL_LOCATIONS}")
    print(f"⏱️ Estimated time with 4 threads: ~{round(TOTAL_LOCATIONS*1.2/4/60,0)} minutes")

    # ── RUN PARALLEL COLLECTION ──
    print("\n🚀 Starting parallel collection...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_location, loc): loc for loc in locations}
        for _future in as_completed(futures):
            pass  # Results handled inside process_location

    elapsed = round((time.time() - start_time) / 60, 1)

    # ── FINAL SAVE ──
    df_soil = pd.DataFrame(results)

    # Remove rows where all soil values are null
    df_soil_clean = df_soil.dropna(subset=["clay_percent","sand_percent","ph_value"])

    raw_path = os.path.join(BASE_DIR, "india_soil_raw.csv")
    clean_path = os.path.join(BASE_DIR, "india_soil_clean.csv")
    df_soil.to_csv(raw_path, index=False)
    df_soil_clean.to_csv(clean_path, index=False)

    print(f"\n🎉 COMPLETE in {elapsed} minutes!")
    print(f"✅ Total collected : {len(df_soil)} locations")
    print(f"✅ Clean data      : {len(df_soil_clean)} locations")
    print(f"❌ Failed          : {len(failed)} locations")
    print(f"\n📊 Score Distribution:")
    print(df_soil_clean["soil_construction_score"].describe())


if __name__ == "__main__":
    main()

