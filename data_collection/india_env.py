# ============================================
# INDIA ENVIRONMENTAL RISK DATA — COMPLETE
# ============================================

import requests
import pandas as pd
import numpy as np
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

BASE_DIR = os.path.dirname(__file__)
SOIL_PATH = os.path.join(BASE_DIR, "india_soil_clean.csv")

df_soil   = pd.read_csv(SOIL_PATH)
locations = df_soil[["location_id","latitude","longitude"]].to_dict("records")
print(f"✅ {len(locations)} locations loaded")

# ══════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ══════════════════════════════════════════
# 1. EARTHQUAKE — USGS API
# ══════════════════════════════════════════
def get_earthquake_risk(lat, lon):
    try:
        url    = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format"      : "geojson",
            "latitude"    : lat, "longitude": lon,
            "maxradiuskm" : 200,
            "minmagnitude": 3.0,
            "starttime"   : "2000-01-01",
            "endtime"     : "2024-01-01",
            "limit"       : 100
        }
        r    = requests.get(url, params=params, timeout=30)
        data = r.json()
        count = data["metadata"]["count"]
        mags  = [f["properties"]["mag"] for f in data["features"] if f["properties"]["mag"]]
        max_mag = round(max(mags), 2) if mags else 0.0
        if max_mag >= 6.0 or count > 50: risk = "High"
        elif max_mag >= 4.5 or count > 20: risk = "Medium"
        else: risk = "Low"
        return count, max_mag, risk
    except:
        return 0, 0.0, "Low"

# ══════════════════════════════════════════
# 2. SEISMIC ZONE — IS 1893 India
# ══════════════════════════════════════════
def get_seismic_zone(lat, lon):
    # Zone V — Very High (Himalayan, NE India, Andaman)
    if lat > 34 or (lat > 24 and lon > 91) or (lat < 14 and lon > 92):
        return "Zone V", 5
    # Zone IV — High
    elif lat > 30 or (lat > 22 and lon > 88) or (20 < lat < 24 and 72 < lon < 75):
        return "Zone IV", 4
    # Zone III — Moderate
    elif lat > 22 or (15 < lat < 22 and 73 < lon < 80):
        return "Zone III", 3
    # Zone II — Low (Most of South India)
    else:
        return "Zone II", 2

# ══════════════════════════════════════════
# 3. MAJOR INDIA RIVERS
# ══════════════════════════════════════════
INDIA_RIVERS = [
    # South India
    {"name":"Cauvery",       "lat":11.1, "lon":78.8},
    {"name":"Krishna",       "lat":16.5, "lon":80.6},
    {"name":"Godavari",      "lat":17.0, "lon":81.8},
    {"name":"Tungabhadra",   "lat":15.9, "lon":76.5},
    {"name":"Periyar",       "lat":10.2, "lon":76.3},
    {"name":"Vaigai",        "lat":9.9,  "lon":78.1},
    # North India
    {"name":"Ganga",         "lat":25.4, "lon":83.0},
    {"name":"Yamuna",        "lat":27.0, "lon":78.5},
    {"name":"Brahmaputra",   "lat":26.5, "lon":92.5},
    {"name":"Indus",         "lat":31.5, "lon":74.0},
    {"name":"Narmada",       "lat":22.5, "lon":76.0},
    {"name":"Tapti",         "lat":21.2, "lon":74.5},
    {"name":"Mahanadi",      "lat":20.5, "lon":83.5},
    {"name":"Damodar",       "lat":23.5, "lon":87.0},
    {"name":"Sabarmati",     "lat":23.0, "lon":72.5},
]

def get_flood_risk(lat, lon):
    dists    = [haversine(lat, lon, r["lat"], r["lon"]) for r in INDIA_RIVERS]
    min_dist = round(min(dists), 2)
    coastal  = lon > 79.5 or lon < 72.5 or lat < 9.0
    # Flood plains
    igp_plain   = 24 < lat < 28 and 75 < lon < 88   # Indo-Gangetic Plain
    brahma_plain = lat > 25 and lon > 89             # Brahmaputra flood plain
    delta       = (16 < lat < 18 and 81 < lon < 83) or (22 < lat < 22.5 and 88 < lon < 89)

    if coastal or igp_plain or brahma_plain or delta or min_dist < 10:
        return "High", min_dist
    elif min_dist < 30:
        return "Medium", min_dist
    else:
        return "Low", min_dist

# ══════════════════════════════════════════
# 4. TSUNAMI RISK
# ══════════════════════════════════════════
TSUNAMI_ZONES = [
    {"lat":13.0,  "lon":80.3,  "risk":"High"},    # Chennai
    {"lat":11.9,  "lon":79.8,  "risk":"High"},    # Puducherry
    {"lat":10.8,  "lon":79.8,  "risk":"High"},    # Nagapattinam
    {"lat":8.5,   "lon":77.5,  "risk":"Medium"},  # Kanyakumari
    {"lat":15.0,  "lon":80.0,  "risk":"High"},    # AP Coast
    {"lat":20.0,  "lon":86.5,  "risk":"High"},    # Odisha Coast
    {"lat":11.6,  "lon":92.7,  "risk":"Very High"}, # Andaman
]

def get_tsunami_risk(lat, lon):
    dists   = [haversine(lat, lon, t["lat"], t["lon"]) for t in TSUNAMI_ZONES]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    if min_d < 20:  return TSUNAMI_ZONES[min_idx]["risk"], min_d
    elif min_d < 100: return "Low", min_d
    else:            return "None", min_d

# ══════════════════════════════════════════
# 5. LANDSLIDE RISK
# ══════════════════════════════════════════
def get_landslide_risk(lat, lon):
    # Himalayan belt — Very High
    if lat > 30 and 75 < lon < 92:
        return "Very High", 10
    # NE India hills
    elif lat > 22 and lon > 90:
        return "High", 25
    # Western Ghats
    elif lon < 77.5 and 8 < lat < 15:
        return "High", 30
    # Eastern Ghats
    elif 15 < lat < 20 and 80 < lon < 83:
        return "Medium", 55
    # Nilgiris
    elif 11 < lat < 11.5 and 76.5 < lon < 77:
        return "High", 30
    else:
        return "Low", 85

# ══════════════════════════════════════════
# 6. COASTAL EROSION RISK
# ══════════════════════════════════════════
EROSION_HOTSPOTS = [
    {"lat":13.0, "lon":80.3, "risk":"High"},    # Chennai
    {"lat":20.3, "lon":86.7, "risk":"Very High"}, # Odisha
    {"lat":22.0, "lon":88.5, "risk":"Very High"}, # Sundarbans
    {"lat":15.5, "lon":80.0, "risk":"High"},    # AP
    {"lat":10.9, "lon":79.8, "risk":"High"},    # TN coast
]

def get_coastal_erosion(lat, lon):
    is_coastal = lon > 79.0 or lon < 73.0 or lat < 9.0
    if not is_coastal:
        return "None", 999
    dists   = [haversine(lat, lon, e["lat"], e["lon"]) for e in EROSION_HOTSPOTS]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    if min_d < 30:  return EROSION_HOTSPOTS[min_idx]["risk"], min_d
    elif min_d < 100: return "Low", min_d
    else:            return "None", min_d

# ══════════════════════════════════════════
# 7. MINING / SUBSIDENCE ZONES
# ══════════════════════════════════════════
MINING_ZONES = [
    {"lat":23.8, "lon":86.4, "name":"Jharia Coalfield",     "risk":"Very High"},
    {"lat":23.5, "lon":85.3, "name":"Ranchi Mining",        "risk":"High"},
    {"lat":22.0, "lon":85.8, "name":"Rourkela Steel Zone",  "risk":"High"},
    {"lat":21.2, "lon":81.6, "name":"Chhattisgarh Coal",    "risk":"High"},
    {"lat":15.3, "lon":76.9, "name":"Bellary Iron Ore",     "risk":"High"},
    {"lat":22.7, "lon":86.2, "name":"Singhbhum Copper",     "risk":"Medium"},
    {"lat":14.4, "lon":78.8, "name":"Kurnool Mines",        "risk":"Medium"},
    {"lat":25.3, "lon":83.0, "name":"Mirzapur Quarries",    "risk":"Medium"},
]

def get_mining_risk(lat, lon):
    dists   = [haversine(lat, lon, m["lat"], m["lon"]) for m in MINING_ZONES]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    if min_d < 10:  return MINING_ZONES[min_idx]["risk"], min_d, MINING_ZONES[min_idx]["name"]
    elif min_d < 25: return "Medium", min_d, MINING_ZONES[min_idx]["name"]
    elif min_d < 50: return "Low", min_d, MINING_ZONES[min_idx]["name"]
    else:            return "None", min_d, "None"

# ══════════════════════════════════════════
# 8. INDUSTRIAL HAZARD ZONES
# ══════════════════════════════════════════
INDUSTRIAL_ZONES = [
    # South India
    {"lat":13.2, "lon":80.3,  "name":"Manali Industrial Chennai",  "risk":"High"},
    {"lat":11.7, "lon":79.7,  "name":"SIPCOT Cuddalore",           "risk":"Very High"},
    {"lat":13.4, "lon":80.1,  "name":"Gummidipoondi SIPCOT",       "risk":"High"},
    {"lat":11.0, "lon":77.0,  "name":"Coimbatore SIDCO",           "risk":"Medium"},
    {"lat":17.4, "lon":78.5,  "name":"Hyderabad Industrial",       "risk":"High"},
    {"lat":12.9, "lon":77.6,  "name":"Bangalore Whitefield",       "risk":"Medium"},
    # West India
    {"lat":19.1, "lon":72.9,  "name":"Mumbai Thane Industrial",    "risk":"Very High"},
    {"lat":22.3, "lon":70.8,  "name":"Jamnagar Petrochemical",     "risk":"Very High"},
    {"lat":21.2, "lon":72.8,  "name":"Surat Industrial",           "risk":"High"},
    # North India
    {"lat":28.7, "lon":77.1,  "name":"Delhi NCR Industrial",       "risk":"High"},
    {"lat":27.5, "lon":77.7,  "name":"Mathura Refinery",           "risk":"Very High"},
    {"lat":22.8, "lon":86.2,  "name":"Jamshedpur Steel",           "risk":"High"},
    # East India
    {"lat":22.6, "lon":88.4,  "name":"Haldia Petrochemical",       "risk":"Very High"},
]

def get_industrial_hazard(lat, lon):
    dists   = [haversine(lat, lon, ind["lat"], ind["lon"]) for ind in INDUSTRIAL_ZONES]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    if min_d < 5:   return INDUSTRIAL_ZONES[min_idx]["risk"], min_d, INDUSTRIAL_ZONES[min_idx]["name"]
    elif min_d < 15: return "Medium", min_d, INDUSTRIAL_ZONES[min_idx]["name"]
    elif min_d < 30: return "Low", min_d, INDUSTRIAL_ZONES[min_idx]["name"]
    else:            return "None", min_d, "None"

# ══════════════════════════════════════════
# 9. WIND ZONE — IS 875 India
# ══════════════════════════════════════════
def get_wind_zone(lat, lon):
    if lat < 10 or (lat < 15 and lon > 79) or (lat > 20 and lon < 70):
        return "Zone VI", 55, "Very High"
    elif (lat < 14 and lon < 77) or (lat > 22 and lon < 72):
        return "Zone V",  50, "High"
    elif lat > 28 or (14 < lat < 20 and lon > 80):
        return "Zone IV", 47, "High"
    elif 20 < lat < 28:
        return "Zone III", 44, "Medium"
    else:
        return "Zone II",  39, "Low"

# ══════════════════════════════════════════
# 10. SLOPE / TERRAIN
# ══════════════════════════════════════════
def get_slope_risk(lat, lon):
    if lat > 30 and 75 < lon < 92:
        return "Very Steep (>30°)", "Very High", 5
    elif (lon < 77.5 and 8 < lat < 15) or (lat > 25 and lon > 90):
        return "Steep (15-30°)", "High", 20
    elif (77.5 < lon < 79 and 10 < lat < 13) or (lat > 22 and lon > 92):
        return "Moderate (5-15°)", "Medium", 55
    elif lon > 79.5 or (22 < lat < 28 and 72 < lon < 76):
        return "Flat-Coastal (<2°)", "Low", 75
    else:
        return "Gentle (2-5°)", "Low", 85

# ══════════════════════════════════════════
# 11. FOREST FIRE RISK
# ══════════════════════════════════════════
def get_forest_fire_risk(lat, lon):
    # High fire risk zones in India
    if 20 < lat < 25 and 80 < lon < 85:
        return "High"    # Central India forests
    elif lat > 25 and 72 < lon < 78:
        return "Medium"  # Rajasthan scrub
    elif lon < 77.5 and 8 < lat < 15:
        return "Medium"  # Western Ghats (dry season)
    elif lat > 26 and lon > 92:
        return "High"    # NE India forests
    else:
        return "Low"

# ══════════════════════════════════════════
# 12. GROUNDWATER DEPTH
# ══════════════════════════════════════════
def get_groundwater_depth(lat, lon):
    if lon > 79.5 or lat < 9:
        return "Shallow (0-3m)", "High Risk"
    elif lat > 28 and 75 < lon < 85:
        return "Medium (3-8m)", "Medium Risk"   # IGP alluvial
    elif 15 < lat < 25 and 74 < lon < 82:
        return "Deep (10-25m)", "Low Risk"      # Deccan plateau
    elif lat < 15 and lon < 77:
        return "Deep (15-30m)", "Low Risk"      # Hard rock terrain
    else:
        return "Medium-Deep (5-15m)", "Low Risk"

# ══════════════════════════════════════════
# 13. DRAINAGE QUALITY
# ══════════════════════════════════════════
def get_drainage_quality(lat, lon, clay_percent=None):
    coastal = lon > 79.5 or lat < 9
    igp     = 24 < lat < 28 and 75 < lon < 88
    clay_high = clay_percent and clay_percent > 35
    if coastal or igp or clay_high:
        return "Poor"
    elif lon < 77.5 and 8 < lat < 15:
        return "Good"    # Western Ghats natural drainage
    else:
        return "Moderate"

# ══════════════════════════════════════════
# 14. AIR QUALITY ZONE
# ══════════════════════════════════════════
def get_air_quality_zone(lat, lon):
    # Most polluted cities in India
    if (28 < lat < 29 and 76 < lon < 78):
        return "Critical", "Delhi NCR"
    elif (22.4 < lat < 22.7 and 88.2 < lon < 88.5):
        return "Very High", "Kolkata"
    elif (19.0 < lat < 19.2 and 72.8 < lon < 73.0):
        return "High", "Mumbai"
    elif (23.7 < lat < 23.9 and 86.3 < lon < 86.5):
        return "High", "Jharia Industrial"
    elif (13.0 < lat < 13.2 and 80.2 < lon < 80.4):
        return "Medium", "Chennai"
    else:
        return "Low", "Clean Zone"

# ══════════════════════════════════════════
# FINAL SCORE
# ══════════════════════════════════════════
def calculate_env_score(eq_risk, flood_risk, landslide_risk,
                         tsunami_risk, coastal_erosion, mining_risk,
                         industrial_risk, wind_severity, slope_risk,
                         forest_fire_risk, drainage, air_quality,
                         seismic_zone_num):
    score = 100

    score += {"Low":0, "Medium":-10, "High":-20}.get(eq_risk, 0)
    score += {"Low":0, "Medium":-10, "High":-20}.get(flood_risk, 0)
    score += {"Low":0, "Medium":-8,  "High":-18, "Very High":-25}.get(landslide_risk, 0)
    score += {"None":0,"Low":-5,  "Medium":-12, "High":-20, "Very High":-30}.get(tsunami_risk, 0)
    score += {"None":0,"Low":-3,  "Medium":-10, "High":-18, "Very High":-25}.get(coastal_erosion, 0)
    score += {"None":0,"Low":-3,  "Medium":-8,  "High":-15, "Very High":-20}.get(mining_risk, 0)
    score += {"None":0,"Low":-3,  "Medium":-8,  "High":-15, "Very High":-20}.get(industrial_risk, 0)
    score += {"Low":0, "Medium":-5, "High":-10, "Very High":-15}.get(wind_severity, 0)
    score += {"Low":0, "Medium":-5, "High":-12, "Very High":-20}.get(slope_risk, 0)
    score += {"Low":0, "Medium":-5, "High":-12}.get(forest_fire_risk, 0)
    score += {"Good":0,"Moderate":-5,"Poor":-10}.get(drainage, 0)
    score += {"Low":0, "Medium":-5, "High":-10, "Very High":-15, "Critical":-20}.get(air_quality, 0)
    if seismic_zone_num >= 5: score -= 20
    elif seismic_zone_num == 4: score -= 10
    elif seismic_zone_num == 3: score -= 5

    return max(0, min(100, round(score, 1)))

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
results_lock = threading.Lock()
results  = []
failed   = []
counter  = [0]
clay_map = dict(zip(df_soil["location_id"], df_soil.get("clay_percent", {})))

def process_location(loc):
    try:
        lat = loc["latitude"]
        lon = loc["longitude"]
        lid = loc["location_id"]

        eq_count, max_mag, eq_risk       = get_earthquake_risk(lat, lon)
        seismic_zone, seismic_num        = get_seismic_zone(lat, lon)
        flood_risk, river_dist           = get_flood_risk(lat, lon)
        landslide_risk, ls_score         = get_landslide_risk(lat, lon)
        tsunami_risk, tsunami_dist       = get_tsunami_risk(lat, lon)
        erosion_risk, erosion_dist       = get_coastal_erosion(lat, lon)
        mining_risk, mine_dist, mine_name= get_mining_risk(lat, lon)
        ind_risk, ind_dist, ind_name     = get_industrial_hazard(lat, lon)
        wind_zone, wind_speed, wind_sev  = get_wind_zone(lat, lon)
        slope_desc, slope_risk, _        = get_slope_risk(lat, lon)
        fire_risk                        = get_forest_fire_risk(lat, lon)
        gw_depth, gw_risk                = get_groundwater_depth(lat, lon)
        clay_val                         = clay_map.get(lid)
        drainage                         = get_drainage_quality(lat, lon, clay_val)
        air_quality, air_zone            = get_air_quality_zone(lat, lon)

        env_score = calculate_env_score(
            eq_risk, flood_risk, landslide_risk,
            tsunami_risk, erosion_risk, mining_risk,
            ind_risk, wind_sev, slope_risk,
            fire_risk, drainage, air_quality,
            seismic_num
        )

        row = {
            "location_id"              : lid,
            "latitude"                 : lat,
            "longitude"                : lon,
            "earthquake_count"         : eq_count,
            "max_earthquake_magnitude" : max_mag,
            "earthquake_risk"          : eq_risk,
            "seismic_zone"             : seismic_zone,
            "seismic_zone_number"      : seismic_num,
            "flood_risk"               : flood_risk,
            "nearest_river_dist_km"    : river_dist,
            "landslide_risk"           : landslide_risk,
            "tsunami_risk"             : tsunami_risk,
            "tsunami_zone_dist_km"     : tsunami_dist,
            "coastal_erosion_risk"     : erosion_risk,
            "erosion_zone_dist_km"     : erosion_dist,
            "mining_subsidence_risk"   : mining_risk,
            "nearest_mining_km"        : mine_dist,
            "nearest_mining_zone"      : mine_name,
            "industrial_hazard_risk"   : ind_risk,
            "nearest_industrial_km"    : ind_dist,
            "nearest_industrial_zone"  : ind_name,
            "wind_zone"                : wind_zone,
            "basic_wind_speed_ms"      : wind_speed,
            "wind_severity"            : wind_sev,
            "terrain_slope"            : slope_desc,
            "slope_risk"               : slope_risk,
            "forest_fire_risk"         : fire_risk,
            "groundwater_depth"        : gw_depth,
            "groundwater_risk"         : gw_risk,
            "drainage_quality"         : drainage,
            "air_quality_zone"         : air_quality,
            "air_quality_area"         : air_zone,
            "env_construction_score"   : env_score,
        }

        with results_lock:
            results.append(row)
            counter[0] += 1
            c = counter[0]
            if c % 50 == 0:
                pct = round(c / len(locations) * 100, 1)
                print(f"✅ {c}/{len(locations)} — {pct}% — Score: {env_score}")
            if c % 200 == 0:
                pd.DataFrame(results).to_csv("india_env_checkpoint.csv", index=False)
                print(f"💾 Checkpoint saved — {c} rows")

    except Exception as e:
        with results_lock:
            failed.append(loc["location_id"])

def main():
    print(f"\n🚀 Starting India Environmental Collection...")
    print(f"📍 Locations : {len(locations)}")
    print(f"⏱️ Estimated  : ~{round(len(locations)*1.5/4/60,0)} minutes\n")

    start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_location, loc): loc for loc in locations}
        for _future in as_completed(futures):
            pass

    elapsed = round((time.time() - start) / 60, 1)

    df_env = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, "india_env_data.csv")
    df_env.to_csv(out_path, index=False)

    print(f"\n🎉 COMPLETE in {elapsed} minutes!")
    print(f"✅ Success  : {len(results)}")
    print(f"❌ Failed   : {len(failed)}")
    print(f"\n📊 Env Score:")
    print(df_env["env_construction_score"].describe())
    print(f"\n⚠️ Risk Summary:")
    print("Earthquake   :", df_env["earthquake_risk"].value_counts().to_dict())
    print("Flood        :", df_env["flood_risk"].value_counts().to_dict())
    print("Landslide    :", df_env["landslide_risk"].value_counts().to_dict())
    print("Tsunami      :", df_env["tsunami_risk"].value_counts().to_dict())
    print("Mining       :", df_env["mining_subsidence_risk"].value_counts().to_dict())
    print("Industrial   :", df_env["industrial_hazard_risk"].value_counts().to_dict())
    print("Wind         :", df_env["wind_severity"].value_counts().to_dict())
    print("Forest Fire  :", df_env["forest_fire_risk"].value_counts().to_dict())
    print("Air Quality  :", df_env["air_quality_zone"].value_counts().to_dict())


if __name__ == "__main__":
    main()

