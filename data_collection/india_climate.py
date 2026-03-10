# ============================================
# INDIA CLIMATE DATA — COMPLETE VERSION
# ============================================

import requests
import pandas as pd
import numpy as np
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

BASE_DIR = os.path.dirname(__file__)
SOIL_PATH = os.path.join(BASE_DIR, "india_soil_clean.csv")

df_soil   = pd.read_csv(SOIL_PATH)
locations = df_soil[["location_id","latitude","longitude"]].to_dict("records")
print(f"✅ {len(locations)} locations loaded")

# ══════════════════════════════════════════
# 1. CYCLONE RISK — India Coast Based
# ══════════════════════════════════════════
# Historical cyclone tracks — Bay of Bengal + Arabian Sea
CYCLONE_PRONE = [
    # Bay of Bengal coast (high risk)
    {"lat": 13.08, "lon": 80.27, "name": "Chennai Coast",      "risk": "High"},
    {"lat": 16.50, "lon": 81.50, "name": "Andhra Coast",       "risk": "High"},
    {"lat": 20.27, "lon": 85.84, "name": "Odisha Coast",       "risk": "Very High"},
    {"lat": 22.57, "lon": 88.36, "name": "West Bengal Coast",  "risk": "High"},
    {"lat": 10.77, "lon": 79.84, "name": "Nagapattinam",       "risk": "High"},
    # Arabian Sea coast (moderate)
    {"lat": 15.34, "lon": 73.83, "name": "Goa Coast",          "risk": "Medium"},
    {"lat": 19.07, "lon": 72.87, "name": "Mumbai Coast",       "risk": "Medium"},
    {"lat": 23.02, "lon": 72.57, "name": "Gujarat Coast",      "risk": "High"},
    {"lat": 22.30, "lon": 69.66, "name": "Kutch Coast",        "risk": "High"},
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_cyclone_risk(lat, lon):
    dists   = [haversine(lat, lon, c["lat"], c["lon"]) for c in CYCLONE_PRONE]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)

    if min_d < 50:   return "Very High", min_d
    elif min_d < 150: return "High",      min_d
    elif min_d < 300: return "Medium",    min_d
    else:             return "Low",       min_d

# ══════════════════════════════════════════
# 2. INDIA CLIMATE ZONE CLASSIFICATION
# ══════════════════════════════════════════
def get_climate_zone(lat, lon):
    # Based on IMD (India Meteorological Dept) zones
    if lat > 32:
        return "Alpine/Sub-Alpine"        # Himalayan
    elif lat > 28:
        if lon < 76:
            return "Semi-Arid"            # Punjab/Haryana dry
        else:
            return "Humid Subtropical"    # UP/Bihar
    elif lat > 23:
        if lon < 70:
            return "Arid"                 # Rajasthan desert
        elif lon < 76:
            return "Semi-Arid"
        else:
            return "Humid Subtropical"
    elif lat > 18:
        if lon < 74:
            return "Semi-Arid"            # Maharashtra interior
        else:
            return "Tropical Wet & Dry"   # Deccan
    elif lat > 12:
        if lon < 76:
            return "Tropical Wet & Dry"   # Karnataka
        else:
            return "Tropical Coastal"     # AP/TN coast
    else:
        return "Tropical Humid"           # Kerala/South TN

# ══════════════════════════════════════════
# 3. MONSOON INTENSITY
# ══════════════════════════════════════════
def get_monsoon_intensity(lat, lon, annual_rain):
    # SW Monsoon (June-Sep) dominant regions
    if lon < 77 and lat < 15:
        zone = "SW Monsoon Heavy"         # Kerala/Karnataka coast
        intensity = "Very High" if annual_rain > 2500 else "High"
    elif 8 < lat < 22 and 80 < lon < 88:
        zone = "NE Monsoon"               # TN/AP coast
        intensity = "High" if annual_rain > 1200 else "Medium"
    elif lat > 20 and 86 < lon < 92:
        zone = "Bay of Bengal Monsoon"
        intensity = "Very High" if annual_rain > 2000 else "High"
    elif lat > 25 and lon < 72:
        zone = "Low Monsoon"              # Rajasthan
        intensity = "Low"
    else:
        zone = "Normal Monsoon"
        intensity = "Medium" if annual_rain > 800 else "Low"

    return zone, intensity

# ══════════════════════════════════════════
# 4. EXTREME HEAT DAYS ESTIMATE
# ══════════════════════════════════════════
def estimate_extreme_heat_days(max_temp, lat):
    # Days above 40°C per year estimate
    if max_temp > 48:     return 60, "Critical"
    elif max_temp > 45:   return 40, "Very High"
    elif max_temp > 42:   return 20, "High"
    elif max_temp > 40:   return 10, "Medium"
    else:                 return 0,  "Low"

# ══════════════════════════════════════════
# 5. DROUGHT RISK
# ══════════════════════════════════════════
def get_drought_risk(annual_rain, lat, lon):
    if annual_rain < 400:
        return "Very High"    # Rajasthan, Kutch
    elif annual_rain < 700:
        return "High"         # Semi-arid zones
    elif annual_rain < 1000:
        return "Medium"
    else:
        return "Low"          # Coastal / high rainfall

# ══════════════════════════════════════════
# 6. LIGHTNING DENSITY
# ══════════════════════════════════════════
def get_lightning_risk(lat, lon):
    # High lightning zones in India
    if 20 < lat < 27 and 80 < lon < 90:
        return "Very High"    # Odisha/Jharkhand/WB — highest in India
    elif 22 < lat < 26 and 85 < lon < 92:
        return "High"
    elif lat > 25 and lon > 88:
        return "High"         # Northeast India
    elif 15 < lat < 20 and 73 < lon < 80:
        return "Medium"       # Maharashtra/MP
    else:
        return "Low"

# ══════════════════════════════════════════
# 7. FOG DAYS ESTIMATE
# ══════════════════════════════════════════
def estimate_fog_days(lat, lon, min_temp):
    # Indo-Gangetic Plain = high fog
    if lat > 25 and lat < 32 and lon > 75 and lon < 88:
        return 40, "High"     # IGP fog belt
    elif lat > 28 and min_temp < 10:
        return 20, "Medium"
    else:
        return 2,  "Low"

# ══════════════════════════════════════════
# 8. HEAT INDEX (Feels Like)
# ══════════════════════════════════════════
def calculate_heat_index(temp, humidity):
    # Rothfusz regression formula
    try:
        hi = (-42.379 + 2.04901523*temp + 10.14333127*humidity
              - 0.22475541*temp*humidity - 0.00683783*temp**2
              - 0.05481717*humidity**2 + 0.00122874*temp**2*humidity
              + 0.00085282*temp*humidity**2 - 0.00000199*temp**2*humidity**2)
        if hi > 54:   return round(hi,1), "Extreme Danger"
        elif hi > 41: return round(hi,1), "Danger"
        elif hi > 32: return round(hi,1), "Extreme Caution"
        else:         return round(hi,1), "Caution"
    except:
        return None, "Unknown"

# ══════════════════════════════════════════
# 9. NASA POWER API — Full Parameters
# ══════════════════════════════════════════
def get_nasa_climate(loc):
    url    = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,WS10M_MAX,WS10M,RH2M,FROST_DAYS,ALLSKY_SFC_UV_INDEX",
        "community" : "RE",
        "longitude" : loc["longitude"],
        "latitude"  : loc["latitude"],
        "format"    : "JSON"
    }
    r     = requests.get(url, params=params, timeout=30)
    data  = r.json()
    return data["properties"]["parameter"]

# ══════════════════════════════════════════
# 10. FINAL CLIMATE CONSTRUCTION SCORE
# ══════════════════════════════════════════
def calculate_climate_score(annual_rain, max_wind, humidity, frost_days,
                             max_temp, min_temp, cyclone_risk,
                             monsoon_intensity, drought_risk,
                             extreme_heat_cat, lightning_risk, fog_risk):
    score = 100

    # Rainfall
    if annual_rain > 3000:    score -= 20
    elif annual_rain > 2000:  score -= 10
    elif annual_rain < 300:   score -= 15
    elif annual_rain < 500:   score -= 8

    # Wind
    if max_wind > 20:         score -= 15
    elif max_wind > 15:       score -= 8

    # Humidity
    if humidity > 90:         score -= 15
    elif humidity > 85:       score -= 8

    # Frost
    if frost_days > 30:       score -= 15
    elif frost_days > 10:     score -= 8

    # Temperature extremes
    if max_temp > 48:         score -= 15
    elif max_temp > 45:       score -= 8
    if min_temp < 0:          score -= 10
    elif min_temp < 5:        score -= 5

    # Cyclone
    cyc_map = {"Low":0, "Medium":-10, "High":-20, "Very High":-30}
    score += cyc_map.get(cyclone_risk, 0)

    # Monsoon
    mon_map = {"Low":0, "Medium":-5, "High":-10, "Very High":-15}
    score += mon_map.get(monsoon_intensity, 0)

    # Drought
    dro_map = {"Low":0, "Medium":-5, "High":-10, "Very High":-15}
    score += dro_map.get(drought_risk, 0)

    # Extreme heat
    heat_map = {"Low":0, "Medium":-5, "High":-10, "Very High":-15, "Critical":-20}
    score += heat_map.get(extreme_heat_cat, 0)

    # Lightning
    lig_map = {"Low":0, "Medium":-5, "High":-10, "Very High":-15}
    score += lig_map.get(lightning_risk, 0)

    # Fog
    fog_map = {"Low":0, "Medium":-3, "High":-8}
    score += fog_map.get(fog_risk, 0)

    return max(0, min(100, round(score, 1)))

# ══════════════════════════════════════════
# MAIN PROCESS FUNCTION
# ══════════════════════════════════════════
def process_location(loc):
    try:
        props = get_nasa_climate(loc)

        lat = loc["latitude"]
        lon = loc["longitude"]

        annual_rain = round(sum(props["PRECTOTCORR"].values()), 2)
        max_wind    = round(max(props["WS10M_MAX"].values()), 2)
        avg_wind    = round(sum(props["WS10M"].values()) / 12, 2)
        humidity    = round(sum(props["RH2M"].values()) / 12, 2)
        frost_days  = round(sum(props["FROST_DAYS"].values()), 2)
        avg_temp    = round(sum(props["T2M"].values()) / 12, 2)
        max_temp    = round(max(props["T2M_MAX"].values()), 2)
        min_temp    = round(min(props["T2M_MIN"].values()), 2)
        max_rain    = round(max(props["PRECTOTCORR"].values()), 2)

        try:
            uv_vals = props.get("ALLSKY_SFC_UV_INDEX", {})
            avg_uv  = round(sum(uv_vals.values()) / len(uv_vals), 2) if uv_vals else None
        except:
            avg_uv = None

        # All derived factors
        cyclone_risk, cyclone_dist       = get_cyclone_risk(lat, lon)
        climate_zone                      = get_climate_zone(lat, lon)
        monsoon_zone, monsoon_intensity   = get_monsoon_intensity(lat, lon, annual_rain)
        extreme_heat_days, extreme_heat_cat = estimate_extreme_heat_days(max_temp, lat)
        drought_risk                      = get_drought_risk(annual_rain, lat, lon)
        lightning_risk                    = get_lightning_risk(lat, lon)
        fog_days, fog_risk                = estimate_fog_days(lat, lon, min_temp)
        heat_index, heat_index_cat        = calculate_heat_index(avg_temp, humidity)

        climate_score = calculate_climate_score(
            annual_rain, max_wind, humidity, frost_days,
            max_temp, min_temp, cyclone_risk,
            monsoon_intensity, drought_risk,
            extreme_heat_cat, lightning_risk, fog_risk
        )

        row = {
            "location_id"               : loc["location_id"],
            "latitude"                  : lat,
            "longitude"                 : lon,

            # NASA Data
            "avg_temp_C"                : avg_temp,
            "max_temp_C"                : max_temp,
            "min_temp_C"                : min_temp,
            "temp_range_C"              : round(max_temp - min_temp, 2),
            "annual_rainfall_mm"        : annual_rain,
            "max_monthly_rain_mm"       : max_rain,
            "avg_wind_speed_ms"         : avg_wind,
            "max_wind_speed_ms"         : max_wind,
            "avg_humidity_percent"      : humidity,
            "frost_days_per_year"       : frost_days,
            "avg_uv_index"              : avg_uv,

            # Derived Factors
            "climate_zone"              : climate_zone,
            "cyclone_risk"              : cyclone_risk,
            "nearest_cyclone_zone_km"   : cyclone_dist,
            "monsoon_zone"              : monsoon_zone,
            "monsoon_intensity"         : monsoon_intensity,
            "extreme_heat_days_per_year": extreme_heat_days,
            "extreme_heat_category"     : extreme_heat_cat,
            "drought_risk"              : drought_risk,
            "lightning_risk"            : lightning_risk,
            "estimated_fog_days"        : fog_days,
            "fog_risk"                  : fog_risk,
            "heat_index_C"              : heat_index,
            "heat_index_category"       : heat_index_cat,

            # Final Score
            "climate_construction_score": climate_score
        }

        with results_lock:
            results.append(row)
            counter[0] += 1
            c = counter[0]
            if c % 50 == 0:
                pct = round(c / len(locations) * 100, 1)
                print(f"✅ {c}/{len(locations)} — {pct}% — Score: {climate_score}")
            if c % 200 == 0:
                pd.DataFrame(results).to_csv("india_climate_checkpoint.csv", index=False)
                print(f"💾 Checkpoint saved — {c} rows")

    except Exception as e:
        with results_lock:
            failed.append(loc["location_id"])

# ══════════════════════════════════════════
# RUN
# ══════════════════════════════════════════
results_lock = threading.Lock()
results  = []
failed   = []
counter  = [0]

def main():
    print(f"\n🚀 Starting India Climate Collection...")
    print(f"📍 Locations : {len(locations)}")
    print(f"⏱️ Estimated  : ~{round(len(locations)*1.2/4/60,0)} minutes\n")

    start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_location, loc): loc for loc in locations}
        for _future in as_completed(futures):
            pass

    elapsed = round((time.time() - start) / 60, 1)

    # Save
    df_climate = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, "india_climate_data.csv")
    df_climate.to_csv(out_path, index=False)

    print(f"\n🎉 COMPLETE in {elapsed} minutes!")
    print(f"✅ Success  : {len(results)}")
    print(f"❌ Failed   : {len(failed)}")
    print(f"\n📊 Climate Score:")
    print(df_climate["climate_construction_score"].describe())
    print(f"\n🌡️ Climate Zones:")
    print(df_climate["climate_zone"].value_counts())
    print(f"\n🌀 Cyclone Risk:")
    print(df_climate["cyclone_risk"].value_counts())
    print(f"\n🌧️ Monsoon Intensity:")
    print(df_climate["monsoon_intensity"].value_counts())
    print(f"\n⚡ Lightning Risk:")
    print(df_climate["lightning_risk"].value_counts())


if __name__ == "__main__":
    main()
