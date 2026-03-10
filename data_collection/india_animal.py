# ============================================
# INDIA ANIMAL ACTIVITY DATA — COMPLETE
# + BUILDING SUCCESS/FAILURE LABELS
# ============================================

import requests
import pandas as pd
import numpy as np
import math
import time
import threading
import warnings
warnings.filterwarnings('ignore')
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
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ══════════════════════════════════════════
# 1. NATIONAL PARKS & WILDLIFE SANCTUARIES
#    (India — 106 National Parks)
# ══════════════════════════════════════════
INDIA_PROTECTED = [
    # South India
    {"name":"Mudumalai NP",           "lat":11.60, "lon":76.60, "type":"National Park"},
    {"name":"Anamalai Tiger Reserve", "lat":10.30, "lon":77.00, "type":"Tiger Reserve"},
    {"name":"Kalakkad Mundanthurai",  "lat":8.70,  "lon":77.30, "type":"Tiger Reserve"},
    {"name":"Sathyamangalam TR",      "lat":11.50, "lon":77.20, "type":"Tiger Reserve"},
    {"name":"Guindy NP",              "lat":13.00, "lon":80.20, "type":"National Park"},
    {"name":"Gulf of Mannar Marine",  "lat":9.10,  "lon":79.10, "type":"Marine Park"},
    {"name":"Nagarhole NP",           "lat":12.10, "lon":76.10, "type":"National Park"},
    {"name":"Bandipur NP",            "lat":11.70, "lon":76.60, "type":"National Park"},
    {"name":"Bhadra TR",              "lat":13.50, "lon":75.60, "type":"Tiger Reserve"},
    {"name":"Periyar TR",             "lat":9.50,  "lon":77.20, "type":"Tiger Reserve"},
    {"name":"Silent Valley NP",       "lat":11.10, "lon":76.50, "type":"National Park"},
    {"name":"Eravikulam NP",          "lat":10.10, "lon":77.10, "type":"National Park"},
    # Central India
    {"name":"Kanha TR",               "lat":22.30, "lon":80.60, "type":"Tiger Reserve"},
    {"name":"Pench TR",               "lat":21.70, "lon":79.30, "type":"Tiger Reserve"},
    {"name":"Satpura NP",             "lat":22.50, "lon":78.30, "type":"National Park"},
    {"name":"Bandhavgarh TR",         "lat":23.70, "lon":81.00, "type":"Tiger Reserve"},
    {"name":"Panna TR",               "lat":24.70, "lon":80.00, "type":"Tiger Reserve"},
    {"name":"Tadoba TR",              "lat":20.20, "lon":79.30, "type":"Tiger Reserve"},
    # North India
    {"name":"Jim Corbett NP",         "lat":29.50, "lon":78.80, "type":"National Park"},
    {"name":"Rajaji NP",              "lat":30.00, "lon":78.20, "type":"National Park"},
    {"name":"Kaziranga NP",           "lat":26.60, "lon":93.40, "type":"National Park"},
    {"name":"Manas NP",               "lat":26.70, "lon":90.70, "type":"National Park"},
    {"name":"Sundarbans TR",          "lat":21.90, "lon":89.00, "type":"Tiger Reserve"},
    {"name":"Ranthambore TR",         "lat":26.00, "lon":76.50, "type":"Tiger Reserve"},
    {"name":"Sariska TR",             "lat":27.30, "lon":76.40, "type":"Tiger Reserve"},
    # West India
    {"name":"Gir NP",                 "lat":21.10, "lon":70.80, "type":"National Park"},
    {"name":"Blackbuck NP Velavadar", "lat":22.00, "lon":72.20, "type":"National Park"},
    {"name":"Marine NP Jamnagar",     "lat":22.50, "lon":70.50, "type":"Marine Park"},
    # Northeast
    {"name":"Namdapha NP",            "lat":27.50, "lon":96.40, "type":"National Park"},
    {"name":"Dibru Saikhowa NP",      "lat":27.50, "lon":95.30, "type":"National Park"},
]

def check_protected_area(lat, lon):
    dists   = [haversine(lat,lon,p["lat"],p["lon"]) for p in INDIA_PROTECTED]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    ptype   = INDIA_PROTECTED[min_idx]["type"]
    pname   = INDIA_PROTECTED[min_idx]["name"]
    if min_d < 5:    risk = "Very High"  # Inside/adjacent
    elif min_d < 15: risk = "High"       # Eco-sensitive zone
    elif min_d < 30: risk = "Medium"
    else:            risk = "Low"
    return min_d, pname, ptype, risk

# ══════════════════════════════════════════
# 2. TIGER CORRIDORS — India
# ══════════════════════════════════════════
TIGER_CORRIDORS = [
    {"name":"Nilgiris-Eastern Ghats",       "lat":11.40, "lon":76.80},
    {"name":"Anamalai-Parambikulam",        "lat":10.40, "lon":77.10},
    {"name":"Central Indian Corridor",      "lat":22.00, "lon":79.50},
    {"name":"Terai Arc Landscape",          "lat":29.00, "lon":79.50},
    {"name":"Sundarbans Corridor",          "lat":22.00, "lon":88.80},
    {"name":"NE Hills Corridor",            "lat":26.50, "lon":93.00},
    {"name":"Western Ghats Corridor",       "lat":11.00, "lon":76.50},
    {"name":"Satpura-Maikal Corridor",      "lat":22.50, "lon":80.00},
    {"name":"Panna-Pench Corridor",         "lat":23.00, "lon":79.80},
    {"name":"Ranthambore Corridor",         "lat":26.20, "lon":76.80},
]

def check_tiger_corridor(lat, lon):
    dists = [haversine(lat,lon,c["lat"],c["lon"]) for c in TIGER_CORRIDORS]
    min_d = round(min(dists), 2)
    name  = TIGER_CORRIDORS[dists.index(min(dists))]["name"]
    if min_d < 8:    risk = "Very High"
    elif min_d < 20: risk = "High"
    elif min_d < 40: risk = "Medium"
    else:            risk = "Low"
    return min_d, name, risk

# ══════════════════════════════════════════
# 3. ELEPHANT CORRIDORS
# ══════════════════════════════════════════
ELEPHANT_CORRIDORS = [
    {"name":"Nilgiris Corridor",            "lat":11.40, "lon":76.80},
    {"name":"Anamalai Corridor",            "lat":10.40, "lon":77.10},
    {"name":"Sathyamangalam Corridor",      "lat":11.60, "lon":77.30},
    {"name":"Kalakkad Corridor",            "lat":8.80,  "lon":77.40},
    {"name":"Mudumalai-Bandipur",           "lat":11.65, "lon":76.65},
    {"name":"Assam Elephant Corridor",      "lat":26.80, "lon":93.50},
    {"name":"Jharkhand Elephant Belt",      "lat":23.50, "lon":85.50},
    {"name":"Odisha Elephant Corridor",     "lat":21.50, "lon":84.50},
    {"name":"North Bengal Corridor",        "lat":26.80, "lon":89.00},
    {"name":"Eastern Ghats Corridor",       "lat":18.00, "lon":83.00},
]

def check_elephant_corridor(lat, lon):
    dists = [haversine(lat,lon,c["lat"],c["lon"]) for c in ELEPHANT_CORRIDORS]
    min_d = round(min(dists), 2)
    name  = ELEPHANT_CORRIDORS[dists.index(min(dists))]["name"]
    if min_d < 10:   risk = "Very High"
    elif min_d < 25: risk = "High"
    elif min_d < 50: risk = "Medium"
    else:            risk = "Low"
    return min_d, name, risk

# ══════════════════════════════════════════
# 4. BIRD SANCTUARIES & FLYWAYS
# ══════════════════════════════════════════
BIRD_ZONES = [
    {"name":"Bharatpur Keoladeo",           "lat":27.20, "lon":77.50},
    {"name":"Chilika Lake",                 "lat":19.70, "lon":85.30},
    {"name":"Point Calimere",               "lat":10.30, "lon":79.80},
    {"name":"Vedanthangal",                 "lat":12.50, "lon":79.90},
    {"name":"Pulicat Lake",                 "lat":13.50, "lon":80.20},
    {"name":"Koonthankulam",                "lat":8.80,  "lon":77.70},
    {"name":"Nal Sarovar Gujarat",          "lat":22.80, "lon":72.00},
    {"name":"Loktak Lake Manipur",          "lat":24.50, "lon":93.80},
    {"name":"Sambhar Lake Rajasthan",       "lat":26.90, "lon":75.10},
    {"name":"Harike Wetland Punjab",        "lat":31.20, "lon":75.20},
    {"name":"Pichavaram Mangrove",          "lat":11.40, "lon":79.80},
    # Central Asian Flyway staging points
    {"name":"Rann of Kutch Staging",        "lat":23.80, "lon":69.80},
    {"name":"Gujarat Coast Flyway",         "lat":22.00, "lon":72.00},
    {"name":"TN Coast Flyway",              "lat":12.00, "lon":80.10},
]

def check_bird_zone(lat, lon):
    dists = [haversine(lat,lon,b["lat"],b["lon"]) for b in BIRD_ZONES]
    min_d = round(min(dists), 2)
    name  = BIRD_ZONES[dists.index(min(dists))]["name"]
    if min_d < 5:    risk = "High"
    elif min_d < 15: risk = "Medium"
    else:            risk = "Low"
    return min_d, name, risk

# ══════════════════════════════════════════
# 5. ENDANGERED SPECIES HABITATS
# ══════════════════════════════════════════
ENDANGERED_HABITATS = [
    {"name":"Nilgiri Tahr",                 "lat":10.10, "lon":77.20, "species":"Nilgiri Tahr"},
    {"name":"Lion-tailed Macaque",          "lat":10.30, "lon":77.00, "species":"Lion-tailed Macaque"},
    {"name":"Asiatic Lion Gir",             "lat":21.10, "lon":70.80, "species":"Asiatic Lion"},
    {"name":"One-horned Rhino",             "lat":26.60, "lon":93.40, "species":"One-horned Rhino"},
    {"name":"Bengal Tiger Core",            "lat":22.30, "lon":80.60, "species":"Bengal Tiger"},
    {"name":"Snow Leopard Habitat",         "lat":33.00, "lon":77.50, "species":"Snow Leopard"},
    {"name":"Gangetic Dolphin",             "lat":25.40, "lon":83.00, "species":"Gangetic Dolphin"},
    {"name":"Dugong Gulf of Mannar",        "lat":9.20,  "lon":79.20, "species":"Dugong"},
    {"name":"Sea Turtle Chennai",           "lat":13.20, "lon":80.30, "species":"Olive Ridley"},
    {"name":"Great Indian Bustard",         "lat":27.00, "lon":71.00, "species":"Great Indian Bustard"},
    {"name":"Red Panda NE India",           "lat":27.20, "lon":88.60, "species":"Red Panda"},
    {"name":"Hangul Kashmir",               "lat":34.00, "lon":75.00, "species":"Kashmir Stag"},
    {"name":"Irrawaddy Dolphin",            "lat":26.50, "lon":95.00, "species":"Irrawaddy Dolphin"},
    {"name":"Leatherback Turtle",           "lat":12.00, "lon":93.00, "species":"Leatherback Turtle"},
    {"name":"Grizzled Squirrel",            "lat":9.50,  "lon":77.50, "species":"Grizzled Squirrel"},
]

def check_endangered_habitat(lat, lon):
    dists   = [haversine(lat,lon,e["lat"],e["lon"]) for e in ENDANGERED_HABITATS]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    species = ENDANGERED_HABITATS[min_idx]["species"]
    name    = ENDANGERED_HABITATS[min_idx]["name"]
    if min_d < 10:   risk = "Very High"
    elif min_d < 25: risk = "High"
    elif min_d < 50: risk = "Medium"
    else:            risk = "Low"
    return min_d, species, name, risk

# ══════════════════════════════════════════
# 6. HUMAN-ANIMAL CONFLICT ZONES
# ══════════════════════════════════════════
CONFLICT_ZONES = [
    {"name":"Nilgiris Fringe",              "lat":11.40, "lon":76.70},
    {"name":"Coimbatore Forest Border",     "lat":11.10, "lon":76.90},
    {"name":"Assam Tea Garden Conflicts",   "lat":26.50, "lon":93.80},
    {"name":"Jharkhand Elephant Conflict",  "lat":23.30, "lon":85.30},
    {"name":"Odisha Elephant Conflict",     "lat":21.50, "lon":84.80},
    {"name":"Sundarbans Tiger Conflict",    "lat":21.80, "lon":89.10},
    {"name":"Uttarakhand Leopard Conflict", "lat":29.50, "lon":79.00},
    {"name":"Karnataka Forest Fringe",      "lat":12.30, "lon":76.40},
    {"name":"Gujarat Lion Fringe",          "lat":21.30, "lon":70.60},
    {"name":"MP Tiger Conflict Zone",       "lat":22.50, "lon":80.50},
]

def check_conflict_zone(lat, lon):
    dists = [haversine(lat,lon,c["lat"],c["lon"]) for c in CONFLICT_ZONES]
    min_d = round(min(dists), 2)
    name  = CONFLICT_ZONES[dists.index(min(dists))]["name"]
    if min_d < 10:   risk = "High"
    elif min_d < 25: risk = "Medium"
    else:            risk = "Low"
    return min_d, name, risk

# ══════════════════════════════════════════
# 7. MARINE PROTECTED AREAS
# ══════════════════════════════════════════
MARINE_ZONES = [
    {"name":"Gulf of Mannar",               "lat":9.10,  "lon":79.10},
    {"name":"Lakshadweep Marine",           "lat":10.60, "lon":72.60},
    {"name":"Marine NP Jamnagar",           "lat":22.50, "lon":70.50},
    {"name":"Malvan Marine Sanctuary",      "lat":16.10, "lon":73.50},
    {"name":"Andaman Marine",               "lat":12.00, "lon":93.00},
    {"name":"Chilika Marine",               "lat":19.70, "lon":85.30},
    {"name":"Mangrove Bhitarkanika",        "lat":20.70, "lon":87.00},
    {"name":"Pichavaram Mangrove TN",       "lat":11.40, "lon":79.80},
]

def check_marine_zone(lat, lon):
    coastal = lon > 79.0 or lon < 73.5 or lat < 9.0
    if not coastal:
        return "None", 999, "Inland"
    dists   = [haversine(lat,lon,m["lat"],m["lon"]) for m in MARINE_ZONES]
    min_idx = dists.index(min(dists))
    min_d   = round(min(dists), 2)
    name    = MARINE_ZONES[min_idx]["name"]
    if min_d < 5:    risk = "Very High"
    elif min_d < 20: risk = "High"
    elif min_d < 50: risk = "Medium"
    else:            risk = "Low"
    return risk, min_d, name

# ══════════════════════════════════════════
# 8. BIODIVERSITY HOTSPOTS
# ══════════════════════════════════════════
def check_biodiversity_hotspot(lat, lon):
    # Western Ghats
    if lon < 78.0 and 8.0 < lat < 21.0:
        return "Yes", "Western Ghats (UNESCO)"
    # Eastern Himalayas
    elif lat > 26.0 and lon > 88.0:
        return "Yes", "Eastern Himalayas (UNESCO)"
    # Indo-Burma (NE India)
    elif lat > 22.0 and lon > 92.0:
        return "Yes", "Indo-Burma Hotspot"
    # Gulf of Mannar
    elif 8.5 < lat < 10.5 and 78.5 < lon < 79.8:
        return "Yes", "Gulf of Mannar"
    # Sundaland (Andaman)
    elif lat < 14.0 and lon > 92.0:
        return "Yes", "Sundaland (Andaman)"
    else:
        return "No", "None"

# ══════════════════════════════════════════
# 9. BURROWING ANIMAL RISK
# ══════════════════════════════════════════
def get_burrowing_risk(lat, lon, clay_percent=None):
    near_forest = lon < 78.5 and 8 < lat < 25
    high_clay   = clay_percent and clay_percent > 30
    arid_zone   = lat > 24 and lon < 73  # Rajasthan — rodent burrows
    if arid_zone:
        return "High"  # Desert rodents
    elif near_forest and high_clay:
        return "High"
    elif near_forest or high_clay:
        return "Medium"
    else:
        return "Low"

# ══════════════════════════════════════════
# 10. GBIF — Live Animal Occurrences
# ══════════════════════════════════════════
def get_gbif_data(lat, lon):
    try:
        url    = "https://api.gbif.org/v1/occurrence/search"
        params = {
            "decimalLatitude" : f"{lat-0.3},{lat+0.3}",
            "decimalLongitude": f"{lon-0.3},{lon+0.3}",
            "kingdomKey"      : 1,
            "hasCoordinate"   : True,
            "limit"           : 100
        }
        r    = requests.get(url, params=params, timeout=20)
        data = r.json()
        records    = data.get("results", [])
        total      = data.get("count", 0)
        threatened = len([x for x in records if x.get("iucnRedListCategory") in ["CR","EN","VU"]])
        species    = len(set([x.get("speciesKey") for x in records if x.get("speciesKey")]))
        mammals    = len([x for x in records if x.get("class") == "Mammalia"])
        birds      = len([x for x in records if x.get("class") == "Aves"])
        return total, threatened, species, mammals, birds
    except:
        return 0, 0, 0, 0, 0

# ══════════════════════════════════════════
# ANIMAL CONSTRUCTION SCORE
# ══════════════════════════════════════════
def calculate_animal_score(pa_risk, tiger_risk, elephant_risk,
                            bird_risk, endangered_risk, conflict_risk,
                            marine_risk, is_hotspot, burrowing_risk,
                            threatened_count):
    score = 100
    score += {"Low":0,"Medium":-15,"High":-25,"Very High":-40}.get(pa_risk, 0)
    score += {"Low":0,"Medium":-10,"High":-20,"Very High":-30}.get(tiger_risk, 0)
    score += {"Low":0,"Medium":-10,"High":-20,"Very High":-25}.get(elephant_risk, 0)
    score += {"Low":0,"Medium":-5, "High":-12}.get(bird_risk, 0)
    score += {"Low":0,"Medium":-8, "High":-15,"Very High":-20}.get(endangered_risk, 0)
    score += {"Low":0,"Medium":-5, "High":-15}.get(conflict_risk, 0)
    score += {"None":0,"Low":-3,  "Medium":-10,"High":-18,"Very High":-25}.get(marine_risk, 0)
    if is_hotspot == "Yes": score -= 10
    score += {"Low":0,"Medium":-3,"High":-8}.get(burrowing_risk, 0)
    if threatened_count > 5:   score -= 10
    elif threatened_count > 2: score -= 5
    return max(0, min(100, round(score, 1)))

# ══════════════════════════════════════════
# BUILDING SUCCESS / FAILURE LABEL
# Method: USGS + OSM + Geography combine
# ══════════════════════════════════════════
def get_building_success_label(lat, lon):
    score  = 0
    weight = 0

    # ── SOURCE 1: USGS Earthquake damage history ──
    try:
        url    = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format"      : "geojson",
            "latitude"    : lat, "longitude": lon,
            "maxradiuskm" : 150,
            "minmagnitude": 5.0,
            "starttime"   : "1990-01-01",
            "endtime"     : "2024-01-01",
        }
        r       = requests.get(url, params=params, timeout=20)
        data    = r.json()
        mags    = [f["properties"]["mag"] for f in data["features"] if f["properties"]["mag"]]
        max_mag = max(mags) if mags else 0
        count   = len(mags)

        if max_mag >= 7.0 or count > 20: usgs_score = 0    # High damage history
        elif max_mag >= 6.0 or count > 10: usgs_score = 30
        elif max_mag >= 5.0:              usgs_score = 60
        else:                              usgs_score = 90  # Safe history

        score  += usgs_score * 0.35  # 35% weight
        weight += 0.35
    except:
        score  += 60 * 0.35
        weight += 0.35

    # ── SOURCE 2: OSM Real Buildings ──
    # Locations near cities = proven construction success
    MAJOR_CITIES = [
        (28.67, 77.21), (19.08, 72.88), (12.97, 77.59),
        (22.57, 88.36), (17.38, 78.48), (13.08, 80.27),
        (23.02, 72.57), (26.91, 75.79), (21.25, 81.63),
        (11.01, 76.96), (9.93,  78.12), (10.79, 78.70),
        (15.34, 75.14), (16.51, 80.64), (20.27, 85.84),
        (25.59, 85.14), (26.85, 80.95), (22.32, 87.32),
    ]
    city_dists   = [haversine(lat,lon,c[0],c[1]) for c in MAJOR_CITIES]
    min_city_d   = min(city_dists)

    if min_city_d < 10:   osm_score = 95   # Major city — proven
    elif min_city_d < 30: osm_score = 80
    elif min_city_d < 60: osm_score = 65
    elif min_city_d < 100: osm_score = 55
    else:                  osm_score = 45

    score  += osm_score * 0.35   # 35% weight
    weight += 0.35

    # ── SOURCE 3: Geography Success Factors ──
    geo_score = 70  # Base

    # Flat terrain — construction friendly
    if 15 < lat < 28 and 74 < lon < 85:  geo_score += 15  # Deccan/IGP
    # Coastal — risky
    if lon > 79.5 or lon < 72.5:         geo_score -= 20
    # Himalayan — very risky
    if lat > 32:                          geo_score -= 35
    # Flood plains
    if 24 < lat < 28 and 80 < lon < 88:  geo_score -= 10
    # Western Ghats — difficult terrain
    if lon < 77.5 and 8 < lat < 15:      geo_score -= 15
    # Earthquake zones
    if lat > 30 or (lat > 24 and lon > 90): geo_score -= 20

    geo_score = max(0, min(100, geo_score))

    score  += geo_score * 0.30   # 30% weight
    weight += 0.30

    final_score = round(score / weight, 1) if weight > 0 else 50

    # Label
    if final_score >= 70:   label, category = 1,   "Success"
    elif final_score >= 45: label, category = 0.5, "Moderate"
    else:                   label, category = 0,   "High Risk"

    return label, category, round(final_score, 1)

# ══════════════════════════════════════════
# MAIN PROCESS
# ══════════════════════════════════════════
results_lock = threading.Lock()
results  = []
failed   = []
counter  = [0]
clay_map = dict(zip(df_soil["location_id"], df_soil.get("clay_percent", pd.Series(dtype=float))))

def process_location(loc):
    try:
        lat = loc["latitude"]
        lon = loc["longitude"]
        lid = loc["location_id"]

        # Animal factors
        pa_dist, pa_name, pa_type, pa_risk        = check_protected_area(lat, lon)
        ti_dist, ti_name, ti_risk                 = check_tiger_corridor(lat, lon)
        el_dist, el_name, el_risk                 = check_elephant_corridor(lat, lon)
        bi_dist, bi_name, bi_risk                 = check_bird_zone(lat, lon)
        en_dist, en_species, en_name, en_risk     = check_endangered_habitat(lat, lon)
        cf_dist, cf_name, cf_risk                 = check_conflict_zone(lat, lon)
        ma_risk, ma_dist, ma_name                 = check_marine_zone(lat, lon)
        is_hotspot, hotspot_name                  = check_biodiversity_hotspot(lat, lon)
        clay_val                                   = clay_map.get(lid)
        burrowing_risk                             = get_burrowing_risk(lat, lon, clay_val)
        total, threatened, species, mammals, birds = get_gbif_data(lat, lon)

        animal_score = calculate_animal_score(
            pa_risk, ti_risk, el_risk, bi_risk,
            en_risk, cf_risk, ma_risk,
            is_hotspot, burrowing_risk, threatened
        )

        # Building success label
        b_label, b_category, b_score = get_building_success_label(lat, lon)

        row = {
            "location_id"                  : lid,
            "latitude"                     : lat,
            "longitude"                    : lon,

            # GBIF Live
            "total_animal_records"         : total,
            "threatened_species_count"     : threatened,
            "unique_species_count"         : species,
            "mammal_count"                 : mammals,
            "bird_count"                   : birds,

            # Protected Area
            "nearest_protected_area_km"    : pa_dist,
            "nearest_protected_area"       : pa_name,
            "protected_area_type"          : pa_type,
            "protected_area_risk"          : pa_risk,

            # Tiger Corridor
            "nearest_tiger_corridor_km"    : ti_dist,
            "nearest_tiger_corridor"       : ti_name,
            "tiger_corridor_risk"          : ti_risk,

            # Elephant Corridor
            "nearest_elephant_corridor_km" : el_dist,
            "nearest_elephant_corridor"    : el_name,
            "elephant_corridor_risk"       : el_risk,

            # Bird Zones
            "nearest_bird_zone_km"         : bi_dist,
            "nearest_bird_zone"            : bi_name,
            "bird_zone_risk"               : bi_risk,

            # Endangered
            "nearest_endangered_habitat_km": en_dist,
            "nearest_endangered_species"   : en_species,
            "endangered_habitat_name"      : en_name,
            "endangered_habitat_risk"      : en_risk,

            # Conflict
            "nearest_conflict_zone_km"     : cf_dist,
            "nearest_conflict_zone"        : cf_name,
            "human_animal_conflict_risk"   : cf_risk,

            # Marine
            "marine_protected_area_risk"   : ma_risk,
            "nearest_marine_area_km"       : ma_dist,
            "nearest_marine_area"          : ma_name,

            # Biodiversity
            "biodiversity_hotspot"         : is_hotspot,
            "hotspot_name"                 : hotspot_name,

            # Burrowing
            "burrowing_animal_risk"        : burrowing_risk,

            # Animal Score
            "animal_construction_score"    : animal_score,

            # ── BUILDING SUCCESS LABEL ──
            "construction_success_label"   : b_label,
            "construction_success_category": b_category,
            "construction_viability_score" : b_score,
        }

        with results_lock:
            results.append(row)
            counter[0] += 1
            c = counter[0]
            if c % 50 == 0:
                pct = round(c/len(locations)*100, 1)
                print(f"✅ {c}/{len(locations)} — {pct}% — Animal:{animal_score} Label:{b_category}")
            if c % 200 == 0:
                pd.DataFrame(results).to_csv("india_animal_checkpoint.csv", index=False)
                print(f"💾 Checkpoint saved — {c} rows")

    except Exception as e:
        with results_lock:
            failed.append(loc["location_id"])

def main():
    print(f"\n🚀 Starting India Animal Activity + Success Labels...")
    print(f"📍 Locations : {len(locations)}")
    print(f"⏱️ Estimated  : ~{round(len(locations)*1.5/4/60,0)} minutes\n")

    start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_location, loc): loc for loc in locations}
        for _future in as_completed(futures):
            pass

    elapsed = round((time.time() - start) / 60, 1)

    df_animal = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, "india_animal_data.csv")
    df_animal.to_csv(out_path, index=False)

    print(f"\n🎉 COMPLETE in {elapsed} minutes!")
    print(f"✅ Success  : {len(results)}")
    print(f"❌ Failed   : {len(failed)}")
    print(f"\n📊 Animal Score:")
    print(df_animal["animal_construction_score"].describe())
    print(f"\n🏗️ Building Success Labels:")
    print(df_animal["construction_success_category"].value_counts())
    print(f"\n⚠️ Risk Summary:")
    print("Protected Area  :", df_animal["protected_area_risk"].value_counts().to_dict())
    print("Tiger Corridor  :", df_animal["tiger_corridor_risk"].value_counts().to_dict())
    print("Elephant        :", df_animal["elephant_corridor_risk"].value_counts().to_dict())
    print("Bird Zone       :", df_animal["bird_zone_risk"].value_counts().to_dict())
    print("Endangered      :", df_animal["endangered_habitat_risk"].value_counts().to_dict())
    print("Marine          :", df_animal["marine_protected_area_risk"].value_counts().to_dict())
    print("Biodiversity    :", df_animal["biodiversity_hotspot"].value_counts().to_dict())
    print("Burrowing       :", df_animal["burrowing_animal_risk"].value_counts().to_dict())


if __name__ == "__main__":
    main()

