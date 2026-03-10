# merge_bmtpc_labels.py -- Spatially merge BMTPC labels into master dataset

import os
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_MASTER = os.path.join(BASE_DIR, "india_master_dataset (1).csv")
DEFAULT_BMTPC = os.path.join(DATA_DIR, "bmtpc_failure_labels.csv")
DEFAULT_OUT = os.path.join(BASE_DIR, "india_master_dataset.csv")


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def _infer_lat_lon(df):
    lat_col = "latitude" if "latitude" in df.columns else "lat" if "lat" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else "lon" if "lon" in df.columns else None
    return lat_col, lon_col


def _compute_severity(row):
    success = row.get("success_label")
    failure_type = str(row.get("failure_type") or "").strip().lower()
    if success in [0, "0", "false", "False"]:
        return 1.0
    if failure_type and failure_type not in {"none", "na", "n/a"}:
        return 1.0
    return 0.0


def merge(master_path, bmtpc_path, output_path, radius_km=25.0):
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master dataset not found: {master_path}")
    if not os.path.exists(bmtpc_path):
        raise FileNotFoundError(f"BMTPC labels not found: {bmtpc_path}")

    master = pd.read_csv(master_path)
    bmtpc = pd.read_csv(bmtpc_path)

    mlat_col, mlon_col = _infer_lat_lon(master)
    blat_col, blon_col = _infer_lat_lon(bmtpc)
    if not mlat_col or not mlon_col:
        raise ValueError("Master dataset missing latitude/longitude columns")
    if not blat_col or not blon_col:
        raise ValueError("BMTPC labels missing latitude/longitude columns")

    if bmtpc.empty:
        master["bmtpc_failure_nearest_km"] = None
        master["bmtpc_failure_count_25km"] = 0
        master["bmtpc_failure_severity_index"] = 0
        master.to_csv(output_path, index=False)
        return

    bmtpc = bmtpc.copy()
    bmtpc["severity_index"] = bmtpc.apply(_compute_severity, axis=1)

    b_lat = bmtpc[blat_col].astype(float).to_numpy()
    b_lon = bmtpc[blon_col].astype(float).to_numpy()
    b_sev = bmtpc["severity_index"].to_numpy()

    nearest_km = []
    count_25 = []
    sev_index = []

    for _, row in master.iterrows():
        mlat = float(row[mlat_col])
        mlon = float(row[mlon_col])
        dists = _haversine_km(mlat, mlon, b_lat, b_lon)
        min_km = float(np.min(dists)) if len(dists) else None
        within = dists <= radius_km
        cnt = int(np.sum(within)) if len(dists) else 0
        sev = float(np.max(b_sev[within])) if np.any(within) else 0.0
        nearest_km.append(round(min_km, 2) if min_km is not None else None)
        count_25.append(cnt)
        sev_index.append(round(sev, 2))

    master["bmtpc_failure_nearest_km"] = nearest_km
    master["bmtpc_failure_count_25km"] = count_25
    master["bmtpc_failure_severity_index"] = sev_index

    master.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=DEFAULT_MASTER, help="Master dataset CSV path")
    parser.add_argument("--bmtpc", default=DEFAULT_BMTPC, help="BMTPC labels CSV path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output merged dataset CSV path")
    parser.add_argument("--radius-km", type=float, default=25.0, help="Radius for label counting")
    args = parser.parse_args()

    merge(args.master, args.bmtpc, args.out, args.radius_km)
    print(f"Merged dataset saved to: {args.out}")
