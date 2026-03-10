# retrain_models.py -- Retrain ensemble with updated datasets

import os
import pickle
import argparse
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_DATASET = os.path.join(BASE_DIR, "india_master_dataset (1).csv")
BMTPC_LABELS = os.path.join(DATA_DIR, "bmtpc_failure_labels.csv")

TARGET_CANDIDATES = {
    "feasibility": ["final_feasibility_score", "feasibility_score", "construction_viability_score"],
    "lifespan": ["predicted_lifespan", "lifespan_years"],
    "success": ["construction_success_label", "success_probability"],
}

LEAKAGE_COLUMNS = {
    "location_id",
    "construction_success_category",
    "construction_viability_score",
    "final_feasibility_score",
    "risk_level",
    "predicted_lifespan",
    "confidence_percent",
}

CITY_BUCKET_DEG = 1.0


def _load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df


def _merge_bmtpc_labels(df):
    if not os.path.exists(BMTPC_LABELS):
        return df
    labels = pd.read_csv(BMTPC_LABELS)
    lat_col = "lat" if "lat" in labels.columns else "latitude" if "latitude" in labels.columns else None
    lon_col = "lon" if "lon" in labels.columns else "longitude" if "longitude" in labels.columns else None
    if not lat_col or not lon_col:
        return df
    # Placeholder join: if dataset already contains BMTPC columns, keep them.
    # Real merge should be spatial; this keeps pipeline stable until labels are expanded.
    for col in ["bmtpc_failure_nearest_km", "bmtpc_failure_count_25km", "bmtpc_failure_severity_index"]:
        if col not in df.columns:
            df[col] = 0
    return df


def _encode_categoricals(df):
    label_encoders = {}
    for col in df.columns:
        if is_object_dtype(df[col]) or is_string_dtype(df[col]):
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("")
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df, label_encoders


def _select_target(df, target_key):
    for col in TARGET_CANDIDATES[target_key]:
        if col in df.columns:
            return col
    return None


def _build_features(df):
    df = df.copy()
    df = _merge_bmtpc_labels(df)

    # Drop rows with missing target values
    selected_targets = {}
    for key in TARGET_CANDIDATES:
        col = _select_target(df, key)
        if col:
            selected_targets[key] = col
            df = df[~df[col].isna()]

    df, label_encoders = _encode_categoricals(df)

    target_cols = set(selected_targets.values())
    feature_cols = [c for c in df.columns if c not in target_cols and c not in LEAKAGE_COLUMNS]
    X = df[feature_cols].fillna(0)
    return X, feature_cols, label_encoders, df, selected_targets


def _city_buckets(df):
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = df["latitude"].astype(float)
        lon = df["longitude"].astype(float)
        lat_bin = (lat / CITY_BUCKET_DEG).round(0).astype(int)
        lon_bin = (lon / CITY_BUCKET_DEG).round(0).astype(int)
        return (lat_bin.astype(str) + "_" + lon_bin.astype(str)).to_numpy()
    return None


def _train_and_eval(X, y, model, cv_splits=5, groups=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    city_r2 = None
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in splitter.split(X, y, groups):
            if hasattr(X, "iloc"):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            y_te_pred = model.predict(X_te)
            city_r2 = r2_score(y_te, y_te_pred)
            break
    return model, r2_train, r2_test, cv_scores, city_r2


def main(dataset_path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = _load_dataset(dataset_path)
    X, feature_list, label_encoders, df, targets = _build_features(df)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    outputs = {}

    city_groups = _city_buckets(df)

    # Feasibility ensemble
    feas_col = targets.get("feasibility")
    if feas_col:
        y = df[feas_col].astype(float)
        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=4,
            min_samples_split=8,
            max_features=0.7,
            random_state=42,
        )
        et = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=3,
            min_samples_split=8,
            max_features=0.7,
            random_state=42,
        )
        xgb = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        rf, rf_r2_train, rf_r2_test, rf_cv, rf_city = _train_and_eval(X, y, rf, groups=city_groups)
        et, et_r2_train, et_r2_test, et_cv, et_city = _train_and_eval(X, y, et, groups=city_groups)
        xgb, xgb_r2_train, xgb_r2_test, xgb_cv, xgb_city = _train_and_eval(X_sc, y, xgb, groups=city_groups)
        outputs["rf"] = rf
        outputs["et"] = et
        outputs["xgb"] = xgb
        print("Feasibility R2 ->")
        print(f"  RF  train: {rf_r2_train:.3f}  test: {rf_r2_test:.3f}  cv(5): {rf_cv.mean():.3f} ± {rf_cv.std():.3f}  city: {rf_city:.3f}")
        print(f"  ET  train: {et_r2_train:.3f}  test: {et_r2_test:.3f}  cv(5): {et_cv.mean():.3f} ± {et_cv.std():.3f}  city: {et_city:.3f}")
        print(f"  XGB train: {xgb_r2_train:.3f}  test: {xgb_r2_test:.3f}  cv(5): {xgb_cv.mean():.3f} ± {xgb_cv.std():.3f}  city: {xgb_city:.3f}")

    # Lifespan model
    life_col = targets.get("lifespan")
    if life_col:
        y = df[life_col].astype(float)
        gb = GradientBoostingRegressor(random_state=42)
        gb, gb_r2_train, gb_r2_test, gb_cv, gb_city = _train_and_eval(X, y, gb, groups=city_groups)
        outputs["gb"] = gb
        print("Lifespan R2 ->")
        print(f"  GB  train: {gb_r2_train:.3f}  test: {gb_r2_test:.3f}  cv(5): {gb_cv.mean():.3f} ± {gb_cv.std():.3f}  city: {gb_city:.3f}")

    # Success probability model
    succ_col = targets.get("success")
    if succ_col:
        y = df[succ_col].astype(float)
        succ = RandomForestRegressor(n_estimators=250, random_state=42)
        succ, succ_r2_train, succ_r2_test, succ_cv, succ_city = _train_and_eval(X, y, succ, groups=city_groups)
        outputs["success"] = succ
        print("Success R2 ->")
        print(f"  RF  train: {succ_r2_train:.3f}  test: {succ_r2_test:.3f}  cv(5): {succ_cv.mean():.3f} ± {succ_cv.std():.3f}  city: {succ_city:.3f}")

    ensemble_weights = {"rf": 0.33, "et": 0.33, "xgb": 0.34}

    with open(os.path.join(MODELS_DIR, "model_feasibility_rf.pkl"), "wb") as f:
        pickle.dump(outputs.get("rf"), f)
    with open(os.path.join(MODELS_DIR, "model_feasibility_et.pkl"), "wb") as f:
        pickle.dump(outputs.get("et"), f)
    with open(os.path.join(MODELS_DIR, "model_feasibility_xgb.pkl"), "wb") as f:
        pickle.dump(outputs.get("xgb"), f)
    with open(os.path.join(MODELS_DIR, "model_lifespan_gb.pkl"), "wb") as f:
        pickle.dump(outputs.get("gb"), f)
    with open(os.path.join(MODELS_DIR, "model_success_rf.pkl"), "wb") as f:
        pickle.dump(outputs.get("success"), f)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(MODELS_DIR, "feature_list.pkl"), "wb") as f:
        pickle.dump(feature_list, f)
    with open(os.path.join(MODELS_DIR, "ensemble_weights.pkl"), "wb") as f:
        pickle.dump(ensemble_weights, f)

    print("Models and artifacts saved to models/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to merged dataset CSV")
    args = parser.parse_args()
    main(args.dataset)
