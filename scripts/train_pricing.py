
# # scripts/train_pricing.py
from __future__ import annotations

import os
import argparse
import json
import sqlite3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import joblib

# --- Defaults (per .env optional überschreibbar) ---
DEFAULT_DB = os.environ.get("DB_PATH", "./data/hausverwaltung.db")
DEFAULT_MODEL_PATH = os.environ.get("PRICING_MODEL_PATH", "./data/pricing_model.pkl")
DEFAULT_METRICS_PATH = "./models/metrics.json"

# --- Features: numerisch + kategorisch (bewusst geringe Kardinalität) ---
NUM_FEATURES = [
    "area_sqm",
    "rooms",
    "floor",
    "year_built",
    "numberOfFloors",
    "building_age",
    "rooms_per_100qm",
    "floor_ratio",
]
CAT_FEATURES = [
    "zip3",
    "zip2",
    "condition",
    "typeOfFlat",
]
TARGET = "rent_cold"


def load_from_sqlite(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM listings_raw", con)
    finally:
        con.close()
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Bereitet das Roh-DF aus listings_raw für das Training vor."""
    # --- PLZ-Buckets aus zip_code ableiten ---
    if "zip_code" in df.columns:
        s = df["zip_code"].astype("string")
        df["zip3"] = s.str.extract(r"(\d{3})", expand=False)
        df["zip2"] = s.str.extract(r"(\d{2})", expand=False)
    else:
        df["zip3"] = None
        df["zip2"] = None

    # --- Felder aus Rohdaten übernehmen/typisieren, falls vorhanden ---
    # (Falls Spalten fehlen, werden sie weiter unten beim Spaltenfilter ignoriert.)
    if "numberOfFloors" in df.columns:
        df["numberOfFloors"] = pd.to_numeric(df["numberOfFloors"], errors="coerce")
    else:
        df["numberOfFloors"] = np.nan

    if "typeOfFlat" in df.columns:
        df["typeOfFlat"] = df["typeOfFlat"].astype("string").str.lower()
    else:
        df["typeOfFlat"] = None

    # Abgeleitete Features
    if "year_built" in df.columns:
        year_built_num = pd.to_numeric(df["year_built"], errors="coerce")
    elif "yearConstructed" in df.columns:
        year_built_num = pd.to_numeric(df["yearConstructed"], errors="coerce")
    else:
        year_built_num = pd.Series(np.nan, index=df.index, dtype="float")

    df["year_built"] = year_built_num
    df["building_age"] = 2025 - year_built_num

    # Basisfelder sicherstellen (rooms/area_sqm/floor)
    if "rooms" in df.columns:
        df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    if "area_sqm" in df.columns:
        df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce")
    if "floor" in df.columns:
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

    # weitere abgeleitete Felder
    df["rooms_per_100qm"] = df["rooms"] / (df["area_sqm"] / 100.0)
    df["floor_ratio"] = np.where(
        (df["numberOfFloors"].notna()) & (pd.to_numeric(df["numberOfFloors"], errors="coerce") > 0),
        df["floor"] / df["numberOfFloors"],
        np.nan,
    )

    # --- nur relevante Spalten behalten ---
    cols = NUM_FEATURES + CAT_FEATURES + [TARGET]
    keep = [c for c in cols if c in df.columns]
    df = df.loc[:, keep].copy()

    # --- Plausibilitäten ---
    if "area_sqm" in df.columns and "rent_cold" in df.columns:
        df = df[
            (df["area_sqm"] > 10) & (df["area_sqm"] < 400) &
            (df["rent_cold"] > 200) & (df["rent_cold"] < 10000)
        ].copy()

    # --- leichte Imputation (zusätzlich zur Pipeline) ---
    if "condition" in df.columns:
        df["condition"] = df["condition"].fillna("unknown")
    if "year_built" in df.columns and df["year_built"].isna().any():
        df["year_built"] = df["year_built"].fillna(df["year_built"].median())
    if "floor" in df.columns and df["floor"].isna().any():
        df["floor"] = df["floor"].fillna(0)
    if "rooms" in df.columns and df["rooms"].isna().any():
        df["rooms"] = df["rooms"].fillna(df["rooms"].median())

    return df


def build_pipeline(verbose: int = 0) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), [c for c in NUM_FEATURES]),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [c for c in CAT_FEATURES],
            ),
        ],
        remainder="drop",
        n_jobs=None,
    )

    model = RandomForestRegressor(
        n_estimators=200,       # schneller als 400, gute Baseline
        min_samples_leaf=2,     # glättet, beschleunigt
        random_state=42,
        n_jobs=-1,
        verbose=verbose,
    )
    return Pipeline(steps=[("prep", pre), ("rf", model)])


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1, denom)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def main():
    parser = argparse.ArgumentParser(description="Train rental price model")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path (table listings_raw)")
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Path to save trained model (joblib)")
    parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--sample", type=int, default=50000, help="Sample size (0 disables sampling)")
    parser.add_argument("--verbose", type=int, default=1, help="RF verbose level (0,1,2)")
    args = parser.parse_args()

    print(f"DB: {args.db}")
    df = load_from_sqlite(args.db)
    print("Loaded:", df.shape)

    df = basic_clean(df)
    print("After clean:", df.shape)

    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled to: {df.shape}")

    feature_cols = [c for c in NUM_FEATURES + CAT_FEATURES if c in df.columns]
    if TARGET not in df.columns or not feature_cols:
        raise RuntimeError("Fehlende Spalten: target oder features nicht gefunden.")

    X = df[feature_cols]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipe = build_pipeline(verbose=args.verbose)
    print("Fitting model...")
    pipe.fit(X_train, y_train)
    print("Predicting...")
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mdae = median_absolute_error(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / np.clip(y_test, 1e-6, None))) * 100.0
    smape_val = smape(y_test.values, pred)

    print(f"MAE : {mae:.2f} €")
    print(f"MdAE: {mdae:.2f} €")
    print(f"MAPE: {mape:.2f} %   sMAPE: {smape_val:.2f} %")
    print(f"R²  : {r2:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"Saved model → {args.out}")

    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae_eur": float(mae),
        "mdae_eur": float(mdae),
        "mape_pct": float(mape),
        "smape_pct": float(smape_val),
        "r2": float(r2),
        "features_used": feature_cols,
        "target": TARGET,
        "sample": int(args.sample),
    }
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics → {args.metrics}")


if __name__ == "__main__":
    main()
