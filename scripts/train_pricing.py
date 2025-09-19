# # scripts/train_pricing.py

# scripts/train_pricing.py
from __future__ import annotations

import os, argparse, json
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

from pathlib import Path

# Projektroot = eine Ebene über /scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH   = str((PROJECT_ROOT / "data"   / "pricing_model.pkl").resolve())
DEFAULT_METRICS_PATH = str((PROJECT_ROOT / "models" / "metrics.json").resolve())
DEFAULT_TRAIN_CSV    = str((PROJECT_ROOT / "data"   / "clean" / "immo_train_joined.csv").resolve())


# --- Ziel & Features ---
TARGET = "rent_cold"
NUM_FEATURES_BASE = [
    "area_sqm", "rooms", "floor", "pricetrend", "serviceCharge",
    "garden", "lift", "hasKitchen",
]
CAT_FEATURES_BASE = [
    "typeOfFlat", "heatingType_clean", "periode_0005",
    # falls du später 0008-Buckets oder weitere Kategorien hast -> hier ergänzen
]

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1, denom)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def load_train_df(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    # Ziel muss existieren
    if TARGET not in df.columns:
        raise RuntimeError(f"Spalte '{TARGET}' fehlt in {train_csv}")

    # Datentypen setzen
    num_cols = [c for c in NUM_FEATURES_BASE if c in df.columns]
    for c in num_cols + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in CAT_FEATURES_BASE:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Zeilen ohne Ziel entfernen
    before = len(df)
    df = df[df[TARGET].notna()].copy()
    after = len(df)
    if after < before:
        print(f"[Info] {before-after} Zeilen ohne {TARGET} entfernt.")

    return df

def build_pipeline(num_features, cat_features, verbose=1) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_features),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_features),
        ],
        remainder="drop",
        n_jobs=None,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=verbose,
    )
    return Pipeline([("prep", pre), ("rf", model)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV, help="Pfad zum bereinigten Trainings-CSV")
    ap.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Zielpfad für das Modell (joblib)")
    ap.add_argument("--metrics", default=DEFAULT_METRICS_PATH, help="Zielpfad für Metriken (JSON)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=50000, help="0 = kein Sampling")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    print(f"Train CSV: {args.train_csv}")
    df = load_train_df(args.train_csv)
    print("Geladen:", df.shape)

    # Verfügbare Features prüfen
    num_features = [c for c in NUM_FEATURES_BASE if c in df.columns]
    cat_features = [c for c in CAT_FEATURES_BASE if c in df.columns]
    missing_num = [c for c in NUM_FEATURES_BASE if c not in num_features]
    missing_cat = [c for c in CAT_FEATURES_BASE if c not in cat_features]
    if missing_num or missing_cat:
        print("[Warnung] Nicht gefunden:",
              {"num_missing": missing_num, "cat_missing": missing_cat})

    feature_cols = num_features + cat_features
    if not feature_cols:
        raise RuntimeError("Keine Features gefunden.")

    # Diagnose: Non-NA-Quoten
    nn = (df[feature_cols].notna().mean()*100).sort_values(ascending=True)
    print("\nNon-NA-Quote der Features (in %):")
    print(nn.to_string())

    # Optionales Sampling
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print("Sampled to:", df.shape)

    X = df[feature_cols]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipe = build_pipeline(num_features, cat_features, verbose=args.verbose)
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
        "train_csv": args.train_csv,
    }
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics → {args.metrics}")

if __name__ == "__main__":
    main()

# from __future__ import annotations

# import os
# import argparse
# import json
# import sqlite3
# import numpy as np
# import pandas as pd

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
# import joblib

# # # # # Hilfsfunktionen # # # # #

# def _city_simple_from_df(df: pd.DataFrame) -> pd.Series:
#     # robust: city / cityName / address / location
#     if "city" in df.columns and df["city"].notna().any():
#         s = df["city"].astype("string")
#     elif "cityName" in df.columns:
#         s = df["cityName"].astype("string")
#     elif "address" in df.columns:
#         s = df["address"].astype("string").str.split(",").str[0]
#     elif "location" in df.columns:
#         s = df["location"].astype("string").str.split(",").str[0]
#     else:
#         s = pd.Series("", index=df.index, dtype="string")
#     return s.str.split(",").str[0].str.strip().str.lower()

# def _bucketize_area(a: float) -> str:
#     if pd.isna(a): return None
#     bins = [(0,30),"Unter 30 m²",(30,40),"30 - 39 m²",(40,50),"40 - 49 m²",
#             (50,60),"50 - 59 m²",(60,70),"60 - 69 m²",(70,80),"70 - 79 m²",
#             (80,90),"80 - 89 m²",(90,100),"90 - 99 m²",(100,110),"100 - 109 m²",
#             (110,120),"110 - 119 m²",(120,130),"120 - 129 m²",(130,140),"130 - 139 m²",
#             (140,150),"140 - 149 m²",(150,160),"150 - 159 m²",(160,170),"160 - 169 m²",
#             (170,180),"170 - 179 m²",(180,1e9),"180 m² und mehr"]
#     for lo, label in zip(bins[::2], bins[1::2]): # pairs (range,label)
#         lo, hi = lo
#         if (a >= lo) and (a < hi): return label
#     return None

# def _bucketize_year(y: float) -> str:
#     if pd.isna(y): return None
#     y = int(y)
#     if y < 1919: return "vor_1919"
#     if 1919 <= y <= 1949: return "1919_1949"
#     if 1950 <= y <= 1959: return "1950_1959"
#     if 1970 <= y <= 1979: return "1970_1979"
#     if 1980 <= y <= 1989: return "1980_1989"
#     if 2000 <= y <= 2009: return "2000_2009"
#     if 2010 <= y <= 2015: return "2010_2015"
#     if y >= 2016: return "2016_plus"
#     return None  # 1960-69, 1990-99 sind in 0005 nicht vorhanden


# # --- Defaults (per .env optional überschreibbar) ---
# DEFAULT_DB = os.environ.get("DB_PATH", "./data/hausverwaltung.db")
# DEFAULT_MODEL_PATH = os.environ.get("PRICING_MODEL_PATH", "./data/pricing_model.pkl")
# DEFAULT_METRICS_PATH = "./models/metrics.json"
# DEFAULT_ZENSUS_PATH = "./data/clean/zensus_0004_clean.csv"

# # --- Features: numerisch + kategorisch ---
# NUM_FEATURES = [
#     "area_sqm",
#     "rooms",
#     "floor",
#     "year_built",
#     "numberOfFloors",
#     "building_age",
#     "rooms_per_100qm",
#     "floor_ratio",
#     "zensus_factor_size",
#     "zensus_factor_age",
#     "zensus_factor_combined",
#     # wird später ggf. ergänzt:
#     # "zensus_miete",
# ]
# CAT_FEATURES = [
#     "zip3",
#     "zip2",
#     "condition",
#     "typeOfFlat",
#     # optional: wenn du State/City im Modell kodieren willst,
#     # kannst du sie hier ergänzen
# ]
# TARGET = "rent_cold"


# def load_from_sqlite(db_path: str) -> pd.DataFrame:
#     con = sqlite3.connect(db_path)
#     try:
#         df = pd.read_sql_query("SELECT * FROM listings_raw", con)
#     finally:
#         con.close()
#     return df


# def norm_name_series(s: pd.Series) -> pd.Series:
#     """Einfaches, robustes Normalisieren von Gemeindenamen."""
#     return (
#         s.fillna("")
#         .astype(str)
#         .str.lower()
#         .str.replace(r"[^a-zäöüß0-9 ]+", " ", regex=True)
#         .str.replace(r"\s+", " ", regex=True)
#         .str.strip()
#     )


# def load_zensus_miete(path: str) -> pd.DataFrame:
#     """
#     Erwartet data/clean/zensus_0004_clean.csv mit Spalten:
#     GKZ, Gemeindename, Miete
#     Bereitet (gemeinde_norm, zensus_miete) vor.
#     """
#     if not os.path.exists(path):
#         print(f"[ZENSUS] Datei nicht gefunden: {path} – Feature wird übersprungen.")
#         return pd.DataFrame(columns=["gemeinde_norm", "zensus_miete", "geo_bln"])

#     z = pd.read_csv(path)
#     need = {"GKZ", "Gemeindename", "Miete"}
#     if not need.issubset(z.columns):
#         print(f"[ZENSUS] Unerwartete Spalten in {path}: {z.columns.tolist()}")
#         return pd.DataFrame(columns=["gemeinde_norm", "zensus_miete", "geo_bln"])

#     # Bundesland aus GKZ ableiten (optional – nur für strengeren Join)
#     BL_CODE_TO_NAME = {
#         "01": "Schleswig-Holstein","02": "Hamburg","03": "Niedersachsen","04": "Bremen",
#         "05": "Nordrhein-Westfalen","06": "Hessen","07": "Rheinland-Pfalz","08": "Baden-Württemberg",
#         "09": "Bayern","10": "Saarland","11": "Berlin","12": "Brandenburg",
#         "13": "Mecklenburg-Vorpommern","14": "Sachsen","15": "Sachsen-Anhalt","16": "Thüringen",
#     }

#     z = z.copy()
#     z["GKZ"] = z["GKZ"].astype(str).str.zfill(11)
#     z["geo_bln"] = z["GKZ"].str[:2].map(BL_CODE_TO_NAME)
#     z["gemeinde_norm"] = norm_name_series(z["Gemeindename"])
#     z = (
#         z[["geo_bln", "gemeinde_norm", "Miete"]]
#         .rename(columns={"Miete": "zensus_miete"})
#         .dropna(subset=["gemeinde_norm"])
#     )

#     # Falls gleiche Namen in mehreren BL vorkommen -> wir behalten beide Zeilen;
#     # der Join kann optional über geo_bln zusätzlich einschränken.
#     # Für reine Ortsnamen-Join mitteln wir je (gemeinde_norm) als Fallsicherung:
#     z_fallback = z.groupby("gemeinde_norm", as_index=False)["zensus_miete"].mean()
#     z_fallback["geo_bln"] = np.nan  # Kennzeichen, dass es ein Fallback ist

#     z_all = pd.concat([z, z_fallback], ignore_index=True)
#     print(f"[ZENSUS] Geladen: {len(z)} Zeilen (+{len(z_fallback)} Fallback-Aggregate)")
#     return z_all


# def join_zensus_feature(
#     df: pd.DataFrame,
#     zensus: pd.DataFrame,
#     city_col: str | None,
#     state_col: str | None,
# ) -> pd.DataFrame:
#     """
#     Hängt 'zensus_miete' an df:
#       1) Wenn state_col vorhanden -> Join auf (state_col normalisiert, city_col normalisiert)
#       2) Sonst -> Join nur auf (city_col normalisiert) via Fallback-Einträge
#     Gibt Abdeckung aus.
#     """
#     if zensus.empty or city_col is None or city_col not in df.columns:
#         df["zensus_miete"] = np.nan
#         print("[ZENSUS] Join übersprungen (keine City-Spalte/keine Zensusdaten).")
#         return df

#     work = df.copy()
#     work["_city_norm"] = norm_name_series(work[city_col])

#     if state_col and state_col in work.columns:
#         work["_state_norm"] = norm_name_series(work[state_col])
#         left = work.merge(
#             zensus.dropna(subset=["geo_bln"]),
#             how="left",
#             left_on=["_state_norm", "_city_norm"],
#             right_on=["geo_bln", "gemeinde_norm"],
#             validate="m:1",
#         )
#         hit1 = left["zensus_miete"].notna().sum()
#     else:
#         left = work.copy()
#         hit1 = 0

#     # Fallback: falls noch NaN, Join nur über City auf die Fallback-Aggregate
#     miss_mask = left["zensus_miete"].isna()
#     if miss_mask.any():
#         only_city = left.loc[miss_mask, ["_city_norm"]].merge(
#             zensus[zensus["geo_bln"].isna()],
#             how="left",
#             left_on="_city_norm",
#             right_on="gemeinde_norm",
#             suffixes=("", "_fb"),
#             validate="m:1",
#         )
#         left.loc[miss_mask, "zensus_miete"] = only_city["zensus_miete"].values

#     coverage = left["zensus_miete"].notna().mean()
#     print(f"[ZENSUS] Join-Abdeckung: {coverage:.0%} (inkl. City-Fallback)")

#     # Aufräumen
#     left = left.drop(columns=["_city_norm"] + (["_state_norm"] if "_state_norm" in left.columns else []))
#     return left


# def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
#     """Bereitet das Roh-DF aus listings_raw für das Training vor."""
#     # --- PLZ-Buckets aus zip_code ableiten ---
#     if "zip_code" in df.columns:
#         s = df["zip_code"].astype("string")
#         df["zip3"] = s.str.extract(r"(\d{3})", expand=False)
#         df["zip2"] = s.str.extract(r"(\d{2})", expand=False)
#     else:
#         df["zip3"] = None
#         df["zip2"] = None

#     # --- Felder aus Rohdaten übernehmen/typisieren, falls vorhanden ---
#     if "numberOfFloors" in df.columns:
#         df["numberOfFloors"] = pd.to_numeric(df["numberOfFloors"], errors="coerce")
#     else:
#         df["numberOfFloors"] = np.nan

#     if "typeOfFlat" in df.columns:
#         df["typeOfFlat"] = df["typeOfFlat"].astype("string").str.lower()
#     else:
#         df["typeOfFlat"] = None

#     # Abgeleitete Features
#     if "year_built" in df.columns:
#         year_built_num = pd.to_numeric(df["year_built"], errors="coerce")
#     elif "yearConstructed" in df.columns:
#         year_built_num = pd.to_numeric(df["yearConstructed"], errors="coerce")
#     else:
#         year_built_num = pd.Series(np.nan, index=df.index, dtype="float")

#     df["year_built"] = year_built_num
#     df["building_age"] = 2025 - year_built_num

#     # Basisfelder sicherstellen (rooms/area_sqm/floor)
#     if "rooms" in df.columns:
#         df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
#     if "area_sqm" in df.columns:
#         df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce")
#     if "floor" in df.columns:
#         df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

#     # weitere abgeleitete Felder
#     df["rooms_per_100qm"] = df["rooms"] / (df["area_sqm"] / 100.0)
#     df["floor_ratio"] = np.where(
#         (df["numberOfFloors"].notna()) & (pd.to_numeric(df["numberOfFloors"], errors="coerce") > 0),
#         df["floor"] / df["numberOfFloors"],
#         np.nan,
#     )

#     # --- nur relevante Spalten behalten ---
#     cols = NUM_FEATURES + CAT_FEATURES + [TARGET]
#     keep = [c for c in cols if c in df.columns]
#     df = df.loc[:, keep].copy()

#     # --- Plausibilitäten ---
#     if "area_sqm" in df.columns and "rent_cold" in df.columns:
#         df = df[
#             (df["area_sqm"] > 10) & (df["area_sqm"] < 400) &
#             (df["rent_cold"] > 200) & (df["rent_cold"] < 10000)
#         ].copy()

#     # --- leichte Imputation (zusätzlich zur Pipeline) ---
#     if "condition" in df.columns:
#         df["condition"] = df["condition"].fillna("unknown")
#     if "year_built" in df.columns and df["year_built"].isna().any():
#         df["year_built"] = df["year_built"].fillna(df["year_built"].median())
#     if "floor" in df.columns and df["floor"].isna().any():
#         df["floor"] = df["floor"].fillna(0)
#     if "rooms" in df.columns and df["rooms"].isna().any():
#         df["rooms"] = df["rooms"].fillna(df["rooms"].median())

#     # --- ZENSUS: Faktoren joinen ---
#     # 1) benötigte Ableitungen:
#     df["city_simple"] = _city_simple_from_df(df)
#     df["area_bucket"] = df["area_sqm"].apply(_bucketize_area)
#     df["year_bucket"] = df["year_built"].apply(_bucketize_year)

#     # 2) Tabellen lesen
#     try:
#         z5 = pd.read_csv("./data/clean/zensus_0005_factors.csv")
#         z8 = pd.read_csv("./data/clean/zensus_0008_factors.csv")
#     except FileNotFoundError:
#         z5 = pd.DataFrame(columns=["city_simple"])
#         z8 = pd.DataFrame(columns=["city_simple"])

#     # 3) Fallback (bundesweit) – Mittelwert je Bucket
#     z5_mean = z5.drop(columns=["city_simple"], errors="ignore").mean(numeric_only=True)
#     z8_mean = z8.drop(columns=["city_simple"], errors="ignore").mean(numeric_only=True)

#     # 4) City-lookup dicts
#     z5_dict = {row["city_simple"]: row.drop("city_simple").to_dict() for _, row in z5.iterrows()}
#     z8_dict = {row["city_simple"]: row.drop("city_simple").to_dict() for _, row in z8.iterrows()}

#     def _lookup_factor(city, bucket, dct, mean_series):
#         if pd.isna(bucket): return np.nan
#         if isinstance(city, str) and city in dct and bucket in dct[city]:
#             return dct[city][bucket]
#         # fallback bundesweit:
#         return float(mean_series.get(bucket, np.nan))

#     df["zensus_factor_age"] = [
#         _lookup_factor(c, b, z5_dict, z5_mean) for c, b in zip(df["city_simple"], df["year_bucket"])
#     ]
#     df["zensus_factor_size"] = [
#         _lookup_factor(c, b, z8_dict, z8_mean) for c, b in zip(df["city_simple"], df["area_bucket"])
#     ]

#     # Kombinierter Faktor (wenn eins fehlt, nimm das andere)
#     df["zensus_factor_combined"] = np.where(
#         df["zensus_factor_age"].notna() & df["zensus_factor_size"].notna(),
#         df["zensus_factor_age"] * df["zensus_factor_size"],
#         df["zensus_factor_age"].fillna(1.0) * df["zensus_factor_size"].fillna(1.0)
#     )


#     return df


# def build_pipeline(available_cols: list[str], verbose: int = 0) -> Pipeline:
#     # tatsächlich vorhandene Spalten bestimmen
#     num_candidates = NUM_FEATURES + ["zensus_miete"]
#     num_feats_present = [c for c in num_candidates if c in available_cols]
#     cat_feats_present = [c for c in CAT_FEATURES if c in available_cols]

#     pre = ColumnTransformer(
#         transformers=[
#             ("num", SimpleImputer(strategy="median"), num_feats_present),
#             (
#                 "cat",
#                 Pipeline(
#                     steps=[
#                         ("imp", SimpleImputer(strategy="most_frequent")),
#                         ("ohe", OneHotEncoder(handle_unknown="ignore")),
#                     ]
#                 ),
#                 cat_feats_present,
#             ),
#         ],
#         remainder="drop",
#         n_jobs=None,
#     )

#     model = RandomForestRegressor(
#         n_estimators=200,
#         min_samples_leaf=2,
#         random_state=42,
#         n_jobs=-1,
#         verbose=verbose,
#     )
#     return Pipeline(steps=[("prep", pre), ("rf", model)])



# def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     denom = (np.abs(y_true) + np.abs(y_pred))
#     denom = np.where(denom == 0, 1, denom)
#     return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


# def main():
#     parser = argparse.ArgumentParser(description="Train rental price model")
#     parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path (table listings_raw)")
#     parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Path to save trained model (joblib)")
#     parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON")
#     parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
#     parser.add_argument("--sample", type=int, default=50000, help="Sample size (0 disables sampling)")
#     parser.add_argument("--verbose", type=int, default=1, help="RF verbose level (0,1,2)")

#     # NEU: Zensus-Optionen
#     parser.add_argument("--zensus_path", default=DEFAULT_ZENSUS_PATH, help="Pfad zu data/clean/zensus_0004_clean.csv")
#     parser.add_argument("--city_col", default=None, help="Spalte in listings_raw mit Gemeindename (z. B. 'regio3' oder 'city')")
#     parser.add_argument("--state_col", default=None, help="Spalte in listings_raw mit Bundesland (z. B. 'geo_bln' oder 'state')")
#     args = parser.parse_args()

#     print(f"DB: {args.db}")
#     df = load_from_sqlite(args.db)
#     print("Loaded:", df.shape)

#     df = basic_clean(df)
#     print("After clean:", df.shape)

#     # -------- ZENSUS: optionaler Join --------
#     zensus = load_zensus_miete(args.zensus_path)
#     # Wenn der Nutzer keine Spalten angegeben hat, versuchen wir sinnvolle Defaults:
#     possible_city_cols = [args.city_col, "regio3", "city", "municipality", "place", "ort"]
#     possible_state_cols = [args.state_col, "geo_bln", "state", "bundesland"]

#     use_city = next((c for c in possible_city_cols if c and c in df.columns), None)
#     use_state = next((c for c in possible_state_cols if c and c in df.columns), None)

#     if use_city:
#         print(f"[ZENSUS] Join über Stadtspalte: '{use_city}'" + (f" und Bundesland: '{use_state}'" if use_state else " (ohne Bundesland)"))
#         df = join_zensus_feature(df, zensus, city_col=use_city, state_col=use_state)
#     else:
#         print("[ZENSUS] Keine geeignete Stadtspalte gefunden – Feature wird NaN.")
#         df["zensus_miete"] = np.nan


#     if args.sample and len(df) > args.sample:
#         df = df.sample(args.sample, random_state=42).reset_index(drop=True)
#         print(f"Sampled to: {df.shape}")

#     # Features zusammenstellen (zensus_miete ggf. ergänzen)
#     feature_cols = [c for c in NUM_FEATURES + CAT_FEATURES + ["zensus_miete"] if c in df.columns]
#     if TARGET not in df.columns or not feature_cols:
#         raise RuntimeError("Fehlende Spalten: target oder features nicht gefunden.")

#     X = df[feature_cols]
#     y = df[TARGET].astype(float)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=args.test_size, random_state=42
#     )

#     pipe = build_pipeline(feature_cols, verbose=args.verbose)
#     print("Fitting model...")
#     pipe.fit(X_train, y_train)
#     print("Predicting...")
#     pred = pipe.predict(X_test)

#     mae = mean_absolute_error(y_test, pred)
#     r2 = r2_score(y_test, pred)
#     mdae = median_absolute_error(y_test, pred)
#     mape = np.mean(np.abs((y_test - pred) / np.clip(y_test, 1e-6, None))) * 100.0
#     smape_val = smape(y_test.values, pred)

#     print(f"MAE : {mae:.2f} €")
#     print(f"MdAE: {mdae:.2f} €")
#     print(f"MAPE: {mape:.2f} %   sMAPE: {smape_val:.2f} %")
#     print(f"R²  : {r2:.3f}")

#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     joblib.dump(pipe, args.out)
#     print(f"Saved model → {args.out}")

#     os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
#     metrics = {
#         "n_train": int(len(X_train)),
#         "n_test": int(len(X_test)),
#         "mae_eur": float(mae),
#         "mdae_eur": float(mdae),
#         "mape_pct": float(mape),
#         "smape_pct": float(smape_val),
#         "r2": float(r2),
#         "features_used": feature_cols,
#         "target": TARGET,
#         "sample": int(args.sample),
#         "zensus_file": args.zensus_path if os.path.exists(args.zensus_path) else None,
#         "zensus_join_city_col": use_city,
#         "zensus_join_state_col": use_state,
#     }
#     with open(args.metrics, "w", encoding="utf-8") as f:
#         json.dump(metrics, f, ensure_ascii=False, indent=2)
#     print(f"Saved metrics → {args.metrics}")


# if __name__ == "__main__":
#     main()
