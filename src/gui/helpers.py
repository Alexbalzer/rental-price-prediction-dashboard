# --- ADD-ONs fürs Dashboard: Laden/Normieren/Statistiken --------------------

# src/gui/helpers.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import difflib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------- Utilities ---------------------------

def _norm(s: str | None) -> str:
    """Einfache Normalisierung für Namen/Orte (für fuzzy matching)."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    repl = {
        "ß": "ss",
        "-": " ",
        "(": "",
        ")": "",
        ".": "",
        ",": "",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = s.replace(" hansestadt", "").replace(" stadt", "")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def _first_existing(candidates: List[str | Path]) -> Optional[Path]:
    """Nimmt die erste existierende Datei aus einer Kandidatenliste."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


# ---------------------- Zensus/AGS – alte API ----------------------

@st.cache_data(show_spinner=False)
def load_zensus0005(path: str | Path | None = None) -> pd.DataFrame:
    """
    Lädt Zensus 0005 (Gemeindeebene) und konvertiert numerische Spalten.

    Erwartete Kerne:
      - GKZ (12-stellig)
      - Gemeindename
      - Insgesamt
      - Perioden-Spalten (z. B. '2016_plus', '1970_1979', ...)
    """
    if path is None:
        path = _first_existing([
            "./data/clean/zensus_0005_clean.csv",
            "./data/clean/zensus_0005_clean1.csv",
            "./data/clean/zensus_0005_clean_de.csv",
        ])
    if path is None:
        raise FileNotFoundError("Zensus 0005-Datei nicht gefunden (data/clean/...).")

    df = pd.read_csv(path, dtype=str)
    for c in df.columns:
        if c not in ("GKZ", "Gemeindename"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["name_norm"] = df["Gemeindename"].map(_norm)
    return df


@st.cache_data(show_spinner=False)
def load_ags_map(path: str | Path | None = None) -> Optional[pd.DataFrame]:
    """
    Optionales Mapping PLZ/Ort → AGS/ARS.
    Erwartete Spalten (variieren je nach Quelle): ARS/AGS, PLZ, Ort, Gemeindename.
    """
    if path is None:
        path = _first_existing([
            "./data/ags_gkz.csv",
            "./data/clean/ags_gkz.csv",
        ])
    if path is None:
        return None

    m = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    cols = {c.lower(): c for c in m.columns}

    def pick(name: str, default: str) -> str:
        return cols.get(name, default)

    m = m.rename(columns={
        pick("ars", "ARS"): "ARS",
        pick("ags", "AGS"): "AGS",
        pick("gemeindename", "Gemeindename"): "Gemeindename",
        pick("plz", "PLZ"): "PLZ",
        pick("ort", "Ort"): "Ort",
    })

    for need in ("Gemeindename", "Ort", "PLZ", "ARS", "AGS"):
        if need not in m.columns:
            return None

    m["name_norm"] = m["Gemeindename"].map(_norm)
    m["ort_norm"] = m["Ort"].map(_norm)
    m = m[["ARS", "AGS", "PLZ", "Gemeindename", "Ort", "name_norm", "ort_norm"]].drop_duplicates()
    return m


def resolve_location_defaults(
    user_city: str,
    periode: str = "2016_plus",
    zensus_path: str | Path | None = None,
    ags_map_path: str | Path | None = None,
) -> Dict[str, Optional[float | str]]:
    """
    Versucht für eine Stadt sinnvolle Default-Werte zu finden:
      - ARS (falls Mapping vorhanden)
      - zensus_miete_total (Insgesamt)
      - zensus_miete_decade (für 'periode')
      - zensus_factor_decade (decade/gesamt)

    Rückgabe: dict mit 'ARS', 'zensus_total', 'zensus_decade', 'zensus_factor'
    """
    out: Dict[str, Optional[float | str]] = {
        "ARS": None,
        "zensus_total": None,
        "zensus_decade": None,
        "zensus_factor": None,
    }
    if not user_city:
        return out

    z5 = load_zensus0005(zensus_path)
    m = load_ags_map(ags_map_path)
    target_norm = _norm(user_city)

    # 1) exakter/fuzzy Treffer im Zensus
    hit = z5[z5["name_norm"] == target_norm]
    if hit.empty:
        choices = z5["name_norm"].unique().tolist()
        best = difflib.get_close_matches(target_norm, choices, n=1, cutoff=0.87)
        if best:
            hit = z5[z5["name_norm"] == best[0]]

    if not hit.empty:
        row = hit.iloc[0]
        if pd.notna(row.get("Insgesamt")):
            out["zensus_total"] = float(row["Insgesamt"])
        if periode in row.index and pd.notna(row.get(periode)):
            out["zensus_decade"] = float(row[periode])
        if out["zensus_total"] and out["zensus_decade"] and out["zensus_total"] not in (0, "0"):
            out["zensus_factor"] = float(out["zensus_decade"]) / float(out["zensus_total"])
        if pd.notna(row.get("GKZ")):
            out["ARS"] = str(row["GKZ"]).strip()

    # 2) ARS ggf. über Mapping
    if not out["ARS"] and m is not None:
        mm = m[(m["name_norm"] == target_norm) | (m["ort_norm"] == target_norm)]
        if mm.empty:
            pool = pd.unique(pd.concat([m["name_norm"], m["ort_norm"]], ignore_index=True)).tolist()
            best = difflib.get_close_matches(target_norm, pool, n=1, cutoff=0.9)
            if best:
                mm = m[(m["name_norm"] == best[0]) | (m["ort_norm"] == best[0])]
        if not mm.empty:
            out["ARS"] = str(mm.iloc[0]["ARS"]).strip()

    return out


# ---------------------- Neue API fürs Dashboard ----------------------

@st.cache_data(show_spinner=False)
def load_dashboard_frames(
    immo_path: str | Path | None = None,
    zensus_path: str | Path | None = None,
    ags_map_path: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Lädt Datenframes, die das Dashboard braucht:
      - immo (Listings + abgeleitete Spalten + Zensus-Merge, soweit möglich)
      - agg_state (Anzahl Listings je Bundesland)
      - agg_city  (Anzahl Listings je Stadt)
      - miete_vs_zensus_state (Ø €/m² aus Listings vs. Zensus je Bundesland)
      - features_by_plz (Summen/Counts je PLZ für Balkon/Küche/Lift/Garten)

    Robust gegen fehlende Dateien/Spalten; füllt dann mit NaNs/Defaults.
    """
    # --- Pfade auflösen
    immo_p = Path(immo_path) if immo_path else _first_existing([
        "./data/immo_data.csv",
        "./data/clean/immo_data.csv",
        "./data/clean/immo_clean8.csv",
    ])
    if immo_p is None:
        raise FileNotFoundError("Kann immo_data.csv nicht finden.")

    zensus_p = Path(zensus_path) if zensus_path else _first_existing([
        "./data/clean/zensus_0005_clean.csv",
        "./data/clean/zensus_0005_clean1.csv",
    ])
    ags_p = Path(ags_map_path) if ags_map_path else _first_existing([
        "./data/ags_gkz.csv",
        "./data/clean/ags_gkz.csv",
    ])

    # --- IMMO laden
    immo = pd.read_csv(immo_p)
    # Spalten robust machen
    def ensure(col: str, default=np.nan):
        if col not in immo.columns:
            immo[col] = default

    for c in ["state", "city", "zipCode", "coldRent", "livingSpace",
              "balcony", "hasKitchen", "lift", "garden"]:
        ensure(c, np.nan)

    # Typen/Normierungen
    immo["state"] = immo["state"].astype(str).str.strip()
    immo["city"] = immo["city"].astype(str).str.strip()
    immo["PLZ"] = immo["zipCode"].astype(str).str.extract(r"(\d{5})", expand=False)

    for c in ["coldRent", "livingSpace"]:
        immo[c] = pd.to_numeric(immo[c], errors="coerce")

    # €/m²
    immo["coldRentPerSqm"] = immo["coldRent"] / immo["livingSpace"]

    # Feature-Spalten zu int (0/1), wenn bool/Strings vorliegen
    for c in ["balcony", "hasKitchen", "lift", "garden"]:
        if c in immo.columns:
            immo[c] = immo[c].map(lambda x: 1 if str(x).strip().lower() in ("1", "true", "yes", "ja") else 0)

    # --- AGS/ARS-Mapping (optional)
    if ags_p is not None and ags_p.exists() and "PLZ" in immo.columns:
        m = load_ags_map(ags_p)
        if m is not None:
            immo = immo.merge(m[["PLZ", "ARS"]], on="PLZ", how="left")

    # --- Zensus join (optional)
    if zensus_p is not None and zensus_p.exists():
        z5 = load_zensus0005(zensus_p)
        if "ARS" in immo.columns:
            immo = immo.merge(z5[["GKZ", "Insgesamt"]], left_on="ARS", right_on="GKZ", how="left")
            immo.rename(columns={"Insgesamt": "zensus_miete_total"}, inplace=True)
        else:
            immo["zensus_miete_total"] = np.nan
    else:
        immo["zensus_miete_total"] = np.nan

    # --- Aggregationen
    agg_state = (
        immo.groupby("state", dropna=False)
            .size()
            .reset_index(name="n_listings")
            .sort_values("n_listings", ascending=False)
    )

    agg_city = (
        immo.groupby("city", dropna=False)
            .size()
            .reset_index(name="n_listings")
            .sort_values("n_listings", ascending=False)
    )

    miete_vs_zensus_state = (
        immo.groupby("state", dropna=False)
            .agg(listing_rent=("coldRentPerSqm", "mean"),
                 zensus_rent=("zensus_miete_total", "mean"),
                 n=("coldRentPerSqm", "size"))
            .reset_index()
            .sort_values("n", ascending=False)
    )

    features_by_plz = (
        immo.groupby("PLZ", dropna=False)
            .agg(n=("coldRentPerSqm", "size"),
                 balcony=("balcony", "sum"),
                 hasKitchen=("hasKitchen", "sum"),
                 lift=("lift", "sum"),
                 garden=("garden", "sum"),
                 city=("city", "first"),
                 state=("state", "first"))
            .reset_index()
            .sort_values("n", ascending=False)
    )

    return {
        "immo": immo,
        "agg_state": agg_state,
        "agg_city": agg_city,
        "miete_vs_zensus_state": miete_vs_zensus_state,
        "features_by_plz": features_by_plz,
    }


# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
# import pandas as pd
# import numpy as np
# import streamlit as st

# # Spaltenkandidaten in verschiedenen Datasets
# _CAND = {
#     "area": ["area_sqm", "livingSpace", "total_living_area", "size_sqm", "area"],
#     "net_rent": ["netRent", "netColdRent", "net_cold_rent", "cold_rent", "kaltmiete"],
#     "warm_rent": ["totalRent", "warmmiete", "warm_rent"],
#     "plz": ["zip_code", "postcode", "plz", "postalCode", "zip"],
#     "city": ["city", "ort", "town"],
#     "state": ["state", "bundesland", "federal_state", "federalState"],
#     "year_built": ["year_built", "constructionYear", "yearOfConstruction", "baujahr"],
#     "periode": ["periode_0005", "periode_5", "build_period"],
#     "lat": ["latitude", "geo_latitude", "lat"],
#     "lon": ["longitude", "geo_longitude", "lon"],
#     "balcony": ["balcony", "hasBalcony"],
#     "kitchen": ["hasKitchen", "builtInKitchen", "kitchen"],
#     "garden": ["garden", "hasGarden"],
#     "lift": ["lift", "hasElevator", "elevator"],
# }

# # AGS 2er-Präfix -> Bundeslandname
# _LAND_BY_PREFIX = {
#     "01": "Schleswig-Holstein", "02": "Hamburg", "03": "Niedersachsen", "04": "Bremen",
#     "05": "Nordrhein-Westfalen", "06": "Hessen", "07": "Rheinland-Pfalz", "08": "Baden-Württemberg",
#     "09": "Bayern", "10": "Saarland", "11": "Berlin", "12": "Brandenburg",
#     "13": "Mecklenburg-Vorpommern", "14": "Sachsen", "15": "Sachsen-Anhalt", "16": "Thüringen",
# }

# def _pick_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
#     for k in keys:
#         if k in df.columns: 
#             return k
#         # lax: case-insensitive hit
#         for c in df.columns:
#             if c.lower() == k.lower():
#                 return c
#     return None

# def _ensure_binary(series: pd.Series) -> pd.Series:
#     """0/1 aus bool/ja/nein/true/false etc. bauen."""
#     s = series.copy()
#     if s.dtype == bool:
#         return s.astype(int)
#     return s.map(lambda x: 1 if str(x).strip().lower() in ("1","true","yes","ja") else 0).fillna(0).astype(int)

# @st.cache_data(show_spinner=False)
# def load_immo_csv(path: str | Path) -> pd.DataFrame:
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"Immo-CSV nicht gefunden: {p}")
#     df = pd.read_csv(p)
#     # Kanonische Spalten erzeugen
#     c_area   = _pick_col(df, _CAND["area"])
#     c_net    = _pick_col(df, _CAND["net_rent"])
#     c_warm   = _pick_col(df, _CAND["warm_rent"])
#     c_plz    = _pick_col(df, _CAND["plz"])
#     c_city   = _pick_col(df, _CAND["city"])
#     c_state  = _pick_col(df, _CAND["state"])
#     c_year   = _pick_col(df, _CAND["year_built"])
#     c_period = _pick_col(df, _CAND["periode"])
#     c_lat    = _pick_col(df, _CAND["lat"])
#     c_lon    = _pick_col(df, _CAND["lon"])
#     c_bal    = _pick_col(df, _CAND["balcony"])
#     c_kitch  = _pick_col(df, _CAND["kitchen"])
#     c_gard   = _pick_col(df, _CAND["garden"])
#     c_lift   = _pick_col(df, _CAND["lift"])

#     out = pd.DataFrame()
#     out["area_sqm"] = df[c_area] if c_area else np.nan
#     out["netRent"]  = df[c_net] if c_net else np.nan
#     out["warmRent"] = df[c_warm] if c_warm else np.nan
#     out["PLZ"]      = df[c_plz].astype(str).str.extract(r"(\d{5})")[0] if c_plz else None
#     out["city"]     = df[c_city] if c_city else None
#     out["state_raw"]= df[c_state] if c_state else None
#     out["year_built"]= df[c_year] if c_year else None
#     out["periode_0005"] = df[c_period] if c_period else None
#     out["lat"] = df[c_lat] if c_lat else None
#     out["lon"] = df[c_lon] if c_lon else None

#     # Features 0/1
#     out["balcony"]    = _ensure_binary(df[c_bal])   if c_bal   else 0
#     out["hasKitchen"] = _ensure_binary(df[c_kitch]) if c_kitch else 0
#     out["garden"]     = _ensure_binary(df[c_gard])  if c_gard  else 0
#     out["lift"]       = _ensure_binary(df[c_lift])  if c_lift  else 0

#     return out

# def _derive_state_from_ags_map(df_immo: pd.DataFrame, ags_map: Optional[pd.DataFrame]) -> pd.Series:
#     """Falls 'state' fehlt: via PLZ -> AGS -> Bundesland bestimmen."""
#     if ags_map is None or "PLZ" not in df_immo.columns:
#         return pd.Series([None]*len(df_immo))

#     m = ags_map[["PLZ", "AGS"]].dropna().copy()
#     m["PLZ"] = m["PLZ"].astype(str).str.extract(r"(\d{5})")[0]
#     m = m.dropna().drop_duplicates(subset=["PLZ"])
#     df = df_immo.merge(m, on="PLZ", how="left")
#     land = df["AGS"].astype(str).str[:2].map(_LAND_BY_PREFIX)
#     return land

# def _periode_from_year(y: Optional[float]) -> Optional[str]:
#     try:
#         y = int(y)
#     except Exception:
#         return None
#     if y < 1919: return "vor_1919"
#     if y <= 1949: return "1919_1949"
#     if y <= 1959: return "1950_1959"
#     if y <= 1969: return "1960_1969"
#     if y <= 1979: return "1970_1979"
#     if y <= 1989: return "1980_1989"
#     if y <= 1999: return "1990_1999"
#     if y <= 2009: return "2000_2009"
#     if y <= 2015: return "2010_2015"
#     return "2016_plus"

# def attach_canonical_regionals(
#     df_immo: pd.DataFrame,
#     ags_map: Optional[pd.DataFrame]
# ) -> pd.DataFrame:
#     df = df_immo.copy()
#     # Bundesland konsolidieren
#     if "state_raw" in df.columns and df["state_raw"].notna().any():
#         df["state"] = df["state_raw"].astype(str).str.strip()
#     else:
#         df["state"] = _derive_state_from_ags_map(df, ags_map)
#     # Periode
#     if "periode_0005" not in df.columns or df["periode_0005"].isna().all():
#         df["periode_0005"] = df["year_built"].map(_periode_from_year)
#     return df

# def attach_zensus_expectation(df: pd.DataFrame, z5: pd.DataFrame) -> pd.DataFrame:
#     """Erwartete Kaltmiete aus Zensus 0005 (€/m²) * Fläche."""
#     if z5 is None or z5.empty:
#         df["zensus_expected"] = np.nan
#         return df
#     # Merge über Gemeindenamen schwierig -> wir rechnen auf Landes-/PLZ-Niveau im Zweifel mit Gesamtwert
#     # Wir nutzen hier nur die EUR/m²-Werte aus zensus_0005 (Insgesamt + Perioden)
#     use = z5[["Gemeindename","Insgesamt","name_norm"] + [c for c in z5.columns if c.startswith(tuple(["vor_1919","1919_1949","1950","1960","1970","1980","1990","2000","2010","2016"]))]].copy()

#     df2 = df.copy()
#     # fallback: expected = Insgesamt * area; wenn periode da und im zensus vorhanden -> nehmen wir die periodespezifische
#     df2["zensus_rate"] = np.nan
#     # Mapping per Periode
#     def _pick_rate(row):
#         per = row.get("periode_0005")
#         if per and per in use.columns:
#             return use.loc[use["name_norm"]==row.get("city_norm",""), per].mean()
#         return use.loc[use["name_norm"]==row.get("city_norm",""), "Insgesamt"].mean()

#     # city_norm zum Join
#     df2["city_norm"] = df2["city"].fillna("").map(lambda x: _norm(str(x)))
#     # Für jede Zeile die Rate holen (mittels Groupby Lookup zur Beschleunigung)
#     city_groups = use.groupby("name_norm").agg({**{c:"mean" for c in use.columns if c not in ["Gemeindename","name_norm"]}})
#     def _rate_fast(row):
#         per = row.get("periode_0005")
#         grp = city_groups.loc[row.get("city_norm","")] if row.get("city_norm","") in city_groups.index else None
#         if grp is not None:
#             if per in city_groups.columns and not np.isnan(grp[per]):
#                 return grp[per]
#             if "Insgesamt" in city_groups.columns and not np.isnan(grp["Insgesamt"]):
#                 return grp["Insgesamt"]
#         return np.nan

#     df2["zensus_rate"] = df2.apply(_rate_fast, axis=1)
#     df2["zensus_expected"] = df2["zensus_rate"] * df2["area_sqm"]
#     return df2

# @st.cache_data(show_spinner=False)
# def load_dashboard_frames(
#     immo_path: str | Path,
#     zensus_path: str = "./data/clean/zensus_0005_clean1.csv",
#     ags_map_path: str = "./data/ags_gkz.csv",
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     immo = load_immo_csv(immo_path)
#     z5 = load_zensus0005(zensus_path)
#     m  = load_ags_map(ags_map_path)
#     immo = attach_canonical_regionals(immo, m)
#     immo = attach_zensus_expectation(immo, z5)
#     return immo, z5

# # src/gui/helpers.py
# from __future__ import annotations
# import pandas as pd
# import numpy as np
# import difflib
# import streamlit as st

# # ---------- Normalisierung für Namen ----------
# def _norm(s: str) -> str:
#     if s is None:
#         return ""
#     s = str(s).lower().strip()
#     repl = {
#         "ß": "ss",
#         "-": " ",
#         "(": "",
#         ")": "",
#         ".": "",
#         ",": "",
#     }
#     for k, v in repl.items():
#         s = s.replace(k, v)
#     # häufige Zusätze entfernen
#     s = s.replace(" hansestadt", "").replace(" stadt", "")
#     while "  " in s:
#         s = s.replace("  ", " ")
#     return s.strip()

# # ---------- Zensus 0005 laden ----------
# @st.cache_data(show_spinner=False)
# def load_zensus0005(path: str = "./data/clean/zensus_0005_clean1.csv") -> pd.DataFrame:
#     z5 = pd.read_csv(path, dtype=str)
#     # numerische Spalten konvertieren
#     for c in z5.columns:
#         if c not in ("GKZ", "Gemeindename"):
#             z5[c] = pd.to_numeric(z5[c], errors="coerce")
#     z5["name_norm"] = z5["Gemeindename"].map(_norm)
#     return z5

@st.cache_data(show_spinner=False)
def load_zensus0005(path: str | Path | None = None) -> pd.DataFrame:
    """
    Zensus 0005 laden und Spalten robust normalisieren.
    Akzeptierte Schlüssel-Spalten: GKZ / ARS / AGS / Amtlicher Gemeindeschlüssel (u.ä.)
    Akzeptierte Namensspalten: Gemeindename / Gemeinde / Name (u.ä.)
    """
    if path is None:
        path = _first_existing([
            "./data/clean/zensus_0005_clean.csv",
            "./data/clean/zensus_0005_clean1.csv",
            "./data/clean/zensus_0005_clean_de.csv",
            "./data/clean/zensus_0005_clean8.csv",
        ])
    if path is None:
        raise FileNotFoundError("Zensus 0005-Datei nicht gefunden (data/clean/...).")

    df = pd.read_csv(path, dtype=str)

    # ---- Spalten-Synonyme erkennen ----
    cols_lc = {c.lower(): c for c in df.columns}

    # GKZ/ARS/AGS finden
    gkz_col = (
        cols_lc.get("gkz")
        or cols_lc.get("ars")
        or cols_lc.get("ags")
        or cols_lc.get("amtlicher gemeindeschlüssel")
        or cols_lc.get("amtlicher gemeindeschluessel")
        or cols_lc.get("gemeindeschlüssel")
        or cols_lc.get("gemeindeschluessel")
    )
    if not gkz_col:
        raise ValueError(f"Keine GKZ/ARS/AGS-Spalte gefunden. Vorhandene Spalten: {list(df.columns)}")

    # Gemeindename finden (optional, aber praktisch)
    name_col = (
        cols_lc.get("gemeindename")
        or cols_lc.get("gemeinde")
        or cols_lc.get("name")
    )

    renames = {gkz_col: "GKZ"}
    if name_col:
        renames[name_col] = "Gemeindename"
    df = df.rename(columns=renames)

    # ---- Numerik konvertieren (alles außer Identifikatoren) ----
    skip = {"GKZ", "Gemeindename"}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normierter Name für fuzzy matching
    if "Gemeindename" in df.columns:
        df["name_norm"] = df["Gemeindename"].map(_norm)
    else:
        df["name_norm"] = ""

    return df


# # ---------- (Optionale) AGS/PLZ-Mapping-Tabelle ----------
# @st.cache_data(show_spinner=False)
# def load_ags_map(path: str = "./data/ags_gkz.csv") -> pd.DataFrame | None:
#     try:
#         m = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
#     except Exception:
#         return None
#     # weiche Spaltennamen tolerieren
#     cols = {c.lower(): c for c in m.columns}
#     # Standardisieren
#     m = m.rename(
#         columns={
#             cols.get("ars", "ARS"): "ARS",
#             cols.get("ags", "AGS"): "AGS",
#             cols.get("gemeindename", "Gemeindename"): "Gemeindename",
#             cols.get("plz", "PLZ"): "PLZ",
#             cols.get("ort", "Ort"): "Ort",
#         }
#     )
#     for c in ("Gemeindename", "Ort", "PLZ", "ARS", "AGS"):
#         if c not in m.columns:
#             # Map ist optional – wenn Felder fehlen, einfach None zurück
#             return None
#     m["name_norm"] = m["Gemeindename"].map(_norm)
#     m["ort_norm"] = m["Ort"].map(_norm)
#     return m[["ARS", "AGS", "PLZ", "Gemeindename", "Ort", "name_norm", "ort_norm"]].drop_duplicates()

# # ---------- Stadt/Ort → ARS auflösen + passende Zensuswerte herausziehen ----------
# def resolve_location_defaults(
#     user_city: str,
#     periode: str = "2016_plus",
#     zensus_path: str = "./data/clean/zensus_0005_clean1.csv",
#     ags_map_path: str = "./data/ags_gkz.csv",
# ) -> dict:
#     """
#     Versucht, für eine Nutzereingabe (Stadt/Ort) sinnvolle Default-Werte zu finden:
#     - ARS (falls Mapping verfügbar)
#     - zensus_miete_total (Insgesamt)
#     - zensus_miete_decade (nach Periode)
#     - zensus_factor_decade (Verhältnis zur Gesamtmiete)

#     Rückgabe: dict mit Keys: 'ARS', 'zensus_total', 'zensus_decade', 'zensus_factor'
#     (fehlende Infos -> None)
#     """
#     out = {"ARS": None, "zensus_total": None, "zensus_decade": None, "zensus_factor": None}

#     if not user_city:
#         return out

#     z5 = load_zensus0005(zensus_path)
#     m = load_ags_map(ags_map_path)

#     target_norm = _norm(user_city)

#     # 1) exakte/eindeutige Treffer im Zensus
#     hit = z5[z5["name_norm"] == target_norm]
#     if hit.empty:
#         # 2) fuzzy im Zensus probieren
#         choices = z5["name_norm"].unique().tolist()
#         best = difflib.get_close_matches(target_norm, choices, n=1, cutoff=0.87)
#         if best:
#             hit = z5[z5["name_norm"] == best[0]]

#     if not hit.empty:
#         # Zensus-Werte
#         row = hit.iloc[0]
#         out["zensus_total"] = float(row.get("Insgesamt")) if pd.notna(row.get("Insgesamt")) else None
#         if periode in row.index:
#             zdec = row[periode]
#             out["zensus_decade"] = float(zdec) if pd.notna(zdec) else None
#             if pd.notna(row.get("Insgesamt")) and pd.notna(zdec) and row["Insgesamt"] not in (0, "0"):
#                 out["zensus_factor"] = float(zdec) / float(row["Insgesamt"])
#         # ARS aus Zensus-GKZ übernehmen (12-stellig)
#         if pd.notna(row.get("GKZ")):
#             out["ARS"] = str(row["GKZ"]).strip()

#     # 3) Falls keine ARS über Zensus: optional AGS/PLZ-Mapping nutzen
#     if not out["ARS"] and m is not None:
#         mm = m[(m["name_norm"] == target_norm) | (m["ort_norm"] == target_norm)]
#         if mm.empty:
#             # fuzzy
#             pool = pd.unique(pd.concat([m["name_norm"], m["ort_norm"]], ignore_index=True)).tolist()
#             best = difflib.get_close_matches(target_norm, pool, n=1, cutoff=0.9)
#             if best:
#                 mm = m[(m["name_norm"] == best[0]) | (m["ort_norm"] == best[0])]
#         if not mm.empty:
#             # Nimm einfach die erste ARS
#             out["ARS"] = str(mm.iloc[0]["ARS"]).strip()

#     return out
