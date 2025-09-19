# --- ADD-ONs fürs Dashboard: Laden/Normieren/Statistiken --------------------
# src/gui/helpers.py
from __future__ import annotations
import pandas as pd
import numpy as np
import difflib
import streamlit as st
from typing import Iterable, Optional, List

# --- robuste Hilfen ---------------------------------------------------------

def _first_existing(columns, candidates):
    """
    Liefert den *ersten* vorhandenen Spaltennamen aus 'candidates' (case-insensitive),
    der in 'columns' existiert. Sonst None.
    """
    if columns is None:
        return None
    cl = {str(c).lower(): c for c in columns}
    for cand in candidates:
        c = cl.get(str(cand).lower())
        if c:
            return c
    return None


# ---------- Normalisierung für Namen ----------
def _norm(s: str) -> str:
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


# ---------- (NEU) Immo-Datensatz laden & vereinheitlichen ----------
@st.cache_data(show_spinner=False)
def load_immo_data(path: str = "./data/clean/immo_train_joined.csv") -> pd.DataFrame:
    """
    Lädt den Immo-Datensatz und normalisiert auf ein einheitliches Schema:
    mind. ['state', 'city', 'coldRent'] (+ optionale Felder, falls vorhanden).

    Wir akzeptieren viele alternative Spaltennamen und berechnen coldRent,
    falls nur Warmmiete + Nebenkosten / baseRent vorhanden sind.
    """
    df = pd.read_csv(path)

    # Kandidatenlisten für typische Quellen (ImmoScout, eigene Joins, etc.)
    state_col   = _first_existing(df.columns, ["state", "federal_state", "bundesland", "region"])
    city_col    = _first_existing(df.columns, ["city", "ort", "municipality", "place", "stadt"])
    area_col    = _first_existing(df.columns, ["area_sqm", "livingSpace", "wohnflaeche", "wohnfläche", "size"])
    rooms_col   = _first_existing(df.columns, ["rooms", "anzahl_zimmer"])
    cold_col    = _first_existing(df.columns, ["coldRent", "netColdRent", "net_rent", "netRent", "baseRent"])
    warm_col    = _first_existing(df.columns, ["totalRent", "warmmiete", "warmRent"])
    nk_col      = _first_existing(df.columns, ["serviceCharge", "nebenkosten"])
    plz_col     = _first_existing(df.columns, ["plz", "zip_code", "postalCode"])
    balc_col    = _first_existing(df.columns, ["balcony", "hasBalcony"])
    kitc_col    = _first_existing(df.columns, ["hasKitchen", "kitchen", "einbaukueche", "einbauküche"])

    # coldRent berechnen, falls nicht vorhanden
    if cold_col is None:
        if warm_col and nk_col:
            df["coldRent"] = pd.to_numeric(df[warm_col], errors="coerce") - pd.to_numeric(df[nk_col], errors="coerce")
            cold_col = "coldRent"
        else:
            # manchmal gibt es 'baseRent'
            base_col = _first_existing(df.columns, ["baseRent", "baserent"])
            if base_col:
                cold_col = base_col

    # Umbenennen ins Zielschema (nur vorhandene Spalten)
    rename = {}
    if state_col: rename[state_col] = "state"
    if city_col:  rename[city_col]  = "city"
    if cold_col:  rename[cold_col]  = "coldRent"
    if area_col:  rename[area_col]  = "area_sqm"
    if rooms_col: rename[rooms_col] = "rooms"
    if plz_col:   rename[plz_col]   = "plz"
    if balc_col:  rename[balc_col]  = "balcony"
    if kitc_col:  rename[kitc_col]  = "hasKitchen"

    df = df.rename(columns=rename)

    # Pflichtfelder prüfen
    missing = [c for c in ["state", "city", "coldRent"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Immo-Datensatz ({path}) konnte nicht normalisiert werden – fehlende Kernspalten: {missing}. "
            f"Passe ggf. die Spalten-Mappings in helpers.load_immo_data an."
        )

    # leichte Bereinigung
    for c in ["state", "city"]:
        df[c] = df[c].astype(str).str.strip()

    df["coldRent"] = pd.to_numeric(df["coldRent"], errors="coerce")

    # Optional-Flags in ints wandeln (0/1), falls vorhanden
    for flag in ["balcony", "hasKitchen"]:
        if flag in df.columns:
            df[flag] = df[flag].astype(float).fillna(0).astype(int)

    return df


# ---- Dashboard-Loader: Immo + Zensus + Aggregationen ----
@st.cache_data(show_spinner=False)
def load_dashboard_frames(
    immo_path: str | None = None,
    zensus_path: str = "./data/clean/zensus_0005_clean1.csv",
):
    """
    Lädt den normalisierten Immo-Datensatz (load_immo_data) und Zensus 0005 (load_zensus0005)
    und erzeugt einige sinnvolle Aggregationen fürs Dashboard.

    Returns
    -------
    immo : pd.DataFrame
        Inserate mit vereinheitlichten Spalten (state, city, plz, area_sqm, rooms, coldRent, ...)
    z5 : pd.DataFrame
        Zensus-Tabelle mit numerischen Spalten + 'GKZ'/'name_norm' wenn vorhanden.
    meta : dict[str, pd.DataFrame]
        Aggregationen:
          - agg_state: Anzahl Inserate je Bundesland
          - agg_city:  Anzahl Inserate je Stadt
          - feat_by_plz: Summen ausgewählter Feature-Flags (balcony, hasKitchen, garden, lift) je PLZ
    """
    # 1) Daten laden
    immo = load_immo_data(immo_path)
    z5 = load_zensus0005(zensus_path)

    # 2) Normspalten
    immo = immo.copy()
    immo["city_norm"] = immo["city"].map(_norm)
    if "name_norm" not in z5.columns and "Gemeindename" in z5.columns:
        z5["name_norm"] = z5["Gemeindename"].map(_norm)

    # 3) Aggregationen
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

    feature_cols = [c for c in ["balcony", "hasKitchen", "garden", "lift"] if c in immo.columns]
    if feature_cols:
        feat_by_plz = (
            immo.groupby("plz", dropna=False)[feature_cols]
                .sum(numeric_only=True)
                .reset_index()
        )
    else:
        feat_by_plz = pd.DataFrame(columns=["plz"])

    meta = {
        "agg_state": agg_state,
        "agg_city": agg_city,
        "feat_by_plz": feat_by_plz,
    }
    return immo, z5, meta


# ---------- Zensus 0005 laden ----------
@st.cache_data(show_spinner=False)
def load_zensus0005(path: str = "./data/clean/zensus_0005_clean1.csv") -> pd.DataFrame:
    """
    Lädt Zensus-0005 (Gemeindemieten nach Bauperiode) und vereinheitlicht Spalten:
    - GKZ/ARS/AGS -> 'GKZ'
    - Gemeindename -> 'Gemeindename'
    - numerische Periode-Spalten in float
    - zusätzliche 'name_norm' für Joins
    """
    z5 = pd.read_csv(path, dtype=str)

    key_gkz   = _first_existing(z5.columns, ["GKZ", "gkz", "ARS", "ars", "AGS", "ags"])
    key_name  = _first_existing(z5.columns, ["Gemeindename", "gemeindename", "Gemeinde", "gemeinde", "name"])
    key_total = _first_existing(z5.columns, ["Insgesamt", "insgesamt", "total"])

    if not key_gkz:
        raise ValueError(f"Keine GKZ/ARS/AGS-Spalte gefunden. Vorhandene Spalten: {list(z5.columns)}")

    if not key_name:
        raise ValueError(f"Keine Gemeindename-Spalte gefunden. Vorhandene Spalten: {list(z5.columns)}")

    z5 = z5.rename(columns={key_gkz: "GKZ", key_name: "Gemeindename"})
    if key_total and key_total != "Insgesamt":
        z5 = z5.rename(columns={key_total: "Insgesamt"})

    # numerische Spalten -> float
    for c in z5.columns:
        if c not in ("GKZ", "Gemeindename"):
            z5[c] = pd.to_numeric(z5[c], errors="coerce")

    # Normalisierter Name für fuzzy-Join
    def _norm(s: str) -> str:
        if s is None:
            return ""
        s = str(s).lower().strip()
        s = (s.replace("ß", "ss")
               .replace("-", " ")
               .replace("(", "").replace(")", "")
               .replace(".", "").replace(",", "")
               .replace(" hansestadt", "").replace(" stadt", ""))
        while "  " in s:
            s = s.replace("  ", " ")
        return s.strip()

    z5["name_norm"] = z5["Gemeindename"].map(_norm)
    return z5



# ---------- (Optionale) AGS/PLZ-Mapping-Tabelle ----------
@st.cache_data(show_spinner=False)
def load_ags_map(path: str = "./data/ags_gkz.csv") -> pd.DataFrame | None:
    try:
        m = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    except Exception:
        return None
    cols = {c.lower(): c for c in m.columns}
    m = m.rename(
        columns={
            cols.get("ars", "ARS"): "ARS",
            cols.get("ags", "AGS"): "AGS",
            cols.get("gemeindename", "Gemeindename"): "Gemeindename",
            cols.get("plz", "PLZ"): "PLZ",
            cols.get("ort", "Ort"): "Ort",
        }
    )
    for c in ("Gemeindename", "Ort", "PLZ", "ARS", "AGS"):
        if c not in m.columns:
            return None
    m["name_norm"] = m["Gemeindename"].map(_norm)
    m["ort_norm"] = m["Ort"].map(_norm)
    return m[["ARS", "AGS", "PLZ", "Gemeindename", "Ort", "name_norm", "ort_norm"]].drop_duplicates()


# ---------- Stadt/Ort → ARS / Zensus-Defaults ----------
def resolve_location_defaults(
    user_city: str,
    periode: str = "2016_plus",
    zensus_path: str = "./data/clean/zensus_0005_clean1.csv",
    ags_map_path: str = "./data/ags_gkz.csv",
) -> dict:
    """
    Liefert sinnvolle Default-Werte aus dem Zensus:
    { 'ARS', 'zensus_total', 'zensus_decade', 'zensus_factor' }
    """
    out = {"ARS": None, "zensus_total": None, "zensus_decade": None, "zensus_factor": None}
    if not user_city:
        return out

    z5 = load_zensus0005(zensus_path)
    m = load_ags_map(ags_map_path)

    target_norm = _norm(user_city)

    # exakter/fuzzy Treffer im Zensus
    hit = z5[z5.get("name_norm", "") == target_norm]
    if hit.empty and "name_norm" in z5.columns:
        choices = z5["name_norm"].dropna().unique().tolist()
        best = difflib.get_close_matches(target_norm, choices, n=1, cutoff=0.87)
        if best:
            hit = z5[z5["name_norm"] == best[0]]

    if not hit.empty:
        row = hit.iloc[0]
        if "Insgesamt" in row.index:
            out["zensus_total"] = float(row.get("Insgesamt")) if pd.notna(row.get("Insgesamt")) else None
        if periode in row.index:
            zdec = row[periode]
            out["zensus_decade"] = float(zdec) if pd.notna(zdec) else None
            if pd.notna(row.get("Insgesamt")) and pd.notna(zdec) and row["Insgesamt"] not in (0, "0"):
                out["zensus_factor"] = float(zdec) / float(row["Insgesamt"])
        if pd.notna(row.get("GKZ")):
            out["ARS"] = str(row["GKZ"]).strip()

    if not out["ARS"] and m is not None:
        mm = m[(m["name_norm"] == target_norm) | (m["ort_norm"] == target_norm)]
        if not mm.empty:
            out["ARS"] = str(mm.iloc[0]["ARS"]).strip()

    return out


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
''