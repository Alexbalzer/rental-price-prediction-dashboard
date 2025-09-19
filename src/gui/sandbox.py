# src/gui/sandbox.py
# Preis-Sandbox: Standort + Zensus-Autofill + Vorhersage

from __future__ import annotations

import json
import os
from typing import List, Optional, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
# statt eigener Loader:
from gui.helpers import load_zensus0005, load_ags_map, resolve_location_defaults


# ---- Pfade ------------------------------------------------------------------

MODEL_PATH   = os.getenv("PRICING_MODEL_PATH", "./data/pricing_model.pkl")
METRICS_PATH = "./models/metrics.json"

ZENSUS_0005_CSV = "./data/clean/zensus_0005_clean1.csv"  # enthält GKZ/ARS, Gemeindename, Insgesamt, Dekaden…
AGS_GKZ_CSV     = "./data/ags_gkz.csv"                   # sep=';' – Spalten u.a. ARS/AGS/Gemeindename/PLZ/Ort

# ---- Featureliste (muss zum trainierten Modell passen) ----------------------

NUM_FEATURES: List[str] = [
    "area_sqm",
    "rooms",
    "floor",
    "pricetrend",
    "serviceCharge",
    # optionale Zensus-Features (numerisch)
    "zensus_miete_total",
    "zensus_miete_decade",
    "zensus_factor_decade",
]
CAT_FEATURES: List[str] = [
    "typeOfFlat",
    "heatingType_clean",
    "periode_0005",
]
BIN_FEATURES: List[str] = ["garden", "lift", "hasKitchen"]

ALL_FEATURES: List[str] = NUM_FEATURES + CAT_FEATURES + BIN_FEATURES

DEKADE_COLS = [
    "vor_1919",
    "1919_1949",
    "1950_1959",
    "1970_1979",
    "1980_1989",
    "2000_2009",
    "2010_2015",
    "2016_plus",
]
PMAP: Dict[str, str] = {c: c for c in DEKADE_COLS}

# ---- Caches -----------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str = MODEL_PATH):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def _load_metrics(path: str = METRICS_PATH) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_zensus0005(path: str = ZENSUS_0005_CSV) -> pd.DataFrame:
    z5 = pd.read_csv(path, dtype=str)
    # GKZ kann "GKZ" heißen (12-stellig). Wir nennen es hier einheitlich ARS.
    if "ARS" not in z5.columns:
        if "GKZ" in z5.columns:
            z5 = z5.rename(columns={"GKZ": "ARS"})
        # sonst: keine ARS-Spalte vorhanden
    # numerische Spalten
    num_cols = [c for c in z5.columns if c not in ("ARS", "Gemeindename")]
    for c in num_cols:
        z5[c] = pd.to_numeric(z5[c], errors="coerce")
    # falls Faktor-Spalten fehlen -> berechnen
    if "Insgesamt" in z5.columns:
        for c in DEKADE_COLS:
            fcol = f"factor_{c}"
            if fcol not in z5.columns and c in z5.columns:
                z5[fcol] = z5[c] / z5["Insgesamt"]
    return z5

@st.cache_data(show_spinner=False)
def _load_ags_map(path: str = AGS_GKZ_CSV) -> pd.DataFrame:
    # Mapping-Datei ist Semikolon-getrennt
    m = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    # Spalten robust normalize
    for col in ["PLZ", "Ort", "Gemeindename", "ARS", "AGS"]:
        if col not in m.columns:
            m[col] = None
    # PLZ auf 5-stellig herausziehen
    m["PLZ"] = m["PLZ"].astype(str).str.extract(r"(\d{5})", expand=False)
    # Norm-Spalten für Namen
    m["ort_norm"] = _norm_city(m["Ort"])
    m["gemeinde_norm"] = _norm_city(m["Gemeindename"])
    # Ein paar Reinigungen
    if "ARS" in m.columns and m["ARS"].notna().any():
        m["ARS"] = m["ARS"].astype(str).str.replace(r"\D", "", regex=True)
    if "AGS" in m.columns and m["AGS"].notna().any():
        m["AGS"] = m["AGS"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
    return m

# ---- Helpers ----------------------------------------------------------------

def _norm_city(s: pd.Series | str) -> pd.Series | str:
    if isinstance(s, str):
        s = pd.Series([s], dtype="string")
        one = True
    else:
        one = False
    s = s.astype("string").str.lower().str.strip()
    s = (
        s.str.replace("ß", "ss")
         .str.replace("-", " ")
         .str.replace("(", "", regex=False)
         .str.replace(")", "", regex=False)
         .str.replace(".", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.replace("  ", " ")
         .str.replace(" stadt", "", regex=False)
         .str.replace(" hansestadt", "", regex=False)
         .str.replace(" kreisfreie stadt", "", regex=False)
         .str.replace(" kreis", "", regex=False)
         .str.replace(" amt ", " ", regex=False)
         .str.strip()
    )
    return s.iloc[0] if one else s

def _defaults_from_city_or_plz(
    city_or_plz: str, periode: str, z5: pd.DataFrame, m: pd.DataFrame
) -> Dict[str, Optional[float]]:
    """
    Liefert Defaults (total, decade, factor) für die gegebene Stadt/PLZ + Dekade.
    Strategie:
      1) Wenn 5-stellige PLZ → falls eindeutig genau 1 ARS → nimm diese.
      2) Sonst Gemeindename normalisieren → wenn eindeutig → nimm diese.
    """
    result = {"total": None, "decade": None, "factor": None}

    if not city_or_plz or periode not in PMAP:
        return result

    key_col = PMAP[periode]
    fcol = f"factor_{key_col}"

    # 1) Versuch: PLZ-eindeutig
    plz = pd.Series([city_or_plz]).astype(str).str.extract(r"(\d{5})", expand=False).iloc[0]
    ars_match = None
    if plz and m["PLZ"].notna().any():
        sub = m[m["PLZ"] == plz]
        # Auswahlregel: wenn genau eine ARS vorkommt → nimm sie
        vals = sub["ARS"].dropna().unique()
        if len(vals) == 1:
            ars_match = vals[0]

    # 2) Versuch: Gemeindename
    if ars_match is None:
        cname = _norm_city(city_or_plz)
        sub = m[m["gemeinde_norm"] == cname]
        vals = sub["ARS"].dropna().unique()
        if len(vals) == 1:
            ars_match = vals[0]

    if ars_match is None:
        return result

    row = z5[z5["ARS"] == ars_match]
    if row.empty:
        return result

    # Werte ziehen
    total  = row["Insgesamt"].iloc[0] if "Insgesamt" in row.columns else np.nan
    decade = row[key_col].iloc[0] if key_col in row.columns else np.nan
    factor = row[fcol].iloc[0] if fcol in row.columns else (decade / total if (pd.notna(decade) and pd.notna(total) and total) else np.nan)

    result["total"]  = float(total)  if pd.notna(total)  else None
    result["decade"] = float(decade) if pd.notna(decade) else None
    result["factor"] = float(factor) if pd.notna(factor) else None
    return result

def _section_header(title: str):
    st.markdown(f"### {title}")

def _number(label: str, value: float, minv: float, maxv: float, step: float = 1.0):
    return st.slider(label, min_value=minv, max_value=maxv, value=value, step=step)

# ---- Public API -------------------------------------------------------------

def render_price_sandbox():
    st.divider()
    st.markdown("### 📊 Preis-Sandbox – Modell testen & (neu) trainieren")

    # Modell laden
    try:
        pipe = _load_model_cached(MODEL_PATH)
        st.success("Modell geladen.")
    except Exception as e:
        st.error("Konnte Modell nicht laden. Trainiere zuerst unter 'Daten & Training'.")
        st.exception(e)
        return

    # Metriken
    metrics = _load_metrics(METRICS_PATH)
    if metrics:
        try:
            st.caption(
                f"Letztes Training · n={metrics.get('n_train','?')}/{metrics.get('n_test','?')} · "
                f"MAE={float(metrics.get('mae_eur', 0.0)):.2f}€ · R²={float(metrics.get('r2', 0.0)):.3f}"
            )
        except Exception:
            pass

    # Daten laden (für Standort-Autofill)
    z5 = _load_zensus0005()
    m  = _load_ags_map()

    # ---------------- Standort & Zensus-Defaults ----------------
    st.subheader("Standort & Baujahr")
    col_loc, col_per = st.columns([2, 1])
    with col_loc:
        city_input = st.text_input(
            "PLZ oder Stadt/Gemeinde (optional)",
            placeholder="z. B. 04109 oder Leipzig, Aachen, Gera …",
        )
    with col_per:
        periode = st.selectbox(
            "Baujahr-Periode (Zensus-0005)",
            DEKADE_COLS,
            index=7,
        )

    # Defaults ermitteln (bevor die Zensus-Inputs gerendert werden!)
    defaults = {"total": None, "decade": None, "factor": None}
    if city_input.strip():
        try:
            defaults = _defaults_from_city_or_plz(city_input.strip(), periode, z5, m)
        except Exception as e:
            st.info("Konnte keine eindeutigen Zensus-Defaults aus dem Standort ableiten.")
            st.caption(str(e))

    # ---------------- Basis-Eingaben ----------------
    st.subheader("Eingaben")

    col_l, col_r = st.columns([2, 2])
    with col_l:
        area_sqm  = _number("Wohnfläche (m²)", 67.0, 10.0, 250.0, 1.0)
        rooms     = _number("Zimmer", 2.0, 1.0, 9.0, 0.5)
        floor     = _number("Etage", 1.0, 0.0, 30.0, 1.0)
        pricetrend = _number("pricetrend", 3.0, 0.0, 10.0, 0.01)
        service   = _number("Nebenkosten (serviceCharge, €)", 180.0, 0.0, 1000.0, 1.0)

    with col_r:
        garden     = st.checkbox("Garten", True)
        lift       = st.checkbox("Aufzug", False)
        hasKitchen = st.checkbox("Einbauküche", True)

        # Wohnungstyp (DE Labels, EN Werte)
        type_options = ["apartment", "roof_storey", "ground_floor", "other"]
        type_labels  = {
            "apartment":    "Wohnung",
            "roof_storey":  "Dachgeschoss",
            "ground_floor": "Erdgeschoss",
            "other":        "Sonstiges",
        }
        type_help = (
            "Erklärung:\n"
            "- Wohnung = apartment\n"
            "- Dachgeschoss = roof_storey\n"
            "- Erdgeschoss = ground_floor\n"
            "- Sonstiges = other"
        )
        typeOfFlat = st.selectbox(
            "Wohnungstyp",
            type_options,
            index=0,
            format_func=lambda v: type_labels.get(v, v),
            help=type_help,
        )

        # Heizungsart (DE Labels, EN Werte)
        heat_options = ["central_heating", "district_heating", "gas_heating", "oil_heating", "other"]
        heat_labels  = {
            "central_heating":  "Zentralheizung",
            "district_heating": "Fern-/Nahwärme",
            "gas_heating":      "Gasheizung",
            "oil_heating":      "Ölheizung",
            "other":            "Elektro/sonstiges",
        }
        heat_help = (
            "Erklärung:\n"
            "- Zentralheizung = central_heating\n"
            "- Fern-/Nahwärme = district_heating\n"
            "- Gasheizung = gas_heating\n"
            "- Ölheizung = oil_heating\n"
            "- Elektro/sonstiges = other"
        )
        heatingType_clean = st.selectbox(
            "Heizungsart",
            heat_options,
            index=0,
            format_func=lambda v: heat_labels.get(v, v),
            help=heat_help,
        )

    # ---------------- Zensus (mit Defaults aus Standort) ----------------
    st.markdown("#### Zensus (optional)")
    z_total_default  = defaults["total"]  if defaults["total"]  is not None else 4.99
    z_decade_default = defaults["decade"] if defaults["decade"] is not None else 4.95
    z_factor_default = defaults["factor"] if defaults["factor"] is not None else 0.99

    z_total  = st.number_input("Zensus: Gesamt €/m² (optional)",  value=float(z_total_default),  min_value=0.0, step=0.01)
    z_decade = st.number_input("Zensus: Dekade €/m² (optional)", value=float(z_decade_default), min_value=0.0, step=0.01)
    z_factor = st.number_input("Zensus: Faktor (Dekade/Gesamt) (optional)", value=float(z_factor_default), min_value=0.0, step=0.01)

    # ---------------- Vorhersage ----------------
    st.markdown("#### Vorhersage")
    try:
        X_user = pd.DataFrame([{
            "area_sqm": area_sqm,
            "rooms": rooms,
            "floor": floor,
            "pricetrend": pricetrend,
            "serviceCharge": service,
            "zensus_miete_total": z_total,
            "zensus_miete_decade": z_decade,
            "zensus_factor_decade": z_factor,
            "typeOfFlat": typeOfFlat,
            "heatingType_clean": heatingType_clean,
            "periode_0005": periode,
            "garden": int(bool(garden)),
            "lift": int(bool(lift)),
            "hasKitchen": int(bool(hasKitchen)),
        }], columns=ALL_FEATURES)

        y_hat = float(pipe.predict(X_user)[0])
        st.success(f"**Prognose Nettokaltmiete:** {y_hat:.2f} €")
        if area_sqm > 0:
            st.caption(f"≈ {y_hat/area_sqm:.2f} €/m²")
    except Exception as e:
        st.error("Vorhersage fehlgeschlagen. Prüfe, ob alle benötigten Feature-Spalten vorhanden sind.")
        st.exception(e)

    # ---------------- Feature Importances ----------------
    with st.expander("Feature-Importances", expanded=False):
        try:
            rf = getattr(pipe, "named_steps", {}).get("rf", None)
            prep = getattr(pipe, "named_steps", {}).get("prep", None)
            if rf is not None and hasattr(rf, "feature_importances_"):
                if prep is not None and hasattr(prep, "get_feature_names_out"):
                    names = prep.get_feature_names_out()
                else:
                    names = [f"f{i}" for i in range(len(rf.feature_importances_))]
                fi = (
                    pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
                    .sort_values("importance", ascending=False)
                    .head(30)
                    .reset_index(drop=True)
                )
                st.dataframe(fi, use_container_width=True)
            else:
                st.caption("Dieses Modell liefert keine Feature-Importances (oder ist kein RandomForest).")
        except Exception as e:
            st.exception(e)
