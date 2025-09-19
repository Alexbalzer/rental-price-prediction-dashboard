# src/gui/sandbox.py
from __future__ import annotations

import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---- Pfade / Defaults -------------------------------------------------------

MODEL_PATH = os.getenv("PRICING_MODEL_PATH", "./data/pricing_model.pkl")
METRICS_PATH = "./models/metrics.json"

# Diese Spaltennamen müssen zu deinem Trainings-Pipeline-Setup passen:
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
BIN_FEATURES: List[str] = [
    "garden",
    "lift",
    "hasKitchen",
]

ALL_FEATURES: List[str] = (
    NUM_FEATURES + CAT_FEATURES + BIN_FEATURES
)


# ---- Caches -----------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str = MODEL_PATH):
    return joblib.load(path)


def _load_metrics(path: str = METRICS_PATH) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---- UI Helper --------------------------------------------------------------

def _section_header(title: str):
    st.markdown(f"### {title}")


def _number(label: str, value: float, minv: float, maxv: float, step: float = 1.0):
    return st.slider(label, min_value=minv, max_value=maxv, value=value, step=step)


# ---- Public API -------------------------------------------------------------

def render_price_sandbox():
    """
    Rendert die interaktive Preis-Sandbox (Modell testen).
    In app.py im Tab „Mietpreis-Vorschlag“ aufrufen:  render_price_sandbox()
    """
    st.divider()
    st.markdown("### 📊 Preis-Sandbox – Modell testen & (neu) trainieren")

    # Modell laden
    pipe = None
    try:
        pipe = _load_model_cached(MODEL_PATH)
        st.success("Modell geladen.")
    except Exception as e:
        st.error("Konnte Modell nicht laden. Trainiere zuerst unter 'Daten & Training'.")
        st.exception(e)
        return  # ohne Modell macht die Sandbox keinen Sinn

    # Optional: letzte Trainings-Metriken anzeigen
    metrics = _load_metrics(METRICS_PATH)
    if metrics:
        st.caption(
            f"Letztes Training · n={metrics.get('n_train','?')}/{metrics.get('n_test','?')} · "
            f"MAE={metrics.get('mae_eur','?'):.2f}€ · R²={metrics.get('r2','?'):.3f}"
        )

    # ---------------- Eingaben ----------------
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

        typeOfFlat = st.selectbox(
            "Wohnungstyp",
            ["apartment", "roof_storey", "ground_floor", "other"],
            index=0,
        )
        heatingType = st.selectbox(
            "Heizungsart",
            ["central_heating", "district_heating", "gas_heating", "other"],
            index=0,
        )
        periode = st.selectbox(
            "Baujahr-Periode (Zensus-0005)",
            ["vor_1919", "1919_1949", "1950_1959", "1970_1979", "1980_1989", "2000_2009", "2010_2015", "2016_plus"],
            index=7,
        )

    st.markdown("#### Zensus (optional)")
    z_total  = st.number_input("Zensus: Gesamt €/m² (optional)", value=4.99, min_value=0.0, step=0.01)
    z_decade = st.number_input("Zensus: Dekade €/m² (optional)", value=4.95, min_value=0.0, step=0.01)
    z_factor = st.number_input("Zensus: Faktor (Dekade/Gesamt) (optional)", value=0.99, min_value=0.0, step=0.01)

    # Eingaben in ein DataFrame (genau die Feature-Spalten, die das Modell kennt)
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
        "heatingType_clean": heatingType,
        "periode_0005": periode,
        "garden": int(bool(garden)),
        "lift": int(bool(lift)),
        "hasKitchen": int(bool(hasKitchen)),
    }], columns=ALL_FEATURES)

    # ---------------- Vorhersage ----------------
    st.markdown("#### Vorhersage")
    try:
        y_hat = float(pipe.predict(X_user)[0])
        st.success(f"**Prognose Nettokaltmiete:** {y_hat:.2f} €")
        if area_sqm > 0:
            st.caption(f"≈ {y_hat/area_sqm:.2f} €/m²")
    except Exception as e:
        st.error("Vorhersage fehlgeschlagen. Prüfe, ob alle benötigten Feature-Spalten vorhanden sind.")
        st.exception(e)

    # ---------------- Feature Importances (falls vorhanden) ----------------
    with st.expander("Feature-Importances", expanded=False):
        try:
            rf = pipe.named_steps.get("rf", None)
            prep = pipe.named_steps.get("prep", None)

            if rf is not None and hasattr(rf, "feature_importances_"):
                # Feature-Namen aus dem Preprocessor holen (falls verfügbar)
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
