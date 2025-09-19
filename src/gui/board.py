# ================================
# file: src/gui/board.py
# ================================
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Unsere robusten Loader kommen aus helpers.py
from gui.helpers import load_immo_data, load_zensus0005


# --------- kleine Helfer ---------
def _err_box(msg: str) -> None:
    st.error(msg)


def _load_sources() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Immo & Zensus laden – Fehler nett anzeigen."""
    try:
        immo = load_immo_data()
    except Exception as e:
        _err_box(f"Immo-Datensatz konnte nicht geladen werden: {e}")
        immo = None

    try:
        z5 = load_zensus0005()
    except Exception as e:
        _err_box(f"Zensus-Datensatz konnte nicht geladen werden: {e}")
        z5 = None

    return immo, z5


# --------- einzelne Kacheln / Sektionen ---------
def _tile_counts(immo: pd.DataFrame) -> None:
    """Inserate je Bundesland & je Stadt (Top-10)."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Inserate je Bundesland (Top 10)")
        by_state = (
            immo.groupby("state", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(by_state)

    with c2:
        st.markdown("#### Inserate je Stadt (Top 10)")
        by_city = (
            immo.groupby("city", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(by_city)


def _tile_immo_vs_zensus(immo: pd.DataFrame, z5: pd.DataFrame) -> None:
    """
    Median €/m² der Immo-Inserate vs. Zensus 'Insgesamt' je Stadt (Top-20 nach Anzahl).
    Join per normalisiertem Stadtnamen (city_norm ↔ name_norm).
    """
    st.markdown("#### Median €/m²: Immo vs. Zensus (Top 20 nach Anzahl)")

    if "rent_per_sqm" not in immo.columns:
        _err_box("In Immo-Daten fehlt 'rent_per_sqm'. Bitte Daten/Mapping prüfen.")
        return

    city_stat = (
        immo.groupby(["city", "city_norm"], dropna=False)
        .agg(n=("city", "size"), immo_median=("rent_per_sqm", "median"))
        .reset_index()
        .sort_values("n", ascending=False)
    )

    comp = city_stat.merge(
        z5[["Gemeindename", "name_norm", "Insgesamt"]],
        left_on="city_norm",
        right_on="name_norm",
        how="left",
    )

    comp["Δ (Immo–Zensus)"] = comp["immo_median"] - comp["Insgesamt"]
    show = comp.head(20)[
        ["city", "n", "immo_median", "Insgesamt", "Δ (Immo–Zensus)", "Gemeindename"]
    ].rename(
        columns={
            "city": "Stadt (Immo)",
            "Gemeindename": "Gemeindename (Zensus)",
            "immo_median": "Immo-Median €/m²",
            "Insgesamt": "Zensus €/m²",
        }
    )

    st.dataframe(show, use_container_width=True)


def _tile_features(immo: pd.DataFrame) -> None:
    """Feature-Statistiken (gesamt & nach PLZ Top-10)."""
    st.markdown("#### Feature-Anteil gesamt")
    feats = ["balcony", "kitchen", "lift", "garden"]
    feat_share = (immo[feats].mean().sort_values(ascending=False) * 100.0).round(1)
    st.bar_chart(feat_share)

    st.markdown("#### Feature-Counts nach PLZ (Top 10)")
    immo["plz_str"] = immo["plz"].astype(str).str.extract(r"(\d{5})")[0]
    by_plz = (
        immo.dropna(subset=["plz_str"])
        .groupby("plz_str")[feats]
        .sum()
        .assign(total=lambda d: d.sum(axis=1))
        .sort_values("total", ascending=False)
        .head(10)
        .drop(columns="total")
    )
    st.dataframe(by_plz, use_container_width=True)


def _tile_heatmap(immo: pd.DataFrame, geojson_path: str | Path = "./data/geo/bundeslaender.geojson") -> None:
    """
    Choropleth Heatmap Deutschland (Bundesländer).
    Erwartet eine passende GeoJSON (properties.NAME sollte den Bundesland-Namen enthalten).
    """
    geo_path = Path(geojson_path)
    st.markdown("#### Heatmap: Inserate je Bundesland")

    if not geo_path.exists():
        st.caption("ℹ️ Für die Karte `data/geo/bundeslaender.geojson` bereitstellen. Zeige stattdessen Balken.")
        by_state = immo.groupby("state", dropna=False).size().reset_index(name="count")
        st.bar_chart(by_state.set_index("state")["count"])
        return

    try:
        import plotly.express as px
    except Exception:
        st.caption("Plotly nicht installiert – zeige Balken statt Karte (`pip install plotly`).")
        by_state = immo.groupby("state", dropna=False).size().reset_index(name="count")
        st.bar_chart(by_state.set_index("state")["count"])
        return

    try:
        with open(geo_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        by_state = immo.groupby("state", dropna=False).size().reset_index(name="count")

        fig = px.choropleth(
            by_state,
            geojson=geojson,
            locations="state",
            color="count",
            featureidkey="properties.NAME",  # ggf. an GeoJSON anpassen
            projection="mercator",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Heatmap nicht verfügbar: {e}. Zeige Balken statt Karte.")
        by_state = immo.groupby("state", dropna=False).size().reset_index(name="count")
        st.bar_chart(by_state.set_index("state")["count"])


# --------- öffentliches API ---------
def render_dashboard() -> None:
    """
    Orchestriert alle Dashboard-Karten.
    Kann direkt aus app.py aufgerufen werden.
    """
    immo, z5 = _load_sources()
    if immo is None:
        return

    st.subheader("Überblick")

    # Zeile 1: Counters / Top-Listen
    _tile_counts(immo)

    # Zeile 2: Immo vs. Zensus
    if z5 is not None:
        _tile_immo_vs_zensus(immo, z5)
    else:
        st.info("Zensus nicht geladen – Übersichtsvergleich entfällt.")

    # Zeile 3: Features
    _tile_features(immo)

    # Zeile 4: (optional) Heatmap
    with st.expander("Deutschland-Heatmap", expanded=False):
        _tile_heatmap(immo)


# ================================
# PATCH für src/gui/app.py
# ================================

# (1) Ganz oben ergänzen:
# from gui.board import render_dashboard

# (2) Im Seiten-Router ersetzen:
# if page == "Dashboard":
#     render_dashboard()
