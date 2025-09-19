from __future__ import annotations  # MUSS Zeile 1 sein

# src zum Importpfad hinzufügen (nur einmal; unkritisch, wenn wir überall 'hausverwaltung.*' importieren)
import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
from gui.helpers import load_zensus0005, load_ags_map, resolve_location_defaults, load_dashboard_frames

from sqlmodel import select
from dotenv import load_dotenv
from datetime import date
import pandas as pd
import numpy as np
import joblib
import subprocess
import textwrap
import matplotlib.pyplot as plt
from pathlib import Path


from hausverwaltung.db import init_db, get_session
from hausverwaltung.models import Owner, Property, Unit, Tenant, Lease, Payment, Charge
from hausverwaltung.ml.pricing import suggest_rent_for_unit
from hausverwaltung.services.docs import render_letter, refine_with_llm, TEMPLATE_REMINDER


load_dotenv()
st.set_page_config(page_title="Hausverwaltung", page_icon="🏠", layout="wide")

# DB initialisieren
init_db()

st.title("🏠 Hausverwaltung – MVP")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Seite",
        ["Dashboard", "Stammdaten", "Verträge & Zahlungen", "Mietpreis-Vorschlag",
         "Dokumente", "Daten & Training", "Preis-Sandbox", "Einstellungen","📖 Projektbeschreibung"]
    )

# with st.sidebar:
#     st.header("Navigation")
#     # in der Navigation:
#     page = st.radio("Seite", ["Dashboard", "Stammdaten", "Verträge & Zahlungen", "Mietpreis-Vorschlag", "Dokumente", "Daten & Training", "Einstellungen"])


# ---------------- Dashboard ----------------
if page == "Dashboard":
    import altair as alt
    import pydeck as pdk

    st.subheader("Überblick")

    # aktive CSV für Dashboard/Sandbox
    DEFAULT_JOINED = Path("./data/clean/immo_train_joined.csv")
    immo_csv = Path(st.session_state.get("data_csv_path", DEFAULT_JOINED))

    try:
        immo, z5 = load_dashboard_frames(immo_csv)
    except Exception as e:
        st.error(f"Dataset konnte nicht geladen werden: {e}")
        st.stop()

    # Kennzahlen
    total_offers = len(immo)
    by_state = immo["state"].fillna("Unbekannt").value_counts().reset_index()
    by_state.columns = ["Bundesland","Anzahl"]
    by_city = immo["city"].fillna("Unbekannt").value_counts().head(15).reset_index()
    by_city.columns = ["Stadt","Anzahl"]

    colA, colB, colC = st.columns(3)
    colA.metric("Inserate gesamt", f"{total_offers:,}".replace(",", "."))
    colB.metric("Bundesländer mit Inseraten", by_state["Bundesland"].nunique())
    colC.metric("Top-Stadt (Anzahl)", f"{by_city.iloc[0]['Stadt'] if len(by_city)>0 else '-'} ({int(by_city.iloc[0]['Anzahl']) if len(by_city)>0 else 0})")

    st.markdown("### Verteilung nach Bundesland")
    chart_state = (
        alt.Chart(by_state)
        .mark_bar()
        .encode(
            x=alt.X("Anzahl:Q", title="Inserate"),
            y=alt.Y("Bundesland:N", sort="-x", title=""),
            tooltip=["Bundesland", "Anzahl"]
        )
        .properties(height=400)
    )
    st.altair_chart(chart_state, use_container_width=True)

    st.markdown("### Top-Städte")
    chart_city = (
        alt.Chart(by_city)
        .mark_bar()
        .encode(
            x=alt.X("Anzahl:Q", title="Inserate"),
            y=alt.Y("Stadt:N", sort="-x", title=""),
            tooltip=["Stadt", "Anzahl"]
        )
        .properties(height=400)
    )
    st.altair_chart(chart_city, use_container_width=True)

    # --- Kaltmiete: Ist vs. Zensus-Erwartung --------------------------------
    st.markdown("### Kaltmiete: Ist vs. Zensus-Erwartung")
    df_rent = immo.copy()
    # robuste Zielspalte (falls netRent fehlt)
    if df_rent["netRent"].isna().all() and "warmRent" in df_rent.columns:
        # optional heuristik: warm->cold (nur Demo)
        df_rent["netRent"] = df_rent["warmRent"] * 0.8

    df_rent = df_rent.dropna(subset=["area_sqm"])
    df_rent["eur_m2"] = df_rent.apply(
        lambda r: r["netRent"]/r["area_sqm"] if pd.notna(r["netRent"]) and r["area_sqm"]>0 else np.nan, axis=1
    )
    view = df_rent.dropna(subset=["eur_m2","zensus_rate"]).copy()
    if len(view) == 0:
        st.info("Keine vergleichbaren Daten vorhanden (netRent/area/zensus_rate fehlen).")
    else:
        view["Bundesland"] = view["state"].fillna("Unbekannt")
        chart_scatter = (
            alt.Chart(view.sample(min(5000, len(view)), random_state=42))
            .mark_circle(size=35, opacity=0.45)
            .encode(
                x=alt.X("zensus_rate:Q", title="Zensus €/m²"),
                y=alt.Y("eur_m2:Q", title="Ist €/m²"),
                color=alt.Color("Bundesland:N", legend=None),
                tooltip=["city","PLZ","Bundesland","area_sqm","eur_m2","zensus_rate"]
            )
            .properties(height=400)
        )
        st.altair_chart(chart_scatter, use_container_width=True)

        # Differenz-Boxplot (Ist - Zensus)
        view["delta_eur_m2"] = view["eur_m2"] - view["zensus_rate"]
        chart_delta = (
            alt.Chart(view)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("Bundesland:N", sort="-y", title=""),
                y=alt.Y("delta_eur_m2:Q", title="Δ €/m² (Ist − Zensus)"),
                tooltip=["Bundesland","delta_eur_m2"]
            )
            .properties(height=380)
        )
        st.altair_chart(chart_delta, use_container_width=True)

    # --- Feature-Statistik nach PLZ -----------------------------------------
    st.markdown("### Feature-Statistiken (nach PLZ)")
    feat = immo.copy()
    # nur PLZ mit Format
    feat = feat[feat["PLZ"].notna()]
    agg = feat.groupby("PLZ").agg(
        n=("PLZ","count"),
        balkon=("balcony","sum"),
        kueche=("hasKitchen","sum"),
        garten=("garden","sum"),
        lift=("lift","sum"),
    ).reset_index()

    top_plz = agg.sort_values("n", ascending=False).head(25)
    chart_feats = (
        alt.Chart(top_plz.melt(id_vars=["PLZ","n"], value_vars=["balkon","kueche","garten","lift"], var_name="Feature", value_name="Anzahl"))
        .mark_bar()
        .encode(
            x=alt.X("Anzahl:Q"),
            y=alt.Y("PLZ:N", sort="-x"),
            color=alt.Color("Feature:N"),
            tooltip=["PLZ","Feature","Anzahl","n"]
        )
        .properties(height=500)
    )
    st.altair_chart(chart_feats, use_container_width=True)

    # --- Interaktive Heatmap (falls Koordinaten vorhanden) -------------------
    st.markdown("### Karte – Dichte der Inserate")
    has_coords = immo[["lat","lon"]].dropna().shape[0] > 0
    if has_coords:
        df_map = immo.dropna(subset=["lat","lon"]).copy()
        init_view = {
            "latitude": float(df_map["lat"].astype(float).median()),
            "longitude": float(df_map["lon"].astype(float).median()),
            "zoom": 5, "pitch": 40
        }
        layer = pdk.Layer(
            "HexagonLayer",
            data=df_map,
            get_position=["lon","lat"],
            radius=9000, elevation_scale=30, elevation_range=[0, 3000],
            extruded=True, pickable=True,
        )
        tooltip = {"html": "<b>Anzahl:</b> {pointCount}", "style": {"color": "white"}}
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/dark-v10",
                                 initial_view_state=init_view,
                                 layers=[layer],
                                 tooltip=tooltip))
        st.caption("Zoom/Drag: Draufzoomen → Dichte wird feiner. Tooltips zeigen Counts.")
    else:
        st.info("Keine Koordinaten in der CSV gefunden – zeige stattdessen die Balkencharts je Bundesland/Stadt (siehe oben).")

# ---------------- Stammdaten ----------------
elif page == "Stammdaten":
    tab1, tab2, tab3, tab4 = st.tabs(["Eigentümer", "Objekte", "Einheiten", "Mieter"])

    with tab1:
        st.write("### Eigentümer")
        with get_session() as sess:
            with st.form("owner_form"):
                name = st.text_input("Name")
                email = st.text_input("E-Mail")
                phone = st.text_input("Telefon")
                if st.form_submit_button("➕ Anlegen"):
                    if name.strip():
                        sess.add(Owner(name=name.strip(), email=email or None, phone=phone or None))
                        sess.commit()
                        st.success("Eigentümer gespeichert.")
                    else:
                        st.error("Name ist Pflicht.")
            owners = sess.exec(select(Owner)).all()
            st.table([{"ID": o.id, "Name": o.name, "E-Mail": o.email, "Telefon": o.phone} for o in owners])

    with tab2:
        st.write("### Objekte")
        with get_session() as sess:
            owners = sess.exec(select(Owner)).all()
            with st.form("prop_form"):
                owner = st.selectbox("Eigentümer", owners, format_func=lambda o: f"{o.id} – {o.name}" if o else "-")
                street = st.text_input("Straße und Hausnr.")
                zip_code = st.text_input("PLZ")
                city = st.text_input("Stadt")
                year_built = st.number_input("Baujahr", step=1, min_value=1800, max_value=2100, value=1990)
                if st.form_submit_button("➕ Anlegen"):
                    if owner and street and zip_code and city:
                        sess.add(Property(owner_id=owner.id, street=street, zip_code=zip_code, city=city, year_built=int(year_built)))
                        sess.commit()
                        st.success("Objekt gespeichert.")
                    else:
                        st.error("Bitte Eigentümer und Adresse angeben.")
            props = sess.exec(select(Property)).all()
            st.table([{"ID": p.id, "Eigentümer": p.owner_id, "Adresse": f"{p.street}, {p.zip_code} {p.city}", "Baujahr": p.year_built} for p in props])

    with tab3:
        st.write("### Einheiten")
        with get_session() as sess:
            props = sess.exec(select(Property)).all()
            with st.form("unit_form"):
                prop = st.selectbox("Objekt", props, format_func=lambda p: f"{p.id} – {p.street}, {p.city}" if p else "-")
                unit_no = st.text_input("Wohnungs-Nr.")
                floor = st.number_input("Etage", step=1, value=0)
                rooms = st.number_input("Zimmer", step=0.5, value=2.0)
                size = st.number_input("Fläche (m²)", step=1.0, value=60.0)
                condition = st.selectbox("Zustand", ["", "neu", "saniert", "renoviert", "gebraucht"])
                if st.form_submit_button("➕ Anlegen"):
                    if prop and unit_no:
                        sess.add(Unit(property_id=prop.id, unit_no=unit_no, floor=int(floor), rooms=float(rooms), size_sqm=float(size), condition=condition or None))
                        sess.commit()
                        st.success("Einheit gespeichert.")
                    else:
                        st.error("Bitte Objekt und Wohnungs-Nr. angeben.")
            units = sess.exec(select(Unit)).all()
            st.table([{"ID": u.id, "Objekt": u.property_id, "Whg": u.unit_no, "Etage": u.floor, "Zimmer": u.rooms, "m²": u.size_sqm, "Zustand": u.condition} for u in units])

    with tab4:
        st.write("### Mieter")
        with get_session() as sess:
            with st.form("tenant_form"):
                fn = st.text_input("Vorname")
                ln = st.text_input("Nachname")
                email = st.text_input("E-Mail")
                phone = st.text_input("Telefon")
                if st.form_submit_button("➕ Anlegen"):
                    if fn.strip() and ln.strip():
                        sess.add(Tenant(first_name=fn.strip(), last_name=ln.strip(), email=email or None, phone=phone or None))
                        sess.commit()
                        st.success("Mieter gespeichert.")
                    else:
                        st.error("Vor- und Nachname sind Pflicht.")
            tenants = sess.exec(select(Tenant)).all()
            st.table([{"ID": t.id, "Name": f"{t.first_name} {t.last_name}", "E-Mail": t.email, "Telefon": t.phone} for t in tenants])

# ---------------- Verträge & Zahlungen ----------------
elif page == "Verträge & Zahlungen":
    st.write("### Verträge")
    with get_session() as sess:
        units = sess.exec(select(Unit)).all()
        tenants = sess.exec(select(Tenant)).all()
        with st.form("lease_form"):
            unit = st.selectbox("Einheit", units, format_func=lambda u: f"{u.id} – {u.unit_no} ({u.property_id})" if u else "-")
            tenant = st.selectbox("Mieter", tenants, format_func=lambda t: f"{t.id} – {t.first_name} {t.last_name}" if t else "-")
            start = st.date_input("Start", value=date.today())
            end = st.date_input("Ende (optional)", value=None)
            rent_base = st.number_input("Nettokaltmiete (€)", step=10.0, value=900.0)
            rent_total = st.number_input("Warmmiete (optional)", step=10.0, value=0.0)
            deposit = st.number_input("Kaution (optional)", step=50.0, value=0.0)
            if st.form_submit_button("➕ Vertrag anlegen"):
                if unit and tenant:
                    sess.add(Lease(unit_id=unit.id, tenant_id=tenant.id, start_date=start, end_date=end if end != date.min else None,
                                   rent_base=float(rent_base), rent_total=(float(rent_total) or None), deposit=(float(deposit) or None)))
                    sess.commit()
                    st.success("Vertrag gespeichert.")
        leases = sess.exec(select(Lease)).all()
        st.table([{"ID": l.id, "Whg": l.unit_id, "Mieter": l.tenant_id, "Start": l.start_date, "Ende": l.end_date, "Kalt (€)": l.rent_base, "Warm (€)": l.rent_total} for l in leases])

    st.write("### Zahlungen")
    with get_session() as sess:
        leases = sess.exec(select(Lease)).all()
        with st.form("pay_form"):
            lease = st.selectbox("Vertrag", leases, format_func=lambda l: f"{l.id} – Unit {l.unit_id} / Tenant {l.tenant_id}" if l else "-")
            paid_on = st.date_input("Datum", value=date.today())
            amount = st.number_input("Betrag (€)", step=10.0, value=900.0)
            ptype = st.selectbox("Typ", ["rent", "deposit", "other"])
            note = st.text_input("Notiz", "")
            if st.form_submit_button("➕ Zahlung buchen"):
                if lease:
                    sess.add(Payment(lease_id=lease.id, paid_on=paid_on, amount=float(amount), type=ptype, note=(note or None)))
                    sess.commit()
                    st.success("Zahlung gespeichert.")
        pays = sess.exec(select(Payment)).all()
        st.table([{"ID": p.id, "Vertrag": p.lease_id, "Datum": p.paid_on, "Betrag": p.amount, "Typ": p.type, "Notiz": p.note} for p in pays])

# ---------------- Mietpreis-Vorschlag ----------------
elif page == "Mietpreis-Vorschlag":
    st.write("### Vorschlag für Nettokaltmiete")
    with get_session() as sess:
        units = sess.exec(select(Unit)).all()
        unit = st.selectbox("Einheit", units, format_func=lambda u: f"{u.id} – {u.unit_no} ({u.size_sqm} m²)")
        if st.button("💡 Vorschlag berechnen") and unit:
            est = suggest_rent_for_unit(sess, unit.id)
            if est is not None:
                st.success(f"Vorgeschlagene Nettokaltmiete: **{est:.2f} €**")
                st.caption("Hinweis: Baseline-Heuristik. Später durch trainiertes Modell ersetzt.")
            else:
                st.error("Keine Einheit gefunden.")

# ---------------- Dokumente ----------------
elif page == "Dokumente":
    st.write("### Zahlungserinnerung erstellen")
    with get_session() as sess:
        tenants = sess.exec(select(Tenant)).all()
        units = sess.exec(select(Unit)).all()
        owners = sess.exec(select(Owner)).all()

    col1, col2 = st.columns(2)
    with col1:
        tenant = st.selectbox("Mieter", tenants, format_func=lambda t: f"{t.first_name} {t.last_name}")
        unit = st.selectbox("Einheit", units, format_func=lambda u: f"{u.unit_no}")
        owner = st.selectbox("Absender (Eigentümer)", owners, format_func=lambda o: o.name)
        period = st.text_input("Zeitraum", "2025-09")
        amount = st.number_input("Offener Betrag (€)", value=900.0, step=10.0)
        due = st.date_input("Fällig bis", value=date.today())
        if st.button("📄 Entwurf generieren"):
            text = render_letter(
                TEMPLATE_REMINDER,
                unit_no=unit.unit_no if unit else "",
                street=unit.property.street if unit else "",
                zip=unit.property.zip_code if unit else "",
                city=unit.property.city if unit else "",
                tenant_name=f"{tenant.first_name} {tenant.last_name}" if tenant else "",
                period=period,
                amount=f"{amount:.2f}",
                due_date=due.strftime("%d.%m.%Y"),
                owner_name=owner.name if owner else "",
            )
            st.session_state["doc_raw"] = text

    with col2:
        raw = st.session_state.get("doc_raw", "")
        st.text_area("Entwurf", value=raw, height=260)
        if st.button("✨ Mit LLM höflich formulieren (optional)"):
            refined = refine_with_llm(raw)
            st.session_state["doc_refined"] = refined
        st.text_area("Final", value=st.session_state.get("doc_refined", raw), height=260)

# ---------------- Einstellungen ----------------
elif page == "Einstellungen":
    st.write("### Einstellungen")
    st.write(f"DB: `{os.getenv('DB_PATH', './data/hausverwaltung.db')}`")
    st.write(f"OPENAI_API_KEY gesetzt: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    st.caption("API-Keys in `.env` setzen. DB-Pfad über `DB_PATH` steuerbar.")


# ---------------- Daten & Training ----------------
elif page == "Daten & Training":
    st.write("### Datensatz laden & Modell trainieren")

    st.markdown("Upload **ZIP** (mit CSV drin) **oder** direkt eine CSV des Kaggle-Datensatzes.")
    up_zip = st.file_uploader("ZIP hochladen", type=["zip"])
    up_csv = st.file_uploader("CSV hochladen", type=["csv"])

    from hausverwaltung.services.ingest import ingest_zip_bytes, ingest_csv_bytes
    from hausverwaltung.ml.pricing import train_pricing_model

    if st.button("➡️ Daten importieren"):
        if not up_zip and not up_csv:
            st.error("Bitte ZIP oder CSV auswählen.")
        else:
            with get_session() as sess:
                try:
                    if up_zip:
                        n = ingest_zip_bytes(sess, up_zip.read(), source="kaggle")
                    else:
                        n = ingest_csv_bytes(sess, up_csv.read(), source="kaggle")
                    st.success(f"{n} Inserate importiert.")
                except Exception as e:
                    st.exception(e)

    st.divider()
    if st.button("🧠 Modell trainieren"):
        with get_session() as sess:
            try:
                mae, r2, n = train_pricing_model(sess)
                st.success(f"Training ok: n={n}, MAE={mae:.2f} €, R²={r2:.3f}")
                st.caption("Das Modell wurde als `data/pricing_model.pkl` gespeichert.")
            except Exception as e:
                st.exception(e)

elif page == "📖 Projektbeschreibung":
    from gui.project_description import description
    st.markdown(description)

# ---------- Preis-Sandbox -------------------------------------------------
st.markdown("## 📈 Preis-Sandbox – Modell testen & (neu) trainieren")

# Info: Modellstatus oben anzeigen (pipe wird weiter unten verwendet)
if "pipe_loaded" not in st.session_state:
    st.session_state["pipe_loaded"] = False

# ---------------- Standort & Zensus-Defaults ----------------
with st.container():
    st.subheader("Eingaben")

    col_loc1, col_loc2 = st.columns([2, 1])
    with col_loc1:
        city_input = st.text_input(
            "Stadt / Gemeinde (optional)",
            placeholder="z. B. Leipzig, Aachen, Gera …",
        )

    # Zensus-Periode (gleich den Spalten in 0005_clean1)
    periode = st.selectbox(
        "Baujahr-Periode (Zensus-0005)",
        ["vor_1919", "1919_1949", "1950_1959", "1970_1979",
         "1980_1989", "2000_2009", "2010_2015", "2016_plus"],
        index=7,
    )

    # Zensus-Defaults über Namen ermitteln (falls möglich)
    defaults = {}
    if city_input.strip():
        try:
            defaults = resolve_location_defaults(city_input, periode=periode)
        except Exception:
            defaults = {}

# ---------------- Basis-Features ----------------
col_l, col_r = st.columns(2)

with col_l:
    area_sqm   = st.slider("Wohnfläche (m²)", 10.0, 250.0, 67.0, 1.0)
    rooms      = st.slider("Zimmer", 1.0, 8.0, 2.0, 0.5)
    floor      = st.slider("Etage",  -1, 50, 1, 1)
    pricetrend = st.slider("pricetrend", 0.0, 10.0, 3.0, 0.01)
    serviceCharge = st.slider("Nebenkosten (ServiceCharge, €)", 0.0, 800.0, 180.0, 1.0)

with col_r:
    garden     = st.checkbox("Garten", True)
    lift       = st.checkbox("Aufzug", False)
    hasKitchen = st.checkbox("Einbauküche", True)

    typeOfFlat = st.selectbox(
        "Wohnungstyp",
        ["apartment", "ground_floor", "loft", "penthouse", "roof_storey",
         "terraced_flat", "maisonette", "other"],
        index=0,
    )

    heatingType_clean = st.selectbox(
        "Heizungsart",
        ["central_heating", "district_heating", "gas_heating", "oil_heating",
         "heat_pump", "floor_heating", "electric_heating", "other"],
        index=0,
    )

# ---------------- Zensus (optional; mit Defaults aus Standort) ----------------
z_total_default  = float(defaults.get("zensus_total") or 0.0)
z_dec_default    = float(defaults.get("zensus_decade") or 0.0)
z_factor_default = float(defaults.get("zensus_factor") or 0.0)

zensus_miete_total  = st.number_input("Zensus: Gesamt €/m² (optional)", value=z_total_default, min_value=0.0, step=0.01)
zensus_miete_decade = st.number_input("Zensus: Dekade €/m² (optional)", value=z_dec_default,    min_value=0.0, step=0.01)
zensus_factor_decade = st.number_input("Zensus: Faktor (Dekade/gesamt) (optional)", value=z_factor_default, min_value=0.0, step=0.01)

# ---------------- Modell laden ----------------
import joblib
MODEL_PATH = "./data/pricing_model.pkl"

pipe = None
try:
    pipe = joblib.load(MODEL_PATH)
    if not st.session_state["pipe_loaded"]:
        st.session_state["pipe_loaded"] = True
        st.success("Modell geladen.")
except Exception as e:
    st.error("Kein Modell gefunden oder Laden fehlgeschlagen.")
    st.caption("Trainiere zuerst unter „Daten & Training“ oder lege eine Datei `data/pricing_model.pkl` ab.")
    st.exception(e)

# ---------------- Vorhersage ----------------
st.markdown("### 🔮 Vorhersage")
if pipe is not None:
    try:
        # Eingabezeile so benannt, wie beim Training erwartet
        X_user = {
            "area_sqm": area_sqm,
            "rooms": rooms,
            "floor": float(floor),
            "pricetrend": pricetrend,
            "serviceCharge": serviceCharge,
            "garden": int(bool(garden)),
            "lift": int(bool(lift)),
            "hasKitchen": int(bool(hasKitchen)),
            "typeOfFlat": typeOfFlat,
            "heatingType_clean": heatingType_clean,
            "periode_0005": periode,
            "zensus_miete_total": zensus_miete_total if zensus_miete_total > 0 else None,
            "zensus_miete_decade": zensus_miete_decade if zensus_miete_decade > 0 else None,
            "zensus_factor_decade": zensus_factor_decade if zensus_factor_decade > 0 else None,
        }
        import pandas as pd
        X_user = pd.DataFrame([X_user])

        y_hat = float(pipe.predict(X_user)[0])
        st.success(f"**Prognose Nettokaltmiete:** {y_hat:,.2f} €")
        st.caption(f"≈ {y_hat/area_sqm:.2f} €/m²")
    except Exception as e:
        st.error("Vorhersage fehlgeschlagen. Prüfe, ob alle benötigten Feature-Spalten vorhanden sind.")
        st.exception(e)

    # ---------------- Feature-Importances (nur falls Modell das kann) ----------------
    with st.expander("Feature-Importances", expanded=False):
        try:
            rf   = pipe.named_steps.get("rf", None)      # Regressor
            prep = pipe.named_steps.get("prep", None)    # ColumnTransformer

            if rf is not None and hasattr(rf, "feature_importances_"):
                # Feature-Namen aus dem Preprocessor holen (falls unterstützt)
                if prep is not None and hasattr(prep, "get_feature_names_out"):
                    names = prep.get_feature_names_out()
                else:
                    names = [f"f{i}" for i in range(len(rf.feature_importances_))]

                import pandas as pd
                fi = (pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
                        .sort_values("importance", ascending=False)
                        .head(25))
                st.dataframe(fi, use_container_width=True)
            else:
                st.info("Dieses Modell stellt keine Feature-Importances bereit.")
        except Exception as e:
            st.caption("Feature-Importances konnten nicht ermittelt werden.")
            st.exception(e)



