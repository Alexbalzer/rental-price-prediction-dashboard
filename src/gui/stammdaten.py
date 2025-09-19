# src/gui/stammdaten.py
from __future__ import annotations

import streamlit as st
from sqlmodel import select

from hausverwaltung.db import get_session
from hausverwaltung.models import Owner, Property, Unit, Tenant


def render_stammdaten() -> None:
    """Zeigt die vier Stammdaten-Tabs (Eigentümer, Objekte, Einheiten, Mieter)."""
    tab1, tab2, tab3, tab4 = st.tabs(["Eigentümer", "Objekte", "Einheiten", "Mieter"])

    # ---------- Eigentümer ----------
    with tab1:
        st.write("### Eigentümer")
        with get_session() as sess:
            with st.form("owner_form"):
                name = st.text_input("Name")
                email = st.text_input("E-Mail")
                phone = st.text_input("Telefon")
                create = st.form_submit_button("➕ Anlegen")
                if create:
                    if name.strip():
                        sess.add(Owner(name=name.strip(), email=email or None, phone=phone or None))
                        sess.commit()
                        st.success("Eigentümer gespeichert.")
                    else:
                        st.error("Name ist Pflicht.")

            owners = sess.exec(select(Owner)).all()
            st.dataframe(
                [{"ID": o.id, "Name": o.name, "E-Mail": o.email, "Telefon": o.phone} for o in owners],
                use_container_width=True,
            )

    # ---------- Objekte ----------
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
                create = st.form_submit_button("➕ Anlegen")
                if create:
                    if owner and street and zip_code and city:
                        sess.add(
                            Property(
                                owner_id=owner.id,
                                street=street,
                                zip_code=zip_code,
                                city=city,
                                year_built=int(year_built),
                            )
                        )
                        sess.commit()
                        st.success("Objekt gespeichert.")
                    else:
                        st.error("Bitte Eigentümer und Adresse angeben.")

            props = sess.exec(select(Property)).all()
            st.dataframe(
                [
                    {
                        "ID": p.id,
                        "Eigentümer": p.owner_id,
                        "Adresse": f"{p.street}, {p.zip_code} {p.city}",
                        "Baujahr": p.year_built,
                    }
                    for p in props
                ],
                use_container_width=True,
            )

    # ---------- Einheiten ----------
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
                create = st.form_submit_button("➕ Anlegen")
                if create:
                    if prop and unit_no:
                        sess.add(
                            Unit(
                                property_id=prop.id,
                                unit_no=unit_no,
                                floor=int(floor),
                                rooms=float(rooms),
                                size_sqm=float(size),
                                condition=condition or None,
                            )
                        )
                        sess.commit()
                        st.success("Einheit gespeichert.")
                    else:
                        st.error("Bitte Objekt und Wohnungs-Nr. angeben.")

            units = sess.exec(select(Unit)).all()
            st.dataframe(
                [
                    {
                        "ID": u.id,
                        "Objekt": u.property_id,
                        "Whg": u.unit_no,
                        "Etage": u.floor,
                        "Zimmer": u.rooms,
                        "m²": u.size_sqm,
                        "Zustand": u.condition,
                    }
                    for u in units
                ],
                use_container_width=True,
            )

    # ---------- Mieter ----------
    with tab4:
        st.write("### Mieter")
        with get_session() as sess:
            with st.form("tenant_form"):
                fn = st.text_input("Vorname")
                ln = st.text_input("Nachname")
                email = st.text_input("E-Mail")
                phone = st.text_input("Telefon")
                create = st.form_submit_button("➕ Anlegen")
                if create:
                    if fn.strip() and ln.strip():
                        sess.add(
                            Tenant(
                                first_name=fn.strip(),
                                last_name=ln.strip(),
                                email=email or None,
                                phone=phone or None,
                            )
                        )
                        sess.commit()
                        st.success("Mieter gespeichert.")
                    else:
                        st.error("Vor- und Nachname sind Pflicht.")

            tenants = sess.exec(select(Tenant)).all()
            st.dataframe(
                [
                    {
                        "ID": t.id,
                        "Name": f"{t.first_name} {t.last_name}",
                        "E-Mail": t.email,
                        "Telefon": t.phone,
                    }
                    for t in tenants
                ],
                use_container_width=True,
            )
