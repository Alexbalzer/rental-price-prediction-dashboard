# src/gui/project_guide.py
guide = r"""
# 🧭 Was ist wo – Funktionen & Verwendung

Diese Seite fasst die wichtigsten Module/Funktionen zusammen und zeigt, wer wen aufruft.

---

## App-Shell

### `src/gui/app.py`
- **Navigiert** zwischen den Seiten (Radio in der Sidebar).
- Ruft die Renderer-Funktionen der Unterseiten auf.

---

## Dashboard

### `src/gui/board.py`
**Öffentliche API**
- `render_dashboard()` → Orchestriert alle Dashboard-Kacheln; von `app.py` aufgerufen.

**Interne Helfer (nur in board.py)**
- `_err_box(msg)` → hübsche Fehlerbox.
- `_load_sources()` → lädt Daten via `helpers.load_immo_data()` und `helpers.load_zensus0005()`.
- `_tile_counts(immo)` → Top-10 Inserate nach Bundesland/Stadt (Balken).
- `_tile_immo_vs_zensus(immo, z5)` → vergleicht Median €/m² aus Inseraten mit Zensus-„Insgesamt“ je Stadt (Top-20).
- `_tile_features(immo)` → Anteile von Balkon/Küche/Lift/Garten (gesamt + Top-PLZ).
- `_tile_heatmap(immo)` → Choropleth nach Bundesland (Fallback auf Balken, wenn Geo-JSON/Plotly fehlt).

### `src/gui/helpers.py` (Loader & Utilities fürs Dashboard)
- `load_immo_data(path=None, candidates=...)` → lädt Inserats-CSV robust; vereinheitlicht Spalten (`state/city/plz/area_sqm/...`, Flags).
- `load_zensus0005(path=None)` → lädt Zensus-0005 robust; erkennt GKZ/ARS/AGS, normalisiert Zahlen, erzeugt `name_norm`.
- `_norm(s)` → Namensnormalisierung (lower, Sonderzeichen raus, Zusätze wie „, Stadt“/„Hansestadt“ entfernen).

*Derzeit ungenutzt (kann bleiben):*
- `load_ags_map(...)`, `resolve_location_defaults(...)` → nützlich für Standort-Autofill, aber vom Dashboard nicht genutzt.
- Auskommentierte Vorlagen: `load_dashboard_frames(...)`, `load_immo_csv(...)`, `attach_canonical_regionals(...)`, `attach_zensus_expectation(...)`.

> **Hinweis:** In `app.py` sollten ungenutzte Importe entfernt werden:
> `from gui.helpers import load_zensus0005, load_ags_map, resolve_location_defaults, load_dashboard_frames`
> – diese werden dort nicht verwendet.

---

## Stammdaten

### `src/gui/stammdaten.py`
- `render_stammdaten()` → rendert die vier Tabs (Eigentümer, Objekte, Einheiten, Mieter) inkl. Formulare/Tabellen.
- Wird in `app.py` aufgerufen.
- Persistenz via **SQLModel/SQLite** (`get_session()`, `select(...)`).

---

## Preis-Sandbox

### `src/gui/sandbox.py`
**Öffentliche API**
- `render_price_sandbox()` → interaktive Vorhersage:
  - lädt Modell (`pricing_model.pkl`) und Metriken (`models/metrics.json`),
  - Standort-Eingabe (PLZ/Stadt) → **Zensus-Autofill** (`zensus_miete_total`, `zensus_miete_decade`, `zensus_factor_decade`),
  - Eingaben (Wohnfläche, Zimmer, etc.),
  - Prediction + (falls RF) Feature-Importances.

**Interne Helfer**
- `_load_model_cached`, `_load_metrics` → Caching/Lesen.
- `_load_zensus0005`, `_load_ags_map` → Sandbox-eigene Loader (separat von helpers.py).
- `_norm_city`, `_defaults_from_city_or_plz` → Standort-Autofill-Logik.
- `_section_header`, `_number` → kleine UI-Hilfen.

---

## Dokumente & Training

- In `app.py`:
  - **Verträge & Zahlungen** → DB-Zugriffe via `get_session()`/`select(...)`.
  - **Dokumente** → `render_letter(...)` und optional `refine_with_llm(...)` für höfliche Formulierungen.
  - **Daten & Training** → CSV/ZIP importieren und ML-Training starten; Metriken werden gespeichert.

---

## Datenfluss (kurz)

1. **Kaggle immo_data.csv** → Bereinigung → `immo_train_ready.csv`.
2. **ARS/GKZ-Mapping** (PLZ/Name) → ~80 % Match.
3. **Zensus 0005** (Dekaden) → Join auf **ARS** → `immo_train_joined.csv`, inkl. `zensus_*`.
4. **Training** (`scripts/train_pricing.py`) → `pricing_model.pkl` + `models/metrics.json`.
5. **App/Sandbox** → Vorhersage + Zensus-Autofill.

---
"""
