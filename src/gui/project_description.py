# src/gui/project_description.py
description = r"""
# Projektbeschreibung

## Ziel
Vorhersage der **Nettokaltmiete** von Wohnungen auf Basis eines bereinigten
Kaggle-Datensatzes (*immo_data.csv*) und externer **Zensus**-Merkmale
(Statistische Ämter des Bundes und der Länder).

---

## Datenquellen

**1) Kaggle: immo_data.csv**  
Ausgangsbasis (~268k Inserate). Wichtigste Aufbereitung:

- Spalten vereinheitlicht/umbenannt:
  - `baseRent` → **rent_cold**
  - `livingSpace` → **area_sqm**
  - `noRooms` → **rooms**
- Booleans zu 0/1: **garden**, **lift**, **hasKitchen**.
- Numerik erzwungen: **rooms**, **area_sqm**, **floor**, **pricetrend**, **serviceCharge** …
- Fehlwerte:
  - `serviceCharge` fehlend → **0**
  - `totalRent` (falls leer) = `baseRent + serviceCharge`
- Stark fehlende/irrelevante Spalten entfernt (u. a. `telekom*`, `electricity*`,
  `petsAllowed`, `interiorQual`, `thermalChar`, `houseNumber`, `streetPlain`, `noParkSpaces`, `condition`).
- Feature **periode_0005** aus Baujahr abgeleitet (Dekaden-Buckets passend zur Zensus-Tabelle).

> Ergebnis der Bereinigung: Trainingsfähiges Set, sehr schnelle Trainingsläufe.

**2) Zensus – zuerst 0004 (gesamt), dann 0005/0008**  
- *Start:* Zensus-**0004** („Durchschnittsmiete insgesamt“) als einfacher Lage-Anker.  
- *Erkenntnis:* **0005** (Mieten nach **Baujahr-Dekaden**) und **0008** (Mieten nach **Wohnungsgröße**) liefern **reichere Features** → 0004 wurde **überflüssig**.  
- Wir nutzen aktuell **0005** produktiv und berechnen je Gemeinde **Faktoren**:
  - `factor_<Dekade> = <Dekadenmiete> / Insgesamt`
- **0008** (Größen-Intervalle) ist vorbereitet und kann später zusätzlich gejoint werden.

**3) Standort-Mapping (GKZ/ARS)**  
- Ziel: Jeder Immo-Zeile einen **ARS/GKZ** (12-stellig) zuordnen.  
- Quellen:
  - `../data/clean/zensus_0005_clean1.csv` mit **ARS (GKZ)**, *Gemeindename*, *Insgesamt*, Dekaden …
  - `../data/ags_gkz.csv` (PLZ↔AGS/ARS/Gemeindename/Ort; Semikolon-CSV)
- Vorgehen:
  1. **PLZ** → eindeutige **ARS** (falls genau eine Gemeinde je PLZ).
  2. **Gemeindename** (normalisiert; „, Stadt“, „Hansestadt“, Bindestriche etc. entfernt) → eindeutige **ARS**.
- Entwicklung der Match-Quote:
  - anfänglich ~40 %  
  - mit PLZ-Mapping + eindeutigen Namens-Treffern ~**80 %** (~211k Zeilen mit ARS)  
- Für das Training behalten wir **nur Zeilen mit ARS**.

---

## Feature-Engineering & Join

- **Train-Basis (`immo_train_ready.csv`)**  
  `['ARS','rent_cold','area_sqm','rooms','floor','pricetrend','serviceCharge',
    'garden','lift','hasKitchen','typeOfFlat','heatingType_clean','periode_0005']`
- **Join mit 0005** auf **ARS** → `immo_train_joined.csv` mit Zensus-Features:
  - `zensus_miete_total` (Insgesamt €/m²),
  - `zensus_miete_decade` (passend zu `periode_0005`),
  - `zensus_factor_decade` (= Dekade/Insgesamt).

---

## Modell

- **RandomForestRegressor** (sklearn) mit Pipeline:
  - Numerik: Median-Imputation, Standardisierung (wo sinnvoll).
  - Kategorisch: One-Hot-Encoding (`typeOfFlat`, `heatingType_clean`, `periode_0005`).
- **Train/Test-Split** und Metrik-Logging (`models/metrics.json`).

**Letzter Stand (Beispiellauf):**  
- **MAE:** ~**54 €**  
- **MdAE:** ~15 €  
- **MAPE/sMAPE:** ~9.6 % / 9.3 %  
- **R²:** ~**0.84**  
- Trainingszeit: Sekunden (nach Bereinigung).

---

## App (Streamlit + SQLite)

- **Stammdaten:** Eigentümer, Objekte, Einheiten, Mieter, Verträge, Zahlungen
  (persistiert via **SQLModel/SQLite**).
- **Preis-Sandbox:** Interaktive Vorhersage mit Schiebereglern.  
  Eingabe von **PLZ/Stadt** füllt Zensus-Werte automatisch (aus 0005) und
  erlaubt manuelle Überschreibung.
- **Training:** CSV importieren & Modell trainieren – Metriken werden gespeichert.
- **Dokumente:** Zahlungserinnerung als Text-Entwurf; optionales „Höflich-Formulieren“.

---

## Gründe für RandomForest
- robuste Baseline, wenig Hyperparameter-Tuning, nicht empfindlich gegen Ausreißer,
  funktioniert gut mit Mix aus **numerischen** und **kategorischen** Features.
- sehr schnelle Trainingsläufe → ideal für iteratives Feature-Engineering.

> Nächste Schritte (Roadmap):
> - **0008** (Größen-Bänder) zusätzlich zum Join nutzen.
> - Alternative Modelle (XGBoost/LightGBM) testen.
> - Zeitliche Kalibrierung nach Marktjahr/Inflation.
> - Standort-Matching weiter verbessern (ambige Namen, Ortsteile).

---

## Reproduzierbarkeit

- Notebooks: `immo_eda.ipynb`, `zensus_eda.ipynb`  
- Datenaufbau: `immo_train_ready.csv` → Join mit 0005 → `immo_train_joined.csv`  
- Training: `scripts/train_pricing.py`  
- App: `src/gui/app.py`, Sandbox: `src/gui/sandbox.py`

"""
