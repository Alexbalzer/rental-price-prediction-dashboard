# gui/project_description.py

description = """
# 📖 Projektbeschreibung

## Datenbeschaffung & Vorbereitung
- Ausgangspunkt war **Zensus-Datensatz 0004**, der die ersten Gebäudedaten enthielt.  
- Bald stellte sich heraus, dass **0004 redundant** zur Spalte *Insgesamt* von **0005** war → daher entfernt.  
- Anschließend kam **0005** (Baujahre) als Kernbasis, ergänzt um weitere Datensätze.  
- Dabei stellte sich heraus, dass **einige Inserate fehlerhafte Angaben** hatten (z. B. Flächen 0 oder unrealistisch hohe Mieten) → wurden bereinigt.  
- Ein besonderes Thema war das **Mapping von Standorten**: GKZ, AGS, PLZ, Bundesland, Gemeindenamen.  
  - Problem: Schreibvarianten wie *St.*, *Bad*, *kreisfreie Stadt*, Ortsteile.  
  - Lösung: Normalisierung und Mapping auf die Spalte **ARS** (eindeutiger Identifier).  

## Datenbereinigung
- Nicht plausible Werte entfernt.  
- Nur Kerneigenschaften behalten: `area_sqm`, `rooms`, `floor`, `heatingType`, etc.  
- Nach der Bereinigung reduzierte sich die Datenmenge deutlich, **Trainingszeit ging von 15 Minuten auf wenige Sekunden**.  

## Machine Learning
- Modell: **RandomForestRegressor** (scikit-learn).  
- Warum RandomForest?
  - robust bei gemischten Features (kategorisch + numerisch)
  - gute Performance ohne viel Hyperparameter-Tuning
  - liefert Feature Importances (Interpretierbarkeit)  
- Features: Größe, Zimmer, Stockwerk, Baujahr-Periode (aus Zensus), Ausstattung (Lift, Küche, Garten), Region (ARS).  
- Output: **geschätzte Nettokaltmiete (€)**.  
- Ergebnisse:
  - MAE ≈ 54 €  
  - MdAE ≈ 15 €  
  - R² ≈ 0.84  

## App (Streamlit)
- Stammdaten zu **Eigentümern, Objekten, Mietern** werden erfasst.  
- Speicherung erfolgt in einer **SQLite-Datenbank**.  
- Verträge und Zahlungen können dokumentiert werden.  
- **Mietpreis-Vorschlag** nutzt das trainierte Modell.  
- Zusätzlich gibt es eine Funktion für **Dokumentenerstellung (Mahnschreiben)** mit optionaler LLM-Verbesserung.  

## Herausforderungen
- Komplexes Mapping der Standorte.  
- Konsistenz zwischen Zensus-Daten und Immobilien-Inseraten.  
- Datenqualität (falsche Inserate).  
- Sicherstellen, dass das Training auch reproduzierbar läuft.  

## Fazit
- Projekt bildet einen **End-to-End-Data-Science-Prozess** ab: von Datenbeschaffung → Bereinigung → Feature Engineering → Modelltraining → Deployment (Streamlit-App).  
- Nächste Schritte: weitere ML-Modelle testen (XGBoost, LightGBM), mehr Interaktivität im Dashboard, Integration zusätzlicher Datenquellen (z. B. Energieverbrauch, Demografie).  
"""
