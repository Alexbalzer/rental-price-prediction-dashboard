
# Rental Price Prediction Dashboard  

A Streamlit application for analyzing apartment rental offers in Germany and predicting rental prices using machine learning models.  

## 🚀 Project Overview  
This project was developed as part of a Data Science capstone.  
It combines:  
- **Data Ingestion**: Import rental offer data (e.g., Kaggle dataset, CSV/ZIP, APIs) into a SQLite database.  
- **Machine Learning**: Train a regression model (Random Forest) to predict cold rent based on features such as size, rooms, year built, condition, and location.  
- **Interactive Dashboard**: Visualize data, explore features, and get rental price recommendations for new units.  

## ✨ Features  
- Upload rental datasets (CSV or ZIP).  
- Automatic cleaning and standardization of data.  
- Train and evaluate ML models (MAE, R² metrics).  
- Interactive dashboard (Streamlit) with filtering and KPIs.  
- Rent recommendation for new units (ML-based, fallback heuristic).  
- Fully integrated with SQLite (sqlmodel).  

## 📊 Tech Stack  
- **Python** (pandas, scikit-learn, sqlmodel)  
- **Streamlit** (interactive dashboard)  
- **SQLite** (data storage)  
- **Joblib** (model persistence)  

## 📂 Project Structure  
```
rental-price-prediction-dashboard/
│
├── hausverwaltung/          # Main package
│   ├── db.py                # DB initialization
│   ├── models.py            # SQLModel models
│   ├── services/ingest.py   # Data ingestion (CSV/ZIP)
│   └── ml/pricing.py        # Training + Prediction
│
├── app.py                   # Streamlit entrypoint
├── requirements.txt         # Dependencies
├── .env.example             # Env config (DB path, model path)
└── README.md                # Project description
```

## 📈 Example Use Cases  
- Estimate a fair rental price for new apartment listings.  
- Explore rental market trends across German cities.  
- Compare features (size, rooms, condition) and their influence on rent.  

## 🏗️ How to Run Locally  
```bash
# clone repo
git clone https://github.com:Alexbalzer/rental-price-prediction-dashboard.git
cd rental-price-prediction-dashboard

# create virtual env & install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

## 📑 Data Source  
Dataset: *Apartment rental offers in Germany* (Kaggle)  
👉 [https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany)  

flowchart LR
  %% Dateien als Cluster
  subgraph APP[src/gui/app.py]
    A1[render_dashboard()  ← import]
    A2[render_stammdaten() ← import]
    A3[render_price_sandbox() ← import]
    A4[Docs/DB: render_letter(), refine_with_llm(), get_session()]
  end

  subgraph BOARD[src/gui/board.py]
    B0[render_dashboard()]
    B1[_load_sources()]
    B2[_tile_counts()]
    B3[_tile_immo_vs_zensus()]
    B4[_tile_features()]
    B5[_tile_heatmap()]
  end

  subgraph STAMM[src/gui/stammdaten.py]
    S1[render_stammdaten()]
  end

  subgraph SANDBOX[src/gui/sandbox.py]
    X0[render_price_sandbox()]
    X1[_load_model_cached()]
    X2[_load_metrics()]
    X3[_defaults_from_city_or_plz()]
  end

  subgraph HELP[src/gui/helpers.py]
    H1[load_immo_data()]
    H2[load_zensus0005()]
    H3[load_ags_map()]
    H4[resolve_location_defaults()]
    H5[_norm()]
  end

  subgraph DESC[src/gui/project_description.py]
    D1[description]
  end

  %% Import-Pfeile (wo wird was benutzt)
  B0 -->|wird importiert in| A1
  S1 -->|wird importiert in| A2
  X0 -->|wird importiert in| A3

  H1 -->|genutzt in| B1
  H2 -->|genutzt in| B1
  H2 -->|genutzt in| X3
  H3 -->|genutzt in| X3
  H4 -->|genutzt in| X3

  B2 -->|UI-Kachel| B0
  B3 -->|UI-Kachel| B0
  B4 -->|UI-Kachel| B0
  B5 -->|UI-Kachel| B0

  D1 -->|Markdown wird angezeigt in| A1
