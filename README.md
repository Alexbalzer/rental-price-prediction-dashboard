
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
