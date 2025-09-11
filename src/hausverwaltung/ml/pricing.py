from __future__ import annotations
import os, joblib, numpy as np
from typing import Optional, Tuple
from sqlmodel import Session, select
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from ..models import Unit, Property, ListingRaw

MODEL_PATH = os.getenv("PRICING_MODEL_PATH", "./data/pricing_model.pkl")

def _build_dataset(session: Session):
    rows = session.exec(select(ListingRaw).where(ListingRaw.rent_cold != None)).all()  # noqa: E711
    if not rows:
        return None, None, None
    import pandas as pd
    df = pd.DataFrame([{
        "city": r.city,
        "zip_code": r.zip_code,
        "district": r.district,
        "lat": r.lat,
        "lon": r.lon,
        "area_sqm": r.area_sqm,
        "rooms": r.rooms,
        "floor": r.floor,
        "year_built": r.year_built if r.year_built is not None else np.nan,
        "condition": (r.condition or "").lower() if r.condition else None,
        "rent_cold": r.rent_cold,
    } for r in rows])

    # einfache Filter: plausibles Mieten-/Flächenfenster
    df = df[(df["area_sqm"] > 10) & (df["area_sqm"] < 400) & (df["rent_cold"] > 200) & (df["rent_cold"] < 10000)]
    y = df["rent_cold"].values
    X = df.drop(columns=["rent_cold"])

    num_features = ["area_sqm","rooms","floor","year_built","lat","lon"]
    cat_features = ["city","zip_code","district","condition"]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")),
                                ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
    ])
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", pre), ("rf", model)])
    return X, y, pipe

def train_pricing_model(session: Session) -> Tuple[float,float,int]:
    data = _build_dataset(session)
    if data is None or data[0] is None:
        raise RuntimeError("Keine Trainingsdaten gefunden (listings_raw leer).")
    X, y, pipe = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    r2 = r2_score(yte, preds)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return mae, r2, len(y)

def _predict_with_model(city: str, zip_code: str, district: str, lat: float, lon: float,
                        area_sqm: float, rooms: float, floor: float, year_built: Optional[int],
                        condition: Optional[str]) -> Optional[float]:
    if not os.path.exists(MODEL_PATH):
        return None
    pipe = joblib.load(MODEL_PATH)
    import pandas as pd
    X = pd.DataFrame([{
        "city": city, "zip_code": zip_code, "district": district,
        "lat": lat, "lon": lon, "area_sqm": area_sqm, "rooms": rooms,
        "floor": floor, "year_built": year_built, "condition": (condition or "").lower()
    }])
    pred = pipe.predict(X)[0]
    return float(pred)

# --- Dein vorhandener Fallback bleibt bestehen ---
from ..models import Property, Lease

def suggest_rent_for_unit(session: Session, unit_id: int) -> Optional[float]:
    unit = session.get(Unit, unit_id)
    if not unit:
        return None
    prop = session.get(Property, unit.property_id)

    # 1) Versuche ML-Modell
    try:
        ml_pred = _predict_with_model(
            city=(prop.city if prop else None),
            zip_code=(prop.zip_code if prop else None),
            district=None,
            lat=None, lon=None,
            area_sqm=(unit.size_sqm or 60.0),
            rooms=(unit.rooms or 2.0),
            floor=(unit.floor or 0),
            year_built=(prop.year_built if prop else None),
            condition=(unit.condition or None),
        )
        if ml_pred is not None and ml_pred > 0:
            return round(ml_pred, 2)
    except Exception:
        pass

    # 2) Fallback: heuristik (wie bisher)
    city_baseline = {
        "Berlin": 13.0, "München": 18.0, "Hamburg": 14.0, "Köln": 13.5, "Frankfurt": 15.0
    }
    base = city_baseline.get((prop.city if prop else "") or "", 11.0)
    size_factor = (unit.size_sqm or 60) ** 0.15
    condition_bonus = 1.0
    if (unit.condition or "").lower() in {"neu", "saniert", "renoviert"}:
        condition_bonus = 1.08
    floor_adj = 1.0 + (0.01 * (unit.floor or 0))
    price_sqm = base * size_factor * condition_bonus * floor_adj
    return round(price_sqm * (unit.size_sqm or 60), 2)
