import io, json, zipfile
import pandas as pd
from typing import Optional
from sqlmodel import Session
from ..models import ListingRaw

# Spaltenmapping für das Kaggle-Set (robust gegen leichte Varianten)
COLUMN_MAP_CANDIDATES = {
    "city": ["city", "Stadt"],
    "zip_code": ["zip", "postal_code", "plz"],
    "district": ["district", "Ortsteil", "neighbourhood"],
    "lat": ["lat", "latitude", "Latitude"],
    "lon": ["lng", "lon", "longitude", "Longitude"],
    "area_sqm": ["livingSpace", "area", "squareMeters", "size", "total_area_sqm"],
    "rooms": ["rooms", "numRooms", "NumRooms"],
    "floor": ["floor", "Floor"],
    "year_built": ["yearConstructed", "year_built", "construction_year"],
    "condition": ["condition", "state"],
    "rent_cold": ["price", "cold_rent", "rent", "Nettokaltmiete"],
    "rent_warm": ["warm_rent", "Warmmiete"],
    "created_at": ["date", "created_at", "inserted_at"],
}

def _pick(series_like, names):
    for n in names:
        if n in series_like:
            return n
    return None

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {key: _pick(df.columns, cand) for key, cand in COLUMN_MAP_CANDIDATES.items()}
    out = pd.DataFrame()
    for k, src in cols.items():
        out[k] = df[src] if src in df.columns else None
    # Cleanup/Typen
    for num in ["lat","lon","area_sqm","rooms","floor","rent_cold","rent_warm"]:
        if num in out:
            out[num] = pd.to_numeric(out[num], errors="coerce")
    if "year_built" in out:
        out["year_built"] = pd.to_numeric(out["year_built"], errors="coerce").astype("Int64")
    return out, cols

def ingest_csv_bytes(session: Session, content: bytes, source: str = "kaggle", max_rows: Optional[int]=None) -> int:
    df = pd.read_csv(io.BytesIO(content))
    if max_rows:
        df = df.head(max_rows)
    std, used_map = _standardize_columns(df)
    # alles, was nicht gemappt wurde, in extra_json ablegen
    used_cols = set(c for c in used_map.values() if c)
    leftovers = df.drop(columns=[c for c in used_cols if c in df.columns], errors="ignore")
    std["extra_json"] = leftovers.apply(lambda r: json.dumps({k: (None if pd.isna(v) else v) for k,v in r.items()}, ensure_ascii=False), axis=1)

    to_add = []
    for rec in std.to_dict(orient="records"):
        to_add.append(ListingRaw(**{**rec, "source": source}))
    session.add_all(to_add)
    session.commit()
    return len(to_add)

def ingest_zip_bytes(session: Session, content: bytes, source: str = "kaggle", max_rows: Optional[int]=None) -> int:
    z = zipfile.ZipFile(io.BytesIO(content))
    # nimm die erste CSV im Archiv
    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        return 0
    with z.open(csv_names[0]) as f:
        return ingest_csv_bytes(session, f.read(), source=source, max_rows=max_rows)
