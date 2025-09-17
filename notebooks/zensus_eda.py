# notebooks/zensus_eda.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from pathlib import Path

# -------------------------------------------
# Einstellungen / Pfade
# -------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DIR.mkdir(exist_ok=True, parents=True)

FILES = {
    "0004": DATA_DIR / "4000W-0004_de.csv",  # Durchschnittliche Nettokaltmiete
    "0005": DATA_DIR / "4000W-0005_de.csv",  # ... nach Baujahr
    "0006": DATA_DIR / "4000W-0006_de.csv",  # ... nach Eigentumsform
    "0008": DATA_DIR / "4000W-0008_de.csv",  # ... nach Wohnfläche (10 m²)
}

# -------------------------------------------
# Hilfsfunktionen
# -------------------------------------------
def _to_float(x):
    """Zensus-Stringzahlen → float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("€/m²", "").replace("€ / m²", "").replace("€", "")
    s = re.sub(r"\(.*?\)", "", s)      # ( ... )
    s = s.replace("e", "")             # Fußnotenmarker
    s = s.replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_csv_robust(path):
    """
    Robustes Einlesen:
    - erkennt Trennzeichen automatisch (sep=None, engine='python')
    - probiert mehrere Encodings
    - überspringt kaputte Zeilen
    - liest OHNE Header (header=None), alles als String
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=None,               # Trennzeichen automatisch erkennen
                engine="python",
                header=None,            # keine Kopfzeile – wir erraten sie später
                dtype=str,
                low_memory=False,
                on_bad_lines="skip",
                encoding=enc,
            )
            return df
        except Exception as e:
            last_err = e
    raise last_err


def load_raw(path):
    """Rohvorschau ohne Header-Erkennung."""
    raw = read_csv_robust(path)
    print(f"\n--- Rohvorschau: {path.name} (shape={raw.shape}) ---")
    print(raw.head(20).to_string(index=False))
    return raw


def find_start_row(df, ags_col=1):
    """Erste Datenzeile finden (wo eine lange Ziffernfolge wie AGS/GKZ steht)."""
    for i in range(len(df)):
        val = str(df.iloc[i, ags_col]) if ags_col < df.shape[1] else ""
        if re.fullmatch(r"\d{6,}", val):
            return i
    return None


def load_table_with_header_guess(path, ags_col=1):
    """
    Einlesen + Header erraten:
    - Nimmt die Zeile VOR der ersten Datenzeile als Header
    - Gibt (df, header_list, start_row_index) zurück
    """
    raw = read_csv_robust(path)
    start = find_start_row(raw, ags_col=ags_col)
    header_row = start - 1 if start and start > 0 else None

    if header_row is None:
        cols = [f"col_{i}" for i in range(raw.shape[1])]
        df = raw.copy()
        df.columns = cols
        df = df.iloc[start:] if start is not None else df
        return df.reset_index(drop=True), None, start

    header = raw.iloc[header_row].tolist()
    df = raw.iloc[start:].copy()
    df.columns = [f"{c}" if c not in (None, np.nan) else f"col_{i}" for i, c in enumerate(header)]
    return df.reset_index(drop=True), header, start


def melt_long(df, ags_col=1, name_col=2, value_cols=None, value_name="miete_qm", var_name="kategorie"):
    """Wide → Long (für 0005/0006/0008)."""
    if value_cols is None:
        value_cols = list(range(3, df.shape[1]))
    work = df.copy()
    work.columns = [f"col_{i}" if (c is None or str(c).strip()=="") else str(c) for i, c in enumerate(work.columns)]
    id_vars = [work.columns[ags_col], work.columns[name_col]]
    long = work.melt(
        id_vars=id_vars,
        value_vars=[work.columns[i] for i in value_cols],
        var_name=var_name,
        value_name=value_name,
    )
    long.rename(columns={work.columns[ags_col]: "ags", work.columns[name_col]: "gemeinde"}, inplace=True)
    long[value_name] = long[value_name].apply(_to_float)
    long = long[long[value_name].between(2, 30)]
    return long


# -------------------------------------------
# 1) Rohvorschau
# -------------------------------------------
for key, path in FILES.items():
    if not path.exists():
        print(f"FEHLT: {path}")
    else:
        print(f"=== {key} → {path.name} ===")
        load_raw(path)

# -------------------------------------------
# 2) 0004 aufbereiten → ags, gemeinde, miete_qm
# -------------------------------------------
print("\n### Aufbereitung: 0004 (Durchschnittliche Nettokaltmiete)")
df4, header4, start4 = load_table_with_header_guess(FILES["0004"], ags_col=1)
print("Startzeile erkannt:", start4)
print("Header (gekürzt):", header4[:10] if header4 else None)
print("Shape raw:", df4.shape)

ags_col = 1
name_col = 2

# erste numerische Spalte nach name_col suchen
num_col = None
for j in range(name_col + 1, df4.shape[1]):
    sample = df4.iloc[:80, j].apply(_to_float)
    if sample.notna().mean() > 0.5:
        num_col = j
        break

if num_col is None:
    raise RuntimeError("Konnte keine Mietspalte in 0004 erkennen. Bitte Datei prüfen.")

clean4 = pd.DataFrame({
    "ags": df4.iloc[:, ags_col].astype(str).str.strip(),
    "gemeinde": df4.iloc[:, name_col].astype(str).str.strip(),
    "miete_qm": df4.iloc[:, num_col].apply(_to_float),
})
clean4 = clean4[clean4["miete_qm"].between(2, 30)]
print("Cleaned shape:", clean4.shape)
print(clean4.head(10).to_string(index=False))

out_path = CLEAN_DIR / "zensus_0004_clean.csv"
clean4.to_csv(out_path, index=False, encoding="utf-8")
print("Gespeichert:", out_path)

# -------------------------------------------
# 3) Überblick 0005/0006/0008
# -------------------------------------------
print("\n### Überblick: 0005/0006/0008 (Header & numerische Spalten)")
for k in ["0005", "0006", "0008"]:
    dfk, headerk, startk = load_table_with_header_guess(FILES[k], ags_col=1)
    print(f"\n=== {k} === start={startk}")
    print("Header (gekürzt):", headerk[:20] if headerk else None)

    num_candidates = []
    upper = min(dfk.shape[1], 80)
    for j in range(3, upper):
        s = dfk.iloc[:80, j].apply(_to_float)
        if s.notna().mean() > 0.5:
            label = headerk[j] if headerk and j < len(headerk) else f"col_{j}"
            num_candidates.append((j, str(label)))
    print("Numerische Spalten (erste 12):", num_candidates[:12])

# -------------------------------------------
# 4) Beispiel: 0005 ins Long-Format (optional)
# -------------------------------------------
print("\n### Beispiel: 0005 ins Long-Format (optional)")
df5, h5, s5 = load_table_with_header_guess(FILES["0005"], ags_col=1)
value_cols = list(range(3, min(df5.shape[1], 40)))
long5 = melt_long(df5, ags_col=1, name_col=2, value_cols=value_cols,
                  var_name="baujahr", value_name="miete_qm")
print("0005 long shape:", long5.shape)
print(long5.head(12).to_string(index=False))

print("\nFertig. Es liegt jetzt 'data/clean/zensus_0004_clean.csv' bereit.")
