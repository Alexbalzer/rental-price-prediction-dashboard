# src/hausverwaltung/db.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event
from dotenv import load_dotenv
import importlib, sys

load_dotenv()

# --- DB-URL zentral steuern (heute SQLite, morgen z.B. Postgres) ---
DB_PATH = os.getenv("DB_PATH", "./data/hausverwaltung.db")
DEFAULT_SQLITE_URL = f"sqlite:///{Path(DB_PATH)}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# --- Engine erstellen; für SQLite Streamlit-kompatibel konfigurieren ---
engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)

# --- SQLite: Foreign Keys aktivieren ---
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

def init_db() -> None:
    # sicherstellen, dass Models genau EINMAL unter diesem Namen geladen sind
    if "hausverwaltung.models" not in sys.modules:
        importlib.import_module("hausverwaltung.models")
    SQLModel.metadata.create_all(engine)

@contextmanager
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session

