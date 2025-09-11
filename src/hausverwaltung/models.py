# KEIN: from __future__ import annotations
from datetime import date
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship


class Owner(SQLModel, table=True):
    __tablename__ = "owners"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

    properties: List["Property"] = Relationship(back_populates="owner")


class Property(SQLModel, table=True):
    __tablename__ = "properties"
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="owners.id", index=True)
    street: str
    zip_code: str
    city: str
    year_built: Optional[int] = None

    owner: "Owner" = Relationship(back_populates="properties")
    units: List["Unit"] = Relationship(back_populates="property")


class Unit(SQLModel, table=True):
    __tablename__ = "units"
    id: Optional[int] = Field(default=None, primary_key=True)
    property_id: int = Field(foreign_key="properties.id", index=True)
    unit_no: str
    floor: Optional[int] = None
    rooms: Optional[float] = None
    size_sqm: Optional[float] = None
    heating_type: Optional[str] = None
    condition: Optional[str] = None

    property: "Property" = Relationship(back_populates="units")
    leases: List["Lease"] = Relationship(back_populates="unit")


class Tenant(SQLModel, table=True):
    __tablename__ = "tenants"
    id: Optional[int] = Field(default=None, primary_key=True)
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None

    leases: List["Lease"] = Relationship(back_populates="tenant")


class Lease(SQLModel, table=True):
    __tablename__ = "leases"
    id: Optional[int] = Field(default=None, primary_key=True)
    unit_id: int = Field(foreign_key="units.id", index=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    start_date: date
    end_date: Optional[date] = None
    rent_base: float
    rent_total: Optional[float] = None
    deposit: Optional[float] = None

    unit: "Unit" = Relationship(back_populates="leases")
    tenant: "Tenant" = Relationship(back_populates="leases")
    payments: List["Payment"] = Relationship(back_populates="lease")
    charges: List["Charge"] = Relationship(back_populates="lease")


class Payment(SQLModel, table=True):
    __tablename__ = "payments"
    id: Optional[int] = Field(default=None, primary_key=True)
    lease_id: int = Field(foreign_key="leases.id", index=True)
    paid_on: date
    amount: float
    type: str = Field(default="rent")
    note: Optional[str] = None

    lease: "Lease" = Relationship(back_populates="payments")


class Charge(SQLModel, table=True):
    __tablename__ = "charges"
    id: Optional[int] = Field(default=None, primary_key=True)
    lease_id: int = Field(foreign_key="leases.id", index=True)
    period: str
    type: str
    amount: float
    note: Optional[str] = None

    lease: "Lease" = Relationship(back_populates="charges")


class ListingRaw(SQLModel, table=True):
    __tablename__ = "listings_raw"
    id: Optional[int] = Field(default=None, primary_key=True)
    source: Optional[str] = Field(default="kaggle", index=True)           # z.B. 'kaggle' / 'scrape'
    city: Optional[str] = Field(default=None, index=True)
    zip_code: Optional[str] = Field(default=None, index=True)
    district: Optional[str] = Field(default=None, index=True)             # Stadtteil/Mikrolage, falls vorhanden
    lat: Optional[float] = Field(default=None, index=True)
    lon: Optional[float] = Field(default=None, index=True)
    area_sqm: Optional[float] = None
    rooms: Optional[float] = None
    floor: Optional[float] = None
    year_built: Optional[int] = None
    condition: Optional[str] = None                                       # 'neu'/'saniert'/...
    rent_warm: Optional[float] = None                                     # falls vorhanden
    rent_cold: Optional[float] = Field(default=None, index=True)          # <- Zielvariable
    created_at: Optional[str] = None                                      # Datum aus Inserat (String ok)
    extra_json: Optional[str] = None                                      # JSON-String für sonstige Spalten
