"""Validación de datos crudos con pandera (solo ingesta / entrenamiento / CI).

Separado de aqi/schemas.py para que la imagen de runtime (API) NO dependa de pandera.
Rechaza nulos, rangos físicos imposibles, columnas faltantes/extra y timestamps duplicados.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa

from .constants import RAW_COLUMNS


def _num(lo: float, hi: float) -> pa.Column:
    return pa.Column(float, checks=pa.Check.in_range(lo, hi), nullable=False, coerce=True)


RAW_SCHEMA = pa.DataFrameSchema(
    {
        "time": pa.Column("datetime64[ns]", nullable=False, unique=True, coerce=True),
        "pm2_5": _num(0, 1000),  # µg/m³
        "pm10": _num(0, 2000),  # µg/m³
        "co": _num(0, 50000),  # µg/m³
        "no2": _num(0, 1000),  # µg/m³
        "so2": _num(0, 2000),  # µg/m³
        "o3": _num(0, 1000),  # µg/m³
        "temperature": _num(-60, 60),  # °C
        "humidity": _num(0, 100),  # %
        "wind_speed": _num(0, 200),  # km/h
        "pressure": _num(300, 1100),  # hPa (presión en superficie; Puno ≈ 640 hPa)
        "us_aqi": _num(0, 1000),
    },
    strict=True,
    coerce=True,
)


class DataValidationError(ValueError):
    """Los datos no cumplen el esquema; la ingesta se aborta sin escribir nada."""


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Valida contra RAW_SCHEMA. Devuelve el frame (coercionado) o lanza DataValidationError."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    try:
        return RAW_SCHEMA.validate(df[RAW_COLUMNS], lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.head(10).to_string()
        raise DataValidationError(f"Schema validation failed:\n{cases}") from exc
