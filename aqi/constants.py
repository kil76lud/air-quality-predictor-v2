"""Constantes compartidas: única fuente de verdad para nombres de columnas y clases."""

from __future__ import annotations

import pandas as pd

# Categorías US AQI. Nota: todo AQI > 200 (incluye "Hazardous") se agrupa en "Very Unhealthy".
CLASSES: list[str] = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
]
# Intervalos (a, b] sobre us_aqi: 0-50, 51-100, 101-150, 151-200, 201+
AQI_BINS: list[float] = [-float("inf"), 50, 100, 150, 200, float("inf")]

POLLUTANTS: list[str] = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]
WEATHER: list[str] = ["temperature", "humidity", "wind_speed", "pressure"]
BASE_COLUMNS: list[str] = POLLUTANTS + WEATHER
TARGET_SOURCE = "us_aqi"  # SOLO se usa para construir el target; nunca es feature
RAW_COLUMNS: list[str] = ["time", *BASE_COLUMNS, TARGET_SOURCE]

LAGS: tuple[int, ...] = (1, 3, 6, 24)  # horas hacia el PASADO
ROLL_WINDOWS: tuple[int, ...] = (6, 24)  # ventanas móviles que terminan en t (inclusive)
CALENDAR: list[str] = ["hour", "day_of_week"]  # day_of_week: lunes=0 ... domingo=6


def lag_name(col: str, lag: int) -> str:
    return f"{col}_lag{lag}h"


def roll_name(col: str, window: int) -> str:
    return f"{col}_roll{window}h"


FEATURE_COLUMNS: list[str] = (
    BASE_COLUMNS
    + CALENDAR
    + [lag_name(c, lag) for c in POLLUTANTS for lag in LAGS]
    + [roll_name(c, w) for c in POLLUTANTS for w in ROLL_WINDOWS]
)


def category_from_aqi(aqi: pd.Series) -> pd.Series:
    """Convierte us_aqi en categoría (str). Los NaN se conservan como NaN."""
    return pd.cut(aqi, bins=AQI_BINS, labels=CLASSES, right=True).astype(object)
