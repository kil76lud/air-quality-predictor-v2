"""Contrato de la API (pydantic), compartido con el diseñador de UI. No cambiar sin avisar.

La validación de DATOS crudos (pandera) vive en aqi/validation.py para que la imagen de
runtime no necesite pandera.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pm2_5: float = Field(ge=0, le=1000, description="PM2.5 (µg/m³)")
    pm10: float = Field(ge=0, le=2000, description="PM10 (µg/m³)")
    co: float = Field(ge=0, le=50000, description="Carbon monoxide (µg/m³)")
    no2: float = Field(ge=0, le=1000, description="Nitrogen dioxide (µg/m³)")
    so2: float = Field(ge=0, le=2000, description="Sulphur dioxide (µg/m³)")
    o3: float = Field(ge=0, le=1000, description="Ozone (µg/m³)")
    temperature: float = Field(ge=-60, le=60, description="Temperature (°C)")
    humidity: float = Field(ge=0, le=100, description="Relative humidity (%)")
    wind_speed: float = Field(ge=0, le=200, description="Wind speed (km/h)")
    pressure: float = Field(ge=300, le=1100, description="Surface pressure (hPa)")
    hour: int = Field(ge=0, le=23, description="Local hour of the day")
    day_of_week: int = Field(ge=0, le=6, description="Monday=0 ... Sunday=6")


class PredictResponse(BaseModel):
    predicted_category: str
    predicted_at_horizon: str = "t+6h"
    probabilities: dict[str, float]
    model_version: str


class LatestResponse(BaseModel):
    city: str
    observed_at: str
    inputs: PredictRequest
    current_category: str
    prediction: PredictResponse


class MetricsOut(BaseModel):
    accuracy: float
    f1_macro: float


class DataRange(BaseModel):
    start: str
    end: str


class ModelInfo(BaseModel):
    city: str
    model_name: str
    version: str
    alias: str = "champion"
    trained_at: str
    metrics: MetricsOut
    horizon_hours: int
    classes: list[str]
    data_range: DataRange
    n_rows: int


class CityInfo(BaseModel):
    slug: str
    name: str
    latitude: float
    longitude: float
    available: bool = Field(description="True when a trained champion exists for this city")
    default: bool = False


class Health(BaseModel):
    status: str = "ok"
