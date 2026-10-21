"""API FastAPI según el CONTRATO compartido con el diseñador de UI.

Endpoints: GET /health, GET /cities, GET /model-info, POST /predict, GET /latest.
Multiciudad: /model-info, /predict y /latest aceptan `?city=<slug>` (por defecto, la ciudad base).
Errores: siempre JSON {"detail": str} con códigos 4xx/5xx.
Todos los textos visibles están en inglés.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import download
from .champion import META_FILE, WINDOW_FILE, load_champion
from .config import Settings, UnknownCityError, champion_dir, get_settings, now_local, to_iso
from .constants import BASE_COLUMNS, CLASSES, category_from_aqi
from .features import build_features, row_from_current
from .model import AqiClassifier
from .schemas import (
    CityInfo,
    Health,
    LatestResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("aqi.api")

app = FastAPI(
    title="Air Quality Category Predictor",
    description="Predicts the US AQI category 6 hours ahead (Open-Meteo gridded reanalysis data).",
    version="1.0.0",
)

# ----------------------------------------------------------------------------- estado (un champion por ciudad)
@dataclass
class _State:
    model: AqiClassifier
    meta: dict[str, Any]


_states: dict[str, _State] = {}
_latest_cache: dict[str, dict[str, Any]] = {}


def reset_state() -> None:
    """Descarta los modelos y la caché (tests / recarga tras redeploy)."""
    _states.clear()
    _latest_cache.clear()


def _settings(city: str | None) -> Settings:
    """Settings de la ciudad pedida (404 si no está en el catálogo)."""
    try:
        return get_settings().for_city(city)
    except UnknownCityError as exc:
        available = ", ".join(c.slug for c in get_settings().cities)
        raise HTTPException(status_code=404, detail=f"Unknown city '{city}'. Available: {available}.") from exc


def get_state(slug: str) -> _State:
    st = _states.get(slug)
    if st is None:
        loaded = load_champion(champion_dir(slug))
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model not available yet for this city. Please try again later.")
        st = _states[slug] = _State(*loaded)
    return st


# ----------------------------------------------------------------------------- versión desplegada
# Render define RENDER_GIT_COMMIT (SHA del deploy). CD lo compara con el commit esperado para
# confirmar que la versión NUEVA ya sirve. Cabecera aditiva: no cambia el contrato JSON.
APP_COMMIT = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT") or "dev"


@app.middleware("http")
async def _commit_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-App-Commit"] = APP_COMMIT
    return response


# ----------------------------------------------------------------------------- errores
@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    parts = [f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid input - " + "; ".join(parts)})


@app.exception_handler(HTTPException)
async def _http_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ----------------------------------------------------------------------------- lógica
def _predict_from_features(X: pd.DataFrame, slug: str) -> PredictResponse:
    st = get_state(slug)
    proba = st.model.predict_proba(X)[0]
    probs = {c: round(float(p), 6) for c, p in zip(CLASSES, proba, strict=True)}
    return PredictResponse(
        predicted_category=max(probs, key=probs.get),
        predicted_at_horizon="t+6h",
        probabilities=probs,
        model_version=str(st.meta["version"]),
    )


def _recent_observations(s) -> pd.DataFrame:
    """Últimas horas reales; si Open-Meteo falla, usa la ventana respaldo del champion."""
    try:
        return download.fetch_recent(s, past_days=4)
    except download.DownloadError as exc:
        log.warning("Live data unavailable (%s); trying fallback window.", exc)
        window = champion_dir(s.city.slug) / WINDOW_FILE
        if window.exists():
            return pd.read_parquet(window)
        raise HTTPException(
            status_code=503, detail="Live air-quality data is temporarily unavailable."
        ) from exc


# ----------------------------------------------------------------------------- endpoints
@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok")


@app.get("/cities", response_model=list[CityInfo])
def cities() -> list[CityInfo]:
    """Catálogo de ciudades; `available` indica si ya existe un champion entrenado."""
    base = get_settings()
    return [
        CityInfo(
            slug=c.slug, name=c.name, latitude=c.latitude, longitude=c.longitude,
            available=(champion_dir(c.slug) / META_FILE).exists(), default=c.slug == base.city.slug,
        )
        for c in base.cities
    ]


@app.get("/model-info", response_model=ModelInfo)
def model_info(city: str | None = None) -> ModelInfo:
    s = _settings(city)
    meta = get_state(s.city.slug).meta
    return ModelInfo(
        city=s.city.name,
        model_name=meta["model_name"],
        version=str(meta["version"]),
        alias="champion",
        trained_at=meta["trained_at"],
        metrics={"accuracy": meta["metrics"]["accuracy"], "f1_macro": meta["metrics"]["f1_macro"]},
        horizon_hours=int(meta["horizon_hours"]),
        classes=list(meta["classes"]),
        data_range=meta["data_range"],
        n_rows=int(meta["n_rows"]),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, city: str | None = None) -> PredictResponse:
    """Modo manual: los lags y promedios móviles se rellenan con el valor ACTUAL de cada
    contaminante (el usuario solo conoce el estado actual). Ver aqi.features.row_from_current."""
    s = _settings(city)
    return _predict_from_features(row_from_current(req.model_dump()), s.city.slug)


@app.get("/latest", response_model=LatestResponse)
def latest(city: str | None = None) -> LatestResponse:
    """Datos reales más recientes de Open-Meteo + predicción (usa historial real para lags)."""
    s = _settings(city)
    slug = s.city.slug
    ttl = float(os.environ.get("LATEST_TTL_SECONDS", "600"))
    cached = _latest_cache.get(slug)
    if cached is not None and time.time() - cached["ts"] < ttl:
        return cached["payload"]

    obs = download.clean_observed(_recent_observations(s), now_local(s))
    X, _ = build_features(obs, s.horizon_hours, with_target=False)
    if X.empty:
        raise HTTPException(status_code=503, detail="Not enough recent data to build a prediction.")
    t = X.index[-1]
    row = X.iloc[[-1]]
    aqi_now = float(obs.set_index("time").loc[t, "us_aqi"])

    inputs = {c: float(row[c].iloc[0]) for c in BASE_COLUMNS}
    inputs["hour"] = int(row["hour"].iloc[0])
    inputs["day_of_week"] = int(row["day_of_week"].iloc[0])
    payload = LatestResponse(
        city=s.city.name,
        observed_at=to_iso(t, s),
        inputs=PredictRequest(**inputs),
        current_category=str(category_from_aqi(pd.Series([aqi_now])).iloc[0]),
        prediction=_predict_from_features(row, slug),
    )
    _latest_cache[slug] = {"ts": time.time(), "payload": payload}
    return payload
