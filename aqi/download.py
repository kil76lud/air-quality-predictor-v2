"""Descarga de datos de Open-Meteo (sin API key).

IMPORTANTE: la Air Quality API NO trae variables meteorológicas. Por eso se combinan:
  * Air Quality API  -> contaminantes + us_aqi (histórico con start/end_date y recientes con past_days)
  * Historical Weather (archive) -> clima histórico (~5 días de retraso)
  * Forecast API (past_days)    -> clima de los últimos días (cubre el retraso del archive)
Los datos son de REANÁLISIS/modelo por cuadrícula (CAMS), no de sensores locales.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .config import Settings, get_settings, now_local, resolve_settings
from .constants import RAW_COLUMNS

log = logging.getLogger(__name__)

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# nombre en Open-Meteo -> nombre interno
AIR_VARS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "us_aqi": "us_aqi",
}
WEATHER_VARS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "surface_pressure": "pressure",
}
ARCHIVE_LAG_DAYS = 7  # el archive se usa hasta hoy-7; lo más reciente sale del forecast API


class DownloadError(RuntimeError):
    """Fallo de red / respuesta inesperada de Open-Meteo."""


def _get_json(url: str, params: dict[str, Any], retries: int = 3, timeout: int = 60) -> dict:
    """GET con reintentos y backoff exponencial (1s, 2s, 4s)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            log.warning("Open-Meteo request failed (%s/%s): %s", attempt + 1, retries, exc)
            time.sleep(2**attempt)
    raise DownloadError(f"Open-Meteo request failed after {retries} attempts: {last}")


def hourly_frame(payload: dict, rename: dict[str, str]) -> pd.DataFrame:
    """Convierte el bloque `hourly` de la respuesta en DataFrame con nombres internos."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise DownloadError(f"Unexpected Open-Meteo payload: {list(payload)[:5]}")
    absent = [k for k in rename if k not in hourly]
    if absent:
        raise DownloadError(f"Variables missing in payload: {absent}")
    df = pd.DataFrame({k: hourly[k] for k in ["time", *rename]})
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns=rename)


def _common(s: Settings) -> dict[str, Any]:
    return {
        "latitude": s.city.latitude,
        "longitude": s.city.longitude,
        "timezone": s.city.timezone,
    }


def fetch_air(s: Settings, **extra: Any) -> pd.DataFrame:
    params = {**_common(s), "hourly": ",".join(AIR_VARS), **extra}
    return hourly_frame(_get_json(AIR_URL, params), AIR_VARS)


def fetch_weather_recent(s: Settings, past_days: int) -> pd.DataFrame:
    params = {
        **_common(s),
        "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days,
        "forecast_days": 1,
    }
    return hourly_frame(_get_json(FORECAST_URL, params), WEATHER_VARS)


def fetch_weather_history(s: Settings, start: date, end: date) -> pd.DataFrame:
    """Clima histórico: archive hasta hoy-7 + forecast(past_days) para los últimos días."""
    archive_end = min(end, date.today() - timedelta(days=ARCHIVE_LAG_DAYS))
    params = {
        **_common(s),
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start.isoformat(),
        "end_date": archive_end.isoformat(),
    }
    archive = hourly_frame(_get_json(ARCHIVE_URL, params), WEATHER_VARS)
    recent = fetch_weather_recent(s, past_days=ARCHIVE_LAG_DAYS + 7)
    recent = recent[recent["time"] > archive["time"].max()]
    return pd.concat([archive, recent], ignore_index=True)


def fetch_history(s: Settings, start: date, end: date) -> pd.DataFrame:
    """Histórico por hora entre `start` y `end` (contaminantes + clima), sin limpiar."""
    air = fetch_air(s, start_date=start.isoformat(), end_date=end.isoformat())
    weather = fetch_weather_history(s, start, end)
    return air.merge(weather, on="time", how="inner")[RAW_COLUMNS]


def fetch_recent(s: Settings | None = None, past_days: int = 3) -> pd.DataFrame:
    """Últimas `past_days` días (incluye horas futuras del día actual: filtrar con clean_observed)."""
    s = s or get_settings()
    air = fetch_air(s, past_days=past_days, forecast_days=1)
    weather = fetch_weather_recent(s, past_days=past_days)
    return air.merge(weather, on="time", how="inner")[RAW_COLUMNS]


def clean_observed(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Descarta horas futuras y filas con cualquier valor nulo; ordena y deduplica."""
    out = df.dropna(subset=[c for c in RAW_COLUMNS if c != "time"])
    out = out[out["time"] <= now]
    return out.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def write_bootstrap_meta(s: Settings, df: pd.DataFrame, source: str) -> None:
    meta = {
        "source": source,
        "city": s.city.name,
        "latitude": s.city.latitude,
        "longitude": s.city.longitude,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "rows": len(df),
        "start": df["time"].min().isoformat(),
        "end": df["time"].max().isoformat(),
    }
    s.paths.raw_meta.parent.mkdir(parents=True, exist_ok=True)
    s.paths.raw_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def bootstrap(months: int | None = None, s: Settings | None = None) -> pd.DataFrame:
    """Descarga los últimos N meses, valida el esquema y guarda el dataset base (data/raw)."""
    from .validation import validate_frame  # lazy: pandera no existe en la imagen de runtime

    s = s or get_settings()
    months = months or s.history_months
    now = now_local(s)
    start = (now - pd.DateOffset(months=months)).date()
    df = clean_observed(fetch_history(s, start, now.date()), now)
    df = validate_frame(df)
    s.paths.raw_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(s.paths.raw_file, index=False)
    write_bootstrap_meta(s, df, source="open-meteo")
    log.info("Bootstrap saved: %s rows (%s -> %s)", len(df), df.time.min(), df.time.max())
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Download the 12-month historical bootstrap dataset.")
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--city", default=None, help="City slug from config.yaml (default: the default city).")
    p.add_argument("--all", action="store_true", help="Download every city in config.yaml.")
    args = p.parse_args()
    failed: list[str] = []
    for s in resolve_settings(args.city, args.all):
        try:
            df = bootstrap(args.months, s)
            print(f"[{s.city.name}] Bootstrap OK: {len(df)} rows, {df.time.min()} -> {df.time.max()}")
        except Exception as exc:  # noqa: BLE001 - una ciudad que falla no detiene a las demás
            print(f"[{s.city.name}] BOOTSTRAP FAILED: {exc}", file=sys.stderr)
            failed.append(s.city.name)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
