"""Ingesta incremental + validación de esquema + respaldo.

Flujo:
  1. Carga el dataset acumulado (data/processed); si no existe, parte del bootstrap (data/raw).
  2. Descarga las últimas horas desde Open-Meteo (past_days calculado según el hueco).
  3. Limpia (sin nulos / sin horas futuras), VALIDA con pandera y agrega solo filas nuevas.
  4. Si la API falla: usa el dataset existente (modo respaldo) salvo que se pida --strict.
  Datos inválidos => DataValidationError y NO se escribe nada.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass

import pandas as pd

from .config import Settings, get_settings, now_local, resolve_settings
from .download import DownloadError, clean_observed, fetch_recent
from .validation import DataValidationError, validate_frame

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    status: str  # "updated" | "up_to_date" | "fallback"
    new_rows: int
    total_rows: int
    message: str


def load_dataset(s: Settings | None = None) -> pd.DataFrame:
    """Dataset acumulado; si no existe se crea desde el bootstrap versionado (respaldo)."""
    s = s or get_settings()
    if s.paths.processed_file.exists():
        return pd.read_parquet(s.paths.processed_file)
    if s.paths.raw_file.exists():
        return reset_from_raw(s)
    raise FileNotFoundError(f"No dataset found for {s.city.name}. Run `python -m aqi.download --city {s.city.slug}` first.")


def reset_from_raw(s: Settings | None = None) -> pd.DataFrame:
    """Reconstruye data/processed a partir del bootstrap en data/raw."""
    s = s or get_settings()
    df = validate_frame(pd.read_parquet(s.paths.raw_file))
    _save(df, s)
    return df


def _save(df: pd.DataFrame, s: Settings) -> None:
    s.paths.processed_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = s.paths.processed_file.with_suffix(".tmp")
    df.sort_values("time").to_parquet(tmp, index=False)
    os.replace(tmp, s.paths.processed_file)  # escritura atómica


def ingest_incremental(s: Settings | None = None, strict: bool = False) -> IngestResult:
    s = s or get_settings()
    current = load_dataset(s)
    last = current["time"].max()
    now = now_local(s)

    gap_days = math.ceil((now - last) / pd.Timedelta(days=1)) + 1
    past_days = max(2, min(gap_days, 92))  # Open-Meteo admite hasta 92 días
    try:
        fresh = fetch_recent(s, past_days=past_days)
    except DownloadError as exc:
        if strict:
            raise
        msg = f"Open-Meteo unavailable, using existing dataset: {exc}"
        log.warning(msg)
        return IngestResult("fallback", 0, len(current), msg)

    new = clean_observed(fresh, now)
    new = new[new["time"] > last]  # solo se agrega lo posterior; el historial no se reescribe
    if new.empty:
        return IngestResult("up_to_date", 0, len(current), "No new rows.")

    new = validate_frame(new)  # lanza DataValidationError si hay datos inválidos
    merged = pd.concat([current, new], ignore_index=True)
    merged = validate_frame(merged.sort_values("time").reset_index(drop=True))
    _save(merged, s)
    return IngestResult("updated", len(new), len(merged), f"Added {len(new)} rows up to {new.time.max()}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Incremental ingestion with schema validation.")
    p.add_argument("--reset", action="store_true", help="Rebuild processed dataset from raw bootstrap.")
    p.add_argument("--strict", action="store_true", help="Fail if the API is unavailable.")
    p.add_argument("--city", default=None, help="City slug from config.yaml (default: the default city).")
    p.add_argument("--all", action="store_true", help="Process every city in config.yaml.")
    args = p.parse_args()
    code = 0
    for s in resolve_settings(args.city, args.all):
        try:
            if args.reset:
                df = reset_from_raw(s)
                print(f"[{s.city.name}] Dataset rebuilt from bootstrap: {len(df)} rows")
            else:
                res = ingest_incremental(s, strict=args.strict)
                print(f"[{s.city.name}] [{res.status}] {res.message} Total rows: {res.total_rows}")
        except DataValidationError as exc:
            print(f"[{s.city.name}] INGEST REJECTED (invalid data): {exc}", file=sys.stderr)
            code = max(code, 2)
        except DownloadError as exc:
            print(f"[{s.city.name}] INGEST FAILED (API): {exc}", file=sys.stderr)
            code = max(code, 3)
        except FileNotFoundError as exc:
            print(f"[{s.city.name}] NO DATA: {exc}", file=sys.stderr)
            code = max(code, 1)
    sys.exit(code)


if __name__ == "__main__":
    main()
