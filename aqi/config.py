"""Carga de configuración (config/config.yaml) y rutas del proyecto.

Multi-ciudad: `city` es la ciudad por defecto y `cities` el catálogo extra. Cada ciudad tiene
su propio dataset, champion y reportes (layout por `slug`, ver `Paths`).
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


class UnknownCityError(ValueError):
    """El slug pedido no está en el catálogo de config.yaml."""


def slugify(name: str) -> str:
    """'Ñuñoa Perú' -> 'nunoa-peru' (ASCII, minúsculas, guiones)."""
    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    timezone: str = "America/Lima"  # todo Perú es UTC-5 sin horario de verano

    @property
    def slug(self) -> str:
        return slugify(self.name)


@dataclass(frozen=True)
class TrainingConfig:
    n_trials: int
    n_splits: int
    test_fraction: float
    seed: int
    promotion_min_delta: float
    quality_gate_f1_macro: float


@dataclass(frozen=True)
class Paths:
    """Layout de carpetas bajo una raíz (configurable para pruebas).

    Con `slug` (producción): data/raw/<slug>.parquet, data/processed/<slug>.parquet,
    models/champion/<slug>/, models/reports/<slug>/.
    Sin `slug` (tests con `with_root`): layout plano de una sola ciudad.
    """

    root: Path
    slug: str = ""

    @property
    def raw_file(self) -> Path:
        return self.root / "data" / "raw" / (f"{self.slug}.parquet" if self.slug else "bootstrap.parquet")

    @property
    def raw_meta(self) -> Path:
        return self.root / "data" / "raw" / (f"{self.slug}.meta.json" if self.slug else "bootstrap.meta.json")

    @property
    def processed_file(self) -> Path:
        return self.root / "data" / "processed" / (f"{self.slug}.parquet" if self.slug else "dataset.parquet")

    @property
    def champion_dir(self) -> Path:
        base = self.root / "models" / "champion"
        return base / self.slug if self.slug else base

    @property
    def reports_dir(self) -> Path:
        base = self.root / "models" / "reports"
        return base / self.slug if self.slug else base

    @property
    def mlflow_db(self) -> Path:
        return self.root / "mlflow.db"

    @property
    def mlflow_artifacts(self) -> Path:
        return self.root / "mlartifacts"


@dataclass(frozen=True)
class Settings:
    city: City
    horizon_hours: int
    history_months: int
    training: TrainingConfig
    paths: Paths
    cities: tuple[City, ...] = ()

    def with_root(self, root: Path) -> Settings:
        """Copia de la configuración con todas las rutas bajo otra raíz (tests)."""
        return replace(self, paths=Paths(Path(root)))

    def for_city(self, slug: str | None = None) -> Settings:
        """Configuración de una ciudad del catálogo (rutas y coordenadas propias)."""
        slug = slug or self.city.slug
        match = next((c for c in (self.cities or (self.city,)) if c.slug == slug), None)
        if match is None:
            raise UnknownCityError(slug)
        return replace(self, city=match, paths=Paths(self.paths.root, match.slug))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg_path = Path(os.environ.get("AQI_CONFIG", ROOT / "config" / "config.yaml"))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    default = City(**raw["city"])
    extra = [City(**c) for c in raw.get("cities", [])]
    cities = (default, *[c for c in extra if c.slug != default.slug])
    return Settings(
        city=default,
        horizon_hours=int(raw["horizon_hours"]),
        history_months=int(raw["history_months"]),
        training=TrainingConfig(**raw["training"]),
        paths=Paths(ROOT, default.slug),
        cities=cities,
    )


def resolve_settings(city: str | None = None, all_cities: bool = False) -> list[Settings]:
    """Ciudades sobre las que actúa un comando: todas (--all), una (--city) o la de por defecto."""
    base = get_settings()
    if all_cities:
        return [base.for_city(c.slug) for c in base.cities]
    return [base.for_city(city)]


def champion_dir(slug: str | None = None) -> Path:
    """Directorio del champion de una ciudad. MODEL_DIR sobreescribe el de la ciudad por defecto
    (tests/Docker)."""
    s = get_settings()
    env = os.environ.get("MODEL_DIR")
    if env and (slug is None or slug == s.city.slug):
        return Path(env)
    return s.for_city(slug).paths.champion_dir


def now_local(settings: Settings | None = None) -> pd.Timestamp:
    """Hora actual local (naive) redondeada hacia abajo a la hora en punto."""
    s = settings or get_settings()
    return pd.Timestamp.now(tz=s.city.timezone).tz_localize(None).floor("h")


def to_iso(ts: pd.Timestamp, settings: Settings | None = None) -> str:
    """Timestamp naive (hora local) -> ISO8601 con offset, p. ej. 2026-10-19T14:00:00-05:00."""
    s = settings or get_settings()
    return pd.Timestamp(ts).tz_localize(s.city.timezone).isoformat()
