"""Gestión del champion: carga, decisión champion/challenger, promoción y bitácora.

Este módulo NO importa MLflow: la API lo usa en producción (imagen ligera).
Artefactos del champion (models/champion/):
  model.joblib       -> AqiClassifier serializado
  metadata.json      -> versión, métricas, rango de datos, features, parámetros...
  last_window.parquet-> últimas 72 h del dataset (respaldo de /latest si Open-Meteo falla)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .model import AqiClassifier

MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"
WINDOW_FILE = "last_window.parquet"
REGISTERED_MODEL_NAME = "aqi-category-classifier"


def load_champion(directory: Path) -> tuple[AqiClassifier, dict[str, Any]] | None:
    """Devuelve (modelo, metadata) o None si aún no hay champion."""
    model_path, meta_path = directory / MODEL_FILE, directory / META_FILE
    if not (model_path.exists() and meta_path.exists()):
        return None
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def decide(champion_f1: float | None, challenger_f1: float, min_delta: float) -> tuple[bool, str]:
    """Regla de promoción: el challenger debe superar al champion en >= min_delta de F1 macro.

    Ambos F1 se miden sobre el MISMO conjunto de prueba (lo garantiza train.py).
    """
    if champion_f1 is None:
        return True, "No incumbent champion: first model is promoted."
    delta = challenger_f1 - champion_f1
    if delta >= min_delta - 1e-12:
        return True, (
            f"Promoted: challenger F1 macro {challenger_f1:.4f} vs champion {champion_f1:.4f} "
            f"(delta {delta:+.4f} >= required {min_delta:+.4f})."
        )
    return False, (
        f"Rejected: challenger F1 macro {challenger_f1:.4f} vs champion {champion_f1:.4f} "
        f"(delta {delta:+.4f} < required {min_delta:+.4f})."
    )


def promote(
    model: AqiClassifier,
    metadata: dict[str, Any],
    directory: Path,
    dataset: pd.DataFrame | None = None,
) -> None:
    """Escribe el nuevo champion en disco (se versiona en git y alimenta al Docker)."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILE, compress=3)
    (directory / META_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if dataset is not None:
        window = dataset.sort_values("time").tail(72)
        window.to_parquet(directory / WINDOW_FILE, index=False)


def append_decision(record: dict[str, Any], reports_dir: Path) -> None:
    """Bitácora append-only de decisiones (promovido/rechazado + motivo)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), **record}
    with (reports_dir / "decisions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
