"""QUALITY GATE de CI sobre el champion VERSIONADO de CADA ciudad (models/champion/<ciudad>).

1. El F1 macro registrado en su metadata debe superar el mínimo configurado.
2. Se REEVALÚA el champion sobre el holdout actual del dataset versionado: si el modelo se
   degrada (por datos nuevos) por debajo del mínimo, CI falla y evita desplegarlo.
Las ciudades aún sin champion entrenado se omiten (skip) en lugar de fallar.
"""

import json
import warnings

import pytest

from aqi.champion import META_FILE, load_champion
from aqi.config import get_settings
from aqi.features import build_features
from aqi.metrics import evaluate
from aqi.train import split_time_ordered

BASE = get_settings()
SLUGS = [c.slug for c in BASE.cities]


@pytest.mark.parametrize("slug", SLUGS)
def test_registered_f1_macro_above_minimum(slug):
    s = BASE.for_city(slug)
    meta_file = s.paths.champion_dir / META_FILE
    if not meta_file.exists():
        pytest.skip(f"no champion committed for {slug}")
    meta = json.loads(meta_file.read_text())
    assert meta["metrics"]["f1_macro"] >= s.training.quality_gate_f1_macro, f"{slug}: F1 macro below gate"
    if meta.get("data_source") == "synthetic":
        warnings.warn(f"{slug}: champion trained on SYNTHETIC data: run `make first-run`.", stacklevel=1)


@pytest.mark.parametrize("slug", SLUGS)
def test_champion_reevaluated_on_current_holdout_above_minimum(slug):
    from aqi.ingest import load_dataset

    s = BASE.for_city(slug)
    if not (s.paths.champion_dir / META_FILE).exists():
        pytest.skip(f"no champion committed for {slug}")
    if not (s.paths.processed_file.exists() or s.paths.raw_file.exists()):
        pytest.skip(f"no dataset committed for {slug}")
    model, _ = load_champion(s.paths.champion_dir)
    X, y = build_features(load_dataset(s), s.horizon_hours)
    _, _, X_te, y_te = split_time_ordered(X, y, s.training.test_fraction, s.horizon_hours)
    f1 = evaluate(y_te, model.predict(X_te))["f1_macro"]
    assert f1 >= s.training.quality_gate_f1_macro, f"{slug}: champion degraded, F1 macro={f1:.3f}"
