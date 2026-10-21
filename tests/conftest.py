"""Fixtures compartidas. Los tests no tocan la red: todo dato externo se simula."""

from __future__ import annotations

import pandas as pd
import pytest

from aqi.config import get_settings
from aqi.synthetic import make_synthetic
from aqi.train import train_pipeline


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic(n_hours=24 * 120, seed=1)


@pytest.fixture(scope="session")
def tmp_settings(tmp_path_factory):
    """Settings con todas las rutas bajo un directorio temporal (aísla modelos y reportes)."""
    return get_settings().with_root(tmp_path_factory.mktemp("repo"))


@pytest.fixture(scope="session")
def trained_champion(synthetic_df, tmp_settings):
    """Champion entrenado rápido (3 trials, sin MLflow) sobre datos sintéticos."""
    decision = train_pipeline(dataset=synthetic_df, settings=tmp_settings, n_trials=3,
                              use_mlflow=False, force_promote=True)
    return tmp_settings.paths.champion_dir, decision


@pytest.fixture()
def client(trained_champion, monkeypatch):
    """TestClient de la API apuntando al champion temporal."""
    from fastapi.testclient import TestClient

    from aqi import api

    monkeypatch.setenv("MODEL_DIR", str(trained_champion[0]))
    monkeypatch.setenv("LATEST_TTL_SECONDS", "0")  # sin caché en tests
    api.reset_state()
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c
    api.reset_state()
