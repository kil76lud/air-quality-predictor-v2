import pandas as pd
import pytest

from aqi import api, download
from aqi.constants import CLASSES, RAW_COLUMNS
from aqi.synthetic import make_synthetic

VALID = dict(pm2_5=18.5, pm10=30.0, co=250.0, no2=9.0, so2=2.0, o3=55.0, temperature=8.0,
             humidity=45.0, wind_speed=12.0, pressure=650.0, hour=14, day_of_week=2)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_model_info_matches_contract(client):
    body = client.get("/model-info").json()
    assert set(body) == {"model_name", "version", "alias", "trained_at", "metrics",
                         "horizon_hours", "classes", "data_range", "n_rows", "city"}
    assert body["alias"] == "champion" and body["horizon_hours"] == 6
    assert set(body["metrics"]) == {"accuracy", "f1_macro"}
    assert body["classes"] == CLASSES and set(body["data_range"]) == {"start", "end"}


def test_predict_matches_contract(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predicted_category", "predicted_at_horizon", "probabilities",
                         "model_version"}
    assert body["predicted_at_horizon"] == "t+6h"
    assert list(body["probabilities"]) == CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["predicted_category"] in CLASSES


@pytest.mark.parametrize("patch", [{"humidity": 150}, {"hour": 24}, {"pm2_5": -1},
                                   {"day_of_week": 7}, {"pm2_5": "abc"}])
def test_predict_invalid_input_returns_detail_string(client, patch):
    r = client.post("/predict", json={**VALID, **patch})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "o3"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422 and "o3" in r.json()["detail"]


def test_unknown_route_has_detail(client):
    r = client.get("/nope")
    assert r.status_code == 404 and isinstance(r.json()["detail"], str)


def _recent(hours=96):
    df = make_synthetic(hours, seed=9)
    shift = pd.Timestamp.now().floor("h") - df["time"].max()
    df["time"] = df["time"] + shift
    return df[RAW_COLUMNS]


def test_latest_returns_inputs_and_prediction(client, monkeypatch):
    monkeypatch.setattr(download, "fetch_recent", lambda *a, **k: _recent())
    monkeypatch.setattr(api, "now_local", lambda *_: pd.Timestamp.now().floor("h"))
    r = client.get("/latest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"city", "observed_at", "inputs", "current_category", "prediction"}
    assert body["current_category"] in CLASSES
    assert body["prediction"]["predicted_at_horizon"] == "t+6h"
    assert {"pm2_5", "hour", "day_of_week"} <= set(body["inputs"])


def test_latest_uses_fallback_window_when_api_down(client, trained_champion, monkeypatch):
    def boom(*a, **k):
        raise download.DownloadError("down")

    monkeypatch.setattr(download, "fetch_recent", boom)
    monkeypatch.setattr(api, "now_local", lambda *_: pd.Timestamp("2100-01-01"))
    r = client.get("/latest")
    assert r.status_code == 200 and r.json()["current_category"] in CLASSES


def test_latest_503_when_api_down_and_no_fallback(client, trained_champion, monkeypatch):
    def boom(*a, **k):
        raise download.DownloadError("down")

    monkeypatch.setattr(download, "fetch_recent", boom)
    (trained_champion[0] / "last_window.parquet").rename(trained_champion[0] / "tmp.parquet")
    try:
        r = client.get("/latest")
        assert r.status_code == 503 and isinstance(r.json()["detail"], str)
    finally:
        (trained_champion[0] / "tmp.parquet").rename(trained_champion[0] / "last_window.parquet")


def test_model_unavailable_returns_503(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    api.reset_state()
    r = TestClient(api.app, raise_server_exceptions=False).get("/model-info")
    api.reset_state()
    assert r.status_code == 503 and "detail" in r.json()


def test_commit_header_identifies_deployed_version(client):
    r = client.get("/health")
    assert r.headers["x-app-commit"]  # "dev" en local; el SHA del deploy en Render
