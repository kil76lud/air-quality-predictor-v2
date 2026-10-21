import pandas as pd
import pytest

from aqi import download, ingest
from aqi.config import get_settings
from aqi.constants import RAW_COLUMNS
from aqi.synthetic import make_synthetic
from aqi.validation import DataValidationError, validate_frame


@pytest.fixture()
def df():
    return make_synthetic(24 * 10, seed=3)


def test_valid_frame_passes(df):
    assert len(validate_frame(df)) == len(df)


@pytest.mark.parametrize(
    "column,value",
    [("pm2_5", -5.0), ("humidity", 150.0), ("temperature", 200.0), ("pressure", 10.0),
     ("us_aqi", -1.0)],
)
def test_invalid_values_are_rejected(df, column, value):
    bad = df.copy()
    bad.loc[5, column] = value
    with pytest.raises(DataValidationError):
        validate_frame(bad)


def test_nulls_missing_columns_and_duplicates_are_rejected(df):
    with pytest.raises(DataValidationError):
        validate_frame(df.assign(pm10=[None] * len(df)))
    with pytest.raises(DataValidationError):
        validate_frame(df.drop(columns=["o3"]))
    with pytest.raises(DataValidationError):
        validate_frame(pd.concat([df, df.iloc[[0]]], ignore_index=True))


def test_parse_payload_and_missing_variable():
    payload = {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                          "pm2_5": [1.0, 2.0], "pm10": [2.0, 3.0], "carbon_monoxide": [100, 110],
                          "nitrogen_dioxide": [1, 1], "sulphur_dioxide": [1, 1], "ozone": [30, 31],
                          "us_aqi": [10, 11]}}
    out = download.hourly_frame(payload, download.AIR_VARS)
    assert list(out.columns) == ["time", *download.AIR_VARS.values()]
    assert out["time"].dtype == "datetime64[ns]"
    payload["hourly"].pop("ozone")
    with pytest.raises(download.DownloadError):
        download.hourly_frame(payload, download.AIR_VARS)


def test_clean_observed_drops_nulls_and_future_hours(df):
    now = df["time"].iloc[-10]
    dirty = df.copy()
    dirty.loc[0, "pm10"] = None
    out = download.clean_observed(dirty, now)
    assert out["time"].max() == now and out["pm10"].notna().all() and len(out) == len(df) - 9 - 1


@pytest.fixture()
def settings(tmp_path, df):
    s = get_settings().with_root(tmp_path)
    s.paths.raw_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(s.paths.raw_file, index=False)
    return s


def test_incremental_appends_only_new_rows(settings, df, monkeypatch):
    last = df["time"].max()
    new = make_synthetic(24 * 2, seed=4)
    new["time"] = pd.date_range(last + pd.Timedelta(hours=1), periods=len(new), freq="h")
    monkeypatch.setattr(ingest, "fetch_recent", lambda *a, **k: new[RAW_COLUMNS])
    monkeypatch.setattr(ingest, "now_local", lambda *_: new["time"].max())
    res = ingest.ingest_incremental(settings)
    assert res.status == "updated" and res.new_rows == len(new)
    again = ingest.ingest_incremental(settings)  # idempotente
    assert again.status == "up_to_date"
    assert ingest.load_dataset(settings)["time"].is_unique


def test_invalid_incoming_data_is_rejected_and_nothing_is_written(settings, df, monkeypatch):
    last = df["time"].max()
    bad = make_synthetic(24, seed=5)
    bad["time"] = pd.date_range(last + pd.Timedelta(hours=1), periods=len(bad), freq="h")
    bad.loc[3, "humidity"] = 999.0
    monkeypatch.setattr(ingest, "fetch_recent", lambda *a, **k: bad[RAW_COLUMNS])
    monkeypatch.setattr(ingest, "now_local", lambda *_: bad["time"].max())
    ingest.load_dataset(settings)
    before = len(pd.read_parquet(settings.paths.processed_file))
    with pytest.raises(DataValidationError):
        ingest.ingest_incremental(settings)
    assert len(pd.read_parquet(settings.paths.processed_file)) == before


def test_api_failure_falls_back_to_existing_dataset(settings, monkeypatch):
    def boom(*a, **k):
        raise download.DownloadError("network down")

    monkeypatch.setattr(ingest, "fetch_recent", boom)
    res = ingest.ingest_incremental(settings)
    assert res.status == "fallback" and res.total_rows > 0
    with pytest.raises(download.DownloadError):
        ingest.ingest_incremental(settings, strict=True)
