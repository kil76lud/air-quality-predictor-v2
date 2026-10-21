import numpy as np
import pandas as pd
import pytest

from aqi.constants import AQI_BINS, CLASSES, FEATURE_COLUMNS, category_from_aqi
from aqi.features import build_features, row_from_current


def test_feature_columns_and_no_nans(synthetic_df):
    X, y = build_features(synthetic_df, 6)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert set(y.unique()) <= set(CLASSES)
    assert len(X) == len(y)


def test_lag_and_rolling_values(synthetic_df):
    X, _ = build_features(synthetic_df, 6)
    d = synthetic_df.set_index("time")
    t = X.index[100]
    assert X.loc[t, "pm2_5_lag3h"] == pytest.approx(d.loc[t - pd.Timedelta(hours=3), "pm2_5"])
    expected = d["pm2_5"].loc[t - pd.Timedelta(hours=5): t].mean()
    assert X.loc[t, "pm2_5_roll6h"] == pytest.approx(expected)
    assert X.loc[t, "hour"] == t.hour and X.loc[t, "day_of_week"] == t.dayofweek


def test_gap_in_data_drops_rows_instead_of_shifting_lags(synthetic_df):
    df = synthetic_df.drop(index=range(200, 210))  # 10 horas faltantes
    X, _ = build_features(df, 6)
    gap_start = synthetic_df.loc[200, "time"]
    # filas cuyo lag de 24 h cae en el hueco no deben existir
    hole = pd.date_range(gap_start, periods=10, freq="h")
    affected = X.index[X.index.isin(hole + pd.Timedelta(hours=24))]
    assert len(affected) == 0


@pytest.mark.parametrize(
    "aqi,expected",
    [(0, "Good"), (50, "Good"), (51, "Moderate"), (100, "Moderate"), (101, CLASSES[2]),
     (150, CLASSES[2]), (151, "Unhealthy"), (200, "Unhealthy"), (201, "Very Unhealthy"),
     (450, "Very Unhealthy")],
)
def test_category_boundaries(aqi, expected):
    assert category_from_aqi(pd.Series([aqi])).iloc[0] == expected
    assert len(AQI_BINS) == len(CLASSES) + 1


def test_row_from_current_fills_lags_with_current_value():
    vals = dict(pm2_5=20, pm10=30, co=300, no2=10, so2=2, o3=50, temperature=8, humidity=40,
                wind_speed=10, pressure=650, hour=14, day_of_week=2)
    row = row_from_current(vals)
    assert list(row.columns) == FEATURE_COLUMNS
    assert row["pm2_5_lag24h"].iloc[0] == 20 and row["o3_roll24h"].iloc[0] == 50
    assert np.isfinite(row.to_numpy()).all()
