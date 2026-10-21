"""PRUEBAS ANTI-LEAKAGE (obligatorias).

1. Ninguna feature en t cambia si se alteran TODOS los datos posteriores a t.
2. El target en t depende únicamente de us_aqi en t+6.
3. us_aqi nunca es feature.
4. El split cronológico no solapa etiquetas de train con el periodo de test.
"""

import numpy as np
import pandas as pd

from aqi.constants import FEATURE_COLUMNS, TARGET_SOURCE, category_from_aqi
from aqi.features import build_features
from aqi.train import split_time_ordered

H = 6


def test_us_aqi_is_not_a_feature():
    assert TARGET_SOURCE not in FEATURE_COLUMNS
    assert not any(c.startswith("us_aqi") for c in FEATURE_COLUMNS)


def test_features_at_t_ignore_future_data(synthetic_df):
    X_ref, _ = build_features(synthetic_df, H)
    cut = X_ref.index[len(X_ref) // 2]
    rng = np.random.default_rng(0)

    corrupted = synthetic_df.copy()
    future = corrupted["time"] > cut
    num_cols = [c for c in corrupted.columns if c != "time"]
    corrupted.loc[future, num_cols] = rng.uniform(1, 400, size=(future.sum(), len(num_cols)))

    X_new, _ = build_features(corrupted, H)
    common = X_ref.index[X_ref.index <= cut]
    pd.testing.assert_frame_equal(X_ref.loc[common], X_new.loc[common])


def test_target_is_category_of_aqi_exactly_h_hours_ahead(synthetic_df):
    X, y = build_features(synthetic_df, H)
    aqi = synthetic_df.set_index("time")[TARGET_SOURCE]
    for t in X.index[::97]:
        assert y.loc[t] == category_from_aqi(pd.Series([aqi.loc[t + pd.Timedelta(hours=H)]])).iloc[0]


def test_changing_future_aqi_changes_only_the_label(synthetic_df):
    X_ref, y_ref = build_features(synthetic_df, H)
    t = X_ref.index[500]
    mod = synthetic_df.copy()
    idx = mod.index[mod["time"] == t + pd.Timedelta(hours=H)][0]
    mod.loc[idx, TARGET_SOURCE] = 400.0  # AQI futuro extremo
    X_new, y_new = build_features(mod, H)
    pd.testing.assert_frame_equal(X_ref.loc[[t]], X_new.loc[[t]])  # features intactas
    assert y_new.loc[t] == "Very Unhealthy"


def test_chronological_split_has_no_label_overlap(synthetic_df):
    X, y = build_features(synthetic_df, H)
    X_tr, _, X_te, _ = split_time_ordered(X, y, 0.2, H)
    assert X_tr.index.max() + pd.Timedelta(hours=H) < X_te.index.min()
    assert X_tr.index.is_monotonic_increasing and X_te.index.is_monotonic_increasing
