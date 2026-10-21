"""Ingeniería de features y target (t+6) con garantías anti-leakage.

Reglas (verificadas por tests/test_no_leakage.py):
  * Toda feature en la fila t usa SOLO información de instantes <= t
    (valores actuales, lags hacia el pasado, medias móviles que terminan en t).
  * El target es la categoría de us_aqi en t+6 (shift(-6) sobre una malla horaria regular).
  * us_aqi NUNCA es feature: solo se usa para construir el target.
  * Se reindexa a una malla horaria regular: un hueco en los datos produce NaN (y se descarta
    la fila) en lugar de desplazar silenciosamente los lags.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .constants import (
    BASE_COLUMNS,
    FEATURE_COLUMNS,
    LAGS,
    POLLUTANTS,
    RAW_COLUMNS,
    ROLL_WINDOWS,
    TARGET_SOURCE,
    category_from_aqi,
    lag_name,
    roll_name,
)


def build_features(
    df: pd.DataFrame, horizon: int = 6, with_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Construye (X, y) indexados por `time`.

    Con with_target=False (inferencia) no se necesita us_aqi futuro y y=None.
    """
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.sort_values("time").drop_duplicates("time", keep="last").set_index("time").asfreq("h")

    feats: dict[str, pd.Series | pd.Index] = {c: d[c] for c in BASE_COLUMNS}
    feats["hour"] = pd.Series(d.index.hour, index=d.index)
    feats["day_of_week"] = pd.Series(d.index.dayofweek, index=d.index)
    for col in POLLUTANTS:
        for lag in LAGS:
            feats[lag_name(col, lag)] = d[col].shift(lag)  # shift(+k) = valor de hace k horas
        for win in ROLL_WINDOWS:
            # ventana [t-win+1, t]; min_periods=win exige ventana completa (sin huecos)
            feats[roll_name(col, win)] = d[col].rolling(win, min_periods=win).mean()

    X = pd.DataFrame(feats)[FEATURE_COLUMNS].astype("float64")
    assert TARGET_SOURCE not in X.columns, "us_aqi must never be a feature"

    if not with_target:
        return X[X.notna().all(axis=1)], None

    # Target: categoría de us_aqi en t+horizon (shift(-h) trae el futuro SOLO a la etiqueta)
    y = category_from_aqi(d[TARGET_SOURCE].shift(-horizon))
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


def row_from_current(values: Mapping[str, float]) -> pd.DataFrame:
    """Fila de inferencia en modo MANUAL: lags y promedios móviles = valor actual.

    En modo manual el usuario solo conoce el estado actual, así que se asume una atmósfera
    estable (lag_k = valor actual y media móvil = valor actual). Es una aproximación
    documentada: el modo /latest usa el historial real de las últimas horas.
    """
    row: dict[str, float] = {c: float(values[c]) for c in BASE_COLUMNS}
    row["hour"] = float(values["hour"])
    row["day_of_week"] = float(values["day_of_week"])
    for col in POLLUTANTS:
        for lag in LAGS:
            row[lag_name(col, lag)] = row[col]
        for win in ROLL_WINDOWS:
            row[roll_name(col, win)] = row[col]
    return pd.DataFrame([row])[FEATURE_COLUMNS]
